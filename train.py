"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU small debug run, example:
$ python -m train.py --compile=False --eval_iters=10 --batch_size=8

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import sys
import os
import time
from contextlib import nullcontext
from datetime import datetime
from functools import partial
import typer
import yaml

import torch
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from model import ModelArgs
from tinystories import Task
from export import model_export
from training_utils import (
    TrainingArgs,
    setup_wandb,
    setup_ddp,
    setup_model_and_optimizer,
    get_lr,
    estimate_loss
)

app = typer.Typer()


@app.command()
def main(
    config_file: str = typer.Option(..., "--config", help="Path to the config file")
):
    print(config_file)
    # Read config file and setup args
    if config_file is not None:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        model_config = config["model_args"]
        training_config = config["training_args"]

        print(model_config["norm_eps"], type(model_config["norm_eps"]))
        assert isinstance(model_config["norm_eps"], float)
    else:
        model_config, training_config = {}, {}

    model_args = ModelArgs(**model_config)
    training_args = TrainingArgs(**training_config)

    # Setup distributed training if needed
    ddp, master_process, seed_offset, ddp_world_size = setup_ddp(training_args)

    # Calculate tokens per iteration
    tokens_per_iter = training_args.gradient_accumulation_steps * ddp_world_size * training_args.batch_size * model_args.max_seq_len
    if master_process:
        print(f"tokens per iteration will be: {tokens_per_iter:,}")
        print(f"breaks down as: {training_args.gradient_accumulation_steps} grad accum steps * {ddp_world_size} processes * {training_args.batch_size} batch size * {model_args.max_seq_len} max seq len")

    # Set random seed
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Setup device and dtype
    device_type = "cuda" if "cuda" in training_args.device else "cpu"
    ptdtype = training_args.get_ptdtype()
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Setup task
    iter_batches = partial(
        Task.iter_batches,
        batch_size=training_args.batch_size,
        device=training_args.device,
        num_workers=0,
        max_seq_len=model_args.max_seq_len,
        vocab_size=model_args.vocab_size,
        vocab_source=model_args.vocab_source,
        num_future_tokens=model_args.num_future_tokens,
    )

    # Setup model and optimizer
    model, optimizer, model_args, iter_num, best_val_loss = setup_model_and_optimizer(model_args,
                                                                        training_args)

    # Setup gradient scaler for fp16
    scaler = torch.amp.GradScaler(enabled=(training_args.dtype == "float16"))

    # Compile model if requested
    if training_args.compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model   # Karpathy has it in the code, but never uses it
        model = torch.compile(model)

    # Wrap model in DDP if needed
    if ddp:
        # Ignore the `freqs_cis` buffer so that DDP does not broadcast it at
        # construction time since NCCL does not support `ComplexFloat`
        prefix = "_orig_mod." if training_args.compile else ""
        model._ddp_params_and_buffers_to_ignore = {prefix + "freqs_cis"}
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])

    # Setup output directory and wandb
    config["model_num_parameters"] = sum(p.numel() for p in model.parameters())
    if master_process:
        # wandb
        if training_args.wandb_log:
            import wandb
            run = setup_wandb(training_args, config)
            run_name = run.name
        else:
            run_name =  "run" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + training_args.additional_info
        print(f"Run name: {run_name}")
        run_save_dir = os.path.join(training_args.out_dir, run_name)
        os.makedirs(run_save_dir, exist_ok=True)

    ###  Training loop
    train_batch_iter = iter_batches(split="train")
    X, Y = next(train_batch_iter)
    t0 = time.time()
    local_iter_num = 0
    raw_model = model.module if ddp else model
    running_mfu = -1.0

    while True:
        # Set learning rate
        lr = get_lr(iter_num, training_args) if training_args.decay_lr else training_args.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Evaluate and save checkpoint (every `eval_interval` iterations)
        if iter_num % training_args.eval_interval == 0 and master_process:
            losses = estimate_loss(model, iter_batches, training_args.eval_iters, ctx, training_args.device)
            # in stdout print the aggregated losses of all heads
            print(f"step {iter_num}: train loss {sum(losses['train'].values())/len(losses['train']):.4f}, val loss {sum(losses['val'].values())/len(losses['val']):.4f}")

            if training_args.wandb_log:
                try:
                    # Create base logging dictionary
                    log_dict = {
                        "iter": iter_num,
                        "tokens": iter_num * tokens_per_iter,
                        "lr": lr,
                        "mfu": running_mfu * 100,
                    }

                    # Add losses for each head
                    for head_name, train_loss in losses["train"].items():
                        log_dict[f"loss/train/{head_name}"] = train_loss
                    for head_name, val_loss in losses["val"].items():
                        log_dict[f"loss/val/{head_name}"] = val_loss

                    # Add average losses
                    log_dict["loss/train/avg"] = sum(losses["train"].values())/len(losses["train"])
                    log_dict["loss/val/avg"] = sum(losses["val"].values())/len(losses["val"])

                    wandb.log(log_dict, step=iter_num)
                except Exception as e:
                    print(f"logging to wandb failed: {e}")

            # Use average validation loss for checkpointing
            avg_val_loss = sum(losses["val"].values())/len(losses["val"])
            if avg_val_loss < best_val_loss or training_args.always_save_checkpoint:
                best_val_loss = avg_val_loss
                if iter_num > 0:
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": model_args,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "config": training_args.__dict__,
                    }
                    print(f"saving checkpoint to {run_save_dir}")
                    torch.save(checkpoint, os.path.join(run_save_dir, "ckpt.pt"))
                    model_export(raw_model, os.path.join(run_save_dir, "model.bin"), version=0)

        if iter_num == 0 and training_args.eval_only:
            break

        # Forward backward update
        for micro_step in range(training_args.gradient_accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = micro_step == training_args.gradient_accumulation_steps - 1
            with ctx:
                logits = model(X, Y)
                loss = raw_model.last_loss
                loss = loss / training_args.gradient_accumulation_steps
            X, Y = next(train_batch_iter)
            scaler.scale(loss).backward()

        # Clip gradients
        if training_args.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_args.grad_clip)

        # Step optimizer
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # Logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % training_args.log_interval == 0 and master_process:
            lossf = loss.item() * training_args.gradient_accumulation_steps
            if local_iter_num >= 5:
                mfu = raw_model.estimate_mfu(training_args.batch_size * training_args.gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            print(f"{iter_num} | loss {lossf:.4f} | lr {lr:e} | {dt*1000:.2f}ms | mfu {running_mfu*100:.2f}%")
            if training_args.wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "loss": lossf,
                    "lr": lr,
                }, step=iter_num)

        iter_num += 1
        local_iter_num += 1

        if iter_num > training_args.max_iters:
            break

    if ddp:
        destroy_process_group()

if __name__ == "__main__":
    app()
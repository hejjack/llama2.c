import os
import math
import torch
import torch.nn as nn
from datetime import datetime
from contextlib import nullcontext
from model import Transformer, ModelArgs
from dataclasses import dataclass


@dataclass
class TrainingArgs:
    # I/O
    out_dir: str = "out"
    eval_interval: int = 500
    log_interval: int = 1
    eval_iters: int = 50
    eval_only: bool = False  # if True, script exits right after the first eval
    always_save_checkpoint: bool = False  # if True, always save a checkpoint after each eval
    init_from: str = "scratch"  # 'scratch' or 'resume'

    # wandb logging
    wandb_log: bool = True  # disabled by default
    wandb_project: str = "bottlecap"
    wandb_run_name: str|None = None  # will be appended with timestamp
    wandb_group: str|None = None
    additional_run_name_info: str|None = None  # will be appended with timestamp
    resume_id: str|None = None  # will be appended with timestamp

    # data
    batch_size: int = 2  # if gradient_accumulation_steps > 1, this is the micro-batch size
    vocab_source: str = "llama2"  # llama2|custom; use Lllama 2 vocab from Meta, or custom trained

    # adamw optimizer
    gradient_accumulation_steps: int = 4  # used to simulate larger batch sizes
    learning_rate: float = 5e-4  # max learning rate
    max_iters: int = 100000  # total number of training iterations
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0  # clip gradients at this value, or disable if == 0.0

    # learning rate decay settings
    decay_lr: bool = True  # whether to decay the learning rate
    warmup_iters: int = 1000  # how many steps to warm up for

    # system
    device: str = "cpu"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype: str = "bfloat16"  # float32|bfloat16|float16
    compile: bool = True  # use PyTorch 2.0 to compile the model to be faster

    # derived attributes (computed from other parameters)
    lr_decay_iters: int | None = None  # will be set to max_iters
    min_lr: float | None = None  # will be set to 0.0

    def __post_init__(self):
        # Set derived attributes
        self.lr_decay_iters = self.max_iters
        self.min_lr = 0.0

    def get_ptdtype(self):
        return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[self.dtype]


def setup_wandb(training_args: TrainingArgs, config: dict, num_future_tokens: int = None) -> None:
    """Setup wandb if requested."""
    import wandb
    run = wandb.init(
        name=training_args.wandb_run_name, # None by default -> automatic wandb name
        id=training_args.resume_id,
        group=training_args.wandb_group,
        resume="must" if training_args.resume_id else "never",
        entity="msgc_boys",
        project=training_args.wandb_project,
        save_code=True,
        config=config,
    )
    # to not add additional info to the run name if it is already there
    if not run.name.endswith(training_args.additional_run_name_info):
        if num_future_tokens is not None:
            run.name = run.name + f"_{num_future_tokens}heads"
        run.name = run.name + training_args.additional_run_name_info

    return run


def setup_ddp(args: TrainingArgs) -> tuple[bool, bool, int, int]:
    """Setup distributed data parallel training if needed."""
    ddp = int(os.environ.get("RANK", -1)) != -1 # is this a ddp run?
    if ddp:
        torch.distributed.init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        args.device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(args.device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert args.gradient_accumulation_steps % ddp_world_size == 0
        args.gradient_accumulation_steps //= ddp_world_size
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
    return ddp, master_process, seed_offset, ddp_world_size


def setup_model_and_optimizer(model_args: ModelArgs, training_args: TrainingArgs) -> tuple[Transformer, torch.optim.Optimizer, ModelArgs, int, float]:
    """Initialize or load the model and initialize or load the optimizer."""
    if training_args.init_from == "scratch":
        print("Initializing a new model from scratch")
        model = Transformer(model_args)
        iter_num = 0
        best_val_loss = 1e9
    elif training_args.init_from == "resume":
        print(f"Resuming training from {training_args.init_from}")
        ckpt_path = os.path.join(training_args.init_from, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=training_args.device)
        checkpoint_model_args = checkpoint["model_args"]
        restored_model_args = {}
        for k in ["dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "hidden_dim", "multiple_of", "norm_eps", "max_seq_len", "dropout", "num_future_tokens"]:
            restored_model_args[k] = checkpoint_model_args[k]
        print("Model_args: ", restored_model_args)
        model = Transformer(ModelArgs(**restored_model_args))
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
        model_args = restored_model_args
    else:
        raise ValueError(f"Invalid init_from: {training_args.init_from}")

    model.to(training_args.device)

    # optimizer
    optimizer = model.configure_optimizers(
        training_args.weight_decay,
        training_args.learning_rate,
        (training_args.beta1, training_args.beta2),
        training_args.device
    )
    if training_args.init_from == "resume" and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    checkpoint = None  # free up memory


    return model, optimizer, model_args, iter_num, best_val_loss


def get_lr(it: int, args: TrainingArgs) -> float:
    """Get learning rate for current iteration."""
    if it < args.warmup_iters:
        return args.learning_rate * it / args.warmup_iters
    if it > args.max_iters:
        return 0.0
    decay_ratio = (it - args.warmup_iters) / (args.max_iters - args.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return 0.0 + coeff * (args.learning_rate - 0.0)

@torch.no_grad()
def estimate_loss(
    model: Transformer,
    iter_batches: callable,
    eval_iters: int,
    ctx: nullcontext,
    device: str
) -> dict[str, dict[str, float]]:
    """Estimate loss on train/val sets for each head separately."""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        batch_iter = iter_batches(split=split)
        # Initialize losses for each head
        num_heads = len(model.output_heads)
        head_losses = {f"head_{i}": torch.zeros(eval_iters) for i in range(num_heads)}

        loss_fct = nn.CrossEntropyLoss()
        for k in range(eval_iters):
            X, Y = next(batch_iter)
            with ctx:
                logits = model(X, Y)  # (batch_size, seq_len, num_heads, vocab_size)
                # Compute loss for each head separately
                for i in range(num_heads):
                    head_logits = logits[:, :, i, :].reshape(-1, model.vocab_size)
                    head_targets = Y[:, :, i].view(-1)
                    head_losses[f"head_{i}"][k] = loss_fct(head_logits, head_targets)

        # Compute mean loss for each head
        split_losses = {f"head_{i}": head_losses[f"head_{i}"].mean().item() for i in range(num_heads)}
        out[split] = split_losses

    model.train()
    return out
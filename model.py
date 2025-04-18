import math
import struct
import inspect
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

@dataclass
class ModelArgs:
    # default hyperparameters for the Llama 7B model
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = 32000
    vocab_source: str = "llama2"
    hidden_dim: Optional[int] = None
    multiple_of: int = 256  # MLP hidden layer size will be multiple of
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    dropout: float = 0.0
    untied_head: bool = False
    num_future_tokens: int = 4  # Number of future tokens to predict (1 + num MTPs)
    lambda_loss: float = 0.3  # weight for scaling the loss of MTPs
    num_mtp_layers: int = 1  # Number of transformer blocks for the MTPs
    mtp_structure: str = "linear"  # "linear" or "tree"
    mtp_info_merge: str = "mean"  # "concat" or "mean"

    def __post_init__(self):
        assert self.vocab_source in ["llama2", "custom"]
        assert self.vocab_source == "custom" or self.vocab_size == 32000, "The vocab from Meta has 32K tokens"


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), f"{freqs_cis.shape} != {(x.shape[1], x.shape[-1])}"
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    # reshape xq and xk to match the complex representation
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # flatten last two dimensions
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        # use flash attention or a manual implementation?
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ):
        bsz, seqlen, _ = x.shape

        # QKV
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # RoPE relative positional embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # grouped multiquery attention: expand out keys and values
        xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        # make heads into a batch dimension
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # flash implementation
        if self.flash:
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            # manual implementation
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            assert hasattr(self, 'mask')
            scores = scores + self.mask[:, :, :seqlen, :seqlen]   # (bs, n_local_heads, seqlen, cache_len + seqlen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin):
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class MTPModule(nn.Module):
    """Multi-Token Prediction Module that predicts tokens further in the future
    sketch of the architecture: https://dataturbo.medium.com/deepseek-technical-analysis-3-multi-token-prediction-f8f3ea7eaf9c
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.past_info_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.current_info_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.linear_proj = nn.Linear(2 * args.dim, args.dim, bias=False)
        self.dropout = nn.Dropout(args.dropout)
        # more transformers blocks for the MTP
        self.mtp_layers = nn.ModuleList([
            TransformerBlock(i, args) for i in range(args.num_mtp_layers)
        ])

    def forward(self, current_info: torch.Tensor, past_info: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
        """
        current_info: (batch_size, seq_len, dim)
            - the shifted embeddings freshly from the shared embedding table
        past_info: (batch_size, seq_len, dim)
            - the past info is the output of the last transformer block of the main model or the previous MTP module
        """
        past_info = self.past_info_norm(past_info)
        current_info = self.current_info_norm(current_info)
        combined = torch.cat([past_info, current_info], dim=-1)
        combined = self.linear_proj(combined)
        combined = self.dropout(combined)
        for layer in self.mtp_layers:
            combined = layer(combined, freqs_cos, freqs_sin)
        return combined


class Transformer(nn.Module):
    last_loss: Optional[torch.Tensor]

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        if params.num_future_tokens > 1:
            self.loss_weighting_factor = params.lambda_loss / (params.num_future_tokens - 1)  # lambda / D   ...   from deepseek paper

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        # add the MTP modules
        self.mtp_modules = nn.ModuleList([
            MTPModule(params) for _ in range(params.num_future_tokens - 1)
        ])

        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # share the unembedding parameters with the embedding parameters
        if not params.untied_head:
            self.tok_embeddings.weight = self.output.weight # https://paperswithcode.com/method/weight-tying

        # some useful precompute for the RoPE relative positional embeddings
        freqs_cos, freqs_sin = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * params.n_layers))

        # Initialize attribute for the loss of the last forward call. This will be set if the forward is called with a targets tensor.
        self.last_loss = None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        _bsz, seqlen = tokens.shape
        embeddings = self.tok_embeddings(tokens)
        embeddings = self.dropout(embeddings)

        # If we are training and we have multiple future tokens to predict, we need to cut the seq-related variables to the window size
        if self.params.num_future_tokens > 1 and targets is not None:
            window_size = seqlen - self.params.num_future_tokens
            h = embeddings[:, :window_size, :]
            freqs_cos = self.freqs_cos[:window_size]
            freqs_sin = self.freqs_sin[:window_size]
        else:
            window_size = seqlen
            h = embeddings
            freqs_cos = self.freqs_cos[:seqlen]
            freqs_sin = self.freqs_sin[:seqlen]

        # Compute the main model's hidden states
        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)

        normed_h = self.norm(h)
        main_model_output = self.output(normed_h) # TODO: Norm here or not?

        if targets is not None:
            if self.params.num_future_tokens > 1:
                # Get predictions from each Module
                all_logits = [main_model_output]
                last_mtp_out = h
                for k, mtp_module in enumerate(self.mtp_modules, 1):
                    last_mtp_out = mtp_module(embeddings[:, k:window_size+k], last_mtp_out, freqs_cos, freqs_sin)
                    all_logits.append(self.output(last_mtp_out))

                # Stack logits: (batch_size, seq_len, num_predictions, vocab_size)
                stacked_logits = torch.stack(all_logits, dim=2)
                assert stacked_logits.shape == (_bsz, window_size, self.params.num_future_tokens, self.vocab_size), f"{stacked_logits.shape} != {(_bsz, window_size, self.params.num_future_tokens, self.vocab_size)}"
                assert targets.shape == (_bsz, window_size, self.params.num_future_tokens), f"{targets.shape} != {(_bsz, window_size, self.params.num_future_tokens)}"

                # Calculate loss across all predictions
                loss_fct = nn.CrossEntropyLoss()
                losses = []
                for i in range(self.params.num_future_tokens):
                    logits_reshaped = stacked_logits[:, :, i, :].view(-1, self.vocab_size)
                    targets_reshaped = targets[:, :, i].view(-1)
                    losses.append(loss_fct(logits_reshaped, targets_reshaped))
                mtp_loss = self.loss_weighting_factor * sum(losses[1:])
                self.last_loss = losses[0] + mtp_loss

                return stacked_logits

            else:
                logits = main_model_output
                self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
                return logits

        else:
            # inference-time: only forward the output on the very last position
            # and only use the first head
            last_h = h[:, [-1], :]  # (batch_size, 1, dim)
            logits = self.output(last_h)  # (batch_size, 1, vocab_size)
            self.last_loss = None
            return logits

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = sum(p.numel() for p in self.parameters())
        cfg = self.params
        L, H, Q, T = cfg.n_layers, cfg.n_heads, cfg.dim//cfg.n_heads, cfg.max_seq_len
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        Also note this is a super inefficient version of sampling with no key/value cache.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)  # (batch_size, 1, vocab_size)
            logits = logits[:, -1, :]  # crop to just the final time step
            if temperature == 0.0:
                # "sample" the single most likely index
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # pluck the logits at the final step and scale by desired temperature
                logits = logits / temperature
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


#### Experiment with Tree of MTPs
class MTPInfoMerge(nn.Module):
    """
    This module is used in the TreeMTPModule to merge the past hidden states of the main module and the MTPs.

    The possible merging strategies so far are:
    - "concat": concatenate the past hidden states of the main module and the MTPs and project to the original dimension
    - "mean": average the past hidden states of the main module and the MTPs
    """
    def __init__(self, args: ModelArgs, module_index: int):
        super().__init__()
        self.args = args
        self.module_index = module_index

        if args.mtp_info_merge == "concat":
            if module_index > 1:
                self.past_combine_proj = nn.Linear((module_index) * args.dim, args.dim)
            else:
                pass
        elif args.mtp_info_merge == "mean":
            pass
        else:
            raise ValueError(f"Invalid mtp_info_merge: {args.mtp_info_merge}")

    def forward(self, past_hidden_states: list[torch.Tensor]) -> torch.Tensor:
        if self.module_index <= 1:
            return past_hidden_states[0]

        if self.args.mtp_info_merge == "concat":
            cat_hidden = torch.cat(past_hidden_states, dim=-1)
            # print(f"cat_hidden.shape: {cat_hidden.shape}")
            # print(f"all past_hidden_states.shape: {[h.shape for h in past_hidden_states]}")
            return self.past_combine_proj(cat_hidden)

        elif self.args.mtp_info_merge == "mean":
            mean_hidden = torch.mean(torch.stack(past_hidden_states, dim=0), dim=0) # TODO: Check if this is correct (should be)
            return mean_hidden


class TreeMTPModule(nn.Module):
    """
    Enhanced MTP module that aggregates information from all previous modules including the main model
    The idea is to get even more gradients to the main model and to do it directly.
    """
    def __init__(self, args: ModelArgs, module_index: int):
        super().__init__()
        self.args = args
        self.module_index = module_index  # 0 is main model, 1 and up are MTP modules
        self.past_info_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.current_info_norm = RMSNorm(args.dim, eps=args.norm_eps)

        # Layer for combining past hidden states
        self.past_combine_proj = MTPInfoMerge(args, module_index)

        # Main processing components
        self.linear_proj = nn.Linear(2 * args.dim, args.dim)
        self.mtp_layers = nn.ModuleList([
            TransformerBlock(i, args) for i in range(args.num_mtp_layers)
        ])


    def forward(self, current_info: torch.Tensor, past_hidden_states: list[torch.Tensor],
                freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
        """
        Args:
            current_info: token embeddings shifted according to module position
            past_hidden_states: list of hidden states from all previous modules
            freqs_cos, freqs_sin: positional encoding tensors
        """
        # Align and combine past hidden states
        aligned_states = self._align_hidden_states(past_hidden_states)
        combined_past = self.past_combine_proj(aligned_states)

        # Normal MTP processing
        past_info = self.past_info_norm(combined_past)
        current_info = self.current_info_norm(current_info)
        combined = torch.cat([past_info, current_info], dim=-1)
        x = self.linear_proj(combined)

        # Process through transformer layers
        for layer in self.mtp_layers:
            x = layer(x, freqs_cos, freqs_sin)

        return x

    def _align_hidden_states(self, past_hidden_states: list[torch.Tensor]) -> torch.Tensor:
        """
        Aligns hidden states from different modules to match current positions.
        Each module processes a different window of the sequence, so we need to
        align them properly before combining.
        """
        batch_size = past_hidden_states[0].shape[0]
        seq_len = past_hidden_states[0].shape[1]

        # Align each hidden state based on its module position
        aligned_states = []
        for idx, hidden_states in enumerate(past_hidden_states):
            # Calculate position shift based on module index difference
            pos_shift = self.module_index - idx - 1 # -1 because there is no shift between neighboring modules and module_index starts at 1

            if pos_shift > 0:
                # Need to shift hidden state backward and pad at the end
                aligned = hidden_states[:, pos_shift:]
                pad_length = pos_shift
                padding = torch.zeros(batch_size, pad_length, self.args.dim, device=hidden_states.device)
                aligned = torch.cat([aligned, padding], dim=1)
            else:
                aligned = hidden_states

            aligned_states.append(aligned)

        # Concatenate all aligned states along feature dimension
        return aligned_states


class TreeTransformer(Transformer):
    """Main transformer model with tree-structured MTP modules"""
    def __init__(self, params: ModelArgs):
        # init not the father but the grandfather
        super(Transformer, self).__init__()
        assert params.num_future_tokens > 2, "Tree structure requires at least 3 future tokens (otherwise linear or simple)"
        self.params = params
        self.vocab_size = params.vocab_size
        self.weighting_factor = params.lambda_loss / (params.num_future_tokens - 1)  # lambda / D   ...   from deepseek paper


        # The main model as in Transformer
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        # Tree-structured MTP modules
        self.mtp_modules = nn.ModuleList([
            TreeMTPModule(params, module_index=i+1)
            for i in range(params.num_future_tokens - 1)
        ])

        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # share the unembedding parameters with the embedding parameters
        if not params.untied_head:
            self.tok_embeddings.weight = self.output.weight # https://paperswithcode.com/method/weight-tying

        # some useful precompute for the RoPE relative positional embeddings
        freqs_cos, freqs_sin = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * params.n_layers))

        # Initialize attribute for the loss of the last forward call. This will be set if the forward is called with a targets tensor.
        self.last_loss = None


    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        _bsz, seqlen = tokens.shape

        if targets is not None:
            window_size = seqlen - self.params.num_future_tokens

            # Get embeddings for all positions
            embeddings = self.tok_embeddings(tokens)
            embeddings = self.dropout(embeddings)

            # Process main model
            h = embeddings[:, :window_size, :]
            freqs_cos = self.freqs_cos[:window_size]
            freqs_sin = self.freqs_sin[:window_size]

            for layer in self.layers:
                h = layer(h, freqs_cos, freqs_sin)
            main_model_output = self.norm(h)

            # Store all hidden states adn logits for tree structure
            all_hidden_states = [main_model_output]
            all_logits = [self.output(main_model_output)]

            # Process MTP modules in tree structure
            for i, mtp_module in enumerate(self.mtp_modules, 1):
                current_tokens = embeddings[:, i:window_size+i]
                mtp_output = mtp_module(current_tokens, all_hidden_states, freqs_cos, freqs_sin)

                all_hidden_states.append(mtp_output)
                all_logits.append(self.output(mtp_output))

            stacked_logits = torch.stack(all_logits, dim=2)
            assert stacked_logits.shape == (_bsz, window_size, self.params.num_future_tokens, self.vocab_size), f"{stacked_logits.shape} != {(_bsz, window_size, self.params.num_future_tokens, self.vocab_size)}"
            assert targets.shape == (_bsz, window_size, self.params.num_future_tokens), f"{targets.shape} != {(_bsz, window_size, self.params.num_future_tokens)}"

            # Compute losses with dynamic weighting
            loss_fct = nn.CrossEntropyLoss()
            losses = []
            for i in range(self.params.num_future_tokens):
                logits_reshaped = stacked_logits[:, :, i, :].view(-1, self.vocab_size)
                targets_reshaped = targets[:, :, i].view(-1)
                losses.append(loss_fct(logits_reshaped, targets_reshaped))
            mtp_loss = self.weighting_factor * sum(losses[1:])
            self.last_loss = losses[0] + mtp_loss

            return stacked_logits

        else:
            # Inference mode - similar to original implementation
            return super().forward(tokens, targets)
"""
Progressive architecture study: GPT-2 → Modern
================================================
Start from GPT-2 baseline and add improvements one at a time.
Control via ARCH_LEVEL environment variable:

  ARCH_LEVEL=0  →  Pure GPT-2 (absolute pos, LayerNorm, MHA, GELU, AdamW)
  ARCH_LEVEL=1  →  + RoPE + RMSNorm
  ARCH_LEVEL=2  →  + Muon optimizer (biggest single win)
  ARCH_LEVEL=3  →  + GQA + learned attn/mlp scales
  ARCH_LEVEL=4  →  + U-Net skip connections + resid_mix
  ARCH_LEVEL=5  →  + logit softcap + ReLU² + CastedLinear (full modern)

Run:
  ARCH_LEVEL=0 torchrun --standalone --nproc_per_node=8 train_progressive.py
  ARCH_LEVEL=2 torchrun --standalone --nproc_per_node=8 train_progressive.py
  etc.
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb

ARCH_LEVEL = int(os.environ.get("ARCH_LEVEL", "5"))

# -----------------------------
# HYPERPARAMETERS
# -----------------------------

class Hyperparameters:
    data_path      = os.environ.get("DATA_PATH",      "./data/datasets/fineweb10B_sp1024")
    train_files    = os.path.join(data_path, "fineweb_train_*.bin")
    val_files      = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id         = os.environ.get("RUN_ID",         str(uuid.uuid4()))
    seed           = int(os.environ.get("SEED", 1337))

    val_batch_size  = int(os.environ.get("VAL_BATCH_SIZE",  524_288))
    val_loss_every  = int(os.environ.get("VAL_LOSS_EVERY",  4000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 500))

    iterations            = int(os.environ.get("ITERATIONS",            20000))
    warmup_steps          = int(os.environ.get("WARMUP_STEPS",          100))
    train_batch_tokens    = int(os.environ.get("TRAIN_BATCH_TOKENS",    524_288))
    train_seq_len         = int(os.environ.get("TRAIN_SEQ_LEN",         1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 3600.0))
    qk_gain_init          = float(os.environ.get("QK_GAIN_INIT",         1.0))

    # Model — same size budget across all levels for fair comparison
    vocab_size    = int(os.environ.get("VOCAB_SIZE",    1024))
    num_layers    = int(os.environ.get("NUM_LAYERS",    9))
    model_dim     = int(os.environ.get("MODEL_DIM",     512))
    num_heads     = int(os.environ.get("NUM_HEADS",     8))
    num_kv_heads  = int(os.environ.get("NUM_KV_HEADS",  4))   # used at ARCH_LEVEL >= 3
    mlp_mult      = int(os.environ.get("MLP_MULT",      4))   # GPT-2 uses 4x; modern uses 2x
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base      = float(os.environ.get("ROPE_BASE",   10000.0))
    logit_softcap  = float(os.environ.get("LOGIT_SOFTCAP", 20.0))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))

    # Optimizer
    # AdamW params (used at all levels, Muon added at level 2+)
    learning_rate  = float(os.environ.get("LEARNING_RATE",  6e-4))
    min_lr_frac    = float(os.environ.get("MIN_LR_FRAC",    0.1))
    weight_decay   = float(os.environ.get("WEIGHT_DECAY",   0.1))
    beta1          = float(os.environ.get("BETA1",          0.9))
    beta2          = float(os.environ.get("BETA2",          0.95))
    adam_eps       = float(os.environ.get("ADAM_EPS",       1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 1.0))

    # Muon params (used at ARCH_LEVEL >= 2)
    matrix_lr                  = float(os.environ.get("MATRIX_LR",                  0.05))
    scalar_lr                  = float(os.environ.get("SCALAR_LR",                  0.05))
    tied_embed_lr              = float(os.environ.get("TIED_EMBED_LR",              0.05))
    embed_lr                   = float(os.environ.get("EMBED_LR",                   0.75))
    head_lr                    = float(os.environ.get("HEAD_LR",                    0.01))
    muon_momentum              = float(os.environ.get("MUON_MOMENTUM",              0.95))
    muon_backend_steps         = int(os.environ.get("MUON_BACKEND_STEPS",           5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.9))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS",   500))

# -----------------------------
# MUON OPTIMIZER (used at ARCH_LEVEL >= 2)
# -----------------------------

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float,
                 backend_steps: int, nesterov: bool = True):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      backend_steps=backend_steps, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank       = dist.get_rank()       if distributed else 0
        for group in self.param_groups:
            params        = group["params"]
            lr            = group["lr"]
            momentum      = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov      = group["nesterov"]
            total_params  = sum(int(p.numel()) for p in params)
            updates_flat  = torch.zeros(total_params, device=params[0].device,
                                        dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g     = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr: curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            curr = 0
            for p in params:
                g = updates_flat[curr: curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss

# -----------------------------
# SHARED BPB / VAL HELPERS
# -----------------------------

CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scale", "attn_scales", "mlp_scale", "mlp_scales",
    "resid_mix", "resid_mixes", "q_gain", "skip_weight", "skip_weights",
)
INT8_KEEP_FLOAT_MAX_NUMEL   = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE    = torch.float16
INT8_CLIP_PERCENTILE        = 99.995
INT8_CLIP_Q                 = INT8_CLIP_PERCENTILE / 100.0
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = CONTROL_TENSOR_NAME_PATTERNS


def build_sentencepiece_luts(sp, vocab_size, device):
    sp_vocab_size = int(sp.vocab_size())
    table_size    = max(sp_vocab_size, vocab_size)
    base_bytes_np        = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones( (table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np,        dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np,  dtype=torch.bool,  device=device),
        torch.tensor(is_boundary_token_np,  dtype=torch.bool,  device=device),
    )


def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens    = int(header[2])
    expected_size = header_bytes + num_tokens * np.dtype("<u2").itemsize
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found: {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[: usable + 1]


class TokenStream:
    def __init__(self, pattern: str):
        self.files    = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files: {pattern}")
        self.file_idx = 0
        self.tokens   = load_data_shard(self.files[0])
        self.pos      = 0

    def _advance_file(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens   = load_data_shard(self.files[self.file_idx])
        self.pos      = 0

    def take(self, n: int) -> Tensor:
        chunks, remaining = [], n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos: self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern, rank, world_size, device):
        self.rank = rank; self.world_size = world_size; self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens, seq_len, grad_accum_steps):
        local_tokens  = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk         = self.stream.take(per_rank_span * self.world_size)
        start         = self.rank * per_rank_span
        local         = chunk[start: start + per_rank_span].to(dtype=torch.int64).pin_memory()
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


def eval_val(args, model, rank, world_size, device, grad_accum_steps,
             val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut):
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    local_batch_seqs   = local_batch_tokens  // args.train_seq_len
    total_seqs  = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start   = (total_seqs * rank)       // world_size
    seq_end     = (total_seqs * (rank + 1)) // world_size
    val_loss_sum    = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count  = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for bss in range(seq_start, seq_end, local_batch_seqs):
            bse   = min(bss + local_batch_seqs, seq_end)
            local = val_tokens[bss * args.train_seq_len: bse * args.train_seq_len + 1]
            local = local.to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            val_loss_sum    += batch_loss.to(torch.float64) * float(y.numel())
            val_token_count += float(y.numel())
            prev_ids     = x.reshape(-1)
            tgt_ids      = y.reshape(-1)
            token_bytes  = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] &
                            ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum,    op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count,  op=dist.ReduceOp.SUM)
    val_loss        = val_loss_sum / val_token_count
    bits_per_token  = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)

# -----------------------------
# QUANTIZATION (unchanged from competition script)
# -----------------------------

def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())

def keep_float_tensor(name, t, passthrough_orig_dtypes):
    if any(p in name for p in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t

def quantize_float_tensor(t):
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
                    if t32.numel() else torch.empty((t32.shape[0],)))
        clipped  = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale    = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q        = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale    = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q        = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale

def quantize_state_dict_int8(state_dict):
    quantized, scales, dtypes, passthrough, passthrough_orig_dtypes, qmeta = {}, {}, {}, {}, {}, {}
    stats = dict.fromkeys(("param_count","num_tensors","num_float_tensors",
                           "num_nonfloat_tensors","baseline_tensor_bytes","int8_payload_bytes"), 0)
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)
        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue
        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q; scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
    obj = {"__quant_format__": "int8_clean_per_row_v1",
           "quantized": quantized, "scales": scales, "dtypes": dtypes,
           "passthrough": passthrough}
    if qmeta: obj["qmeta"] = qmeta
    if passthrough_orig_dtypes: obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats

def dequantize_state_dict_int8(obj):
    out = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            out[name] = (q.float() * float(s.item())).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig  = passthrough_orig_dtypes.get(name)
        if isinstance(orig, str):
            out_t = out_t.to(dtype=getattr(torch, orig)).contiguous()
        out[name] = out_t
    return out

# -----------------------------
# ARCHITECTURE BUILDING BLOCKS
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps=None):
        super().__init__()
        self.eps = eps
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x):
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module):
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS)) \
                    and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = self._sin_cached = None

    def forward(self, seq_len, device, dtype):
        if self._cos_cached is None or self._seq_len_cached != seq_len \
                or self._cos_cached.device != device:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x, cos, sin):
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)

# -----------------------------
# ATTENTION — supports all arch levels
# -----------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, use_rope, use_qk_norm,
                 use_qk_gain, rope_base, qk_gain_init, use_casted_linear):
        super().__init__()
        assert dim % num_heads == 0
        assert num_heads % num_kv_heads == 0
        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim     = dim // num_heads
        self.use_rope     = use_rope
        self.use_qk_norm  = use_qk_norm
        self.use_qk_gain  = use_qk_gain

        Linear = CastedLinear if use_casted_linear else nn.Linear
        kv_dim = num_kv_heads * self.head_dim

        if num_kv_heads == num_heads:
            # Fused QKV for standard MHA (GPT-2 style, faster)
            self.c_attn = Linear(dim, 3 * dim, bias=False)
            self.c_q = self.c_k = self.c_v = None
        else:
            # Separate projections for GQA
            self.c_attn = None
            self.c_q = Linear(dim, dim,    bias=False)
            self.c_k = Linear(dim, kv_dim, bias=False)
            self.c_v = Linear(dim, kv_dim, bias=False)

        self.proj = Linear(dim, dim, bias=False)
        if use_casted_linear:
            self.proj._zero_init = True

        if use_qk_gain:
            self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init,
                                                   dtype=torch.float32))
        if use_rope:
            self.rotary = Rotary(self.head_dim, base=rope_base)
        else:
            # Absolute position is handled at the embedding level (GPT-2 style)
            pass

    def forward(self, x):
        B, T, C = x.shape
        if self.c_attn is not None:
            # Fused QKV (standard MHA)
            qkv = self.c_attn(x)
            q, k, v = qkv.split(C, dim=2)
            q = q.view(B, T, self.num_heads,    self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        else:
            q = self.c_q(x).view(B, T, self.num_heads,    self.head_dim).transpose(1, 2)
            k = self.c_k(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = self.c_v(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            q = F.rms_norm(q, (q.size(-1),))
            k = F.rms_norm(k, (k.size(-1),))

        if self.use_rope:
            cos, sin = self.rotary(T, x.device, q.dtype)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)

        if self.use_qk_gain:
            q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

# -----------------------------
# MLP — supports relu², gelu
# -----------------------------

class MLP(nn.Module):
    def __init__(self, dim, mlp_mult, activation, use_casted_linear):
        super().__init__()
        hidden = mlp_mult * dim
        Linear = CastedLinear if use_casted_linear else nn.Linear
        self.fc   = Linear(dim, hidden, bias=False)
        self.proj = Linear(hidden, dim,  bias=False)
        if use_casted_linear:
            self.proj._zero_init = True
        self.activation = activation  # "relu2" or "gelu"

    def forward(self, x):
        x = self.fc(x)
        if self.activation == "relu2":
            x = torch.relu(x).square()
        else:
            x = F.gelu(x, approximate='tanh')
        return self.proj(x)

# -----------------------------
# BLOCK — supports all arch levels
# -----------------------------

class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult,
                 use_rope, use_qk_norm, use_qk_gain, use_rmsnorm,
                 use_learned_scales, use_resid_mix, use_casted_linear,
                 activation, rope_base, qk_gain_init):
        super().__init__()
        Norm = RMSNorm if use_rmsnorm else nn.LayerNorm

        self.attn_norm = Norm()
        self.mlp_norm  = Norm()
        self.attn = CausalSelfAttention(
            dim, num_heads, num_kv_heads,
            use_rope, use_qk_norm, use_qk_gain,
            rope_base, qk_gain_init, use_casted_linear,
        )
        self.mlp = MLP(dim, mlp_mult, activation, use_casted_linear)

        self.use_learned_scales = use_learned_scales
        self.use_resid_mix      = use_resid_mix

        if use_learned_scales:
            self.attn_scale = nn.Parameter(torch.ones(dim,  dtype=torch.float32))
            self.mlp_scale  = nn.Parameter(torch.ones(dim,  dtype=torch.float32))
        if use_resid_mix:
            self.resid_mix = nn.Parameter(
                torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x, x0=None):
        if self.use_resid_mix and x0 is not None:
            mix = self.resid_mix.to(dtype=x.dtype)
            x   = mix[0][None, None, :] * x + mix[1][None, None, :] * x0

        attn_out = self.attn(self.attn_norm(x))
        if self.use_learned_scales:
            x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        else:
            x = x + attn_out

        mlp_out = self.mlp(self.mlp_norm(x))
        if self.use_learned_scales:
            x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * mlp_out
        else:
            x = x + mlp_out

        return x

# -----------------------------
# FULL MODEL — selects features by arch level
# -----------------------------

class ProgressiveGPT(nn.Module):
    """
    arch_level controls which features are active:
      0: pure GPT-2
      1: + RoPE, RMSNorm, qk_norm
      2: same as 1 (Muon is optimizer-side, not architecture)
      3: + GQA, learned attn/mlp scales, qk_gain
      4: + U-Net skips, resid_mix, CastedLinear
      5: + logit softcap, relu², tied embed init (full modern)
    """
    def __init__(self, args: Hyperparameters, arch_level: int):
        super().__init__()
        self.arch_level = arch_level

        # Feature flags
        use_rope           = arch_level >= 1
        use_rmsnorm        = arch_level >= 1
        use_qk_norm        = arch_level >= 1
        use_gqa            = arch_level >= 3
        use_learned_scales = arch_level >= 3
        use_qk_gain        = arch_level >= 3
        use_unet_skips     = arch_level >= 4
        use_resid_mix      = arch_level >= 4
        use_casted_linear  = arch_level >= 4
        use_logit_softcap  = arch_level >= 5
        use_relu2          = arch_level >= 5

        # At level 5 drop mlp_mult to 2 (modern default), else keep 4 (GPT-2 default)
        mlp_mult   = 2 if arch_level >= 5 else args.mlp_mult
        num_kv_heads = args.num_kv_heads if use_gqa else args.num_heads
        activation   = "relu2" if use_relu2 else "gelu"

        self.use_unet_skips    = use_unet_skips
        self.use_logit_softcap = use_logit_softcap
        self.logit_softcap     = args.logit_softcap
        self.tie_embeddings    = args.tie_embeddings
        self.use_rope          = use_rope

        self.tok_emb = nn.Embedding(args.vocab_size, args.model_dim)

        # Absolute position embedding only for GPT-2 level
        if not use_rope:
            self.pos_emb = nn.Embedding(args.train_seq_len, args.model_dim)
        else:
            self.pos_emb = None

        # U-Net skip weights
        if use_unet_skips:
            num_enc = args.num_layers // 2
            num_dec = args.num_layers - num_enc
            self.num_encoder_layers = num_enc
            self.num_decoder_layers = num_dec
            self.num_skip_weights   = min(num_enc, num_dec)
            self.skip_weights = nn.Parameter(
                torch.ones(self.num_skip_weights, args.model_dim, dtype=torch.float32))
        else:
            self.num_encoder_layers = 0
            self.num_decoder_layers = 0

        self.blocks = nn.ModuleList([
            Block(
                args.model_dim, args.num_heads, num_kv_heads, mlp_mult,
                use_rope, use_qk_norm, use_qk_gain, use_rmsnorm,
                use_learned_scales, use_resid_mix, use_casted_linear,
                activation, args.rope_base, args.qk_gain_init,
            )
            for _ in range(args.num_layers)
        ])

        Norm = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.final_norm = Norm()

        self.lm_head = (None if args.tie_embeddings
                        else (CastedLinear if use_casted_linear else nn.Linear)(
                            args.model_dim, args.vocab_size, bias=False))
        if self.lm_head is not None and use_casted_linear:
            self.lm_head._zero_init = True

        self._init_weights(args, arch_level)

    def _init_weights(self, args, arch_level):
        if arch_level >= 5:
            # Modern: tiny init for tied embeddings
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=args.tied_embed_init_std)
        else:
            # GPT-2 style init
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
            if self.pos_emb is not None:
                nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        if arch_level >= 4:
            for m in self.modules():
                if isinstance(m, CastedLinear) and getattr(m, "_zero_init", False):
                    nn.init.zeros_(m.weight)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        B, T = input_ids.shape
        x = self.tok_emb(input_ids)

        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(T, device=input_ids.device))

        if self.arch_level >= 4:
            x = F.rms_norm(x, (x.size(-1),))

        x0 = x

        if self.use_unet_skips:
            skips = []
            for i in range(self.num_encoder_layers):
                x = self.blocks[i](x, x0)
                skips.append(x)
            for i in range(self.num_decoder_layers):
                if skips:
                    x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
                x = self.blocks[self.num_encoder_layers + i](x, x0)
        else:
            for block in self.blocks:
                x = block(x, x0 if self.arch_level >= 4 else None)

        x       = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)

        if self.tie_embeddings:
            logits = F.linear(x, self.tok_emb.weight)
        else:
            logits = self.lm_head(x)

        if self.use_logit_softcap:
            logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)

        return F.cross_entropy(logits.float(), targets, reduction="mean")

# -----------------------------
# OPTIMIZER BUILDER
# -----------------------------

def build_optimizers(base_model, args, arch_level):
    """
    level < 2:  AdamW only (GPT-2 style)
    level >= 2: Muon for matrices + Adam for embeddings/scalars
    """
    if arch_level < 2:
        decay_params   = [p for n, p in base_model.named_parameters()
                          if p.ndim >= 2 and p.requires_grad]
        nodecay_params = [p for n, p in base_model.named_parameters()
                          if p.ndim <  2 and p.requires_grad]
        optimizer = torch.optim.AdamW(
            [{"params": decay_params,   "weight_decay": args.weight_decay},
             {"params": nodecay_params, "weight_decay": 0.0}],
            lr=args.learning_rate, betas=(args.beta1, args.beta2),
            eps=args.adam_eps, fused=True,
        )
        return [optimizer], "adamw"

    # Muon split
    block_named = list(base_model.blocks.named_parameters())
    matrix_params = [p for n, p in block_named
                     if p.ndim == 2 and not any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    scalar_params  = [p for n, p in block_named
                      if p.ndim < 2 or any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]

    if hasattr(base_model, "skip_weights") and base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr

    opt_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    opt_muon = Muon(matrix_params, lr=args.matrix_lr,
                    momentum=args.muon_momentum, backend_steps=args.muon_backend_steps)
    for g in opt_muon.param_groups:
        g["base_lr"] = args.matrix_lr

    opt_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    opts = [opt_tok, opt_muon, opt_scalar]

    if base_model.lm_head is not None:
        opt_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
        )
        opts.insert(1, opt_head)

    return opts, "muon"

# -----------------------------
# LR SCHEDULE
# -----------------------------

def make_lr_fns(args, arch_level):
    """Returns (lr_mul_fn, adamw_lr_fn).
    lr_mul_fn: used for Muon-style optimizers (multiplier on base_lr)
    adamw_lr_fn: used for AdamW (absolute LR)
    """
    min_lr = args.learning_rate * args.min_lr_frac

    def adamw_lr(step):
        if step < args.warmup_steps:
            return args.learning_rate * (step + 1) / args.warmup_steps
        if step >= args.iterations:
            return min_lr
        t = (step - args.warmup_steps) / max(1, args.iterations - args.warmup_steps)
        return min_lr + 0.5 * (1.0 + math.cos(math.pi * t)) * (args.learning_rate - min_lr)

    def muon_lr_mul(step, elapsed_ms):
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        # flat hold for first 5% of training after warmup
        hold_steps = args.iterations // 20
        if step < args.warmup_steps + hold_steps:
            return 1.0
        progress = (step - args.warmup_steps - hold_steps) / max(
            1, args.iterations - args.warmup_steps - hold_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return muon_lr_mul, adamw_lr

# -----------------------------
# MAIN
# -----------------------------

def main():
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    distributed  = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank         = int(os.environ.get("RANK",       "0"))
    world_size   = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank   = int(os.environ.get("LOCAL_RANK", "0"))
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8")
    grad_accum_steps = 8 // world_size
    grad_scale       = 1.0 / grad_accum_steps

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    from torch.backends.cuda import (enable_cudnn_sdp, enable_flash_sdp,
                                      enable_math_sdp, enable_mem_efficient_sdp)
    enable_cudnn_sdp(False); enable_flash_sdp(True)
    enable_mem_efficient_sdp(False); enable_math_sdp(False)

    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{args.run_id}.txt" if master_process else None

    def log0(msg, console=True):
        if not master_process: return
        if console: print(msg)
        if logfile:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"vocab_size mismatch")

    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = \
        build_sentencepiece_luts(sp, args.vocab_size, device)

    base_model = ProgressiveGPT(args, ARCH_LEVEL).to(device).bfloat16()

    if ARCH_LEVEL >= 4:
        for m in base_model.modules():
            if isinstance(m, CastedLinear):
                m.float()
        restore_low_dim_params_to_fp32(base_model)
    else:
        for m in base_model.modules():
            if isinstance(m, nn.LayerNorm):
                m.float()

    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model = (DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False,
                 gradient_as_bucket_view=True)
             if distributed else compiled_model)

    optimizers, opt_style = build_optimizers(base_model, args, ARCH_LEVEL)
    muon_lr_mul, adamw_lr = make_lr_fns(args, ARCH_LEVEL)

    n_params = sum(p.numel() for p in base_model.parameters())

    arch_description = {
        0: "pure GPT-2 (abs_pos, LayerNorm, MHA, GELU, AdamW)",
        1: "GPT-2 + RoPE + RMSNorm + QK-norm",
        2: "GPT-2 + RoPE + RMSNorm + QK-norm + MUON",
        3: "level-2 + GQA + learned attn/mlp scales + QK-gain",
        4: "level-3 + U-Net skips + resid_mix + CastedLinear",
        5: "FULL MODERN (level-4 + logit softcap + ReLU² + mlp_mult=2)",
    }[ARCH_LEVEL]

    if master_process:
        wandb.init(
            project="Llm_training_andrej_ver",
            name=f"arch_level{ARCH_LEVEL}_{int(time.time())}",
            tags=[f"arch_level_{ARCH_LEVEL}", opt_style],
            config={
                "arch_level": ARCH_LEVEL,
                "arch_description": arch_description,
                "num_layers": args.num_layers,
                "model_dim": args.model_dim,
                "num_heads": args.num_heads,
                "mlp_mult": args.mlp_mult,
                "vocab_size": args.vocab_size,
                "optimizer": opt_style,
                "iterations": args.iterations,
                "warmup_steps": args.warmup_steps,
                "train_batch_tokens": args.train_batch_tokens,
                "model_params": n_params,
                "world_size": world_size,
            },
            save_code=True,
        )

    log0(f"arch_level:{ARCH_LEVEL} — {arch_description}")
    log0(f"model_params:{n_params}  optimizer:{opt_style}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds

    def zero_grad_all():
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    # Warmup
    if args.warmup_steps > 0:
        initial_model_state = {n: t.detach().cpu().clone()
                                for n, t in base_model.state_dict().items()}
        initial_opt_states  = [copy.deepcopy(o.state_dict()) for o in optimizers]
        model.train()
        for ws in range(args.warmup_steps):
            zero_grad_all()
            for ms in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = ms == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens,
                                               args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    loss = model(x, y)
                (loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if (ws + 1) % 10 == 0 or ws + 1 == args.warmup_steps:
                log0(f"warmup_step:{ws + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, st in zip(optimizers, initial_opt_states):
            opt.load_state_dict(st)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # Main loop
    training_time_ms = 0.0
    stop_after_step  = None
    torch.cuda.synchronize()
    t0   = time.perf_counter()
    step = 0

    while True:
        last_step = (step == args.iterations or
                     (stop_after_step is not None and step >= stop_after_step))

        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args, model, rank, world_size, device, grad_accum_steps,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            log0(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                 f"train_time:{training_time_ms:.0f}ms "
                 f"step_avg:{training_time_ms / max(step, 1):.2f}ms")
            if master_process:
                wandb.log({"step": step, "val_loss": val_loss, "val_bpb": val_bpb,
                           "train_time_ms": training_time_ms,
                           "step_avg_ms": training_time_ms / max(step, 1)}, step=step)
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock_cap step:{step}/{args.iterations}")
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)

        # LR update
        if opt_style == "adamw":
            lr = adamw_lr(step)
            for opt in optimizers:
                for g in opt.param_groups:
                    g["lr"] = lr
        else:
            scale = muon_lr_mul(step, elapsed_ms)
            frac  = min(step / max(args.muon_momentum_warmup_steps, 1), 1.0)
            mom   = ((1 - frac) * args.muon_momentum_warmup_start
                     + frac * args.muon_momentum)
            for g in optimizers[1].param_groups if len(optimizers) > 1 else []:
                g["momentum"] = mom
            # Find muon optimizer (it's a Muon instance)
            for opt in optimizers:
                if isinstance(opt, Muon):
                    for g in opt.param_groups:
                        g["momentum"] = mom
                else:
                    for g in opt.param_groups:
                        if "base_lr" in g:
                            g["lr"] = g["base_lr"] * scale

        zero_grad_all()
        train_loss = torch.zeros((), device=device)

        for ms in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = ms == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens,
                                           args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()

        train_loss /= grad_accum_steps

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)

        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)

        if (args.train_log_every > 0 and
                (step <= 10 or step % args.train_log_every == 0)):
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                 f"train_time:{approx_ms:.0f}ms step_avg:{approx_ms / step:.2f}ms")
            if master_process:
                wandb.log({"step": step, "train_loss": train_loss.item(),
                           "train_time_ms": approx_ms,
                           "step_avg_ms": approx_ms / step}, step=step)

        reached_cap = max_wallclock_ms is not None and approx_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            t_cap = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(t_cap, op=dist.ReduceOp.MAX)
            reached_cap = bool(t_cap.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(f"peak memory: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    # Serialize + roundtrip
    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Code size: {code_bytes} bytes")

    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw  = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)

    if master_process:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int8.ptz")
        code_bytes       = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(f"Serialized int8+zlib: {quant_file_bytes} bytes (ratio:{ratio:.2f}x)")
        log0(f"Total submission size: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)

    q_val_loss, q_val_bpb = eval_val(
        args, model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    log0(f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f}")
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    if master_process:
        wandb.log({"final_val_bpb": q_val_bpb, "final_val_loss": q_val_loss,
                   "arch_level": ARCH_LEVEL})
        wandb.finish()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
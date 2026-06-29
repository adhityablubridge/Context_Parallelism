"""
Context Parallel GPT-2 — Pure FP32, _AttentionContextParallel ParallelStyle variant
=================================================================================
This is an experimental fork of gpt2_cp_headtail_fp32.py that uses the
private `_AttentionContextParallel` ParallelStyle instead of the legacy
`context_parallel()` context manager.

WHY: the legacy `context_parallel()` API installs the CP dispatcher only
within its `with` block. When that block exits (before `loss.backward()`),
the dispatcher is torn down — so the CP-specific backward function
`_templated_ring_attention_backward` never runs, and cross-rank dK/dV
contributions are silently dropped. The `_AttentionContextParallel`
ParallelStyle uses input/output backward hooks to keep the CP dispatcher
LIVE around the attention module's backward as well as its forward.

KNOWN CAVEATS:
- `_AttentionContextParallel` is a private API (underscore prefix)
- It uses `placement = [Replicate()]` with `run_check=False` as a hack
  (the local data is actually sequence-sharded but tagged as Replicate)
- Only flash attention backend is officially supported (we'll let efficient
  attention dispatch and hope for the best — should work in practice)
- Buffer sharding (idx, pos, targets) must be done MANUALLY since we
  removed the `with context_parallel(...)` block

Launch:
  torchrun --standalone --nnodes=1 --nproc-per-node=<N> gpt2_cp_attnstyle_fp32.py
"""

import os
import sys
import math
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.experimental._attention import (
    _cp_options,
    set_rotate_method,
    _AttentionContextParallel,
    _context_parallel_buffers,
    _generate_round_robin_indices,
)
from torch.distributed.tensor.parallel import parallelize_module

# ── Instrumentation: count CP backward invocations ─────────────────────────
# We expect the count to be NON-ZERO this time (the whole point of the
# _AttentionContextParallel ParallelStyle is to make this fire).
import torch.distributed.tensor.experimental._attention as _cp_attn

_bwd_call_count = [0]
_patched = {"bwd": False}

if hasattr(_cp_attn, '_templated_ring_attention_backward'):
    _original_bwd = _cp_attn._templated_ring_attention_backward
    def _instrumented_bwd(*args, **kwargs):
        _bwd_call_count[0] += 1
        return _original_bwd(*args, **kwargs)
    _cp_attn._templated_ring_attention_backward = _instrumented_bwd
    _patched["bwd"] = True

import atexit
def _print_counts():
    import torch.distributed as _dist
    try:
        r = _dist.get_rank()
    except Exception:
        r = -1
    print(f"\n[INSTRUMENTATION rank={r}] patched: {_patched}", flush=True)
    print(f"[INSTRUMENTATION rank={r}] CP backward called {_bwd_call_count[0]} times", flush=True)
atexit.register(_print_counts)

# Env-gated: print the ACTUAL per-ring-step SDPA q/k/v shapes inside
# _templated_ring_attention (the sub-chunking happens here, not in forward()).
# CP_PRINT_SDPA_SHAPES=1 to enable; prints the first 12 op calls on each rank.
if int(os.environ.get("CP_PRINT_SDPA_SHAPES", "0")) == 1 and hasattr(
    _cp_attn, "_templated_ring_attention"
):
    _orig_tra = _cp_attn._templated_ring_attention
    _sdpa_step_prints = [0]
    def _traced_tra(group, seq_dim, op, query, key, value, *a, **kw):
        def _traced_op(q, k, v, *oa, **okw):
            if _sdpa_step_prints[0] < 12:
                _sdpa_step_prints[0] += 1
                try:
                    import torch.distributed as _d
                    _r = _d.get_rank()
                except Exception:
                    _r = -1
                print(f"[SDPA STEP rank={_r}] q={tuple(q.shape)} k={tuple(k.shape)} "
                      f"v={tuple(v.shape)} is_causal={okw.get('is_causal')}", flush=True)
            return op(q, k, v, *oa, **okw)
        return _orig_tra(group, seq_dim, _traced_op, query, key, value, *a, **kw)
    _cp_attn._templated_ring_attention = _traced_tra
# ── end instrumentation ────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# Distributed init
# ══════════════════════════════════════════════════════════════════════════════

torch.distributed.init_process_group(backend="nccl")

local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)

cp_world_size = torch.distributed.get_world_size()
cp_rank       = torch.distributed.get_rank()
cp_group      = torch.distributed.group.WORLD

device         = torch.device("cuda", local_rank)
master_process = (cp_rank == 0)

# ── Match C++ TF32 matmul precision (ON by default; env-gated) ──
# The C++ model runs Linear GEMMs in TF32, so we default PT to TF32 too for
# parity. Default PyTorch on Ampere would use fp32 matmul (allow_tf32=False).
# Set PT_TF32=0 to force plain fp32 matmul (e.g. to isolate the TF32-vs-fp32 gap).
if int(os.environ.get("PT_TF32", "1")) == 1:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if master_process:
        print("[PT_TF32] allow_tf32=True (matching C++ TF32 matmul precision)")

torch.manual_seed(1234)

# ── Context Parallel mesh + options ──
cp_mesh = init_device_mesh("cuda", (cp_world_size,))
_cp_options.enable_load_balance = True
_cp_options.convert_to_f32      = True
# Rotate method: ROTATE_METHOD = alltoall | allgather (PyTorch CP supports two).
ROTATE_METHOD = os.environ.get("ROTATE_METHOD", "alltoall").lower()
if ROTATE_METHOD not in ("alltoall", "allgather"):
    ROTATE_METHOD = "alltoall"
set_rotate_method(ROTATE_METHOD)


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════
nsys_report =False

@dataclass
class GPTConfig:
    n_embd:       int  = 768
    block_size:   int  = 1024
    vocab_size:   int  = 50304
    n_layer:      int  = 12
    n_head:       int  = 12
    weight_tying: bool = False


# ══════════════════════════════════════════════════════════════════════════════
# CUDA Timer
# ══════════════════════════════════════════════════════════════════════════════

class CudaTimer:
    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event   = torch.cuda.Event(enable_timing=True)
    def start(self):
        self.start_event.record()
    def elapsed_seconds(self):
        self.end_event.record()
        torch.cuda.synchronize()
        return self.start_event.elapsed_time(self.end_event) / 1000.0


# ══════════════════════════════════════════════════════════════════════════════
# Attention — same as headtail variant, but no internal context_parallel block
# ══════════════════════════════════════════════════════════════════════════════

class CPAttention(nn.Module):
    """
    Self-attention module. Forward takes a single tensor x (already sequence-
    sharded by the caller). The _AttentionContextParallel ParallelStyle wraps
    this module's forward/backward with DTensor and the CP dispatcher.
    """
    _raw_dumped = False  # class-level: dump raw SDPA output once (block 0, DUMP_FWD)

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head   = config.n_head
        self.n_embd   = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.ln     = nn.LayerNorm(config.n_embd)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)

        residual_std = 0.02 / math.sqrt(2.0 * config.n_layer)
        self.c_proj  = nn.Linear(config.n_embd, config.n_embd, bias=True)

        nn.init.normal_(self.c_attn.weight, std=0.02)
        nn.init.zeros_(self.c_attn.bias)
        nn.init.normal_(self.c_proj.weight, std=residual_std)
        nn.init.zeros_(self.c_proj.bias)

        self.t_attn = 0.0
        self._ts = torch.cuda.Event(enable_timing=True)
        self._te = torch.cuda.Event(enable_timing=True)

    def reset_t_attn(self):
        self.t_attn = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Path A: all surrounding ops are plain local tensors (no DTensor
        # weights). Only SDPA itself is wrapped as DTensor inside an active
        # _enable_cp_dispatcher() so the CP handler runs ring forward AND
        # registers the backward node that routes through
        # _templated_ring_attention_backward.
        from torch.distributed.tensor import DTensor as _DT, Replicate as _Rep
        from torch.distributed.tensor.experimental._attention import (
            _enable_cp_dispatcher,
        )

        B, T_local, C = x.size()

        h   = self.ln(x)
        qkv = self.c_attn(h)
        q, k, v = qkv.split(self.n_embd, dim=2)

        def to_heads(t):
            return t.view(B, T_local, self.n_head, self.head_dim).transpose(1, 2).contiguous()

        q = to_heads(q)
        k = to_heads(k)
        v = to_heads(v)

        self._ts.record()

        # Enter dispatcher manually for SDPA forward
        cp_fwd_ctx = _enable_cp_dispatcher()
        cp_fwd_ctx.__enter__()
        try:
            q_dt = _DT.from_local(q, cp_mesh, [_Rep()], run_check=False)
            k_dt = _DT.from_local(k, cp_mesh, [_Rep()], run_check=False)
            v_dt = _DT.from_local(v, cp_mesh, [_Rep()], run_check=False)
            y_dt = F.scaled_dot_product_attention(
                q_dt, k_dt, v_dt, attn_mask=None, dropout_p=0.0, is_causal=True,
            )
        finally:
            cp_fwd_ctx.__exit__(None, None, None)

        # Backward bracket: re-enter dispatcher when grad arrives at y_dt,
        # exit after SDPA backward produces grad for q_dt. Register exit on
        # q_dt only (k_dt/v_dt grads land in the same backward step; one
        # exit is enough).
        bwd_state = {'ctx': None}

        def _enter_bwd(_grad):
            bwd_state['ctx'] = _enable_cp_dispatcher()
            bwd_state['ctx'].__enter__()
            return None  # do not modify the grad

        def _exit_bwd(_grad):
            ctx = bwd_state['ctx']
            if ctx is not None:
                ctx.__exit__(None, None, None)
                bwd_state['ctx'] = None
            return None

        # Only register hooks when grad is being tracked (skipped under no_grad
        # / inference paths like validation).
        if y_dt.requires_grad:
            y_dt.register_hook(_enter_bwd)
            q_dt.register_hook(_exit_bwd)

        self._te.record()
        torch.cuda.synchronize()
        self.t_attn += self._ts.elapsed_time(self._te) / 1000.0

        # Back to local for trailing ops. to_local() is autograd-aware so
        # grad will flow back into y_dt (firing _enter_bwd) before SDPA's
        # backward node consumes it.
        y = y_dt.to_local()
        y = y.transpose(1, 2).contiguous().view(B, T_local, C)
        if int(os.environ.get("DUMP_FWD", "0")) == 1 and not CPAttention._raw_dumped:
            # Raw SDPA output (block 0), before c_proj + residual — isolates the
            # attention operator from the projection.
            np.save("fwd_sdpa_pt.npy", y.detach().to(torch.float32).cpu().numpy())
            CPAttention._raw_dumped = True
            print("[DUMP_FWD] saved fwd_sdpa_pt.npy (raw attn output, pre-c_proj)",
                  flush=True)
        return x + self.c_proj(y)


# ══════════════════════════════════════════════════════════════════════════════
# MLP
# ══════════════════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        residual_std = 0.02 / math.sqrt(2.0 * config.n_layer)
        self.ln     = nn.LayerNorm(config.n_embd)
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=True)
        self.gelu   = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=True)

        nn.init.normal_(self.c_fc.weight,   std=0.02)
        nn.init.zeros_(self.c_fc.bias)
        nn.init.normal_(self.c_proj.weight, std=residual_std)
        nn.init.zeros_(self.c_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        h = self.c_fc(h)
        h = self.gelu(h)
        h = self.c_proj(h)
        return x + h


# ══════════════════════════════════════════════════════════════════════════════
# Transformer Block
# ══════════════════════════════════════════════════════════════════════════════

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attn = CPAttention(config)
        self.mlp  = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        torch.cuda.nvtx.range_push("Attention")
        x = self.attn(x)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("MLP")
        x = self.mlp(x)
        torch.cuda.nvtx.range_pop()
        return x


# ══════════════════════════════════════════════════════════════════════════════
# GPT Model — no `with context_parallel(...)` block. Sharding done by caller.
# ══════════════════════════════════════════════════════════════════════════════

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.t_tok_emb = 0.0
        self.t_pos_emb = 0.0
        self.t_attn    = 0.0
        self.t_mlp     = 0.0
        self.t_ln_f    = 0.0
        self.t_lm_head = 0.0

        self.transformer = nn.ModuleDict({
            'wte':  nn.Embedding(config.vocab_size, config.n_embd),
            'wpe':  nn.Embedding(config.block_size, config.n_embd),
            'ln_f': nn.LayerNorm(config.n_embd),
            'h':    nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        })

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.weight_tying:
            self.lm_head.weight = self.transformer['wte'].weight
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.transformer['wte'].weight, std=0.02)
        nn.init.normal_(self.transformer['wpe'].weight, std=0.02)
        if not self.config.weight_tying:
            nn.init.normal_(self.lm_head.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def collect_attn_timing(self):
        self.t_attn = sum(b.attn.t_attn for b in self.transformer['h'])

    def reset_timing(self):
        self.t_tok_emb = 0.0
        self.t_pos_emb = 0.0
        self.t_attn    = 0.0
        self.t_mlp     = 0.0
        self.t_ln_f    = 0.0
        self.t_lm_head = 0.0
        for b in self.transformer['h']:
            b.attn.reset_t_attn()

    def forward(self, idx: torch.Tensor, targets=None):
        """
        Forward pass. Takes full (idx, targets) of shape [B, T] and shards
        them internally along the sequence dimension (HeadTail-rearranged if
        LB is enabled). Same external contract as gpt2_cp_headtail_fp32.py.
        """
        B, T = idx.size()

        # Shard buffers along sequence dim. This replaces the legacy
        # `with context_parallel(buffers=...)` block — same effect, but
        # without installing the dispatcher (which would exit before
        # backward). _AttentionContextParallel handles the dispatcher
        # per-attention-module with backward hooks.
        if _cp_options.enable_load_balance:
            load_balance_indices = _generate_round_robin_indices(
                seq_length=T,
                cp_world_size=cp_world_size,
                device=idx.device,
            )
        else:
            load_balance_indices = None

        # Global positions [B, T]; sharded through the SAME LB machinery as idx
        # so each rank embeds the correct GLOBAL position of its local tokens.
        # (Previously pos = arange(0, T_local) used LOCAL indices, which at ws>1
        # gave every rank positions 0..T_local-1 -> wpe[T_local:] received zero
        # gradient and second-half tokens got the wrong positional encoding.)
        pos_full = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_full = pos_full.unsqueeze(0).expand(B, -1).contiguous()

        buffers = [idx]
        buffer_seq_dims = [1]
        if targets is not None:
            buffers.append(targets)
            buffer_seq_dims.append(1)
        buffers.append(pos_full)
        buffer_seq_dims.append(1)

        sharded = _context_parallel_buffers(
            cp_mesh, buffers, buffer_seq_dims, load_balance_indices
        )
        idx = sharded[0]
        if targets is not None:
            targets = sharded[1]
            pos = sharded[2]
        else:
            pos = sharded[1]

        T_local = idx.size(1)

        _t  = torch.cuda.Event(enable_timing=True)
        _t2 = torch.cuda.Event(enable_timing=True)

        torch.cuda.nvtx.range_push("TokEmb")
        _t.record()
        tok_emb = self.transformer['wte'](idx)
        _t2.record(); torch.cuda.synchronize()
        self.t_tok_emb += _t.elapsed_time(_t2) / 1000.0
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("PosEmb")
        _t.record()
        pos_emb = self.transformer['wpe'](pos)
        _t2.record(); torch.cuda.synchronize()
        self.t_pos_emb += _t.elapsed_time(_t2) / 1000.0
        torch.cuda.nvtx.range_pop()

        x = tok_emb + pos_emb

        # Env-gated forward-parity dump (ws=1 base-model check). Captures ONCE.
        _dump_fwd = int(os.environ.get("DUMP_FWD", "0")) == 1 and not getattr(
            GPT, "_fwd_dumped", False)
        if _dump_fwd:
            np.save("fwd_emb_pt.npy", x.detach().to(torch.float32).cpu().numpy())
            print(f"[DUMP_FWD] embedded idx[:8]={idx.reshape(-1)[:8].tolist()} "
                  f"pos[:8]={pos.reshape(-1)[:8].tolist()}", flush=True)

        _t.record()
        for i, block in enumerate(self.transformer['h']):
            torch.cuda.nvtx.range_push(f"Block_{i}")
            if _dump_fwd and i == 0:
                # Intra-block-0 bisection: capture post-attention (pre-MLP) too.
                x = block.attn(x)
                np.save("fwd_blk0attn_pt.npy", x.detach().to(torch.float32).cpu().numpy())
                x = block.mlp(x)
            else:
                x = block(x)
            torch.cuda.nvtx.range_pop()
            if _dump_fwd and i == 0:
                np.save("fwd_blk0_pt.npy", x.detach().to(torch.float32).cpu().numpy())
        _t2.record(); torch.cuda.synchronize()
        self.t_mlp += _t.elapsed_time(_t2) / 1000.0

        torch.cuda.nvtx.range_push("LN_Final")
        _t.record()
        x = self.transformer['ln_f'](x)
        if _dump_fwd:
            np.save("fwd_lnf_pt.npy", x.detach().to(torch.float32).cpu().numpy())
            GPT._fwd_dumped = True
            print("[DUMP_FWD] saved fwd_emb_pt.npy, fwd_blk0_pt.npy, fwd_lnf_pt.npy",
                  flush=True)
        _t2.record(); torch.cuda.synchronize()
        self.t_ln_f += _t.elapsed_time(_t2) / 1000.0
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("LMHead")
        _t.record()
        logits = self.lm_head(x)
        _t2.record(); torch.cuda.synchronize()
        self.t_lm_head += _t.elapsed_time(_t2) / 1000.0
        torch.cuda.nvtx.range_pop()

        loss = None
        loss_log = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
            loss_log = loss.detach().clone()
            torch.distributed.all_reduce(
                loss_log, op=torch.distributed.ReduceOp.SUM, group=cp_group
            )
            loss_log = loss_log / cp_world_size

        return logits, loss, loss_log

    def configure_optimizers(self, weight_decay, learning_rate):
        param_dict     = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params   = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params,   'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        return torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95),
            eps=1e-8, fused=False, foreach=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Data Loader (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def load_tokens(filename):
    npt = np.fromfile(filename, dtype=np.uint16).astype(np.int32)
    return torch.tensor(npt, dtype=torch.long)


class DataLoaderLite:
    def __init__(self, B, T, split):
        self.B = B
        self.T = T
        data_root = "/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/Data_Loader/Data"
        shards = sorted(
            os.path.join(data_root, s)
            for s in os.listdir(data_root)
            if split in s
        )
        assert len(shards) > 0, f"No '{split}' shards found in {data_root}"
        self.shards = shards
        self.reset()
    def reset(self):
        self.current_shard    = 0
        self.tokens           = load_tokens(self.shards[self.current_shard])
        self.current_position = 0
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T
        if self.current_position + B * T + 1 > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = 0
        return x, y


# ══════════════════════════════════════════════════════════════════════════════
# Buffer sharding helper — replaces the legacy `with context_parallel(...)` block
# ══════════════════════════════════════════════════════════════════════════════

def shard_for_cp(idx: torch.Tensor, targets: torch.Tensor):
    """
    HeadTail-rearrange + sequence-shard idx and targets across cp_mesh.
    Returns local shards on each rank. Mirrors what context_parallel() does
    to the buffers internally.
    """
    seq_length = idx.shape[1]
    if _cp_options.enable_load_balance:
        load_balance_indices = _generate_round_robin_indices(
            seq_length=seq_length,
            cp_world_size=cp_world_size,
            device=idx.device,
        )
    else:
        load_balance_indices = None

    buffers = [idx, targets]
    buffer_seq_dims = [1, 1]
    sharded = _context_parallel_buffers(
        cp_mesh, buffers, buffer_seq_dims, load_balance_indices
    )
    return sharded[0], sharded[1]


# ══════════════════════════════════════════════════════════════════════════════
# Training Setup
# ══════════════════════════════════════════════════════════════════════════════

# MODEL_44M=1 selects the small (~44M) config; mirrors EVERYTHING C++
# fourtyfour=true changes: dims (n_embd=384, n_layer=3, n_head=6), global batch
# (65536), and max_steps (6768). Default is the ~161M config (768/12/12).
_is_44m = int(os.environ.get("MODEL_44M", "0")) == 1

MODEL_TAG = "44M" if _is_44m else "161M"

# ── Memory-scaling sweep parametrization (env-driven) ──────────────────────
# All optional; defaults preserve the original behavior. Used by the memory
# occupancy sweep harness (mem_scaling) to vary T and the model config without
# editing this file per run.
#   T              : sequence length (default 2048; sweep 1024,2048,4096,...)
#   N_EMBD/N_LAYER/N_HEAD : override model dims (else fall back to 44M/161M)
#   WEIGHT_TYING   : 1/0 to tie lm_head to wte (overrides config default)
#   MODEL_LABEL    : free-text label embedded in snapshots/logs (e.g. "124M")
#   MEM_PROBE      : 1 = run exactly MEM_PROBE_STEPS steps (grad_accum=1), skip
#                    validation/token-gen, snapshot nvidia-smi + peak mem, exit.
#   MEM_PROBE_STEPS: steps to run in probe mode (default 2).
_env_T          = os.environ.get("T")
_env_n_embd     = os.environ.get("N_EMBD")
_env_n_layer    = os.environ.get("N_LAYER")
_env_n_head     = os.environ.get("N_HEAD")
_env_tying      = os.environ.get("WEIGHT_TYING")
_mem_probe      = int(os.environ.get("MEM_PROBE", "0")) == 1
_mem_probe_steps = int(os.environ.get("MEM_PROBE_STEPS", "2"))
MODEL_LABEL     = os.environ.get("MODEL_LABEL", MODEL_TAG)

B = 4
T = int(_env_T) if _env_T else 2048
total_batch_size = 65536 if _is_44m else 524288   # C++ global_batch
# In probe mode we only care about per-microstep peak memory, so force
# grad_accum_steps = 1 (one micro-batch of B*T tokens). Activation memory of a
# normal grad-accum loop is identical (the loop is sequential), so this is a
# faithful memory measurement that runs fast.
if _mem_probe:
    total_batch_size = B * T

# ── Sequence-length sanity checks (generalize T past the default 1024) ──
# 1. CP shards the sequence across ranks, so T must divide evenly.
assert T % cp_world_size == 0, (
    f"T={T} must be divisible by cp_world_size={cp_world_size}"
)
# 2. grad_accum is integer token bookkeeping; B*T must divide the global batch.
assert total_batch_size % (B * T) == 0, (
    f"B*T={B * T} must divide total_batch_size={total_batch_size}"
)
grad_accum_steps = total_batch_size // (B * T)

# block_size (== positional-embedding table rows, wpe) must be >= T, otherwise
# pos = arange(0, T) indexes past the end of wpe and the embedding lookup throws
# a device-side assert. Derive it from T so any sequence length "just works".
# NOTE: changing block_size changes wpe's shape -> the model has DIFFERENT params
# and is NOT parity-compatible with init_weights_named_*.bin or the C++ ref
# (both assume block_size=1024). Override with BLOCK_SIZE env if you need a
# fixed table larger than T (e.g. to keep the 1024 layout while testing T<1024).
_block_size = int(os.environ.get("BLOCK_SIZE", str(max(1024, T))))
assert _block_size >= T, f"block_size={_block_size} must be >= T={T}"

# Base dims from the 44M/161M presets, then apply any per-dim env overrides so
# the sweep can hit arbitrary configs (25M/124M/etc) by toggling tying.
if _is_44m:
    _base_embd, _base_layer, _base_head, _base_tie = 384, 3, 6, False
else:
    _base_embd, _base_layer, _base_head, _base_tie = 768, 12, 12, False

_cfg_embd  = int(_env_n_embd)  if _env_n_embd  else _base_embd
_cfg_layer = int(_env_n_layer) if _env_n_layer else _base_layer
_cfg_head  = int(_env_n_head)  if _env_n_head  else _base_head
_cfg_tie   = (_env_tying == "1") if _env_tying is not None else _base_tie

config = GPTConfig(n_embd=_cfg_embd, block_size=_block_size, vocab_size=50304,
                   n_layer=_cfg_layer, n_head=_cfg_head, weight_tying=_cfg_tie)
model = GPT(config)
model.to(device)

# ── Path A: manual CP dispatcher bracketing around SDPA only ───────────────
# We do NOT call parallelize_module(_AttentionContextParallel) — that would
# convert ln/c_attn/c_proj params to DTensors and trip the incomplete view
# sharding rules in PT 2.9 (see PT source TODO: "this should be Shard(2),
# need to fix Linear layer rules"). Instead each attention module's forward
# manually enters _enable_cp_dispatcher() around the SDPA op only, wraps
# q/k/v as DTensors at that point, and registers backward hooks so the
# dispatcher is re-active when SDPA's backward node fires. All other ops
# (LN, c_attn, c_proj, residual) run as plain local tensors.

for _i, _blk in enumerate(model.transformer['h']):
    setattr(_blk.attn, '_cp_layer_idx', _i)
# ───────────────────────────────────────────────────────────────────────────

if master_process:
    print("=== GPT-2 CP via _AttentionContextParallel (FP32, HeadTail LB) ===")
    print(f"  Applied _AttentionContextParallel to {config.n_layer} blocks")

num_params         = sum(p.numel() for p in model.parameters())
num_params_per_gpu = num_params
max_steps    = 6768 if _is_44m else 1555   # C++ fourtyfour max_steps
if _mem_probe:
    max_steps = _mem_probe_steps
warmup_steps = max(1, max_steps // 10)

if nsys_report:
    max_steps = 1
    warmup_steps = 0

max_lr = 6e-4
min_lr = max_lr * 0.1
VAL_FREQ = 100

if master_process:
    print(f"Configuration:")
    print(f"  vocab_size:     {config.vocab_size}")
    print(f"  context_length: {config.block_size}")
    print(f"  n_embd:         {config.n_embd}")
    print(f"  n_layers:       {config.n_layer}")
    print(f"  n_heads:        {config.n_head}")
    print(f"  B={B}, T={T}")
    print(f"  cp_world_size:  {cp_world_size}")
    print(f"  global_batch:   {total_batch_size}")
    print(f"  grad_accum_steps: {grad_accum_steps}")
    print(f"  Parameters:          {num_params}")
    print(f"  max_steps:      {max_steps}")
    print(f"  warmup_steps:   {warmup_steps}")
    print(f"  CP API:         _AttentionContextParallel (ParallelStyle)")

optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr)

# ── Save init weights for C++ parity check ──
if int(os.environ.get("SAVE_INIT_WEIGHTS", "0")) == 1:
    if master_process:
        print("Saving init weights to init_weights.bin")
    import io
    buf = io.BytesIO()
    for p in model.parameters():
        p_cpu = p.detach().to(torch.float32).cpu()
        buf.write(p_cpu.numpy().tobytes())
    with open("init_weights.bin", "wb") as f:
        f.write(buf.getvalue())
    if master_process:
        print("  saved init_weights.bin")

# ── Save NAME-KEYED init weights (correct parity: positional load scrambles
#    because C++ enumerates params in a different order). Record format:
#    <int32 name_len><name><int32 ndim><int64 dims...><float32 data (PT layout)>
if int(os.environ.get("SAVE_INIT_NAMED", "0")) == 1 and master_process:
    import struct
    _named_fn = f"init_weights_named_{MODEL_TAG}.bin"
    with open(_named_fn, "wb") as f:
        n = 0
        for name, p in model.named_parameters():
            arr = p.detach().to(torch.float32).cpu().numpy()
            nm = name.encode()
            f.write(struct.pack("<i", len(nm)))
            f.write(nm)
            f.write(struct.pack("<i", arr.ndim))
            for d in arr.shape:
                f.write(struct.pack("<q", int(d)))
            f.write(arr.tobytes())
            n += 1
    print(f"  saved {_named_fn} ({n} named params)")


# ══════════════════════════════════════════════════════════════════════════════
# LR Schedule
# ══════════════════════════════════════════════════════════════════════════════

def get_lr(step):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# ══════════════════════════════════════════════════════════════════════════════
# Data Loaders + CSV Logging
# ══════════════════════════════════════════════════════════════════════════════

train_loader = DataLoaderLite(B, T, "train")
val_loader   = DataLoaderLite(B, T, "val")

log_file = None
log_filename = ""

if master_process:
    os.makedirs("Pytorch_CP_AttnStyle_FP32_Training_logs", exist_ok=True)
    log_idx = 1
    while True:
        log_filename = f"Pytorch_CP_AttnStyle_FP32_Training_logs/Pytorch_CP_AttnStyle_FP32_Training_log{log_idx}.csv"
        if not os.path.exists(log_filename):
            break
        log_idx += 1
    print(f"Saving logs to: {log_filename}")
    log_file = open(log_filename, 'w', newline='')
    log_file.write(
        "step,loss,val_loss,lr,grad_norm,dt_ms,tok_per_sec,"
        "timer_data,timer_fwd,timer_loss,timer_bwd,timer_clip,timer_optim,"
        "timer_tok_emb,timer_pos_emb,timer_attn_cp,timer_mlp,timer_ln_f,timer_lm_head\n"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Step Timers
# ══════════════════════════════════════════════════════════════════════════════

timer_step  = CudaTimer()
timer_data  = CudaTimer()
timer_fwd   = CudaTimer()
timer_bwd   = CudaTimer()
timer_clip  = CudaTimer()
timer_optim = CudaTimer()


# ══════════════════════════════════════════════════════════════════════════════
# Training Loop
# ══════════════════════════════════════════════════════════════════════════════

if master_process:
    print("\nStarting training...")

# ── Step 0 gradient dump (for C++ parity) ──
if int(os.environ.get("DUMP_STEP0_GRADS", "0")) == 1:
    if master_process:
        print("Running step 0 to capture gradients...")
    model.train()
    optimizer.zero_grad()
    x, y = train_loader.next_batch()
    x = x.to(device)
    y = y.to(device)
    if master_process:
        print(f"[DUMP] first 8 token ids: {x.view(-1)[:8].tolist()}")
    logits, loss, loss_log = model(x, y)
    # Single micro-batch, UNSCALED (no 1/grad_accum) to match C++ dump pass.
    loss.backward()
    # AVG-all-reduce PARAM grads across ranks so the dumped gradient is the
    # FULL-batch gradient (sum over all 1024 tokens), independent of how each
    # impl splits tokens across ranks. Required for a valid per-element compare:
    # rank0's local shard differs between PT and C++ HeadTail LB.
    for _p in model.parameters():
        if _p.grad is not None:
            torch.distributed.all_reduce(
                _p.grad, op=torch.distributed.ReduceOp.AVG, group=cp_group)

    if master_process:
        print("Dumping step 0 gradients:")
        import pickle
        dump = {}
        for name, p in model.named_parameters():
            if p.grad is not None:
                dump[name] = p.grad.detach().to(torch.float32).cpu().numpy()
        with open("step0_grads_pt_attnstyle.pkl", "wb") as f:
            pickle.dump(dump, f)
        print("  saved step0_grads_pt_attnstyle.pkl")
        # Raw arrays for cosine check (PT native layout: weight is [out,in])
        np.save("raw_ln_f_weight_pt.npy", dump["transformer.ln_f.weight"])
        np.save("raw_h0_c_attn_weight_pt.npy",
                dump["transformer.h.0.attn.c_attn.weight"])
        print("  saved raw_ln_f_weight_pt.npy, raw_h0_c_attn_weight_pt.npy")

    # Reset for actual training
    train_loader.reset()
    optimizer.zero_grad()
    model.train()

val_loss_accum_log = -1.0

for step in range(max_steps):
    timer_step.start()

    # ---- Validation ---- (skipped entirely in memory-probe mode)
    if not _mem_probe and (step % VAL_FREQ == 0 or step == max_steps - 1):
        model.eval()
        val_loader.reset()
        val_loss_accum = 0.0
        val_loss_steps = 5
        with torch.no_grad():
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x = x.to(device)
                y = y.to(device)
                # NOTE: do NOT shard here. GPT.forward shards once internally
                # (pre-embedding), matching legacy/C++. Sharding here too would
                # double-shard (1024->512->256).
                _, _, loss_log = model(x, y)
                val_loss_accum += loss_log.item() / val_loss_steps
        if master_process:
            print(f"validation loss: {val_loss_accum:.4f}")
        val_loss_accum_log = val_loss_accum

    # ---- Training ----
    model.train()
    optimizer.zero_grad()
    loss_accum    = 0.0
    time_data     = 0.0
    time_forward  = 0.0
    time_loss     = 0.0
    time_backward = 0.0
    model.reset_timing()

    for micro_step in range(grad_accum_steps):
        torch.cuda.nvtx.range_push("DataLoad")
        timer_data.start()
        x, y = train_loader.next_batch()
        x = x.to(device)
        y = y.to(device)
        # NOTE: do NOT shard here. GPT.forward shards once internally
        # (pre-embedding), matching legacy/C++. Sharding here too would
        # double-shard (1024->512->256).
        time_data += timer_data.elapsed_seconds()
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("Forward")
        timer_fwd.start()
        logits, loss, loss_log = model(x, y)
        time_forward += timer_fwd.elapsed_seconds()
        torch.cuda.nvtx.range_pop()

        loss_accum += loss_log.item() / grad_accum_steps

        torch.cuda.nvtx.range_push("Backward")
        timer_bwd.start()
        (loss / grad_accum_steps).backward()
        time_backward += timer_bwd.elapsed_seconds()
        torch.cuda.nvtx.range_pop()

    model.collect_attn_timing()

    # [GRAD PARITY DUMP] step 0 only: save first-layer c_attn.weight.grad
    # (accumulated over all micro-steps, before clip/opt) so we can compare
    # the CP gradient against the single-GPU exact reference. AVG of rank0+rank1
    # should equal the single-GPU grad IFF the CP backward is mathematically
    # correct. Tagged: attnstyle, cp<world>, rank<r>.
    if step == 0 and int(os.environ.get("GRAD_PARITY_DUMP", "0")) == 1:
        _g = model.transformer['h'][0].attn.c_attn.weight.grad
        _gp = _g.detach().to(torch.float32).cpu().numpy()
        _fn = f"gradparity_attnstyle_cp{cp_world_size}_rank{cp_rank}.npy"
        np.save(_fn, _gp)
        print(f"[GRAD PARITY DUMP] saved {_fn} shape={_gp.shape} "
              f"L2={float((_gp**2).sum()**0.5):.6e}", flush=True)

    # ---- Optional param-grad AVG all-reduce across CP ranks (toggle) ----
    # CP_GRAD_ALLREDUCE=1 syncs param grads (AVG) so both replicas stay
    # identical. Default OFF (matches the curve runs). Use to A/B the
    # "divergent-replica ensemble" hypothesis for Q1.
    if int(os.environ.get("CP_GRAD_ALLREDUCE", "0")) == 1:
        for p in model.parameters():
            if p.grad is not None:
                torch.distributed.all_reduce(
                    p.grad, op=torch.distributed.ReduceOp.AVG, group=cp_group
                )

    torch.cuda.nvtx.range_push("GradClip")
    timer_clip.start()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    time_clip = timer_clip.elapsed_seconds()
    torch.cuda.nvtx.range_pop()

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    torch.cuda.nvtx.range_push("Optimizer")
    timer_optim.start()
    optimizer.step()
    time_optim = timer_optim.elapsed_seconds()
    torch.cuda.nvtx.range_pop()

    torch.cuda.synchronize()
    dt = timer_step.elapsed_seconds()

    tokens_processed = B * T * grad_accum_steps
    tokens_per_sec   = tokens_processed / dt

    total_sec = int((max_steps - step) * dt)
    h = total_sec // 3600
    m = (total_sec % 3600) // 60

    if master_process:
        print(
            f"step {step:5d} | loss: {loss_accum:.6f} | lr {lr:.4e} "
            f"| norm: {norm:.4f} | dt: {dt * 1000.0:.2f}ms "
            f"| tok/sec: {tokens_per_sec:.1f} "
            f"| Time Left: {h:02d} hrs : {m:02d} mins"
        )
        print(
            f"  [TIMING] data: {time_data * 1000.0:.1f}ms"
            f" | fwd: {time_forward * 1000.0:.1f}ms"
            f" | bwd: {time_backward * 1000.0:.1f}ms"
            f" | clip: {time_clip * 1000.0:.1f}ms"
            f" | optim: {time_optim * 1000.0:.1f}ms"
        )
        print(
            f"  [LAYER] tok_emb: {model.t_tok_emb * 1000.0:.1f}ms"
            f" | pos_emb: {model.t_pos_emb * 1000.0:.1f}ms"
            f" | attn_cp: {model.t_attn * 1000.0:.1f}ms"
            f" | mlp: {model.t_mlp * 1000.0:.1f}ms"
            f" | ln_f: {model.t_ln_f * 1000.0:.1f}ms"
            f" | lm_head: {model.t_lm_head * 1000.0:.1f}ms"
        )

        if log_file:
            log_file.write(
                f"{step},{loss_accum:.6f},{val_loss_accum_log:.6f},"
                f"{lr},{norm},{dt * 1000.0},{tokens_per_sec},"
                f"{time_data * 1000.0},{time_forward * 1000.0},"
                f"0.0,{time_backward * 1000.0},"
                f"{time_clip * 1000.0},{time_optim * 1000.0},"
                f"{model.t_tok_emb * 1000.0},{model.t_pos_emb * 1000.0},"
                f"{model.t_attn * 1000.0},{model.t_mlp * 1000.0},"
                f"{model.t_ln_f * 1000.0},{model.t_lm_head * 1000.0}\n"
            )
            log_file.flush()

    val_loss_accum_log = -1.0

    # ── Memory-probe snapshot: after the final probe step, capture nvidia-smi
    #    + per-rank torch peak memory, then stop. ───────────────────────────
    if _mem_probe and step == max_steps - 1:
        torch.cuda.synchronize()
        peak_alloc_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        peak_resv_mb  = torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0)
        snap_dir = os.environ.get("MEM_SNAPSHOT_DIR", "mem_scaling_runs")
        tag = f"PT_{MODEL_LABEL}_{ROTATE_METHOD}_T{T}_ws{cp_world_size}"
        # All ranks print their peak so the harness can read per-GPU usage.
        print(f"[MEM_PROBE rank={cp_rank}] tag={tag} "
              f"peak_alloc_mb={peak_alloc_mb:.1f} peak_reserved_mb={peak_resv_mb:.1f}",
              flush=True)
        if master_process:
            os.makedirs(snap_dir, exist_ok=True)
            snap_path = os.path.join(snap_dir, f"{tag}.txt")
            import subprocess
            with open(snap_path, "w") as sf:
                sf.write(f"# MEM PROBE SNAPSHOT  tag={tag}\n")
                sf.write(f"# impl=PyTorch label={MODEL_LABEL} rotator={ROTATE_METHOD} "
                         f"n_embd={config.n_embd} n_layer={config.n_layer} "
                         f"n_head={config.n_head} weight_tying={config.weight_tying}\n")
                sf.write(f"# B={B} T={T} cp_world_size={cp_world_size} "
                         f"params={num_params}\n")
                sf.write(f"# torch.peak_alloc_mb(rank0)={peak_alloc_mb:.1f} "
                         f"torch.peak_reserved_mb(rank0)={peak_resv_mb:.1f}\n")
                # Machine-readable live per-GPU used MiB (parsed by the table gen).
                # nvidia-smi ignores CUDA_VISIBLE_DEVICES and lists ALL physical
                # GPUs, so on a shared box we filter to only the GPUs this run
                # uses (the physical indices in CUDA_VISIBLE_DEVICES). Without
                # this, a co-tenant's memory would pollute the reported number.
                _vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
                _vis_set = {v.strip() for v in _vis.split(",") if v.strip() != ""} \
                    if _vis else None
                try:
                    q = subprocess.run(
                        ["nvidia-smi", "--query-gpu=index,memory.used",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=30).stdout.strip()
                    keep = []
                    for ln in q.splitlines():
                        parts = [p.strip() for p in ln.split(",")]
                        if len(parts) >= 2 and (_vis_set is None or parts[0] in _vis_set):
                            keep.append(",".join(parts))
                    smi_used = ";".join(keep)
                except Exception as e:
                    smi_used = f"query_failed:{e}"
                sf.write(f"# SMI_USED_MB_PER_GPU={smi_used}\n")
                sf.write("# ---- nvidia-smi ----\n")
                try:
                    smi = subprocess.run(["nvidia-smi"], capture_output=True,
                                         text=True, timeout=30).stdout
                except Exception as e:
                    smi = f"nvidia-smi failed: {e}\n"
                sf.write(smi)
            print(f"[MEM_PROBE] wrote snapshot {snap_path}", flush=True)
        torch.distributed.barrier()
        break


# ══════════════════════════════════════════════════════════════════════════════
# Cleanup
# ══════════════════════════════════════════════════════════════════════════════

if master_process:
    if log_file:
        log_file.close()
    print(f"\nTraining log saved to: {log_filename}")
    print("\n=== CP via _AttentionContextParallel — Training Complete ===")

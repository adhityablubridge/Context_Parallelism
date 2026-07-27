"""usp_model.py -- a minimal faithful transformer block stack driven by USP
(yunchang) hybrid Ulysses x Ring attention, for apples-to-apples comparison
against bluscriptCP's hybrid CP.

The block mirrors bluscriptCP's model shape: RMSNorm -> GQA attention (USP) ->
residual -> RMSNorm -> SwiGLU MLP -> residual, at the SAME dims (d_model,
n_layers, q_heads, kv_heads, head_dim, ffn). The sequence is sharded across the
CP mesh exactly as USP expects (EXTRACT_FUNC for the chosen ring layout), so
each rank holds T/cp_size tokens and only attention communicates across ranks --
the same division of labor as the fused CP path we compare against.

Shared by usp_probe.py (2-step memory snapshot) and train_usp_compare.py
(logged training run). NOT a training-quality model -- weights are random; this
measures the SYSTEMS behavior (throughput / peak memory / max context) of the
USP hybrid attention mechanism at a given (ring, ulysses, T).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from yunchang import LongContextAttention, EXTRACT_FUNC_DICT
from yunchang.kernels import AttnType


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).type_as(self.weight)


class USPBlock(nn.Module):
    """One transformer block using USP hybrid attention for the attn op."""

    def __init__(self, d, qh, kvh, hd, ffn, ring_impl_type):
        super().__init__()
        self.qh, self.kvh, self.hd = qh, kvh, hd
        self.attn_norm = RMSNorm(d)
        self.q_proj = nn.Linear(d, qh * hd, bias=False)
        self.k_proj = nn.Linear(d, kvh * hd, bias=False)
        self.v_proj = nn.Linear(d, kvh * hd, bias=False)
        self.o_proj = nn.Linear(qh * hd, d, bias=False)
        self.q_norm = RMSNorm(hd, eps=1e-6)  # Qwen3-style per-head qk norm
        self.k_norm = RMSNorm(hd, eps=1e-6)
        self.mlp_norm = RMSNorm(d)
        self.gate_proj = nn.Linear(d, ffn, bias=False)
        self.up_proj = nn.Linear(d, ffn, bias=False)
        self.down_proj = nn.Linear(ffn, d, bias=False)
        # One USP attention op per block (holds the CP process groups internally).
        self.attn = LongContextAttention(ring_impl_type=ring_impl_type, attn_type=AttnType.FA)

    def forward(self, x):  # x: [B, T_local, d]
        B, Tl, _ = x.shape
        h = self.attn_norm(x)
        q = self.q_norm(self.q_proj(h).view(B, Tl, self.qh, self.hd))
        k = self.k_norm(self.k_proj(h).view(B, Tl, self.kvh, self.hd))
        v = self.v_proj(h).view(B, Tl, self.kvh, self.hd)
        # USP consumes local shards [B, T_local, H, D] and gathers across the CP mesh.
        o = self.attn(q, k, v, dropout_p=0.0, causal=True)  # -> [B, T_local, qh, hd]
        x = x + self.o_proj(o.reshape(B, Tl, self.qh * self.hd))
        h = self.mlp_norm(x)
        x = x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        return x


class USPModel(nn.Module):
    def __init__(self, d, n_layers, qh, kvh, hd, ffn, ring_impl_type):
        super().__init__()
        self.blocks = nn.ModuleList(
            [USPBlock(d, qh, kvh, hd, ffn, ring_impl_type) for _ in range(n_layers)]
        )
        self.norm_f = RMSNorm(d)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.norm_f(x)


def shard_sequence(global_hidden, rank, world_size, ring_degree, ulysses_degree, ring_impl_type):
    """Shard a global [B, T, d] hidden state into this rank's local [B, T_local, d]
    using the SAME layout USP expects for the chosen ring impl (zigzag reorders;
    basic is a contiguous chunk). Trick: view d as a single head so we can reuse
    EXTRACT_FUNC_DICT, then squeeze it back."""
    x = global_hidden.unsqueeze(2)  # [B, T, 1, d]
    local = EXTRACT_FUNC_DICT[ring_impl_type](
        x, rank, world_size=world_size, rd=ring_degree, ud=ulysses_degree
    )
    return local.squeeze(2).contiguous()  # [B, T_local, d]


def count_params(model):
    return sum(p.numel() for p in model.parameters())

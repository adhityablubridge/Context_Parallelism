# Attention-Kernel Design Requirements for Context Parallelism (CP)

**Audience:** the team building the fused attention kernel used by `BluTrain/bluscript.cpp:203-207`
(`autograd::scaled_dot_product_attention(...)` → `gqa_fused_flash_attn_forward/backward`).
**Author context:** I am layering ring-based Context Parallelism on top of this kernel.
**Date:** 2026-06-29

This document states *exactly* what the single-GPU fused kernel must expose so that a ring-CP
layer can wrap it without re-implementing attention. It is written by tracing both:

- the **production fused kernel** (`Tensor-Implementations/src/Kernels/cuda/attention/arch/GQA_fused_fwd_sm103.cu`,
  `GQA_fused_bwd_sm103.cu`, reached via `AttentionOps.cpp:547` `scaled_dot_product_attention`), and
- my **existing CP machinery** (`CP/context_parallel/` — `ContextParallel.h`, `SDPAMerger.h`,
  `RingRotator.h`, `LoadBalancer.h`, and the CP-aware `AttentionForward_sm89.cu` / `AttentionBackward_sm89.cu`).

> **STATUS / SCOPE CORRECTION (read first).** The CP layer *as it exists today* is a **GPT-2 with
> *learned* positional embeddings (`wpe`)** wrapping a **de-fused sm89 SDPA kernel** that has **no RoPE
> and no QK-norm**. Verified: a grep for `rope|rotary` across `context_parallel/` and `gpt2_cp_test.cpp`
> returns nothing; the per-token `pos_local` array feeds `wpe.forward(pos_idx)` (ContextParallel.h:525),
> not a rotary cache. **Therefore the CP layer has never handled RoPE.** Supporting the fused RoPE kernel
> of `bluscript.cpp` is **net-new work on both the kernel and the CP layer** — it is *not* already solved.
> Earlier drafts of this doc claimed "my CP already de-fuses RoPE / ships pre-rotated K"; that was wrong
> and is corrected in §3.5 / §4.1 / §4.2 below. The *de-fused sm89 contract* is real and proven for the
> GPT-2/wpe path; the *RoPE-fused* requirements are a design, not a description of existing code.

---

## 0. TL;DR — the five things that matter most

1. **Two independent position origins, not one.** Q and K/V in a ring step come from *different*
   global positions. The kernel needs **`q_offset` AND `k_offset`** (the production kernel has only
   a single `pos_offset`). Both the causal mask *and* RoPE must use them.
2. **Causal mask must be global.** Mask on `k_offset + k_local  >  q_offset + q_local`, not on
   local tile indices. The production kernel's `pos_offset` is applied to RoPE but **NOT to the mask**
   — that has to change.
3. **`T_q` and `T_k` must be independent.** Ring load-balancing calls attention with Q-length ≠
   KV-length (e.g. `Q[T/2:]` against full KV, or full Q against `KV[:T/2]`). The production kernel
   assumes square `T_q == T_k`.
4. **LSE must be a first-class *output* of the forward op**, returned to the caller (not just stashed
   inside the backward node). The cross-block merge is impossible without it.
5. **RoPE-fused-into-attention is the biggest friction point — and fusion is the chosen path here.**
   Keeping RoPE fused IS viable for CP/load-balancing (§4.1, §4.2): the kernel must replace the single
   `pos_offset` with the **4-delta interface** (per-side head/tail deltas, computed locally, no cache
   transfer), and must accumulate `dq_gamma`/`dk_gamma` across ring steps in backward (§3.5b). De-fusing
   (rotate Q/K before attention, pure-SDPA kernel) is the simpler alternative but is **not** the path
   taken; it is documented in §4 "Recommended resolution" only as the rejected option.

Everything below expands these.

---

## 1. What the production kernel does today (traced)

`scaled_dot_product_attention(query, key, value, q_gamma, k_gamma, cos_sin_cache, Nq_heads, Nkv_heads,
is_causal, interleaved, eps, pos_offset, backend)` → `gqa_fused_flash_attn_forward(...)`:

| Aspect | Current behavior | File:line |
|---|---|---|
| Fusion | QK-norm (RMSNorm over `hd`) + RoPE + causal GQA, all in one kernel | `GQA_fused_fwd_sm103.cu:49-92,149-179` |
| Position origin | single scalar `pos_offset`; used **only** for RoPE | `…fwd:149-150,178-179` |
| RoPE Q position | `q_row0 + pos_offset` | `…fwd:149` |
| RoPE K position | `kc*Bc + pos_offset` (same offset as Q) | `…fwd:178` |
| Causal mask | `(kc*Bc + j) > q_idx`, `q_idx = q_row0 + lane` — local/absolute, **no offset** | `…fwd:217` |
| T_q vs T_k | assumed equal; tile bounds derived from one shared `S = T` | `…fwd:116,159-161` |
| Online softmax | YES — running max `sm[]`, running sum `sl[]`, per-tile rescale | `…fwd:221-235` |
| LSE | computed `LSE = m + log(l)`, shape `[B, Nq, T]` fp32, saved for backward | `…fwd:272-274` |
| LSE exposure | NOT returned by `scaled_dot_product_attention`; hidden in backward node | `AttentionBackward.h:144-201` |
| GQA | `hkv = hq / G`, group broadcast of K/V; bwd accumulates dK/dV over group | `…fwd:113`, `…bwd:172-177` |
| Precision | fp32 in/out, bf16 internal (WMMA) + fp32 accum | `AttentionOps.cpp:392-416` |
| Backward inputs | its *own* saved `O, LSE, q_rstd, k_rstd, q/k_gamma, cache` + `dO` | `AttentionBackward.h:144-201` |
| Backward outputs | dQ, dK, dV, (dq_gamma, dk_gamma) | `GQA_fused_bwd_sm103.cu` |

**Good news:** online softmax + LSE + absolute-index masking + a `pos_offset` parameter mean the kernel
is already ~70% of the way to CP-ready. The gaps are specific and listed next.

---

## 2. What the CP ring layer requires (traced from my implementation)

My CP-aware kernel (`AttentionForward_sm89.h:13-24`) already encodes the target contract:

```
mem_efficient_attn_forward_tc_sm89_strided(
    q, q_strideB/M/H,  k, k_strideB/M/H,  v, v_strideB/M/H,
    out, o_strideB/M/H,  lse, lse_strideB/H,
    B, nh, T_q, T_k, q_offset, k_offset, hd, is_causal, dropout_p, dropout_mask)
```

Mask logic (`AttentionForward_sm89.cu:317-333`):
```cpp
const int qi_global = q_offset + qi_local;
const int k_global  = k_offset + kj_block + col;
if (is_causal && k_global > qi_global) v = -INFINITY;   // GLOBAL causal
```

The merge (`SDPAMerger.h:108-121`, Ring-Attention / Liu et al. 2023):
```
new_out = accum_out - sigmoid(block_lse - accum_lse) * (accum_out - block_out)
new_lse = accum_lse - log(sigmoid(accum_lse - block_lse))
```
It consumes **per-block `out` [B,H,T_q,D]** and **per-block `lse` [B,H,T_q,1]** each ring step.

Ring loop (`ContextParallel.h:472-586`): per step `i` on rank `r` it picks one of three modes —
causal (diagonal, `i==0`), non-causal full block (past KV), or **skip** (future KV, "skip-future"),
and under load balance it sub-chunks Q or KV to half length. It passes
`q_off = r * T_local`, `k_off = source_rank * T_local` into the kernel.

So the CP layer is "kernel-light": it does all communication, sequencing, and merging *outside* the
kernel. The kernel just has to be a **correct, offset-aware, asymmetric, LSE-emitting block primitive.**

---

## 3. The concrete kernel changes required

### 3.1 Separate `T_q` and `T_k` (square assumption must go)
Ring load-balancing (the zig-zag / HeadTail scheme, `LoadBalancer.h:37-50`) calls attention with
Q-length ≠ KV-length:
- past KV steps: full `Q` (`T_q = T_local`) against `KV[:T_local/2]` (`T_k = T_local/2`)
- future KV steps: `Q[T_local/2:]` (`T_q = T_local/2`) against full `KV` (`T_k = T_local`)

The production kernel derives all tile bounds from one shared `S` (`…fwd:116,159-161`). It must take
`seqlen_q` and `seqlen_k` separately and bound the Q-tile loop by `T_q`, the KV-tile loop by `T_k`,
and the diagonal/causal cut by both. This is the single most invasive structural change.

### 3.2 Dual position offsets feeding the **causal mask**
Replace the single `pos_offset` with `q_offset` and `k_offset`, and apply them to the mask:
```cpp
const int q_global = q_offset + q_row0 + lane;   // was: q_idx = q_row0 + lane
const int k_global = k_offset + kc*Bc + j;        // was: kc*Bc + j
if (is_causal && k_global > q_global) score = -INFINITY;
```
With `q_offset == k_offset == 0` this is byte-identical to today's behavior, so the single-GPU path is
unaffected. (Verify the diagonal pre-filter / `max_kj`-style early-out also uses the global delta
`q_offset + q_row0 + Br - 1 - k_offset` so future KV tiles are skipped — see my
`AttentionForward_sm89.cu:193-210`.)

> **Scope caveat:** scalar `q_offset`/`k_offset` for the mask are sufficient **only for contiguous /
> round-robin shards**, where each rank's local sequence is one contiguous global range. They are
> **NOT** sufficient under HeadTail **load balancing**, where the local buffer is two *disjoint* global
> ranges `[head | tail]` — see §4.1. In my CP layer the load-balanced path actually passes
> `q_off = k_off = 0` and relies on sub-chunk *selection* (not a positional mask) to encode causality
> (`ContextParallel.h:562-568`). The offsets matter for the **non-LB** path; RoPE is the part that LB
> genuinely breaks, covered next.

### 3.3 `is_causal == false` must be a *correct full* attention
CP calls the kernel non-causal for off-diagonal past-KV blocks. Confirm the `is_causal=false` path does
true unmasked attention over all `T_k` with no residual diagonal assumption. (The production kernel has
the flag; just needs validation under `T_q != T_k`.)

### 3.4 LSE as a returned output
`scaled_dot_product_attention` currently returns only the output tensor and buries LSE in the backward
node. CP needs the forward op to **return `(out, lse)`** (or accept an output `lse` pointer). Shape:
production uses `[B, Nq, T]`; my merger wants `[B, H, T_q, 1]` (keepdim). Either is fine — just expose
it and document the layout/stride. The softmax scale used to form LSE must equal the scale used in the
merge (`1/sqrt(hd)`), consistently across every block.

### 3.5 Backward contract — TARGET (de-fused sm89) vs the FUSED production kernel
**This subsection describes two different things; do not conflate them.**

**(a) The target de-fused contract (already real in the sm89 CP path).** A CP-friendly backward is a
**pure function of supplied tensors**: it accepts `Q, K, V, O, dO, LSE, q_offset, k_offset, T_q, T_k,
is_causal` and recomputes `P` under the global mask. CP then relies on:
- **dQ**: atomic-add into a caller-zeroed buffer (accumulated across ring steps for a fixed Q shard).
- **dK, dV**: returned per-block; CP rotates/accumulates them around the ring
  (`ContextParallelBackward.h:476-557`).
- a scratch `D = rowsum(dO ∘ O)` `[BH, T_q]` buffer (caller-provided).
This is exactly `AttentionBackward_sm89.{h,cu}` (sig at `:23-38`; global-causal recompute + `q_loop_start =
max(0, k_offset + kv_base - q_offset)` at `:250-255,315-326`; D computed by the launcher at `:91-137`). It
has **no RoPE / QK-norm / gamma** — it is a pure SDPA backward.

**(b) The fused production backward does the OPPOSITE on every point** (`GQA_fused_bwd_sm103.cu`): it
recomputes RoPE + QK-norm **internally** via `norm_rope_tile` keyed on a single `q_row0 + pos_offset`
(`:177-178`), computes its **own** `D = rowsum(dO*O)` internally (`:181-187`, *not* caller-provided), and
runs off its **own** saved LSE, single `pos_offset`, and single square `S`. So making the fused kernel
CP-correct is a **large** change, not a wrapper: it must (i) thread per-token / per-side positions for the
internal RoPE recompute (the §4.2 deltas), (ii) take `q_offset/k_offset/T_q/T_k`, and (iii) **accumulate
`dq_gamma`/`dk_gamma` across all ring steps** (each ring step is a partial contribution to the same gamma),
which the single-GPU kernel does in one shot. The CP caller must own that gamma accumulation.

**LSE-for-backward caveat (correction).** It is **not** simply true that "the merged LSE is the correct
LSE for backward." For load-balanced **partial** steps, feeding the merged LSE corrupts that step's
gradient — which is why the CP code has a `USE_PER_STEP_LSE` path that saves per-step `out`/`lse`
(`Claude Logs.md:56`; `ContextParallelBackward.h` around `:291-297` and the forward note `:677-686`).
The repo itself is not internally consistent here (default path uses merged; the per-step path exists for
the partial-step bug). **Requirement:** the kernel must let the caller supply *whichever* LSE it chooses
per step (merged or per-step), and the CP layer decides — the kernel must not assume merged-LSE is valid.

### 3.6 Strided / view inputs (nice-to-have, not blocking)
Sharding and sub-chunking produce **non-contiguous views** (narrow on the sequence dim). The production
kernel repacks Q/K/V into one dense bf16 `[Q|K|V]` buffer (`AttentionOps.cpp:392-395`). CP can `.clone()`
to contiguous, but a kernel that accepts per-tensor `(strideB, strideM, strideH)` avoids a repack on
every ring step. Recommended, not required.

---

## 4. RoPE — the central design decision (read this twice)

RoPE fused *inside* the attention kernel, keyed on a single `pos_offset`, is **fundamentally at odds
with ring CP**, for three compounding reasons:

**(a) Q and K need different RoPE positions per ring step.** In step `i`, Q is rank `r`'s shard
(global positions starting at `q_offset`) and K is *source rank*'s shard (positions starting at
`k_offset ≠ q_offset`). The production kernel rotates both Q and K with the *same* `pos_offset`
(`…fwd:149,178`) — wrong for any step where the KV came from another rank. A single offset cannot
express this; you need `q_pos_offset` and `k_pos_offset`.

**(b) Re-rotating K every ring step is wasted work and corrupts gradients.** K travels the ring and is
visited by `world_size` ranks. If RoPE is applied inside attention, the same K shard gets re-rotated on
every rank that sees it. Numerically idempotent only if each applies K's *own* fixed global position
(not the local one) — which again requires shipping K's positions alongside it. Worse, because RoPE is
fused with QK-norm, the **`dk_gamma` / RoPE gradients become entangled with the per-block recompute**,
so the gamma gradient has to be correctly accumulated across all ring steps instead of computed once.

**(c) Load-balanced shards are NON-CONTIGUOUS in global position.** Under HeadTail balancing
(`LoadBalancer.h:37-50`, `ContextParallel.cpp:83-129`) rank `r` owns chunks `(r, 2N-1-r)` — two disjoint
global ranges. A token's RoPE position within a local shard is therefore *not* `pos_offset + local_idx`;
it jumps. A single scalar offset cannot describe it; you'd need a **per-token position index array**
indexing `cos_sin_cache`. (Analogous precedent: in the GPT-2/wpe PyTorch reference, the *learned*
positional-embedding lookup had a `pos=arange(0,T_local)` bug under LB that broke parity until the global
positions were sharded through the CP machinery — `Claude Logs.md:75`. That was learned `wpe`, **not** a
RoPE cache; the principle "shard position-dependent buffers in lockstep with the tokens" transfers, but no
RoPE code in this repo does it yet.)

### 4.1 The load-balancing + sub-chunk (zig-zag round-robin) case — where fused RoPE truly breaks
This is the case that makes de-fusing non-negotiable, beyond the static non-contiguity of §4(c).

HeadTail balancing (`LoadBalancer.h:37-50`) gives rank `r` chunks `(r, 2N-1-r)`, laid out `[head | tail]`
where head global positions ≈ `r·cs` and tail ≈ `(2N-1-r)·cs` (`cs = T/2N`). So a rank's *local* Q/K is
**two disjoint global position ranges**, and during the ring the loader slices half-views of it
(`ContextParallel.h:508-529`):
- `i==0`: full `[head|tail]` Q vs full `[head|tail]` K/V (causal)
- `0 < i ≤ rank`: full Q vs `K/V[:T/2]` = the **head** half (non-causal)
- `i  > rank`: `Q[T/2:]` = the **tail** half vs full K/V (non-causal, partial)

via `make_shards_inplace_axis(2, seq_dim)`. Two failures result if RoPE is fused on a scalar offset:

**(1) Scalar offset can't index two ranges.** A fused kernel computing `pos = base + local_idx` rotates
the tail half with head-range angles. No single `pos_offset` (or even `q_offset`+`k_offset`) is correct,
because head and tail with *different* position values coexist inside one call (the `i==0` diagonal).

**(2) Sub-chunk slicing destroys the `base + arange` assumption.** When the ring hands the kernel
`q_halves[1]` (tail) or `k_halves[0]` (head), a fused kernel sees a `T/2` slice and would rotate it as
`base + arange(T/2)` — but `q_halves[1]` is the *tail* chunk, whose true positions are the late range,
not `base..base+T/2`. The angle is simply wrong, and selection can't fix it.

**Why the causal *mask* survives but RoPE doesn't.** HeadTail preserves global ordering inside the local
buffer (head always precedes tail, in both layout and global position), so a plain *local* causal mask is
globally correct on the diagonal, and off-diagonal causality is encoded by *which half is selected* — the
sub-chunk selection **is** the mask. That's why the LB path passes `q_off = k_off = 0`
(`ContextParallel.h:562-568`). Masking only needs *ordering*, which selection provides. RoPE needs the
actual position *value* to look up `cos_sin_cache`, which selection cannot fabricate. So the zig-zag gives
the mask a free ride and gives RoPE nothing.

**What the CP layer actually does today (correction).** It builds a per-token position list `pos_local`
`[1, T_local]` int64 (`ContextParallel.h:75,94-137` — "first half = early global positions, second half =
late"), but that array feeds the **learned `wpe` positional embedding** (`wpe.forward(pos_idx)`,
ContextParallel.h:525) — **there is no RoPE in the CP layer at all.** The attention kernel it wraps
(`AttentionForward_sm89`) has **no RoPE inside** and only takes scalar `q_off/k_off` for the mask (and
even those are 0 under LB). So the existing code sidesteps the RoPE problem only because the model is
GPT-2/wpe, not because RoPE was solved. For the Llama/RoPE model the rotation must be made
position-correct either by de-fusing (rotate Q/K before attention with true global positions) **or** by
the fused 4-delta scheme in §4.2 — neither exists in the repo yet.

The two failures above are failures of a **single scalar offset + `base + arange` assumption** — not of
fusion itself. Fused RoPE *is* viable for load balancing: see §4.2. The fix is to pass **two offsets per
side** (head_base, tail_base), both computed locally from the deterministic source-rank index, so the
kernel indexes the full resident cache at the correct piecewise global position. No cache transfer and no
per-token array are needed for the standard 2-chunk zigzag. (My own CP layer instead de-fuses — it
pre-rotates with `pos_local` before attention — which is one valid choice, but not the only one.)

### 4.2 How production frameworks handle this (PyTorch / Megatron-LM / TransformerEngine)
Researched against actual source (June 2026). The headline finding:

> **No major framework fuses RoPE *into* the attention kernel for CP training.** PyTorch (torchtitan),
> Megatron-LM and TransformerEngine all apply RoPE as a **separate op before** the attention kernel, and
> make load balancing correct by **sharding/reordering the rotary cos/sin cache with the exact same
> zigzag permutation as the tokens.** By the time Q/K reach the (RoPE-agnostic) flash kernel, they are
> already rotated with their true global positions.

- **Megatron-LM** — `get_pos_emb_on_this_cp_rank(pos_emb, seq_dim, cp_group)` (in
  `megatron/core/models/common/embeddings/rope_utils.py`) reshapes the rope cache to expose `2*cp_size`
  chunks and does `index_select(seq_dim, [cp_rank, 2*cp_size-cp_rank-1])` — **the identical zigzag index
  used to shard the tokens**. The sliced cache is then handed to `apply_rotary_pos_emb` on Q/K *before*
  `TEDotProductAttention`/flash. (Issue #560 is users hitting exactly this "is the rope split correct"
  question; the function is the answer.)
- **PyTorch torchtitan / `torch.distributed.context_parallel`** — `freqs_cis` is registered as a
  context-parallel buffer and **sharded over the sequence dim in lockstep with the input**
  (`_context_parallel_buffers`); `apply_rotary_emb` runs before SDPA. Their **Round-Robin load balancer**
  reorders query blocks (shortest+longest → rank 0, etc.) and `freqs_cis` follows the same reorder.
  (My own GPT-2 PyTorch reference applies the *same buffer-sharding principle* — `Claude Logs.md:75` —
  but to the **learned `wpe` position buffer**, not a RoPE `freqs_cis`. Same idea, different buffer; the
  RoPE version is not yet implemented here.)
- **The one fused-RoPE-in-kernel that exists** — flash-attn's `flash_attn_with_kvcache` — parameterizes
  rotary by a **scalar/per-batch `cache_seqlens` offset** (contiguous decode positions). It cannot
  express the zigzag's two disjoint ranges, so it is **not** used for load-balanced CP training. The
  regular training entry points (`flash_attn_func` / `flash_attn_varlen_func`) apply **no** RoPE inside
  the kernel at all — RoPE must be applied beforehand.

**What this means if we keep RoPE fused (the team's constraint).** Fusion does not let us avoid the
position bookkeeping — it **moves it inside the kernel**. To match what Megatron does outside, the fused
kernel must index `cos_sin_cache` by the **actual reordered global position of each token**, not by a
contiguous scalar `pos_offset`. Concretely, replace the scalar with:
- a **per-token position-id (or pre-gathered cos/sin) array for Q**, already zigzag-reordered to the
  rank's `[head | tail]` layout; and
- a **per-token position-id array for K that travels the ring with K** and is sliced in lockstep when
  sub-chunking takes `K[:T/2]` / `K[T/2:]`.

Then per Q row look up `cos/sin` at `q_pos[row]`, per K col at `k_pos[col]`. The **causal mask** can keep
using the zigzag-selection trick (`q_off=k_off=0`, `is_causal` only on the diagonal) because ordering is
preserved; only the **rotation** needs the gathered positions. This is functionally identical to
de-fusing — the production frameworks simply chose to express the cache-sharding at the Python/buffer
level (`get_pos_emb_on_this_cp_rank` / `_context_parallel_buffers`) because it is cleaner there.

**Preferred concrete implementation — full cache resident on every rank + per-side offsets. NO transfer.**
The full cos/sin cache `[T_full, hd]` is tiny, so every rank holds the whole thing and indexes it by
**absolute global position**. The fused kernel never needs the cache shipped — it only needs to know which
global positions the resident Q/K shards occupy, which is pure offset metadata computable locally. The only
nuance vs. a single naive offset is **how many offsets** the (possibly two-piece) shard layout needs:

- **Contiguous sharding:** source rank `s`'s K shard is one contiguous range → **one scalar
  `k_offset = s*T_local`**. Local K index `j` → global `s*T_local + j` → `cache[that]`. Q likewise with
  `q_offset = r*T_local`. This is the straightforward case.
- **HeadTail / zigzag load balancing:** each shard is **two disjoint chunks** (`head=s`, `tail=2N-1-s`)
  laid out `[head_half | tail_half]`, so the local-index→global-position map is *piecewise* and needs
  **two offsets per side**:
  ```
  j <  T_local/2 :  global = head_base(s) + j
  j >= T_local/2 :  global = tail_base(s) + (j - T_local/2)
  head_base(s) = s*cs ,  tail_base(s) = (2N-1-s)*cs ,  cs = T/(2N)
  ```
  **`s` is known locally** at ring step `i` on rank `r`: `s = (r - i) mod N`, and the bases are closed-form.
  So each rank computes its own K-offsets **per ring step with zero communication and zero cache transfer**;
  Q-offsets (`head_base(r)`, `tail_base(r)`) are static. Sub-chunk slicing just selects which half/base
  applies. (This is the in-kernel equivalent of Megatron's `get_pos_emb_on_this_cp_rank`, which rearranges
  the cache rows at the Python level instead — same positions, different layer.)

**Kernel signature delta to keep fusion CP-correct — the unified 4-delta interface (recommended).**
Interpret the offsets as **additive deltas to the local index**, with the kernel splitting the passed
sequence at its midpoint. This gives ONE code path that degenerates cleanly to contiguous-CP and non-CP:
```cpp
// instead of:  int pos_offset
int q_delta0, q_delta1;   // delta for Q's first / second half (split at len_q/2)
int k_delta0, k_delta1;   // delta for K's first / second half (split at len_k/2)
// kernel: global_pos(j) = j + ( j >= len/2 ? delta1 : delta0 );  then index the resident full cos/sin cache.
```
Degenerate values (all computed locally, NO communication, NO cache transfer):

| Regime | q_delta0,q_delta1 | k_delta0,k_delta1 |
|---|---|---|
| **Non-CP (single GPU)** | `0, 0` | `0, 0`  → `global = j`, recovers current kernel exactly |
| **Contiguous CP** | `r*T_local, r*T_local` | `s*T_local, s*T_local`  (two equal per side) |
| **HeadTail LB** | `head_base(r), tail_base(r)-T_local/2` | `head_base(s), tail_base(s)-T_local/2` |

with `head_base(x)=x*cs`, `tail_base(x)=(2N-1-x)*cs`, `cs=T/(2N)`, `s=(r-i) mod N` (locally known each step).

Composes with **sub-chunking** for free: when the ring passes a single half-chunk (`K[:T/2]` or `Q[T/2:]`),
that side's two deltas are set **equal** (to `head_base(s)` or `tail_base(r)`) so the split is a no-op; when
a full two-chunk shard is passed, the two deltas differ. **Bonus:** because the kernel now has true global
positions per token, it can *optionally* use one uniform global causal mask `k_global <= q_global` for all
regimes instead of the zigzag-selection trick — or keep the existing `use_causal`/skip logic and use the
deltas for RoPE only.

Cost note: with RoPE fused, each rank re-rotates the K shard resident at its step (minor redundant compute
vs. rotating once up front) — but **no extra communication** and no cache movement. (An earlier draft said
the K-cache must "travel the ring"; that was wrong — the full cache is resident everywhere and shard
positions are a deterministic function of the locally-known source rank, so only these 4 deltas are needed.)

**Why 4 slots and not 2 (the {Q,K}×{head,tail} axes are independent).** It is tempting to carry only
`(q_offset, k_offset)`. That fails on the `i=0` **diagonal** under HeadTail: the local shard is two
*non-adjacent* global chunks `[head | tail]`, and the diagonal block contains **tail-query attending
head-key** (e.g. N=2, rank 0 shard = global `{0,1,6,7}`: query@6 attends key@0). RoPE is relative, so the
score depends on the *true* gap `6-0=6`; a single offset places the tail contiguously after the head
(gap≈2) and **silently corrupts every cross-chunk term**. So head and tail need distinct offsets even at
`i=0` (there Q and K share the shard, so 2 *unique* values, applied per-half). At `i>0`, K comes from
source rank `s≠r`, so the *full* side (always a 2-chunk shard) needs its own head/tail pair while the
sub-chunked side is a single chunk — and the full side alternates between Q (`i≤rank`) and K (`i>rank`).
Hence both sides must expose a head/tail pair → **4 slots** (collapsing to ≤3 distinct per step).

**The only way 2 offsets suffice:** decompose every computation into single-chunk × single-chunk blocks
(chunk-level ring instead of shard-level). Then each call is a contiguous rectangle/triangle on both sides
and `(q_offset, k_offset)` works — but `i=0` becomes **three** sub-calls (head×head causal, tail×tail
causal, tail×head full rectangle), i.e. more, smaller launches. Design fork: **4-slot kernel** (one call
per shard-pair, kernel splits head/tail internally) vs **2-slot kernel + caller-side chunk decomposition**
(simpler kernel, more launches). Both correct.

### Recommended resolution (if de-fusing were allowed): expose a **de-fused** building-block path
Keep the fully-fused kernel for the non-CP single-GPU fast path (it's great there). But for CP, provide
either:

- **Option A (preferred): de-fuse.** Expose standalone `qk_norm` and `rope_apply(x, positions)` ops that
  take an explicit **per-token position tensor** (so HeadTail's disjoint ranges work), and an attention
  kernel that does **pure SDPA** (no RoPE, no QK-norm) but *does* take `q_offset/k_offset/T_q/T_k/LSE`.
  CP then: at shard time, applies QK-norm + RoPE to Q and K using each token's true global position;
  ships **already-normalized, already-rotated** K around the ring; the attention kernel just attends.
  This is exactly what my `AttentionForward_sm89` does (it has **no** RoPE inside) and it is the clean,
  proven path.
- **Option B (if fusion must stay): generalize the fused kernel** to take `q_pos_offset`, `k_pos_offset`
  *and* optional per-token position index arrays for Q and K, and make QK-norm idempotent-safe under
  repeated K visits, with gamma-grad accumulation handled by the caller. Strictly harder; only worth it
  if the fused kernel's perf advantage is large and measured.

My recommendation: **ship Option A's de-fused primitives**; keep the fused kernel as the single-GPU
default. RoPE/QK-norm applied once at shard time is cheaper than re-applying inside every ring step
anyway.

---

## 5. GQA + ring communication volume (keep GQA in the kernel)
The fused kernel's GQA (`hkv = hq/G`) is *beneficial* for CP, not just single-GPU: the ring ships **K/V**,
so shipping `kv_heads` (4) instead of `q_heads` (12) is **3× less inter-rank traffic**. So:
- Keep the attention block primitive **GQA-aware** (accept `Nq_heads`, `Nkv_heads`), and
- keep K/V **grouped** through the ring rotation (do not expand KV heads before communication).

Note my current CP `sm89` kernel takes a single `nh` and is *not* GQA-aware — that means today CP would
have to expand KV heads (more comm + more memory). A GQA-aware CP attention primitive is the better
end state; flag this as a joint design item.

---

## 6. Numerical / precision notes
- **Two different internal precisions are in play — don't conflate them.** The **production** fused kernel
  is **bf16-WMMA internal, fp32 accum**, with LSE = `m + log(l)` in fp32 (`…fwd:272-274`) and output O
  stored as **bf16** (`…fwd:270`). The **sm89 CP kernel the merge currently wraps is TF32-internal**
  (`sm89cp_mma_tf32`, with the `+0x1000u` round-to-nearest at `AttentionForward_sm89.cu:232/241/250`).
  So statements about "bf16-internal" describe the production kernel, while the CP path is TF32. Any
  unified CP+fused plan must pick and document one.
- **The merge consumes a rounded per-block O.** SDPAMerger combines per-block outputs in fp32, but each
  block's O arrives **bf16-rounded** (production) or TF32-rounded (sm89) before the fp32 merge. The merge
  being fp32 does not recover the per-block rounding — a parity-risk worth a test, given the history below.
- The per-block LSE must be a true fp32 logsumexp (production's is); keep the softmax scale identical
  across all blocks and inside the merger.
- History (`Claude Logs.md:64-69`) shows TF32-vs-fp32 placement caused C++↔PyTorch parity drift in CP.
  For CP correctness work, expose a way to force the attention math to fp32 (an `ATTN_FP32`-style toggle)
  so parity tests can isolate algorithm bugs from precision noise.

---

## 7. Proposed CP-ready kernel API (satisfies both single-GPU and CP)

**Forward** (returns out + LSE; offsets default 0 ⇒ identical to today):
```cpp
struct SDPAResult { Tensor out; Tensor lse; };   // lse: [B, Nq, T_q] (or keepdim [.,.,.,1])
SDPAResult sdpa_block_forward(
    Tensor q, Tensor k, Tensor v,                 // [B, Nq|Nkv, T_q|T_k, hd], may be views (strided)
    int Nq_heads, int Nkv_heads,
    int64_t T_q, int64_t T_k,
    int q_offset, int k_offset,                   // GLOBAL positions; feed mask
    bool is_causal, float scale);
// RoPE + QK-norm applied OUTSIDE (de-fused, per-token positions) — see §4 Option A.
```

**Backward** (pure function of supplied tensors):
```cpp
std::tuple<Tensor,Tensor,Tensor> sdpa_block_backward(   // dQ, dK, dV
    Tensor q, Tensor k, Tensor v, Tensor out, Tensor dout, Tensor lse,
    int Nq_heads, int Nkv_heads,
    int64_t T_q, int64_t T_k,
    int q_offset, int k_offset,
    bool is_causal, float scale,
    Tensor D_scratch /* [B*Nq, T_q] */);
```

If the team keeps the fused kernel, the same parameters apply but with added `q_pos_offset`,
`k_pos_offset`, and optional `q_positions[]`, `k_positions[]` per §4 Option B, plus
`q_gamma/k_gamma/cos_sin_cache` and `dq_gamma/dk_gamma` outputs whose accumulation the CP caller owns.

---

## 8. Checklist for the kernel team
- [ ] Add independent `T_q` / `T_k`; bound Q-loop by `T_q`, KV-loop by `T_k`, diagonal by both.
- [ ] Add `q_offset` + `k_offset`; apply to the **causal mask** as `k_offset+k > q_offset+q` (not just RoPE).
- [ ] Verify `is_causal=false` is correct full attention under `T_q != T_k`.
- [ ] **Return LSE** from the forward op (document shape + stride); keep it true fp32 logsumexp.
- [ ] Backward: accept supplied `(Q,K,V,O,dO,LSE,offsets,T_q,T_k,is_causal)`; dQ atomic-add, dK/dV per-block. NOTE the fused kernel today computes `D` and recomputes RoPE/QK-norm internally (§3.5b) — to be CP-correct it must take the per-side position deltas and **accumulate `dq_gamma`/`dk_gamma` across ring steps**.
- [ ] Let the caller choose the LSE fed to backward per step (merged vs per-step) — do **not** assume merged-LSE is valid; it corrupts load-balanced partial steps (§3.5).
- [ ] RoPE strategy = **fused, via the 4-delta interface** (§4.2). (De-fusing is the rejected alternative.)
- [ ] Keep the primitive **GQA-aware**; keep K/V grouped through the ring (3× less comm).
- [ ] Pin **internal precision**: production is bf16-WMMA, the sm89 CP path is TF32 — pick one and document it; flag per-block-O rounding as a parity item (§6).
- [ ] (Nice-to-have) accept **strided** Q/K/V/O to avoid per-ring-step repacks.
- [ ] Provide an **fp32 math toggle** for CP↔reference parity testing.

---

## 9. Source map (for whoever picks this up)
- Production fused fwd/bwd: `Tensor-Implementations/src/Kernels/cuda/attention/arch/GQA_fused_{fwd,bwd}_sm103.cu`
- Production op + dispatch: `Tensor-Implementations/src/autograd/operations/AttentionOps.cpp:328-578`, `RopeOps.cpp:20-47`
- Production backward node: `Tensor-Implementations/include/autograd/backward/AttentionBackward.h:144-201`
- CP-aware kernel (de-fused, offset+asym, target contract): `CP/context_parallel/AttentionForward_sm89.{h,cu}`, `AttentionBackward_sm89.{h,cu}`
- CP orchestration: `CP/context_parallel/ContextParallel.h:425-720`, `ContextParallelBackward.h:240-568`
- Cross-block merge: `CP/context_parallel/SDPAMerger.h:62-172`
- Ring comm: `CP/context_parallel/RingRotator.h`; load balance: `LoadBalancer.h`, `ContextParallel.cpp:83-129`

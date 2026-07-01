# Implementation Plan — Offset-Based RoPE for the Fused CP Attention Kernel

**Goal:** make the fused (QK-norm + RoPE + GQA + causal) attention kernel context-parallel-correct by
replacing its single scalar `pos_offset` with the **4-delta** position interface, and wire the CP layer
to drive it (forward + backward, contiguous + HeadTail load balancing).
**Companion doc:** `cp_attention_kernel_requirements.md` (the contract + rationale).
**Date:** 2026-06-29

> Reminder of ground truth (verified): the CP layer today is **GPT-2 / learned `wpe` + de-fused sm89 SDPA,
> NO RoPE**. This plan is net-new work spanning the kernel (team) and the CP layer (me).

---

## A. Strategy: wait vs. parallel

**Parallelize, contract-first.** Only the *interface* is a true dependency on the team; the CP-side logic,
the delta math, and the parity harness are all independent of their kernel internals.

- **Block on:** agreeing the kernel signature (Phase 0). Cheap, one sync.
- **Do NOT wait for:** their kernel body. Build the CP plumbing + a de-fused stand-in oracle meanwhile.
- **De-risk early:** the stand-in (RoPE applied in the CP layer + existing pure-SDPA kernel) validates the
  whole CP-under-RoPE ring before the fused kernel exists, and becomes the oracle to test it against.

---

## B. The math being implemented (single source of truth)

Per-token global position from a per-side delta pair, split at the midpoint of the *passed* sequence:
```
global_pos(j) = j + ( j >= len/2 ? delta1 : delta0 )     // len = length of the Q (or K) tensor passed
```
Delta values (`cs = T/(2N)`, `head_base(x)=x*cs`, `tail_base(x)=(2N-1-x)*cs`):

| Regime | (q_delta0, q_delta1) | (k_delta0, k_delta1) |
|---|---|---|
| non-CP | `0, 0` | `0, 0` |
| contiguous CP (rank r, source s) | `r*T_local, r*T_local` | `s*T_local, s*T_local` |
| HeadTail LB (rank r, source s=(r-i)%N) | `head_base(r), tail_base(r)-T_local/2` | `head_base(s), tail_base(s)-T_local/2` |

Sub-chunk rule: when only a half-chunk is passed (`K[:T/2]`=head(s), `Q[T/2:]`=tail(r)), set that side's
two deltas **equal** (`head_base(s)` or `tail_base(r)`) so the split is a no-op. The kernel uses
`global_pos` to (a) index `cos_sin_cache` for RoPE on Q and K, and optionally (b) drive a global causal
mask `k_global <= q_global`. QK-norm (RMSNorm over hd) is position-independent — unaffected.

Backward: the fused backward recomputes RoPE with the *same* `global_pos`, so the deltas must be threaded
identically. `dq_gamma`/`dk_gamma` are **partial** per ring step → CP must accumulate them (dq_gamma like
dQ: per-Q-shard; dk_gamma like dK: travels the ring with the K shard).

---

## C. Phases

### Phase 0 — Lock the interface contract (BLOCKING, ~1 sync)
Deliverable: a frozen header (both sides build to it), even if stubbed.
- Forward: `(q,k,v, Nq,Nkv, T_q,T_k, q_delta0,q_delta1,k_delta0,k_delta1, is_causal, scale, eps,
  q_gamma,k_gamma, cos_sin_cache) -> (out, lse)`. LSE returned, shape/stride documented.
- Backward: `(q,k,v,out,dO,lse, deltas..., T_q,T_k, is_causal, scale, gammas, cache)
  -> (dQ,dK,dV, dq_gamma,dk_gamma)`; dQ atomic-add into caller-zeroed buffer; dK/dV per-block;
  dq_gamma/dk_gamma per-call (caller accumulates). Caller supplies the LSE per step (merged or per-step).
- Decisions to pin: LSE shape (`[B,Nq,T_q]` vs keepdim `[.,.,.,1]`); internal precision (bf16 vs TF32) +
  an fp32 parity toggle; whether mask is global-from-deltas or selection-trick (recommend: kernel supports
  global mask, CP can still pass is_causal/skip).
- Exit criteria: header compiles on both sides; a no-op stub returns correct shapes.

### Phase 1 — CP-layer delta derivation + plumbing (parallel; mine)
- 1a. `RopeDeltas` helper: `(N, r, i, T_local, scheme, sub_chunk_kind) -> (q_d0,q_d1,k_d0,k_d1)`.
  Pure function, no device code. **TDD: write the table in §B as unit tests first.**
- 1b. Replace the `q_off/k_off` scalars at the SDPA call site (`ContextParallel.h:562-586`) with the
  4 deltas; handle the three sub-chunk cases (`i==0`, `i<=rank`, `i>rank`).
- 1c. Backward: extend `ContextParallelBackward.h` to accumulate `dq_gamma` (per-Q-shard, alongside dQ)
  and `dk_gamma` (ring-rotated, alongside dK at `:476-557`). Add the gamma buffers to the saved state.
- 1d. Decide & implement the per-step-vs-merged LSE feed (honor the `USE_PER_STEP_LSE` lesson — partial
  steps need per-step LSE; see requirements §3.5).
- Exit criteria: delta unit tests green; plumbing compiles against the Phase-0 stub.

### Phase 2 — De-fused stand-in ORACLE (parallel; mine — highest de-risk value)
- 2a. In the CP layer, apply RoPE to Q and K with their true `global_pos` (built from the same deltas /
  per-token positions), then call the **existing pure-SDPA sm89 kernel** (no RoPE). Throwaway path.
- 2b. PyTorch/single-GPU reference: ws=1 full-RoPE Llama attention is the ground truth.
- 2c. Parity gates (mirror the proven ws=1-vs-ws=2 methodology in the logs):
  - forward cosine ws=1 vs ws=2/4 ~ 1.0, contiguous AND HeadTail LB;
  - backward dQ/dK/dV/d_gamma parity;
  - both is_causal diagonal and off-diagonal/skip steps.
- Exit criteria: CP ring is byte/cosine-correct under RoPE **without** the fused kernel. This proves the
  orchestration + delta math + gamma accumulation independently, and is the oracle for Phase 3.

### Phase 3 — Integrate the team's fused kernel (when it lands)
- 3a. Wire it into `FusedSDPAOp` for the Llama/RoPE model (replace the stand-in's RoPE+SDPA with one call).
- 3b. Validate against the Phase-2 stand-in AND ws=1: forward + backward + gamma grads, fp32 toggle on.
- 3c. Bisection harness ready (per-block out/LSE dumps) reusing the existing CP debug toggles
  (`CP_DEBUG_SHAPES`, `CP_DRAIN_AT`, `ATTN_FP32`).
- Exit criteria: fused-kernel CP matches stand-in and ws=1 within precision tolerance.

### Phase 4 — GQA + performance (after correctness)
- GQA-aware ring: ship `kv_heads` not expanded heads (3x less comm); kernel takes `Nq/Nkv`.
- Re-enable shared forward rotator / overlap; confirm no regression vs Phase-3 parity.

---

## D. Risks & mitigations
- **Interface churn** → Phase 0 freeze; both sides to the stub.
- **Merged-vs-per-step LSE backward** (corrupts LB partial steps) → Phase 1d decision + per-step path.
- **Gamma-grad accumulation** (fused-only complexity) → explicit accumulators mirroring dQ/dK; tested in Ph2 stand-in.
- **Precision parity** (bf16/TF32 vs fp32) → fp32 toggle + per-block-O rounding flagged as a test, not an afterthought.
- **HeadTail >2-chunk schemes** → 4-delta is exact only for the 2-chunk zigzag; if the scheme changes, fall
  back to per-token position arrays (kept as the general escape hatch).

---

## E. Dependency summary (what blocks what)
- Phase 0 blocks 1b/1c integration *signatures* (not the delta math 1a or the stand-in 2a/2b).
- Phase 2 needs only the *existing* pure-SDPA kernel → start immediately.
- Phase 3 is the only phase that truly needs the team's deliverable.
- Critical path if working solo-parallel: 0 → (1a, 2b in parallel) → 2a/2c → 3.

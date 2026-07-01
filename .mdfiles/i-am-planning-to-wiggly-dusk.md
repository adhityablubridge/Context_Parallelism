# Plan: Additive Ulysses (DeepSpeed-style) Sequence-Parallel Attention in the CP framework

## Context

The CP framework (`/home/blu-bridge25/CP/context_parallel/`) today implements **ring**
context-parallel attention: the sequence is sharded across ranks, KV travels a ring
(`RingRotator`), each step runs an offset/asymmetric block SDPA, and partial outputs are
merged via online-softmax LSE (`SDPAMerger`). HeadTail zig-zag load-balancing keeps the ring
work even.

DeepSpeed **Ulysses** is a *different* sequence-parallel scheme and we want it as a
**second, opt-in attention mode** alongside ring — strictly additive, non-destructive.
Instead of rotating KV, Ulysses does a single layout swap:

1. **all-to-all** turns the sequence-sharded layout `[B, H, T_local, D]` (all heads, my
   sequence slice) into a head-sharded layout `[B, H_local, T, D]` (my head group, the full
   sequence);
2. each rank runs **one ordinary full-sequence causal SDPA** over `H_local` heads (no ring,
   no LSE merge, no load-balancing);
3. a second **all-to-all** swaps back to `[B, H, T_local, D]`.

Why add it: Ulysses has lower communication volume and a much simpler attention kernel
contract (square `T_q==T_k`, `q_offset==k_offset==0`, plain causal — none of the ring's
4-delta/asymmetry/merge machinery), and positions become trivial (after the gather every rank
holds the full contiguous sequence `0..T-1`, so the existing GPT-2/`wpe` path works unchanged
with no zig-zag position bookkeeping). It is the better choice when `H % world_size == 0` and
head count is large enough to shard.

**Confirmed scope (v1):** blocking (synchronous) all-to-all first; **MHA only**
(`H_q == H_kv`, GQA/MQA noted as future); building block **+ standalone parity test** (no
`gpt2_cp_test` model wiring); local SDPA **reuses the existing CP attention kernels** — **NO
new attention kernel is written**.

**Attention kernel reuse (explicit, per user direction).** The gathered full-sequence local
attention calls the SAME wrappers the ring path uses, `sdpa_fused_forward` /
`sdpa_fused_backward` in [FusedSDPAOp.h](context_parallel/FusedSDPAOp.h), which wrap
`OwnTensor::cp::cuda::mem_efficient_attn_forward_tc_strided` /
`mem_efficient_attn_backward_strided` defined in the existing
[AttentionForward.cu](context_parallel/AttentionForward.cu),
[AttentionBackward.cu](context_parallel/AttentionBackward.cu) (and the sm89 variants
[AttentionForward_sm89.cu](context_parallel/AttentionForward_sm89.cu),
[AttentionBackward_sm89.cu](context_parallel/AttentionBackward_sm89.cu)). Ulysses' local call is
the square causal regime `T_q == T_k == T`, `q_offset == k_offset == 0`, `is_causal == true` —
which [AttentionForward.cu:990-1003](context_parallel/AttentionForward.cu#L990-L1003)
transparently routes to Tensor-Implementations' own `fused_attn_forward_tc_sm89_cuda` on Ada,
and otherwise to the CP-port WMMA kernel. These kernels already emit LSE; Ulysses is a single
block (no ring), so **no `SDPAMerger` / LSE-merge step** is needed — `out`/`lse` from the one
call are used directly, and `lse` is saved for the one backward call.

---

## Design (mirrors the proven `enable_rope()` additive pattern)

All ring code paths stay byte-identical; Ulysses is reached only when explicitly enabled.

### New files (additive)

- **`context_parallel/UlyssesAttention.h`** — header-only. Free functions:
  - the two all-to-all layout transforms (combine / partition) as **raw collective + reshape**
    helpers (returning the transformed tensor + the count/displacement metadata), and
  - `ulysses_attention_forward(pg, q_local, k_local, v_local, scale, is_causal, ...)` that runs
    combine → `sdpa_fused_forward` → partition, builds the backward node, wires edges, returns
    the local output. This keeps the edits inside `ContextParallel.h` tiny.
- **`context_parallel/UlyssesAttentionBackward.h`** — `class UlyssesAttentionBackward : public Node`
  mirroring `ContextParallelBackward` (manual collective orchestration inside `apply()`, the
  established style in this repo — *not* composed autograd collective ops).

### All-to-all layout math (MHA, equal-split `alltoall`)

`P = world_size`, `T_local = T/P`, `H_local = H/P`. Reuse the `AlltoAllRingRotator`
count/displacement convention (`RingRotator.h:205-281`) but **uniform** (every peer gets an
equal block), so the simple `alltoall(sendbuff, recvbuff, count, dtype, sync=true)` on
`cp_comm_` suffices.

**Combine (seq-sharded → head-sharded)** input `[B, H, T_local, D]` → output `[B, H_local, T, D]`:
1. view `[B, P, H_local, T_local, D]` (split `H = P·H_local`; group `g` is destined for rank `g`),
2. permute → `[P, B, H_local, T_local, D]`, `.contiguous()` (block for rank `g` now contiguous),
3. `alltoall`, per-peer count `= B·H_local·T_local·D`; recv `[P=s, B, H_local, T_local, D]` (`s`=source rank),
4. permute → `[B, H_local, P=s, T_local, D]`, `.contiguous()`, view `[B, H_local, T, D]`
   (merging `(s, T_local)` in rank order reconstructs the **contiguous** global sequence).

**Partition (head-sharded → seq-sharded)** is the exact structural inverse (swap the two
permutes around the same `alltoall`). Because the collective is symmetric, the **backward of
combine is partition and vice-versa** — store the shape metadata and reverse the permutes.

### Forward integration in `ContextParallel.h` (additive)

- Add members mirroring RoPE: `bool use_ulysses_ = false;` and an `enable_ulysses()` setter.
- At the **top of `forward_cp`** (before any ring/shard work, ~line 248+) add a single gate:
  `if (use_ulysses_) return forward_ulysses(q, k, v, unshard, pre_sharded);`
  so the entire ring path is untouched.
- `forward_ulysses` (new private method, body delegating to `UlyssesAttention.h`):
  - **contiguous** sequence shard if `!pre_sharded` (`make_shards_inplace_axis(world_size_, 2)`,
    take `[rank_]`) — **HeadTail/zig-zag is NOT used** (Ulysses needs contiguous shards);
  - **`pre_sharded == true` is FORBIDDEN in Ulysses v1** — `throw` a clear error. (Critique
    point 1: the ring path's callers hand in HeadTail/zig-zag-sharded tensors; accepting
    `pre_sharded` would silently mis-order the gathered sequence — wrong mask, wrong grads, no
    crash. Re-enabling it later requires a tag distinguishing contiguous- vs zig-zag-sharded.)
  - asserts: `H % P == 0`, `T % P == 0`, and (MHA) `q/k/v` head counts equal;
  - call `ulysses_attention_forward(...)`;
  - **unshard with a PLAIN gather, NOT the ring unshard helper.** When `unshard == true`,
    mirror only the plain part of [ContextParallel.h:907-928](context_parallel/ContextParallel.h#L907-L928)
    (`all_gather` + contiguous rank-order reassembly into `[B,H,T,D]`) and **skip the
    `load_balancer_.unloadbalance()` call** at :942 — Ulysses output is already in linear order,
    so de-zigzagging would scramble it. (Critique point 2.) When `unshard == false`, return the
    seq-sharded `[B,H,T_local,D]`.

### Backward (`UlyssesAttentionBackward::apply`)

Saves `q_g, k_g, v_g` (head-sharded), `out_g`, `lse`, the all-to-all metadata, `pg_`, `scale`,
`is_causal`, the original input shapes, **and `unshard_`** — the bool passed into
`forward_ulysses`. `unshard_` MUST be persisted on the node: step 1 branches on it, and it is the
only state that branch can key on. Do **not** infer it from `grad_out`'s shape (fragile — breaks
the moment a third shape variant appears) and do **not** assume reuse of `ContextParallelBackward`
supplies it (the carve-outs above show ring state misfires for Ulysses). The `ws=2`/`ws=4` parity
test drives both `unshard=true` and `unshard=false` through this exact path, so both branches are
exercised. Steps:
1. take incoming `grad_out`; **if `unshard_` is true**, re-shard the full grad to local by
   **slicing this rank's contiguous `[B,H,T_local,D]` block** (`narrow` on the seq dim at
   `rank_*T_local`) — the plain inverse of the contiguous gather, with **no `unloadbalance`
   de-zigzag** (consistent with the forward; critique point 2); if `unshard_` is false, `grad_out`
   is already `[B,H,T_local,D]` and is used as-is,
2. **partition-backward (= combine)** on `grad_out_local` → `grad_out_g` `[B, H_local, T, D]`,
3. local `sdpa_fused_backward(q_g, k_g, v_g, grad_out_g, out_g, lse, is_causal, scale, 0, 0)`
   → `dq_g, dk_g, dv_g`,
4. **combine-backward (= partition)** on each → `dq_l, dk_l, dv_l` `[B, H, T_local, D]`,
5. form `full_grad_q/k/v` matching the shapes the `next_edges` point to. The forward sharded
   internally (`pre_sharded` is forbidden), so gather each local `[B,H,T_local,D]` grad back to
   full `[B,H,T,D]` with a **plain rank-ordered all-gather** (reuse `all_gather_along_seq`,
   [ContextParallelBackward.h:850-880](context_parallel/ContextParallelBackward.h#L850-L880)) and
   **unconditionally skip the `unloadbalance` de-zigzag** at
   [:778-785](context_parallel/ContextParallelBackward.h#L778-L785). Do **NOT** "mirror exactly"
   the ring backward's tail — that branch re-zigzags when `load_balance_` is set, which would
   silently corrupt Ulysses gradients in the original token order (critique point: the forward
   carve-out recurs here, and is harder to spot because forward numbers look fine). Then
   `return {full_grad_q, full_grad_k, full_grad_v}` (Node(3)).

Edge wiring in `forward_ulysses` copies the ring pattern: `set_next_edge(0/1/2, get_grad_edge(q/k/v))`
guarded by `requires_grad()`, then `output.set_grad_fn(node); output.set_requires_grad(true)`.

### Reused existing utilities (do not re-implement)

- `ProcessGroupNCCL::alltoall(...)` sync on `cp_comm_`; `mesh.get_process_group(0)`, `get_rank/get_worldsize`.
- Count/displacement convention from `AlltoAllRingRotator` (`RingRotator.h:205-281`).
- Tensor `view/reshape/transpose/contiguous` (`core/Tensor.h`); `make_shards_inplace_axis`.
- `sdpa_fused_forward` / `sdpa_fused_backward` (`FusedSDPAOp.h:51`) — `SDPAResult{out, lse}`.
- `Node` + `set_next_edge` + `get_grad_edge` (`autograd/Node.h`), and the `full_grad_*`
  gather/shard logic already in `ContextParallelBackward.h`.
- The forward `unshard` helper the ring path calls (keep identical contract).

---

## Pseudocode

Notation: `P = world_size`, `Tl = T/P` (local seq), `Hl = H/P` (local heads). All tensors fp32,
layout `[B, H, T, D]`. Raw collective = `pg->alltoall(send, recv, count, dtype, /*sync=*/true)`
on `cp_comm_`, uniform equal-split, `count = B*Hl*Tl*D` elements per peer.

### A. All-to-all layout helpers (`UlyssesAttention.h`) — raw, no autograd

```cpp
// seq-sharded -> head-sharded.  in:[B,H,Tl,D]  ->  out:[B,Hl,T,D]
Tensor ulysses_combine(ProcessGroupNCCL* pg, const Tensor& in, int P) {
  auto [B,H,Tl,D] = dims(in);  int Hl = H/P;
  // 1. expose the destination-rank axis on the head dim (group g -> rank g)
  Tensor s = in.reshape({B, P, Hl, Tl, D})        // split H = P*Hl
               .transpose(0,1)                     // [P, B, Hl, Tl, D]  (P outermost)
               .contiguous();                      // block g is contiguous -> goes to rank g
  // 2. uniform all-to-all: send block r to rank r; recv block s from rank s
  Tensor r = Tensor::empty({P, B, Hl, Tl, D}, in.opts());   // [src=s, B, Hl, Tl, D]
  pg->alltoall(s.data(), r.data(), B*Hl*Tl*D, in.dtype(), /*sync=*/true);
  // 3. merge (src, Tl) -> full contiguous T (src in rank order == global order)
  return r.transpose(0,1)            // [B, src, Hl, Tl, D]
          .transpose(1,2)            // [B, Hl, src, Tl, D]
          .contiguous()
          .reshape({B, Hl, T, D});   // T = P*Tl
}

// head-sharded -> seq-sharded.  in:[B,Hl,T,D]  ->  out:[B,H,Tl,D]   (exact inverse of combine)
Tensor ulysses_partition(ProcessGroupNCCL* pg, const Tensor& in, int P) {
  auto [B,Hl,T,D] = dims(in);  int Tl = T/P;  int H = Hl*P;
  Tensor s = in.reshape({B, Hl, P, Tl, D})   // split T = P*Tl (axis 2 = destination rank)
               .transpose(0,2)               // [P, Hl, B, Tl, D]
               .transpose(1,2)               // [P, B, Hl, Tl, D]
               .contiguous();
  Tensor r = Tensor::empty({P, B, Hl, Tl, D}, in.opts());   // [src=group s, B, Hl, Tl, D]
  pg->alltoall(s.data(), r.data(), B*Hl*Tl*D, in.dtype(), /*sync=*/true);
  return r.transpose(0,1)            // [B, src=group, Hl, Tl, D]
          .contiguous()
          .reshape({B, H, Tl, D});   // H = P*Hl, group s -> heads [s*Hl:(s+1)*Hl]
}
```
`ulysses_combine` backward == `ulysses_partition` (and vice-versa): a pure permutation's adjoint
is the inverse permutation, no scaling (confirmed by the critique).

### B. Forward (`ContextParallel::forward_ulysses`, gated at top of `forward_cp`)

```cpp
Tensor forward_cp(Tensor& q, Tensor& k, Tensor& v, bool unshard, bool pre_sharded) {
  if (use_ulysses_) return forward_ulysses(q, k, v, unshard, pre_sharded);
  ... existing ring path unchanged ...
}

Tensor forward_ulysses(Tensor& q, Tensor& k, Tensor& v, bool unshard, bool pre_sharded) {
  if (pre_sharded) throw std::runtime_error(
      "Ulysses v1: pre_sharded unsupported (would accept zig-zag-sharded input); pass full [B,H,T,D]");
  int P = world_size_;
  // shapes from the FULL tensors (q is [B,H,T,D])
  assert(H % P == 0 && T % P == 0);
  assert(q.heads == k.heads && k.heads == v.heads);     // MHA only

  // 1. contiguous sequence shard (NO HeadTail/zig-zag): rank r takes seq block r
  Tensor ql = q.make_shards_inplace_axis(P, /*seq_dim=*/2)[rank_];   // [B,H,Tl,D]
  Tensor kl = k.make_shards_inplace_axis(P, 2)[rank_];
  Tensor vl = v.make_shards_inplace_axis(P, 2)[rank_];

  // 2. combine: seq-sharded -> head-sharded full sequence
  Tensor qg = ulysses_combine(pg_, ql, P);   // [B,Hl,T,D]
  Tensor kg = ulysses_combine(pg_, kl, P);
  Tensor vg = ulysses_combine(pg_, vl, P);

  // 3. ONE full square causal SDPA — reuse existing kernel; no offsets, no ring, no LSE merge
  SDPAResult res = sdpa_fused_forward(qg, kg, vg, /*is_causal=*/true, attn_scale_,
                                      /*q_off=*/0, /*k_off=*/0);   // res.out [B,Hl,T,D], res.lse [B,Hl,T,1]
  Tensor out_g = res.out;

  // 4. partition: head-sharded -> seq-sharded local output
  Tensor out_l = ulysses_partition(pg_, out_g, P);   // [B,H,Tl,D]

  // 5. optional unshard: PLAIN rank-order all-gather; NO unloadbalance de-zigzag
  Tensor output = unshard ? plain_all_gather_seq(out_l)   // [B,H,T,D] (mirror CP.h:907-928, skip :942)
                          : out_l;                        // [B,H,Tl,D]

  // 6. autograd wiring (mirror ring pattern)
  if (q/k/v.requires_grad()) {
    auto node = make_shared<UlyssesAttentionBackward>(
        qg, kg, vg, out_g, res.lse, /*P,B,H,T,D,*/ pg_, attn_scale_, /*is_causal=*/true,
        rank_, unshard /*persisted*/);
    if (q.requires_grad()) node->set_next_edge(0, get_grad_edge(q));
    if (k.requires_grad()) node->set_next_edge(1, get_grad_edge(k));
    if (v.requires_grad()) node->set_next_edge(2, get_grad_edge(v));
    output.set_grad_fn(node); output.set_requires_grad(true);
  }
  return output;
}
```

### C. Backward (`UlyssesAttentionBackward : Node`, `UlyssesAttentionBackward.h`)

```cpp
class UlyssesAttentionBackward : public Node {
  // SAVED: qg, kg, vg, out_g, lse, P,B,H,T,D, pg_, scale, is_causal, rank_, unshard_
  UlyssesAttentionBackward(...) : Node(3) { /* store all of the above, incl. unshard_ */ }

  vector<Tensor> apply(vector<Tensor>&& grads) override {
    Tensor grad_out = grads[0];

    // 1. branch on the PERSISTED unshard_ flag (NOT inferred from shape)
    Tensor grad_out_l = unshard_
        ? grad_out.narrow(/*seq_dim=*/2, rank_*Tl, Tl).contiguous()  // [B,H,Tl,D], no de-zigzag
        : grad_out;                                                  // already [B,H,Tl,D]

    // 2. partition-backward == combine: seq-sharded grad -> head-sharded
    Tensor grad_out_g = ulysses_combine(pg_, grad_out_l, P);         // [B,Hl,T,D]

    // 3. ONE local SDPA backward — reuse existing kernel (square, no offsets)
    auto [dqg, dkg, dvg] = sdpa_fused_backward(qg, kg, vg, grad_out_g, out_g, lse,
                                               is_causal_, scale_, /*q_off=*/0, /*k_off=*/0);

    // 4. combine-backward == partition: head-sharded grads -> seq-sharded local
    Tensor dql = ulysses_partition(pg_, dqg, P);   // [B,H,Tl,D]
    Tensor dkl = ulysses_partition(pg_, dkg, P);
    Tensor dvl = ulysses_partition(pg_, dvg, P);

    // 5. gather to match upstream edge shape [B,H,T,D]: PLAIN rank-order all-gather,
    //    UNCONDITIONALLY skip unloadbalance (do NOT mirror ring's load_balance_ branch)
    Tensor full_dq = plain_all_gather_seq(dql);
    Tensor full_dk = plain_all_gather_seq(dkl);
    Tensor full_dv = plain_all_gather_seq(dvl);
    return {full_dq, full_dk, full_dv};
  }
};
```

> `plain_all_gather_seq` = `all_gather` + contiguous rank-order reassembly into `[B,H,T,D]`,
> i.e. `all_gather_along_seq` ([ContextParallelBackward.h:850-880](context_parallel/ContextParallelBackward.h#L850-L880))
> WITHOUT the `unloadbalance` step.

---

## Files to create / modify

| File | Change |
|---|---|
| `context_parallel/UlyssesAttention.h` | **NEW** — all-to-all combine/partition helpers + `ulysses_attention_forward` |
| `context_parallel/UlyssesAttentionBackward.h` | **NEW** — `UlyssesAttentionBackward : Node` (reverse all-to-all + local SDPA bwd) |
| `context_parallel/ContextParallel.h` | **+** `use_ulysses_` member, `enable_ulysses()` setter, top-of-`forward_cp` gate, `forward_ulysses` method, include new header |
| `Tests/cp_ulysses_parity.cpp` | **NEW** — ws=2 parity test (forward + dQ/dK/dV vs single-GPU full causal SDPA) |
| `Makefile` | **+** additive targets `cp-ulysses` / `run-cp-ulysses` (mirror `cp-rope-standin`) |

No existing function bodies are altered; every change is a new symbol or a single gated branch.
Default builds (`use_ulysses_ == false`) are byte-identical to today.

---

## Verification

1. **Static**: `mpic++ -fsyntax-only` over `ContextParallel.h` + new headers in default mode
   AND with Ulysses enabled; confirm the ring/GPT-2 path is unchanged (gate defaults off).
2. **Parity (the real gate)** — `Tests/cp_ulysses_parity.cpp`, modeled on the proven
   `Tests/cp_rope_standin_parity.cpp`:
   - build a single-GPU reference: full `[B,H,T,D]` causal SDPA forward + backward;
   - run the Ulysses CP path by feeding **full `[B,H,T,D]` tensors with `pre_sharded=false`** and
     letting `forward_ulysses` do its own internal contiguous split (do NOT pass pre-sharded
     tensors — `pre_sharded=true` throws by design);
   - assert **forward** output and **dQ/dK/dV** match the reference (`cos ≈ 1.0`, maxdiff within
     TF32 noise `1e-4..1e-6`), after all-gathering the sharded CP results.
   - **Run at both `ws=2` AND `ws=4`** (`make run-cp-ulysses NP=2` and `NP=4`). The `ws=4` run
     is mandatory: at `P=2` the all-to-all `P`-axis permute has only one possible swap, so a
     transposed `P` vs `B`/`H_local`/`T_local` axis bug can stay correct by symmetry and only
     surface at `P≥4` (critique point 3). Requires `H % 4 == 0` and `T % 4 == 0` in the test config.
   - test **both `unshard=true` and `unshard=false`** for coverage of both return shapes
     (`unshard=false` is the building block for future layer-chaining once `pre_sharded`
     re-enablement lands). `pre_sharded=true` is forbidden in v1, so it is not exercised.
3. **Regression**: run an existing ring test/`run-cp-rope-standin` to confirm the ring path
   still passes (no behavioral change).

## Memory note (capacity planning)
Unlike the ring path's incrementally-sized blocks, Ulysses' local SDPA operates on the full
`T×T` causal attention per call. FLOPs/compute are comparable, but **peak activation memory per
rank rises** (each rank holds the full `[B, H_local, T, D]` Q/K/V/O), which matters at large `P`
with few heads per rank. Worth stating so it doesn't surprise capacity planning. (Critique point 4.)

## Out of scope (documented future work)
- **Fuse the three Q/K/V all-to-alls into one** (concat along head/last dim before a single
  collective) — correctness-neutral; an easy throughput win alongside the overlap work below.
  v1 does three separate blocking all-to-alls for clarity. (Critique point 5.)
- Overlapped all-to-all (`alltoallv_async_stream` on `cpRingStream()` + `recordStream`/`streamWait`).
- GQA/MQA — **now planned; see "GQA Extension (v2)" below.**
- Re-enabling `pre_sharded=true` for Ulysses — **DONE** (accepted when `!load_balance_`; zig-zag still guarded).
- Wiring `CP_ATTN_MODE=ulysses` into `gpt2_cp_test.cpp` — **DONE** (env gate forces `load_balancing=false`).

---

# GQA Extension (v2) — grouped/multi-query Ulysses, DeepSpeed-style

## Context
v1 Ulysses is MHA-only (`nkv == nq`). GQA/MQA has `nkv < nq` (group size `g = nq/nkv`). We want the
**DeepSpeed strategy** (verified against `ulysses_sp.py`): **shard-first (Strategy B)** — the KV
all-to-all carries only `nkv` heads (comm **× 1**, never the `× g` of a full nkv→nq expansion), and
the grouped broadcast happens **locally** after the gather. The only nuance: when `nkv < P` you can't
give each rank a distinct KV head, so KV is **partially replicated** by `rep = P/nkv` (to exactly `P`
heads, one per rank) — never all the way to `nq`.

Our CP SDPA is MHA-only (equal head counts; [FusedSDPAOp.h:69-74](context_parallel/FusedSDPAOp.h#L69-L74)
throws on mismatch), and OwnTensor has **no `repeat_interleave`**. So DeepSpeed's `enable_gqa` local
broadcast is done **explicitly**: expand KV `kv_local → nq_local` heads before the MHA SDPA, and
**sum each group** in backward (the adjoint of a broadcast). This stays additive and reuses the same
existing kernel — still **no new attention kernel**.

## STRICTLY ADDITIVE structure (no existing body is modified)
GQA is a **separate, parallel path** — the proven MHA v1 code is left byte-for-byte:
- **New** free helpers in `UlyssesAttention.h` (`head_repeat_interleave`, `head_group_reduce`); the
  existing `ulysses_combine`/`partition`/`gather_seq` are untouched (already head-count-generic).
- **New** method `ContextParallel::forward_ulysses_gqa(...)` — the MHA `forward_ulysses` body is
  unchanged except for **one added early-return line** at its top (the same additive-gate idiom as the
  `use_ulysses_` gate in `forward_cp`):
  `if (k.shape().dims[1] != q.shape().dims[1]) return forward_ulysses_gqa(q,k,v,unshard,pre_sharded);`
- **New** node class `UlyssesGQAAttentionBackward` in a **new** header
  `UlyssesGQAAttentionBackward.h` — the MHA `UlyssesAttentionBackward` is not touched.
- **New** GQA cases appended to `cp_ulysses_parity.cpp`; existing MHA cases unchanged.
So MHA (`nkv == nq`) never enters any new code; GQA (`nkv < nq`) never enters the MHA code.

## Head math (all derived from shapes; validated)
`P=world_size`, `g = nq/nkv`. Two regimes, mirroring DeepSpeed:
- **`nkv ≥ P`** (require `nkv % P == 0`): no replication, `eff_kv = nkv`, `kv_local = nkv/P`.
- **`nkv < P`** (require `P % nkv == 0`): replicate KV by `rep = P/nkv` → `eff_kv = P`, `kv_local = 1`.

Then `nq_local = nq/P`, and the **local group size** `g_local = nq_local / kv_local`. Contiguous
head-scatter is automatically group-aligned, so rank `r`'s query heads and its `kv_local` head(s)
belong to the same global GQA group, and `g_local` equals the queries-per-local-KV-head. (Checked for
MHA `g=1`, GQA `nkv≥P`, and MQA/`nkv<P`.) MHA degenerates to `rep=1, g_local=1` ⇒ the v1 path
byte-for-byte.

Divisibility guard (DeepSpeed's exact rule): `nq % P == 0` **and** `nq % nkv == 0` **and**
(`nkv % P == 0` **or** `P % nkv == 0`).

## Two new head helpers (raw, non-autograd) in `UlyssesAttention.h`
OwnTensor has no `repeat_interleave`; build it from `narrow`+`cat`, and its adjoint from
`reshape`+`reduce_sum` ([Reduction.h:23](BluTrain/Tensor-Implementations/include/ops/UnaryOps/Reduction.h#L23)).
```cpp
// repeat_interleave along head dim: [B,H,T,D] -> [B,H*r,T,D], out head e = in head (e / r).
inline Tensor head_repeat_interleave(const Tensor& x, int r) {
  if (r == 1) return x;                                  // MHA no-op (v1 path unchanged)
  const auto& d = x.shape().dims; int64_t H = d[1];
  std::vector<Tensor> parts; parts.reserve(H * r);
  for (int64_t h = 0; h < H; ++h)
    for (int j = 0; j < r; ++j) parts.push_back(x.narrow(1, h, 1)); // [B,1,T,D]
  return Tensor::cat(parts, 1).contiguous();             // [B,H*r,T,D]
}
// adjoint (sum each group of r): [B,groups*r,T,D] -> [B,groups,T,D].
inline Tensor head_group_reduce(const Tensor& x, int64_t groups, int r) {
  if (r == 1) return x;
  const auto& d = x.shape().dims; int64_t B=d[0], T=d[2], D=d[3];
  Tensor g5 = x.reshape(Shape({{B, groups, (int64_t)r, T, D}}));
  return OwnTensor::reduce_sum(g5, /*axes=*/{2}, /*keepdim=*/false).contiguous(); // [B,groups,T,D]
}
```
`ulysses_combine`/`ulysses_partition`/`ulysses_gather_seq` are already generic on head count — they
work unchanged for `nq`, `eff_kv`, or `nkv` heads.

## Forward (NEW method `forward_ulysses_gqa`; entered only when `nkv < nq`)
```cpp
const int64_t nq = q.dims[1], nkv = k.dims[1];          // v.dims[1] == nkv
require(nq % P == 0 && nq % nkv == 0 && (nkv % P == 0 || P % nkv == 0));
int rep     = (nkv < P) ? (P / nkv) : 1;                // DeepSpeed kv_replication_factor path
// (obtain local contiguous shards ql[B,nq,Tl,D], kl/vl[B,nkv,Tl,D] as in v1: narrow or pre_sharded)

// 1. optional PARTIAL replication BEFORE the KV all-to-all (only when nkv < P)
Tensor kl_r = head_repeat_interleave(kl, rep);          // [B, eff_kv, Tl, D], eff_kv = nkv*rep
Tensor vl_r = head_repeat_interleave(vl, rep);

// 2. combine: Q by nq, KV by eff_kv (SEPARATE all-to-alls; KV carries eff_kv <= P heads => Strategy B)
Tensor qg = ulysses_combine(pg_, ql,   P);              // [B, nq/P,     T, D]
Tensor kg = ulysses_combine(pg_, kl_r, P);              // [B, kv_local, T, D]   (kv_local=eff_kv/P)
Tensor vg = ulysses_combine(pg_, vl_r, P);
int kv_local = kg.dims[1], nq_local = qg.dims[1], g_local = nq_local / kv_local;

// 3. LOCAL broadcast kv_local -> nq_local (the "enable_gqa" broadcast, done explicitly), then MHA SDPA
Tensor kg_e = head_repeat_interleave(kg, g_local);      // [B, nq/P, T, D]
Tensor vg_e = head_repeat_interleave(vg, g_local);
SDPAResult res = sdpa_fused_forward(qg, kg_e, vg_e, /*causal=*/true, scale, 0, 0);

// 4. partition Q-shaped output back; 5. unshard/gather exactly as v1
Tensor out_l = ulysses_partition(pg_, res.out, P);      // [B, nq, Tl, D]
... (unshard plain-gather or local) ...
// forward_ulysses_gqa handles pre_sharded IDENTICALLY to MHA forward_ulysses:
//   throw if (pre_sharded && load_balance_)  [zig-zag guard];
//   if pre_sharded -> use q/k/v as local contiguous shards; else narrow this rank's slice.
// node saves (name EVERY flag explicitly — do NOT bury in "flags"; the MHA round proved this bites):
//   qg, kg, vg (UN-expanded), res.out, res.lse, scale, g_local, kv_local, nkv, rep, nq, P, rank_,
//   and the THREE booleans, each persisted (backward branches on them, NEVER infers from shape):
//     unshard_      -> step 1: narrow incoming grad to local slice iff true
//     pre_sharded_  -> step 8: return LOCAL grads iff true, else all-gather to full
//     is_causal_    -> passed to sdpa_fused_backward (always true for Ulysses, but stored, not assumed)
```

## Backward (NEW node `UlyssesGQAAttentionBackward`; MHA node untouched)
```cpp
grad_out_l = unshard_ ? narrow(grad_out) : grad_out;
grad_out_g = ulysses_combine(pg_, grad_out_l, P);                 // [B, nq/P, T, D]
kg_e = head_repeat_interleave(kg, g_local);                      // re-expand (cheap, deterministic)
vg_e = head_repeat_interleave(vg, g_local);
auto [dqg, dkg_e, dvg_e] = sdpa_fused_backward(qg, kg_e, vg_e, grad_out_g, out_g, lse, causal, scale);
Tensor dkg = head_group_reduce(dkg_e, kv_local, g_local);        // adjoint of local broadcast
Tensor dvg = head_group_reduce(dvg_e, kv_local, g_local);        // [B, kv_local, T, D]
Tensor dq_l = ulysses_partition(pg_, dqg, P);                    // [B, nq,     Tl, D]
Tensor dk_c = ulysses_partition(pg_, dkg, P);                    // [B, eff_kv, Tl, D]
Tensor dv_c = ulysses_partition(pg_, dvg, P);
Tensor dk_l = head_group_reduce(dk_c, nkv, rep);                 // adjoint of replication (no-op if rep==1)
Tensor dv_l = head_group_reduce(dv_c, nkv, rep);                 // [B, nkv, Tl, D]
// 8. return shape = upstream edge shape, branch on the PERSISTED pre_sharded_:
//    pre_sharded_ -> {dq_l, dk_l, dv_l} LOCAL (no gather);
//    else         -> gather each to full (q by nq heads, k/v by nkv heads). Both edge-shape correct.
```
`unshard_` and `pre_sharded_` are stored on the node exactly as in the MHA `UlyssesAttentionBackward`
(step 1 narrows the incoming grad iff `unshard_`; step 8 gathers iff `!pre_sharded_`). Neither is
inferred from tensor shape.
Two adjoints, in reverse order of the two broadcasts: local group-reduce (undo step 3) then
replication group-reduce (undo step 1). Summation is the correct adjoint of a broadcast/copy.

## Files (all additive)
| File | Change |
|---|---|
| `context_parallel/UlyssesAttention.h` | **+** `head_repeat_interleave`, `head_group_reduce`; **+** include `ops/UnaryOps/Reduction.h`. combine/partition/gather **unchanged**. |
| `context_parallel/UlyssesGQAAttentionBackward.h` | **NEW** — `UlyssesGQAAttentionBackward : Node` (re-expand → sdpa_fused_backward → two group-reduces → partition → gather/local). |
| `context_parallel/ContextParallel.h` | **+** new private `forward_ulysses_gqa`; **+** one early-return guard line at top of `forward_ulysses`; **+** include the new header. MHA `forward_ulysses` body **unchanged**. |
| `Tests/cp_ulysses_parity.cpp` | **+** GQA/MQA cases (new calls). Existing MHA cases **unchanged**. |

`UlyssesAttentionBackward.h` and the MHA `forward_ulysses` body are **not modified**. No existing
function body changes except the single additive guard line.

## Verification
- **Parity** (extend `cp_ulysses_parity`): add **GQA `nkv≥P`** and **MQA `nkv<P`** cases.
  - **ws=2**: GQA `nq=8,nkv=2,g=4` (⇒ `kv_local=1`) and MQA `nkv=1` (⇒ `rep=2`).
  - **ws=4 (MANDATORY axis-coverage — same reasoning as MHA; a transposed P-axis bug in the Q or KV
    `combine`/`partition` is invisible at P=2):** GQA `nq=16,nkv=4,P=4` (`nkv≥P`, `kv_local=1,g_local=4`)
    and MQA `nq=16,nkv=1,P=4` (`rep=4,kv_local=1`). Need `nq%4==0`, `nkv|nq`, `T%4==0`.
  - Reference = single-GPU causal SDPA with KV `head_repeat_interleave`d to `nq` (plain MHA over the
    expanded KV); the reference dK/dV are `head_group_reduce`d back to `[B,nkv,T,D]` (same group-sum the
    CP path applies) before comparison. Assert forward + dQ/dK/dV `cos≈1.0`.
  - Keep the existing MHA cases (regression). ws=4 needs a 4-GPU box; the test's device mapping is
    already oversubscription-safe so it runs on 2 GPUs via GPU-sharing if needed.
- **Regression**: MHA Ulysses parity + ring `run-cp-rope-standin` still green (GQA path is `nkv<nq`
  only; `nkv==nq` degenerates to the verified v1 code).

## Out of scope (v2)
- A **GQA GPT-2 model** in `gpt2_cp_test.cpp` (needs a separate `nkv`-head KV projection — a model
  change beyond CP). GQA here is a CP-layer capability exercised by the parity test; `CP_ATTN_MODE=
  ulysses` training stays MHA.
- GQA-aware **fused** local kernel (would avoid the explicit expand/reduce) — only if measured worth it.

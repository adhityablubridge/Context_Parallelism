#pragma once

// ---------------------------------------------------------------------------
// UlyssesAttentionBackward.h
//
// Autograd node for the backward pass of Ulysses sequence-parallel attention
// (additive; only constructed when ContextParallel::enable_ulysses() is set).
//
// The Ulysses forward is: shard(seq) -> combine(all-to-all) -> full causal SDPA
// -> partition(all-to-all) -> [optional plain gather]. Each all-to-all is a pure
// permutation, so its adjoint is the inverse permutation with no scaling:
//   - d/d(combine input)   == partition(grad)
//   - d/d(partition input) == combine(grad)
// The local SDPA backward re-uses the EXISTING CP kernel via sdpa_fused_backward
// (FusedSDPAOp.h) -- no new kernel.
//
// apply() steps (mirrors the forward in reverse):
//   1. if unshard_: narrow the full grad to this rank's contiguous seq slice;
//      else grad is already local [B,H,Tl,D].   (branch on the PERSISTED flag)
//   2. combine (= partition-backward): seq-sharded grad -> head-sharded.
//   3. ONE sdpa_fused_backward on the gathered head group (square, no offsets).
//   4. partition (= combine-backward): head-sharded grads -> seq-sharded local.
//   5. plain rank-order gather to full [B,H,T,D] to match the upstream edges;
//      NO unloadbalance de-zigzag (Ulysses shards contiguously).
// ---------------------------------------------------------------------------

#include "autograd/Node.h"
#include "core/Tensor.h"
#include "process_group/ProcessGroupNCCL.h"

#include "context_parallel/FusedSDPAOp.h"
#include "context_parallel/UlyssesAttention.h"

#include <memory>
#include <stdexcept>
#include <vector>

using namespace OwnTensor;

class UlyssesAttentionBackward : public Node {
public:
  UlyssesAttentionBackward(Tensor q_g,        // [B, Hl, T, D] head-sharded Q
                           Tensor k_g,        // [B, Hl, T, D]
                           Tensor v_g,        // [B, Hl, T, D]
                           Tensor out_g,      // [B, Hl, T, D] local SDPA output
                           Tensor lse,        // [B, Hl, T, 1]
                           std::shared_ptr<ProcessGroupNCCL> pg, float attn_scale,
                           bool is_causal, int world_size, int rank,
                           bool unshard, bool pre_sharded)
      : Node(3), q_g_(q_g), k_g_(k_g), v_g_(v_g), out_g_(out_g), lse_(lse),
        pg_(pg), attn_scale_(attn_scale), is_causal_(is_causal),
        world_size_(world_size), rank_(rank), unshard_(unshard),
        pre_sharded_(pre_sharded) {}

  const char *name() const override { return "UlyssesAttentionBackward"; }

  std::vector<Tensor> apply(std::vector<Tensor> &&grads) override {
    if (grads.empty() || !grads[0].is_valid())
      throw std::runtime_error("UlyssesAttentionBackward: no gradient provided");

    const int P = world_size_;
    Tensor grad_out = grads[0];

    // 1. Branch on the PERSISTED unshard_ flag (never inferred from shape).
    Tensor grad_out_l;
    if (unshard_) {
      // grad_out is full [B,H,T,D]; take this rank's contiguous seq slice.
      const int64_t Tl = grad_out.shape().dims[2] / P;
      grad_out_l =
          grad_out.narrow(2, static_cast<int64_t>(rank_) * Tl, Tl).contiguous();
    } else {
      grad_out_l = grad_out.contiguous(); // already [B,H,Tl,D]
    }

    // 2. combine (= partition-backward): seq-sharded grad -> head-sharded.
    Tensor grad_out_g = cp::ulysses_combine(pg_, grad_out_l, P); // [B,Hl,T,D]

    // 3. ONE local SDPA backward -- reuse existing kernel (square, no offsets).
    std::vector<Tensor> dqkv = sdpa_fused_backward(
        q_g_, k_g_, v_g_, grad_out_g, out_g_, lse_, is_causal_, attn_scale_,
        /*q_offset=*/0, /*k_offset=*/0);

    // 4. partition (= combine-backward): head-sharded grads -> seq-sharded local.
    Tensor dq_l = cp::ulysses_partition(pg_, dqkv[0], P); // [B,H,Tl,D]
    Tensor dk_l = cp::ulysses_partition(pg_, dqkv[1], P);
    Tensor dv_l = cp::ulysses_partition(pg_, dqkv[2], P);

    // 5. Return shape must match the upstream edges (the tensors forward_ulysses
    //    set edges on):
    //    - pre_sharded_  : edges are LOCAL [B,H,Tl,D] -> return local, NO gather.
    //    - else (full)   : edges are FULL  [B,H,T,D]  -> plain rank-order gather.
    if (pre_sharded_)
      return {dq_l, dk_l, dv_l};

    Tensor full_dq = cp::ulysses_gather_seq(pg_, dq_l, P);
    Tensor full_dk = cp::ulysses_gather_seq(pg_, dk_l, P);
    Tensor full_dv = cp::ulysses_gather_seq(pg_, dv_l, P);
    return {full_dq, full_dk, full_dv};
  }

private:
  Tensor q_g_, k_g_, v_g_, out_g_, lse_;
  std::shared_ptr<ProcessGroupNCCL> pg_;
  float attn_scale_;
  bool is_causal_;
  int world_size_;
  int rank_;
  bool unshard_;
  bool pre_sharded_;
};

#pragma once

// ---------------------------------------------------------------------------
// UlyssesGQAAttentionBackward.h
//
// Autograd node for the backward pass of GQA/MQA Ulysses attention (additive;
// only constructed when nkv < nq via ContextParallel::forward_ulysses_gqa). The
// MHA UlyssesAttentionBackward is a SEPARATE node and is not touched.
//
// Forward (GQA) was:
//   [optional] KV partial-replication (rep = P/nkv, only when nkv < P)
//   -> combine (all-to-all)  Q by nq, KV by eff_kv
//   -> local broadcast kv_local -> nq_local  (repeat_interleave by g_local)
//   -> ONE MHA sdpa_fused_forward  -> partition -> [optional] plain gather.
//
// apply() reverses it (a broadcast's adjoint is a group-sum):
//   1. narrow incoming grad to local iff unshard_  (persisted, never inferred).
//   2. combine (= partition-backward): seq-sharded grad -> head-sharded.
//   3. re-expand kg/vg to nq_local, ONE sdpa_fused_backward (existing kernel).
//   4. group-reduce dkv over g_local  (adjoint of local broadcast).
//   5. partition (= combine-backward): head-sharded grads -> seq-sharded local.
//   6. group-reduce dk/dv over rep    (adjoint of KV replication; no-op if rep==1).
//   7. return LOCAL grads iff pre_sharded_, else plain rank-order gather to full.
// ---------------------------------------------------------------------------

#include "autograd/Node.h"
#include "core/Tensor.h"
#include "ProcessGroupNCCL.h"   // canonical BluTrain PG (via -IBluTrain/dist/communication/include)

#include "context_parallel/FusedSDPAOp.h"
#include "context_parallel/UlyssesAttention.h"

#include <memory>
#include <stdexcept>
#include <vector>

using namespace OwnTensor;

class UlyssesGQAAttentionBackward : public Node {
public:
  UlyssesGQAAttentionBackward(Tensor q_g,   // [B, nq_local,  T, D]
                              Tensor k_g,   // [B, kv_local,  T, D]  (UN-expanded)
                              Tensor v_g,   // [B, kv_local,  T, D]
                              Tensor out_g, // [B, nq_local,  T, D]
                              Tensor lse,   // [B, nq_local,  T, 1]
                              std::shared_ptr<ProcessGroupNCCL> pg,
                              float attn_scale, bool is_causal, int world_size,
                              int rank, bool unshard, bool pre_sharded,
                              int g_local, int64_t kv_local, int64_t nkv, int rep,
                              int64_t nq)
      : Node(3), q_g_(q_g), k_g_(k_g), v_g_(v_g), out_g_(out_g), lse_(lse),
        pg_(pg), attn_scale_(attn_scale), is_causal_(is_causal),
        world_size_(world_size), rank_(rank), unshard_(unshard),
        pre_sharded_(pre_sharded), g_local_(g_local), kv_local_(kv_local),
        nkv_(nkv), rep_(rep), nq_(nq) {}

  const char *name() const override { return "UlyssesGQAAttentionBackward"; }

  std::vector<Tensor> apply(std::vector<Tensor> &&grads) override {
    if (grads.empty() || !grads[0].is_valid())
      throw std::runtime_error("UlyssesGQAAttentionBackward: no gradient provided");

    const int P = world_size_;
    Tensor grad_out = grads[0];

    // 1. Branch on the PERSISTED unshard_ flag (never inferred from shape).
    Tensor grad_out_l;
    if (unshard_) {
      const int64_t Tl = grad_out.shape().dims[2] / P;
      grad_out_l =
          grad_out.narrow(2, static_cast<int64_t>(rank_) * Tl, Tl).contiguous();
    } else {
      grad_out_l = grad_out.contiguous(); // already [B, nq, Tl, D]
    }

    // 2. combine (= partition-backward): seq-sharded grad -> head-sharded.
    Tensor grad_out_g = cp::ulysses_combine(pg_, grad_out_l, P); // [B, nq_local, T, D]

    // 3. Re-expand kv_local -> nq_local (same broadcast the forward used), then
    //    ONE MHA SDPA backward on the existing kernel.
    Tensor kg_e = cp::head_repeat_interleave(k_g_, g_local_); // [B, nq_local, T, D]
    Tensor vg_e = cp::head_repeat_interleave(v_g_, g_local_);
    std::vector<Tensor> dqkv =
        sdpa_fused_backward(q_g_, kg_e, vg_e, grad_out_g, out_g_, lse_,
                            is_causal_, attn_scale_, /*q_offset=*/0, /*k_offset=*/0);

    // 4. Adjoint of the local broadcast: sum each group of g_local -> kv_local.
    Tensor dkg = cp::head_group_reduce(dqkv[1], kv_local_, g_local_); // [B, kv_local, T, D]
    Tensor dvg = cp::head_group_reduce(dqkv[2], kv_local_, g_local_);

    // 5. partition (= combine-backward): head-sharded grads -> seq-sharded local.
    Tensor dq_l = cp::ulysses_partition(pg_, dqkv[0], P); // [B, nq,     Tl, D]
    Tensor dk_c = cp::ulysses_partition(pg_, dkg, P);     // [B, eff_kv, Tl, D]
    Tensor dv_c = cp::ulysses_partition(pg_, dvg, P);

    // 6. Adjoint of the KV replication: sum each group of rep -> nkv (no-op if rep==1).
    Tensor dk_l = cp::head_group_reduce(dk_c, nkv_, rep_); // [B, nkv, Tl, D]
    Tensor dv_l = cp::head_group_reduce(dv_c, nkv_, rep_);

    // 7. Return shape must match the upstream edges:
    //    - pre_sharded_ : LOCAL [B,nq/nkv,Tl,D] (no gather);
    //    - else         : plain rank-order gather to full (q by nq, k/v by nkv heads).
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
  int g_local_;
  int64_t kv_local_;
  int64_t nkv_;
  int rep_;
  int64_t nq_;
};

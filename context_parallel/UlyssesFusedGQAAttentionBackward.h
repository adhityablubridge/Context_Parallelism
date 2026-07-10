#pragma once

// ---------------------------------------------------------------------------
// UlyssesFusedGQAAttentionBackward.h
//
// Autograd node for the backward of the FUSED (RoPE + QK-norm + GQA) Ulysses
// path (additive; only constructed by ContextParallel::forward_ulysses_fused).
// Calls the team's bf16 fused backward kernel directly. The v1 MHA and v2
// plain-GQA Ulysses nodes are separate and untouched.
//
// Forward was:  [nkv<P: replicate KV by rep] -> combine (all-to-all) ->
//               pack bf16 -> gqa_fused_flash_attn_forward -> partition -> gather.
// apply() reverses it:
//   1. narrow incoming grad to local iff unshard_ (persisted, never inferred).
//   2. combine (= partition-backward): seq-sharded grad -> head-sharded.
//   3. bf16 pack dO, call gqa_fused_flash_attn_backward -> dQ/dK/dV (+dgammas).
//   4. partition (= combine-backward) each grad -> seq-sharded local.
//   5. replication adjoint (head_group_reduce by rep; no-op if rep==1).
//   6. gammas: SUM all-reduce over the CP group ([hd] shared param, head-partitioned).
//   7. return LOCAL grads iff pre_sharded_, else PLAIN rank-order gather (no
//      unloadbalance de-zigzag) to full; append gamma grads when has_gamma.
// ---------------------------------------------------------------------------

#include "autograd/Node.h"
#include "core/Tensor.h"
#include "dtype/Types.h"
#include "ops/helpers/AttentionKernels.h"
#include "ProcessGroupNCCL.h"   // canonical BluTrain PG (via -IBluTrain/dist/communication/include)

#include "context_parallel/UlyssesAttention.h"

#include <memory>
#include <stdexcept>
#include <vector>

using namespace OwnTensor;

class UlyssesFusedGQAAttentionBackward : public Node {
public:
  UlyssesFusedGQAAttentionBackward(
      Tensor packed,   // bf16 [Q|K|V] (raw, pre-norm, pre-rope) as fed to forward
      Tensor out_bf,   // bf16 [B, nq_local, T, hd] forward output
      Tensor lse,      // fp32 [B, nq_local, T]
      Tensor q_rstd,   // fp32 [B, nq_local, T]
      Tensor k_rstd,   // fp32 [B, kv_local, T]
      Tensor cos_sin_cache, Tensor q_gamma, Tensor k_gamma,
      std::shared_ptr<ProcessGroupNCCL> pg, bool is_causal, int world_size,
      int rank, bool unshard, bool pre_sharded, int64_t B, int64_t nq_local,
      int64_t kv_local, int64_t hd, int cache_seq_len, float eps,
      bool interleaved, int64_t nkv, int rep, int64_t nq)
      : Node((q_gamma.is_valid() && k_gamma.is_valid()) ? 5 : 3),
        packed_(packed), out_bf_(out_bf), lse_(lse), q_rstd_(q_rstd),
        k_rstd_(k_rstd), cache_(cos_sin_cache), q_gamma_(q_gamma),
        k_gamma_(k_gamma), pg_(pg), is_causal_(is_causal),
        world_size_(world_size), rank_(rank), unshard_(unshard),
        pre_sharded_(pre_sharded), B_(B), nq_local_(nq_local),
        kv_local_(kv_local), hd_(hd), cache_seq_len_(cache_seq_len), eps_(eps),
        interleaved_(interleaved), nkv_(nkv), rep_(rep), nq_(nq),
        has_gamma_(q_gamma.is_valid() && k_gamma.is_valid()) {}

  const char *name() const override { return "UlyssesFusedGQAAttentionBackward"; }

  std::vector<Tensor> apply(std::vector<Tensor> &&grads) override {
    if (grads.empty() || !grads[0].is_valid())
      throw std::runtime_error("UlyssesFusedGQAAttentionBackward: no gradient");

    const int P = world_size_;
    const int64_t T = out_bf_.shape().dims[2];
    Tensor grad_out = grads[0];

    // 1. branch on the PERSISTED unshard_ flag (never inferred from shape).
    Tensor grad_out_l;
    if (unshard_) {
      const int64_t Tl = grad_out.shape().dims[2] / P;
      grad_out_l =
          grad_out.narrow(2, static_cast<int64_t>(rank_) * Tl, Tl).contiguous();
    } else {
      grad_out_l = grad_out.contiguous(); // already [B, nq, Tl, D]
    }

    // 2. combine (= partition-backward): seq-sharded grad -> head-sharded.
    Tensor grad_out_g = cp::ulysses_combine(pg_, grad_out_l, P); // [B,nq_local,T,hd]
    Tensor grad_out_bf = grad_out_g.contiguous().as_type(Dtype::Bfloat16);

    // 3. fused backward kernel (bf16, packed [dQ|dK|dV]).
    const int64_t qElems = B_ * nq_local_ * T * hd_;
    const int64_t kElems = B_ * kv_local_ * T * hd_;
    const int64_t total = qElems + 2 * kElems;
    TensorOptions bf = out_bf_.opts().with_dtype(Dtype::Bfloat16).with_req_grad(false);
    TensorOptions fp = out_bf_.opts().with_dtype(Dtype::Float32).with_req_grad(false);
    Tensor grad_qkv = Tensor::zeros(Shape({{total}}), bf);
    Tensor D_buf = Tensor::empty(Shape({{B_, nq_local_, T}}), fp);
    Tensor dq_gamma = has_gamma_ ? Tensor::zeros(Shape({{hd_}}), fp) : Tensor();
    Tensor dk_gamma = has_gamma_ ? Tensor::zeros(Shape({{hd_}}), fp) : Tensor();

    OwnTensor::cuda::gqa_fused_flash_attn_backward(
        packed_.data<bfloat16_t>(), cache_.data<float>(),
        has_gamma_ ? q_gamma_.data<float>() : nullptr,
        has_gamma_ ? k_gamma_.data<float>() : nullptr,
        out_bf_.data<bfloat16_t>(), grad_out_bf.data<bfloat16_t>(),
        lse_.data<float>(), q_rstd_.data<float>(), k_rstd_.data<float>(),
        grad_qkv.data<bfloat16_t>(),
        has_gamma_ ? dq_gamma.data<float>() : nullptr,
        has_gamma_ ? dk_gamma.data<float>() : nullptr, D_buf.data<float>(),
        static_cast<int>(B_), static_cast<int>(nq_local_),
        static_cast<int>(kv_local_), static_cast<int>(T), static_cast<int>(hd_),
        cache_seq_len_, /*pos_offset=*/0, eps_, interleaved_, is_causal_);

    // split [dQ|dK|dV] (mirror AttentionBackward.cpp): narrow_view + reshape + fp32.
    Tensor dQg = grad_qkv.narrow_view(0, 0, qElems)
                     .reshape(Shape({{B_, nq_local_, T, hd_}}))
                     .as_type(Dtype::Float32);
    Tensor dKg = grad_qkv.narrow_view(0, qElems, kElems)
                     .reshape(Shape({{B_, kv_local_, T, hd_}}))
                     .as_type(Dtype::Float32);
    Tensor dVg = grad_qkv.narrow_view(0, qElems + kElems, kElems)
                     .reshape(Shape({{B_, kv_local_, T, hd_}}))
                     .as_type(Dtype::Float32);

    // 4. partition (= combine-backward) -> seq-sharded local.
    Tensor dq_l = cp::ulysses_partition(pg_, dQg, P); // [B, nq,     Tl, hd]
    Tensor dk_c = cp::ulysses_partition(pg_, dKg, P); // [B, eff_kv, Tl, hd]
    Tensor dv_c = cp::ulysses_partition(pg_, dVg, P);

    // 5. replication adjoint (no-op if rep==1).
    Tensor dk_l = cp::head_group_reduce(dk_c, nkv_, rep_); // [B, nkv, Tl, hd]
    Tensor dv_l = cp::head_group_reduce(dv_c, nkv_, rep_);

    // 6. gammas: shared [hd] param over a HEAD-partitioned axis -> cross-rank SUM.
    if (has_gamma_) {
      pg_->all_reduce(dq_gamma.data<float>(), dq_gamma.data<float>(),
                      static_cast<size_t>(hd_), Dtype::Float32, op_t::sum,
                      /*sync=*/true);
      pg_->all_reduce(dk_gamma.data<float>(), dk_gamma.data<float>(),
                      static_cast<size_t>(hd_), Dtype::Float32, op_t::sum,
                      /*sync=*/true);
    }

    // 7. return shape = upstream edge shape (branch on PERSISTED pre_sharded_).
    Tensor gq, gk, gv;
    if (pre_sharded_) {
      gq = dq_l;
      gk = dk_l;
      gv = dv_l; // LOCAL, no gather
    } else {
      // PLAIN rank-order gather (skip unloadbalance/de-zigzag) — same carve-out
      // as v1/v2. q by nq heads, k/v by nkv heads.
      gq = cp::ulysses_gather_seq(pg_, dq_l, P);
      gk = cp::ulysses_gather_seq(pg_, dk_l, P);
      gv = cp::ulysses_gather_seq(pg_, dv_l, P);
    }

    if (has_gamma_)
      return {gq, gk, gv, dq_gamma, dk_gamma};
    return {gq, gk, gv};
  }

private:
  Tensor packed_, out_bf_, lse_, q_rstd_, k_rstd_, cache_, q_gamma_, k_gamma_;
  std::shared_ptr<ProcessGroupNCCL> pg_;
  bool is_causal_;
  int world_size_;
  int rank_;
  bool unshard_;
  bool pre_sharded_;
  int64_t B_, nq_local_, kv_local_, hd_;
  int cache_seq_len_;
  float eps_;
  bool interleaved_;
  int64_t nkv_;
  int rep_;
  int64_t nq_;
  bool has_gamma_;
};

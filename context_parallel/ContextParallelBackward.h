#pragma once

#include "autograd/Engine.h"
#include "autograd/Node.h"
#include "autograd/ops_template.h"
#include "core/Tensor.h"
#include "process_group/ProcessGroupNCCL.h"
// #include "tensor/dtensor.h"
#include "context_parallel/LoadBalancer.h"

#include "context_parallel/FusedSDPAOp.h"
#include "context_parallel/RingRotator.h"
#include "context_parallel/SDPAMerger.h"
#include "context_parallel/SDPAOp.h"
#include "context_parallel/RopeDeltas.h"      // 4-delta RoPE positions (additive)
#include "context_parallel/FusedRoPESDPA.h"   // sdpa_fused_backward_rope (gated by CP_FUSED_ROPE)

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <nvtx3/nvToolsExt.h>

using namespace OwnTensor;

// ---------------------------------------------------------------------------
// ContextParallelBackward
//
// Autograd node for the backward pass of context parallel ring attention.
//
// Key corrections vs naive backward:
//   1. Shards the incoming full gradient [B,H,T,D] to local [B,H,T/n,D]
//   2. Applies merger rescaling: each step's grad is weighted by
//      exp(step_lse - merged_lse) to account for the online softmax merge
//   3. Communicates dK/dV back to source ranks via sendrecv
//   4. All-gathers local gradients to reconstruct full [B,H,T,D] gradients
// ---------------------------------------------------------------------------
class ContextParallelBackward : public Node {
public:
  ContextParallelBackward(
      // Saved tensors from forward
      Tensor saved_q,                         // [B, H, T/n, D]
      std::vector<Tensor> saved_k_chunks,     // K chunks per ring step
      std::vector<Tensor> saved_v_chunks,     // V chunks per ring step
      std::vector<bool> saved_causal_flags,   // causal flag per ring step
      std::vector<bool> saved_partial_flags,  // partial flag per ring step
      std::vector<Tensor> saved_lse_per_step, // LSE per ring step [B,H,T/n,1]
      std::vector<Tensor> saved_out_per_step, // OUT per ring step [B,H,T/n,D]
      Tensor merged_lse,                      // final merged LSE [B,H,T/n,1]
      Tensor merged_out,                      // final merged out [B,H,T/n,D]
      // Process group and config
      std::shared_ptr<ProcessGroupNCCL> pg, float attn_scale, bool is_causal,
      int rotator_type, bool load_balance, int world_size,
      int rank, bool unshard = true,
      bool recompute_k = true,
      bool sub_chunk_active = false,
      bool external_balanced = false,
      // RoPE opt-in (additive). When use_rope is false (GPT-2/wpe), this node
      // behaves byte-identically to before: Node(3), 3-grad apply(), gamma
      // members unused. When true, Node(5) and apply() also returns
      // {dq_gamma, dk_gamma} on edges 3,4 (wired in ContextParallel.h).
      bool use_rope = false, Tensor q_gamma = Tensor(),
      Tensor k_gamma = Tensor(), Tensor cos_sin_cache = Tensor(),
      float rope_eps = 1e-6f)
      // 3 outputs (q,k,v), or 5 (+q_gamma,k_gamma) when RoPE + valid gammas.
      : Node((use_rope && q_gamma.is_valid() && k_gamma.is_valid()) ? 5 : 3),
        saved_q_(saved_q), saved_k_chunks_(saved_k_chunks),
        saved_v_chunks_(saved_v_chunks),
        saved_causal_flags_(saved_causal_flags),
        saved_partial_flags_(saved_partial_flags),
        saved_lse_per_step_(saved_lse_per_step),
        saved_out_per_step_(saved_out_per_step), merged_lse_(merged_lse),
        merged_out_(merged_out), pg_(pg), attn_scale_(attn_scale),
        is_causal_(is_causal), rotator_type_(rotator_type),
        load_balance_(load_balance), world_size_(world_size), rank_(rank),
        unshard_(unshard), recompute_k_(recompute_k),
        sub_chunk_active_(sub_chunk_active),
        external_balanced_(external_balanced),
        use_rope_(use_rope), q_gamma_(q_gamma), k_gamma_(k_gamma),
        cos_sin_cache_(cos_sin_cache), rope_eps_(rope_eps),
        has_gamma_(use_rope && q_gamma.is_valid() && k_gamma.is_valid()) {}

  const char *name() const override { return "ContextParallelBackward"; }

  std::vector<Tensor> apply(std::vector<Tensor> &&grads) override {
    if (grads.empty()) {
      throw std::runtime_error(
          "ContextParallelBackward: no gradients provided");
    }

    // Env flag: SKIP_DKV_ROTATION=1 disables the dkv ring rotation under LB.
    // Mimics PyTorch's effective behavior under context_parallel() — its
    // _templated_ring_attention_backward never runs, so dkv stays local.
    // WARNING: experimental. May produce partial gradients missing cross-rank
    // contributions. Verify parity vs the rotated path before trusting.
    const char *_skip_env = std::getenv("SKIP_DKV_ROTATION");
    bool skip_dkv_nccl = (_skip_env && _skip_env[0] == '1');

    // Env flag: USE_PER_STEP_LSE=1 feeds the per-step (out, lse) captured at
    // forward time into sdpa_fused_backward instead of the merged ones. PT's
    // local aten backward uses per-step quantities because each inner
    // aten._scaled_dot_product call has its own autograd backward seeing only
    // that step's outputs. Empirically PT converges to slightly lower loss
    // than our with-rotation+merged path. This flag lets us test whether the
    // per-step LSE choice is the cause.
    const char *_perstep_env = std::getenv("USE_PER_STEP_LSE");
    const bool use_per_step_lse = (_perstep_env && _perstep_env[0] == '1');

    // Env flag: ITER0_ONLY_DKV=1 accumulates dK/dV ONLY from ring iter i==0
    // (i.e. local K/V contribution). For i > 0, dK_step/dV_step (which are
    // for rotated K/V = peer's K/V from this rank's perspective) are dropped
    // entirely. This precisely mimics PT's behavior under context_parallel():
    // PT's local aten backward at iter > 0 produces dK_step for the rotated
    // K, but permute_tensor (non-autograd variant) provides no autograd link
    // back to the source rank's K parameter, so the gradient is silently
    // dropped. dQ accumulation is unchanged (PT accumulates dQ at all iters).
    //
    // Implies SKIP_DKV_ROTATION=1 — no point rotating an accumulator that
    // never receives non-zero updates after iter 0.
    const char *_iter0_env = std::getenv("ITER0_ONLY_DKV");
    const bool iter0_only_dkv = (_iter0_env && _iter0_env[0] == '1');
    if (iter0_only_dkv) {
      skip_dkv_nccl = true;
    }

    // ----- Phase 0: Shard the full gradient to local chunk -----
    Tensor grad_local;
    if (unshard_) {
      Tensor grad_output_full = grads[0].contiguous();
      std::vector<Tensor> grad_chunks =
          grad_output_full.make_shards_inplace_axis(
              static_cast<size_t>(world_size_), 2);
      grad_local = grad_chunks[rank_].contiguous();
    } else {
      grad_local = grads[0].contiguous();
    }

    // ----- Phase 1: Gradient Buffer Init -----
    // Single travelling accumulators for dK/dV (rotated through ring).
    // This matches PyTorch's pipelined dkv_rotater protocol.
    const int seq_dim = 2;
    const int64_t T_local_bwd = saved_q_.shape().dims[seq_dim];

    Tensor grad_q = Tensor::zeros(saved_q_.shape(), saved_q_.opts());
    Tensor grad_key = Tensor::zeros(
        saved_k_chunks_[0].shape(), saved_k_chunks_[0].opts());
    Tensor grad_value = Tensor::zeros(
        saved_v_chunks_[0].shape(), saved_v_chunks_[0].opts());

    // RoPE QK-norm gamma grad accumulators (additive; only used when use_rope_).
    // Replicated-parameter grads: accumulated LOCALLY over ring steps (NO
    // rotation — see RopeGammaAccum.h); the per-rank partial is reduced across
    // CP ranks by the existing model.parameters() all-reduce, not here.
    Tensor grad_q_gamma, grad_k_gamma;
    if (use_rope_) {
      const int64_t hd = saved_q_.shape().dims[3];
      if (q_gamma_.is_valid())
        grad_q_gamma = Tensor::zeros(Shape({{hd}}), saved_q_.opts());
      if (k_gamma_.is_valid())
        grad_k_gamma = Tensor::zeros(Shape({{hd}}), saved_q_.opts());
    }

    // dkv_rotater: pipelined rotation of dK/dV (only for LB path)
    std::unique_ptr<RingRotatorBase> dkv_rotater;
    // Batch accumulators for non-LB path (old Phase 3 approach)
    std::vector<Tensor> grad_k_accum;
    std::vector<Tensor> grad_v_accum;

    // Sub-chunk dispatch (round-robin Q/K/V halving + pipelined dkv_rotater)
    // is driven by the forward's sub_chunk_active flag, not by raw load_balance_.
    // This decouples LB sub-chunking from the legacy "HeadTail kernel applied"
    // semantics, which is what previously caused LB+causal to be disabled.
    bool lb_active_bwd = sub_chunk_active_;
    if (sub_chunk_active_ && (T_local_bwd % 2 != 0)) {
      throw std::runtime_error(
          "ContextParallelBackward: sub_chunk_active requires even T_local");
    }
    if (lb_active_bwd && !skip_dkv_nccl) {
      // The backward dK/dV is a TRAVELLING ACCUMULATOR (see above): a buffer
      // that is rotated around the ring and has the local contribution added
      // at every step. AllGatherRingRotator communicates ONLY on its first
      // exchange_buffers call (it caches a one-shot all_gather snapshot) and
      // every subsequent call is a no-op serving cached slices. That is correct
      // for the read-only FORWARD KV ring, but it CANNOT carry a write-
      // accumulate gradient buffer: per-step dK/dV updates are never re-
      // communicated, so cross-rank gradient contributions from steps > 0 are
      // silently dropped -> wrong gradients -> the model trains worse (observed
      // as a monotonically growing validation-loss gap, AllGather-only).
      // PyTorch sidesteps this identically: _templated_ring_attention_backward
      // FORCES its dkv_rotater to _RotateMethod.ALL_TO_ALL regardless of the
      // configured forward rotate method. Mirror that here -- force a true
      // per-step ring for the gradient accumulator when forward uses AllGather.
      // (rotator_type_: 0=P2P, 1=AlltoAll, 2=AllGather; see create_rotator().)
      dkv_rotater = (rotator_type_ == 2)
          ? std::make_unique<AlltoAllRingRotator>(pg_)
          : create_rotator();
    } else if (!lb_active_bwd) {
      grad_k_accum.resize(world_size_);
      grad_v_accum.resize(world_size_);
    }
    // When skip_dkv_nccl is true under LB, no rotator is created — grad_key /
    // grad_value just accumulate local step contributions without any exchange.

    // Optional kv_rotater for recompute_k mode
    std::unique_ptr<RingRotatorBase> kv_rotater;
    Tensor curr_k, curr_v;
    // Persistent (static) double-buffered send staging for the recompute_k
    // re-rotation, REUSED across all layer-backward calls. The forward already
    // keeps persistent ring buffers (ContextParallel.h Phase 2); the backward
    // recompute path used a fresh Tensor::empty per ring step (and thus per
    // layer), churning/fragmenting the caching pool so the single large logits
    // allocation fails at the frontier even when total memory is lower. Static
    // reuse removes that churn. Grown (never shrunk) to the largest KV seen.
    static thread_local Tensor s_kv_send[2];
    static thread_local int64_t s_kv_send_numel = 0;
    std::shared_ptr<Work> kv_send_work[2] = {nullptr, nullptr};
    if (recompute_k_) {
      kv_rotater = create_rotator();
      curr_k = saved_k_chunks_[0];
      curr_v = saved_v_chunks_[0];
      int64_t need = curr_k.numel() * 2;
      if (s_kv_send_numel < need) {
        for (int s = 0; s < 2; ++s)
          s_kv_send[s] = Tensor::empty(Shape({{need}}), curr_k.opts());
        s_kv_send_numel = need;
      }
    }

    // ---- Compute/comm overlap setup (LB dK/dV travelling accumulator) ----
    // The dK/dV exchange is a serial recv->add->send chain, but the SDPA-backward
    // of step i (independent; uses saved K/V) overlaps the in-flight accumulator
    // transfer from step i-1 when we (a) consume GPU-side and (b) keep the send
    // staging buffer alive across the transfer (persistent double-buffer instead
    // of the previous fresh-per-step alloc that relied on the caching allocator).
    // CP_NO_OVERLAP=1 reverts to the CPU-blocking path for A/B.
    // Value-aware: unset OR "0" => overlap ON; any other value => OFF.
    const char *_cp_no_ovl = std::getenv("CP_NO_OVERLAP");
    bool OVERLAP =
        (_cp_no_ovl == nullptr) || (_cp_no_ovl[0] == '0' && _cp_no_ovl[1] == '\0');
    // Independent BACKWARD overlap gate (for bisecting fwd-vs-bwd races):
    // CP_NO_OVERLAP_BWD=1 forces the backward ring onto the blocking/CPU path
    // even when forward overlap is on.
    const char *_no_bwd = std::getenv("CP_NO_OVERLAP_BWD");
    if (_no_bwd != nullptr && !(_no_bwd[0] == '0' && _no_bwd[1] == '\0'))
        OVERLAP = false;
    {
      static bool _once = false;
      if (!_once) { _once = true;
        fprintf(stderr, "[CP bwd] overlap=%s\n", OVERLAP ? "ON" : "OFF"); }
    }
    cudaStream_t compute_stream = OwnTensor::cuda::getCurrentStream();
    Tensor dkv_send[2];
    std::shared_ptr<Work> dkv_work[2] = {nullptr, nullptr};
    if (OVERLAP && lb_active_bwd && !skip_dkv_nccl) {
      int64_t kv2 = grad_key.numel() * 2;
      for (int s = 0; s < 2; ++s)
        dkv_send[s] = Tensor::empty(Shape({{kv2}}), grad_key.opts());
    }

    // ----- Phase 2: Ring Loop (backward) -----
    for (int i = 0; i < world_size_; ++i) {
      // --- K/V access ---
      Tensor step_k, step_v;
      if (recompute_k_) {
        // Recompute: rotate K/V through ring (same protocol as forward)
        if (i > 0) {
          Tensor next_kv = kv_rotater->next_buffer();
          Tensor kv_flat = next_kv.flatten();
          int64_t k_numel = curr_k.numel();
          curr_k = kv_flat.narrow(0, 0, k_numel).reshape(curr_k.shape());
          curr_v = kv_flat.narrow(0, k_numel, k_numel).reshape(curr_v.shape());
        }
        if (i < (world_size_ - 1)) {
          int64_t k_numel = curr_k.numel();
          size_t k_bytes = static_cast<size_t>(k_numel) * sizeof(float);
          int s = i & 1;
          // Reuse the persistent slot; ensure its prior send (step i-2) drained
          // before overwriting (no-op at world_size<=2 where each slot sends once).
          if (kv_send_work[s]) kv_send_work[s]->streamWait(0);
          Tensor kv_send = s_kv_send[s];
          cudaMemcpyAsync(kv_send.data<float>(), curr_k.data<float>(),
                          k_bytes, cudaMemcpyDeviceToDevice, 0);
          cudaMemcpyAsync(kv_send.data<float>() + k_numel,
                          curr_v.data<float>(), k_bytes,
                          cudaMemcpyDeviceToDevice, 0);
          kv_send_work[s] = kv_rotater->exchange_buffers(kv_send);
        }
        step_k = curr_k;
        step_v = curr_v;
      } else {
        // Save-K path (default): use pre-saved K/V
        // NOTE: Do NOT continue here -- dkv_rotater communication below must
        // happen on every step to avoid NCCL deadlocks across ranks.
        step_k = saved_k_chunks_[i];
        step_v = saved_v_chunks_[i];
      }

      // If this step was skipped in forward (invalid K/V), skip SDPA but
      // still participate in dkv_rotater communication below.
      bool step_skipped = !step_k.is_valid();

      bool use_causal = saved_causal_flags_[i];
      bool use_partial = saved_partial_flags_[i];

      // --- Compute SDPA backward (skip if step was skipped in forward) ---
      Tensor grad_q_step, grad_k_step, grad_v_step;

      if (!step_skipped) {
        Tensor q_bwd = saved_q_;
        Tensor k_bwd = step_k;
        Tensor v_bwd = step_v;
        // Default (merged) path — attn = exp(QK^T - merged_lse) gives the
        // *fractional* weight of this step's K within the full softmax.
        // Per-step path (USE_PER_STEP_LSE=1) — mirrors PT's local aten backward
        // which uses each step's own (out, lse). Per-step lse sums to 1 over
        // only that step's K so it over-weighs the step's contribution; in
        // PT this empirically converges better because each rank only gets
        // iter-0's dK anyway and the over-weight compensates.
        Tensor out_bwd = use_per_step_lse ? saved_out_per_step_[i] : merged_out_;
        Tensor grad_out_bwd = grad_local;
        Tensor lse_bwd = use_per_step_lse ? saved_lse_per_step_[i] : merged_lse_;

        if (lb_active_bwd && i > 0) {
          if (i <= rank_) {
            // Past with LB: full Q, first half K/V, full out/grad_out/lse.
            // sdpa_fused_backward reads per-tensor B/M/H strides from K/V,
            // so strided half-views can be passed directly — no copy needed.
            std::vector<Tensor> k_halves =
                step_k.make_shards_inplace_axis(2, seq_dim);
            std::vector<Tensor> v_halves =
                step_v.make_shards_inplace_axis(2, seq_dim);
            k_bwd = k_halves[0];
            v_bwd = v_halves[0];
          } else {
            // Future with LB: 2nd half of Q, out, grad_out, lse; full K/V.
            // Q/out/grad_out: backward wrapper reads strides from the tensor
            // (FusedSDPAOp.h:198-208), so strided half-views are safe.
            // lse: backward wrapper hard-codes lse strides as
            // (H * T_q, T_q) at FusedSDPAOp.h:209 — those assume a freshly
            // allocated contiguous LSE of the kernel's T_q. A half-view of
            // merged_lse_ keeps the parent's stride pattern, which would
            // miscompute LSE_bh offsets. Until the wrapper reads LSE strides
            // from the tensor, lse_bwd must remain materialized.
            std::vector<Tensor> q_halves =
                saved_q_.make_shards_inplace_axis(2, seq_dim);
            q_bwd = q_halves[1];

            std::vector<Tensor> grad_halves =
                grad_local.make_shards_inplace_axis(2, seq_dim);
            grad_out_bwd = grad_halves[1];

            // Under per-step mode, saved_out_per_step_[i] / saved_lse_per_step_[i]
            // are already the half-shaped tensors produced when the forward
            // SDPA ran with half-Q. No additional halving needed and the
            // wrapper's hard-coded lse stride (H * T_q, T_q) with T_q = T/2
            // matches the freshly-allocated per-step lse layout.
            if (!use_per_step_lse) {
              std::vector<Tensor> out_halves =
                  merged_out_.make_shards_inplace_axis(2, seq_dim);
              out_bwd = out_halves[1];

              std::vector<Tensor> lse_halves =
                  merged_lse_.make_shards_inplace_axis(2, seq_dim);
              lse_bwd = lse_halves[1].contiguous();
            }
          }
        }

        int q_off = 0, k_off = 0;
        if (!lb_active_bwd && is_causal_) {
          int source_rank =
              ((rank_ - i) % world_size_ + world_size_) % world_size_;
          q_off = rank_ * static_cast<int>(T_local_bwd);
          k_off = source_rank * static_cast<int>(T_local_bwd);
        }

        if (use_rope_) {
          // RoPE path (additive): same 4-delta derivation as the forward ring
          // loop (must match per step). Returns dQ/dK/dV + per-call gamma
          // partials, which accumulate LOCALLY (no rotation).
          OwnTensor::cp::SubChunk sc =
              (i == 0) ? OwnTensor::cp::SubChunk::Full
                       : (lb_active_bwd
                              ? (i <= rank_ ? OwnTensor::cp::SubChunk::KHeadHalf
                                            : OwnTensor::cp::SubChunk::QTailHalf)
                              : OwnTensor::cp::SubChunk::Full);
          OwnTensor::cp::SdpaDeltas dl = OwnTensor::cp::compute_deltas(
              rank_, i, world_size_, static_cast<int>(T_local_bwd),
              /*lb=*/lb_active_bwd, sc);
          SDPARoPEBackwardResult rg = sdpa_fused_backward_rope(
              q_bwd, k_bwd, v_bwd, grad_out_bwd, out_bwd, lse_bwd,
              use_causal, attn_scale_, dl.q.d0, dl.q.d1, dl.k.d0, dl.k.d1,
              cos_sin_cache_, q_gamma_, k_gamma_, rope_eps_);
          grad_q_step = rg.dQ;
          grad_k_step = rg.dK;
          grad_v_step = rg.dV;
          if (grad_q_gamma.is_valid())
            grad_q_gamma = grad_q_gamma + rg.dq_gamma;  // local sum, no rotation
          if (grad_k_gamma.is_valid())
            grad_k_gamma = grad_k_gamma + rg.dk_gamma;  // local sum, no rotation
        } else {
          std::vector<Tensor> step_grads = sdpa_fused_backward(
              q_bwd, k_bwd, v_bwd, grad_out_bwd, out_bwd, lse_bwd,
              use_causal, attn_scale_, q_off, k_off);

          grad_q_step = step_grads[0];
          grad_k_step = step_grads[1];
          grad_v_step = step_grads[2];
        }

        // --- Per-step parity probe (gated by DUMP_CP_STEPS=1) ---
        // Dumps grad_q_step/grad_k_step/grad_v_step for each ring step,
        // tagged with rank, layer-backward call index, and step index i.
        // Used to localize which LB sub-chunk step (i=0 causal-full,
        // 0<i<=rank head-half, i>rank partial) produces divergent grads.
        {
          static std::atomic<int> _step_call_idx{0};
          if (i == 0) _step_call_idx.fetch_add(0);  // no-op, marker
          const char *env = std::getenv("DUMP_CP_STEPS");
          if (env && env[0] == '1') {
            auto dump16 = [](std::ofstream &f, const char *label,
                             const Tensor &t) {
              if (!t.is_valid()) {
                f << label << " <invalid>\n";
                return;
              }
              Tensor host = t.to_cpu();
              const float *p = host.data<float>();
              const auto &d = t.shape().dims;
              f << label << " shape=[";
              for (size_t k = 0; k < d.size(); ++k) {
                f << d[k] << (k + 1 == d.size() ? "" : ",");
              }
              f << "]\n";
              // Slice 1: first16 = position [0, 0, 0, 0..15] = chunk_0 head row 0
              int64_t n0 = std::min<int64_t>(16, t.numel());
              f << "  " << label << " first16 (head/chunk_0): [";
              for (int64_t k = 0; k < n0; ++k) {
                f << p[k] << (k == n0 - 1 ? "" : ", ");
              }
              f << "]\n";
              // 4 sample slices covering both heads (H=0, H=1) and both
              // chunks (chunk_0 head, chunk_3 tail) for 4D [B,H,T,D].
              if (d.size() == 4 && d[2] >= 2 && d[1] >= 1) {
                int64_t T_seq = d[2];
                int64_t Dd = d[3];
                int64_t H_stride = T_seq * Dd;
                auto dump_slice = [&](const char *sublabel, int64_t off) {
                  int64_t n1 = std::min<int64_t>(16, t.numel() - off);
                  f << "  " << label << " " << sublabel << " (offset=" << off
                    << "): [";
                  for (int64_t k = 0; k < n1; ++k) {
                    f << p[off + k] << (k == n1 - 1 ? "" : ", ");
                  }
                  f << "]\n";
                };
                dump_slice("H0_chunk3_start", (T_seq / 2) * Dd);
                if (d[1] >= 2) {
                  dump_slice("H1_chunk0_start", H_stride);
                  dump_slice("H1_chunk3_start", H_stride + (T_seq / 2) * Dd);
                }
              }
            };
            // call_idx_for_rank = how many times THIS rank's apply was called
            // before this one; combined with i below to get unique tag.
            static thread_local int _layer_call_idx = 0;
            int my_layer = _layer_call_idx;
            if (i == 0) {
              // bump on first step of each backward call
              _layer_call_idx++;
            }
            std::string path = "step_bw_rank" + std::to_string(rank_) + ".md";
            std::ofstream df(path, std::ios::app);
            df << "\n## layer_call=" << my_layer << " step_i=" << i
               << " use_causal=" << use_causal
               << " use_partial=" << use_partial
               << " src_rank="
               << (((rank_ - i) % world_size_ + world_size_) % world_size_)
               << "\n";
            dump16(df, "grad_q_step", grad_q_step);
            dump16(df, "grad_k_step", grad_k_step);
            dump16(df, "grad_v_step", grad_v_step);
          }
        }

        // --- DUMP_CP_DEEP: save full tensors for deep parity audit ---
        // Saves grad_q (before), grad_q_step, plus partial-accumulation
        // intermediates (gq_1st_clone, gq_2nd_clone, gq_2nd_plus_step) so a
        // Python diff can pinpoint the exact op where C++ diverges from PT.
        const char *deep_env = std::getenv("DUMP_CP_DEEP");
        bool deep_dump = (deep_env && deep_env[0] == '1');
        auto save_full = [&](const char *label, const Tensor &t) {
          if (!deep_dump || !t.is_valid()) return;
          std::string path = "/tmp/cp_bwd_test/deep/cpp_" + std::string(label) +
                             "_step" + std::to_string(i) + "_rank" +
                             std::to_string(rank_) + ".bin";
          Tensor host = t.to_cpu();
          std::ofstream f(path, std::ios::binary);
          f.write(reinterpret_cast<const char *>(host.data<float>()),
                  host.numel() * sizeof(float));
        };
        save_full("grad_q_before", grad_q);
        save_full("grad_q_step", grad_q_step);

        // --- Accumulate dQ (local, never rotated) ---
        if (lb_active_bwd && i > rank_) {
          int64_t half_T = T_local_bwd / 2;
          Tensor gq_1st = grad_q.narrow_view(seq_dim, 0, half_T);
          Tensor gq_2nd = grad_q.narrow_view(seq_dim, half_T, half_T);
          Tensor gq_1st_clone = gq_1st.clone();
          Tensor gq_2nd_clone = gq_2nd.clone();
          Tensor gq_2nd_plus_step = gq_2nd_clone + grad_q_step;
          save_full("gq_1st_clone", gq_1st_clone);
          save_full("gq_2nd_clone", gq_2nd_clone);
          save_full("gq_2nd_plus_step", gq_2nd_plus_step);
          grad_q = Tensor::cat({gq_1st_clone, gq_2nd_plus_step}, seq_dim);
        } else if (grad_q_step.is_valid()) {
          grad_q = grad_q + grad_q_step;
        }
        save_full("grad_q_after", grad_q);
      }

      // --- dK/dV accumulation ---
      if (lb_active_bwd) {
        // LB path: pipelined rotation via dkv_rotater (unless skip_dkv_nccl)
        if (i == 0) {
          if (grad_k_step.is_valid()) {
            grad_key = grad_key + grad_k_step;
          }
          if (grad_v_step.is_valid()) {
            grad_value = grad_value + grad_v_step;
          }
        } else if (iter0_only_dkv) {
          // PT mimicry: drop iter > 0 step contributions entirely. Their
          // grad_k_step / grad_v_step are for rotated K/V (= peer's K/V),
          // and in PT those gradients have no autograd link back so they
          // are silently dropped. We do the same here. grad_key/grad_value
          // already hold this rank's iter-0 contribution; nothing to add.
        } else {
          if (!skip_dkv_nccl) {
            int64_t k_numel = grad_key.numel();
            // GPU-side consume: SDPA-backward(i) above already overlapped the
            // accumulator transfer posted at step i-1; only wait GPU-side here.
            Tensor next_grad_kv = OVERLAP
                ? dkv_rotater->next_buffer_streamordered(compute_stream)
                : dkv_rotater->next_buffer();
            Tensor gkv_flat = next_grad_kv.flatten();
            grad_key =
                gkv_flat.narrow(0, 0, k_numel).reshape(grad_key.shape());
            grad_value =
                gkv_flat.narrow(0, k_numel, k_numel).reshape(grad_value.shape());
          }
          // When skip_dkv_nccl: grad_key / grad_value carry forward from the
          // previous iter's local accumulation — no overwrite from peers.

          if (i <= rank_ && grad_k_step.is_valid()) {
            int64_t half_T = T_local_bwd / 2;
            Tensor zeros_half_k = Tensor::zeros(grad_k_step.shape(), grad_k_step.opts());
            Tensor gk_padded = Tensor::cat({grad_k_step, zeros_half_k}, seq_dim);
            Tensor zeros_half_v = Tensor::zeros(grad_v_step.shape(), grad_v_step.opts());
            Tensor gv_padded = Tensor::cat({grad_v_step, zeros_half_v}, seq_dim);
            grad_key = grad_key + gk_padded;
            grad_value = grad_value + gv_padded;
          } else {
            if (grad_k_step.is_valid()) {
              grad_key = grad_key + grad_k_step;
            }
            if (grad_v_step.is_valid()) {
              grad_value = grad_value + grad_v_step;
            }
          }
        }

        // Send current grad_key/grad_value to next rank (skipped under
        // SKIP_DKV_ROTATION; in that mode each rank keeps its own local
        // accumulator with no cross-rank exchange)
        if (!skip_dkv_nccl) {
          int64_t k_numel_send = grad_key.numel();
          size_t k_bytes_send =
              static_cast<size_t>(k_numel_send) * sizeof(float);
          int s = i & 1;
          // Persistent double-buffered send staging when overlapping (was
          // fresh-per-step, relying on the caching allocator to keep the buffer
          // alive while the async send drained — fragile under true overlap).
          // Under CP_NO_OVERLAP: fresh alloc, exactly the original behavior.
          // (Tensor assignment is shallow/ref-counted, so gks shares dkv_send[s].)
          Tensor gks = OVERLAP
              ? dkv_send[s]
              : Tensor::empty(Shape({{k_numel_send * 2}}), grad_key.opts());
          cudaStream_t pack_stream = OVERLAP ? compute_stream : 0;
          // [#1] guard: don't repack slot s until step i-2's send drained.
          if (OVERLAP && dkv_work[s]) dkv_work[s]->streamWait(compute_stream);
          cudaMemcpyAsync(gks.data<float>(), grad_key.data<float>(),
                          k_bytes_send, cudaMemcpyDeviceToDevice, pack_stream);
          cudaMemcpyAsync(gks.data<float>() + k_numel_send,
                          grad_value.data<float>(), k_bytes_send,
                          cudaMemcpyDeviceToDevice, pack_stream);
          nvtxRangePushA("CP.bwd.LB.ring.exchange_buffers");
          // Explicit overlap flag + pack (compute) stream; the rotator owns the
          // persistent per-slot pack event (no per-call create/destroy UB).
          std::shared_ptr<Work> w =
              dkv_rotater->exchange_buffers(gks, OVERLAP, compute_stream);
          if (OVERLAP) dkv_work[s] = w;
          nvtxRangePop();
        }
      } else {
        // Non-LB path: batch accumulate per step, single sendrecv at end.
        // Under ITER0_ONLY_DKV, only iter 0's contribution is kept — peer-K
        // gradients from i > 0 are dropped (mimics PT's autograd behavior).
        if (!step_skipped && grad_k_step.is_valid() &&
            !(iter0_only_dkv && i > 0)) {
          grad_k_accum[i] = grad_k_step;
          grad_v_accum[i] = grad_v_step;
        }
      }
    }

    // --- Final: collect dK/dV from ring ---
    if (lb_active_bwd && !skip_dkv_nccl) {
      // LB: receive completed travelling accumulator
      int64_t k_numel = grad_key.numel();
      nvtxRangePushA("CP.bwd.LB.post_loop.next_buffer");
      Tensor final_grad_kv = OVERLAP
          ? dkv_rotater->next_buffer_streamordered(compute_stream)
          : dkv_rotater->next_buffer();
      nvtxRangePop();
      Tensor final_flat = final_grad_kv.flatten();
      // MUST .clone(): final_grad_kv is a view into dkv_rotater's recv buffer,
      // and dkv_rotater is a LOCAL unique_ptr destroyed when this function
      // returns. grad_key/grad_value are returned to the autograd engine and
      // consumed LATER by the downstream c_attn (Linear) backward. Without the
      // clone they dangle into the freed rotator buffer; under no-overlap the
      // CPU-blocking timing + caching allocator hide it, but under overlap the
      // CPU races ahead and the freed buffer can be reused before consumption ->
      // rare corruption that compounds into the ~step-140 grad-norm explosion.
      grad_key =
          final_flat.narrow(0, 0, k_numel).reshape(grad_key.shape()).clone();
      grad_value =
          final_flat.narrow(0, k_numel, k_numel).reshape(grad_value.shape()).clone();
    } else if (lb_active_bwd && skip_dkv_nccl) {
      // SKIP_DKV_ROTATION path: grad_key / grad_value already hold this
      // rank's locally accumulated partials. No cross-rank collection.
    } else {
      // Non-LB: step 0 is local, steps 1..N-1 exchanged via packed sendrecv.
      // grad_key/grad_value (initialized to zeros at lines 88-92) accumulate results.

      // Step 0: local contribution (our own K chunk)
      if (grad_k_accum[0].is_valid()) {
        grad_key = grad_key + grad_k_accum[0];
        grad_value = grad_value + grad_v_accum[0];
      }

      // Steps 1..N-1: pack K+V, single sendrecv per step
      for (int i = 1; i < world_size_; ++i) {
        int source_rank = ((rank_ - i) % world_size_ + world_size_) % world_size_;
        int dest_rank = ((rank_ + i) % world_size_ + world_size_) % world_size_;

        int64_t k_numel = saved_k_chunks_[0].numel();
        size_t k_bytes = static_cast<size_t>(k_numel) * sizeof(float);

        // Pack K+V into one send buffer
        Tensor send_buf = Tensor::zeros(Shape({{k_numel * 2}}),
                                        saved_k_chunks_[0].opts());
        if (grad_k_accum[i].is_valid()) {
          cudaMemcpyAsync(send_buf.data<float>(), grad_k_accum[i].data<float>(),
                          k_bytes, cudaMemcpyDeviceToDevice, 0);
          cudaMemcpyAsync(send_buf.data<float>() + k_numel,
                          grad_v_accum[i].data<float>(),
                          k_bytes, cudaMemcpyDeviceToDevice, 0);
        }

        Tensor recv_buf = Tensor::empty(Shape({{k_numel * 2}}),
                                        saved_k_chunks_[0].opts());

        // Single packed sendrecv
        nvtxRangePushA("CP.bwd.nonLB.sendrecv");
        pg_->sendrecv(send_buf.data<float>(), recv_buf.data<float>(),
                      source_rank, dest_rank,
                      static_cast<size_t>(k_numel * 2),
                      saved_k_chunks_[0].dtype(), true);
        nvtxRangePop();

        // Unpack and accumulate
        Tensor recv_k = recv_buf.narrow(0, 0, k_numel).reshape(
            saved_k_chunks_[0].shape());
        Tensor recv_v = recv_buf.narrow(0, k_numel, k_numel).reshape(
            saved_v_chunks_[0].shape());

        grad_key = grad_key + recv_k;
        grad_value = grad_value + recv_v;
      }
    }

    // [#2] Drain pending dK/dV ring sends before dkv_send[2] frees at return.
    // Both slots drained (in-loop guard covers reuse; at most 2 in flight here).
    // CPU-wait (not GPU-side streamWait): the freed dkv_send/recv blocks can be
    // reused by allocations whose NCCL writes come from a different stream than
    // the compute stream under full fwd+bwd overlap, so a compute-stream-only
    // drain leaves a reuse-vs-in-flight-send race. See the matching Context
    // Parallel.h post-loop comment for the full rationale.
    if (OVERLAP) {
      for (int s = 0; s < 2; ++s)
        if (dkv_work[s]) dkv_work[s]->streamWait(compute_stream);
    }

    // Cast back to original dtype
    if (grad_q.dtype() != saved_q_.dtype()) {
      grad_q = grad_q.as_type(saved_q_.dtype());
    }
    if (grad_key.dtype() != saved_k_chunks_[0].dtype()) {
      grad_key = grad_key.as_type(saved_k_chunks_[0].dtype());
      grad_value = grad_value.as_type(saved_v_chunks_[0].dtype());
    }

    // Optional backward parity probe (gated by DUMP_CP_OUT=1).
    // Appends dY + per-rank local dQ/dK/dV first16 for each backward call.
    // Block 0's backward = LAST call (highest call_idx) in step 0.
    // Placed BEFORE the unshard branch so it fires on both paths.
    {
      static std::atomic<int> _bw_call_idx{0};
      int call_idx = _bw_call_idx.fetch_add(1);
      const char *env = std::getenv("DUMP_CP_OUT");
      if (call_idx < 24 && env && env[0] == '1') {
        auto dump16 = [](std::ofstream &f, const char *label,
                         const Tensor &t) {
          Tensor host = t.to_cpu();
          const float *p = host.data<float>();
          int64_t n = std::min<int64_t>(16, t.numel());
          f << label << " first16: [";
          for (int64_t i = 0; i < n; ++i) {
            f << p[i] << (i == n - 1 ? "" : ", ");
          }
          f << "]\n";
        };

        std::string path = "block_bw_rank" + std::to_string(rank_) + ".md";
        std::ofstream df(path, std::ios::app);
        df << "\n## call_idx=" << call_idx << "\n";
        const auto &gd = grad_local.shape().dims;
        df << "grad_local shape=[" << gd[0] << "," << gd[1] << "," << gd[2]
           << "," << gd[3] << "]\n";
        dump16(df, "dY", grad_local);
        dump16(df, "dQ_local", grad_q);
        dump16(df, "dK_local", grad_key);
        dump16(df, "dV_local", grad_value);
      }
    }

    // ----- Phase 4: Unshard gradients -----
    // Gamma grads are [hd] replicated-param partials (NOT sequence-sharded), so
    // they are returned as-is in both unshard modes; cross-rank reduction is the
    // existing model.parameters() all-reduce, not the per-shard all-gather.
    if (!unshard_) {
      if (has_gamma_)
        return {grad_q, grad_key, grad_value, grad_q_gamma, grad_k_gamma};
      return {grad_q, grad_key, grad_value};
    }

    nvtxRangePushA("CP.bwd.unshard.all_gather.grad_q");
    Tensor full_grad_q = all_gather_along_seq(grad_q);
    nvtxRangePop();
    nvtxRangePushA("CP.bwd.unshard.all_gather.grad_k");
    Tensor full_grad_k = all_gather_along_seq(grad_key);
    nvtxRangePop();
    nvtxRangePushA("CP.bwd.unshard.all_gather.grad_v");
    Tensor full_grad_v = all_gather_along_seq(grad_value);
    nvtxRangePop();

    // Mirror the forward: only unloadbalance the all-gathered grads when CP
    // itself applied HeadTail (i.e. external pre-shard did NOT). If
    // external_balanced_ is true, grads are left in HeadTail layout to match
    // the caller's expectation (parameter grads already trained that way).
    if (load_balance_ && !external_balanced_) {
      HeadTail lb;
      lb.set_world_size(world_size_);
      lb.set_chunk_dim(2);
      lb.set_stream(0);
      lb.unloadbalance(full_grad_q);
      lb.unloadbalance(full_grad_k);
      lb.unloadbalance(full_grad_v);
    }

    if (has_gamma_)
      return {full_grad_q, full_grad_k, full_grad_v, grad_q_gamma, grad_k_gamma};
    return {full_grad_q, full_grad_k, full_grad_v};
  }

  void release_saved_variables() override {
    saved_q_ = Tensor();
    saved_k_chunks_.clear();
    saved_v_chunks_.clear();
    saved_lse_per_step_.clear();
    saved_out_per_step_.clear();
    merged_lse_ = Tensor();
    merged_out_ = Tensor();
    q_gamma_ = Tensor();
    k_gamma_ = Tensor();
    cos_sin_cache_ = Tensor();
  }

private:
  Tensor saved_q_;
  std::vector<Tensor> saved_k_chunks_;
  std::vector<Tensor> saved_v_chunks_;
  std::vector<bool> saved_causal_flags_;
  std::vector<bool> saved_partial_flags_;
  std::vector<Tensor> saved_lse_per_step_;
  std::vector<Tensor> saved_out_per_step_;
  Tensor merged_lse_;
  Tensor merged_out_;

  std::shared_ptr<ProcessGroupNCCL> pg_;
  float attn_scale_;
  bool is_causal_;
  int rotator_type_;
  bool load_balance_;
  int world_size_;
  int rank_;
  bool unshard_;
  bool recompute_k_;
  bool sub_chunk_active_;
  bool external_balanced_;

  // RoPE opt-in (additive; default off => unused, node behaves as 3-output).
  bool use_rope_ = false;
  Tensor q_gamma_;        // [hd] QK-norm scale (or invalid => skip)
  Tensor k_gamma_;        // [hd] QK-norm scale (or invalid => skip)
  Tensor cos_sin_cache_;  // [cache_len, hd] cos/sin cache
  float rope_eps_ = 1e-6f;
  bool has_gamma_ = false;  // use_rope_ && both gammas valid => node has 5 outputs

  std::unique_ptr<RingRotatorBase> create_rotator() const {
    switch (rotator_type_) {
    case 0:
      return std::make_unique<P2PRingRotator>(pg_);
    case 1:
      return std::make_unique<AlltoAllRingRotator>(pg_);
    case 2:
      return std::make_unique<AllGatherRingRotator>(pg_);
    default:
      throw std::runtime_error("Unknown rotator type");
    }
  }

  Tensor all_gather_along_seq(Tensor &local) {
    size_t local_count = static_cast<size_t>(local.numel());
    size_t total_count = local_count * static_cast<size_t>(world_size_);

    Shape flat_shape({{static_cast<int64_t>(total_count)}});
    Tensor gathered_flat = Tensor::empty(flat_shape, local.opts());

    pg_->all_gather(local.data<float>(), gathered_flat.data<float>(),
                    local_count, local.dtype(), true);

    int64_t B = local.shape().dims[0];
    int64_t H = local.shape().dims[1];
    int64_t T_local = local.shape().dims[2];
    int64_t D = local.shape().dims[3];
    int64_t T_full = T_local * world_size_;

    Shape full_shape({{B, H, T_full, D}});
    Tensor full = Tensor::empty(full_shape, local.opts());

    // Per-(b,h)-slice copy with correct strides (see ContextParallel.h)
    size_t slice_bytes = static_cast<size_t>(T_local * D) * sizeof(float);
    for (int r = 0; r < world_size_; ++r) {
      for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
          float *src = gathered_flat.data<float>() + r * (B * H * T_local * D) +
                       b * (H * T_local * D) + h * (T_local * D);
          float *dst = full.data<float>() + b * (H * T_full * D) +
                       h * (T_full * D) + r * (T_local * D);
          cudaMemcpyAsync(dst, src, slice_bytes, cudaMemcpyDeviceToDevice, 0);
        }
      }
    }
    cudaStreamSynchronize(0);

    return full;
  }
};

#pragma once

// ---------------------------------------------------------------------------
// FusedSDPAOp.h  (sm89 / Ada Lovelace build)
//
// SAME interface as the generic FusedSDPAOp.h, but the hot path calls the
// CP-aware Ada PTX-MMA attention kernels:
//
//   forward  -> OwnTensor::cp::cuda::mem_efficient_attn_forward_tc_sm89_strided
//               (AttentionForward_sm89.cu)
//   backward -> OwnTensor::cp::cuda::mem_efficient_attn_backward_sm89_strided
//               (AttentionBackward_sm89.cu)
//
// Unlike the bare arch/*_sm89.cu kernels, these CP variants accept T_q != T_k
// and non-zero q_offset / k_offset, so they handle CP ring-attention steps with
// global-position causal masking — no fallback / throw needed.
//
// There is NO runtime arch gate here: drop this file in (rename to
// FusedSDPAOp.h) ONLY on an sm89 (Ada: A6000 Ada, RTX 4090, L40S) build.
//
// Interface:
//
//   SDPAResult sdpa_fused_forward(
//       Tensor& q,          // [B, H, T_q, D]
//       Tensor& k,          // [B, H, T_k, D]
//       Tensor& v,          // [B, H, T_k, D]
//       bool    is_causal,
//       float   scale,
//       int     q_offset = 0,   // global sequence start of Q chunk (CP use)
//       int     k_offset = 0);  // global sequence start of K chunk (CP use)
//
// LSE shape: [B, H, T_q, 1]  -- matches SDPAMerger::step() expectations.
// ---------------------------------------------------------------------------

#include "core/Tensor.h"
#include "ops/helpers/AttentionKernels.h"
#include "context_parallel/AttentionBackward.h"
#include "context_parallel/AttentionForward.h"
#include "context_parallel/AttentionBackward_sm89.h"
#include "context_parallel/AttentionForward_sm89.h"
#include "context_parallel/FusedSDPABackwardKernel.h"
#include "context_parallel/FusedSDPAKernel.h"
#include "context_parallel/SDPAOp.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

using namespace OwnTensor;

// ---------------------------------------------------------------------------
// sdpa_fused_forward
// ---------------------------------------------------------------------------
inline SDPAResult sdpa_fused_forward(Tensor &q, Tensor &k, Tensor &v,
                                     bool is_causal, float scale,
                                     int q_offset = 0, int k_offset = 0) {
  // --- input validation ---------------------------------------------------
  if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4) {
    throw std::runtime_error(
        "sdpa_fused_forward: Q, K, V must be 4-D tensors [B, H, T, D]");
  }
  if (q.dtype() != Dtype::Float32) {
    throw std::runtime_error("sdpa_fused_forward: only Float32 is supported");
  }

  const int64_t B = q.shape().dims[0];
  const int64_t H = q.shape().dims[1];
  const int64_t T_q = q.shape().dims[2];
  const int64_t D = q.shape().dims[3];
  const int64_t T_k = k.shape().dims[2];

  if (k.shape().dims[0] != B || k.shape().dims[1] != H ||
      k.shape().dims[3] != D || v.shape().dims[0] != B ||
      v.shape().dims[1] != H || v.shape().dims[2] != T_k ||
      v.shape().dims[3] != D) {
    throw std::runtime_error("sdpa_fused_forward: K/V shape mismatch with Q");
  }
  if (D > 256) {
    throw std::runtime_error(
        "sdpa_fused_forward: head dim > 256 not supported");
  }

  // --- allocate output tensors on the same device as Q -------------------
  TensorOptions base_opts = q.opts().with_req_grad(false);

  Shape out_shape({{B, H, T_q, D}});
  Tensor out = Tensor::empty(out_shape, base_opts);

  // LSE shape: [B, H, T_q, 1]  (keepdim convention used by SDPAMerger)
  Shape lse_shape({{B, H, T_q, 1}});
  Tensor lse = Tensor::empty(lse_shape, base_opts);

  // --- get raw float* pointers (device memory) ---------------------------
  const float *Q_ptr = q.data<float>();
  const float *K_ptr = k.data<float>();
  const float *V_ptr = v.data<float>();
  float *O_ptr = out.data<float>();
  float *LSE_ptr = lse.data<float>();

  // Extract strides from Q/K/V so the kernel can read non-contiguous inputs
  // (e.g. transposed views) without an upstream contiguous copy.
  const auto& qs = q.stride().strides;
  const auto& ks = k.stride().strides;
  const auto& vs = v.stride().strides;
  // [B, H, T, D] -> strides[0]=B, strides[1]=H, strides[2]=T, strides[3]=D(=1).
  const int64_t q_sB = qs[0], q_sH = qs[1], q_sM = qs[2];
  const int64_t k_sB = ks[0], k_sH = ks[1], k_sM = ks[2];
  const int64_t v_sB = vs[0], v_sH = vs[1], v_sM = vs[2];
  const int64_t o_sB = H * T_q * D, o_sH = T_q * D, o_sM = D;
  const int64_t lse_sB = H * T_q,  lse_sH = T_q;

  // Ada PTX-MMA CP forward (handles T_q != T_k and non-zero offsets).
  // NOTE: the generic scalar-fp32 (ATTN_FP32) fallback is intentionally NOT
  // wired here — this is a pure-sm89 build that does not link the generic CP
  // AttentionForward.cu. Use the generic FusedSDPAOp.h for that parity path.
  OwnTensor::cp::cuda::mem_efficient_attn_forward_tc_sm89_strided(
    Q_ptr, q_sB, q_sM, q_sH,
    K_ptr, k_sB, k_sM, k_sH,
    V_ptr, v_sB, v_sM, v_sH,
    O_ptr, o_sB, o_sM, o_sH,
    LSE_ptr, lse_sB, lse_sH,
    B, H, T_q, T_k, q_offset, k_offset, D,
    is_causal, 0.0f, nullptr);

  return SDPAResult{out, lse};
}

// ---------------------------------------------------------------------------
// sdpa_fused_backward
//
// Fused FlashAttention backward (Ada-tuned, CP-aware).  Replaces
// sdpa_backward_op_manual() for the hot path in ContextParallelBackward.
//
// Inputs:
//   q, k, v        -- saved forward inputs      [B, H, T_q/k, D]
//   grad_out       -- incoming gradient dO       [B, H, T_q, D]
//   out            -- merged forward output O    [B, H, T_q, D]
//   merged_lse     -- merged log-sum-exp         [B, H, T_q, 1]
// ---------------------------------------------------------------------------
inline std::vector<Tensor>
sdpa_fused_backward(const Tensor &q, const Tensor &k, const Tensor &v,
                    const Tensor &grad_out, const Tensor &out,
                    const Tensor &merged_lse, bool is_causal, float scale,
                    int q_offset = 0, int k_offset = 0) {
  const int64_t B = q.shape().dims[0];
  const int64_t H = q.shape().dims[1];
  const int64_t T_q = q.shape().dims[2];
  const int64_t D = q.shape().dims[3];
  const int64_t T_k = k.shape().dims[2];
  const int BH = static_cast<int>(B * H);

  if (D > 256) {
    throw std::runtime_error(
        "sdpa_fused_backward: head dim > 256 not supported");
  }

  TensorOptions base_opts = q.opts().with_req_grad(false);

  Tensor dQ = Tensor::empty(q.shape(), base_opts);
  Tensor dK = Tensor::empty(k.shape(), base_opts);
  Tensor dV = Tensor::empty(v.shape(), base_opts);

  // D_buf [BH, T_q] -- scratch written by precompute, read by dK/dV phase.
  Shape d_shape({{static_cast<int64_t>(BH), T_q}});
  Tensor D_buf = Tensor::empty(d_shape, base_opts);

  // Extract input strides; outputs (dQ, dK, dV) are freshly allocated above
  // and contiguous so their strides are the contiguous ones.
  const auto& qs   = q.stride().strides;
  const auto& ks   = k.stride().strides;
  const auto& vs   = v.stride().strides;
  const auto& os   = out.stride().strides;
  const auto& dos_ = grad_out.stride().strides;
  // [B, H, T, D] -> strides[0]=B, strides[1]=H, strides[2]=T, strides[3]=D(=1).
  const int64_t q_sB = qs[0],  q_sH = qs[1],  q_sM = qs[2];
  const int64_t k_sB = ks[0],  k_sH = ks[1],  k_sM = ks[2];
  const int64_t v_sB = vs[0],  v_sH = vs[1],  v_sM = vs[2];
  const int64_t o_sB = os[0],  o_sH = os[1],  o_sM = os[2];
  const int64_t do_sB = dos_[0], do_sH = dos_[1], do_sM = dos_[2];
  const int64_t lse_sB = H * T_q,  lse_sH = T_q;
  const int64_t dq_sB = H * T_q * D, dq_sH = T_q * D, dq_sM = D;
  const int64_t dk_sB = H * T_k * D, dk_sH = T_k * D, dk_sM = D;
  const int64_t dv_sB = H * T_k * D, dv_sH = T_k * D, dv_sM = D;

  OwnTensor::cp::cuda::mem_efficient_attn_backward_sm89_strided(
    q.data<float>(), q_sB, q_sM, q_sH,
    k.data<float>(), k_sB, k_sM, k_sH,
    v.data<float>(), v_sB, v_sM, v_sH,
    out.data<float>(),       o_sB,  o_sM,  o_sH,
    grad_out.data<float>(),  do_sB, do_sM, do_sH,
    merged_lse.data<float>(), lse_sB, lse_sH,
    dQ.data<float>(), dq_sB, dq_sM, dq_sH,
    dK.data<float>(), dk_sB, dk_sM, dk_sH,
    dV.data<float>(), dv_sB, dv_sM, dv_sH,
    D_buf.data<float>(),
    B, H, T_q, T_k, q_offset, k_offset, D,
    is_causal);

  return {dQ, dK, dV};
}

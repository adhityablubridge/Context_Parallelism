#pragma once

// ---------------------------------------------------------------------------
// FusedSDPAOp.h
//
// C++ wrapper that calls the FlashAttention CUDA kernel and returns the same
// SDPAResult{out, lse} struct as sdpa_forward() in SDPAOp.h.
//
// Drop-in replacement for sdpa_forward() in the hot path.  The returned
// tensors are NOT connected to an autograd graph (the kernel bypasses it).
// For the backward pass continue using sdpa_backward_op_manual() from
// SDPAOp.h -- it recomputes the unfused forward internally.
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
// q_offset / k_offset are used only for causal masking during ring-attention
// CP steps where Q and K/V may come from different positions in the sequence.
// For standard (non-CP) use leave them at 0.
//
// LSE shape: [B, H, T_q, 1]  -- matches SDPAMerger::step() expectations.
// ---------------------------------------------------------------------------

#include "core/Tensor.h"
#include "ops/helpers/AttentionKernels.h"
#include "context_parallel/AttentionBackward.h"
#include "context_parallel/AttentionForward.h"
#include "context_parallel/FusedSDPABackwardKernel.h"
#include "context_parallel/FusedSDPAKernel.h"
#include "context_parallel/SDPAOp.h"

#include <cuda_runtime.h>
#include <cstdio>
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

  const int BH = static_cast<int>(B * H);

  // --- allocate output tensors on the same device as Q -------------------
  TensorOptions base_opts = q.opts().with_req_grad(false);

  Shape out_shape({{B, H, T_q, D}});
  Tensor out = Tensor::empty(out_shape, base_opts);

  // LSE shape: [B, H, T_q, 1]  (keepdim convention used by SDPAMerger)
  Shape lse_shape({{B, H, T_q, 1}});
  Tensor lse = Tensor::empty(lse_shape, base_opts);

  // --- get raw float* pointers (device memory) ---------------------------
  // The kernel treats the layout as [BH, T, D] by multiplying the B and H
  // dimensions into the first axis -- valid because Tensor storage is
  // row-major and contiguous.
  const float *Q_ptr = q.data<float>();
  const float *K_ptr = k.data<float>();
  const float *V_ptr = v.data<float>();
  float *O_ptr = out.data<float>();

  // LSE kernel output is flat [BH, T_q]; we write into the lse tensor's
  // storage directly (its total element count equals BH * T_q because the
  // trailing 1-dim is just a keepdim view).
  float *LSE_ptr = lse.data<float>();

  // --- launch kernel ------------------------------------------------------
  // Always use launch_flash_attn_fwd_f32 (FusedSDPAKernel.cu).
  // The WMMA TC path (mem_efficient_attn_forward_tc from AttentionKernels.cu)
  // is pending bug investigation and disabled until verified correct.

  // launch_flash_attn_fwd_f32(Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr, BH,
  //                           static_cast<int>(T_q), static_cast<int>(T_k),
  //                           static_cast<int>(D), scale, is_causal, q_offset,
  //                           k_offset);

  // Extract strides from Q/K/V so the kernel can read non-contiguous inputs
  // (e.g. transposed views) without an upstream contiguous copy.
  // Output `out` and `lse` are freshly allocated above and packed; their
  // strides are derived from shape.
  const auto& qs = q.stride().strides;
  const auto& ks = k.stride().strides;
  const auto& vs = v.stride().strides;
  // [B, H, T, D] → strides[0]=B, strides[1]=H, strides[2]=T, strides[3]=D(=1).
  const int64_t q_sB = qs[0], q_sH = qs[1], q_sM = qs[2];
  const int64_t k_sB = ks[0], k_sH = ks[1], k_sM = ks[2];
  const int64_t v_sB = vs[0], v_sH = vs[1], v_sM = vs[2];
  const int64_t o_sB = H * T_q * D, o_sH = T_q * D, o_sM = D;
  const int64_t lse_sB = H * T_q,  lse_sH = T_q;

  // Parity test: ATTN_FP32=1 forces the scalar fp32 attention kernel (no TF32
  // WMMA) to isolate whether the C++<->PT attention gap is TF32-vs-fp32.
  static const bool attn_fp32 = (std::getenv("ATTN_FP32") != nullptr);
  {
    // One-time confirmation of the active forward attention path (mirrors the
    // [CP ring ...] prints). Set ATTN_FP32 (any value) to force the scalar fp32
    // kernel; UNSET to use the TF32 WMMA/cp.async TC path.
    static bool _fp32_once = false;
    if (!_fp32_once) { _fp32_once = true;
      fprintf(stderr, "[CP sdpa fwd] %s\n", attn_fp32
          ? "ATTN_FP32=on -> scalar fp32 kernel (no cp.async/WMMA)"
          : "ATTN_FP32=off -> TF32 WMMA/cp.async TC kernel"); }
  }
  if (attn_fp32) {
    OwnTensor::cp::cuda::mem_efficient_attn_forward_strided(
      Q_ptr, q_sB, q_sM, q_sH,
      K_ptr, k_sB, k_sM, k_sH,
      V_ptr, v_sB, v_sM, v_sH,
      O_ptr, o_sB, o_sM, o_sH,
      LSE_ptr, lse_sB, lse_sH,
      B, H, T_q, T_k, q_offset, k_offset, D,
      is_causal, 0.0f, nullptr);
  } else {
    OwnTensor::cp::cuda::mem_efficient_attn_forward_tc_strided(
      Q_ptr, q_sB, q_sM, q_sH,
      K_ptr, k_sB, k_sM, k_sH,
      V_ptr, v_sB, v_sM, v_sH,
      O_ptr, o_sB, o_sM, o_sH,
      LSE_ptr, lse_sB, lse_sH,
      B, H, T_q, T_k, q_offset, k_offset, D,
      is_causal, 0.0f, nullptr);
  }

  // Synchronise so the caller can safely read results immediately
  // cudaDeviceSynchronize();

  return SDPAResult{out, lse};
}

// ---------------------------------------------------------------------------
// sdpa_fused_backward
//
// Fused FlashAttention backward.  Replaces sdpa_backward_op_manual() for the
// hot path in ContextParallelBackward.
//
// Inputs:
//   q, k, v        -- saved forward inputs      [B, H, T_q/k, D]
//   grad_out       -- incoming gradient dO       [B, H, T_q, D]
//   out            -- merged forward output O    [B, H, T_q, D]
//   merged_lse     -- merged log-sum-exp         [B, H, T_q, 1]
//
// For CP ring steps, P_ij = exp(s_ij - merged_lse_i) -- the merger
// rescaling (exp(step_lse - merged_lse)) is baked into the LSE directly,
// so no separate lse_diff tensor is needed.
//
// Supported D: 32, 64, 128.
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

  // D_buf [BH, T_q] -- scratch written by dQ kernel, read by dK/dV kernel
  Shape d_shape({{static_cast<int64_t>(BH), T_q}});
  Tensor D_buf = Tensor::empty(d_shape, base_opts);

  // launch_flash_attn_bwd_f32(
  //     q.data<float>(), k.data<float>(), v.data<float>(), out.data<float>(),
  //     grad_out.data<float>(), merged_lse.data<float>(), dQ.data<float>(),
  //     dK.data<float>(), dV.data<float>(), D_buf.data<float>(), BH,
  //     static_cast<int>(T_q), static_cast<int>(T_k), static_cast<int>(D), scale,
  //     is_causal, q_offset, k_offset);

  // Extract input strides; outputs (dQ, dK, dV) are freshly allocated above
  // and contiguous so their strides are the contiguous ones.
  const auto& qs   = q.stride().strides;
  const auto& ks   = k.stride().strides;
  const auto& vs   = v.stride().strides;
  const auto& os   = out.stride().strides;
  const auto& dos_ = grad_out.stride().strides;
  // [B, H, T, D] → strides[0]=B, strides[1]=H, strides[2]=T, strides[3]=D(=1).
  const int64_t q_sB = qs[0],  q_sH = qs[1],  q_sM = qs[2];
  const int64_t k_sB = ks[0],  k_sH = ks[1],  k_sM = ks[2];
  const int64_t v_sB = vs[0],  v_sH = vs[1],  v_sM = vs[2];
  const int64_t o_sB = os[0],  o_sH = os[1],  o_sM = os[2];
  const int64_t do_sB = dos_[0], do_sH = dos_[1], do_sM = dos_[2];
  const int64_t lse_sB = H * T_q,  lse_sH = T_q;
  const int64_t dq_sB = H * T_q * D, dq_sH = T_q * D, dq_sM = D;
  const int64_t dk_sB = H * T_k * D, dk_sH = T_k * D, dk_sM = D;
  const int64_t dv_sB = H * T_k * D, dv_sH = T_k * D, dv_sM = D;

  OwnTensor::cp::cuda::mem_efficient_attn_backward_strided(
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

  // cudaDeviceSynchronize();

  return {dQ, dK, dV};
}


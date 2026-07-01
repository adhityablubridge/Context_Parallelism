#pragma once
// ---------------------------------------------------------------------------
// FusedRoPESDPA.h — CP wrapper for the team's FUSED (QK-norm + RoPE + GQA +
// causal) attention kernel, extended with the 4-delta position interface.
// Phase 1c of the fused-RoPE CP plan.
//
// SPECULATIVE: built against the PROPOSED Phase-0 signature before the kernel
// team has frozen it. All kernel-touching arguments are funneled through the
// two wrappers below, so if the agreed signature shifts, the change is here in
// one place — not scattered across the ring loop.
//
// NON-DESTRUCTIVE: this is a NEW file. The existing sdpa_fused_forward(...) /
// sdpa_fused_backward(...) in FusedSDPAOp.h are NOT modified. The extern kernel
// symbol and the wrapper bodies are compiled only under -DCP_FUSED_ROPE, so the
// default build (GPT-2/wpe, no RoPE) neither requires the team's kernel symbol
// nor changes behavior. Calling a wrapper without the macro throws a clear
// error (it is never reached on the default path: ContextParallel only calls it
// when use_rope_ is set via enable_rope()).
//
// Position model (see RopeDeltas.h): for a PASSED tensor of length len,
//   global_pos(j) = j + (j >= len/2 ? d1 : d0)
// applied per side (q_d0/q_d1, k_d0/k_d1). Degenerates to today's single
// pos_offset when d0==d1 (contiguous) and to identity when all deltas are 0.
// ---------------------------------------------------------------------------

#include "core/Tensor.h"
#include "context_parallel/SDPAOp.h"   // SDPAResult{out, lse}

#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace OwnTensor;

#if defined(CP_FUSED_ROPE)
namespace OwnTensor {
namespace cp {
namespace cuda {

// ---- FROZEN CONTRACT (Phase 0) — implemented by the kernel team -------------
// Fused forward: QK-norm(gamma) + RoPE(4-delta global positions) + GQA + causal
// attention. fp32 in/out; internal precision is the kernel's choice (must
// reduce to today's single-GPU output when q_d0==q_d1, k_d0==k_d1, offsets 0).
// Last dim of every tensor has stride 1.
void gqa_fused_rope_cp_forward(
    const float* query, int64_t q_sB, int64_t q_sM, int64_t q_sH,
    const float* key,   int64_t k_sB, int64_t k_sM, int64_t k_sH,
    const float* value, int64_t v_sB, int64_t v_sM, int64_t v_sH,
    float* output,      int64_t o_sB, int64_t o_sM, int64_t o_sH,
    float* lse,         int64_t lse_sB, int64_t lse_sH,         // [B,Nq,T_q] (keepdim [.,.,.,1])
    const float* cos_sin_cache,                                // [cache_len, hd]
    const float* q_gamma, const float* k_gamma,                // [hd] each (nullptr => skip QK-norm)
    int64_t B, int64_t Nq_heads, int64_t Nkv_heads,
    int64_t T_q, int64_t T_k,
    int q_d0, int q_d1, int k_d0, int k_d1,                     // per-side head/tail deltas
    int64_t hd, int cache_len,
    bool is_causal, float scale, float eps);

// Fused backward: recomputes RoPE/QK-norm with the SAME deltas. Caller supplies
// the LSE to use (merged or per-step). Returns dQ/dK/dV and the PER-CALL partial
// gamma grads dq_gamma/dk_gamma (caller accumulates locally + rides the existing
// param-grad all-reduce — see RopeGammaAccum.h). D_buf is caller scratch [B*Nq,T_q].
void gqa_fused_rope_cp_backward(
    const float* query, int64_t q_sB, int64_t q_sM, int64_t q_sH,
    const float* key,   int64_t k_sB, int64_t k_sM, int64_t k_sH,
    const float* value, int64_t v_sB, int64_t v_sM, int64_t v_sH,
    const float* output,      int64_t o_sB, int64_t o_sM, int64_t o_sH,
    const float* grad_output, int64_t do_sB, int64_t do_sM, int64_t do_sH,
    const float* lse,         int64_t lse_sB, int64_t lse_sH,
    const float* cos_sin_cache, const float* q_gamma, const float* k_gamma,
    float* grad_query, int64_t dq_sB, int64_t dq_sM, int64_t dq_sH,
    float* grad_key,   int64_t dk_sB, int64_t dk_sM, int64_t dk_sH,
    float* grad_value, int64_t dv_sB, int64_t dv_sM, int64_t dv_sH,
    float* dq_gamma, float* dk_gamma,                          // [hd] each, per-call partials
    float* D_buf,
    int64_t B, int64_t Nq_heads, int64_t Nkv_heads,
    int64_t T_q, int64_t T_k,
    int q_d0, int q_d1, int k_d0, int k_d1,
    int64_t hd, int cache_len,
    bool is_causal, float scale, float eps);

} // namespace cuda
} // namespace cp
} // namespace OwnTensor
#endif // CP_FUSED_ROPE

// Result of the RoPE-fused backward: activation grads + per-call gamma partials.
struct SDPARoPEBackwardResult {
  Tensor dQ, dK, dV;          // [B,H,T_q/k,D]
  Tensor dq_gamma, dk_gamma;  // [hd] per-call partials (caller accumulates)
};

// ---------------------------------------------------------------------------
// sdpa_fused_forward_rope — forward wrapper (mirrors sdpa_fused_forward).
// q,k,v: [B, Nq|Nkv, T_q|T_k, D] fp32. q_gamma/k_gamma/cos_sin_cache fp32.
// Returns {out [B,Nq,T_q,D], lse [B,Nq,T_q,1]}.
// ---------------------------------------------------------------------------
inline SDPAResult sdpa_fused_forward_rope(
    Tensor& q, Tensor& k, Tensor& v, bool is_causal, float scale,
    int q_d0, int q_d1, int k_d0, int k_d1,
    const Tensor& cos_sin_cache, const Tensor& q_gamma, const Tensor& k_gamma,
    float eps) {
#if defined(CP_FUSED_ROPE)
  if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4)
    throw std::runtime_error("sdpa_fused_forward_rope: Q,K,V must be 4-D [B,H,T,D]");
  if (q.dtype() != Dtype::Float32)
    throw std::runtime_error("sdpa_fused_forward_rope: only Float32 supported");

  const int64_t B   = q.shape().dims[0];
  const int64_t Nq  = q.shape().dims[1];
  const int64_t T_q = q.shape().dims[2];
  const int64_t D   = q.shape().dims[3];
  const int64_t Nkv = k.shape().dims[1];
  const int64_t T_k = k.shape().dims[2];

  TensorOptions base = q.opts().with_req_grad(false);
  Tensor out = Tensor::empty(Shape({{B, Nq, T_q, D}}), base);
  Tensor lse = Tensor::empty(Shape({{B, Nq, T_q, 1}}), base);  // keepdim (SDPAMerger)

  const auto& qs = q.stride().strides; const auto& ks = k.stride().strides;
  const auto& vs = v.stride().strides;
  const int64_t o_sB = Nq * T_q * D, o_sH = T_q * D, o_sM = D;
  const int64_t lse_sB = Nq * T_q, lse_sH = T_q;
  const int   cache_len = static_cast<int>(cos_sin_cache.shape().dims[0]);
  const float* qg = q_gamma.is_valid() ? q_gamma.data<float>() : nullptr;
  const float* kg = k_gamma.is_valid() ? k_gamma.data<float>() : nullptr;

  OwnTensor::cp::cuda::gqa_fused_rope_cp_forward(
      q.data<float>(), qs[0], qs[2], qs[1],
      k.data<float>(), ks[0], ks[2], ks[1],
      v.data<float>(), vs[0], vs[2], vs[1],
      out.data<float>(), o_sB, o_sM, o_sH,
      lse.data<float>(), lse_sB, lse_sH,
      cos_sin_cache.data<float>(), qg, kg,
      B, Nq, Nkv, T_q, T_k,
      q_d0, q_d1, k_d0, k_d1,
      D, cache_len, is_causal, scale, eps);
  return SDPAResult{out, lse};
#else
  (void)q; (void)k; (void)v; (void)is_causal; (void)scale;
  (void)q_d0; (void)q_d1; (void)k_d0; (void)k_d1;
  (void)cos_sin_cache; (void)q_gamma; (void)k_gamma; (void)eps;
  throw std::runtime_error(
      "sdpa_fused_forward_rope: build with -DCP_FUSED_ROPE and link the team's "
      "fused RoPE kernel (gqa_fused_rope_cp_forward)");
#endif
}

// ---------------------------------------------------------------------------
// sdpa_fused_backward_rope — backward wrapper. `lse` is the caller-chosen LSE
// (merged or per-step). Returns dQ/dK/dV + per-call gamma partials.
// ---------------------------------------------------------------------------
inline SDPARoPEBackwardResult sdpa_fused_backward_rope(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& grad_out,
    const Tensor& out, const Tensor& lse, bool is_causal, float scale,
    int q_d0, int q_d1, int k_d0, int k_d1,
    const Tensor& cos_sin_cache, const Tensor& q_gamma, const Tensor& k_gamma,
    float eps) {
#if defined(CP_FUSED_ROPE)
  const int64_t B   = q.shape().dims[0];
  const int64_t Nq  = q.shape().dims[1];
  const int64_t T_q = q.shape().dims[2];
  const int64_t D   = q.shape().dims[3];
  const int64_t Nkv = k.shape().dims[1];
  const int64_t T_k = k.shape().dims[2];

  TensorOptions base = q.opts().with_req_grad(false);
  Tensor dQ = Tensor::empty(q.shape(), base);
  Tensor dK = Tensor::empty(k.shape(), base);
  Tensor dV = Tensor::empty(v.shape(), base);
  Tensor dq_gamma = Tensor::zeros(Shape({{D}}), base);
  Tensor dk_gamma = Tensor::zeros(Shape({{D}}), base);
  Tensor D_buf = Tensor::empty(Shape({{B * Nq, T_q}}), base);

  const auto& qs = q.stride().strides; const auto& ks = k.stride().strides;
  const auto& vs = v.stride().strides; const auto& os = out.stride().strides;
  const auto& dos = grad_out.stride().strides;
  const int64_t lse_sB = Nq * T_q, lse_sH = T_q;
  const int64_t dq_sB = Nq * T_q * D, dq_sH = T_q * D, dq_sM = D;
  const int64_t dk_sB = Nkv * T_k * D, dk_sH = T_k * D, dk_sM = D;
  const int   cache_len = static_cast<int>(cos_sin_cache.shape().dims[0]);
  const float* qg = q_gamma.is_valid() ? q_gamma.data<float>() : nullptr;
  const float* kg = k_gamma.is_valid() ? k_gamma.data<float>() : nullptr;

  OwnTensor::cp::cuda::gqa_fused_rope_cp_backward(
      q.data<float>(), qs[0], qs[2], qs[1],
      k.data<float>(), ks[0], ks[2], ks[1],
      v.data<float>(), vs[0], vs[2], vs[1],
      out.data<float>(), os[0], os[2], os[1],
      grad_out.data<float>(), dos[0], dos[2], dos[1],
      lse.data<float>(), lse_sB, lse_sH,
      cos_sin_cache.data<float>(), qg, kg,
      dQ.data<float>(), dq_sB, dq_sM, dq_sH,
      dK.data<float>(), dk_sB, dk_sM, dk_sH,
      dV.data<float>(), dk_sB, dk_sM, dk_sH,
      dq_gamma.data<float>(), dk_gamma.data<float>(),
      D_buf.data<float>(),
      B, Nq, Nkv, T_q, T_k,
      q_d0, q_d1, k_d0, k_d1,
      D, cache_len, is_causal, scale, eps);
  return SDPARoPEBackwardResult{dQ, dK, dV, dq_gamma, dk_gamma};
#else
  (void)q; (void)k; (void)v; (void)grad_out; (void)out; (void)lse;
  (void)is_causal; (void)scale; (void)q_d0; (void)q_d1; (void)k_d0; (void)k_d1;
  (void)cos_sin_cache; (void)q_gamma; (void)k_gamma; (void)eps;
  throw std::runtime_error(
      "sdpa_fused_backward_rope: build with -DCP_FUSED_ROPE and link the team's "
      "fused RoPE kernel (gqa_fused_rope_cp_backward)");
#endif
}

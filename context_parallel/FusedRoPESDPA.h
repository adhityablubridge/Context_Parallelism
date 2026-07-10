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
#include "dtype/Types.h"               // bfloat16_t

#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace OwnTensor;

#if defined(CP_FUSED_ROPE)
namespace OwnTensor {
namespace cp {
namespace cuda {

// ---- CONTRACT (Phase 3) — implemented in GQA_fused_{fwd,bwd}_sm103_cp.cu -----
// Fused forward: QK-norm(gamma) + RoPE(4-delta global positions) + GQA + causal.
// bf16 Q/K/V (separate, contiguous [B,H,T,D]); bf16 output; fp32 lse. Internal
// math is bf16 WMMA (matches the production GQA kernel). Reduces to the non-CP
// kernel's output when q_d0==q_d1, k_d0==k_d1, and all deltas are 0. The four
// deltas index RoPE ONLY; causal masking uses local indices (the CP driver
// guarantees locally-monotonic sub-chunks). `interleaved` false => NeoX pairing
// (matches build_rope_cache). See RopeDeltas.h for the delta semantics.
void gqa_fused_rope_cp_forward(
    const bfloat16_t* Q, const bfloat16_t* K, const bfloat16_t* V,
    const float* cos_sin_cache, const float* q_gamma, const float* k_gamma,  // [hd]/nullptr
    bfloat16_t* output, float* lse,                            // lse [B,Nq,T_q] contiguous
    int B, int Nq_heads, int Nkv_heads, int T_q, int T_k, int hd, int cache_seq_len,
    int q_d0, int q_d1, int k_d0, int k_d1,                     // per-side head/tail deltas
    float eps, bool interleaved, bool is_causal, float scale);

// Fused backward: recomputes RoPE/QK-norm (and rstd) with the SAME deltas. lse
// is the caller-chosen LSE (merged or per-step). Returns dQ/dK/dV (bf16) and the
// PER-CALL partial gamma grads dq_gamma/dk_gamma (fp32; caller accumulates
// locally + rides the param-grad all-reduce — see RopeGammaAccum.h).
void gqa_fused_rope_cp_backward(
    const bfloat16_t* Q, const bfloat16_t* K, const bfloat16_t* V,
    const float* cos_sin_cache, const float* q_gamma, const float* k_gamma,
    const bfloat16_t* output, const bfloat16_t* grad_output, const float* lse,
    bfloat16_t* grad_Q, bfloat16_t* grad_K, bfloat16_t* grad_V,
    float* dq_gamma, float* dk_gamma,                          // [hd] each, per-call partials
    int B, int Nq_heads, int Nkv_heads, int T_q, int T_k, int hd, int cache_seq_len,
    int q_d0, int q_d1, int k_d0, int k_d1,
    float eps, bool interleaved, bool is_causal, float scale);

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

  // Cast to contiguous bf16 for the WMMA kernel; lse stays fp32.
  Tensor qb = q.contiguous().as_type(Dtype::Bfloat16);
  Tensor kb = k.contiguous().as_type(Dtype::Bfloat16);
  Tensor vb = v.contiguous().as_type(Dtype::Bfloat16);
  TensorOptions bf = q.opts().with_dtype(Dtype::Bfloat16).with_req_grad(false);
  TensorOptions fp = q.opts().with_dtype(Dtype::Float32).with_req_grad(false);
  Tensor out_bf = Tensor::empty(Shape({{B, Nq, T_q, D}}), bf);
  Tensor lse    = Tensor::empty(Shape({{B, Nq, T_q, 1}}), fp);  // keepdim (SDPAMerger)

  const int   cache_len = static_cast<int>(cos_sin_cache.shape().dims[0]);
  const float* qg = q_gamma.is_valid() ? q_gamma.data<float>() : nullptr;
  const float* kg = k_gamma.is_valid() ? k_gamma.data<float>() : nullptr;

  OwnTensor::cp::cuda::gqa_fused_rope_cp_forward(
      qb.data<bfloat16_t>(), kb.data<bfloat16_t>(), vb.data<bfloat16_t>(),
      cos_sin_cache.data<float>(), qg, kg,
      out_bf.data<bfloat16_t>(), lse.data<float>(),
      static_cast<int>(B), static_cast<int>(Nq), static_cast<int>(Nkv),
      static_cast<int>(T_q), static_cast<int>(T_k), static_cast<int>(D), cache_len,
      q_d0, q_d1, k_d0, k_d1,
      eps, /*interleaved=*/false, is_causal, scale);
  Tensor out = out_bf.as_type(Dtype::Float32);
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

  // Cast inputs to contiguous bf16 for the WMMA kernel.
  Tensor qb  = q.contiguous().as_type(Dtype::Bfloat16);
  Tensor kb  = k.contiguous().as_type(Dtype::Bfloat16);
  Tensor vb  = v.contiguous().as_type(Dtype::Bfloat16);
  Tensor ob  = out.contiguous().as_type(Dtype::Bfloat16);
  Tensor dob = grad_out.contiguous().as_type(Dtype::Bfloat16);
  TensorOptions bf = q.opts().with_dtype(Dtype::Bfloat16).with_req_grad(false);
  TensorOptions fp = q.opts().with_dtype(Dtype::Float32).with_req_grad(false);
  Tensor dQb = Tensor::empty(Shape({{B, Nq,  T_q, D}}), bf);
  Tensor dKb = Tensor::empty(Shape({{B, Nkv, T_k, D}}), bf);
  Tensor dVb = Tensor::empty(Shape({{B, Nkv, T_k, D}}), bf);
  Tensor dq_gamma = Tensor::zeros(Shape({{D}}), fp);
  Tensor dk_gamma = Tensor::zeros(Shape({{D}}), fp);

  const int   cache_len = static_cast<int>(cos_sin_cache.shape().dims[0]);
  const float* qg = q_gamma.is_valid() ? q_gamma.data<float>() : nullptr;
  const float* kg = k_gamma.is_valid() ? k_gamma.data<float>() : nullptr;

  OwnTensor::cp::cuda::gqa_fused_rope_cp_backward(
      qb.data<bfloat16_t>(), kb.data<bfloat16_t>(), vb.data<bfloat16_t>(),
      cos_sin_cache.data<float>(), qg, kg,
      ob.data<bfloat16_t>(), dob.data<bfloat16_t>(), lse.data<float>(),
      dQb.data<bfloat16_t>(), dKb.data<bfloat16_t>(), dVb.data<bfloat16_t>(),
      dq_gamma.data<float>(), dk_gamma.data<float>(),
      static_cast<int>(B), static_cast<int>(Nq), static_cast<int>(Nkv),
      static_cast<int>(T_q), static_cast<int>(T_k), static_cast<int>(D), cache_len,
      q_d0, q_d1, k_d0, k_d1,
      eps, /*interleaved=*/false, is_causal, scale);
  return SDPARoPEBackwardResult{
      dQb.as_type(Dtype::Float32), dKb.as_type(Dtype::Float32), dVb.as_type(Dtype::Float32),
      dq_gamma, dk_gamma};
#else
  (void)q; (void)k; (void)v; (void)grad_out; (void)out; (void)lse;
  (void)is_causal; (void)scale; (void)q_d0; (void)q_d1; (void)k_d0; (void)k_d1;
  (void)cos_sin_cache; (void)q_gamma; (void)k_gamma; (void)eps;
  throw std::runtime_error(
      "sdpa_fused_backward_rope: build with -DCP_FUSED_ROPE and link the team's "
      "fused RoPE kernel (gqa_fused_rope_cp_backward)");
#endif
}

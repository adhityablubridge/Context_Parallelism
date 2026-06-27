#pragma once

namespace OwnTensor {
namespace cp {
namespace cuda {

// CP-aware Ada (sm89) PTX-MMA backward. Same interface as
// mem_efficient_attn_backward_strided but implemented with the Ada-tuned
// mma.sync.m16n8k8 TF32 "exp12" kernel. Supports T_q != T_k and non-zero
// q_offset / k_offset (global-position causal masking).
//
// dQ is zeroed internally and accumulated via atomicAdd within the call;
// dK/dV are written by direct assignment. dQ/dK/dV must be CONTIGUOUS
// allocations (typical: Tensor::empty). Last dim of each tensor stride=1.
// Build/run on sm89 only.
void mem_efficient_attn_backward_sm89_strided(
    const float* query, int64_t q_strideB, int64_t q_strideM, int64_t q_strideH,
    const float* key,   int64_t k_strideB, int64_t k_strideM, int64_t k_strideH,
    const float* value, int64_t v_strideB, int64_t v_strideM, int64_t v_strideH,
    const float* output,      int64_t o_strideB,  int64_t o_strideM,  int64_t o_strideH,
    const float* grad_output, int64_t do_strideB, int64_t do_strideM, int64_t do_strideH,
    const float* lse,         int64_t lse_strideB, int64_t lse_strideH,
    float* grad_query,  int64_t dq_strideB, int64_t dq_strideM, int64_t dq_strideH,
    float* grad_key,    int64_t dk_strideB, int64_t dk_strideM, int64_t dk_strideH,
    float* grad_value,  int64_t dv_strideB, int64_t dv_strideM, int64_t dv_strideH,
    float* D_buf,
    int64_t B, int64_t nh,
    int64_t T_q, int64_t T_k,
    int q_offset, int k_offset,
    int64_t hd,
    bool is_causal);

} // namespace cuda
} // namespace cp
} // namespace OwnTensor

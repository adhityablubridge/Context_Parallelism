#pragma once

namespace OwnTensor {
namespace cp {
namespace cuda {

// CP-aware Ada (sm89) PTX-MMA forward. Same interface as
// mem_efficient_attn_forward_tc_strided but implemented with the Ada-tuned
// mma.sync.m16n8k8 TF32 kernel. Supports T_q != T_k and non-zero
// q_offset / k_offset (global-position causal masking).
//
// Last dim of each tensor must have stride=1. Build/run on sm89 only.
void mem_efficient_attn_forward_tc_sm89_strided(
    const float* query, int64_t q_strideB, int64_t q_strideM, int64_t q_strideH,
    const float* key,   int64_t k_strideB, int64_t k_strideM, int64_t k_strideH,
    const float* value, int64_t v_strideB, int64_t v_strideM, int64_t v_strideH,
    float* output,      int64_t o_strideB, int64_t o_strideM, int64_t o_strideH,
    float* lse,         int64_t lse_strideB, int64_t lse_strideH,
    int64_t B, int64_t nh,
    int64_t T_q, int64_t T_k,
    int q_offset, int k_offset,
    int64_t hd,
    bool is_causal,
    float dropout_p, const float* dropout_mask);

} // namespace cuda
} // namespace cp
} // namespace OwnTensor

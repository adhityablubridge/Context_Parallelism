#pragma once

namespace OwnTensor {
namespace cp {
namespace cuda {

// Contiguous wrapper — assumes packed [B, nh, T, hd] layout, last dim stride=1.
void mem_efficient_attn_forward_tc(
    const float* query, const float* key, const float* value,
    float* output, float* lse,
    int64_t B, int64_t nh,
    int64_t T_q, int64_t T_k,
    int q_offset, int k_offset,
    int64_t hd,
    bool is_causal,
    float dropout_p, const float* dropout_mask);

// Strided variant — caller passes per-tensor B/M/H strides.
// Last dim of each tensor must have stride=1 (head-dim contig).
void mem_efficient_attn_forward_tc_strided(
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

// Scalar fp32 strided variant (no TF32 WMMA) — used for ATTN_FP32 parity test.
void mem_efficient_attn_forward_strided(
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

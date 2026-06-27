#pragma once

namespace OwnTensor {
namespace cp {
namespace cuda {

// Contiguous wrapper — assumes packed [B, nh, T, hd] layout.
void mem_efficient_attn_backward(
    const float* query, const float* key, const float* value,
    const float* output, const float* grad_output, const float* lse,
    float* grad_query, float* grad_key, float* grad_value,
    float* D_buf,
    int64_t B, int64_t nh,
    int64_t T_q, int64_t T_k,
    int q_offset, int k_offset,
    int64_t hd,
    bool is_causal);

// Strided variant — caller passes per-tensor B/M/H strides for all I/O tensors
// (last dim must have stride=1). dQ/dK/dV are still cleared via cudaMemsetAsync
// inside the kernel launch, so they must be CONTIGUOUS allocations (typical
// case: freshly allocated via Tensor::empty).
void mem_efficient_attn_backward_strided(
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

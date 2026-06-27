#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace OwnTensor {
namespace cp {
namespace cuda {

// ---------------------------------------------------------------------------
// Pack non-contiguous K and V into a flat contiguous send buffer for the CP
// ring rotator. Output layout: [2 * B * H * T * D] floats — K first, then V.
//
// Source K, V have shape [B, H, T, D] with arbitrary B/H/M strides; the last
// (D) dim must have stride=1 (which the upstream tensors always satisfy).
//
// For contiguous sources this is bandwidth-equivalent to cudaMemcpyAsync; for
// strided sources it gathers correctly so the receiver always reshapes back
// into a contiguous logical [B, H, T, D].
// ---------------------------------------------------------------------------
void launch_pack_kv_send(
    const float* K, int64_t k_strideB, int64_t k_strideH, int64_t k_strideM,
    const float* V, int64_t v_strideB, int64_t v_strideH, int64_t v_strideM,
    float* out,
    int B, int H, int T, int D,
    cudaStream_t stream);

} // namespace cuda
} // namespace cp
} // namespace OwnTensor

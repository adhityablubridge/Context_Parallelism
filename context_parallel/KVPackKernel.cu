#include "context_parallel/KVPackKernel.h"

namespace OwnTensor {
namespace cp {

// ---------------------------------------------------------------------------
// Grid: (B*H*T, 2)  — one block per (b,h,t,tensor_id), tensor_id 0=K, 1=V
// Block: blockDim.x threads grid-stride over D
//
// Reads via per-tensor strides; writes contiguously into out at
// [tensor_id, b, h, t, d] flattened.
// ---------------------------------------------------------------------------
__global__ void pack_kv_send_buf_kernel(
    const float* __restrict__ K, int64_t k_sB, int64_t k_sH, int64_t k_sM,
    const float* __restrict__ V, int64_t v_sB, int64_t v_sH, int64_t v_sM,
    float* __restrict__ out,
    int B, int H, int T, int D)
{
    const int row       = blockIdx.x;        // 0 .. B*H*T - 1
    const int tensor_id = blockIdx.y;        // 0=K, 1=V

    const int t = row % T;
    const int h = (row / T) % H;
    const int b = row / (T * H);

    const int64_t src_off = (tensor_id == 0)
        ? (int64_t)b * k_sB + (int64_t)h * k_sH + (int64_t)t * k_sM
        : (int64_t)b * v_sB + (int64_t)h * v_sH + (int64_t)t * v_sM;
    const float* src = (tensor_id == 0 ? K : V) + src_off;

    const int64_t per_tensor = (int64_t)B * H * T * D;
    float* dst = out + (int64_t)tensor_id * per_tensor + (int64_t)row * D;

    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        dst[d] = src[d];
    }
}

namespace cuda {

void launch_pack_kv_send(
    const float* K, int64_t k_strideB, int64_t k_strideH, int64_t k_strideM,
    const float* V, int64_t v_strideB, int64_t v_strideH, int64_t v_strideM,
    float* out,
    int B, int H, int T, int D,
    cudaStream_t stream)
{
    const int rows = B * H * T;
    if (rows <= 0 || D <= 0) return;

    dim3 grid(rows, 2);
    // Pick a block size that comfortably covers D with grid-stride on overflow.
    int threads;
    if (D >= 256)       threads = 256;
    else if (D >= 128)  threads = 128;
    else if (D >= 64)   threads = 64;
    else                threads = 32;

    pack_kv_send_buf_kernel<<<grid, threads, 0, stream>>>(
        K, k_strideB, k_strideH, k_strideM,
        V, v_strideB, v_strideH, v_strideM,
        out, B, H, T, D);
}

} // namespace cuda
} // namespace cp
} // namespace OwnTensor

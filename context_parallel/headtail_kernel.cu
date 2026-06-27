#include <cuda_runtime.h>
#include <cassert>
#include "headtail_kernel.cuh"

// ---------------------------------------------------------------------------
// HeadTail loadbalance kernel (chunk-level, PyTorch-parity)
//
// For a tensor [outer, T, inner] (outer = B*H, inner = D) with world_size N:
// split T into 2*N equal chunks of size chunk_sz = T/(2*N). Rank r owns
// chunks (r, 2N-1-r) concatenated as [head_chunk, tail_chunk]. The global
// permutation that achieves this layout is:
//
//   For each output position g in [0, T):
//     r = g / T_local;   i = g % T_local;   T_local = T/N = 2*chunk_sz
//     if (i < chunk_sz)  src = r * chunk_sz + i              (head_chunk = r)
//     else               src = (2N-1-r)*chunk_sz + (i-chunk_sz)  (tail_chunk = 2N-1-r)
//
// After this permutation, splitting along T into N equal chunks gives each
// rank a contiguous head-half (early global positions) and tail-half (late
// global positions), matching the sub-chunk causal dispatch in
// ContextParallel::forward_cp.
//
// Matches PyTorch _rearrange_seq_for_load_balance @ torch/distributed/tensor/
// experimental/_context_parallel/_attention.py (commit e9ebbd3b).
// ---------------------------------------------------------------------------
__global__ void headtail_loadbalance_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int64_t outer_size,
    int64_t seq_len,
    int64_t inner_size,
    int64_t world_size,
    int64_t total_elements)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    // Decompose flat index into (outer, seq, inner)
    int64_t inner_idx = idx % inner_size;
    int64_t temp = idx / inner_size;
    int64_t seq_idx = temp % seq_len;
    int64_t outer_idx = temp / seq_len;

    // Chunk-level HeadTail mapping (PyTorch parity).
    int64_t t_local = seq_len / world_size;
    int64_t chunk_sz = t_local / 2;
    int64_t r = seq_idx / t_local;
    int64_t i = seq_idx % t_local;
    int64_t src_seq = (i < chunk_sz)
        ? (r * chunk_sz + i)
        : ((2 * world_size - 1 - r) * chunk_sz + (i - chunk_sz));

    int64_t src_idx = outer_idx * (seq_len * inner_size)
                    + src_seq * inner_size
                    + inner_idx;

    dst[idx] = src[src_idx];
}

// ---------------------------------------------------------------------------
// HeadTail unloadbalance kernel (inverse of chunk-level loadbalance)
//
// For each output position g in [0, T):
//   c = g / chunk_sz;   o = g % chunk_sz
//   if (c < N)  r = c,             src = r * T_local + o            (head half of rank r)
//   else        r = 2N-1-c,        src = r * T_local + chunk_sz + o (tail half of rank r)
// ---------------------------------------------------------------------------
__global__ void headtail_unloadbalance_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int64_t outer_size,
    int64_t seq_len,
    int64_t inner_size,
    int64_t world_size,
    int64_t total_elements)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    // Decompose flat index into (outer, seq, inner)
    int64_t inner_idx = idx % inner_size;
    int64_t temp = idx / inner_size;
    int64_t dst_seq = temp % seq_len;
    int64_t outer_idx = temp / seq_len;

    // Chunk-level HeadTail inverse mapping.
    int64_t t_local = seq_len / world_size;
    int64_t chunk_sz = t_local / 2;
    int64_t c = dst_seq / chunk_sz;
    int64_t o = dst_seq % chunk_sz;
    int64_t src_seq;
    if (c < world_size) {
        int64_t r = c;
        src_seq = r * t_local + o;
    } else {
        int64_t r = 2 * world_size - 1 - c;
        src_seq = r * t_local + chunk_sz + o;
    }

    int64_t src_idx = outer_idx * (seq_len * inner_size)
                    + src_seq * inner_size
                    + inner_idx;

    dst[idx] = src[src_idx];
}

// ---------------------------------------------------------------------------
// Launcher functions
// ---------------------------------------------------------------------------
void launch_headtail_loadbalance(
    const float* src,
    float* dst,
    int64_t outer_size,
    int64_t seq_len,
    int64_t inner_size,
    int64_t world_size,
    cudaStream_t stream)
{
    assert(world_size > 0);
    assert(seq_len % (2 * world_size) == 0);

    int64_t total = outer_size * seq_len * inner_size;
    int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);

    headtail_loadbalance_kernel<<<blocks, threads, 0, stream>>>(
        src, dst, outer_size, seq_len, inner_size, world_size, total);
}

void launch_headtail_unloadbalance(
    const float* src,
    float* dst,
    int64_t outer_size,
    int64_t seq_len,
    int64_t inner_size,
    int64_t world_size,
    cudaStream_t stream)
{
    assert(world_size > 0);
    assert(seq_len % (2 * world_size) == 0);

    int64_t total = outer_size * seq_len * inner_size;
    int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);

    headtail_unloadbalance_kernel<<<blocks, threads, 0, stream>>>(
        src, dst, outer_size, seq_len, inner_size, world_size, total);
}

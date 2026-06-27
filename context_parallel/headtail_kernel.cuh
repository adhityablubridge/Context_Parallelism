#pragma once

#include <cuda_runtime.h>

// HeadTail permutation kernel launcher declarations (chunk-level, PyTorch parity).
//
// For sequence length T, world_size N, chunk_sz = T/(2*N), T_local = T/N:
// Splits T into 2*N equal chunks; rank r owns chunks (r, 2N-1-r) concatenated
// as [head_chunk, tail_chunk]. After loadbalance + split-by-N, each rank's
// slice has an early-positions head half and late-positions tail half.
//
// Example (N=2, T=8): perm = [0,1, 6,7, 2,3, 4,5]
// Example (N=4, T=16): perm = [0,1, 14,15, 2,3, 12,13, 4,5, 10,11, 6,7, 8,9]
//
// Matches PyTorch _rearrange_seq_for_load_balance (commit e9ebbd3b).

void launch_headtail_loadbalance(
    const float* src,
    float* dst,
    int64_t outer_size,   // product of dims before chunkdim (B * H for 4D with chunkdim=2)
    int64_t seq_len,      // size of chunkdim (T)
    int64_t inner_size,   // product of dims after chunkdim (D for 4D with chunkdim=2)
    int64_t world_size,   // N (number of CP ranks)
    cudaStream_t stream);

void launch_headtail_unloadbalance(
    const float* src,
    float* dst,
    int64_t outer_size,
    int64_t seq_len,
    int64_t inner_size,
    int64_t world_size,
    cudaStream_t stream);

// ---------------------------------------------------------------------------
// FusedSDPAKernel.cu
//
// CUDA implementation of the FlashAttention forward pass.
//
// Algorithm: Online softmax with shared-memory tiling (Dao et al. 2022).
// Q, K, V tiles are loaded into shared memory; the full [T_q x T_k] attention
// matrix is never materialized in HBM.  Only the final output O and the
// log-sum-exp LSE are written back to global memory.
//
// Kernel grid/block layout:
//   Grid  : (ceil(T_q / BLOCK_Q),  BH)    -- BH = B*H (batch-heads merged)
//   Block : (BLOCK_Q)                      -- one thread owns one query row
//
// Shared memory (per block):
//   K_tile : float[BLOCK_K][D]  (cooperative load, one row per thread)
//   V_tile : float[BLOCK_K][D]  (cooperative load, one row per thread)
//   Total  : 2 * BLOCK_K * D * 4 bytes
//            For D=64, BLOCK_K=32 : 16 KB  (well within 48 KB Ampere default)
//
// Register arrays per thread (compiler allocates):
//   q_regs  : float[HEAD_DIM]  -- loaded once, read many
//   o_regs  : float[HEAD_DIM]  -- running unnormalised output
//   scores  : float[BLOCK_K]   -- per-tile attention logits
//
// The template parameter HEAD_DIM is resolved at kernel-launch time by the
// dispatcher below.  Supported values: 32, 64, 128.
// ---------------------------------------------------------------------------

#include "FusedSDPAKernel.h"
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

static constexpr int BLOCK_Q = 32;
static constexpr int BLOCK_K = 32;

// ---------------------------------------------------------------------------
// flash_attn_fwd_kernel
//
// One thread handles one query row.  Iterates over K/V tiles in a left-to-right
// sweep, maintaining the online softmax running state (m_i, l_i, o_i).
//
// After the sweep:
//   o_i  is normalised  (divided by l_i)
//   LSE  = m_i + log(l_i)
// ---------------------------------------------------------------------------
template <int HEAD_DIM>
__global__ void flash_attn_fwd_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__       O,
    float* __restrict__       LSE,
    int T_q, int T_k,
    float scale,
    bool  is_causal,
    int   q_offset,
    int   k_offset)
{
    // --- index computation ---------------------------------------------------
    const int bh      = blockIdx.y;
    const int q_local = blockIdx.x * BLOCK_Q + threadIdx.x;
    const int q_global = q_offset + q_local;

    if (q_local >= T_q) return;

    // --- shared memory: K_tile and V_tile -----------------------------------
    extern __shared__ float smem[];
    float* K_tile = smem;                       // [BLOCK_K][HEAD_DIM]
    float* V_tile = smem + BLOCK_K * HEAD_DIM;  // [BLOCK_K][HEAD_DIM]

    // --- load Q row into registers -------------------------------------------
    const float* Q_ptr = Q + (bh * T_q + q_local) * HEAD_DIM;
    float q_regs[HEAD_DIM];
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) q_regs[d] = Q_ptr[d];

    // --- running accumulators ------------------------------------------------
    float m_i = -1e30f;   // running row-max
    float l_i = 0.0f;     // running sum of exp(score - max)
    float o_regs[HEAD_DIM];
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) o_regs[d] = 0.0f;

    // --- sweep over K/V tiles ------------------------------------------------
    const int num_k_blocks = (T_k + BLOCK_K - 1) / BLOCK_K;

    for (int kb = 0; kb < num_k_blocks; ++kb) {
        const int k_block_start = kb * BLOCK_K;
        const int k_global_start = k_offset + k_block_start;

        // Causal fast-exit: entire tile is in the future, no more tiles matter
        if (is_causal && k_global_start > q_global) break;

        // --- cooperative tile load -------------------------------------------
        // Thread t loads K_tile[t] and V_tile[t] (one row each)
        const int k_local = k_block_start + threadIdx.x;
        if (k_local < T_k) {
            const float* Kptr = K + (bh * T_k + k_local) * HEAD_DIM;
            const float* Vptr = V + (bh * T_k + k_local) * HEAD_DIM;
            float* Ks = K_tile + threadIdx.x * HEAD_DIM;
            float* Vs = V_tile + threadIdx.x * HEAD_DIM;
#pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) { Ks[d] = Kptr[d]; Vs[d] = Vptr[d]; }
        } else {
            // Out-of-bounds tile slot: zero out so it contributes nothing
            float* Ks = K_tile + threadIdx.x * HEAD_DIM;
            float* Vs = V_tile + threadIdx.x * HEAD_DIM;
#pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) { Ks[d] = 0.0f; Vs[d] = 0.0f; }
        }
        __syncthreads();

        // --- compute per-tile attention logits -------------------------------
        const int tile_size = (k_block_start + BLOCK_K <= T_k)
                            ? BLOCK_K
                            : (T_k - k_block_start);

        float scores[BLOCK_K];
        float tile_max = -1e30f;

        for (int j = 0; j < tile_size; ++j) {
            const int kg = k_global_start + j;
            if (is_causal && kg > q_global) {
                scores[j] = -1e30f;
                continue;
            }
            float dot = 0.0f;
            const float* Kj = K_tile + j * HEAD_DIM;
#pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) dot += q_regs[d] * Kj[d];
            scores[j] = dot * scale;
            if (scores[j] > tile_max) tile_max = scores[j];
        }
        // Fill out-of-bounds slot with -inf (no contribution)
        for (int j = tile_size; j < BLOCK_K; ++j) scores[j] = -1e30f;

        // All positions masked: skip V accumulation
        if (tile_max <= -1e29f) { __syncthreads(); continue; }

        // --- online softmax update -------------------------------------------
        const float m_new     = (tile_max > m_i) ? tile_max : m_i;
        const float exp_shift = expf(m_i - m_new);   // rescale for old state

        // Rescale old unnormalised output
#pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) o_regs[d] *= exp_shift;

        // Accumulate new tile contributions into o_regs and l_inc
        float l_inc = 0.0f;
        for (int j = 0; j < tile_size; ++j) {
            if (scores[j] <= -1e29f) continue;
            const float p_j = expf(scores[j] - m_new);
            l_inc += p_j;
            const float* Vj = V_tile + j * HEAD_DIM;
#pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) o_regs[d] += p_j * Vj[d];
        }

        l_i = l_i * exp_shift + l_inc;
        m_i = m_new;

        __syncthreads();
    }

    // --- normalise and write output -----------------------------------------
    const float inv_l = (l_i > 1e-30f) ? (1.0f / l_i) : 0.0f;
    float* O_ptr = O + (bh * T_q + q_local) * HEAD_DIM;
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) O_ptr[d] = o_regs[d] * inv_l;

    // LSE = m + log(l)  (same convention as SDPAOp.h / SDPAMerger)
    LSE[bh * T_q + q_local] = (l_i > 1e-30f) ? (m_i + logf(l_i)) : -1e30f;
}

// ---------------------------------------------------------------------------
// flash_attn_fwd_kernel_generic
//
// Non-templated fallback for head dims that are not 32, 64, or 128.
// Uses fixed-max arrays (MAX_D=512) and loops to runtime D.
// Correct for any D <= 512, no #pragma unroll.
// ---------------------------------------------------------------------------
static constexpr int MAX_D = 512;

__global__ void flash_attn_fwd_kernel_generic(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__       O,
    float* __restrict__       LSE,
    int T_q, int T_k, int D,
    float scale,
    bool  is_causal,
    int   q_offset,
    int   k_offset)
{
    const int bh      = blockIdx.y;
    const int q_local = blockIdx.x * BLOCK_Q + threadIdx.x;
    const int q_global = q_offset + q_local;

    if (q_local >= T_q) return;

    extern __shared__ float smem[];
    float* K_tile = smem;
    float* V_tile = smem + BLOCK_K * D;

    const float* Q_ptr = Q + (bh * T_q + q_local) * D;
    float q_regs[MAX_D];
    float o_regs[MAX_D];
    for (int d = 0; d < D; ++d) { q_regs[d] = Q_ptr[d]; o_regs[d] = 0.0f; }

    float m_i = -1e30f;
    float l_i = 0.0f;

    const int num_k_blocks = (T_k + BLOCK_K - 1) / BLOCK_K;

    for (int kb = 0; kb < num_k_blocks; ++kb) {
        const int k_block_start  = kb * BLOCK_K;
        const int k_global_start = k_offset + k_block_start;

        if (is_causal && k_global_start > q_global) break;

        const int k_local = k_block_start + threadIdx.x;
        if (k_local < T_k) {
            const float* Kp = K + (bh * T_k + k_local) * D;
            const float* Vp = V + (bh * T_k + k_local) * D;
            float* Ks = K_tile + threadIdx.x * D;
            float* Vs = V_tile + threadIdx.x * D;
            for (int d = 0; d < D; ++d) { Ks[d] = Kp[d]; Vs[d] = Vp[d]; }
        } else {
            float* Ks = K_tile + threadIdx.x * D;
            float* Vs = V_tile + threadIdx.x * D;
            for (int d = 0; d < D; ++d) { Ks[d] = 0.0f; Vs[d] = 0.0f; }
        }
        __syncthreads();

        const int tile_size = (k_block_start + BLOCK_K <= T_k)
                            ? BLOCK_K : (T_k - k_block_start);

        float scores[BLOCK_K];
        float tile_max = -1e30f;

        for (int j = 0; j < tile_size; ++j) {
            const int kg = k_global_start + j;
            if (is_causal && kg > q_global) { scores[j] = -1e30f; continue; }
            float dot = 0.0f;
            const float* Kj = K_tile + j * D;
            for (int d = 0; d < D; ++d) dot += q_regs[d] * Kj[d];
            scores[j] = dot * scale;
            if (scores[j] > tile_max) tile_max = scores[j];
        }
        for (int j = tile_size; j < BLOCK_K; ++j) scores[j] = -1e30f;

        if (tile_max <= -1e29f) { __syncthreads(); continue; }

        const float m_new     = (tile_max > m_i) ? tile_max : m_i;
        const float exp_shift = expf(m_i - m_new);
        for (int d = 0; d < D; ++d) o_regs[d] *= exp_shift;

        float l_inc = 0.0f;
        for (int j = 0; j < tile_size; ++j) {
            if (scores[j] <= -1e29f) continue;
            const float p_j = expf(scores[j] - m_new);
            l_inc += p_j;
            const float* Vj = V_tile + j * D;
            for (int d = 0; d < D; ++d) o_regs[d] += p_j * Vj[d];
        }
        l_i = l_i * exp_shift + l_inc;
        m_i = m_new;
        __syncthreads();
    }

    const float inv_l = (l_i > 1e-30f) ? (1.0f / l_i) : 0.0f;
    float* O_ptr = O + (bh * T_q + q_local) * D;
    for (int d = 0; d < D; ++d) O_ptr[d] = o_regs[d] * inv_l;
    LSE[bh * T_q + q_local] = (l_i > 1e-30f) ? (m_i + logf(l_i)) : -1e30f;
}

// ---------------------------------------------------------------------------
// launch_flash_attn_fwd_f32
//
// Dispatcher: selects the HEAD_DIM template specialisation and launches the
// kernel with the correct shared-memory allocation.
// D=32, 64, 128 use the unrolled template path.
// All other D <= 512 use the generic runtime-loop path.
// ---------------------------------------------------------------------------
void launch_flash_attn_fwd_f32(
    const float* Q,
    const float* K,
    const float* V,
    float*       O,
    float*       LSE,
    int BH, int T_q, int T_k, int D,
    float scale,
    bool  is_causal,
    int   q_offset,
    int   k_offset)
{
    if (D > MAX_D) {
        return;
    }
    const dim3 grid(( T_q + BLOCK_Q - 1) / BLOCK_Q, BH);
    const dim3 block(BLOCK_Q);
    const int  smem_bytes = 2 * BLOCK_K * D * sizeof(float);

    switch (D) {
    case 32:
        flash_attn_fwd_kernel<32><<<grid, block, smem_bytes>>>(
            Q, K, V, O, LSE, T_q, T_k, scale, is_causal, q_offset, k_offset);
        break;
    case 64:
        flash_attn_fwd_kernel<64><<<grid, block, smem_bytes>>>(
            Q, K, V, O, LSE, T_q, T_k, scale, is_causal, q_offset, k_offset);
        break;
    case 128:
        flash_attn_fwd_kernel<128><<<grid, block, smem_bytes>>>(
            Q, K, V, O, LSE, T_q, T_k, scale, is_causal, q_offset, k_offset);
        break;
    default:
        // Generic path: any D <= 512, no unroll.
        // Set extended shared memory limit so kernels with D > 64 can run.
        cudaFuncSetAttribute(flash_attn_fwd_kernel_generic,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_bytes);
        flash_attn_fwd_kernel_generic<<<grid, block, smem_bytes>>>(
            Q, K, V, O, LSE, T_q, T_k, D, scale, is_causal, q_offset, k_offset);
        break;
    }
}
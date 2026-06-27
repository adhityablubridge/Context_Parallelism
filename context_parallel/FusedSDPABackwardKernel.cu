// ---------------------------------------------------------------------------
// FusedSDPABackwardKernel.cu
//
// CUDA implementation of the FlashAttention backward pass.
//
// Avoids materialising the [T_q x T_k] attention matrix in HBM by tiling
// over K/V (for dQ) and Q (for dK/dV), recomputing P_ij on-the-fly from
// the saved merged LSE.
//
// Algorithm (Dao et al. 2022, FlashAttention backward):
//
//   D_i     = rowsum(dO_i * O_i)                    [precomputed in dQ kernel]
//   P_ij    = exp(s_ij - LSE_i)                      s_ij = dot(Q_i, K_j)*scale
//   dV_j   += P_ij * dO_i
//   ds_ij   = P_ij * (dot(dO_i, V_j) - D_i)
//   dQ_i   += ds_ij * scale * K_j
//   dK_j   += ds_ij * scale * Q_i
//
// Two kernels:
//   flash_attn_bwd_dq_kernel   -- one thread per Q row, sweeps K/V tiles.
//                                 Also writes D_buf[BH, T_q].
//   flash_attn_bwd_dkdv_kernel -- one thread per K row, sweeps Q tiles.
//                                 Reads D_buf.
//
// Both kernels launched in stream 0 (sequential).  No explicit sync between
// them -- same-stream ordering guarantees D_buf is ready before dK/dV reads.
//
// Template parameter HEAD_DIM is dispatched at launch for 32, 64, 128.
// ---------------------------------------------------------------------------

#include "FusedSDPABackwardKernel.h"
#include <cuda_runtime.h>
#include <mma.h>
#include <math.h>
#include <cstdio>

static constexpr int BLOCK_Q = 32;
static constexpr int BLOCK_K = 32;

// =====================================================================
// TF32 WMMA Backward — Constants
// =====================================================================
static constexpr int BWD_TC_BQ      = 32;
static constexpr int BWD_TC_BK      = 32;
static constexpr int BWD_TC_THREADS = 256;   // 8 warps
static constexpr int BWD_TC_WMMA_M  = 16;
static constexpr int BWD_TC_WMMA_N  = 16;
static constexpr int BWD_TC_WMMA_K  = 8;
static constexpr int BWD_TC_PAD     = 0;     // smem bank-conflict pad

// ---------------------------------------------------------------------------
// flash_attn_bwd_dq_kernel
//
// Thread threadIdx.x owns Q row q_local = blockIdx.x * BLOCK_Q + threadIdx.x.
// Sweeps all K/V tiles left-to-right (respecting causal mask with early exit).
// Writes dQ and D_buf.
// ---------------------------------------------------------------------------
template <int HEAD_DIM>
__global__ void flash_attn_bwd_dq_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ O,
    const float* __restrict__ dO,
    const float* __restrict__ LSE,
    float* __restrict__       dQ,
    float* __restrict__       D_buf,
    int T_q, int T_k,
    float scale,
    bool  is_causal,
    int   q_offset,
    int   k_offset)
{
    const int bh      = blockIdx.y;
    const int q_local = blockIdx.x * BLOCK_Q + threadIdx.x;

    if (q_local >= T_q) return;

    const int q_global = q_offset + q_local;

    // Shared memory: K_tile and V_tile
    extern __shared__ float smem[];
    float* K_tile = smem;
    float* V_tile = smem + BLOCK_K * HEAD_DIM;

    // Load Q, dO, O rows into registers
    const float* Q_ptr  = Q  + (bh * T_q + q_local) * HEAD_DIM;
    const float* dO_ptr = dO + (bh * T_q + q_local) * HEAD_DIM;
    const float* O_ptr  = O  + (bh * T_q + q_local) * HEAD_DIM;

    float q_regs[HEAD_DIM];
    float do_regs[HEAD_DIM];
    float dq_regs[HEAD_DIM];

#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        q_regs[d]  = Q_ptr[d];
        do_regs[d] = dO_ptr[d];
        dq_regs[d] = 0.0f;
    }

    // D_i = dot(dO_i, O_i) -- scalar, written to D_buf for dK/dV kernel
    float D_i = 0.0f;
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) D_i += do_regs[d] * O_ptr[d];

    D_buf[bh * T_q + q_local] = D_i;

    const float lse_i = LSE[bh * T_q + q_local];

    // Sweep over K/V tiles
    const int num_k_blocks = (T_k + BLOCK_K - 1) / BLOCK_K;

    for (int kb = 0; kb < num_k_blocks; ++kb) {
        const int k_block_start  = kb * BLOCK_K;
        const int k_global_start = k_offset + k_block_start;

        // Causal early exit: entire tile is strictly in the future
        if (is_causal && k_global_start > q_global) break;

        // Cooperative tile load: thread t loads K[t] and V[t]
        const int k_local_t = k_block_start + threadIdx.x;
        if (k_local_t < T_k) {
            const float* Kp = K + (bh * T_k + k_local_t) * HEAD_DIM;
            const float* Vp = V + (bh * T_k + k_local_t) * HEAD_DIM;
            float* Ks = K_tile + threadIdx.x * HEAD_DIM;
            float* Vs = V_tile + threadIdx.x * HEAD_DIM;
#pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) { Ks[d] = Kp[d]; Vs[d] = Vp[d]; }
        } else {
            float* Ks = K_tile + threadIdx.x * HEAD_DIM;
            float* Vs = V_tile + threadIdx.x * HEAD_DIM;
#pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) { Ks[d] = 0.0f; Vs[d] = 0.0f; }
        }
        __syncthreads();

        const int tile_size = (k_block_start + BLOCK_K <= T_k)
                            ? BLOCK_K : (T_k - k_block_start);

        for (int j = 0; j < tile_size; ++j) {
            const int k_global_j = k_global_start + j;
            if (is_causal && k_global_j > q_global) continue;

            const float* Kj = K_tile + j * HEAD_DIM;
            const float* Vj = V_tile + j * HEAD_DIM;

            // s_ij = dot(Q_i, K_j) * scale
            float dot_qk = 0.0f;
#pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) dot_qk += q_regs[d] * Kj[d];
            const float p_ij = expf(dot_qk * scale - lse_i);

            // dp_ij = dot(dO_i, V_j)
            float dp_ij = 0.0f;
#pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) dp_ij += do_regs[d] * Vj[d];

            // ds_ij = p_ij * (dp_ij - D_i)
            const float ds_scaled = p_ij * (dp_ij - D_i) * scale;

#pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) dq_regs[d] += ds_scaled * Kj[d];
        }

        __syncthreads();
    }

    // Write dQ
    float* dQ_ptr = dQ + (bh * T_q + q_local) * HEAD_DIM;
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) dQ_ptr[d] = dq_regs[d];
}

// ---------------------------------------------------------------------------
// flash_attn_bwd_dkdv_kernel
//
// Thread threadIdx.x owns K row k_local = blockIdx.x * BLOCK_K + threadIdx.x.
// Sweeps all Q tiles, accumulating dK_j and dV_j in registers.
// Reads D_buf (written by dQ kernel, guaranteed by same-stream ordering).
//
// Shared memory: Q_tile [BLOCK_Q*HEAD_DIM], dO_tile [BLOCK_Q*HEAD_DIM],
//                LSE_tile [BLOCK_Q], D_tile [BLOCK_Q].
// ---------------------------------------------------------------------------
template <int HEAD_DIM>
__global__ void flash_attn_bwd_dkdv_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ V,
    const float* __restrict__ dO,
    const float* __restrict__ LSE,
    const float* __restrict__ D_buf,
    float* __restrict__       dK,
    float* __restrict__       dV,
    int T_q, int T_k,
    float scale,
    bool  is_causal,
    int   q_offset,
    int   k_offset,
    const float* __restrict__ K)
{
    const int bh      = blockIdx.y;
    const int k_local = blockIdx.x * BLOCK_K + threadIdx.x;

    if (k_local >= T_k) return;

    const int k_global = k_offset + k_local;

    // Shared memory layout:
    //   [0               .. BLOCK_Q*HEAD_DIM) : Q_tile
    //   [BLOCK_Q*HEAD_DIM.. 2*BLOCK_Q*HD)     : dO_tile
    //   [2*BLOCK_Q*HD    .. 2*BLOCK_Q*HD+BQ)  : LSE_tile (floats)
    //   [2*BLOCK_Q*HD+BQ .. 2*BLOCK_Q*HD+2BQ) : D_tile  (floats)
    extern __shared__ float smem[];
    float* Q_tile   = smem;
    float* dO_tile  = smem + BLOCK_Q * HEAD_DIM;
    float* LSE_tile = smem + 2 * BLOCK_Q * HEAD_DIM;
    float* D_tile   = LSE_tile + BLOCK_Q;

    // Load K_j, V_j into registers (fixed across Q tiles)
    const float* K_ptr = K + (bh * T_k + k_local) * HEAD_DIM;
    const float* V_ptr = V + (bh * T_k + k_local) * HEAD_DIM;

    float k_regs[HEAD_DIM];
    float v_regs[HEAD_DIM];
    float dk_regs[HEAD_DIM];
    float dv_regs[HEAD_DIM];

#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        k_regs[d]  = K_ptr[d];
        v_regs[d]  = V_ptr[d];
        dk_regs[d] = 0.0f;
        dv_regs[d] = 0.0f;
    }

    const int num_q_blocks = (T_q + BLOCK_Q - 1) / BLOCK_Q;

    for (int qb = 0; qb < num_q_blocks; ++qb) {
        const int q_block_start  = qb * BLOCK_Q;
        const int q_global_start = q_offset + q_block_start;

        // Causal: skip Q tile if all Q rows are strictly before k_global
        // max q_global in tile = q_global_start + BLOCK_Q - 1
        if (is_causal && (q_global_start + BLOCK_Q - 1) < k_global) continue;

        // Cooperative tile load: thread t loads Q row and dO row at
        // q_block_start + threadIdx.x
        const int q_local_t = q_block_start + threadIdx.x;
        if (q_local_t < T_q) {
            const float* Qp  = Q  + (bh * T_q + q_local_t) * HEAD_DIM;
            const float* dOp = dO + (bh * T_q + q_local_t) * HEAD_DIM;
            float* Qt  = Q_tile  + threadIdx.x * HEAD_DIM;
            float* dOt = dO_tile + threadIdx.x * HEAD_DIM;
#pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) { Qt[d] = Qp[d]; dOt[d] = dOp[d]; }
            LSE_tile[threadIdx.x] = LSE[bh * T_q + q_local_t];
            D_tile[threadIdx.x]   = D_buf[bh * T_q + q_local_t];
        } else {
            float* Qt  = Q_tile  + threadIdx.x * HEAD_DIM;
            float* dOt = dO_tile + threadIdx.x * HEAD_DIM;
#pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) { Qt[d] = 0.0f; dOt[d] = 0.0f; }
            // Large LSE makes p_ij -> 0 for out-of-bounds rows
            LSE_tile[threadIdx.x] = 1e30f;
            D_tile[threadIdx.x]   = 0.0f;
        }
        __syncthreads();

        const int tile_size = (q_block_start + BLOCK_Q <= T_q)
                            ? BLOCK_Q : (T_q - q_block_start);

        for (int i = 0; i < tile_size; ++i) {
            const int q_global_i = q_global_start + i;
            // Causal: K position j attends Q position i only if k_global <= q_global_i
            if (is_causal && k_global > q_global_i) continue;

            const float* Qi  = Q_tile  + i * HEAD_DIM;
            const float* dOi = dO_tile + i * HEAD_DIM;
            const float  lse_i = LSE_tile[i];
            const float  D_i   = D_tile[i];

            // s_ij = dot(Q_i, k_j) * scale
            float dot_qk = 0.0f;
#pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) dot_qk += Qi[d] * k_regs[d];
            const float p_ij = expf(dot_qk * scale - lse_i);

            // dV_j += p_ij * dO_i
#pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) dv_regs[d] += p_ij * dOi[d];

            // dp_ij = dot(dO_i, v_j)
            float dp_ij = 0.0f;
#pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) dp_ij += dOi[d] * v_regs[d];

            // ds_ij = p_ij * (dp_ij - D_i)
            const float ds_scaled = p_ij * (dp_ij - D_i) * scale;

            // dK_j += ds_scaled * Q_i
#pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) dk_regs[d] += ds_scaled * Qi[d];
        }

        __syncthreads();
    }

    // Write dK_j, dV_j
    float* dK_ptr = dK + (bh * T_k + k_local) * HEAD_DIM;
    float* dV_ptr = dV + (bh * T_k + k_local) * HEAD_DIM;
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) { dK_ptr[d] = dk_regs[d]; dV_ptr[d] = dv_regs[d]; }
}

// ---------------------------------------------------------------------------
// Generic backward kernels (non-templated, any D <= 512)
// ---------------------------------------------------------------------------
static constexpr int MAX_D_BWD = 512;

__global__ void flash_attn_bwd_dq_kernel_generic(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ O,
    const float* __restrict__ dO,
    const float* __restrict__ LSE,
    float* __restrict__       dQ,
    float* __restrict__       D_buf,
    int T_q, int T_k, int D,
    float scale,
    bool  is_causal,
    int   q_offset,
    int   k_offset)
{
    const int bh      = blockIdx.y;
    const int q_local = blockIdx.x * BLOCK_Q + threadIdx.x;
    if (q_local >= T_q) return;
    const int q_global = q_offset + q_local;

    extern __shared__ float smem[];
    float* K_tile = smem;
    float* V_tile = smem + BLOCK_K * D;

    const float* Q_ptr  = Q  + (bh * T_q + q_local) * D;
    const float* dO_ptr = dO + (bh * T_q + q_local) * D;
    const float* O_ptr  = O  + (bh * T_q + q_local) * D;

    float q_regs[MAX_D_BWD];
    float do_regs[MAX_D_BWD];
    float dq_regs[MAX_D_BWD];
    for (int d = 0; d < D; ++d) { q_regs[d] = Q_ptr[d]; do_regs[d] = dO_ptr[d]; dq_regs[d] = 0.0f; }

    float D_i = 0.0f;
    for (int d = 0; d < D; ++d) D_i += do_regs[d] * O_ptr[d];
    D_buf[bh * T_q + q_local] = D_i;

    const float lse_i = LSE[bh * T_q + q_local];
    const int num_k_blocks = (T_k + BLOCK_K - 1) / BLOCK_K;

    for (int kb = 0; kb < num_k_blocks; ++kb) {
        const int k_block_start  = kb * BLOCK_K;
        const int k_global_start = k_offset + k_block_start;
        if (is_causal && k_global_start > q_global) break;

        const int k_local_t = k_block_start + threadIdx.x;
        if (k_local_t < T_k) {
            const float* Kp = K + (bh * T_k + k_local_t) * D;
            const float* Vp = V + (bh * T_k + k_local_t) * D;
            float* Ks = K_tile + threadIdx.x * D;
            float* Vs = V_tile + threadIdx.x * D;
            for (int d = 0; d < D; ++d) { Ks[d] = Kp[d]; Vs[d] = Vp[d]; }
        } else {
            float* Ks = K_tile + threadIdx.x * D;
            float* Vs = V_tile + threadIdx.x * D;
            for (int d = 0; d < D; ++d) { Ks[d] = 0.0f; Vs[d] = 0.0f; }
        }
        __syncthreads();

        const int tile_size = (k_block_start + BLOCK_K <= T_k) ? BLOCK_K : (T_k - k_block_start);
        for (int j = 0; j < tile_size; ++j) {
            const int k_global_j = k_global_start + j;
            if (is_causal && k_global_j > q_global) continue;
            const float* Kj = K_tile + j * D;
            const float* Vj = V_tile + j * D;
            float dot_qk = 0.0f;
            for (int d = 0; d < D; ++d) dot_qk += q_regs[d] * Kj[d];
            const float p_ij = expf(dot_qk * scale - lse_i);
            float dp_ij = 0.0f;
            for (int d = 0; d < D; ++d) dp_ij += do_regs[d] * Vj[d];
            const float ds_scaled = p_ij * (dp_ij - D_i) * scale;
            for (int d = 0; d < D; ++d) dq_regs[d] += ds_scaled * Kj[d];
        }
        __syncthreads();
    }

    float* dQ_ptr = dQ + (bh * T_q + q_local) * D;
    for (int d = 0; d < D; ++d) dQ_ptr[d] = dq_regs[d];
}

__global__ void flash_attn_bwd_dkdv_kernel_generic(
    const float* __restrict__ Q,
    const float* __restrict__ V,
    const float* __restrict__ dO,
    const float* __restrict__ LSE,
    const float* __restrict__ D_buf,
    float* __restrict__       dK,
    float* __restrict__       dV,
    int T_q, int T_k, int D,
    float scale,
    bool  is_causal,
    int   q_offset,
    int   k_offset,
    const float* __restrict__ K)
{
    const int bh      = blockIdx.y;
    const int k_local = blockIdx.x * BLOCK_K + threadIdx.x;
    if (k_local >= T_k) return;
    const int k_global = k_offset + k_local;

    // Shared: Q_tile [BLOCK_Q*D] + dO_tile [BLOCK_Q*D] + LSE_tile [BLOCK_Q] + D_tile [BLOCK_Q]
    extern __shared__ float smem[];
    float* Q_tile   = smem;
    float* dO_tile  = smem + BLOCK_Q * D;
    float* LSE_tile = smem + 2 * BLOCK_Q * D;
    float* D_tile   = LSE_tile + BLOCK_Q;

    const float* K_ptr = K + (bh * T_k + k_local) * D;
    const float* V_ptr = V + (bh * T_k + k_local) * D;

    float k_regs[MAX_D_BWD];
    float v_regs[MAX_D_BWD];
    float dk_regs[MAX_D_BWD];
    float dv_regs[MAX_D_BWD];
    for (int d = 0; d < D; ++d) { k_regs[d] = K_ptr[d]; v_regs[d] = V_ptr[d]; dk_regs[d] = 0.0f; dv_regs[d] = 0.0f; }

    const int num_q_blocks = (T_q + BLOCK_Q - 1) / BLOCK_Q;

    for (int qb = 0; qb < num_q_blocks; ++qb) {
        const int q_block_start  = qb * BLOCK_Q;
        const int q_global_start = q_offset + q_block_start;
        if (is_causal && (q_global_start + BLOCK_Q - 1) < k_global) continue;

        const int q_local_t = q_block_start + threadIdx.x;
        if (q_local_t < T_q) {
            const float* Qp  = Q  + (bh * T_q + q_local_t) * D;
            const float* dOp = dO + (bh * T_q + q_local_t) * D;
            float* Qt  = Q_tile  + threadIdx.x * D;
            float* dOt = dO_tile + threadIdx.x * D;
            for (int d = 0; d < D; ++d) { Qt[d] = Qp[d]; dOt[d] = dOp[d]; }
            LSE_tile[threadIdx.x] = LSE[bh * T_q + q_local_t];
            D_tile[threadIdx.x]   = D_buf[bh * T_q + q_local_t];
        } else {
            float* Qt  = Q_tile  + threadIdx.x * D;
            float* dOt = dO_tile + threadIdx.x * D;
            for (int d = 0; d < D; ++d) { Qt[d] = 0.0f; dOt[d] = 0.0f; }
            LSE_tile[threadIdx.x] = 1e30f;
            D_tile[threadIdx.x]   = 0.0f;
        }
        __syncthreads();

        const int tile_size = (q_block_start + BLOCK_Q <= T_q) ? BLOCK_Q : (T_q - q_block_start);
        for (int i = 0; i < tile_size; ++i) {
            const int q_global_i = q_global_start + i;
            if (is_causal && k_global > q_global_i) continue;
            const float* Qi  = Q_tile  + i * D;
            const float* dOi = dO_tile + i * D;
            const float  lse_i = LSE_tile[i];
            const float  Di    = D_tile[i];
            float dot_qk = 0.0f;
            for (int d = 0; d < D; ++d) dot_qk += Qi[d] * k_regs[d];
            const float p_ij = expf(dot_qk * scale - lse_i);
            for (int d = 0; d < D; ++d) dv_regs[d] += p_ij * dOi[d];
            float dp_ij = 0.0f;
            for (int d = 0; d < D; ++d) dp_ij += dOi[d] * v_regs[d];
            const float ds_scaled = p_ij * (dp_ij - Di) * scale;
            for (int d = 0; d < D; ++d) dk_regs[d] += ds_scaled * Qi[d];
        }
        __syncthreads();
    }

    float* dK_ptr = dK + (bh * T_k + k_local) * D;
    float* dV_ptr = dV + (bh * T_k + k_local) * D;
    for (int d = 0; d < D; ++d) { dK_ptr[d] = dk_regs[d]; dV_ptr[d] = dv_regs[d]; }
}

// =====================================================================
// TF32 WMMA Backward — D precompute
//   D[i] = dot(dO[i], O[i]).  One warp per row, 32 lanes split HeadDim.
// =====================================================================
template <int HD>
__global__ void bwd_tc_precompute_D(const float* dO, const float* O, float* D_buf, int T) {
    const int bh  = blockIdx.y;
    // Each warp processes one row
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * (blockDim.x / 32) + warp_id;
    
    if (row >= T) return;
    
    const int64_t off = (int64_t)bh * T * HD + (int64_t)row * HD;
    float s = 0.0f;
    #pragma unroll
    for (int d = lane; d < HD; d += 32) s += dO[off + d] * O[off + d];
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
    if (lane == 0) D_buf[(long long)bh * T + row] = s;
}

// =====================================================================
// TF32 WMMA Backward — dQ kernel  (Q-tile-centric, sweeps KV tiles)
//
// Per KV tile:
//   1. Concurrent: Score = Q@K^T (warps 0-3), dpV = dO@V^T (warps 4-7)
//   2. Scalar: P = exp(score*scale - LSE), ds = P*(dpV - D)*scale
//   3. WMMA: dQ_acc += ds @ K  (register accumulation, all 8 warps)
//
// Requires: HD % 16 == 0, HD <= 64  (single-pass: 2*(HD/16) <= 8 warps)
// =====================================================================
template <int HD>
__global__ void flash_attn_bwd_dq_kernel_tc(
    const float* __restrict__ Q,  const float* __restrict__ K,
    const float* __restrict__ V,  const float* __restrict__ dO,
    const float* __restrict__ LSE, const float* __restrict__ D_buf,
    float* __restrict__ dQ,
    int T_q, int T_k, float scale, bool is_causal, int q_off, int k_off)
{
    using namespace nvcuda;
    constexpr int HP       = HD + BWD_TC_PAD;
    constexpr int SC_N     = BWD_TC_BK / BWD_TC_WMMA_N;          // 2
    constexpr int SC_K     = HD       / BWD_TC_WMMA_K;           // HD/8
    constexpr int RPW      = BWD_TC_BQ / (BWD_TC_THREADS / 32);  // 4 rows/warp

    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, tid = threadIdx.x;
    const int bh = blockIdx.y;
    const int64_t qi_blk = (int64_t)blockIdx.x * BWD_TC_BQ;

    const float* Q_b = Q + bh*T_q*HD,  *K_b = K + bh*T_k*HD;
    const float* V_b = V + bh*T_k*HD,  *dO_b = dO + bh*T_q*HD;
    const float* LSE_b = LSE + bh*T_q,  *D_b = D_buf + bh*T_q;
    float* dQ_b = dQ + bh*T_q*HD;

    extern __shared__ float sm[];
    float* s_q  = sm;
    float* s_do = s_q  + BWD_TC_BQ*HP;
    float* s_k  = s_do + BWD_TC_BQ*HP;
    float* s_v  = s_k  + BWD_TC_BK*HP;
    float* s_A  = s_v  + BWD_TC_BK*HP;          // [BQ×BK] scores/P/ds
    float* s_B  = s_A  + BWD_TC_BQ*BWD_TC_BK;   // [BQ×BK] dpV
    float* s_ls = s_B  + BWD_TC_BQ*BWD_TC_BK;   // [BQ]
    float* s_d  = s_ls + BWD_TC_BQ;              // [BQ]
    float* s_dq = s_d  + BWD_TC_BQ;              // [BQ×HP] writeback

    const int aq = (int)min((int64_t)BWD_TC_BQ, (int64_t)T_q - qi_blk);
    if (aq <= 0) return;

    // Load Q, dO, LSE, D  (Q and dO persistent across KV tiles)
    for (int i = tid; i < BWD_TC_BQ*HD; i += BWD_TC_THREADS) {
        int r = i/HD, d = i%HD;  int64_t g = qi_blk + r;
        s_q [r*HP+d] = (g<T_q) ? Q_b [g*HD+d] : 0.f;
        s_do[r*HP+d] = (g<T_q) ? dO_b[g*HD+d] : 0.f;
    }
    for (int i = tid; i < BWD_TC_BQ; i += BWD_TC_THREADS) {
        int64_t g = qi_blk + i;
        s_ls[i] = (g<T_q) ? LSE_b[g] : 1e30f;
        s_d [i] = (g<T_q) ? D_b  [g] : 0.f;
    }
    __syncthreads();

    // dQ WMMA register accumulators: 2x4 = 8 fragments (one per warp)
    // Output tile: [BQ=32 x HD=64] = [2*16 x 4*16]
    constexpr int DQ_N   = HD / BWD_TC_WMMA_N;         // 4
    constexpr int DQ_K   = BWD_TC_BK / BWD_TC_WMMA_K;  // 4
    const int dq_m = warp / DQ_N;   // 0 or 1
    const int dq_n = warp % DQ_N;   // 0..3
    wmma::fragment<wmma::accumulator, 16,16,8, float> dq_acc;
    wmma::fill_fragment(dq_acc, 0.f);

    const int64_t max_kj = is_causal
        ? max((int64_t)0, min((int64_t)q_off + qi_blk + aq - k_off, (int64_t)T_k))
        : (int64_t)T_k;

    // ---- Main loop over KV tiles ----
    for (int64_t kj = 0; kj < max_kj; kj += BWD_TC_BK) {
        int blen = (int)min((int64_t)BWD_TC_BK, (int64_t)T_k - kj);

        // Load K → s_k, V → s_v
        for (int i = tid; i < BWD_TC_BK*HD; i += BWD_TC_THREADS) {
            int r = i/HD, d = i%HD;  int64_t g = kj + r;
            s_k[r*HP+d] = (g<T_k) ? K_b[g*HD+d] : 0.f;
            s_v[r*HP+d] = (g<T_k) ? V_b[g*HD+d] : 0.f;
        }
        __syncthreads();

        // Concurrent GEMMs: warps 0-3 → Score(Q@K^T→s_A), warps 4-7 → dpV(dO@V^T→s_B)
        {
            const bool is_score = (warp < 4);
            int w4  = is_score ? warp : (warp - 4);
            int m_t = w4 / SC_N, n_t = w4 % SC_N;
            const float* srcA = is_score ? s_q  : s_do;
            const float* srcB = is_score ? s_k  : s_v;
            float*       dst  = is_score ? s_A  : s_B;

            wmma::fragment<wmma::accumulator, 16,16,8, float> acc;
            wmma::fill_fragment(acc, 0.f);
            #pragma unroll
            for (int k = 0; k < SC_K; ++k) {
                wmma::fragment<wmma::matrix_a, 16,16,8, wmma::precision::tf32, wmma::row_major> fa;
                wmma::fragment<wmma::matrix_b, 16,16,8, wmma::precision::tf32, wmma::col_major> fb;
                wmma::load_matrix_sync(fa, srcA + m_t*16*HP + k*8, HP);
                wmma::load_matrix_sync(fb, srcB + n_t*16*HP + k*8, HP);
                wmma::mma_sync(acc, fa, fb, acc);
            }
            wmma::store_matrix_sync(dst + m_t*16*BWD_TC_BK + n_t*16, acc, BWD_TC_BK, wmma::mem_row_major);
        }
        __syncthreads();   // s_A=scores, s_B=dpV

        // Scalar: P = exp(score*scale - LSE),  ds = P*(dpV - D)*scale → s_A
        for (int r = 0; r < RPW; ++r) {
            int row = warp*RPW + r;
            int64_t qi_g = qi_blk + row;
            bool ok = (qi_g < T_q) && (lane < blen);
            if (ok) {
                float sv = s_A[row*BWD_TC_BK + lane] * scale;
                if (is_causal && (k_off + kj + lane) > (q_off + qi_g)) sv = -1e30f;
                float p  = (sv > -1e20f) ? expf(sv - s_ls[row]) : 0.f;
                float ds = p * (s_B[row*BWD_TC_BK + lane] - s_d[row]) * scale;
                s_A[row*BWD_TC_BK + lane] = ds;
            } else {
                s_A[row*BWD_TC_BK + lane] = 0.f;
            }
        }
        __syncthreads();   // s_A = ds

        // dQ WMMA accumulation: dq_acc += ds[BQ×BK] @ K[BK×HD]
        #pragma unroll
        for (int k = 0; k < DQ_K; ++k) {
            wmma::fragment<wmma::matrix_a, 16,16,8, wmma::precision::tf32, wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b, 16,16,8, wmma::precision::tf32, wmma::row_major> fb;
            wmma::load_matrix_sync(fa, s_A + dq_m*16*BWD_TC_BK + k*8, BWD_TC_BK);
            wmma::load_matrix_sync(fb, s_k + k*8*HP + dq_n*16, HP);
            wmma::mma_sync(dq_acc, fa, fb, dq_acc);
        }
        __syncthreads();   // before next KV tile
    }

    // Write dQ: fragment → s_dq → global
    for (int i = tid; i < BWD_TC_BQ*HD; i += BWD_TC_THREADS)
        s_dq[(i/HD)*HP + (i%HD)] = 0.f;
    __syncthreads();
    wmma::store_matrix_sync(s_dq + dq_m*16*HP + dq_n*16, dq_acc, HP, wmma::mem_row_major);
    __syncthreads();
    for (int i = tid; i < aq*HD; i += BWD_TC_THREADS) {
        int r = i/HD, d = i%HD;
        dQ_b[(qi_blk+r)*HD + d] = s_dq[r*HP + d];
    }
}

// =====================================================================
// TF32 WMMA Backward — dK/dV kernel  (K-tile-centric, sweeps Q tiles)
//
// Per Q tile:
//   1. Concurrent: Score = Q@K^T (warps 0-3), dpV = dO@V^T (warps 4-7)
//   2. Scalar: P, ds
//   3. WMMA: dV_acc += P^T @ dO   (register accumulation)
//   4. WMMA: dK_acc += ds^T @ Q   (register accumulation)
// =====================================================================
template <int HD>
__global__ void flash_attn_bwd_dkdv_kernel_tc(
    const float* __restrict__ Q,  const float* __restrict__ K,
    const float* __restrict__ V,  const float* __restrict__ dO,
    const float* __restrict__ LSE, const float* __restrict__ D_buf,
    float* __restrict__ dK, float* __restrict__ dV,
    int T_q, int T_k, float scale, bool is_causal, int q_off, int k_off)
{
    using namespace nvcuda;
    constexpr int HP     = HD + BWD_TC_PAD;
    constexpr int SC_N   = BWD_TC_BK / BWD_TC_WMMA_N;
    constexpr int SC_K   = HD       / BWD_TC_WMMA_K;
    constexpr int GR_N   = HD       / BWD_TC_WMMA_N;
    constexpr int GR_TOT = 2 * GR_N;
    constexpr int GR_K   = BWD_TC_BQ / BWD_TC_WMMA_K;  // 4
    constexpr int RPW    = BWD_TC_BQ / (BWD_TC_THREADS / 32);

    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, tid = threadIdx.x;
    const int bh = blockIdx.y;
    const int64_t kj_blk = (int64_t)blockIdx.x * BWD_TC_BK;

    const float* Q_b = Q + bh*T_q*HD,  *K_b = K + bh*T_k*HD;
    const float* V_b = V + bh*T_k*HD,  *dO_b = dO + bh*T_q*HD;
    const float* LSE_b = LSE + bh*T_q,  *D_b = D_buf + bh*T_q;
    float* dK_b = dK + bh*T_k*HD;
    float* dV_b = dV + bh*T_k*HD;

    extern __shared__ float sm[];
    float* s_k  = sm;
    float* s_v  = s_k  + BWD_TC_BK*HP;
    float* s_q  = s_v  + BWD_TC_BK*HP;
    float* s_do = s_q  + BWD_TC_BQ*HP;
    float* s_A  = s_do + BWD_TC_BQ*HP;
    float* s_B  = s_A  + BWD_TC_BQ*BWD_TC_BK;
    float* s_ls = s_B  + BWD_TC_BQ*BWD_TC_BK;
    float* s_d  = s_ls + BWD_TC_BQ;
    float* s_dk = s_d  + BWD_TC_BQ;
    float* s_dv = s_dk + BWD_TC_BK*HP;

    const int ak = (int)min((int64_t)BWD_TC_BK, (int64_t)T_k - kj_blk);
    if (ak <= 0) return;

    // Load K, V (persistent)
    for (int i = tid; i < BWD_TC_BK*HD; i += BWD_TC_THREADS) {
        int r = i/HD, d = i%HD;  int64_t g = kj_blk + r;
        s_k[r*HP+d] = (g<T_k) ? K_b[g*HD+d] : 0.f;
        s_v[r*HP+d] = (g<T_k) ? V_b[g*HD+d] : 0.f;
    }
    __syncthreads();

    // dK, dV accumulators in registers
    const int gr_tid = warp;
    const int gr_m   = gr_tid / GR_N;
    const int gr_n   = gr_tid % GR_N;
    wmma::fragment<wmma::accumulator, 16,16,8, float> dk_acc, dv_acc;
    wmma::fill_fragment(dk_acc, 0.f);
    wmma::fill_fragment(dv_acc, 0.f);

    // Sweep Q tiles
    const int num_q = (T_q + BWD_TC_BQ - 1) / BWD_TC_BQ;
    for (int qb = 0; qb < num_q; ++qb) {
        const int64_t qi_blk = (int64_t)qb * BWD_TC_BQ;
        const int64_t qi_end = qi_blk + BWD_TC_BQ - 1;
        // Causal skip: all Q rows before this K block
        if (is_causal && (q_off + qi_end) < (k_off + kj_blk)) continue;

        int aq = (int)min((int64_t)BWD_TC_BQ, (int64_t)T_q - qi_blk);

        // Load Q, dO, LSE, D
        for (int i = tid; i < BWD_TC_BQ*HD; i += BWD_TC_THREADS) {
            int r = i/HD, d = i%HD;  int64_t g = qi_blk + r;
            s_q [r*HP+d] = (g<T_q) ? Q_b [g*HD+d] : 0.f;
            s_do[r*HP+d] = (g<T_q) ? dO_b[g*HD+d] : 0.f;
        }
        for (int i = tid; i < BWD_TC_BQ; i += BWD_TC_THREADS) {
            int64_t g = qi_blk + i;
            s_ls[i] = (g<T_q) ? LSE_b[g] : 1e30f;
            s_d [i] = (g<T_q) ? D_b  [g] : 0.f;
        }
        __syncthreads();

        // Concurrent Score + dpV GEMMs
        {
            const bool is_sc = (warp < 4);
            int w4  = is_sc ? warp : (warp - 4);
            int m_t = w4 / SC_N, n_t = w4 % SC_N;
            const float* srcA = is_sc ? s_q  : s_do;
            const float* srcB = is_sc ? s_k  : s_v;
            float*       dst  = is_sc ? s_A  : s_B;
            wmma::fragment<wmma::accumulator, 16,16,8, float> acc;
            wmma::fill_fragment(acc, 0.f);
            #pragma unroll
            for (int k = 0; k < SC_K; ++k) {
                wmma::fragment<wmma::matrix_a, 16,16,8, wmma::precision::tf32, wmma::row_major> fa;
                wmma::fragment<wmma::matrix_b, 16,16,8, wmma::precision::tf32, wmma::col_major> fb;
                wmma::load_matrix_sync(fa, srcA + m_t*16*HP + k*8, HP);
                wmma::load_matrix_sync(fb, srcB + n_t*16*HP + k*8, HP);
                wmma::mma_sync(acc, fa, fb, acc);
            }
            wmma::store_matrix_sync(dst + m_t*16*BWD_TC_BK + n_t*16, acc, BWD_TC_BK, wmma::mem_row_major);
        }
        __syncthreads();

        // Scalar: P = exp(score*scale - LSE)
        for (int r = 0; r < RPW; ++r) {
            int row = warp*RPW + r;
            int64_t qi_g = qi_blk + row;
            bool ok = (qi_g < T_q) && (lane < ak);
            if (ok) {
                float sv = s_A[row*BWD_TC_BK + lane] * scale;
                if (is_causal && (k_off + kj_blk + lane) > (q_off + qi_g)) sv = -1e30f;
                s_A[row*BWD_TC_BK + lane] = (sv > -1e20f) ? expf(sv - s_ls[row]) : 0.f;
            } else {
                s_A[row*BWD_TC_BK + lane] = 0.f;
            }
        }
        __syncthreads();   // s_A = P

        // dV_acc += P^T[BK×BQ] @ dO[BQ×HD]
        //   load P col_major from s_A to get P^T as matrix_a
        if (gr_tid < GR_TOT) {
            #pragma unroll
            for (int k = 0; k < GR_K; ++k) {
                wmma::fragment<wmma::matrix_a, 16,16,8, wmma::precision::tf32, wmma::col_major> fa;
                wmma::fragment<wmma::matrix_b, 16,16,8, wmma::precision::tf32, wmma::row_major> fb;
                wmma::load_matrix_sync(fa, s_A + k*8*BWD_TC_BK + gr_m*16, BWD_TC_BK);
                wmma::load_matrix_sync(fb, s_do + k*8*HP + gr_n*16, HP);
                wmma::mma_sync(dv_acc, fa, fb, dv_acc);
            }
        }
        __syncthreads();   // ensure P fully read before overwrite

        // Scalar: ds = P * (dpV - D) * scale → overwrite s_A
        for (int r = 0; r < RPW; ++r) {
            int row = warp*RPW + r;
            float p  = s_A[row*BWD_TC_BK + lane];
            float dp = s_B[row*BWD_TC_BK + lane];
            s_A[row*BWD_TC_BK + lane] = p * (dp - s_d[row]) * scale;
        }
        __syncthreads();   // s_A = ds

        // dK_acc += ds^T[BK×BQ] @ Q[BQ×HD]
        if (gr_tid < GR_TOT) {
            #pragma unroll
            for (int k = 0; k < GR_K; ++k) {
                wmma::fragment<wmma::matrix_a, 16,16,8, wmma::precision::tf32, wmma::col_major> fa;
                wmma::fragment<wmma::matrix_b, 16,16,8, wmma::precision::tf32, wmma::row_major> fb;
                wmma::load_matrix_sync(fa, s_A + k*8*BWD_TC_BK + gr_m*16, BWD_TC_BK);
                wmma::load_matrix_sync(fb, s_q + k*8*HP + gr_n*16, HP);
                wmma::mma_sync(dk_acc, fa, fb, dk_acc);
            }
        }
        __syncthreads();
    }

    // Write dK, dV: fragments → smem → global
    for (int i = tid; i < BWD_TC_BK*HD; i += BWD_TC_THREADS) {
        s_dk[(i/HD)*HP + (i%HD)] = 0.f;
        s_dv[(i/HD)*HP + (i%HD)] = 0.f;
    }
    __syncthreads();
    if (gr_tid < GR_TOT) {
        wmma::store_matrix_sync(s_dk + gr_m*16*HP + gr_n*16, dk_acc, HP, wmma::mem_row_major);
        wmma::store_matrix_sync(s_dv + gr_m*16*HP + gr_n*16, dv_acc, HP, wmma::mem_row_major);
    }
    __syncthreads();
    for (int i = tid; i < ak*HD; i += BWD_TC_THREADS) {
        int r = i/HD, d = i%HD;
        dK_b[(kj_blk+r)*HD + d] = s_dk[r*HP + d];
        dV_b[(kj_blk+r)*HD + d] = s_dv[r*HP + d];
    }
}

// =====================================================================
// TF32 WMMA Backward — Unified kernel  (KV-tile-centric, sweeps Q tiles)
//
// Single-pass: computes dK, dV (register acc) and dQ (atomicAdd into
// fp32 scratch dQ_acc) without duplicate Score / dpV recomputation.
//
// Per Q tile:
//   1. Concurrent: Score = Q@K^T (warps 0-3), dpV = dO@V^T (warps 4-7)
//   2. Scalar: P = exp(score*scale - LSE)
//   3. WMMA: dV_acc += P^T @ dO          (register accumulation)
//   4. Scalar: ds = P*(dpV - D)*scale
//   5. WMMA: dK_acc += ds^T @ Q          (register accumulation)
//   6. WMMA: dQ_inc = ds @ K -> s_q      (s_q reused as temp buffer)
//   7. atomicAdd s_q -> dQ_acc
//
// Prerequisites:
//   - bwd_tc_precompute_D must run before this kernel (D_buf populated)
//   - dQ_acc must be zeroed (cudaMemset) before launch
// After: bwd_dq_epilogue copies dQ_acc -> dQ.
//
// Requires: HD % 16 == 0, HD <= 64
// =====================================================================
template <int HD>
__global__ void flash_attn_bwd_unified_qparallel_tc(
    const float* __restrict__ Q,   const float* __restrict__ K,
    const float* __restrict__ V,   const float* __restrict__ dO,
    const float* __restrict__ LSE, const float* __restrict__ D_buf,
    float* __restrict__ dQ,        float* __restrict__ dK,        float* __restrict__ dV,
    int T_q, int T_k, float scale, bool is_causal, int q_off, int k_off)
{
    using namespace nvcuda;
    constexpr int HP      = HD + BWD_TC_PAD;
    constexpr int SC_N    = BWD_TC_BK / BWD_TC_WMMA_N;  // 2
    constexpr int SC_K    = HD / BWD_TC_WMMA_K;         // 8
    constexpr int G_N     = HD / BWD_TC_WMMA_N;          // 4
    constexpr int G_K     = BWD_TC_BQ / BWD_TC_WMMA_K;  // 4
    constexpr int RPW     = BWD_TC_BQ / (BWD_TC_THREADS / 32); // 4

    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, tid = threadIdx.x;
    const int bh = blockIdx.y;
    const int64_t qi_blk = (int64_t)blockIdx.x * BWD_TC_BQ;

    const float* Q_b   = Q   + bh*T_q*HD;  const float* K_b   = K   + bh*T_k*HD;
    const float* V_b   = V   + bh*T_k*HD;  const float* dO_b  = dO  + bh*T_q*HD;
    const float* LSE_b = LSE + bh*T_q;     const float* D_b   = D_buf + bh*T_q;
    float* dQ_b = dQ + bh*T_q*HD;  float* dK_b = dK + bh*T_k*HD;  float* dV_b = dV + bh*T_k*HD;

    extern __shared__ float sm[];
    float* s_q  = sm;
    float* s_do = s_q  + BWD_TC_BQ*HP;
    float* s_k  = s_do + BWD_TC_BQ*HP;
    float* s_v  = s_k  + BWD_TC_BK*HP;
    float* s_A  = s_v  + BWD_TC_BK*HP;           // scores/P/ds [32x32]
    float* s_B  = s_A  + BWD_TC_BQ*BWD_TC_BK;    // dpV [32x32]
    float* s_ls = s_B  + BWD_TC_BQ*BWD_TC_BK;
    float* s_d  = s_ls + BWD_TC_BQ;
    float* s_wr = s_d  + BWD_TC_BQ;              // [32xHP] writeback for dK/dV

    const int aq = (int)min((int64_t)BWD_TC_BQ, (int64_t)T_q - qi_blk);
    if (aq <= 0) return;

    // Load Q, dO, LSE, D (persistent for this block)
    for (int i = tid; i < BWD_TC_BQ*HD; i += BWD_TC_THREADS) {
        int r = i/HD, d = i%HD; int64_t g = qi_blk + r;
        s_q [r*HP+d] = (g<T_q) ? Q_b [g*HD+d] : 0.f;
        s_do[r*HP+d] = (g<T_q) ? dO_b[g*HD+d] : 0.f;
    }
    for (int i = tid; i < BWD_TC_BQ; i += BWD_TC_THREADS) {
        int64_t g = qi_blk + i;
        s_ls[i] = (g<T_q) ? LSE_b[g] : 1e30f;
        s_d [i] = (g<T_q) ? D_b  [g] : 0.f;
    }
    __syncthreads();

    // dQ register accumulators: 32 rows x 64 cols -> 2x4 fragments
    wmma::fragment<wmma::accumulator, 16,16,8, float> dq_acc[2][4];
    for (int i=0; i<2; ++i) for (int j=0; j<4; ++j) wmma::fill_fragment(dq_acc[i][j], 0.f);

    const int64_t max_kj = is_causal 
        ? max((int64_t)0, min((int64_t)q_off + qi_blk + aq - k_off, (int64_t)T_k))
        : (int64_t)T_k;

    // Main loop over KV tiles (staggered to prevent atomic collision on dK/dV)
    const int num_k_steps = (max_kj + BWD_TC_BK - 1) / BWD_TC_BK;
    const int start_step = blockIdx.x % max(1, num_k_steps);

    for (int step = 0; step < num_k_steps; ++step) {
        int64_t kj_blk = ((start_step + step) % num_k_steps) * BWD_TC_BK;
        if (kj_blk >= max_kj) continue;
        int ak = (int)min((int64_t)BWD_TC_BK, (int64_t)T_k - kj_blk);
        // Load K, V
        for (int i = tid; i < BWD_TC_BK*HD; i += BWD_TC_THREADS) {
            int r = i/HD, d = i%HD; int64_t g = kj_blk + r;
            s_k[r*HP+d] = (g<T_k) ? K_b[g*HD+d] : 0.f;
            s_v[r*HP+d] = (g<T_k) ? V_b[g*HD+d] : 0.f;
        }
        __syncthreads();

        // 1. Concurrent Score + dpV
        {
            const bool is_sc = (warp < 4);
            int w4 = is_sc ? warp : (warp-4);
            int m_t = w4/SC_N, n_t = w4%SC_N;
            const float* srcA = is_sc ? s_q : s_do;
            const float* srcB = is_sc ? s_k : s_v;
            float* dst = is_sc ? s_A : s_B;
            wmma::fragment<wmma::accumulator, 16,16,8, float> acc;
            wmma::fill_fragment(acc, 0.f);
            #pragma unroll
            for (int k=0; k<SC_K; ++k) {
                wmma::fragment<wmma::matrix_a, 16,16,8, wmma::precision::tf32, wmma::row_major> fa;
                wmma::fragment<wmma::matrix_b, 16,16,8, wmma::precision::tf32, wmma::col_major> fb;
                wmma::load_matrix_sync(fa, srcA + m_t*16*HP + k*8, HP);
                wmma::load_matrix_sync(fb, srcB + n_t*16*HP + k*8, HP);
                wmma::mma_sync(acc, fa, fb, acc);
            }
            wmma::store_matrix_sync(dst + m_t*16*BWD_TC_BK + n_t*16, acc, BWD_TC_BK, wmma::mem_row_major);
        }
        __syncthreads();

        // 2. Scalar P/ds
        for (int r = 0; r < RPW; ++r) {
            int row = warp*RPW + r; int64_t qi_g = qi_blk + row;
            if (row < aq) {
                float sv = s_A[row*BWD_TC_BK + lane] * scale;
                if (is_causal && (k_off + kj_blk + lane) > (q_off + qi_g)) sv = -1e30f;
                float p = (sv > -1e30f) ? expf(sv - s_ls[row]) : 0.f;
                s_A[row*BWD_TC_BK + lane] = p; // Store P for dV calculation
                s_B[row*BWD_TC_BK + lane] = p * (s_B[row*BWD_TC_BK + lane] - s_d[row]) * scale; // Store ds for dQ/dK calculation
            }
        }
        __syncthreads();

        // 3. dQ Update (Register Accumulation)
        {
            // Warps 0-7 each handle 1 dQ fragment (32x64 output / 16x16 fragment = 8 fragments)
            int m_f = warp / G_N, n_f = warp % G_N;
            #pragma unroll
            for (int k=0; k<4; ++k) { // BK=32 / 8 = 4
                wmma::fragment<wmma::matrix_a, 16,16,8, wmma::precision::tf32, wmma::row_major> fa;
                wmma::fragment<wmma::matrix_b, 16,16,8, wmma::precision::tf32, wmma::row_major> fb;
                wmma::load_matrix_sync(fa, s_B + m_f*16*BWD_TC_BK + k*8, BWD_TC_BK);
                wmma::load_matrix_sync(fb, s_k + k*8*HP + n_f*16, HP);
                wmma::mma_sync(dq_acc[m_f][n_f], fa, fb, dq_acc[m_f][n_f]);
            }
        }

        // 4. dK Update (atomicAdd)
        {
            // Compute dK_tile = ds^T @ Q -> s_wr
            int m_f = warp / G_N, n_f = warp % G_N;
            wmma::fragment<wmma::accumulator, 16,16,8, float> acc;
            wmma::fill_fragment(acc, 0.f);
            #pragma unroll
            for (int k=0; k<G_K; ++k) {
                wmma::fragment<wmma::matrix_a, 16,16,8, wmma::precision::tf32, wmma::col_major> fa;
                wmma::fragment<wmma::matrix_b, 16,16,8, wmma::precision::tf32, wmma::row_major> fb;
                wmma::load_matrix_sync(fa, s_B + k*8*BWD_TC_BK + m_f*16, BWD_TC_BK);
                wmma::load_matrix_sync(fb, s_q + k*8*HP + n_f*16, HP);
                wmma::mma_sync(acc, fa, fb, acc);
            }
            __syncthreads();
            wmma::store_matrix_sync(s_wr + m_f*16*HP + n_f*16, acc, HP, wmma::mem_row_major);
            __syncthreads();
            // dK global atomicAdd
            for (int i = tid; i < ak*HD; i += BWD_TC_THREADS) {
                int r = i/HD, d = i%HD;
                atomicAdd(&dK_b[(kj_blk+r)*HD + d], s_wr[r*HP + d]);
            }
        }

        // 5. dV Update (atomicAdd)
        {
            // Compute dV_tile = P^T @ dO -> s_wr
            int m_f = warp / G_N, n_f = warp % G_N;
            wmma::fragment<wmma::accumulator, 16,16,8, float> acc;
            wmma::fill_fragment(acc, 0.f);
            #pragma unroll
            for (int k=0; k<G_K; ++k) {
                wmma::fragment<wmma::matrix_a, 16,16,8, wmma::precision::tf32, wmma::col_major> fa;
                wmma::fragment<wmma::matrix_b, 16,16,8, wmma::precision::tf32, wmma::row_major> fb;
                wmma::load_matrix_sync(fa, s_A + k*8*BWD_TC_BK + m_f*16, BWD_TC_BK);
                wmma::load_matrix_sync(fb, s_do + k*8*HP + n_f*16, HP);
                wmma::mma_sync(acc, fa, fb, acc);
            }
            __syncthreads();
            wmma::store_matrix_sync(s_wr + m_f*16*HP + n_f*16, acc, HP, wmma::mem_row_major);
            __syncthreads();
            // dV global atomicAdd
            for (int i = tid; i < ak*HD; i += BWD_TC_THREADS) {
                int r = i/HD, d = i%HD;
                atomicAdd(&dV_b[(kj_blk+r)*HD + d], s_wr[r*HP + d]);
            }
        }
        __syncthreads();
    }

    // Write final accumulated dQ from registers
    int m_f = warp / G_N, n_f = warp % G_N;
    wmma::store_matrix_sync(s_q + m_f*16*HP + n_f*16, dq_acc[m_f][n_f], HP, wmma::mem_row_major);
    __syncthreads();
    for (int i = tid; i < aq*HD; i += BWD_TC_THREADS) {
        int r = i/HD, d = i%HD;
        dQ_b[(qi_blk+r)*HD + d] = s_q[r*HP + d];
    }
}

// ---------------------------------------------------------------------------
// launch_flash_attn_bwd_f32
// ---------------------------------------------------------------------------
void launch_flash_attn_bwd_f32(
    const float* Q,
    const float* K,
    const float* V,
    const float* O,
    const float* dO,
    const float* LSE,
    float*       dQ,
    float*       dK,
    float*       dV,
    float*       D_buf,
    int BH, int T_q, int T_k, int D,
    float scale,
    bool  is_causal,
    int   q_offset,
    int   k_offset)
{
    // ---- TF32 TC path: two-kernel FlashAttention-2 structure ----
    // Uses flash_attn_bwd_dq_kernel_tc  (Q-parallel, dQ in register accumulators)
    // and  flash_attn_bwd_dkdv_kernel_tc (K-parallel, dK/dV in register accumulators).
    // No atomicAdd -- each output tile is owned by exactly one block.
    // Enabled for D that are multiples of 16 up to 64 (WMMA [16,16,8] requirement).
    if (D % 16 == 0 && D <= 64) {
        const int warps_per_blk = BWD_TC_THREADS / 32;  // 8

        // Step 1: Precompute D[i] = dot(dO[i], O[i]) into D_buf
        {
            const dim3 grid_d((T_q + warps_per_blk - 1) / warps_per_blk, BH);
            const dim3 block_d(BWD_TC_THREADS);
            switch (D) {
            case 16: bwd_tc_precompute_D<16><<<grid_d, block_d>>>(dO, O, D_buf, T_q); break;
            case 32: bwd_tc_precompute_D<32><<<grid_d, block_d>>>(dO, O, D_buf, T_q); break;
            case 48: bwd_tc_precompute_D<48><<<grid_d, block_d>>>(dO, O, D_buf, T_q); break;
            case 64: bwd_tc_precompute_D<64><<<grid_d, block_d>>>(dO, O, D_buf, T_q); break;
            }
        }

        // Step 2: dQ kernel (Q-parallel, register accumulation, direct write)
        {
            const dim3 grid_q((T_q + BWD_TC_BQ - 1) / BWD_TC_BQ, BH);
            const dim3 block_q(BWD_TC_THREADS);
            const size_t smem_dq = (size_t)(
                3 * BWD_TC_BQ * (D + BWD_TC_PAD) +
                2 * BWD_TC_BK * (D + BWD_TC_PAD) +
                2 * BWD_TC_BQ * BWD_TC_BK +
                2 * BWD_TC_BQ
            ) * sizeof(float);

#define LAUNCH_TC_DQ(HD) \
            do { \
                auto* k = flash_attn_bwd_dq_kernel_tc<HD>; \
                cudaError_t e1 = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_dq); \
                if (e1 != cudaSuccess) printf("[TC_DQ] FuncSetAttr err %d: %s\n", (int)e1, cudaGetErrorString(e1)); \
                k<<<grid_q, block_q, smem_dq>>>(Q, K, V, dO, LSE, D_buf, dQ, \
                    T_q, T_k, scale, is_causal, q_offset, k_offset); \
                cudaError_t e2 = cudaGetLastError(); \
                if (e2 != cudaSuccess) printf("[TC_DQ] Launch err %d: %s\n", (int)e2, cudaGetErrorString(e2)); \
            } while (0)

            switch (D) {
            case 16: LAUNCH_TC_DQ(16); break;
            case 32: LAUNCH_TC_DQ(32); break;
            case 48: LAUNCH_TC_DQ(48); break;
            case 64: LAUNCH_TC_DQ(64); break;
            }
#undef LAUNCH_TC_DQ
        }

        // Step 3: dK/dV TC kernel (K-parallel, register accumulation)
        {
            const dim3 grid_kv((T_k + BWD_TC_BK - 1) / BWD_TC_BK, BH);
            const dim3 block_kv(BWD_TC_THREADS);
            const size_t smem_dkdv = (size_t)(
                2 * BWD_TC_BK * (D + BWD_TC_PAD) +   // s_k, s_v
                2 * BWD_TC_BQ * (D + BWD_TC_PAD) +   // s_q, s_do
                2 * BWD_TC_BQ * BWD_TC_BK +           // s_A, s_B
                2 * BWD_TC_BQ +                       // s_ls, s_d
                2 * BWD_TC_BK * (D + BWD_TC_PAD)      // s_dk, s_dv
            ) * sizeof(float);

#define LAUNCH_TC_DKDV(HD) \
            do { \
                auto* k = flash_attn_bwd_dkdv_kernel_tc<HD>; \
                cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_dkdv); \
                k<<<grid_kv, block_kv, smem_dkdv>>>(Q, K, V, dO, LSE, D_buf, dK, dV, \
                    T_q, T_k, scale, is_causal, q_offset, k_offset); \
            } while (0)

            switch (D) {
            case 16: LAUNCH_TC_DKDV(16); break;
            case 32: LAUNCH_TC_DKDV(32); break;
            case 48: LAUNCH_TC_DKDV(48); break;
            case 64: LAUNCH_TC_DKDV(64); break;
            }
#undef LAUNCH_TC_DKDV
        }
        return;
    }


    // ---- Scalar FP32 fallback ----
    // --- Kernel 1: dQ (also computes D_buf) ---
    {
        const dim3 grid((T_q + BLOCK_Q - 1) / BLOCK_Q, BH);
        const dim3 block(BLOCK_Q);
        const int  smem = 2 * BLOCK_K * D * sizeof(float);  // K_tile + V_tile

        switch (D) {
        case 32:
            flash_attn_bwd_dq_kernel<32><<<grid, block, smem>>>(
                Q, K, V, O, dO, LSE, dQ, D_buf,
                T_q, T_k, scale, is_causal, q_offset, k_offset);
            break;
        case 64:
            flash_attn_bwd_dq_kernel<64><<<grid, block, smem>>>(
                Q, K, V, O, dO, LSE, dQ, D_buf,
                T_q, T_k, scale, is_causal, q_offset, k_offset);
            break;
        case 128:
            flash_attn_bwd_dq_kernel<128><<<grid, block, smem>>>(
                Q, K, V, O, dO, LSE, dQ, D_buf,
                T_q, T_k, scale, is_causal, q_offset, k_offset);
            break;
        default:
            // Set extended shared memory limit for large D (e.g. D=384 needs 96KB).
            cudaFuncSetAttribute(flash_attn_bwd_dq_kernel_generic,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 smem);
            flash_attn_bwd_dq_kernel_generic<<<grid, block, smem>>>(
                Q, K, V, O, dO, LSE, dQ, D_buf,
                T_q, T_k, D, scale, is_causal, q_offset, k_offset);
            break;
        }
    }

    // --- Kernel 2: dK, dV (reads D_buf written by Kernel 1) ---
    // Same stream 0 -- sequential by default, D_buf is ready.
    {
        const dim3 grid((T_k + BLOCK_K - 1) / BLOCK_K, BH);
        const dim3 block(BLOCK_K);
        // Q_tile [BLOCK_Q*D] + dO_tile [BLOCK_Q*D] + LSE_tile [BLOCK_Q] + D_tile [BLOCK_Q]
        const int  smem = (2 * BLOCK_Q * D + 2 * BLOCK_Q) * sizeof(float);

        switch (D) {
        case 32:
            flash_attn_bwd_dkdv_kernel<32><<<grid, block, smem>>>(
                Q, V, dO, LSE, D_buf, dK, dV,
                T_q, T_k, scale, is_causal, q_offset, k_offset, K);
            break;
        case 64:
            flash_attn_bwd_dkdv_kernel<64><<<grid, block, smem>>>(
                Q, V, dO, LSE, D_buf, dK, dV,
                T_q, T_k, scale, is_causal, q_offset, k_offset, K);
            break;
        case 128:
            flash_attn_bwd_dkdv_kernel<128><<<grid, block, smem>>>(
                Q, V, dO, LSE, D_buf, dK, dV,
                T_q, T_k, scale, is_causal, q_offset, k_offset, K);
            break;
        default:
            // Set extended shared memory limit for large D (e.g. D=384 needs 96KB).
            cudaFuncSetAttribute(flash_attn_bwd_dkdv_kernel_generic,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 smem);
            flash_attn_bwd_dkdv_kernel_generic<<<grid, block, smem>>>(
                Q, V, dO, LSE, D_buf, dK, dV,
                T_q, T_k, D, scale, is_causal, q_offset, k_offset, K);
            break;
        }
    }
}

// In-place addition kernel for zero-allocation accumulations
__global__ void add_inplace_kernel(float* dst, const float* src, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        dst[idx] += src[idx];
    }
}

extern "C" void add_inplace_f32(float* dst, const float* src, int numel) {
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    add_inplace_kernel<<<blocks, threads>>>(dst, src, numel);
}

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

#include "device/DeviceCore.h"   // OwnTensor::cuda::getCurrentStream()

// ============================================================================
// Context-Parallel aware copy of arch/AttentionBackward_sm89.cu (Ada / sm89).
//
// Mirrors the TI->CP transformation already applied to the generic backward
// kernel (context_parallel/AttentionBackward.cu): adds separate
// T_q / T_k plus q_offset / k_offset so the Ada PTX-MMA "exp12" backward can
// serve CP ring-attention sub-chunks.
//
// The MMA / ldmatrix / scatter core is identical to the arch kernel (tile-local,
// independent of global position). Only bounds and causal masking change:
//   single T  -> T_q (Q side) / T_k (K/V side)
//   causal    -> global positions: q_global = q_offset + qi, k_global = k_offset + kj
//
// Gradient accumulation contract (matches the generic CP backward):
//   - dQ uses atomicAdd across the KV-tile blocks of a single call, so the
//     launcher zeroes dQ before launch. Each call therefore produces this
//     step's dQ contribution; the CP ring driver accumulates across steps.
//   - dK/dV are written by direct assignment (exp12 is KV-tile-centric: exactly
//     one block owns each KV row and accumulates its Q-loop contributions in
//     registers), so no memset is needed for them.
//
// Wrapped in namespace OwnTensor::cp to avoid ODR conflicts with the arch
// kernel (namespace OwnTensor).
// ============================================================================

namespace OwnTensor {
namespace cp {

static constexpr int SM89CP_BWD_WARP_SZ     = 32;
static constexpr int SM89CP_BWD_NUM_THREADS = 256;   // 8 warps
static constexpr int SM89CP_BWD_BLOCK_M_D   = 8;

// CP backward params (strided I/O, last dim stride=1; D flat [BH, T_q]).
struct CPBwdParamsSm89 {
    const float* Q;
    const float* K;
    const float* V;
    const float* dO;
    const float* O;
    const float* LSE;
    const float* D;     // flat scratch [BH * T_q], stride = T_q per (b,h)
    float* dQ;
    float* dK;
    float* dV;
    int B;
    int nh;
    int T_q;
    int T_k;
    int q_offset;
    int k_offset;
    int64_t q_strideB,  q_strideM,  q_strideH;
    int64_t k_strideB,  k_strideM,  k_strideH;
    int64_t v_strideB,  v_strideM,  v_strideH;
    int64_t do_strideB, do_strideM, do_strideH;
    int64_t o_strideB,  o_strideM,  o_strideH;
    int64_t lse_strideB, lse_strideH;
    int64_t dq_strideB, dq_strideM, dq_strideH;
    int64_t dk_strideB, dk_strideM, dk_strideH;
    int64_t dv_strideB, dv_strideM, dv_strideH;
    float scale;
    bool is_causal;
};

__inline__ __device__ float sm89cp_bwd_warp_sum(float val) {
    #pragma unroll
    for (int offset = SM89CP_BWD_WARP_SZ / 2; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// ── PTX mma.sync.aligned.m16n8k8 (TF32->f32 accumulate) ──────────────────────
__device__ __forceinline__
void sm89cp_bwd_mma_tf32(float& d0, float& d1, float& d2, float& d3,
                         uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                         uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// ── Precompute D = rowsum(dO ⊙ O) — rows bounded by T_q, D flat [BH, T_q] ─────
template<int HeadDim>
__global__ void mem_efficient_bwd_precompute_D_sm89_cp(CPBwdParamsSm89 params)
{
    const float* __restrict__ dO = params.dO;
    const float* __restrict__ O  = params.O;
    const int T_q = params.T_q;
    const int nh  = params.nh;

    constexpr int LocalN = (HeadDim + SM89CP_BWD_WARP_SZ - 1) / SM89CP_BWD_WARP_SZ;

    const int bh      = blockIdx.y;
    const int warp_id = threadIdx.x / SM89CP_BWD_WARP_SZ;
    const int lane_id = threadIdx.x % SM89CP_BWD_WARP_SZ;
    const int b       = bh / nh;
    const int h       = bh - b * nh;

    const int base_row = blockIdx.x * (SM89CP_BWD_BLOCK_M_D * 2) + warp_id * 2;
    const int row0     = base_row;
    const int row1     = base_row + 1;
    const bool v0      = (row0 < T_q);
    const bool v1      = (row1 < T_q);

    const float* dO_bh = dO + b * params.do_strideB + h * params.do_strideH;
    const float* O_bh  = O  + b * params.o_strideB  + h * params.o_strideH;
    const long long dO_off0 = (long long)row0 * params.do_strideM;
    const long long dO_off1 = (long long)row1 * params.do_strideM;
    const long long O_off0  = (long long)row0 * params.o_strideM;
    const long long O_off1  = (long long)row1 * params.o_strideM;

    float sum0 = 0.f, sum1 = 0.f;
    #pragma unroll
    for (int i = 0; i < LocalN; ++i) {
        const int k = lane_id + i * SM89CP_BWD_WARP_SZ;
        if (k < HeadDim) {
            if (v0) sum0 += __ldg(&dO_bh[dO_off0 + k]) * __ldg(&O_bh[O_off0 + k]);
            if (v1) sum1 += __ldg(&dO_bh[dO_off1 + k]) * __ldg(&O_bh[O_off1 + k]);
        }
    }
    sum0 = sm89cp_bwd_warp_sum(sum0);
    sum1 = sm89cp_bwd_warp_sum(sum1);
    if (lane_id == 0) {
        float* D_bh = (float*)params.D + (long long)bh * params.T_q;
        if (v0) D_bh[row0] = sum0;
        if (v1) D_bh[row1] = sum1;
    }
}

// =============================================================================
// exp12 (CP): KV-tile-centric backward, TF32 MMA.
//   BM=32, BN=16. dK/dV in persistent register accumulators (direct assign);
//   dQ via atomicAdd. Causal uses global positions.
// =============================================================================
template <int HeadDim, bool Causal>
__launch_bounds__(256, 2)
__global__ void mem_efficient_bwd_unified_kernel_exp12_cp(CPBwdParamsSm89 params)
{
    constexpr int BlockN    = 16;
    constexpr int BM_WMMA   = 32;
    constexpr int BM_TILES  = BM_WMMA / 16;
    constexpr int HD_CHUNKS = HeadDim / 16;
    constexpr int HD_PAD    = HeadDim + 4;
    constexpr int BKN_PAD   = BlockN  + 4;
    constexpr int BM_PAD    = BM_WMMA + 4;
    constexpr float BWD_LOG2E = 1.4426950408889634074f;

    extern __shared__ float smem_f[];
    float* Q_sm    = smem_f;
    float* dO_sm   = Q_sm    + BM_WMMA * HD_PAD;
    float* Ks      = dO_sm   + BM_WMMA * HD_PAD;
    float* Vs      = Ks      + BlockN   * HD_PAD;
    float* ds_qd   = Vs      + BlockN   * HD_PAD;
    float* DPV_sm  = ds_qd   + BM_WMMA  * BKN_PAD;
    float* ds_kd   = DPV_sm  + BM_WMMA  * BKN_PAD;
    float* p_kd    = ds_kd   + BlockN   * BM_PAD;
    float* tile_st = p_kd    + BlockN   * BM_PAD;
    float* LSE_sm  = tile_st + BM_WMMA  * HD_PAD;
    float* D_sm    = LSE_sm  + BM_WMMA;

    const int bh           = blockIdx.y;
    const int kv_tile      = blockIdx.x;
    const int kv_base      = kv_tile * BlockN;
    const int kv_tile_size = min(BlockN, params.T_k - kv_base);
    if (kv_base >= params.T_k) return;

    const int warp_id = threadIdx.x / SM89CP_BWD_WARP_SZ;
    const int lane    = threadIdx.x % SM89CP_BWD_WARP_SZ;
    const int chunk   = warp_id % HD_CHUNKS;
    const int b       = bh / params.nh;
    const int h       = bh - b * params.nh;

    const float* Q_bh   = params.Q   + b * params.q_strideB   + h * params.q_strideH;
    const float* K_bh   = params.K   + b * params.k_strideB   + h * params.k_strideH;
    const float* V_bh   = params.V   + b * params.v_strideB   + h * params.v_strideH;
    const float* dO_bh  = params.dO  + b * params.do_strideB  + h * params.do_strideH;
    const float* LSE_bh = params.LSE + b * params.lse_strideB + h * params.lse_strideH;
    const float* D_bh   = params.D   + (long long)bh * params.T_q;
    float*       dQ_bh  = params.dQ  + b * params.dq_strideB  + h * params.dq_strideH;
    float*       dK_bh  = params.dK  + b * params.dk_strideB  + h * params.dk_strideH;
    float*       dV_bh  = params.dV  + b * params.dv_strideB  + h * params.dv_strideH;

    const int64_t q_sM  = params.q_strideM;
    const int64_t k_sM  = params.k_strideM;
    const int64_t v_sM  = params.v_strideM;
    const int64_t do_sM = params.do_strideM;
    const int64_t dq_sM = params.dq_strideM;
    const int64_t dk_sM = params.dk_strideM;
    const int64_t dv_sM = params.dv_strideM;

    auto loadA = [&](uint32_t r[4], int m_base, int k8,
                     const float* As, int stride) __attribute__((always_inline)) {
        const int row  = m_base + (lane & 15);
        const int col  = k8 + (lane >> 4) * 4;
        const uint32_t addr = (uint32_t)__cvta_generic_to_shared(&As[row * stride + col]);
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];"
            : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
        r[0] += 0x1000u; r[1] += 0x1000u; r[2] += 0x1000u; r[3] += 0x1000u;
    };

    auto loadB_nt = [&](uint32_t r[2], int n_base, int k8,
                        const float* Bs, int stride) __attribute__((always_inline)) {
        const int n = n_base + (lane >> 2);
        const int k = k8 + (lane & 3);
        r[0] = *(const uint32_t*)(&Bs[n * stride + k]);
        r[1] = *(const uint32_t*)(&Bs[n * stride + k + 4]);
        r[0] += 0x1000u; r[1] += 0x1000u;
    };

    auto loadB_nn = [&](uint32_t r[2], int n_base, int k8,
                        const float* Bs, int stride) __attribute__((always_inline)) {
        const int n = n_base + (lane >> 2);
        const int k = k8 + (lane & 3);
        r[0] = *(const uint32_t*)(&Bs[k       * stride + n]);
        r[1] = *(const uint32_t*)(&Bs[(k + 4) * stride + n]);
        r[0] += 0x1000u; r[1] += 0x1000u;
    };

    auto scatter = [&](const float d[4], float* dst, int r_base, int c_base,
                       int stride) __attribute__((always_inline)) {
        const int r0 = r_base + (lane >> 2), r1 = r0 + 8;
        const int c  = c_base + (lane & 3) * 2;
        dst[r0 * stride + c]     = d[0];
        dst[r0 * stride + c + 1] = d[1];
        dst[r1 * stride + c]     = d[2];
        dst[r1 * stride + c + 1] = d[3];
    };

    // Step 0: load K/V tiles into smem (K/V side bounded by T_k)
    for (int idx = threadIdx.x; idx < BlockN * HeadDim; idx += blockDim.x) {
        const int r = idx / HeadDim, k = idx % HeadDim;
        const int g = kv_base + r;
        Ks[r * HD_PAD + k] = (g < params.T_k) ? K_bh[g * k_sM + k] : 0.f;
        Vs[r * HD_PAD + k] = (g < params.T_k) ? V_bh[g * v_sM + k] : 0.f;
    }
    __syncthreads();

    float dk_acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float dv_acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    // Causal Q-loop start in LOCAL Q coords, using global positions:
    //   need q_global >= k_global of this tile's first row
    //   q_offset + q >= k_offset + kv_base  ->  q >= k_offset + kv_base - q_offset
    const int q_loop_start = Causal
        ? max(0, params.k_offset + kv_base - params.q_offset)
        : 0;

    for (int q_base = q_loop_start; q_base < params.T_q; q_base += BM_WMMA) {
        __syncthreads();  // (a)

        const int q_tile_size = min(BM_WMMA, params.T_q - q_base);

        // Load Q, dO, LSE, D for this Q tile (Q side bounded by T_q)
        for (int idx = threadIdx.x; idx < BM_WMMA * HeadDim; idx += blockDim.x) {
            const int r  = idx / HeadDim, k = idx % HeadDim;
            const int qi = q_base + r;
            const bool vq = (qi < params.T_q);
            Q_sm [r * HD_PAD + k] = vq ? Q_bh [qi * q_sM  + k] : 0.f;
            dO_sm[r * HD_PAD + k] = vq ? dO_bh[qi * do_sM + k] : 0.f;
        }
        if (threadIdx.x < BM_WMMA) {
            const int qi  = q_base + threadIdx.x;
            const bool vq = (qi < params.T_q);
            LSE_sm[threadIdx.x] = vq ? LSE_bh[qi] : 0.f;
            D_sm  [threadIdx.x] = vq ? D_bh  [qi] : 0.f;
        }
        __syncthreads();  // (b)

        // ── Phase A: warps 0-3 — S = Q@K^T and DPV = dO@V^T (NT layout) ──────
        if (warp_id < 2 * BM_TILES) {
            const int  rg       = warp_id % BM_TILES;
            const bool is_qk    = (warp_id < BM_TILES);
            const float* src_sm = (is_qk ? Q_sm  : dO_sm) + rg * 16 * HD_PAD;
            const float* kv_sm  =  is_qk ? Ks    : Vs;
            float*       dst_sm = (is_qk ? ds_qd : DPV_sm) + rg * 16 * BKN_PAD;

            float acc_l[4] = {0.f, 0.f, 0.f, 0.f};
            float acc_r[4] = {0.f, 0.f, 0.f, 0.f};
            uint32_t frA[4], frB_l[2], frB_r[2];

            constexpr int KS_TOTAL = 2 * HD_CHUNKS;
            #pragma unroll
            for (int ks = 0; ks < KS_TOTAL; ks++) {
                const int k8 = ks * 8;
                loadA   (frA,   0, k8, src_sm, HD_PAD);
                loadB_nt(frB_l, 0,     k8, kv_sm, HD_PAD);
                loadB_nt(frB_r, 8,     k8, kv_sm, HD_PAD);
                sm89cp_bwd_mma_tf32(acc_l[0], acc_l[1], acc_l[2], acc_l[3],
                             frA[0], frA[1], frA[2], frA[3], frB_l[0], frB_l[1]);
                sm89cp_bwd_mma_tf32(acc_r[0], acc_r[1], acc_r[2], acc_r[3],
                             frA[0], frA[1], frA[2], frA[3], frB_r[0], frB_r[1]);
            }
            scatter(acc_l, dst_sm, 0, 0, BKN_PAD);
            scatter(acc_r, dst_sm, 0, 8, BKN_PAD);
        }
        __syncthreads();  // (c1)

        // Scalar post-process: compute p, ds with GLOBAL causal positions.
        for (int elem = threadIdx.x; elem < BM_WMMA * BlockN; elem += blockDim.x) {
            const int qi_local = elem / BlockN;
            const int j_local  = elem % BlockN;
            const float raw_s  = ds_qd [qi_local * BKN_PAD + j_local];
            const float dpv    = DPV_sm[qi_local * BKN_PAD + j_local];
            const float L      = LSE_sm[qi_local];
            const float D_val  = D_sm  [qi_local];
            const int  qi_global = params.q_offset + q_base   + qi_local;
            const int  j_global  = params.k_offset + kv_base  + j_local;
            const bool qi_ok   = ((q_base + qi_local) < params.T_q);
            const bool j_ok    = (j_local < kv_tile_size);
            const bool cok     = !Causal || (j_global <= qi_global);
            float p = 0.f;
            if (qi_ok && j_ok && cok)
                p = exp2f(BWD_LOG2E * (raw_s * params.scale - L));
            const float ds = p * (dpv - D_val) * params.scale;
            ds_qd[qi_local * BKN_PAD + j_local] = ds;
            ds_kd[j_local  * BM_PAD  + qi_local] = ds;
            p_kd [j_local  * BM_PAD  + qi_local] = p;
        }
        __syncthreads();  // (c2)

        // ── Phase B1: warps 0..HD_CHUNKS-1 -> dQ (per tile, via tile_st);
        //              warps HD_CHUNKS.. -> dK (persistent dk_acc) ───────────
        {
            uint32_t frA[4], frB_l[2], frB_r[2];

            if (warp_id < HD_CHUNKS) {
                const int n_base = chunk * 16;
                float dq_l[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
                float dq_r[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};

                #pragma unroll
                for (int rg = 0; rg < BM_TILES; rg++) {
                    const float* a_base = ds_qd + rg * 16 * BKN_PAD;
                    #pragma unroll
                    for (int ks = 0; ks < 2; ks++) {
                        const int k8 = ks * 8;
                        loadA   (frA,   0, k8, a_base, BKN_PAD);
                        loadB_nn(frB_l, n_base,     k8, Ks, HD_PAD);
                        loadB_nn(frB_r, n_base + 8, k8, Ks, HD_PAD);
                        sm89cp_bwd_mma_tf32(dq_l[rg][0], dq_l[rg][1], dq_l[rg][2], dq_l[rg][3],
                                     frA[0], frA[1], frA[2], frA[3], frB_l[0], frB_l[1]);
                        sm89cp_bwd_mma_tf32(dq_r[rg][0], dq_r[rg][1], dq_r[rg][2], dq_r[rg][3],
                                     frA[0], frA[1], frA[2], frA[3], frB_r[0], frB_r[1]);
                    }
                    scatter(dq_l[rg], tile_st, rg * 16, n_base,     HD_PAD);
                    scatter(dq_r[rg], tile_st, rg * 16, n_base + 8, HD_PAD);
                }
            } else {
                const int n_base = chunk * 16;
                #pragma unroll
                for (int ks = 0; ks < BM_TILES * 2; ks++) {
                    const int k8 = ks * 8;
                    loadA   (frA,   0, k8, ds_kd, BM_PAD);
                    loadB_nn(frB_l, n_base,     k8, Q_sm, HD_PAD);
                    loadB_nn(frB_r, n_base + 8, k8, Q_sm, HD_PAD);
                    sm89cp_bwd_mma_tf32(dk_acc[0], dk_acc[1], dk_acc[2], dk_acc[3],
                                 frA[0], frA[1], frA[2], frA[3], frB_l[0], frB_l[1]);
                    sm89cp_bwd_mma_tf32(dk_acc[4], dk_acc[5], dk_acc[6], dk_acc[7],
                                 frA[0], frA[1], frA[2], frA[3], frB_r[0], frB_r[1]);
                }
            }
        }
        __syncthreads();  // (d)

        // dQ: atomicAdd this Q-tile contribution to global (bounded by T_q).
        for (int idx = threadIdx.x; idx < q_tile_size * HeadDim; idx += blockDim.x) {
            const int r = idx / HeadDim, k = idx % HeadDim;
            atomicAdd(&dQ_bh[(q_base + r) * dq_sM + k], tile_st[r * HD_PAD + k]);
        }

        // ── Phase B2: warps HD_CHUNKS.. -> dV (persistent dv_acc) ───────────
        if (warp_id >= HD_CHUNKS) {
            uint32_t frA[4], frB_l[2], frB_r[2];
            const int n_base = chunk * 16;
            #pragma unroll
            for (int ks = 0; ks < BM_TILES * 2; ks++) {
                const int k8 = ks * 8;
                loadA   (frA,   0, k8, p_kd, BM_PAD);
                loadB_nn(frB_l, n_base,     k8, dO_sm, HD_PAD);
                loadB_nn(frB_r, n_base + 8, k8, dO_sm, HD_PAD);
                sm89cp_bwd_mma_tf32(dv_acc[0], dv_acc[1], dv_acc[2], dv_acc[3],
                             frA[0], frA[1], frA[2], frA[3], frB_l[0], frB_l[1]);
                sm89cp_bwd_mma_tf32(dv_acc[4], dv_acc[5], dv_acc[6], dv_acc[7],
                             frA[0], frA[1], frA[2], frA[3], frB_r[0], frB_r[1]);
            }
        }
        // sync(a) at top of next iteration covers dO_sm and atomicAdd reads.
    } // end Q-tile loop

    // ── Final: scatter dk_acc/dv_acc -> tile_st -> global dK/dV (direct) ─────
    __syncthreads();

    if (warp_id >= HD_CHUNKS) {
        const int n_base = chunk * 16;
        scatter(dk_acc,     tile_st, 0, n_base,     HD_PAD);
        scatter(dk_acc + 4, tile_st, 0, n_base + 8, HD_PAD);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < kv_tile_size * HeadDim; idx += blockDim.x) {
        const int r = idx / HeadDim, k = idx % HeadDim;
        dK_bh[(kv_base + r) * dk_sM + k] = tile_st[r * HD_PAD + k];
    }
    __syncthreads();

    if (warp_id >= HD_CHUNKS) {
        const int n_base = chunk * 16;
        scatter(dv_acc,     tile_st, 0, n_base,     HD_PAD);
        scatter(dv_acc + 4, tile_st, 0, n_base + 8, HD_PAD);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < kv_tile_size * HeadDim; idx += blockDim.x) {
        const int r = idx / HeadDim, k = idx % HeadDim;
        dV_bh[(kv_base + r) * dv_sM + k] = tile_st[r * HD_PAD + k];
    }
}

// =============================================================================
// Public API (CP) — namespace OwnTensor::cp::cuda
// =============================================================================
namespace cuda {

// Strided variant. dQ is zeroed here (atomicAdd target within a call); dK/dV are
// written by direct assignment (exactly one block owns each KV row). For CP ring
// accumulation across steps the driver accumulates the returned per-step grads.
void mem_efficient_attn_backward_sm89_strided(
    const float* query, int64_t q_strideB, int64_t q_strideM, int64_t q_strideH,
    const float* key,   int64_t k_strideB, int64_t k_strideM, int64_t k_strideH,
    const float* value, int64_t v_strideB, int64_t v_strideM, int64_t v_strideH,
    const float* output,    int64_t o_strideB,  int64_t o_strideM,  int64_t o_strideH,
    const float* grad_output, int64_t do_strideB, int64_t do_strideM, int64_t do_strideH,
    const float* lse,   int64_t lse_strideB, int64_t lse_strideH,
    float* grad_query,  int64_t dq_strideB, int64_t dq_strideM, int64_t dq_strideH,
    float* grad_key,    int64_t dk_strideB, int64_t dk_strideM, int64_t dk_strideH,
    float* grad_value,  int64_t dv_strideB, int64_t dv_strideM, int64_t dv_strideH,
    float* D_buf,
    int64_t B, int64_t nh,
    int64_t T_q, int64_t T_k,
    int q_offset, int k_offset,
    int64_t hd,
    bool is_causal)
{
    if (hd % 16 != 0) {
        printf("cp::mem_efficient_attn_backward_sm89: hd=%d must be a multiple "
               "of 16\n", (int)hd);
        return;
    }

    const int BH = (int)(B * nh);
    dim3 block_cfg(SM89CP_BWD_NUM_THREADS);
    dim3 grid_D(((int)T_q + (SM89CP_BWD_BLOCK_M_D * 2) - 1) / (SM89CP_BWD_BLOCK_M_D * 2), BH);

    CPBwdParamsSm89 params{};
    params.Q  = query;
    params.K  = key;
    params.V  = value;
    params.dO = grad_output;
    params.O  = output;
    params.LSE = lse;
    params.D   = D_buf;
    params.dQ  = grad_query;
    params.dK  = grad_key;
    params.dV  = grad_value;
    params.B  = (int)B;
    params.nh = (int)nh;
    params.T_q = (int)T_q;
    params.T_k = (int)T_k;
    params.q_offset = q_offset;
    params.k_offset = k_offset;
    params.q_strideB  = q_strideB;  params.q_strideM  = q_strideM;  params.q_strideH  = q_strideH;
    params.k_strideB  = k_strideB;  params.k_strideM  = k_strideM;  params.k_strideH  = k_strideH;
    params.v_strideB  = v_strideB;  params.v_strideM  = v_strideM;  params.v_strideH  = v_strideH;
    params.do_strideB = do_strideB; params.do_strideM = do_strideM; params.do_strideH = do_strideH;
    params.o_strideB  = o_strideB;  params.o_strideM  = o_strideM;  params.o_strideH  = o_strideH;
    params.lse_strideB = lse_strideB; params.lse_strideH = lse_strideH;
    params.dq_strideB = dq_strideB; params.dq_strideM = dq_strideM; params.dq_strideH = dq_strideH;
    params.dk_strideB = dk_strideB; params.dk_strideM = dk_strideM; params.dk_strideH = dk_strideH;
    params.dv_strideB = dv_strideB; params.dv_strideM = dv_strideM; params.dv_strideH = dv_strideH;
    params.scale = 1.0f / sqrtf(static_cast<float>(hd));
    params.is_causal = is_causal;

    #define LAUNCH_CP_BWD_SM89(HD) \
    do { \
        constexpr int BN12 = 16, BM12 = 32; \
        const size_t shmem12 = \
            (2ULL * BM12 * ((HD) + 4)   \
           + 2ULL * BN12 * ((HD) + 4)   \
           + 2ULL * BM12 * ((BN12) + 4) \
           + 2ULL * BN12 * ((BM12) + 4) \
           + 1ULL * BM12 * ((HD) + 4)   \
           + 2ULL * BM12                \
            ) * sizeof(float); \
        const int kv12 = ((int)T_k + BN12 - 1) / BN12; \
        dim3 grid_kv12(kv12, BH); \
        cudaFuncSetAttribute( \
            mem_efficient_bwd_unified_kernel_exp12_cp<HD, false>, \
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem12); \
        cudaFuncSetAttribute( \
            mem_efficient_bwd_unified_kernel_exp12_cp<HD, true>, \
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem12); \
        cudaStream_t stream = ::OwnTensor::cuda::getCurrentStream(); \
        cudaMemsetAsync(params.dQ, 0, (size_t)BH * (int)T_q * (HD) * sizeof(float), stream); \
        mem_efficient_bwd_precompute_D_sm89_cp<HD><<<grid_D, block_cfg, 0, stream>>>(params); \
        if (is_causal) { \
            mem_efficient_bwd_unified_kernel_exp12_cp<HD, true> \
                <<<grid_kv12, block_cfg, shmem12, stream>>>(params); \
        } else { \
            mem_efficient_bwd_unified_kernel_exp12_cp<HD, false> \
                <<<grid_kv12, block_cfg, shmem12, stream>>>(params); \
        } \
    } while (0)

    switch ((int)hd) {
        case  16: LAUNCH_CP_BWD_SM89( 16); break;
        case  32: LAUNCH_CP_BWD_SM89( 32); break;
        case  48: LAUNCH_CP_BWD_SM89( 48); break;
        case  64: LAUNCH_CP_BWD_SM89( 64); break;
        case  80: LAUNCH_CP_BWD_SM89( 80); break;
        case  96: LAUNCH_CP_BWD_SM89( 96); break;
        case 128: LAUNCH_CP_BWD_SM89(128); break;
        case 160: LAUNCH_CP_BWD_SM89(160); break;
        case 192: LAUNCH_CP_BWD_SM89(192); break;
        case 256: LAUNCH_CP_BWD_SM89(256); break;
        default:
            printf("cp::mem_efficient_attn_backward_sm89: unsupported head_dim %d\n", (int)hd);
            break;
    }
    #undef LAUNCH_CP_BWD_SM89
}

} // namespace cuda
} // namespace cp
} // namespace OwnTensor

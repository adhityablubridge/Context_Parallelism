#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

// ============================================================================
// Context-Parallel aware copy of arch/AttentionForward_sm89.cu (Ada / sm89).
//
// Mirrors the TI->CP transformation already applied to the generic kernels
// (context_parallel/AttentionForward.cu): adds separate T_q / T_k
// plus q_offset / k_offset so the same Ada PTX-MMA kernel can serve CP
// ring-attention sub-chunks where Q and K/V come from different global
// positions.
//
// The MMA / ldmatrix / scatter-gather core is byte-for-byte identical to the
// arch kernel — those operate on tile-local smem and are independent of global
// sequence position. Only two things change:
//   1. Bounds: single T -> T_q (Q side) and T_k (K/V side).
//   2. Causal masking uses GLOBAL positions:
//        q_global = q_offset + (qi_block + row)
//        k_global = k_offset + (kj_block + col)
//      and masks when k_global > q_global.
//
// Wrapped in namespace OwnTensor::cp to avoid ODR conflicts with the arch
// kernel (namespace OwnTensor).
// ============================================================================

namespace OwnTensor {
namespace cp {

// CP forward params (strided I/O, last dim stride=1).
struct CPFwdParamsSm89 {
    const float* Q;
    const float* K;
    const float* V;
    float*       O;
    float*       LSE;
    int          nh;
    int          T_q;
    int          T_k;
    int          q_offset;
    int          k_offset;
    float        scale;
    bool         is_causal;
    float        dropout_p;
    const float* dropout_mask;
    int64_t q_strideB, q_strideM, q_strideH;
    int64_t k_strideB, k_strideM, k_strideH;
    int64_t v_strideB, v_strideM, v_strideH;
    int64_t o_strideB, o_strideM, o_strideH;
    int64_t lse_strideB, lse_strideH;
};

// ── cp.async helpers ────────────────────────────────────────────────────────
__device__ __forceinline__ void sm89cp_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}
template<int N>
__device__ __forceinline__ void sm89cp_cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}
__device__ __forceinline__ void sm89cp_cp_async_l16(
    void* smem_ptr, const void* global_ptr, bool pred)
{
    uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %2, 0;\n"
        "  @p cp.async.cg.shared.global [%0], [%1], 16;\n"
        "  @!p st.shared.v4.u32 [%0], {0,0,0,0};\n"
        "}\n"
        : : "r"(smem_addr), "l"(global_ptr), "r"((int)pred) : "memory");
}

// ── Constants ────────────────────────────────────────────────────────────────
static constexpr int SM89CP_NUM_THREADS = 256;  // 8 warps
static constexpr int SM89CP_WMMA_M      = 16;
static constexpr int SM89CP_WMMA_N      = 16;
static constexpr int SM89CP_WMMA_K      = 8;
static constexpr int SM89CP_SMEM_PAD    = 4;
static constexpr int SM89CP_BK_PAD      = 8;

// ── PTX mma.sync.aligned.m16n8k8 (TF32->f32 accumulate) ──────────────────────
__device__ __forceinline__
void sm89cp_mma_tf32_m16n8k8(float& d0, float& d1, float& d2, float& d3,
                             uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                             uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// =============================================================================
// CP forward kernel — identical MMA core to the arch kernel; only bounds and
// causal masking use T_q/T_k + q_offset/k_offset.
// =============================================================================
template<int HeadDim, int BQ_TILE, int BK_TILE, int MaxBlocksPerSM>
__global__ void __launch_bounds__(SM89CP_NUM_THREADS, MaxBlocksPerSM)
fused_attn_forward_kernel_tc_sm89_cp(CPFwdParamsSm89 params)
{
    const float* __restrict__ Q            = params.Q;
    const float* __restrict__ K            = params.K;
    const float* __restrict__ V            = params.V;
    float*       __restrict__ O            = params.O;
    float*       __restrict__ LSE          = params.LSE;
    const int     T_q                      = params.T_q;
    const int     T_k                      = params.T_k;
    const int     q_offset                 = params.q_offset;
    const int     k_offset                 = params.k_offset;
    const int     nh                       = params.nh;
    const float   scale                    = params.scale;
    const bool    is_causal                = params.is_causal;
    const float   dropout_p                = params.dropout_p;
    const float* __restrict__ dropout_mask = params.dropout_mask;

    static_assert(HeadDim % SM89CP_WMMA_N == 0, "HeadDim must be divisible by 16");
    static_assert(BQ_TILE % SM89CP_WMMA_M == 0, "BQ_TILE must be divisible by 16");
    static_assert(BK_TILE % 32           == 0, "BK_TILE must be a multiple of 32");

    constexpr int HD_PAD    = HeadDim + SM89CP_SMEM_PAD;
    constexpr int NUM_WARPS = SM89CP_NUM_THREADS / 32;

    constexpr int SCORE_M_TILES = BQ_TILE / SM89CP_WMMA_M;
    constexpr int SCORE_N_TILES = BK_TILE / SM89CP_WMMA_N;
    constexpr int SCORE_K_TILES = HeadDim / SM89CP_WMMA_K;
    constexpr int SCORE_TOTAL   = SCORE_M_TILES * SCORE_N_TILES;
    constexpr int BK_STRIDE     = BK_TILE + SM89CP_BK_PAD;

    constexpr int PV_M_TILES = BQ_TILE / SM89CP_WMMA_M;
    constexpr int PV_N_TILES = HeadDim / SM89CP_WMMA_N;
    constexpr int PV_TOTAL   = PV_M_TILES * PV_N_TILES;
    constexpr int PV_K_TILES = BK_TILE   / SM89CP_WMMA_K;
    constexpr int PV_PASSES  = (PV_TOTAL + NUM_WARPS - 1) / NUM_WARPS;

    constexpr int ROWS_PER_WARP   = BQ_TILE / NUM_WARPS;
    constexpr int COLS_PER_THREAD = BK_TILE / 32;

    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;
    const int tid     = threadIdx.x;

    const int64_t qi_block = (int64_t)blockIdx.x * BQ_TILE;
    const int     bnh      = blockIdx.y;
    const int     b        = bnh / nh;
    const int     h        = bnh - b * nh;

    const float* Q_bnh   = Q   + b * params.q_strideB   + h * params.q_strideH;
    const float* K_bnh   = K   + b * params.k_strideB   + h * params.k_strideH;
    const float* V_bnh   = V   + b * params.v_strideB   + h * params.v_strideH;
    float*       O_bnh   = O   + b * params.o_strideB   + h * params.o_strideH;
    float*       LSE_bnh = LSE + b * params.lse_strideB + h * params.lse_strideH;

    const int64_t q_sM = params.q_strideM;
    const int64_t k_sM = params.k_strideM;
    const int64_t v_sM = params.v_strideM;
    const int64_t o_sM = params.o_strideM;

    extern __shared__ float smem[];
    float* s_q       = smem;
    float* s_kv_base = s_q      + BQ_TILE * HD_PAD;
    float* s_kv[2]   = { s_kv_base, s_kv_base + BK_TILE * HD_PAD };
    float* s_scores  = s_kv_base + 2 * BK_TILE * HD_PAD;
    float* s_m       = s_scores  + BQ_TILE * BK_STRIDE;
    float* s_l       = s_m       + BQ_TILE;
    float* s_out     = s_l       + BQ_TILE;

    for (int i = tid; i < BQ_TILE; i += SM89CP_NUM_THREADS) {
        s_m[i] = -INFINITY;
        s_l[i] =  0.0f;
    }
    {
        const int vt = (BQ_TILE * HD_PAD) / 4;
        for (int i = tid; i < vt; i += SM89CP_NUM_THREADS)
            *(float4*)&s_out[i * 4] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // ── Step 1: Async load Q -> s_q (Q side bounded by T_q) ──────────────────
    {
        const int vt = (BQ_TILE * HeadDim) / 4;
        for (int i = tid; i < vt; i += SM89CP_NUM_THREADS) {
            const int q = (i * 4) / HeadDim, d = (i * 4) % HeadDim;
            sm89cp_cp_async_l16(&s_q[q * HD_PAD + d],
                                &Q_bnh[(qi_block + q) * q_sM + d],
                                (qi_block + q < T_q));
        }
        sm89cp_cp_async_commit();
    }

    const int actual_q = (int)(
        ((int64_t)BQ_TILE < ((int64_t)T_q - qi_block)) ? (int64_t)BQ_TILE
                                                       : ((int64_t)T_q - qi_block));
    if (actual_q <= 0) return;

    // Causal KV upper bound in LOCAL K coords, using global positions:
    //   last global Q pos in tile = q_offset + qi_block + actual_q - 1
    //   max local K index (exclusive) = (q_offset + qi_block + actual_q) - k_offset
    //   clamped to [0, T_k]
    int64_t max_kj;
    if (is_causal) {
        int64_t e = (int64_t)q_offset + qi_block + actual_q - (int64_t)k_offset;
        if (e < 0)            e = 0;
        if (e > (int64_t)T_k) e = (int64_t)T_k;
        max_kj = e;
    } else {
        max_kj = (int64_t)T_k;
    }

    // Pre-fetch first K tile (K side bounded by T_k)
    {
        const int vt = (BK_TILE * HeadDim) / 4;
        for (int i = tid; i < vt; i += SM89CP_NUM_THREADS) {
            const int k = (i * 4) / HeadDim, d = (i * 4) % HeadDim;
            sm89cp_cp_async_l16(&s_kv[0][k * HD_PAD + d],
                                &K_bnh[(int64_t)k * k_sM + d], (int64_t)k < T_k);
        }
        sm89cp_cp_async_commit();
    }
    sm89cp_cp_async_wait_group<0>();
    __syncthreads();

    auto loadA = [&](uint32_t r[4], int m_tile, int k8,
                     const float* As, int stride) __attribute__((always_inline)) {
        const int row  = m_tile * 16 + (lane & 15);
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

    auto gather = [&](float d[4], const float* src, int r_base, int c_base,
                      int stride) __attribute__((always_inline)) {
        const int r0 = r_base + (lane >> 2), r1 = r0 + 8;
        const int c  = c_base + (lane & 3) * 2;
        d[0] = src[r0 * stride + c];
        d[1] = src[r0 * stride + c + 1];
        d[2] = src[r1 * stride + c];
        d[3] = src[r1 * stride + c + 1];
    };

    // ── Step 2: Main KV-tile loop ─────────────────────────────────────────────
    for (int64_t kj_block = 0; kj_block < max_kj; kj_block += BK_TILE) {
        const int block_len = (int)(
            ((int64_t)BK_TILE < ((int64_t)T_k - kj_block)) ? (int64_t)BK_TILE
                                                           : ((int64_t)T_k - kj_block));
        const int64_t next_kj_block = kj_block + BK_TILE;
        const bool    has_next      = (next_kj_block < max_kj);

        // 2a: Async load V -> s_kv[1] (K/V side bounded by T_k)
        {
            const int vt = (BK_TILE * HeadDim) / 4;
            for (int i = tid; i < vt; i += SM89CP_NUM_THREADS) {
                const int v = (i * 4) / HeadDim, d = (i * 4) % HeadDim;
                const int64_t g = kj_block + v;
                sm89cp_cp_async_l16(&s_kv[1][v * HD_PAD + d], &V_bnh[g * v_sM + d], g < T_k);
            }
            sm89cp_cp_async_commit();
        }

        // 2b: Score GEMM — Q[BQ×HD] @ K[BK×HD]^T -> s_scores[BQ×BK]
        for (int tile_idx = warp_id; tile_idx < SCORE_TOTAL; tile_idx += NUM_WARPS) {
            const int m_tile = tile_idx / SCORE_N_TILES;
            const int n_tile = tile_idx % SCORE_N_TILES;

            float acc_l[4] = {0.f, 0.f, 0.f, 0.f};
            float acc_r[4] = {0.f, 0.f, 0.f, 0.f};
            uint32_t frA[4], frB_l[2], frB_r[2];

            #pragma unroll
            for (int k = 0; k < SCORE_K_TILES; k++) {
                const int k8 = k * SM89CP_WMMA_K;
                loadA   (frA,   m_tile,          k8, s_q,     HD_PAD);
                loadB_nt(frB_l, n_tile * 16,     k8, s_kv[0], HD_PAD);
                loadB_nt(frB_r, n_tile * 16 + 8, k8, s_kv[0], HD_PAD);
                sm89cp_mma_tf32_m16n8k8(acc_l[0], acc_l[1], acc_l[2], acc_l[3],
                                 frA[0], frA[1], frA[2], frA[3], frB_l[0], frB_l[1]);
                sm89cp_mma_tf32_m16n8k8(acc_r[0], acc_r[1], acc_r[2], acc_r[3],
                                 frA[0], frA[1], frA[2], frA[3], frB_r[0], frB_r[1]);
            }
            scatter(acc_l, s_scores, m_tile * 16, n_tile * 16,     BK_STRIDE);
            scatter(acc_r, s_scores, m_tile * 16, n_tile * 16 + 8, BK_STRIDE);
        }
        __syncthreads();

        // 2c: Online softmax — causal mask uses GLOBAL positions.
        {
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                const int     row       = warp_id * ROWS_PER_WARP + r;
                const int     qi_local  = (int)qi_block + row;
                const int     qi_global = q_offset + qi_local;
                const bool    qi_valid  = (qi_local < T_q);

                float cached_score[COLS_PER_THREAD];
                float row_max = -INFINITY;
                #pragma unroll
                for (int j = 0; j < COLS_PER_THREAD; ++j) {
                    const int col      = j * 32 + lane;
                    const int k_global = k_offset + (int)kj_block + col;
                    float v = (col < block_len && qi_valid)
                              ? s_scores[row * BK_STRIDE + col] * scale : -INFINITY;
                    if (is_causal && k_global > qi_global) v = -INFINITY;
                    cached_score[j] = v;
                    row_max = fmaxf(row_max, v);
                }
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffff, row_max, off));

                const float m_old = s_m[row];
                const float m_new = qi_valid ? fmaxf(m_old, row_max) : m_old;
                const float alpha = (m_old == -INFINITY) ? 0.0f
                                  : (m_old == m_new)     ? 1.0f
                                  :                        expf(m_old - m_new);

                #pragma unroll
                for (int d_base = 0; d_base < HeadDim; d_base += 64) {
                    const int d = d_base + lane * 2;
                    if (d < HeadDim) {
                        float2* ptr = (float2*)&s_out[row * HD_PAD + d];
                        float2 vo = *ptr;
                        vo.x *= alpha; vo.y *= alpha;
                        *ptr = vo;
                    }
                }

                float row_sum = 0.0f;
                #pragma unroll
                for (int j = 0; j < COLS_PER_THREAD; ++j) {
                    const int col = j * 32 + lane;
                    float exp_s = 0.0f;
                    if (cached_score[j] > -INFINITY && qi_valid && m_new > -INFINITY) {
                        exp_s = expf(cached_score[j] - m_new);
                        if (dropout_p > 0.0f && dropout_mask != nullptr) {
                            exp_s *= dropout_mask[
                                ((int64_t)(bnh * T_q + qi_local)) * T_k
                                + ((int)kj_block + col)];
                        }
                    }
                    s_scores[row * BK_STRIDE + col] = exp_s;
                    row_sum += exp_s;
                }
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    row_sum += __shfl_xor_sync(0xffffffff, row_sum, off);

                if (lane == 0) {
                    s_l[row] = alpha * s_l[row] + row_sum;
                    s_m[row] = m_new;
                }
            }
        }
        __syncthreads();

        // 2d: Pre-fetch next K tile (bounded by T_k)
        if (has_next) {
            const int vt = (BK_TILE * HeadDim) / 4;
            for (int i = tid; i < vt; i += SM89CP_NUM_THREADS) {
                const int k = (i * 4) / HeadDim, d = (i * 4) % HeadDim;
                const int64_t g = next_kj_block + k;
                sm89cp_cp_async_l16(&s_kv[0][k * HD_PAD + d], &K_bnh[g * k_sM + d], g < T_k);
            }
            sm89cp_cp_async_commit();
        }

        // 2e: Wait for V, compute P@V fused into s_out.
        //   V was committed (2a) as a cp.async group; the next-K prefetch (2d, if
        //   has_next) was committed AFTER it. wait_group<1> waits for "all groups
        //   except the 1 most recent", i.e. waits for V while leaving the next-K
        //   prefetch in flight -- correct ONLY while a next-K group exists.
        //   On the final KV iteration has_next is false, so no next-K group was
        //   committed and V is the single pending group; wait_group<1> would then
        //   wait for NOTHING, letting P@V read a not-yet-arrived V tile. That race
        //   is benign under exclusive occupancy but corrupts output under CP
        //   co-execution (the ring NCCL kernel delays the copy). Wait for ALL
        //   groups when there is no next-K to keep in flight.
        if (has_next) sm89cp_cp_async_wait_group<1>();
        else          sm89cp_cp_async_wait_group<0>();
        __syncthreads();

        for (int pass = 0; pass < PV_PASSES; ++pass) {
            const int tile_id = warp_id + pass * NUM_WARPS;
            if (tile_id < PV_TOTAL) {
                const int m_tile = tile_id / PV_N_TILES;
                const int n_tile = tile_id % PV_N_TILES;

                float acc_l[4], acc_r[4];
                gather(acc_l, s_out, m_tile * 16, n_tile * 16,     HD_PAD);
                gather(acc_r, s_out, m_tile * 16, n_tile * 16 + 8, HD_PAD);

                uint32_t frA[4], frB_l[2], frB_r[2];
                #pragma unroll
                for (int k = 0; k < PV_K_TILES; k++) {
                    const int k8 = k * SM89CP_WMMA_K;
                    loadA   (frA,   m_tile,          k8, s_scores, BK_STRIDE);
                    loadB_nn(frB_l, n_tile * 16,     k8, s_kv[1],  HD_PAD);
                    loadB_nn(frB_r, n_tile * 16 + 8, k8, s_kv[1],  HD_PAD);
                    sm89cp_mma_tf32_m16n8k8(acc_l[0], acc_l[1], acc_l[2], acc_l[3],
                                     frA[0], frA[1], frA[2], frA[3], frB_l[0], frB_l[1]);
                    sm89cp_mma_tf32_m16n8k8(acc_r[0], acc_r[1], acc_r[2], acc_r[3],
                                     frA[0], frA[1], frA[2], frA[3], frB_r[0], frB_r[1]);
                }
                scatter(acc_l, s_out, m_tile * 16, n_tile * 16,     HD_PAD);
                scatter(acc_r, s_out, m_tile * 16, n_tile * 16 + 8, HD_PAD);
            }
        }
        __syncthreads();
    } // end KV-tile loop

    // ── Step 3: Normalise and write O (rows bounded by actual_q) ─────────────
    {
        const int vt = (actual_q * HeadDim) / 4;
        for (int i = tid; i < vt; i += SM89CP_NUM_THREADS) {
            const int q = (i * 4) / HeadDim, d = (i * 4) % HeadDim;
            const float inv_l = (s_l[q] > 0.0f) ? (1.0f / s_l[q]) : 0.0f;
            float4* ptr = (float4*)&s_out[q * HD_PAD + d];
            float4 v = *ptr;
            v.x *= inv_l; v.y *= inv_l; v.z *= inv_l; v.w *= inv_l;
            *(float4*)&O_bnh[(qi_block + q) * o_sM + d] = v;
        }
    }
    __syncthreads();

    // ── Step 4: Write LSE (indexed by T_q) ────────────────────────────────────
    for (int i = tid; i < actual_q; i += SM89CP_NUM_THREADS) {
        const float m = s_m[i], l = s_l[i];
        LSE_bnh[qi_block + i] = (l > 0.0f) ? (m + logf(l)) : -INFINITY;
    }
}

// =============================================================================
// Shared-memory size helper
// =============================================================================
template<int BQ_TILE, int BK_TILE>
static size_t sm89cp_compute_smem(int64_t hd) {
    const size_t hd_pad = (size_t)hd + SM89CP_SMEM_PAD;
    return (2ULL * BQ_TILE * hd_pad
          + 2ULL * BK_TILE * hd_pad
          + (size_t)BQ_TILE * (BK_TILE + SM89CP_BK_PAD)
          + 2ULL * BQ_TILE)
         * sizeof(float);
}

// =============================================================================
// Public API (CP) — namespace OwnTensor::cp::cuda
// =============================================================================
namespace cuda {

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
    float dropout_p, const float* dropout_mask)
{
    if (hd % SM89CP_WMMA_N != 0) {
        printf("cp::mem_efficient_attn_forward_tc_sm89: hd=%d must be a multiple "
               "of 16\n", (int)hd);
        return;
    }

    int deviceId = 0;
    cudaGetDevice(&deviceId);
    int max_smem = 0;
    cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, deviceId);

    CPFwdParamsSm89 params{};
    params.Q = query; params.K = key; params.V = value;
    params.O = output; params.LSE = lse;
    params.nh = (int)nh;
    params.T_q = (int)T_q; params.T_k = (int)T_k;
    params.q_offset = q_offset; params.k_offset = k_offset;
    params.scale = 1.0f / sqrtf(static_cast<float>(hd));
    params.is_causal = is_causal;
    params.dropout_p = dropout_p; params.dropout_mask = dropout_mask;
    params.q_strideB = q_strideB; params.q_strideM = q_strideM; params.q_strideH = q_strideH;
    params.k_strideB = k_strideB; params.k_strideM = k_strideM; params.k_strideH = k_strideH;
    params.v_strideB = v_strideB; params.v_strideM = v_strideM; params.v_strideH = v_strideH;
    params.o_strideB = o_strideB; params.o_strideM = o_strideM; params.o_strideH = o_strideH;
    params.lse_strideB = lse_strideB; params.lse_strideH = lse_strideH;

    const int grid_y = (int)(B * nh);

    #define LAUNCH_SM89CP(HD, BQ_V, BK_V, MB, SMEM, GRID) \
    do { \
        auto* kptr = fused_attn_forward_kernel_tc_sm89_cp<HD, BQ_V, BK_V, MB>; \
        cudaFuncSetAttribute(kptr, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(SMEM)); \
        cudaFuncSetAttribute(kptr, cudaFuncAttributePreferredSharedMemoryCarveout, \
                             cudaSharedmemCarveoutMaxShared); \
        kptr<<<(GRID), SM89CP_NUM_THREADS, (SMEM)>>>(params); \
        return; \
    } while (0)

    if (hd <= 96) {
        const dim3 grid((int)((T_q + 63) / 64), grid_y);
        switch ((int)hd) {
            case  16: { const size_t s = sm89cp_compute_smem<64,64>(16); LAUNCH_SM89CP( 16,64,64,2,s,grid); }
            case  32: { const size_t s = sm89cp_compute_smem<64,32>(32); LAUNCH_SM89CP( 32,64,32,2,s,grid); }
            case  48: { const size_t s = sm89cp_compute_smem<64,64>(48); LAUNCH_SM89CP( 48,64,64,1,s,grid); }
            case  64: { const size_t s = sm89cp_compute_smem<64,64>(64); LAUNCH_SM89CP( 64,64,64,1,s,grid); }
            case  80: { const size_t s = sm89cp_compute_smem<64,32>(80); LAUNCH_SM89CP( 80,64,32,1,s,grid); }
            case  96: { const size_t s = sm89cp_compute_smem<64,32>(96); LAUNCH_SM89CP( 96,64,32,1,s,grid); }
            default: break;
        }
    }

    {
        const size_t s = sm89cp_compute_smem<32,32>(hd);
        if ((int)s > max_smem) {
            printf("cp::mem_efficient_attn_forward_tc_sm89: hd=%d requires %zu B "
                   "smem, device max is %d B. Aborting.\n", (int)hd, s, max_smem);
            return;
        }
        const dim3 grid((int)((T_q + 31) / 32), grid_y);
        switch ((int)hd) {
            case  16: LAUNCH_SM89CP( 16,32,32,1,s,grid);
            case  32: LAUNCH_SM89CP( 32,32,32,1,s,grid);
            case  48: LAUNCH_SM89CP( 48,32,32,1,s,grid);
            case  64: LAUNCH_SM89CP( 64,32,32,1,s,grid);
            case  80: LAUNCH_SM89CP( 80,32,32,1,s,grid);
            case  96: LAUNCH_SM89CP( 96,32,32,1,s,grid);
            case 128: LAUNCH_SM89CP(128,32,32,1,s,grid);
            default:
                printf("cp::mem_efficient_attn_forward_tc_sm89: unsupported hd=%d\n", (int)hd);
                break;
        }
    }

    #undef LAUNCH_SM89CP
}

} // namespace cuda
} // namespace cp
} // namespace OwnTensor

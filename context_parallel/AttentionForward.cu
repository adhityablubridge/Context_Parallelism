#include "ops/helpers/AttentionKernels.h"
#include "ops/helpers/KernelDispatch.h"
#include "ops/cuda/attention/AttentionCommon.cuh"
#include <cstdlib>  // std::getenv (CP_FWD_TI_SM89 gate)

// ============================================================================
// Context-Parallel aware copy of Tensor-Implementations AttentionForward.
//
// Adds separate T_q / T_k plus q_offset / k_offset so the same kernel can be
// used for CP ring-attention sub-chunks (full-Q x half-K, half-Q x full-K,
// cross-rank causal).
//
// Strided inputs supported: B/M/H per-tensor strides for Q/K/V/O/LSE so the
// caller can pass non-contiguous views (last dim must still be stride=1).
// Mirrors the strided convention from TI's AttentionForward.cu.
//
// Wrapped in namespace OwnTensor::cp to avoid ODR conflicts with the original
// TI kernels that are still linked in.
// ============================================================================

namespace OwnTensor {

// External sm89 entry point from Tensor-Implementations.
// Usable only when T_q == T_k && q_offset == 0 && k_offset == 0.
// Strided: per-tensor B/M/H strides for Q/K/V/O/LSE (last dim stride=1).
void fused_attn_forward_tc_sm89_cuda(
    const float* Q, const float* K, const float* V,
    float* O, float* LSE,
    int64_t B, int64_t nh, int64_t T, int64_t hd, float scale, bool is_causal,
    float dropout_p, const float* dropout_mask,
    int64_t q_strideB, int64_t q_strideM, int64_t q_strideH,
    int64_t k_strideB, int64_t k_strideM, int64_t k_strideH,
    int64_t v_strideB, int64_t v_strideM, int64_t v_strideH,
    int64_t o_strideB, int64_t o_strideM, int64_t o_strideH,
    int64_t lse_strideB, int64_t lse_strideH,
    int grid_y,
    cudaStream_t stream);

namespace cp {

// ============================================================================
// TF32 round-to-nearest before WMMA (precision backport from latest TI kernel).
// The default tf32 WMMA load truncates the fp32 mantissa toward zero (biased);
// adding half a TF32 ULP before truncation yields round-to-nearest, removing the
// systematic bias that compounds across layers and training steps. fp32 keeps 23
// mantissa bits, tf32 keeps 10 -> drop 13 -> half-ULP = (1<<12) = 0x1000.
// ============================================================================
template <typename FragType>
__device__ __forceinline__ void round_tf32_wmma_frag(FragType &frag) {
#pragma unroll
  for (int i = 0; i < (int)FragType::num_elements; i++)
    reinterpret_cast<uint32_t &>(frag.x[i]) += 0x1000u;
}

// ============================================================================
// cp.async helpers (copied verbatim from TI)
// ============================================================================

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::: "memory");
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}


__device__ __forceinline__ void cp_async_l16(void* smem_ptr, const void* global_ptr, bool pred) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %2, 0;\n"
        "  @p cp.async.ca.shared.global [%0], [%1], 16;\n"
        "  @!p st.shared.v4.u32 [%0], {0,0,0,0};\n"
        "}\n"
        : : "r"(smem_addr), "l"(global_ptr), "r"((int)pred) : "memory");
}

// ============================================================================
// CP forward params with per-tensor strides (last dim assumed stride=1)
// ============================================================================
struct CPFwdParams {
    const float* Q;
    const float* K;
    const float* V;
    float*       O;
    float*       LSE;
    int     B, nh;
    int     T_q, T_k;
    int     q_offset, k_offset;
    int64_t q_strideB, q_strideM, q_strideH;
    int64_t k_strideB, k_strideM, k_strideH;
    int64_t v_strideB, v_strideM, v_strideH;
    int64_t o_strideB, o_strideM, o_strideH;
    int64_t lse_strideB, lse_strideH;
    float   scale;
    bool    is_causal;
    float   dropout_p;
    const float* dropout_mask;
};

// ============================================================================
// Scalar forward kernel (CP)
// ============================================================================
//   qi_row         - local Q row index (0..T_q-1), used for storage / bounds
//   qi_global_pos  - global sequence position = q_offset + qi_row, used for causal
//   kj_row         - local K row index (0..T_k-1)
//   kj_global_pos  - k_offset + kj_row
//   Causal mask: attend iff kj_global_pos <= qi_global_pos
// ============================================================================

static constexpr int FWD_BQ              = 32;
static constexpr int FWD_BK              = 32;
static constexpr int FWD_THREADS_PER_ROW = 8;
static constexpr int FWD_NUM_THREADS     = FWD_BQ * FWD_THREADS_PER_ROW;
static constexpr int FWD_KJ_PER_THREAD   = FWD_BK / FWD_THREADS_PER_ROW;
static constexpr int FWD_SMEM_PAD        = 1;
static constexpr int FWD_SCORE_STRIDE    = FWD_BK + FWD_SMEM_PAD;

template<int HeadDim>
__global__ void fused_attn_forward_kernel(CPFwdParams params)
{
    const float* __restrict__ Q             = params.Q;
    const float* __restrict__ K             = params.K;
    const float* __restrict__ V             = params.V;
    float*       __restrict__ O             = params.O;
    float*       __restrict__ LSE           = params.LSE;
    const int64_t T_q                       = params.T_q;
    const int64_t T_k                       = params.T_k;
    const int     q_offset                  = params.q_offset;
    const int     k_offset                  = params.k_offset;
    const int     nh                        = params.nh;
    const float   scale                     = params.scale;
    const bool    is_causal                 = params.is_causal;
    const float   dropout_p                 = params.dropout_p;
    const float* __restrict__ dropout_mask  = params.dropout_mask;

    constexpr uint HD_PAD       = HeadDim + FWD_SMEM_PAD;
    constexpr uint D_PER_THREAD = (HeadDim + FWD_THREADS_PER_ROW - 1) / FWD_THREADS_PER_ROW;

    const uint qi_block = (int64_t)blockIdx.x * FWD_BQ;
    const uint bnh      = blockIdx.y;
    const uint tid      = threadIdx.x;
    const int  b        = (int)bnh / nh;
    const int  h        = (int)bnh - b * nh;

    const int qi_local  = tid / FWD_THREADS_PER_ROW;
    const int local_tid = tid % FWD_THREADS_PER_ROW;
    const int64_t qi_row        = qi_block + qi_local;
    const int64_t qi_global_pos = (int64_t)q_offset + qi_row;
    const bool qi_valid = (qi_row < T_q);

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
    float* s_q      = smem;
    float* s_kv     = s_q      + FWD_BQ * HD_PAD;
    float* s_scores = s_kv     + FWD_BK * HD_PAD;
    float* s_m      = s_scores + FWD_BQ * FWD_SCORE_STRIDE;
    float* s_l      = s_m      + FWD_BQ;
    float* s_out    = s_l      + FWD_BQ;

    if (local_tid == 0) {
        s_m[qi_local] = -INFINITY;
        s_l[qi_local] = 0.0f;
    }

    float reg_out[D_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < D_PER_THREAD; i++) reg_out[i] = 0.0f;

    // Cooperative load Q tile
    {
        constexpr int64_t total = (int64_t)FWD_BQ * HeadDim;
        for (int64_t i = tid; i < total; i += FWD_NUM_THREADS) {
            int q = (int)(i / HeadDim);
            int d = (int)(i % HeadDim);
            s_q[q * HD_PAD + d] = (qi_block + q < T_q) ? Q_bnh[(qi_block + q) * q_sM + d] : 0.0f;
        }
    }
    __syncthreads();

    const int actual_q = ((int)(T_q - qi_block) < FWD_BQ)
                         ? (int)(T_q - qi_block) : FWD_BQ;
    if (actual_q <= 0) return;

    // Causal early termination in GLOBAL coordinates:
    //   last global Q pos in this tile = q_offset + qi_block + actual_q - 1
    //   max local K index (exclusive) = (q_offset + qi_block + actual_q) - k_offset
    //   clamp to [0, T_k]
    int64_t max_kj_signed = is_causal
        ? ((int64_t)q_offset + qi_block + actual_q - (int64_t)k_offset)
        : T_k;
    if (max_kj_signed < 0) max_kj_signed = 0;
    if (max_kj_signed > T_k) max_kj_signed = T_k;
    const int64_t max_kj = max_kj_signed;

    for (int64_t kj_block = 0; kj_block < max_kj; kj_block += FWD_BK) {
        const int block_len = ((int)(T_k - kj_block) < FWD_BK)
                              ? (int)(T_k - kj_block) : FWD_BK;

        // Cooperative K tile load
        {
            const int64_t total = (int64_t)block_len * HeadDim;
            for (int64_t i = tid; i < total; i += FWD_NUM_THREADS) {
                int k = (int)(i / HeadDim);
                int d = (int)(i % HeadDim);
                s_kv[k * HD_PAD + d] = K_bnh[(kj_block + k) * k_sM + d];
            }
        }
        __syncthreads();

        // Score GEMM
        {
            const int kj_start = local_tid * FWD_KJ_PER_THREAD;
            #pragma unroll
            for (int kk = 0; kk < FWD_KJ_PER_THREAD; kk++) {
                int kj = kj_start + kk;
                float dot;
                if (kj < block_len && qi_valid) {
                    dot = 0.0f;
                    #pragma unroll
                    for (int d = 0; d < HeadDim; d++)
                        dot += s_q[qi_local * HD_PAD + d] * s_kv[kj * HD_PAD + d];
                    dot *= scale;
                    // Causal mask in global coordinates
                    if (is_causal && ((int64_t)k_offset + kj_block + kj) > qi_global_pos)
                        dot = -INFINITY;
                } else {
                    dot = -INFINITY;
                }
                s_scores[qi_local * FWD_SCORE_STRIDE + kj] = dot;
            }
        }
        __syncthreads();

        // Online softmax
        {
            const int kj_start = local_tid * FWD_KJ_PER_THREAD;
            float partial_max = -INFINITY;
            #pragma unroll
            for (int kk = 0; kk < FWD_KJ_PER_THREAD; kk++) {
                int kj = kj_start + kk;
                if (kj < block_len)
                    partial_max = fmaxf(partial_max,
                                        s_scores[qi_local * FWD_SCORE_STRIDE + kj]);
            }

            float row_max = partial_max;
            row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffff, row_max, 1));
            row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffff, row_max, 2));
            row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffff, row_max, 4));

            float m_old = s_m[qi_local];
            float m_new = fmaxf(m_old, row_max);

            float alpha;
            if (m_old == -INFINITY)       alpha = 0.0f;
            else if (m_old == m_new)      alpha = 1.0f;
            else                          alpha = expf(m_old - m_new);

            #pragma unroll
            for (int i = 0; i < D_PER_THREAD; i++)
                reg_out[i] *= alpha;

            float partial_sum = 0.0f;
            #pragma unroll
            for (int kk = 0; kk < FWD_KJ_PER_THREAD; kk++) {
                int kj = kj_start + kk;
                float exp_s;
                if (kj < block_len && m_new > -INFINITY) {
                    exp_s = expf(s_scores[qi_local * FWD_SCORE_STRIDE + kj] - m_new);
                    if (dropout_p > 0.0f && dropout_mask != nullptr) {
                        // Mask indexed by local [T_q, T_k] rows
                        float m_val = dropout_mask[(bnh * T_q + qi_row) * T_k
                                                   + (kj_block + kj)];
                        exp_s *= m_val;
                    }
                } else {
                    exp_s = 0.0f;
                }
                s_scores[qi_local * FWD_SCORE_STRIDE + kj] = exp_s;
                partial_sum += exp_s;
            }
            #pragma unroll
            for (int kk = 0; kk < FWD_KJ_PER_THREAD; kk++) {
                int kj = kj_start + kk;
                if (kj >= block_len)
                    s_scores[qi_local * FWD_SCORE_STRIDE + kj] = 0.0f;
            }

            float row_sum = partial_sum;
            row_sum += __shfl_xor_sync(0xffffffff, row_sum, 1);
            row_sum += __shfl_xor_sync(0xffffffff, row_sum, 2);
            row_sum += __shfl_xor_sync(0xffffffff, row_sum, 4);

            if (local_tid == 0) {
                s_l[qi_local] = alpha * s_l[qi_local] + row_sum;
                s_m[qi_local] = m_new;
            }
        }
        __syncthreads();

        // Cooperative V tile load (overwrites K)
        {
            const int64_t total = (int64_t)block_len * HeadDim;
            for (int64_t i = tid; i < total; i += FWD_NUM_THREADS) {
                int v = (int)(i / HeadDim);
                int d = (int)(i % HeadDim);
                s_kv[v * HD_PAD + d] = V_bnh[(kj_block + v) * v_sM + d];
            }
        }
        __syncthreads();

        // P@V GEMM
        #pragma unroll
        for (int dd = 0; dd < D_PER_THREAD; dd++) {
            int d = local_tid + dd * FWD_THREADS_PER_ROW;
            if (d < HeadDim) {
                float acc = 0.0f;
                #pragma unroll 4
                for (int kj = 0; kj < FWD_BK; kj++) {
                    if (kj < block_len)
                        acc += s_scores[qi_local * FWD_SCORE_STRIDE + kj]
                             * s_kv[kj * HD_PAD + d];
                }
                reg_out[dd] += acc;
            }
        }
        __syncthreads();
    }

    // Normalize & stage into s_out
    if (qi_valid) {
        float li    = s_l[qi_local];
        float inv_l = (li > 0.0f) ? (1.0f / li) : 0.0f;
        #pragma unroll
        for (int dd = 0; dd < D_PER_THREAD; dd++) {
            int d = local_tid + dd * FWD_THREADS_PER_ROW;
            if (d < HeadDim)
                s_out[qi_local * HD_PAD + d] = reg_out[dd] * inv_l;
        }
    }
    __syncthreads();

    // Coalesced writeback
    {
        const int64_t total = (int64_t)actual_q * HeadDim;
        for (int64_t i = (int64_t)tid; i < total; i += FWD_NUM_THREADS) {
            int q = (int)(i / HeadDim);
            int d = (int)(i % HeadDim);
            O_bnh[(qi_block + q) * o_sM + d] = s_out[q * HD_PAD + d];
        }
    }

    if (local_tid == 0 && qi_valid) {
        float m = s_m[qi_local];
        float l = s_l[qi_local];
        LSE_bnh[qi_row] = (l > 0.0f) ? (m + logf(l)) : -INFINITY;
    }
}

// ============================================================================
// Scalar launch function
// ============================================================================

static size_t compute_fwd_smem(int64_t hd) {
    int64_t hd_pad = hd + FWD_SMEM_PAD;
    return ((size_t)(FWD_BQ + FWD_BK + FWD_BQ) * hd_pad
          + (size_t)FWD_BQ * FWD_SCORE_STRIDE
          + (size_t)2 * FWD_BQ) * sizeof(float);
}

static void launch_fwd_kernel(const CPFwdParams& params, int grid_y)
{
    int deviceId;
    cudaGetDevice(&deviceId);
    int max_smem = 0;
    cudaDeviceGetAttribute(&max_smem,
                           cudaDevAttrMaxSharedMemoryPerBlockOptin, deviceId);

    const int64_t hd = 0;  // resolved by template dispatch below
    (void)hd;
    // The smem size is keyed off of HeadDim — recomputed inside LAUNCH_FWD.
    int grid_x = (int)((params.T_q + FWD_BQ - 1) / FWD_BQ);
    dim3 grid(grid_x, grid_y);

    #define LAUNCH_FWD(HD) \
    do { \
        size_t smem = compute_fwd_smem(HD); \
        if ((int)smem > max_smem) { \
            printf("cp::fused_attn_forward: hd=%d needs %zu bytes smem, max is %d.\n", \
                   (int)(HD), smem, max_smem); \
            return; \
        } \
        auto* kernel = fused_attn_forward_kernel<HD>; \
        cudaFuncSetAttribute(kernel, \
                             cudaFuncAttributeMaxDynamicSharedMemorySize, \
                             (int)smem); \
        kernel<<<grid, FWD_NUM_THREADS, smem>>>(params); \
    } while (0)

    // The HeadDim is dispatched by inspecting params via the public API caller;
    // the macro is invoked with the compile-time constant from the switch below.
    // (Caller-supplied HeadDim flows in through the switch in the public API.)

    // Default: caller dispatches HeadDim before this; if reached without
    // a known HeadDim, this is a programming error.
    (void)max_smem;
    #undef LAUNCH_FWD
}

// Helper templated on HeadDim — actual launcher used by the public API switch.
template<int HeadDim>
static void launch_fwd_kernel_hd(const CPFwdParams& params, int grid_y)
{
    int deviceId;
    cudaGetDevice(&deviceId);
    int max_smem = 0;
    cudaDeviceGetAttribute(&max_smem,
                           cudaDevAttrMaxSharedMemoryPerBlockOptin, deviceId);

    size_t smem = compute_fwd_smem(HeadDim);
    if ((int)smem > max_smem) {
        printf("cp::fused_attn_forward: hd=%d needs %zu bytes smem, max is %d.\n",
               (int)HeadDim, smem, max_smem);
        return;
    }

    int grid_x = (int)((params.T_q + FWD_BQ - 1) / FWD_BQ);
    dim3 grid(grid_x, grid_y);

    auto* kernel = fused_attn_forward_kernel<HeadDim>;
    cudaFuncSetAttribute(kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem);
    kernel<<<grid, FWD_NUM_THREADS, smem>>>(params);
}

static void launch_fwd_kernel_dispatch(const CPFwdParams& params, int64_t hd, int grid_y)
{
    switch ((int)hd) {
        case   8: launch_fwd_kernel_hd<  8>(params, grid_y); break;
        case  16: launch_fwd_kernel_hd< 16>(params, grid_y); break;
        case  24: launch_fwd_kernel_hd< 24>(params, grid_y); break;
        case  32: launch_fwd_kernel_hd< 32>(params, grid_y); break;
        case  40: launch_fwd_kernel_hd< 40>(params, grid_y); break;
        case  48: launch_fwd_kernel_hd< 48>(params, grid_y); break;
        case  56: launch_fwd_kernel_hd< 56>(params, grid_y); break;
        case  64: launch_fwd_kernel_hd< 64>(params, grid_y); break;
        case  80: launch_fwd_kernel_hd< 80>(params, grid_y); break;
        case  96: launch_fwd_kernel_hd< 96>(params, grid_y); break;
        case 128: launch_fwd_kernel_hd<128>(params, grid_y); break;
        case 160: launch_fwd_kernel_hd<160>(params, grid_y); break;
        case 192: launch_fwd_kernel_hd<192>(params, grid_y); break;
        case 256: launch_fwd_kernel_hd<256>(params, grid_y); break;
        default:
            printf("cp::fused_attn_forward: unsupported head_dim %d\n", (int)hd);
            break;
    }
}

// ============================================================================
// TC (TF32 WMMA) forward kernel (CP)
// ============================================================================

static constexpr int FWD_TC_BQ          = 32;
static constexpr int FWD_TC_BK          = 32;
static constexpr int FWD_TC_NUM_THREADS = 256;
static constexpr int FWD_TC_WMMA_M      = 16;
static constexpr int FWD_TC_WMMA_N      = 16;
static constexpr int FWD_TC_WMMA_K      = 8;
static constexpr int FWD_TC_SMEM_PAD    = 4;

template<int HeadDim>
__global__ void fused_attn_forward_kernel_tc(CPFwdParams params)
{
    const float* __restrict__ Q             = params.Q;
    const float* __restrict__ K             = params.K;
    const float* __restrict__ V             = params.V;
    float*       __restrict__ O             = params.O;
    float*       __restrict__ LSE           = params.LSE;
    const int64_t T_q                       = params.T_q;
    const int64_t T_k                       = params.T_k;
    const int     q_offset                  = params.q_offset;
    const int     k_offset                  = params.k_offset;
    const int     nh                        = params.nh;
    const float   scale                     = params.scale;
    const bool    is_causal                 = params.is_causal;
    const float   dropout_p                 = params.dropout_p;
    const float* __restrict__ dropout_mask  = params.dropout_mask;

    static_assert(HeadDim % FWD_TC_WMMA_N == 0,
                  "fused_attn_forward_kernel_tc: HeadDim must be divisible by 16");
    using namespace nvcuda;

    constexpr int HD_PAD        = HeadDim + FWD_TC_SMEM_PAD;
    constexpr int SCORE_N_TILES = FWD_TC_BK  / FWD_TC_WMMA_N;
    constexpr int SCORE_K_TILES = HeadDim    / FWD_TC_WMMA_K;
    constexpr int PV_N_TILES    = HeadDim    / FWD_TC_WMMA_N;
    constexpr int PV_K_TILES    = FWD_TC_BK  / FWD_TC_WMMA_K;
    constexpr int PV_TOTAL      = 2 * PV_N_TILES;
    constexpr int PV_PASSES     = (PV_TOTAL + 7) / 8;
    constexpr int ROWS_PER_WARP = FWD_TC_BQ / (FWD_TC_NUM_THREADS / 32);

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int tid     = threadIdx.x;

    const int64_t qi_block = (int64_t)blockIdx.x * FWD_TC_BQ;
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
    float* s_q      = smem;
    float* s_kv_base = s_q      + FWD_TC_BQ * HD_PAD;
    float* s_kv[2]  = { s_kv_base, s_kv_base + FWD_TC_BK * HD_PAD };
    float* s_scores = s_kv_base + 2 * FWD_TC_BK * HD_PAD;
    float* s_m      = s_scores + FWD_TC_BQ * FWD_TC_BK;
    float* s_l      = s_m      + FWD_TC_BQ;
    float* s_out    = s_l      + FWD_TC_BQ;
    float* s_pv     = s_out    + FWD_TC_BQ * HD_PAD;

    for (int i = tid; i < FWD_TC_BQ; i += FWD_TC_NUM_THREADS) {
        s_m[i] = -INFINITY;
        s_l[i] =  0.0f;
    }
    for (int i = tid; i < FWD_TC_BQ * HD_PAD; i += FWD_TC_NUM_THREADS) {
        s_out[i] = 0.0f;
    }

    // Load Q tile (vectorized when q_strideM == HeadDim, else scalar)
    {
        const bool q_packed = (q_sM == (int64_t)HeadDim);
        if (q_packed) {
            const int vec_total = (FWD_TC_BQ * HeadDim) / 4;
            for (int i = tid; i < vec_total; i += FWD_TC_NUM_THREADS) {
                const int q = (i * 4) / HeadDim;
                const int d = (i * 4) % HeadDim;

                float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (qi_block + q < T_q) {
                    val = *(const float4*)&Q_bnh[(qi_block + q) * q_sM + d];
                }
                *(float4*)&s_q[q * HD_PAD + d] = val;
            }
        } else {
            const int total = FWD_TC_BQ * HeadDim;
            for (int i = tid; i < total; i += FWD_TC_NUM_THREADS) {
                const int q = i / HeadDim;
                const int d = i % HeadDim;
                s_q[q * HD_PAD + d] = (qi_block + q < T_q)
                    ? Q_bnh[(qi_block + q) * q_sM + d]
                    : 0.0f;
            }
        }
    }
    __syncthreads();

    const int actual_q = (int)(( (int64_t)FWD_TC_BQ < (T_q - qi_block) ) ? (int64_t)FWD_TC_BQ : (T_q - qi_block));
    if (actual_q <= 0) return;

    // Causal early termination in global coords
    int64_t max_kj_signed = is_causal
        ? ((int64_t)q_offset + qi_block + actual_q - (int64_t)k_offset)
        : T_k;
    if (max_kj_signed < 0) max_kj_signed = 0;
    if (max_kj_signed > T_k) max_kj_signed = T_k;
    const int64_t max_kj = max_kj_signed;

    // Whether K/V last-row stride matches HeadDim — enables cp.async 16B path.
    const bool k_packed = (k_sM == (int64_t)HeadDim);
    const bool v_packed = (v_sM == (int64_t)HeadDim);

    // Pre-fetch first K tile
    if (k_packed) {
        const int vec_total = (FWD_TC_BK * HeadDim) / 4;
        for (int i = tid; i < vec_total; i += FWD_TC_NUM_THREADS) {
            const int k = (i * 4) / HeadDim;
            const int d = (i * 4) % HeadDim;
            const int64_t g = 0LL + k;
            cp_async_l16(&s_kv[0][k * HD_PAD + d], &K_bnh[g * k_sM + d], g < T_k);
        }
        cp_async_commit();
    } else {
        const int total = FWD_TC_BK * HeadDim;
        for (int i = tid; i < total; i += FWD_TC_NUM_THREADS) {
            const int k = i / HeadDim;
            const int d = i % HeadDim;
            const int64_t g = 0LL + k;
            s_kv[0][k * HD_PAD + d] = (g < T_k) ? K_bnh[g * k_sM + d] : 0.0f;
        }
        // No cp.async commit when scalar; sync handled by __syncthreads below.
    }

    for (int64_t kj_block = 0; kj_block < max_kj; kj_block += FWD_TC_BK) {
        const int block_len = (int)(( (int64_t)FWD_TC_BK < (T_k - kj_block) ) ? (int64_t)FWD_TC_BK : (T_k - kj_block));
        const int64_t next_kj_block = kj_block + FWD_TC_BK;
        const bool has_next = (next_kj_block < max_kj);

        if (k_packed) cp_async_wait_group<0>();
        __syncthreads();

        // V load into s_kv[1]
        if (v_packed) {
            const int vec_total = (FWD_TC_BK * HeadDim) / 4;
            for (int i = tid; i < vec_total; i += FWD_TC_NUM_THREADS) {
                const int v = (i * 4) / HeadDim;
                const int d = (i * 4) % HeadDim;
                const int64_t g = kj_block + v;
                cp_async_l16(&s_kv[1][v * HD_PAD + d], &V_bnh[g * v_sM + d], g < T_k);
            }
            cp_async_commit();
        } else {
            const int total = FWD_TC_BK * HeadDim;
            for (int i = tid; i < total; i += FWD_TC_NUM_THREADS) {
                const int v = i / HeadDim;
                const int d = i % HeadDim;
                const int64_t g = kj_block + v;
                s_kv[1][v * HD_PAD + d] = (g < T_k) ? V_bnh[g * v_sM + d] : 0.0f;
            }
        }

        // Score GEMM via WMMA (warps 0-3)
        if (warp_id < 4) {
            const int m_tile = warp_id / SCORE_N_TILES;
            const int n_tile = warp_id % SCORE_N_TILES;
            wmma::fragment<wmma::accumulator, FWD_TC_WMMA_M, FWD_TC_WMMA_N, FWD_TC_WMMA_K, float> acc;
            wmma::fill_fragment(acc, 0.0f);

            #pragma unroll
            for (int k = 0; k < SCORE_K_TILES; ++k) {
                wmma::fragment<wmma::matrix_a, FWD_TC_WMMA_M, FWD_TC_WMMA_N, FWD_TC_WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, FWD_TC_WMMA_M, FWD_TC_WMMA_N, FWD_TC_WMMA_K, wmma::precision::tf32, wmma::col_major> b_frag;

                wmma::load_matrix_sync(a_frag, s_q + m_tile * FWD_TC_WMMA_M * HD_PAD + k * FWD_TC_WMMA_K, HD_PAD);
                wmma::load_matrix_sync(b_frag, s_kv[0] + n_tile * FWD_TC_WMMA_N * HD_PAD + k * FWD_TC_WMMA_K, HD_PAD);
                round_tf32_wmma_frag(a_frag); round_tf32_wmma_frag(b_frag);
                wmma::mma_sync(acc, a_frag, b_frag, acc);
            }
            wmma::store_matrix_sync(s_scores + m_tile * FWD_TC_WMMA_M * FWD_TC_BK + n_tile * FWD_TC_WMMA_N, acc, FWD_TC_BK, wmma::mem_row_major);
        }
        __syncthreads();

        // Online softmax
        {
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                const int row = warp_id * ROWS_PER_WARP + r;
                const int64_t qi_row        = qi_block + row;
                const int64_t qi_global_pos = (int64_t)q_offset + qi_row;
                const bool qi_valid = (qi_row < T_q);

                float val = (lane_id < block_len && qi_valid) ? s_scores[row * FWD_TC_BK + lane_id] * scale : -INFINITY;
                if (is_causal && ((int64_t)k_offset + kj_block + lane_id) > qi_global_pos) val = -INFINITY;

                float row_max = val;
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffff, row_max, off));

                const float m_old = s_m[row];
                const float m_new = qi_valid ? fmaxf(m_old, row_max) : m_old;
                const float alpha = (m_old == -INFINITY) ? 0.0f : ((m_old == m_new) ? 1.0f : expf(m_old - m_new));

                #pragma unroll
                for (int d_base = 0; d_base < HeadDim; d_base += 128) {
                    int d = d_base + lane_id * 4;
                    if (d < HeadDim) {
                        float4* ptr = (float4*)&s_out[row * HD_PAD + d];
                        float4 v_out = *ptr;
                        v_out.x *= alpha; v_out.y *= alpha; v_out.z *= alpha; v_out.w *= alpha;
                        *ptr = v_out;
                    }
                }

                float exp_s = (lane_id < block_len && qi_valid && m_new > -INFINITY) ? expf(val - m_new) : 0.0f;
                if (dropout_p > 0.0f && dropout_mask != nullptr && qi_valid && lane_id < block_len)
                    exp_s *= dropout_mask[(bnh * T_q + qi_row) * T_k + (kj_block + lane_id)];
                s_scores[row * FWD_TC_BK + lane_id] = exp_s;

                float row_sum = exp_s;
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    row_sum += __shfl_xor_sync(0xffffffff, row_sum, off);

                if (lane_id == 0) {
                    s_l[row] = alpha * s_l[row] + row_sum;
                    s_m[row] = m_new;
                }
            }
        }
        __syncthreads();

        // Pre-fetch next K tile
        if (has_next) {
            if (k_packed) {
                const int vec_total = (FWD_TC_BK * HeadDim) / 4;
                for (int i = tid; i < vec_total; i += FWD_TC_NUM_THREADS) {
                    const int k = (i * 4) / HeadDim;
                    const int d = (i * 4) % HeadDim;
                    const int64_t g = next_kj_block + k;
                    cp_async_l16(&s_kv[0][k * HD_PAD + d], &K_bnh[g * k_sM + d], g < T_k);
                }
                cp_async_commit();
            } else {
                const int total = FWD_TC_BK * HeadDim;
                for (int i = tid; i < total; i += FWD_TC_NUM_THREADS) {
                    const int k = i / HeadDim;
                    const int d = i % HeadDim;
                    const int64_t g = next_kj_block + k;
                    s_kv[0][k * HD_PAD + d] = (g < T_k) ? K_bnh[g * k_sM + d] : 0.0f;
                }
            }
        }

        // Wait for V tile, compute P@V.
        // V was committed as a cp.async group above; the next-K prefetch (if
        // has_next) was committed AFTER it. wait_group<1> waits for "all groups
        // except the 1 most recent", i.e. it waits for V while leaving the
        // next-K prefetch in flight -- correct ONLY while a next-K group exists.
        // On the final KV iteration has_next is false, so no next-K group was
        // committed and V is the single pending group; wait_group<1> would then
        // wait for NOTHING, letting P@V read a not-yet-arrived V tile. That race
        // is benign under exclusive occupancy (the copy finishes during the
        // score/softmax work) but corrupts the output when co-execution (e.g.
        // the CP ring NCCL kernel) delays the copy -- lse stays correct (K is
        // always fully waited at the top of the loop), only the V->out path is
        // hit. Wait for ALL groups when there is no next-K to keep in flight.
        if (v_packed) {
            if (has_next) cp_async_wait_group<1>();
            else          cp_async_wait_group<0>();
        }
        __syncthreads();

        {
            for (int pass = 0; pass < PV_PASSES; ++pass) {
                const int tile_id = warp_id + pass * 8;
                if (tile_id < PV_TOTAL) {
                    const int m_tile = tile_id / PV_N_TILES;
                    const int n_tile = tile_id % PV_N_TILES;
                    wmma::fragment<wmma::accumulator, FWD_TC_WMMA_M, FWD_TC_WMMA_N, FWD_TC_WMMA_K, float> acc;
                    wmma::fill_fragment(acc, 0.0f);

                    #pragma unroll
                    for (int k = 0; k < PV_K_TILES; ++k) {
                        wmma::fragment<wmma::matrix_a, FWD_TC_WMMA_M, FWD_TC_WMMA_N, FWD_TC_WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag;
                        wmma::fragment<wmma::matrix_b, FWD_TC_WMMA_M, FWD_TC_WMMA_N, FWD_TC_WMMA_K, wmma::precision::tf32, wmma::row_major> b_frag;
                        wmma::load_matrix_sync(a_frag, s_scores + m_tile * FWD_TC_WMMA_M * FWD_TC_BK + k * FWD_TC_WMMA_K, FWD_TC_BK);
                        wmma::load_matrix_sync(b_frag, s_kv[1] + k * FWD_TC_WMMA_K * HD_PAD + n_tile * FWD_TC_WMMA_N, HD_PAD);
                        round_tf32_wmma_frag(a_frag); round_tf32_wmma_frag(b_frag);
                        wmma::mma_sync(acc, a_frag, b_frag, acc);
                    }
                    wmma::store_matrix_sync(s_pv + m_tile * FWD_TC_WMMA_M * HD_PAD + n_tile * FWD_TC_WMMA_N, acc, HD_PAD, wmma::mem_row_major);
                }
            }
        }
        __syncthreads();

        // Accumulate P@V into s_out
        {
            const int vec_total = (FWD_TC_BQ * HeadDim) / 4;
            for (int i = tid; i < vec_total; i += FWD_TC_NUM_THREADS) {
                const int q = (i * 4) / HeadDim;
                const int d = (i * 4) % HeadDim;
                float4* out_ptr = (float4*)&s_out[q * HD_PAD + d];
                float4* pv_ptr  = (float4*)&s_pv[q * HD_PAD + d];
                float4 v_out = *out_ptr;
                float4 v_pv  = *pv_ptr;
                v_out.x += v_pv.x; v_out.y += v_pv.y; v_out.z += v_pv.z; v_out.w += v_pv.w;
                *out_ptr = v_out;
            }
        }
        __syncthreads();
    }

    // Normalize s_out by s_l
    {
        const int vec_total = (actual_q * HeadDim) / 4;
        for (int i = tid; i < vec_total; i += FWD_TC_NUM_THREADS) {
            const int q = (i * 4) / HeadDim;
            const int d = (i * 4) % HeadDim;
            const float li = s_l[q];
            const float inv_l = (li > 0.0f) ? (1.0f / li) : 0.0f;

            float4* out_ptr = (float4*)&s_out[q * HD_PAD + d];
            float4 val = *out_ptr;
            val.x *= inv_l; val.y *= inv_l; val.z *= inv_l; val.w *= inv_l;
            *out_ptr = val;
        }
    }
    __syncthreads();

    // Writeback O — vectorized when o_strideM == HeadDim, else scalar
    const bool o_packed = (o_sM == (int64_t)HeadDim);
    if (o_packed) {
        const int vec_total = (actual_q * HeadDim) / 4;
        for (int i = tid; i < vec_total; i += FWD_TC_NUM_THREADS) {
            const int q = (i * 4) / HeadDim;
            const int d = (i * 4) % HeadDim;
            *(float4*)&O_bnh[(qi_block + q) * o_sM + d] = *(float4*)&s_out[q * HD_PAD + d];
        }
    } else {
        const int total = actual_q * HeadDim;
        for (int i = tid; i < total; i += FWD_TC_NUM_THREADS) {
            const int q = i / HeadDim;
            const int d = i % HeadDim;
            O_bnh[(qi_block + q) * o_sM + d] = s_out[q * HD_PAD + d];
        }
    }

    // Writeback LSE
    for (int i = tid; i < actual_q; i += FWD_TC_NUM_THREADS) {
        const float m = s_m[i];
        const float l = s_l[i];
        LSE_bnh[qi_block + i] = (l > 0.0f) ? (m + logf(l)) : -INFINITY;
    }
}

static size_t compute_fwd_tc_smem(int64_t hd) {
    const size_t hd_pad = (size_t)hd + FWD_TC_SMEM_PAD;
    return (5ULL * FWD_TC_BQ * hd_pad
          + (size_t)FWD_TC_BQ * FWD_TC_BK
          + 2ULL * FWD_TC_BQ)
         * sizeof(float);
}

template<int HeadDim>
static void launch_fwd_tc_kernel_hd(const CPFwdParams& params, int grid_y)
{
    int deviceId;
    cudaGetDevice(&deviceId);
    int max_smem = 0;
    cudaDeviceGetAttribute(&max_smem,
                           cudaDevAttrMaxSharedMemoryPerBlockOptin, deviceId);

    const size_t smem = compute_fwd_tc_smem(HeadDim);
    if ((int)smem > max_smem) {
        printf("cp::fused_attn_forward_tc: hd=%d needs %zu bytes smem, max is %d. Falling back to scalar.\n",
               (int)HeadDim, (size_t)smem, max_smem);
        launch_fwd_kernel_hd<HeadDim>(params, grid_y);
        return;
    }

    const int grid_x = (int)((params.T_q + FWD_TC_BQ - 1) / FWD_TC_BQ);
    const dim3 grid(grid_x, grid_y);

    auto* kernel = fused_attn_forward_kernel_tc<HeadDim>;
    cudaFuncSetAttribute(kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem);
    kernel<<<grid, FWD_TC_NUM_THREADS, smem>>>(params);
}

static void launch_fwd_tc_kernel_dispatch(const CPFwdParams& params, int64_t hd, int grid_y)
{
    if (hd % FWD_TC_WMMA_N != 0) {
        launch_fwd_kernel_dispatch(params, hd, grid_y);
        return;
    }
    switch ((int)hd) {
        case  16: launch_fwd_tc_kernel_hd< 16>(params, grid_y); break;
        case  32: launch_fwd_tc_kernel_hd< 32>(params, grid_y); break;
        case  48: launch_fwd_tc_kernel_hd< 48>(params, grid_y); break;
        case  64: launch_fwd_tc_kernel_hd< 64>(params, grid_y); break;
        case  80: launch_fwd_tc_kernel_hd< 80>(params, grid_y); break;
        case  96: launch_fwd_tc_kernel_hd< 96>(params, grid_y); break;
        case 128: launch_fwd_tc_kernel_hd<128>(params, grid_y); break;
        case 160: launch_fwd_tc_kernel_hd<160>(params, grid_y); break;
        case 192: launch_fwd_tc_kernel_hd<192>(params, grid_y); break;
        case 256: launch_fwd_tc_kernel_hd<256>(params, grid_y); break;
        default:
            launch_fwd_kernel_dispatch(params, hd, grid_y);
            break;
    }
}

} // namespace cp

// ============================================================================
// Public APIs (CP) live in namespace OwnTensor::cp::cuda
// ============================================================================

namespace cp {
namespace cuda {

// Strided variant — caller passes per-tensor strides.
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
    float dropout_p, const float* dropout_mask)
{
    if (hd > MAX_HD) {
        printf("cp::mem_efficient_attn_forward: hd=%d exceeds MAX_HD=%d\n",
            (int)hd, MAX_HD);
        return;
    }
    ::OwnTensor::cp::CPFwdParams params{};
    params.Q = query; params.K = key; params.V = value;
    params.O = output; params.LSE = lse;
    params.B = (int)B; params.nh = (int)nh;
    params.T_q = (int)T_q; params.T_k = (int)T_k;
    params.q_offset = q_offset; params.k_offset = k_offset;
    params.q_strideB = q_strideB; params.q_strideM = q_strideM; params.q_strideH = q_strideH;
    params.k_strideB = k_strideB; params.k_strideM = k_strideM; params.k_strideH = k_strideH;
    params.v_strideB = v_strideB; params.v_strideM = v_strideM; params.v_strideH = v_strideH;
    params.o_strideB = o_strideB; params.o_strideM = o_strideM; params.o_strideH = o_strideH;
    params.lse_strideB = lse_strideB; params.lse_strideH = lse_strideH;
    params.scale = 1.0f / sqrtf(static_cast<float>(hd));
    params.is_causal = is_causal;
    params.dropout_p = dropout_p;
    params.dropout_mask = dropout_mask;

    int grid_y = (int)(B * nh);
    ::OwnTensor::cp::launch_fwd_kernel_dispatch(params, hd, grid_y);
}

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
    float dropout_p, const float* dropout_mask)
{
    if (hd > MAX_HD) {
        printf("cp::mem_efficient_attn_forward_tc: hd=%d exceeds MAX_HD=%d\n",
            (int)hd, MAX_HD);
        return;
    }
    ::OwnTensor::cp::CPFwdParams params{};
    params.Q = query; params.K = key; params.V = value;
    params.O = output; params.LSE = lse;
    params.B = (int)B; params.nh = (int)nh;
    params.T_q = (int)T_q; params.T_k = (int)T_k;
    params.q_offset = q_offset; params.k_offset = k_offset;
    params.q_strideB = q_strideB; params.q_strideM = q_strideM; params.q_strideH = q_strideH;
    params.k_strideB = k_strideB; params.k_strideM = k_strideM; params.k_strideH = k_strideH;
    params.v_strideB = v_strideB; params.v_strideM = v_strideM; params.v_strideH = v_strideH;
    params.o_strideB = o_strideB; params.o_strideM = o_strideM; params.o_strideH = o_strideH;
    params.lse_strideB = lse_strideB; params.lse_strideH = lse_strideH;
    params.scale = 1.0f / sqrtf(static_cast<float>(hd));
    params.is_causal = is_causal;
    params.dropout_p = dropout_p;
    params.dropout_mask = dropout_mask;

    int grid_y = (int)(B * nh);

    // TI sm89 (Ada) kernel is a faster square-case path, BUT it ends its KV loop
    // with an unconditional cp.async.wait_group<1>, so on the final KV tile it
    // does NOT wait for the V copy. Benign under exclusive occupancy, but under CP
    // compute/comm OVERLAP the ring NCCL kernel delays that copy and P@V reads a
    // not-yet-arrived V tile -> intermittent corruption -> grad-norm explosion
    // (world_size>=3, overlap on). We cannot patch the TI kernel here, so by
    // DEFAULT route the square case to the CP kernel (fused_attn_forward_kernel_tc,
    // which has the has_next-guarded wait and handles square fine). Opt back into
    // the TI fast path with CP_FWD_TI_SM89=1 (safe only without forward overlap).
    static const bool use_ti_sm89 = (std::getenv("CP_FWD_TI_SM89") != nullptr);
    if (use_ti_sm89 && T_q == T_k && q_offset == 0 && k_offset == 0 &&
        ::OwnTensor::cuda::get_arch() == ::OwnTensor::cuda::ArchFamily::Ada) {
        ::OwnTensor::fused_attn_forward_tc_sm89_cuda(
            query, key, value, output, lse,
            B, nh, T_q, hd, params.scale, is_causal,
            dropout_p, dropout_mask,
            q_strideB, q_strideM, q_strideH,
            k_strideB, k_strideM, k_strideH,
            v_strideB, v_strideM, v_strideH,
            o_strideB, o_strideM, o_strideH,
            lse_strideB, lse_strideH,
            grid_y, 0);
        return;
    }

    ::OwnTensor::cp::launch_fwd_tc_kernel_dispatch(params, hd, grid_y);
}

// Backward-compat: contiguous wrappers (deduce strides from B/nh/T_q/T_k/hd).
void mem_efficient_attn_forward(
    const float* query, const float* key, const float* value,
    float* output, float* lse,
    int64_t B, int64_t nh,
    int64_t T_q, int64_t T_k,
    int q_offset, int k_offset,
    int64_t hd,
    bool is_causal,
    float dropout_p, const float* dropout_mask)
{
    mem_efficient_attn_forward_strided(
        query, nh * T_q * hd, hd, T_q * hd,
        key,   nh * T_k * hd, hd, T_k * hd,
        value, nh * T_k * hd, hd, T_k * hd,
        output,nh * T_q * hd, hd, T_q * hd,
        lse,   nh * T_q,            T_q,
        B, nh, T_q, T_k, q_offset, k_offset, hd,
        is_causal, dropout_p, dropout_mask);
}

void mem_efficient_attn_forward_tc(
    const float* query, const float* key, const float* value,
    float* output, float* lse,
    int64_t B, int64_t nh,
    int64_t T_q, int64_t T_k,
    int q_offset, int k_offset,
    int64_t hd,
    bool is_causal,
    float dropout_p, const float* dropout_mask)
{
    mem_efficient_attn_forward_tc_strided(
        query, nh * T_q * hd, hd, T_q * hd,
        key,   nh * T_k * hd, hd, T_k * hd,
        value, nh * T_k * hd, hd, T_k * hd,
        output,nh * T_q * hd, hd, T_q * hd,
        lse,   nh * T_q,            T_q,
        B, nh, T_q, T_k, q_offset, k_offset, hd,
        is_causal, dropout_p, dropout_mask);
}

} // namespace cuda
} // namespace cp

} // namespace OwnTensor

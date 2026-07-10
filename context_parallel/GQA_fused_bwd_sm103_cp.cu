// =============================================================================
// CP FUSED GQA Flash Attention BACKWARD — attention-bwd -> RoPE-bwd -> QK-norm-bwd
//
// Context-parallel sibling of GQA_fused_bwd_sm103.cu. Implements the frozen CP
// symbol  OwnTensor::cp::cuda::gqa_fused_rope_cp_backward  (see FusedRoPESDPA.h),
// renamed to avoid the libtensor collision. The attention-backward core (dQ, dK,
// dV via recompute-P) is copied verbatim; the ONLY functional changes are:
//
//   * RoPE cache index uses the 4-delta scheme (per-side head/tail), byte-
//     identical to RopeDeltas.h::rope_global_pos:
//         pos = local_idx + (local_idx >= len/2 ? d1 : d0)
//     Q uses (q_d0,q_d1,T_q); K uses (k_d0,k_d1,T_k). Independent Q/K, T_q/T_k.
//   * rstd is RECOMPUTED in-kernel during the norm+RoPE reconstruct and stashed
//     in smem (sQrstd / sKrstd), so the CP signature needs NO saved q_rstd/k_rstd.
//
// Causal masking is UNCHANGED (local-index compare + tile-skip); correct only
// because the CP driver slices locally-monotonic sub-chunks and sets is_causal
// per step (see forward-kernel header / ContextParallel.h). Deltas index RoPE
// ONLY, never masking.
//
// dV is unchanged (V is never normed/rotated). gamma==nullptr => QK-norm was OFF
// (RoPE-bwd only). dq_gamma/dk_gamma are [hd] fp32 per-call partials (atomicAdd;
// zeroed by the wrapper). Precision: bf16 storage, fp32 math. Compiles only
// under -DCP_FUSED_ROPE (default build => empty TU => no libtensor collision).
// =============================================================================
#if defined(CP_FUSED_ROPE)

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cmath>
#include <cstdio>
#include <stdexcept>

// NOTE: do NOT include FusedRoPESDPA.h here — it pulls SDPAOp.h (autograd deps)
// that don't compile in a .cu TU. This file DEFINES the cp::cuda symbol declared
// there; a signature mismatch surfaces as an undefined-symbol link error.
#include "dtype/Types.h"   // bfloat16_t

using namespace nvcuda;

namespace OwnTensor {
namespace cp {
namespace cuda {
namespace {  // internal linkage

#if defined(CP_ROPE_DEBUG)
__device__ unsigned long long g_rope_oor_count;   // see forward kernel header
#endif

// ---- Forward norm+RoPE on a [rows, D] bf16 smem tile (one lane per row) ----
// Reconstructs Qr/Kr. rstd (recomputed) is written to rstd_out if non-null (used
// by the finalize instead of a saved rstd tensor). pos uses the 4-delta scheme.
template<bool IS_NEOX, int D>
__device__ __forceinline__ void norm_rope_tile(
    __nv_bfloat16* s, int rows, int lane,
    const float* __restrict__ cos_sin_cache,
    const float* __restrict__ gamma,
    int tile_row0, int d0, int d1, int len,
    int cache_seq_len, float eps,
    float* __restrict__ rstd_out)      // smem per-row rstd dest, or nullptr
{
    const int half = D / 2;
    if (lane < rows) {
        __nv_bfloat16* row = s + lane * D;
        if (gamma != nullptr) {
            float ss = 0.0f;
            for (int d = 0; d < D; ++d) { float v = __bfloat162float(row[d]); ss += v * v; }
            float rstd = rsqrtf(ss / (float)D + eps);
            if (rstd_out != nullptr) rstd_out[lane] = rstd;
            for (int d = 0; d < D; ++d)
                row[d] = __float2bfloat16(__bfloat162float(row[d]) * rstd * gamma[d]);
        }
        // 4-delta global position (matches RopeDeltas.h::rope_global_pos).
        const int local_idx = tile_row0 + lane;
        const int pos = local_idx + (local_idx >= (len >> 1) ? d1 : d0);
        if (pos >= 0 && pos < cache_seq_len) {
            const float* cache_row = cos_sin_cache + (long)pos * D;
            for (int i = 0; i < half; ++i) {
                int a_idx = IS_NEOX ? i        : 2 * i;
                int b_idx = IS_NEOX ? i + half : 2 * i + 1;
                float x = __bfloat162float(row[a_idx]);
                float y = __bfloat162float(row[b_idx]);
                float c = cache_row[i];
                float sn = cache_row[i + half];
                row[a_idx] = __float2bfloat16(x * c - y * sn);
                row[b_idx] = __float2bfloat16(x * sn + y * c);
            }
        }
#if defined(CP_ROPE_DEBUG)
        else { atomicAdd(&g_rope_oor_count, 1ULL); }
#endif
    }
}

// ---- Chain one row: dXr (grad wrt rotated+normed) -> dX_raw (bf16) ----------
// Identical to the non-CP kernel. cache_row is the SAME 4-delta global row.
template<bool IS_NEOX, int D>
__device__ __forceinline__ void chain_bwd_row(
    float* dxr, const __nv_bfloat16* raw, float rstd,
    const float* __restrict__ gamma, const float* __restrict__ cache_row,
    __nv_bfloat16* d_out, float* dgamma_accum)
{
    const int half = D / 2;
    for (int i = 0; i < half; ++i) {
        int a_idx = IS_NEOX ? i        : 2 * i;
        int b_idx = IS_NEOX ? i + half : 2 * i + 1;
        float c  = cache_row[i];
        float sn = cache_row[i + half];
        float da = dxr[a_idx], db = dxr[b_idx];
        dxr[a_idx] =  da * c + db * sn;
        dxr[b_idx] = -da * sn + db * c;
    }
    if (gamma == nullptr) {
        for (int d = 0; d < D; ++d) d_out[d] = __float2bfloat16(dxr[d]);
        return;
    }
    float dot = 0.0f;
    for (int d = 0; d < D; ++d)
        dot += dxr[d] * gamma[d] * __bfloat162float(raw[d]);
    float rs3 = rstd * rstd * rstd;
    float ic  = 1.0f / (float)D;
    for (int d = 0; d < D; ++d) {
        float x  = __bfloat162float(raw[d]);
        float gx = rstd * dxr[d] * gamma[d] - rs3 * x * dot * ic;
        d_out[d] = __float2bfloat16(gx);
        if (dgamma_accum != nullptr)
            atomicAdd(&dgamma_accum[d], dxr[d] * x * rstd);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1 — dQ. Grid (B, Hq, T_q/Br), one warp per Q tile.
// ─────────────────────────────────────────────────────────────────────────────
template<int Br, int Bc, int D, bool IS_NEOX>
__global__ void gqa_fused_rope_bwd_dQ(
    const __nv_bfloat16 *d_Q, const __nv_bfloat16 *d_K, const __nv_bfloat16 *d_V,
    const __nv_bfloat16 *d_O, const __nv_bfloat16 *d_dO, const float *d_LSE,
          __nv_bfloat16 *d_dQ,
    const float *d_cache, const float *d_qg, const float *d_kg,
    float *d_dqg,
    int B, int Hq, int Hkv, int G, int T_q, int T_k,
    int cache_seq_len, int q_d0, int q_d1, int k_d0, int k_d1,
    float eps, float scale, bool is_causal
){
    const int b       = blockIdx.x;
    const int hq      = blockIdx.y;
    const int q_tile  = blockIdx.z;
    const int hkv     = hq / G;
    const int lane    = threadIdx.x;
    const int q_row0  = q_tile * Br;
    const int nKTiles = T_k / Bc;
    const int kc_end = is_causal
        ? (((q_row0 + Br - 1) / Bc + 1) < nKTiles ? ((q_row0 + Br - 1) / Bc + 1) : nKTiles)
        : nKTiles;

    const long qBase  = ((long)(b * Hq  + hq)  * T_q + q_row0) * D;
    const long kvBase = ((long)(b * Hkv + hkv) * T_k) * D;
    const long lBase  =  (long)(b * Hq  + hq)  * T_q + q_row0;

    __shared__ __align__(16) __nv_bfloat16 sQ  [Br * D];
    __shared__ __align__(16) __nv_bfloat16 sdO [Br * D];
    __shared__ __align__(16) __nv_bfloat16 sO  [Br * D];
    __shared__ __align__(16) __nv_bfloat16 sK  [Bc * D];
    __shared__ __align__(16) __nv_bfloat16 sV  [Bc * D];
    __shared__ __align__(16) float          sS  [Br * Bc];
    __shared__ __align__(16) float          sdP [Br * Bc];
    __shared__ __align__(16) __nv_bfloat16 sP  [Br * Bc];
    __shared__ __align__(16) __nv_bfloat16 sdS [Br * Bc];
    __shared__ float sLSE [Br];
    __shared__ float sD   [Br];
    __shared__ float sQrstd[Br];   // recomputed rstd (replaces saved d_qrstd)

    for(int i = lane; i < Br * D; i += 32){
        sQ [i] = d_Q [qBase + i];
        sdO[i] = d_dO[qBase + i];
        sO [i] = d_O [qBase + i];
    }
    if(lane < Br){ sLSE[lane] = d_LSE[lBase + lane]; sQrstd[lane] = 1.0f; }
    __syncwarp();

    // FUSED: reconstruct Qr in sQ (norm + RoPE, q-side deltas), stash rstd.
    norm_rope_tile<IS_NEOX, D>(sQ, Br, lane, d_cache, d_qg,
                               q_row0, q_d0, q_d1, T_q, cache_seq_len, eps, sQrstd);
    __syncwarp();

    if(lane < Br){
        float d = 0.0f;
        for(int j = 0; j < D; j++)
            d += __bfloat162float(sdO[lane * D + j]) * __bfloat162float(sO[lane * D + j]);
        sD[lane] = d;
    }
    __syncwarp();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dQ_acc[D / 16];
    for(int t = 0; t < D / 16; t++) wmma::fill_fragment(dQ_acc[t], 0.0f);

    for(int kc = 0; kc < kc_end; kc++){
        const long kBase = kvBase + (long)kc * Bc * D;
        for(int i = lane; i < Bc * D; i += 32){
            sK[i] = d_K[kBase + i];
            sV[i] = d_V[kBase + i];
        }
        __syncwarp();

        // FUSED: reconstruct Kr in sK (norm + RoPE, k-side deltas). K rstd unused here.
        norm_rope_tile<IS_NEOX, D>(sK, Bc, lane, d_cache, d_kg,
                                   kc * Bc, k_d0, k_d1, T_k, cache_seq_len, eps, nullptr);
        __syncwarp();

        // S = Qr @ Kr^T * scale
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> qf;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> kf;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            for(int nt = 0; nt < Bc / 16; nt++){
                wmma::fill_fragment(acc, 0.0f);
                for(int kt = 0; kt < D / 16; kt++){
                    wmma::load_matrix_sync(qf, sQ + kt * 16,            D);
                    wmma::load_matrix_sync(kf, sK + nt * 16 * D + kt * 16, D);
                    wmma::mma_sync(acc, qf, kf, acc);
                }
                for(int t = 0; t < acc.num_elements; t++) acc.x[t] *= scale;
                wmma::store_matrix_sync(sS + nt * 16, acc, Bc, wmma::mem_row_major);
            }
        }
        __syncwarp();

        if(lane < Br){
            float lse = sLSE[lane];
            const int q_idx = q_row0 + lane;
            for(int j = 0; j < Bc; j++){
                float p = (is_causal && (kc * Bc + j) > q_idx)
                          ? 0.0f : expf(sS[lane * Bc + j] - lse);
                sP[lane * Bc + j] = __float2bfloat16(p);
            }
        }
        __syncwarp();

        // dP = dO @ Vr^T  (V raw)
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> dof;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> vf;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            for(int nt = 0; nt < Bc / 16; nt++){
                wmma::fill_fragment(acc, 0.0f);
                for(int kt = 0; kt < D / 16; kt++){
                    wmma::load_matrix_sync(dof, sdO + kt * 16,             D);
                    wmma::load_matrix_sync(vf,  sV  + nt * 16 * D + kt * 16, D);
                    wmma::mma_sync(acc, dof, vf, acc);
                }
                wmma::store_matrix_sync(sdP + nt * 16, acc, Bc, wmma::mem_row_major);
            }
        }
        __syncwarp();

        if(lane < Br){
            float d = sD[lane];
            for(int j = 0; j < Bc; j++){
                float p  = __bfloat162float(sP [lane * Bc + j]);
                float dp = sdP[lane * Bc + j];
                sdS[lane * Bc + j] = __float2bfloat16(p * (dp - d));
            }
        }
        __syncwarp();

        // dQr_acc += dS @ Kr
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> dsf;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> kf;
            for(int nt = 0; nt < D / 16; nt++){
                for(int kt = 0; kt < Bc / 16; kt++){
                    wmma::load_matrix_sync(dsf, sdS + kt * 16,             Bc);
                    wmma::load_matrix_sync(kf,  sK  + kt * 16 * D + nt * 16, D);
                    wmma::mma_sync(dQ_acc[nt], dsf, kf, dQ_acc[nt]);
                }
            }
        }
        __syncwarp();
    }

    // Materialize dQr (scaled) into fp32 scratch (reuse sK).
    float* dqr = reinterpret_cast<float*>(sK);
    for(int nt = 0; nt < D / 16; nt++){
        for(int t = 0; t < dQ_acc[nt].num_elements; t++) dQ_acc[nt].x[t] *= scale;
        wmma::store_matrix_sync(dqr + nt * 16, dQ_acc[nt], D, wmma::mem_row_major);
    }
    __syncwarp();

    // Reload raw Q (reuse sV region) for the norm-backward.
    __nv_bfloat16* rawQ = sV;
    for(int i = lane; i < Br * D; i += 32) rawQ[i] = d_Q[qBase + i];
    __syncwarp();

    // Chain RoPE-bwd -> RMSNorm-bwd per row, write dQ_raw (bf16). 4-delta pos.
    if(lane < Br){
        const int local_idx = q_row0 + lane;
        const int pos = local_idx + (local_idx >= (T_q >> 1) ? q_d1 : q_d0);
        const float* cache_row =
            (pos >= 0 && pos < cache_seq_len) ? d_cache + (long)pos * D : nullptr;
        const float rstd = sQrstd[lane];   // recomputed above (1.0 if gamma null)
        if (cache_row != nullptr) {
            chain_bwd_row<IS_NEOX, D>(dqr + lane * D, rawQ + lane * D, rstd,
                                      d_qg, cache_row, d_dQ + qBase + lane * D, d_dqg);
        } else {
#if defined(CP_ROPE_DEBUG)
            atomicAdd(&g_rope_oor_count, 1ULL);
#endif
            float* dxr = dqr + lane * D;
            const __nv_bfloat16* raw = rawQ + lane * D;
            __nv_bfloat16* out = d_dQ + qBase + lane * D;
            if (d_qg == nullptr) {
                for(int d=0; d<D; ++d) out[d] = __float2bfloat16(dxr[d]);
            } else {
                float dot=0.f; for(int d=0; d<D; ++d) dot += dxr[d]*d_qg[d]*__bfloat162float(raw[d]);
                float rs3=rstd*rstd*rstd, icc=1.0f/(float)D;
                for(int d=0; d<D; ++d){
                    float x=__bfloat162float(raw[d]);
                    out[d]=__float2bfloat16(rstd*dxr[d]*d_qg[d]-rs3*x*dot*icc);
                    if(d_dqg) atomicAdd(&d_dqg[d], dxr[d]*x*rstd);
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2 — dK, dV. Grid (B, Hkv, T_k/Bc), one warp per K tile.
// ─────────────────────────────────────────────────────────────────────────────
template<int Br, int Bc, int D, bool IS_NEOX>
__global__ void gqa_fused_rope_bwd_dKdV(
    const __nv_bfloat16 *d_Q, const __nv_bfloat16 *d_K, const __nv_bfloat16 *d_V,
    const __nv_bfloat16 *d_O, const __nv_bfloat16 *d_dO, const float *d_LSE,
          __nv_bfloat16 *d_dK, __nv_bfloat16 *d_dV,
    const float *d_cache, const float *d_qg, const float *d_kg,
    float *d_dkg,
    int B, int Hq, int Hkv, int G, int T_q, int T_k,
    int cache_seq_len, int q_d0, int q_d1, int k_d0, int k_d1,
    float eps, float scale, bool is_causal
){
    const int b      = blockIdx.x;
    const int hkv    = blockIdx.y;
    const int k_tile = blockIdx.z;
    const int lane   = threadIdx.x;
    const int k_row0 = k_tile * Bc;
    const int nQTiles = T_q / Br;
    int qc_start = 0;
    if (is_causal) {
        const int need = k_row0 - (Br - 1);
        qc_start = (need <= 0) ? 0 : (need + Br - 1) / Br;
    }

    const long kvBase = ((long)(b * Hkv + hkv) * T_k + k_row0) * D;

    __shared__ __align__(16) __nv_bfloat16 sK  [Bc * D];
    __shared__ __align__(16) __nv_bfloat16 sV  [Bc * D];
    __shared__ __align__(16) __nv_bfloat16 sQ  [Br * D];
    __shared__ __align__(16) __nv_bfloat16 sdO [Br * D];
    __shared__ __align__(16) __nv_bfloat16 sO  [Br * D];
    __shared__ __align__(16) float          sS  [Br * Bc];
    __shared__ __align__(16) float          sdP [Br * Bc];
    __shared__ __align__(16) __nv_bfloat16 sP  [Br * Bc];
    __shared__ __align__(16) __nv_bfloat16 sdS [Br * Bc];
    __shared__ float sLSE  [Br];
    __shared__ float sD    [Br];
    __shared__ float sKrstd[Bc];   // recomputed rstd (replaces saved d_krstd)

    for(int i = lane; i < Bc * D; i += 32){
        sK[i] = d_K[kvBase + i];
        sV[i] = d_V[kvBase + i];
    }
    if(lane < Bc) sKrstd[lane] = 1.0f;
    __syncwarp();

    // FUSED: reconstruct Kr in sK (norm + RoPE, k-side deltas) once, stash rstd.
    norm_rope_tile<IS_NEOX, D>(sK, Bc, lane, d_cache, d_kg,
                               k_row0, k_d0, k_d1, T_k, cache_seq_len, eps, sKrstd);
    __syncwarp();

    constexpr int nAccTiles = (Bc / 16) * (D / 16);
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dV_acc[nAccTiles];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dK_acc[nAccTiles];
    for(int t = 0; t < nAccTiles; t++){
        wmma::fill_fragment(dV_acc[t], 0.0f);
        wmma::fill_fragment(dK_acc[t], 0.0f);
    }

    for(int g = 0; g < G; g++){
        const int hq = hkv * G + g;
        for(int qc = qc_start; qc < nQTiles; qc++){
            const int  q_row0 = qc * Br;
            const long qBase  = ((long)(b * Hq + hq) * T_q + q_row0) * D;
            const long lBase  =  (long)(b * Hq + hq) * T_q + q_row0;

            for(int i = lane; i < Br * D; i += 32){
                sQ [i] = d_Q [qBase + i];
                sdO[i] = d_dO[qBase + i];
                sO [i] = d_O [qBase + i];
            }
            if(lane < Br) sLSE[lane] = d_LSE[lBase + lane];
            __syncwarp();

            // FUSED: reconstruct Qr in sQ (norm + RoPE, q-side deltas). Q rstd unused here.
            norm_rope_tile<IS_NEOX, D>(sQ, Br, lane, d_cache, d_qg,
                                       q_row0, q_d0, q_d1, T_q, cache_seq_len, eps, nullptr);
            __syncwarp();

            if(lane < Br){
                float d = 0.0f;
                for(int j = 0; j < D; j++)
                    d += __bfloat162float(sdO[lane * D + j]) * __bfloat162float(sO[lane * D + j]);
                sD[lane] = d;
            }
            __syncwarp();

            // S = Qr @ Kr^T * scale
            {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> qf;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> kf;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
                for(int nt = 0; nt < Bc / 16; nt++){
                    wmma::fill_fragment(acc, 0.0f);
                    for(int kt = 0; kt < D / 16; kt++){
                        wmma::load_matrix_sync(qf, sQ + kt * 16,            D);
                        wmma::load_matrix_sync(kf, sK + nt * 16 * D + kt * 16, D);
                        wmma::mma_sync(acc, qf, kf, acc);
                    }
                    for(int t = 0; t < acc.num_elements; t++) acc.x[t] *= scale;
                    wmma::store_matrix_sync(sS + nt * 16, acc, Bc, wmma::mem_row_major);
                }
            }
            __syncwarp();

            if(lane < Br){
                float lse = sLSE[lane];
                const int q_idx = q_row0 + lane;
                for(int j = 0; j < Bc; j++){
                    float p = (is_causal && (k_row0 + j) > q_idx)
                              ? 0.0f : expf(sS[lane * Bc + j] - lse);
                    sP[lane * Bc + j] = __float2bfloat16(p);
                }
            }
            __syncwarp();

            // dV_acc += P^T @ dO
            {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> pf;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> dof;
                for(int vt = 0; vt < Bc / 16; vt++){
                    for(int nt = 0; nt < D / 16; nt++){
                        for(int kt = 0; kt < Br / 16; kt++){
                            wmma::load_matrix_sync(pf,  sP  + kt * 16 * Bc + vt * 16, Bc);
                            wmma::load_matrix_sync(dof, sdO + kt * 16 * D  + nt * 16, D);
                            wmma::mma_sync(dV_acc[vt * (D/16) + nt], pf, dof,
                                           dV_acc[vt * (D/16) + nt]);
                        }
                    }
                }
            }
            __syncwarp();

            // dP = dO @ V^T
            {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> dof;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> vf;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
                for(int nt = 0; nt < Bc / 16; nt++){
                    wmma::fill_fragment(acc, 0.0f);
                    for(int kt = 0; kt < D / 16; kt++){
                        wmma::load_matrix_sync(dof, sdO + kt * 16,             D);
                        wmma::load_matrix_sync(vf,  sV  + nt * 16 * D + kt * 16, D);
                        wmma::mma_sync(acc, dof, vf, acc);
                    }
                    wmma::store_matrix_sync(sdP + nt * 16, acc, Bc, wmma::mem_row_major);
                }
            }
            __syncwarp();

            // dS = P ⊙ (dP - D)
            if(lane < Br){
                float d = sD[lane];
                for(int j = 0; j < Bc; j++){
                    float p  = __bfloat162float(sP [lane * Bc + j]);
                    float dp = sdP[lane * Bc + j];
                    sdS[lane * Bc + j] = __float2bfloat16(p * (dp - d));
                }
            }
            __syncwarp();

            // dKr_acc += dS^T @ Qr
            {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> dsf;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> qf;
                for(int vt = 0; vt < Bc / 16; vt++){
                    for(int nt = 0; nt < D / 16; nt++){
                        for(int kt = 0; kt < Br / 16; kt++){
                            wmma::load_matrix_sync(dsf, sdS + kt * 16 * Bc + vt * 16, Bc);
                            wmma::load_matrix_sync(qf,  sQ  + kt * 16 * D  + nt * 16, D);
                            wmma::mma_sync(dK_acc[vt * (D/16) + nt], dsf, qf,
                                           dK_acc[vt * (D/16) + nt]);
                        }
                    }
                }
            }
            __syncwarp();
        }
    }

    // Write dV (unchanged) and dKr->dK_raw (chained) per 16-row subtile.
    float* dkr = reinterpret_cast<float*>(sV);
    __nv_bfloat16* rawK = sQ;

    for(int vt = 0; vt < Bc / 16; vt++){
        for(int nt = 0; nt < D / 16; nt++){
            wmma::store_matrix_sync(sdP, dV_acc[vt * (D/16) + nt], 16, wmma::mem_row_major);
            __syncwarp();
            for(int i = lane; i < 16 * 16; i += 32){
                int r = i / 16, c = i % 16;
                d_dV[kvBase + (vt * 16 + r) * D + nt * 16 + c] = __float2bfloat16(sdP[i]);
            }
            __syncwarp();
        }

        for(int nt = 0; nt < D / 16; nt++){
            for(int t = 0; t < dK_acc[vt * (D/16) + nt].num_elements; t++)
                dK_acc[vt * (D/16) + nt].x[t] *= scale;
            wmma::store_matrix_sync(dkr + nt * 16, dK_acc[vt * (D/16) + nt], D, wmma::mem_row_major);
        }
        __syncwarp();

        for(int i = lane; i < 16 * D; i += 32)
            rawK[i] = d_K[kvBase + vt * 16 * D + i];
        __syncwarp();

        // Chain per row of this subtile, 4-delta pos + recomputed rstd from smem.
        if(lane < 16){
            const int krow = k_row0 + vt * 16 + lane;              // local K index
            const int pos  = krow + (krow >= (T_k >> 1) ? k_d1 : k_d0);
            const float rstd = sKrstd[vt * 16 + lane];             // recomputed above
            __nv_bfloat16* out = d_dK + kvBase + (vt * 16 + lane) * D;
            float* dxr = dkr + lane * D;
            const __nv_bfloat16* raw = rawK + lane * D;
            if (pos >= 0 && pos < cache_seq_len) {
                const float* cache_row = d_cache + (long)pos * D;
                chain_bwd_row<IS_NEOX, D>(dxr, raw, rstd, d_kg, cache_row, out, d_dkg);
            } else {
#if defined(CP_ROPE_DEBUG)
                atomicAdd(&g_rope_oor_count, 1ULL);
#endif
                if (d_kg == nullptr) {
                    for(int d=0; d<D; ++d) out[d] = __float2bfloat16(dxr[d]);
                } else {
                    float dot=0.f; for(int d=0; d<D; ++d) dot += dxr[d]*d_kg[d]*__bfloat162float(raw[d]);
                    float rs3=rstd*rstd*rstd, icc=1.0f/(float)D;
                    for(int d=0; d<D; ++d){
                        float x=__bfloat162float(raw[d]);
                        out[d]=__float2bfloat16(rstd*dxr[d]*d_kg[d]-rs3*x*dot*icc);
                        if(d_dkg) atomicAdd(&d_dkg[d], dxr[d]*x*rstd);
                    }
                }
            }
        }
        __syncwarp();
    }
}

template<int Br, int Bc, int D>
void launch_gqa_fused_rope_bwd(
    const __nv_bfloat16 *d_Q, const __nv_bfloat16 *d_K, const __nv_bfloat16 *d_V,
    const __nv_bfloat16 *d_O, const __nv_bfloat16 *d_dO, const float *d_LSE,
          __nv_bfloat16 *d_dQ, __nv_bfloat16 *d_dK, __nv_bfloat16 *d_dV,
    const float *d_cache, const float *d_qg, const float *d_kg,
    float *d_dqg, float *d_dkg,
    int B, int Hq, int Hkv, int G, int T_q, int T_k,
    int cache_seq_len, int q_d0, int q_d1, int k_d0, int k_d1,
    float eps, bool interleaved, float scale, bool is_causal
){
    static_assert(Br % 16 == 0, "Br must be a multiple of 16 (WMMA tile)");
    static_assert(Bc % 16 == 0, "Bc must be a multiple of 16 (WMMA tile)");
    static_assert(D  % 16 == 0, "D  must be a multiple of 16 (WMMA tile)");
    static_assert(Br == 16, "gqa fused backward processes exactly 16 query rows per Q tile");
    static_assert(D % 2 == 0, "head_dim must be even for RoPE");

    dim3 BLOCK(32);
    dim3 GRID1(B, Hq,  T_q / Br);
    dim3 GRID2(B, Hkv, T_k / Bc);

    if (interleaved) {
        gqa_fused_rope_bwd_dQ<Br, Bc, D, false><<<GRID1, BLOCK>>>(
            d_Q, d_K, d_V, d_O, d_dO, d_LSE, d_dQ, d_cache, d_qg, d_kg, d_dqg,
            B, Hq, Hkv, G, T_q, T_k, cache_seq_len, q_d0, q_d1, k_d0, k_d1, eps, scale, is_causal);
        gqa_fused_rope_bwd_dKdV<Br, Bc, D, false><<<GRID2, BLOCK>>>(
            d_Q, d_K, d_V, d_O, d_dO, d_LSE, d_dK, d_dV, d_cache, d_qg, d_kg, d_dkg,
            B, Hq, Hkv, G, T_q, T_k, cache_seq_len, q_d0, q_d1, k_d0, k_d1, eps, scale, is_causal);
    } else {
        gqa_fused_rope_bwd_dQ<Br, Bc, D, true><<<GRID1, BLOCK>>>(
            d_Q, d_K, d_V, d_O, d_dO, d_LSE, d_dQ, d_cache, d_qg, d_kg, d_dqg,
            B, Hq, Hkv, G, T_q, T_k, cache_seq_len, q_d0, q_d1, k_d0, k_d1, eps, scale, is_causal);
        gqa_fused_rope_bwd_dKdV<Br, Bc, D, true><<<GRID2, BLOCK>>>(
            d_Q, d_K, d_V, d_O, d_dO, d_LSE, d_dK, d_dV, d_cache, d_qg, d_kg, d_dkg,
            B, Hq, Hkv, G, T_q, T_k, cache_seq_len, q_d0, q_d1, k_d0, k_d1, eps, scale, is_causal);
    }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Library-facing CP wrapper. See context_parallel/FusedRoPESDPA.h.
// bf16 Q/K/V + grad_Q/K/V (separate, contiguous); fp32 lse + dq/dk_gamma.
// ---------------------------------------------------------------------------
void gqa_fused_rope_cp_backward(
    const bfloat16_t* Q, const bfloat16_t* K, const bfloat16_t* V,
    const float* cos_sin_cache, const float* q_gamma, const float* k_gamma,
    const bfloat16_t* output, const bfloat16_t* grad_output, const float* lse,
    bfloat16_t* grad_Q, bfloat16_t* grad_K, bfloat16_t* grad_V,
    float* dq_gamma, float* dk_gamma,
    int B, int Nq_heads, int Nkv_heads, int T_q, int T_k, int hd, int cache_seq_len,
    int q_d0, int q_d1, int k_d0, int k_d1,
    float eps, bool interleaved, bool is_causal, float scale)
{
    if (Nkv_heads <= 0 || Nq_heads % Nkv_heads != 0)
        throw std::runtime_error(
            "gqa_fused_rope_cp_backward: Nq_heads must be a positive multiple of Nkv_heads");
    if (T_q % 32 != 0 || T_k % 32 != 0)
        throw std::runtime_error(
            "gqa_fused_rope_cp_backward: T_q and T_k must be multiples of 32 (tile size)");
    if ((T_q & 1) || (T_k & 1))
        throw std::runtime_error(
            "gqa_fused_rope_cp_backward: T_q and T_k must be even (head/tail split point)");

    const int G = Nq_heads / Nkv_heads;

    const __nv_bfloat16* dQ  = reinterpret_cast<const __nv_bfloat16*>(Q);
    const __nv_bfloat16* dK  = reinterpret_cast<const __nv_bfloat16*>(K);
    const __nv_bfloat16* dV  = reinterpret_cast<const __nv_bfloat16*>(V);
    const __nv_bfloat16* dO  = reinterpret_cast<const __nv_bfloat16*>(output);
    const __nv_bfloat16* ddO = reinterpret_cast<const __nv_bfloat16*>(grad_output);
    __nv_bfloat16* dQ_out = reinterpret_cast<__nv_bfloat16*>(grad_Q);
    __nv_bfloat16* dK_out = reinterpret_cast<__nv_bfloat16*>(grad_K);
    __nv_bfloat16* dV_out = reinterpret_cast<__nv_bfloat16*>(grad_V);

    // dq_gamma / dk_gamma accumulate over all tokens/heads (per-call partials) -> zero first.
    if (dq_gamma != nullptr) cudaMemset(dq_gamma, 0, sizeof(float) * hd);
    if (dk_gamma != nullptr) cudaMemset(dk_gamma, 0, sizeof(float) * hd);

#if defined(CP_ROPE_DEBUG)
    { unsigned long long z = 0ULL; cudaMemcpyToSymbol(g_rope_oor_count, &z, sizeof(z)); }
#endif

    switch (hd) {
        case 64:
            launch_gqa_fused_rope_bwd<16, 32, 64>(
                dQ, dK, dV, dO, ddO, lse, dQ_out, dK_out, dV_out,
                cos_sin_cache, q_gamma, k_gamma, dq_gamma, dk_gamma,
                B, Nq_heads, Nkv_heads, G, T_q, T_k, cache_seq_len,
                q_d0, q_d1, k_d0, k_d1, eps, interleaved, scale, is_causal);
            break;
        case 128:
            launch_gqa_fused_rope_bwd<16, 32, 128>(
                dQ, dK, dV, dO, ddO, lse, dQ_out, dK_out, dV_out,
                cos_sin_cache, q_gamma, k_gamma, dq_gamma, dk_gamma,
                B, Nq_heads, Nkv_heads, G, T_q, T_k, cache_seq_len,
                q_d0, q_d1, k_d0, k_d1, eps, interleaved, scale, is_causal);
            break;
        default:
            throw std::runtime_error(
                "gqa_fused_rope_cp_backward: only head_dim 64 or 128 are compiled");
    }

#if defined(CP_ROPE_DEBUG)
    {
        cudaDeviceSynchronize();
        unsigned long long oor = 0ULL;
        cudaMemcpyFromSymbol(&oor, g_rope_oor_count, sizeof(oor));
        if (oor != 0ULL)
            std::fprintf(stderr,
                "[CP-RoPE][bwd][ERROR] %llu RoPE cache indices out of range\n", oor);
    }
#endif
}

} // namespace cuda
} // namespace cp
} // namespace OwnTensor

#endif // CP_FUSED_ROPE

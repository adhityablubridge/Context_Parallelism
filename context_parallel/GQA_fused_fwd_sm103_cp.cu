// =============================================================================
// CP FUSED GQA Flash Attention FORWARD — QK-norm + RoPE(4-delta) + attention
//
// Context-parallel sibling of GQA_fused_fwd_sm103.cu. Implements the frozen CP
// symbol  OwnTensor::cp::cuda::gqa_fused_rope_cp_forward  (see FusedRoPESDPA.h),
// renamed so it does NOT collide with libtensor's OwnTensor::cuda::
// gqa_fused_flash_attn_forward. The flash-attention core (tiling, online
// softmax, WMMA GEMMs, bf16 math) is copied verbatim from the non-CP kernel;
// the ONLY functional change is the RoPE cache index:
//
//     scalar:   pos = local_idx + pos_offset                 (one contiguous run)
//     4-delta:  pos = local_idx + (local_idx >= len/2 ? d1 : d0)   (head|tail)
//
// applied INDEPENDENTLY to Q (len=T_q, q_d0/q_d1) and K (len=T_k, k_d0/k_d1).
// This lets each rank's non-adjacent [head | tail] shard (HeadTail load
// balancing) rotate at its TRUE global positions while the full cos/sin cache
// stays resident. The device formula is byte-identical to the host source of
// truth RopeDeltas.h::rope_global_pos (same `>= len/2` split).
//
// CAUSAL MASKING IS UNCHANGED (local-index comparison + tile-skip). This is
// correct ONLY because the CP driver slices each ring step into sub-chunks
// (Full / KHeadHalf / QTailHalf) so that every kernel call sees a locally-
// monotonic (== relative-global-order) tensor, and sets is_causal per step.
// >>> If you change how sub-chunks are sliced in ContextParallel.h, you MUST
// >>> re-confirm local order == relative global order, or this mask is silently
// >>> wrong. <<<  The 4 deltas are used ONLY for RoPE indexing, never masking.
//
// Precision: bf16 storage, fp32 math (norm, rotation, softmax, accum). Inputs
// are bf16 Q/K/V; output is bf16; lse is fp32. The whole TU compiles only under
// -DCP_FUSED_ROPE (default build => empty TU => no libtensor collision).
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
// Debug-only: counts RoPE cache indices that fall out of range. Must stay 0 in
// correct runs; a nonzero value flags a bad delta (off-by-one chunk / wrong
// source rank) that would SILENTLY skip rotation (unrotated Q/K, no crash).
__device__ unsigned long long g_rope_oor_count;
#endif

// ---------------------------------------------------------------------------
// QK-norm (RMSNorm over hd) then RoPE on `rows` head vectors in smem tile `s`
// [rows, D] (row-major, bf16). One lane per row. base_pos replaced by the
// 4-delta (tile_row0, d0, d1, len) so pos = (tile_row0+lane) + half-split delta.
// ---------------------------------------------------------------------------
template<bool IS_NEOX, int D>
__device__ __forceinline__ void norm_rope_tile(
    __nv_bfloat16* s, int rows, int lane,
    const float* __restrict__ cos_sin_cache,
    const float* __restrict__ gamma,   // [D] or nullptr
    int tile_row0, int d0, int d1, int len,   // 4-delta offset (was: int base_pos)
    int cache_seq_len, float eps,
    float* __restrict__ rstd_out)      // per-row stride-1 dest, or nullptr
{
    const int half = D / 2;
    if (lane < rows) {
        __nv_bfloat16* row = s + lane * D;

        // --- QK-norm over hd ---
        float rstd = 1.0f;
        if (gamma != nullptr) {
            float ss = 0.0f;
            for (int d = 0; d < D; ++d) {
                float v = __bfloat162float(row[d]);
                ss += v * v;
            }
            rstd = rsqrtf(ss / (float)D + eps);
            if (rstd_out != nullptr) rstd_out[lane] = rstd;
            for (int d = 0; d < D; ++d)
                row[d] = __float2bfloat16(__bfloat162float(row[d]) * rstd * gamma[d]);
        }

        // --- RoPE (in place), 4-delta global position ---
        // MUST match RopeDeltas.h::rope_global_pos (same `>= len/2` split).
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
        else {
            atomicAdd(&g_rope_oor_count, 1ULL);
        }
#endif
    }
}

template<int Br, int Bc, int D, bool IS_NEOX>
__global__ void gqa_fused_rope_v2(
  const __nv_bfloat16 *d_Q,
  const __nv_bfloat16 *d_K,
  const __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O,
  float *d_LSE,
  const float *d_cache,
  const float *d_qg,        // q_gamma [D] or nullptr
  const float *d_kg,        // k_gamma [D] or nullptr
  int B, int Hq, int Hkv, int G, int T_q, int T_k,
  int cache_seq_len, int q_d0, int q_d1, int k_d0, int k_d1, float eps,
  float scale, bool is_causal
){
  const int b      = blockIdx.x;
  const int hq     = blockIdx.y;
  const int q_tile = blockIdx.z;
  const int hkv    = hq / G;
  const int lane   = threadIdx.x;

  const int q_row0   = q_tile * Br;
  const int nKVTiles = T_k / Bc;

  const long qBase  = ((long)(b * Hq  + hq)  * T_q + q_row0) * D;
  const long kvBase = ((long)(b * Hkv + hkv) * T_k) * D;
  const long lBase  = ((long)(b * Hq  + hq)  * T_q) + q_row0;

  __shared__ __align__(16) __nv_bfloat16 sQ[Br * D];
  __shared__ __align__(16) __nv_bfloat16 sK[Bc * D];
  __shared__ __align__(16) __nv_bfloat16 sV[Bc * D];
  __shared__ __align__(16) __nv_bfloat16 sP[Br * Bc];
  __shared__ __align__(16) float         sS[Br * Bc];
  __shared__ __align__(16) float         sO[Br * D];
  __shared__ __align__(16) float         sPV[Br * D];
  __shared__ float sm[Br];
  __shared__ float sl[Br];
  __shared__ float sCorr[Br];

  // Load Q tile, zero running output, init stats.
  for(int i = lane; i < Br * D; i += 32){
    sQ[i] = d_Q[qBase + i];
    sO[i] = 0.0f;
  }
  if(lane < Br){
    sm[lane] = -INFINITY;
    sl[lane] = 0.0f;
  }
  __syncwarp();

  // === FUSED: QK-norm + RoPE on Q (once; Q tile reused across all KV tiles) ===
  // rstd not saved for CP (backward recomputes) -> rstd_out = nullptr.
  norm_rope_tile<IS_NEOX, D>(sQ, Br, lane, d_cache, d_qg,
                             q_row0, q_d0, q_d1, T_q, cache_seq_len, eps, nullptr);
  __syncwarp();

  // Causal masking: LOCAL-index tile-skip (valid only because the CP driver
  // guarantees locally-monotonic sub-chunks; see file header). pos deltas shift
  // only the RoPE cache, never the causal order.
  const int kc_end = is_causal
      ? (((q_row0 + Br - 1) / Bc + 1) < nKVTiles ? ((q_row0 + Br - 1) / Bc + 1) : nKVTiles)
      : nKVTiles;

  for(int kc = 0; kc < kc_end; ++kc){
    const long kBase = kvBase + (long)kc * Bc * D;
    for(int i = lane; i < Bc * D; i += 32){
      sK[i] = d_K[kBase + i];
      sV[i] = d_V[kBase + i];
    }
    __syncwarp();

    // === FUSED: QK-norm + RoPE on this K tile (before QK^T), K-side deltas ===
    norm_rope_tile<IS_NEOX, D>(sK, Bc, lane, d_cache, d_kg,
                               kc * Bc, k_d0, k_d1, T_k, cache_seq_len, eps, nullptr);
    __syncwarp();

    // S = (Q @ K^T) * scale  ->  sS [Br, Bc]
    {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> qf;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> kf;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;

      for(int nt = 0; nt < Bc / 16; ++nt){
        wmma::fill_fragment(acc, 0.0f);
        for(int kt = 0; kt < D / 16; ++kt){
          wmma::load_matrix_sync(qf, sQ + kt * 16, D);
          wmma::load_matrix_sync(kf, sK + nt * 16 * D + kt * 16, D);
          wmma::mma_sync(acc, qf, kf, acc);
        }
        for(int t = 0; t < acc.num_elements; ++t){
          acc.x[t] *= scale;
        }
        wmma::store_matrix_sync(sS + nt * 16, acc, Bc, wmma::mem_row_major);
      }
    }
    __syncwarp();

    // Online-softmax stats + unnormalized P, one lane per query row. Causal mask
    // by LOCAL in-sequence index (kc*Bc + j) vs (q_row0 + lane).
    if(lane < Br){
      const float m_old = sm[lane];
      const float l_old = sl[lane];
      const int   q_idx = q_row0 + lane;

      float tile_max = -INFINITY;
      for(int j = 0; j < Bc; ++j){
        float s = sS[lane * Bc + j];
        if(is_causal && (kc * Bc + j) > q_idx) s = -INFINITY;
        tile_max = fmaxf(tile_max, s);
      }

      const float m_new = fmaxf(m_old, tile_max);
      const float corr  = __expf(m_old - m_new);

      float p_sum = 0.0f;
      for(int j = 0; j < Bc; ++j){
        float p;
        if(is_causal && (kc * Bc + j) > q_idx) p = 0.0f;
        else                                   p = __expf(sS[lane * Bc + j] - m_new);
        sP[lane * Bc + j] = __float2bfloat16(p);
        p_sum += p;
      }

      sm[lane]    = m_new;
      sl[lane]    = l_old * corr + p_sum;
      sCorr[lane] = corr;
    }
    __syncwarp();

    for(int i = lane; i < Br * D; i += 32){
      sO[i] *= sCorr[i / D];
    }
    __syncwarp();

    // PV = P @ V  ->  sPV [Br, D]
    {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> pf;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> vf;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> oacc[D/16];

      for(int nt = 0; nt < D/16; ++nt){
        wmma::fill_fragment(oacc[nt], 0.0f);
        for(int vt = 0; vt < Bc/16; ++vt){
          wmma::load_matrix_sync(pf, sP + vt * 16, Bc);
          wmma::load_matrix_sync(vf, sV + vt * 16 * D + nt * 16, D);
          wmma::mma_sync(oacc[nt], pf, vf, oacc[nt]);
        }
        wmma::store_matrix_sync(sPV + nt * 16, oacc[nt], D, wmma::mem_row_major);
      }
    }
    __syncwarp();

    for(int i = lane; i < Br * D; i += 32){
      sO[i] += sPV[i];
    }
    __syncwarp();
  } // end KV loop

  // Finalize: normalize by running denominator and write LSE.
  for(int i = lane; i < Br * D; i += 32){
    d_O[qBase + i] = __float2bfloat16(sO[i] / sl[i / D]);
  }
  if(lane < Br){
    d_LSE[lBase + lane] = sm[lane] + logf(sl[lane]);
  }
}

template<int Br, int Bc, int D>
void launch_gqa_fused_rope(
  const __nv_bfloat16 *d_Q, const __nv_bfloat16 *d_K, const __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O, float *d_LSE,
  const float *d_cache, const float *d_qg, const float *d_kg,
  int B, int Hq, int Hkv, int G, int T_q, int T_k,
  int cache_seq_len, int q_d0, int q_d1, int k_d0, int k_d1,
  float eps, bool interleaved, float scale, bool is_causal
){
  static_assert(Br % 16 == 0, "Br must be a multiple of 16 (WMMA tile)");
  static_assert(Bc % 16 == 0, "Bc must be a multiple of 16 (WMMA tile)");
  static_assert(D  % 16 == 0, "D  must be a multiple of 16 (WMMA tile)");
  static_assert(Br == 16, "gqa_fused_rope_v2 processes exactly 16 query rows per block");
  static_assert(D % 2 == 0, "head_dim must be even for RoPE");

  dim3 GRID(B, Hq, T_q / Br);
  dim3 BLOCK(32);
  if (interleaved) {
    gqa_fused_rope_v2<Br, Bc, D, false><<<GRID, BLOCK>>>(
        d_Q, d_K, d_V, d_O, d_LSE, d_cache, d_qg, d_kg,
        B, Hq, Hkv, G, T_q, T_k, cache_seq_len, q_d0, q_d1, k_d0, k_d1, eps,
        scale, is_causal);
  } else {
    gqa_fused_rope_v2<Br, Bc, D, true><<<GRID, BLOCK>>>(
        d_Q, d_K, d_V, d_O, d_LSE, d_cache, d_qg, d_kg,
        B, Hq, Hkv, G, T_q, T_k, cache_seq_len, q_d0, q_d1, k_d0, k_d1, eps,
        scale, is_causal);
  }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Library-facing CP wrapper. See context_parallel/FusedRoPESDPA.h.
// bf16 Q/K/V (separate, contiguous [B,H,T,D]); bf16 output; fp32 lse.
// ---------------------------------------------------------------------------
void gqa_fused_rope_cp_forward(
    const bfloat16_t* Q, const bfloat16_t* K, const bfloat16_t* V,
    const float* cos_sin_cache, const float* q_gamma, const float* k_gamma,
    bfloat16_t* output, float* lse,
    int B, int Nq_heads, int Nkv_heads, int T_q, int T_k, int hd, int cache_seq_len,
    int q_d0, int q_d1, int k_d0, int k_d1,
    float eps, bool interleaved, bool is_causal, float scale)
{
    if (Nkv_heads <= 0 || Nq_heads % Nkv_heads != 0)
        throw std::runtime_error(
            "gqa_fused_rope_cp_forward: Nq_heads must be a positive multiple of Nkv_heads");
    if (T_q % 32 != 0 || T_k % 32 != 0)
        throw std::runtime_error(
            "gqa_fused_rope_cp_forward: T_q and T_k must be multiples of 32 (tile size)");
    if ((T_q & 1) || (T_k & 1))
        throw std::runtime_error(
            "gqa_fused_rope_cp_forward: T_q and T_k must be even (head/tail split point)");
    // cache_seq_len must cover the max reachable global pos (= global_seq_len-1).
    // Runtime OOR is additionally caught by the CP_ROPE_DEBUG counter below.

    const int   G  = Nq_heads / Nkv_heads;

    const __nv_bfloat16* dQ = reinterpret_cast<const __nv_bfloat16*>(Q);
    const __nv_bfloat16* dK = reinterpret_cast<const __nv_bfloat16*>(K);
    const __nv_bfloat16* dV = reinterpret_cast<const __nv_bfloat16*>(V);
    __nv_bfloat16* dO = reinterpret_cast<__nv_bfloat16*>(output);

#if defined(CP_ROPE_DEBUG)
    { unsigned long long z = 0ULL; cudaMemcpyToSymbol(g_rope_oor_count, &z, sizeof(z)); }
#endif

    switch (hd) {
        case 64:
            launch_gqa_fused_rope<16, 32, 64>(
                dQ, dK, dV, dO, lse, cos_sin_cache, q_gamma, k_gamma,
                B, Nq_heads, Nkv_heads, G, T_q, T_k, cache_seq_len,
                q_d0, q_d1, k_d0, k_d1, eps, interleaved, scale, is_causal);
            break;
        case 128:
            launch_gqa_fused_rope<16, 32, 128>(
                dQ, dK, dV, dO, lse, cos_sin_cache, q_gamma, k_gamma,
                B, Nq_heads, Nkv_heads, G, T_q, T_k, cache_seq_len,
                q_d0, q_d1, k_d0, k_d1, eps, interleaved, scale, is_causal);
            break;
        default:
            throw std::runtime_error(
                "gqa_fused_rope_cp_forward: only head_dim 64 or 128 are compiled");
    }

#if defined(CP_ROPE_DEBUG)
    {
        cudaDeviceSynchronize();
        unsigned long long oor = 0ULL;
        cudaMemcpyFromSymbol(&oor, g_rope_oor_count, sizeof(oor));
        if (oor != 0ULL)
            std::fprintf(stderr,
                "[CP-RoPE][fwd][ERROR] %llu RoPE cache indices out of range "
                "(bad delta / cache too small) — RoPE silently skipped for those rows\n", oor);
    }
#endif
}

} // namespace cuda
} // namespace cp
} // namespace OwnTensor

#endif // CP_FUSED_ROPE

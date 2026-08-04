#pragma once
// ---------------------------------------------------------------------------
// LongRoPEOps.h -- declaration of the LongRoPE cos/sin cache builder.
//
// LongRoPE (Ding et al. 2024, arXiv:2402.13753) reparametrizes RoPE with a
// SEARCHED per-dimension rescale vector `lambda` (length head_dim/2) plus an
// initial-token threshold `n_hat`. For position n and dim i (pure RoPE freq
// base_freq_i = base^(-2i/d)):
//
//     angle(n,i) = n * base_freq_i                 if n <  n_hat   (original)
//     angle(n,i) = n * base_freq_i / lambda_i      if n >= n_hat   (rescaled)
//
// lambda_i >= 1, monotone non-decreasing across dims. Unlike YaRN, LongRoPE does
// NOT bake an attention-temperature `m`. Layout matches build_rope_cache exactly
// (NeoX half-split: cos in [0,half), sin in [half,head_dim)), so the fused kernel
// indexes it identically. The definition lives in YARNOps.cpp next to
// build_rope_cache. This is a NEW, additive symbol -- nothing existing changes.
// ---------------------------------------------------------------------------

#include "core/Tensor.h"

#include <cstdint>
#include <vector>

namespace OwnTensor {
namespace autograd {

// `mscale` (default 1.0 = paper-faithful LongRoPE, NO temperature) optionally
// bakes YaRN's attention-temperature into cos/sin (cos*mscale, sin*mscale) so a
// LongRoPE+m hybrid can be compared apples-to-apples against YaRN (which uses
// m = 0.1*ln(s)+1). mscale=1.0 leaves the paper behavior unchanged.
Tensor build_rope_cache_longrope(int64_t seq_len, int64_t head_dim, float base,
                                 const std::vector<float>& lambda, int64_t n_hat,
                                 DeviceIndex device, float mscale = 1.0f);

}  // namespace autograd
}  // namespace OwnTensor

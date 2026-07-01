#pragma once
// ---------------------------------------------------------------------------
// RopeGammaAccum.h — QK-norm gamma gradient accumulation for context-parallel
// ring attention (Phase 1d / Verification 1b of the fused-RoPE CP plan).
//
// q_gamma / k_gamma are REPLICATED parameters (one [head_dim] vector per layer,
// used identically by every CP rank), NOT sharded activations like dQ/dK/dV.
// Consequences (this is the whole point of the helper):
//
//   * WITHIN the ring, on each rank: the per-step partials produced by the
//     fused backward accumulate LOCALLY by plain addition — NO ring rotation.
//     Each (q,k) pair is computed exactly once, on the rank holding q at the
//     step holding k, so a local sum over ring steps gives that rank's full
//     partial. (Rotating it like dK would mis-attribute / double-count.)
//
//   * ACROSS ranks: the per-rank partials are reduced by the EXISTING
//     parameter-gradient all-reduce that loops over model.parameters()
//     (gpt2_cp_test.cpp:~1407, op_t::avg over the CP communicator). AVG (not
//     SUM) is correct because each rank's loss is the mean over its own shard;
//     gamma shares that loss path. This file does NOT perform the cross-rank
//     reduction — it only provides the local primitive + documents/encodes the
//     reduction semantics so the unit test can verify them as a spec.
//
// gamma_reduce_reference() below is the REFERENCE semantics used by the test;
// real training relies on the existing NCCL all-reduce, not this function.
//
// Pure CPU, header-only. Vectors are [head_dim].
// ---------------------------------------------------------------------------

#include <vector>
#include <cstddef>
#include <stdexcept>

namespace OwnTensor {
namespace cp {

// Fold one ring step's per-step gamma partial into the running LOCAL gradient.
// This is the real primitive the CP backward calls inside the ring loop:
//   for each ring step: gamma_accumulate_step(grad_q_gamma_local, step.dq_gamma);
// Plain elementwise add — no rotation, no rescale.
inline void gamma_accumulate_step(std::vector<float>& local,
                                  const std::vector<float>& step_partial) {
  if (local.empty()) { local = step_partial; return; }
  if (local.size() != step_partial.size())
    throw std::invalid_argument("gamma_accumulate_step: size mismatch");
  for (std::size_t h = 0; h < local.size(); ++h) local[h] += step_partial[h];
}

// REFERENCE cross-rank reduction (test-only): AVG over ranks, mirroring the
// existing op_t::avg all-reduce. Input: one local partial per rank. Output: the
// reduced [head_dim] gradient every rank should hold after the all-reduce.
inline std::vector<float>
gamma_reduce_reference(const std::vector<std::vector<float>>& per_rank_local) {
  const int N = static_cast<int>(per_rank_local.size());
  if (N == 0) return {};
  const std::size_t hd = per_rank_local[0].size();
  std::vector<float> out(hd, 0.0f);
  for (int r = 0; r < N; ++r) {
    if (per_rank_local[r].size() != hd)
      throw std::invalid_argument("gamma_reduce_reference: ragged ranks");
    for (std::size_t h = 0; h < hd; ++h) out[h] += per_rank_local[r][h];
  }
  for (std::size_t h = 0; h < hd; ++h) out[h] /= static_cast<float>(N);  // AVG, not SUM
  return out;
}

} // namespace cp
} // namespace OwnTensor

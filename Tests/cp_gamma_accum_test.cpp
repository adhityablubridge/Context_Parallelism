// =============================================================================
// cp_gamma_accum_test.cpp — Verification 1b: QK-norm gamma gradient accumulation
// for context-parallel ring attention. Pure CPU, no CUDA, no framework.
//
// Build & run (from CP/ repo root):
//   g++ -std=c++17 -I. Tests/cp_gamma_accum_test.cpp -o /tmp/cp_gamma_accum_test \
//       && /tmp/cp_gamma_accum_test
//
// Catches the gamma bug class identified in review BEFORE any GPU/multi-rank
// job exists, by exercising the REAL accumulation primitive
// (context_parallel/RopeGammaAccum.h) against a closed-form reference:
//   * "summed when should AVG / vice-versa"   -> negative SUM discriminator
//   * "kept only the last step's partial"     -> per-rank local-sum check
//   * "accidentally rotated dk_gamma"         -> per-rank partial must equal
//                                                the LOCAL sum, not a rotated one
//
// Model: N ranks, each doing N ring steps. The fused backward at step (r,i)
// emits a deterministic dummy per-step partial dq_gamma_step[r][i] (a [hd]
// vector). The CP backward folds these LOCALLY (no rotation) into a per-rank
// partial; the existing all-reduce then AVGs across ranks. The reduced result
// must equal the ws=1 reference = (1/N) * sum over all (r,i) partials.
// =============================================================================

#include "context_parallel/RopeGammaAccum.h"

#include <cstdio>
#include <vector>
#include <cmath>

using namespace OwnTensor::cp;

static int g_failures = 0;

static bool vec_eq(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (std::fabs(a[i] - b[i]) > 1e-4f * (1.0f + std::fabs(b[i]))) return false;
  return true;
}

// Deterministic dummy per-step partial: distinct per (rank, step, head-dim)
// so that dropping a step, mis-summing, or mis-attributing surfaces.
static std::vector<float> dummy_step(int r, int i, int hd) {
  std::vector<float> v(hd);
  for (int h = 0; h < hd; ++h) v[h] = static_cast<float>(r * 1000 + i * 10 + h);
  return v;
}

static void run_case(int N, int hd) {
  char tag[64];

  // ---- simulate per-rank LOCAL accumulation over the ring (no rotation) ----
  std::vector<std::vector<float>> per_rank_local(N);
  for (int r = 0; r < N; ++r) {
    std::vector<float> local;                       // starts empty
    for (int i = 0; i < N; ++i) {                   // one fold per ring step
      gamma_accumulate_step(local, dummy_step(r, i, hd));
    }
    // Per-rank partial must equal the plain local sum over all steps.
    std::vector<float> expect_local(hd, 0.0f);
    for (int i = 0; i < N; ++i)
      for (int h = 0; h < hd; ++h) expect_local[h] += dummy_step(r, i, hd)[h];

    std::snprintf(tag, sizeof tag, "N%d local-partial r%d", N, r);
    if (!vec_eq(local, expect_local)) {
      ++g_failures; std::printf("FAIL %-26s\n", tag);
    } else {
      std::printf("ok   %-26s\n", tag);
    }
    per_rank_local[r] = local;
  }

  // ---- cross-rank reduction (AVG) must equal the ws=1 reference ----
  std::vector<float> reduced = gamma_reduce_reference(per_rank_local);

  // ws=1 reference = (1/N) * sum over ALL (r,i) contributions (mean-loss path).
  std::vector<float> ref(hd, 0.0f);
  for (int r = 0; r < N; ++r)
    for (int i = 0; i < N; ++i)
      for (int h = 0; h < hd; ++h) ref[h] += dummy_step(r, i, hd)[h];
  for (int h = 0; h < hd; ++h) ref[h] /= static_cast<float>(N);

  std::snprintf(tag, sizeof tag, "N%d AVG == ws1 ref", N);
  if (!vec_eq(reduced, ref)) { ++g_failures; std::printf("FAIL %-26s\n", tag); }
  else                       { std::printf("ok   %-26s\n", tag); }

  // ---- NEGATIVE discriminator: a SUM reduction must be WRONG for N>1 ----
  // (confirms the gate actually catches the AVG-vs-SUM factor-N bug).
  if (N > 1) {
    std::vector<float> sum_reduce(hd, 0.0f);
    for (int r = 0; r < N; ++r)
      for (int h = 0; h < hd; ++h) sum_reduce[h] += per_rank_local[r][h];
    std::snprintf(tag, sizeof tag, "N%d SUM != ref (caught)", N);
    if (vec_eq(sum_reduce, ref)) { ++g_failures; std::printf("FAIL %-26s (SUM matched ref!)\n", tag); }
    else                         { std::printf("ok   %-26s\n", tag); }
  }
}

int main() {
  for (int N : {1, 2, 4}) run_case(N, /*hd=*/4);

  if (g_failures == 0) { std::printf("\nALL cp_gamma_accum tests passed.\n"); return 0; }
  std::printf("\n%d cp_gamma_accum test(s) FAILED.\n", g_failures);
  return 1;
}

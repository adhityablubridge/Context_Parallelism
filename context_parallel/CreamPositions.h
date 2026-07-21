#pragma once
// ---------------------------------------------------------------------------
// CreamPositions.h -- CPU generators for manipulated RoPE position LABELS used
// by context-window-extension fine-tuning (CREAM / PoSE / RandPos).
//
// These produce a length-T vector L of absolute position labels for a physical
// sequence of T tokens. Feeding cos/sin cache rows gathered at L[j] into the
// existing RoPE path makes the model compute rotations at the (larger) label
// positions while the attention matrix stays TxT -- see the CREAM plan.
//
// PORT FIDELITY: this is a line-by-line port of bigai-nlco/CREAM src/train.py
// (mirrored at /tmp/cream_train.py). The Python original relies on several
// language-level details that DIFFER in naive C++; each is called out inline:
//   - `rand_factor = np.interp(...).astype(int)`  -> TRUNCATION toward zero,
//     NOT rounding. (train.py:215)
//   - `rand_factor / 2`                            -> TRUE (float) division.
//     (train.py:219)
//   - `random.randint(a, b)`                       -> inclusive on BOTH ends.
//   - `np.interp(u, cdf, x)`                        -> look u up against cdf
//     (the domain), return the interpolated x (the range).
//
// With model_max := T (the reference truncates each chunk to model_max, so
// T == model_max always) and scaled_max := factor * T. `factor` is the RoPE
// scale (== YARN_SCALE), an integer >= 2.
//
// Header-only, std-only (no Tensor dependency) so it unit-tests standalone,
// value-returning (immutable), deterministic given an explicit seed. No hidden
// global RNG state; RNG-stream parity with Python is deliberately NOT relied on
// (the deterministic math layers are golden-tested with injected draws).
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace OwnTensor {
namespace cp {
namespace cream {

// Head/tail block length for CREAM, per segmentation branch (train.py:198-203):
//   branch 0 -> model_max/3   (big continuity blocks)
//   branch 1 -> 4*factor      (small-k regime; factor=8 -> k=32, the paper's k)
inline int cream_head_tail_len(int T, int factor, int branch) {
  return (branch == 0) ? (T / 3) : (4 * factor);
}

// ---- guards: reject a configuration that would yield a malformed (negative
// middle / non-monotonic) label vector, rather than emit one silently. --------
inline void cream_validate(int T, int factor, int branch) {
  if (factor < 2)
    throw std::invalid_argument(
        "cream: factor(=YARN_SCALE) must be >= 2, got " + std::to_string(factor));
  const int ht = cream_head_tail_len(T, factor, branch);
  const int len_chunk = T - 2 * ht;
  if (2 * ht >= T)
    throw std::invalid_argument(
        "cream: head+tail(=" + std::to_string(2 * ht) +
        ") >= T(=" + std::to_string(T) + ") -- degenerate middle");
  if (len_chunk < 1)
    throw std::invalid_argument("cream: len_chunk < 1");
}

// ===========================================================================
// (a) Pure math layer -- deterministic label assembly from resolved draws.
//     Golden-tested against tuples captured from the Python reference.
//     Mirrors train.py:205-222 (pos_ids_1 ++ pos_ids_2 ++ pos_ids_3).
// ===========================================================================
inline std::vector<int64_t> cream_labels_from_draws(int T, int factor, int branch,
                                                    int rand_factor, int64_t end_id) {
  cream_validate(T, factor, branch);
  (void)rand_factor;  // rand_factor selects end_id upstream; not needed here.
  const int ht = cream_head_tail_len(T, factor, branch);
  const int head = ht, tail = ht;
  const int len_chunk = T - head - tail;

  std::vector<int64_t> L;
  L.reserve(static_cast<size_t>(T));
  // pos_ids_1 = [0, head)
  for (int j = 0; j < head; ++j) L.push_back(j);
  // pos_ids_2 = [end_id - (len_chunk-1), end_id]  (inclusive => len_chunk entries)
  for (int64_t p = end_id - (len_chunk - 1); p <= end_id; ++p) L.push_back(p);
  // pos_ids_3 = [factor*T - tail, factor*T)
  const int64_t scaled_max = static_cast<int64_t>(factor) * T;
  for (int64_t p = scaled_max - tail; p < scaled_max; ++p) L.push_back(p);
  return L;  // length == head + len_chunk + tail == T
}

// ===========================================================================
// (b) Range / CDF layer -- the two cross-language-hazard spots.
// ===========================================================================

// numpy.interp(query, xp, fp): xp must be strictly increasing. Clamps to the
// endpoints outside [xp.front(), xp.back()]. Argument order matches the call
// site np.interp(u, cdf, x): query=u, xp=cdf, fp=x.
inline double np_interp(double query, const std::vector<double>& xp,
                        const std::vector<double>& fp) {
  const size_t n = xp.size();
  if (n == 0) return 0.0;
  if (query <= xp.front()) return fp.front();
  if (query >= xp.back()) return fp.back();
  // xp increasing -> first index with xp[i] >= query
  size_t hi = 1;
  while (hi < n && xp[hi] < query) ++hi;
  const size_t lo = hi - 1;
  const double t = (query - xp[lo]) / (xp[hi] - xp[lo]);
  return fp[lo] + t * (fp[hi] - fp[lo]);
}

// rand_factor from a uniform draw u ~ U(0,1). Truncated-Gaussian inverse CDF.
// train.py:209-215:
//   mu = 1 + factor; x = linspace(2, 2*factor, 1000)
//   cdf = 0.5*(1 + erf((x-mu)/(sigma*sqrt2)))
//   rand_factor = np.interp(u, cdf, x).astype(int)   <-- TRUNCATION
inline int rand_factor_from_u(double u, int factor, double sigma) {
  const double mu = 1.0 + factor;
  const double sqrt2 = std::sqrt(2.0);
  const int N = 1000;
  const double x0 = 2.0, x1 = 2.0 * factor;
  const double step = (x1 - x0) / (N - 1);
  std::vector<double> x(N), cdf(N);
  for (int i = 0; i < N; ++i) {
    x[i] = x0 + i * step;
    cdf[i] = 0.5 * (1.0 + std::erf((x[i] - mu) / (sigma * sqrt2)));
  }
  const double val = np_interp(u, cdf, x);
  int rf = static_cast<int>(val);  // .astype(int) truncates toward zero
  // interp is clamped to [2, 2*factor], so rf is already in-range; clamp
  // defensively against fp edge effects.
  if (rf < 2) rf = 2;
  if (rf > 2 * factor) rf = 2 * factor;
  return rf;
}

// Inclusive [lo, hi] range for the end_id draw (train.py:219):
//   lo = int( head + (len_chunk-1) * (rand_factor/2) )   <-- int() truncates
//   hi = (rand_factor/2) * model_max - tail - 1          <-- integral by ctor
// rand_factor/2 is TRUE division at both sites.
inline std::pair<int64_t, int64_t> end_id_range(int T, int factor, int branch,
                                                int rand_factor) {
  const int ht = cream_head_tail_len(T, factor, branch);
  const int head = ht, tail = ht;
  const int len_chunk = T - head - tail;
  const double half = static_cast<double>(rand_factor) / 2.0;  // true division
  const int64_t lo =
      static_cast<int64_t>(head + (len_chunk - 1) * half);      // trunc, matches int()
  const int64_t hi =
      static_cast<int64_t>(std::llround(half * T - tail - 1));   // integral by ctor
  return {lo, hi};
}

// ===========================================================================
// (c) Sampling wrapper -- draws the three randoms, calls the layers above.
//     Uses std::uniform_int_distribution (inclusive both ends, matching
//     Python random.randint) -- never modulo (which excludes the upper bound).
// ===========================================================================
inline std::vector<int64_t> generate_cream_positions(int T, int factor, double sigma,
                                                     uint64_t seed) {
  std::mt19937_64 rng(seed);
  const int branch = std::uniform_int_distribution<int>(0, 1)(rng);  // randint(0,1)
  cream_validate(T, factor, branch);
  const double u = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
  const int rand_factor = rand_factor_from_u(u, factor, sigma);
  const auto [lo, hi] = end_id_range(T, factor, branch, rand_factor);
  const int64_t end_id =
      std::uniform_int_distribution<int64_t>(lo, hi)(rng);  // randint(lo,hi)
  return cream_labels_from_draws(T, factor, branch, rand_factor, end_id);
}

// ---------------------------------------------------------------------------
// PoSE (train.py:150-183) -- position-only relabel of a length-T sequence.
// lt1 = lt = 0; split at rt1; second chunk skipped forward by rt.
//   labels = [0 .. rt1) ++ (arange(rt1, T) + rt)
//   rt1 = randint(1, (T+1)//2);  rt = randint(0, scaled_max - T)
// ---------------------------------------------------------------------------
inline std::vector<int64_t> generate_pose_positions(int T, int factor, uint64_t seed) {
  if (factor < 2)
    throw std::invalid_argument("pose: factor must be >= 2");
  const int64_t scaled_max = static_cast<int64_t>(factor) * T;
  std::mt19937_64 rng(seed);
  const int rt1 = std::uniform_int_distribution<int>(1, (T + 1) / 2)(rng);
  const int64_t rt =
      std::uniform_int_distribution<int64_t>(0, scaled_max - T)(rng);
  std::vector<int64_t> L;
  L.reserve(static_cast<size_t>(T));
  for (int j = 0; j < rt1; ++j) L.push_back(j);          // [0, rt1)
  for (int j = rt1; j < T; ++j) L.push_back(j + rt);     // arange(rt1,T)+rt
  return L;
}

// ---------------------------------------------------------------------------
// RandPos (train.py:126-143) -- T unique positions sampled from [0, scaled_max)
// then SORTED (line 141) so labels stay monotonic for causal attention.
// ---------------------------------------------------------------------------
inline std::vector<int64_t> generate_randpos_positions(int T, int factor, uint64_t seed) {
  if (factor < 2)
    throw std::invalid_argument("randpos: factor must be >= 2");
  const int64_t scaled_max = static_cast<int64_t>(factor) * T;
  if (T > scaled_max)
    throw std::invalid_argument("randpos: T > scaled_max");
  std::mt19937_64 rng(seed);
  // random.sample without replacement: partial Fisher-Yates over [0, scaled_max).
  std::vector<int64_t> pool(static_cast<size_t>(scaled_max));
  for (int64_t i = 0; i < scaled_max; ++i) pool[static_cast<size_t>(i)] = i;
  for (int j = 0; j < T; ++j) {
    std::uniform_int_distribution<int64_t> pick(j, scaled_max - 1);
    const int64_t k = pick(rng);
    std::swap(pool[static_cast<size_t>(j)], pool[static_cast<size_t>(k)]);
  }
  std::vector<int64_t> L(pool.begin(), pool.begin() + T);
  std::sort(L.begin(), L.end());  // train.py:141 new_pos_list.sort()
  return L;
}

}  // namespace cream
}  // namespace cp
}  // namespace OwnTensor

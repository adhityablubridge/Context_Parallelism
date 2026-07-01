#pragma once
// ---------------------------------------------------------------------------
// RopeDeltas.h — offset-based RoPE position deltas for context-parallel
// ring attention (Phase 1a of the fused-RoPE CP plan).
//
// Replaces a single scalar `pos_offset` with a per-side (head/tail) delta pair
// so that, under HeadTail load balancing, each rank's local shard — which is
// TWO non-adjacent global chunks [head | tail] — is rotated at its TRUE global
// positions. The kernel computes, for a PASSED tensor of length `len`:
//
//     global_pos(j) = j + ( j >= len/2 ? d1 : d0 )
//
// and indexes cos_sin_cache at that global position. This is PURE CPU metadata
// derivation — no device code, no communication, no cache transfer. The full
// cos/sin cache stays resident on every rank.
//
// Degenerate cases (verified by Tests/cp_rope_deltas_test.cpp):
//   non-CP        -> (0,0)/(0,0)            -> global_pos(j)=j (identical to today)
//   contiguous CP -> (r*Tl,r*Tl)/(s*Tl,s*Tl) (two equal per side = one range)
//   HeadTail LB   -> piecewise head/tail; sub-chunked side collapses to equal
//
// Header-only; spelled with explicit OwnTensor::cp namespace so include sites
// need no `using`.
// ---------------------------------------------------------------------------

namespace OwnTensor {
namespace cp {

// Per-side delta pair: d0 applies to the first half of the passed sequence,
// d1 to the second half (split at len/2).
struct Deltas { int d0; int d1; };

// Deltas for one SDPA call: Q side and K side.
struct SdpaDeltas { Deltas q; Deltas k; };

// Which (if any) side was sub-chunked to a single half this ring step.
//   Full       : i==0 diagonal — full [head|tail] Q vs full [head|tail] K
//   KHeadHalf  : 0<i<=rank (LB) — full Q vs K[:Tl/2] (head half of source s)
//   QTailHalf  : i>rank   (LB)  — Q[Tl/2:] (tail half of rank r) vs full K
enum class SubChunk { Full, KHeadHalf, QTailHalf };

// Source rank whose K/V shard is resident at ring step i on rank r.
inline int rope_source_rank(int r, int i, int N) {
  return ((r - i) % N + N) % N;
}

// Global position of local index j in a PASSED tensor of length `len`.
inline int rope_global_pos(int j, int len, Deltas d) {
  return j + (j >= len / 2 ? d.d1 : d.d0);
}

// Compute the Q/K delta pairs for ring step i on rank r.
//   r  = rank, i = ring step, N = world_size
//   Tl = T_local = T/N  (local sequence length; chunk size cs = Tl/2 = T/(2N))
//   lb = HeadTail load-balancing active
//   sc = which side (if any) was sub-chunked this step
inline SdpaDeltas compute_deltas(int r, int i, int N, int Tl, bool lb, SubChunk sc) {
  const int s = rope_source_rank(r, i, N);

  if (!lb) {
    // Contiguous (and non-CP when N==1 => r==s==0 => all zero). Each shard is
    // one contiguous global range [x*Tl, (x+1)*Tl): both halves share one delta.
    const int qd = r * Tl;
    const int kd = s * Tl;
    return SdpaDeltas{ Deltas{qd, qd}, Deltas{kd, kd} };
  }

  // HeadTail: rank x owns chunks (x, 2N-1-x), laid out [head | tail].
  const int cs = Tl / 2;                              // chunk size = T/(2N)
  auto head_base = [&](int x) { return x * cs; };     // global start of head chunk
  auto tail_base = [&](int x) { return (2 * N - 1 - x) * cs; };  // global start of tail chunk

  // Full [head|tail] shard: head occupies local [0,Tl/2), tail [Tl/2,Tl).
  //   head local j  -> head_base(x)+j            => d0 = head_base(x)
  //   tail local j  -> tail_base(x)+(j - Tl/2)   => d1 = tail_base(x) - Tl/2
  Deltas q{ head_base(r), tail_base(r) - Tl / 2 };
  Deltas k{ head_base(s), tail_base(s) - Tl / 2 };

  // A sub-chunked side passes a single contiguous half-chunk whose local index
  // restarts at 0, so both its deltas collapse to that chunk's base.
  switch (sc) {
    case SubChunk::KHeadHalf: k = Deltas{ head_base(s), head_base(s) }; break;  // K[:Tl/2]
    case SubChunk::QTailHalf: q = Deltas{ tail_base(r), tail_base(r) }; break;  // Q[Tl/2:]
    case SubChunk::Full: break;
  }
  return SdpaDeltas{ q, k };
}

} // namespace cp
} // namespace OwnTensor

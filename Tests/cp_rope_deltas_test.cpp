// =============================================================================
// cp_rope_deltas_test.cpp — unit tests for the offset-based RoPE delta helper
// (Phase 1a of the CP fused-RoPE plan). Pure CPU, no CUDA, no framework.
//
// Build & run (standalone):
//   g++ -std=c++17 -I.. Tests/cp_rope_deltas_test.cpp -o /tmp/cp_rope_deltas_test \
//       && /tmp/cp_rope_deltas_test
// (run from the CP/ repo root; -I gives access to context_parallel/RopeDeltas.h)
//
// Asserts the §"core math" table from the plan for N in {1,2,4}, all ranks,
// all ring steps, and all SubChunk cases, plus end-to-end global-position
// reconstruction for the canonical N=2 {0,1,6,7} example.
// =============================================================================

#include "context_parallel/RopeDeltas.h"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <set>

using namespace OwnTensor::cp;

static int g_failures = 0;

static void check_deltas(const char* tag, SdpaDeltas got,
                         int eq0, int eq1, int ek0, int ek1) {
  const bool ok = got.q.d0 == eq0 && got.q.d1 == eq1 &&
                  got.k.d0 == ek0 && got.k.d1 == ek1;
  if (!ok) {
    ++g_failures;
    std::printf("FAIL %-28s got q=(%d,%d) k=(%d,%d)  want q=(%d,%d) k=(%d,%d)\n",
                tag, got.q.d0, got.q.d1, got.k.d0, got.k.d1, eq0, eq1, ek0, ek1);
  } else {
    std::printf("ok   %-28s q=(%d,%d) k=(%d,%d)\n",
                tag, got.q.d0, got.q.d1, got.k.d0, got.k.d1);
  }
}

// Reconstruct the global position set of a PASSED tensor of length `len`
// using the per-side delta pair and the midpoint-split rule.
static std::set<int> global_set(int len, Deltas d) {
  std::set<int> s;
  for (int j = 0; j < len; ++j) s.insert(rope_global_pos(j, len, d));
  return s;
}

static void check_set(const char* tag, std::set<int> got, std::set<int> want) {
  if (got != want) {
    ++g_failures;
    std::printf("FAIL %-28s global set mismatch\n", tag);
  } else {
    std::printf("ok   %-28s global set OK\n", tag);
  }
}

int main() {
  // ---- N=1 (non-CP): everything is zero ----
  check_deltas("N1 noncp r0 i0", compute_deltas(0, 0, 1, 8, /*lb=*/false, SubChunk::Full),
               0, 0, 0, 0);

  // ---- N=2 contiguous (lb=false), Tl=4 (T=8) ----
  check_deltas("N2 contig r0 i0", compute_deltas(0, 0, 2, 4, false, SubChunk::Full), 0,0, 0,0);
  check_deltas("N2 contig r0 i1", compute_deltas(0, 1, 2, 4, false, SubChunk::Full), 0,0, 4,4);
  check_deltas("N2 contig r1 i0", compute_deltas(1, 0, 2, 4, false, SubChunk::Full), 4,4, 4,4);
  check_deltas("N2 contig r1 i1", compute_deltas(1, 1, 2, 4, false, SubChunk::Full), 4,4, 0,0);

  // ---- N=2 HeadTail LB, Tl=4, cs=2; head_base=x*2, tail_base=(3-x)*2 ----
  // rank0 shard = chunks (0,3) = global {0,1,6,7}; rank1 shard = chunks (1,2) = {2,3,4,5}
  check_deltas("N2 LB r0 i0 Full",  compute_deltas(0, 0, 2, 4, true, SubChunk::Full),     0,4, 0,4);
  check_deltas("N2 LB r1 i0 Full",  compute_deltas(1, 0, 2, 4, true, SubChunk::Full),     2,2, 2,2);
  // rank0 i1 -> i>rank -> QTailHalf: Q=tail(0)={6,7}, K=full source s=1={2,3,4,5}
  check_deltas("N2 LB r0 i1 QTail", compute_deltas(0, 1, 2, 4, true, SubChunk::QTailHalf), 6,6, 2,2);
  // rank1 i1 -> i<=rank -> KHeadHalf: Q=full rank1, K=head(s=0)={0,1}
  check_deltas("N2 LB r1 i1 KHead", compute_deltas(1, 1, 2, 4, true, SubChunk::KHeadHalf), 2,2, 0,0);

  // ---- N=4 HeadTail LB, Tl=4, cs=2; head_base=x*2, tail_base=(7-x)*2 ----
  // rank1 shard = chunks (1,6) = {2,3,12,13}
  check_deltas("N4 LB r1 i0 Full",  compute_deltas(1, 0, 4, 4, true, SubChunk::Full),     2,10, 2,10);
  // rank1 i1 -> KHeadHalf, s=(1-1)%4=0: K head(0)={0,1}
  check_deltas("N4 LB r1 i1 KHead", compute_deltas(1, 1, 4, 4, true, SubChunk::KHeadHalf), 2,10, 0,0);
  // rank1 i2 -> QTailHalf, s=(1-2)%4=3: Q tail(1)={12,13}, K full source3 chunks(3,4)={6,7,8,9}->(6,6)
  check_deltas("N4 LB r1 i2 QTail", compute_deltas(1, 2, 4, 4, true, SubChunk::QTailHalf), 12,12, 6,6);

  // ---- end-to-end global-position reconstruction (the canonical example) ----
  // N=2 LB rank0 i0 Full: local [0..3] must map to global {0,1,6,7} on both sides
  {
    SdpaDeltas d = compute_deltas(0, 0, 2, 4, true, SubChunk::Full);
    check_set("N2 LB r0 i0 Q->{0,1,6,7}", global_set(4, d.q), std::set<int>{0,1,6,7});
    check_set("N2 LB r0 i0 K->{0,1,6,7}", global_set(4, d.k), std::set<int>{0,1,6,7});
  }
  // N2 LB r1 i0 Full -> contiguous {2,3,4,5}
  {
    SdpaDeltas d = compute_deltas(1, 0, 2, 4, true, SubChunk::Full);
    check_set("N2 LB r1 i0 Q->{2,3,4,5}", global_set(4, d.q), std::set<int>{2,3,4,5});
  }
  // N2 LB r0 i1 QTail -> passed Q len 2 (tail half) -> {6,7}; K len 4 -> {2,3,4,5}
  {
    SdpaDeltas d = compute_deltas(0, 1, 2, 4, true, SubChunk::QTailHalf);
    check_set("N2 LB r0 i1 Qtail->{6,7}", global_set(2, d.q), std::set<int>{6,7});
    check_set("N2 LB r0 i1 Kfull->{2..5}", global_set(4, d.k), std::set<int>{2,3,4,5});
  }
  // N2 LB r1 i1 KHead -> K passed len 2 (head half of s=0) -> {0,1}
  {
    SdpaDeltas d = compute_deltas(1, 1, 2, 4, true, SubChunk::KHeadHalf);
    check_set("N2 LB r1 i1 Khead->{0,1}", global_set(2, d.k), std::set<int>{0,1});
  }
  // N4 LB r1 i0 Full -> {2,3,12,13}
  {
    SdpaDeltas d = compute_deltas(1, 0, 4, 4, true, SubChunk::Full);
    check_set("N4 LB r1 i0 Q->{2,3,12,13}", global_set(4, d.q), std::set<int>{2,3,12,13});
  }

  if (g_failures == 0) {
    std::printf("\nALL cp_rope_deltas tests passed.\n");
    return 0;
  }
  std::printf("\n%d cp_rope_deltas test(s) FAILED.\n", g_failures);
  return 1;
}

#pragma once

// ---------------------------------------------------------------------------
// HybridUlyssesBackward.h
//
// Autograd nodes for the INNER Ulysses all-to-all transforms used by the 3-D
// hybrid context-parallel path (ContextParallel::enable_hybrid): Ulysses
// (all-to-all, inner) nested inside Ring (P2P rotation, outer).
//
// Unlike the single-axis Ulysses path (forward_ulysses + UlyssesAttentionBackward,
// one monolithic node covering combine -> SDPA -> partition), the hybrid path
// COMPOSES: combine -> [ring forward_cp, which builds its own ContextParallelBackward]
// -> partition. For autograd to chain through, the combine and partition
// all-to-alls must each be their own graph node so the ring's backward slots in
// the middle. These two nodes provide exactly that.
//
// Each all-to-all is a pure permutation, so its adjoint is the inverse permute
// with no scaling:
//   - d/d(combine input)   == partition(grad)        (UlyssesCombineBackward)
//   - d/d(partition input) == combine(grad)          (UlyssesPartitionBackward)
// Both re-use the EXISTING raw collectives in UlyssesAttention.h -- no new comm.
//
// The process group passed here is the ULYSSES sub-group (the inner all-to-all
// group, e.g. intra-node), NOT the ring group. P == ulysses degree U.
// ---------------------------------------------------------------------------

#include "autograd/Node.h"
#include "core/Tensor.h"
#include "ProcessGroupNCCL.h"   // canonical BluTrain PG (via -IBluTrain/dist/communication/include)

#include "context_parallel/UlyssesAttention.h"

#include <memory>
#include <stdexcept>
#include <vector>

using namespace OwnTensor;

// Backward for a standalone ulysses_combine (seq-sharded -> head-sharded).
//   forward:  in [B,H,Tl,D]  ->  out [B,Hl,T,D]
//   backward: d_in = partition(d_out)  [B,Hl,T,D] -> [B,H,Tl,D]
class UlyssesCombineBackward : public Node {
public:
  UlyssesCombineBackward(std::shared_ptr<ProcessGroupNCCL> pg, int P)
      : Node(1), pg_(pg), P_(P) {}

  const char *name() const override { return "UlyssesCombineBackward"; }

  std::vector<Tensor> apply(std::vector<Tensor> &&grads) override {
    if (grads.empty() || !grads[0].is_valid())
      throw std::runtime_error("UlyssesCombineBackward: no gradient provided");
    Tensor g = grads[0].contiguous();                 // [B,Hl,T,D]
    return {cp::ulysses_partition(pg_, g, P_)};        // -> [B,H,Tl,D]
  }

private:
  std::shared_ptr<ProcessGroupNCCL> pg_;
  int P_;
};

// Backward for a standalone ulysses_partition (head-sharded -> seq-sharded).
//   forward:  in [B,Hl,T,D]  ->  out [B,H,Tl,D]
//   backward: d_in = combine(d_out)  [B,H,Tl,D] -> [B,Hl,T,D]
class UlyssesPartitionBackward : public Node {
public:
  UlyssesPartitionBackward(std::shared_ptr<ProcessGroupNCCL> pg, int P)
      : Node(1), pg_(pg), P_(P) {}

  const char *name() const override { return "UlyssesPartitionBackward"; }

  std::vector<Tensor> apply(std::vector<Tensor> &&grads) override {
    if (grads.empty() || !grads[0].is_valid())
      throw std::runtime_error("UlyssesPartitionBackward: no gradient provided");
    Tensor g = grads[0].contiguous();                 // [B,H,Tl,D]
    return {cp::ulysses_combine(pg_, g, P_)};          // -> [B,Hl,T,D]
  }

private:
  std::shared_ptr<ProcessGroupNCCL> pg_;
  int P_;
};

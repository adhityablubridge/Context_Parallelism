#pragma once

// ---------------------------------------------------------------------------
// UlyssesAttention.h
//
// DeepSpeed-style Ulysses sequence-parallel attention layout transforms
// (additive; opt-in via ContextParallel::enable_ulysses()).
//
// Ulysses is an ALTERNATIVE to ring attention: instead of rotating K/V around
// a ring, it does a single all-to-all that swaps a sequence-sharded layout for
// a head-sharded one, runs ONE ordinary full-sequence causal SDPA over the
// local head group, then all-to-alls back. No ring, no LSE merge, no zig-zag
// load balancing. Positions are trivial: after the gather every rank holds the
// full contiguous sequence 0..T-1 for its head group.
//
// This header provides the two layout transforms (combine / partition) and a
// plain rank-ordered sequence gather, all as RAW collectives + reshapes (no
// autograd graph). Gradient flow is handled by UlyssesAttentionBackward, which
// re-uses these same helpers in reverse order (a pure permutation's adjoint is
// the inverse permutation). The local attention re-uses the EXISTING CP kernels
// via sdpa_fused_forward / sdpa_fused_backward (FusedSDPAOp.h) -- no new kernel.
//
// Shapes (MHA): P = world_size, Tl = T/P (local seq), Hl = H/P (local heads).
//   combine   : [B, H,  Tl, D]  ->  [B, Hl, T,  D]
//   partition : [B, Hl, T,  D]  ->  [B, H,  Tl, D]   (exact inverse of combine)
//   gather_seq: [B, H,  Tl, D]  ->  [B, H,  T,  D]   (plain rank-order concat)
//
// The uniform equal-split alltoall (ncclAllToAll) sends an equal block to each
// peer; count = B*Hl*Tl*D elements per peer.
// ---------------------------------------------------------------------------

#include "core/Tensor.h"
#include "ProcessGroupNCCL.h"   // canonical BluTrain PG (via -IBluTrain/dist/communication/include)

#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

namespace OwnTensor {
namespace cp {

// seq-sharded -> head-sharded.  in:[B,H,Tl,D] (contiguous)  ->  out:[B,Hl,T,D]
inline Tensor ulysses_combine(const std::shared_ptr<ProcessGroupNCCL> &pg,
                              const Tensor &in, int P) {
  const auto &d = in.shape().dims;
  const int64_t B = d[0], H = d[1], Tl = d[2], D = d[3];
  if (H % P != 0)
    throw std::runtime_error("ulysses_combine: H must be divisible by world_size");
  const int64_t Hl = H / P;
  const int64_t T = Tl * P;

  // 1. expose the destination-rank axis on the head dim (group g -> rank g),
  //    then make the P axis outermost so each rank's block is contiguous.
  Tensor s = in.reshape(Shape({{B, P, Hl, Tl, D}}))   // split H = P*Hl
                 .transpose(0, 1)                       // [P, B, Hl, Tl, D]
                 .contiguous();

  // 2. uniform all-to-all: send block r to rank r; recv block s from rank s.
  Tensor r = Tensor::empty(Shape({{(int64_t)P, B, Hl, Tl, D}}), in.opts());
  const size_t count = static_cast<size_t>(B * Hl * Tl * D);
  pg->alltoall(s.data<float>(), r.data<float>(), count, in.dtype(), /*sync=*/true);

  // 3. merge (src, Tl) -> full contiguous T (src in rank order == global order).
  return r.transpose(0, 1)        // [B, src, Hl, Tl, D]
           .transpose(1, 2)       // [B, Hl, src, Tl, D]
           .contiguous()
           .reshape(Shape({{B, Hl, T, D}}));
}

// head-sharded -> seq-sharded.  in:[B,Hl,T,D] (contiguous)  ->  out:[B,H,Tl,D]
// Exact structural inverse of ulysses_combine.
inline Tensor ulysses_partition(const std::shared_ptr<ProcessGroupNCCL> &pg,
                                const Tensor &in, int P) {
  const auto &d = in.shape().dims;
  const int64_t B = d[0], Hl = d[1], T = d[2], D = d[3];
  if (T % P != 0)
    throw std::runtime_error("ulysses_partition: T must be divisible by world_size");
  const int64_t Tl = T / P;
  const int64_t H = Hl * P;

  // 1. expose the destination-rank axis on the seq dim (slice p -> rank p),
  //    then make the P axis outermost so each rank's block is contiguous.
  Tensor s = in.reshape(Shape({{B, Hl, (int64_t)P, Tl, D}}))  // split T = P*Tl
                 .transpose(0, 2)                              // [P, Hl, B, Tl, D]
                 .transpose(1, 2)                              // [P, B, Hl, Tl, D]
                 .contiguous();

  // 2. uniform all-to-all.
  Tensor r = Tensor::empty(Shape({{(int64_t)P, B, Hl, Tl, D}}), in.opts());
  const size_t count = static_cast<size_t>(B * Hl * Tl * D);
  pg->alltoall(s.data<float>(), r.data<float>(), count, in.dtype(), /*sync=*/true);

  // 3. src axis (= head group s) folds back into H: group s -> heads [s*Hl:(s+1)*Hl].
  return r.transpose(0, 1)        // [B, src=group, Hl, Tl, D]
           .contiguous()
           .reshape(Shape({{B, H, Tl, D}}));
}

// Plain rank-ordered sequence gather.  local:[B,H,Tl,D]  ->  full:[B,H,T,D]
// NO unloadbalance / de-zigzag: Ulysses shards contiguously, so a straight
// rank-order concatenation along the seq dim restores the global order. This
// mirrors all_gather_along_seq (ContextParallelBackward.h) WITHOUT the ring
// path's conditional unloadbalance.
inline Tensor ulysses_gather_seq(const std::shared_ptr<ProcessGroupNCCL> &pg,
                                 const Tensor &local, int P) {
  const auto &d = local.shape().dims;
  const int64_t B = d[0], H = d[1], Tl = d[2], D = d[3];
  const int64_t T = Tl * P;

  const size_t local_count = static_cast<size_t>(local.numel());
  Tensor gathered_flat =
      Tensor::empty(Shape({{(int64_t)(local_count * (size_t)P)}}), local.opts());
  pg->all_gather(local.data<float>(), gathered_flat.data<float>(), local_count,
                 local.dtype(), /*sync=*/true);

  Tensor full = Tensor::empty(Shape({{B, H, T, D}}), local.opts());
  const size_t slice_bytes = static_cast<size_t>(Tl * D) * sizeof(float);
  for (int r = 0; r < P; ++r) {
    for (int64_t b = 0; b < B; ++b) {
      for (int64_t h = 0; h < H; ++h) {
        float *src = gathered_flat.data<float>() + (int64_t)r * (B * H * Tl * D) +
                     b * (H * Tl * D) + h * (Tl * D);
        float *dst = full.data<float>() + b * (H * T * D) + h * (T * D) +
                     (int64_t)r * (Tl * D);
        cudaMemcpyAsync(dst, src, slice_bytes, cudaMemcpyDeviceToDevice, 0);
      }
    }
  }
  cudaStreamSynchronize(0);
  return full;
}

// ---------------------------------------------------------------------------
// GQA head helpers (additive; used only by the GQA Ulysses path). OwnTensor has
// no repeat_interleave, so build it from narrow+cat and its adjoint from
// reshape+reduce_sum. Both are RAW (no autograd) — the GQA backward node calls
// them manually.
// ---------------------------------------------------------------------------

// repeat_interleave along the head dim (dim 1):
//   in  [B, H,   T, D]  ->  out [B, H*r, T, D], where out head e == in head (e / r).
// r == 1 is a no-op (returns the input) so the MHA path stays byte-identical.
inline Tensor head_repeat_interleave(const Tensor &x, int r) {
  if (r <= 1) return x;
  Tensor xm = x; // narrow() is non-const; shallow handle copy shares storage
  const int64_t H = xm.shape().dims[1];
  std::vector<Tensor> parts;
  parts.reserve(static_cast<size_t>(H) * r);
  for (int64_t h = 0; h < H; ++h)
    for (int j = 0; j < r; ++j)
      parts.push_back(xm.narrow(1, h, 1)); // [B, 1, T, D] view
  return Tensor::cat(parts, 1).contiguous(); // [B, H*r, T, D]
}

// Adjoint of head_repeat_interleave: sum each contiguous group of r heads.
//   in [B, groups*r, T, D]  ->  out [B, groups, T, D].
// r == 1 is a no-op. Implemented as an explicit sum of the r sub-slices via the
// raw Tensor operator+ (same primitive the ring backward uses for grad accum) to
// avoid any middle-axis reduce_sum ambiguity.
inline Tensor head_group_reduce(const Tensor &x, int64_t groups, int r) {
  if (r <= 1) return x;
  const auto &d = x.shape().dims;
  const int64_t B = d[0], T = d[2], D = d[3];
  Tensor g5 = x.reshape(Shape({{B, groups, static_cast<int64_t>(r), T, D}}));
  Tensor acc = g5.narrow(2, 0, 1).contiguous(); // [B, groups, 1, T, D]
  for (int j = 1; j < r; ++j)
    acc = acc + g5.narrow(2, j, 1).contiguous(); // raw elementwise add
  return acc.reshape(Shape({{B, groups, T, D}})).contiguous();
}

} // namespace cp
} // namespace OwnTensor

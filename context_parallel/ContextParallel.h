#pragma once

#include "autograd/AutogradOps.h"
#include "autograd/operations/BinaryOps.h"
#include "autograd/operations/ReshapeOps.h"
#include "core/Tensor.h"
#include "dtype/DtypeTraits.h"   // get_dtype_name() -- DTYPE_PROBE diagnostics only
// #include "dnn/DistributedNN.h"
#include "ops/IndexingOps.h"
#include "ProcessGroupNCCL.h"   // canonical BluTrain PG (via -IBluTrain/dist/communication/include)
#include "process_group/device_mesh.h"
#include "context_parallel/LoadBalancer.h"

#include "context_parallel/ContextParallelBackward.h"
#include "context_parallel/FusedSDPAOp.h"
#include "context_parallel/KVPackKernel.h"
#include "context_parallel/RingRotator.h"
#include "context_parallel/SDPAMerger.h"
#include "context_parallel/SDPAOp.h"
#include "context_parallel/RopeDeltas.h"      // 4-delta RoPE position helper (additive)
#include "context_parallel/FusedRoPESDPA.h"   // sdpa_fused_forward_rope (gated by CP_FUSED_ROPE)
#include "context_parallel/UlyssesAttention.h"        // Ulysses all-to-all layout transforms (additive)
#include "context_parallel/UlyssesAttentionBackward.h" // Ulysses backward node (additive)
#include "context_parallel/HybridUlyssesBackward.h"    // hybrid (Ulysses-inner) combine/partition backward nodes
#include "context_parallel/UlyssesGQAAttentionBackward.h" // Ulysses GQA backward node (additive)
#include "context_parallel/UlyssesFusedGQAAttentionBackward.h" // Ulysses fused RoPE+QKnorm+GQA node (additive)

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <nvtx3/nvToolsExt.h>

using namespace OwnTensor;

// ---------------------------------------------------------------------------
// RotatorType
//
// Selects which ring communication strategy to use.
// ---------------------------------------------------------------------------
enum class RotatorType {
  P2P,      // Point-to-point ncclSend/ncclRecv
  AlltoAll, // sendrecv-based ring shift
  AllGather // Single all_gather, then index
};

// ---------------------------------------------------------------------------
// CausalBehavior
//
// Controls how causal masking is applied during ring attention.
//
// FULL_CAUSAL:  Apply causal mask on every ring step (correct for
//               chunks where q_idx >= k_idx in the global sequence).
// SKIP_FUTURE:  Skip SDPA entirely for ring steps where the K,V chunk
//               is entirely in the "future" relative to the Q chunk.
// NOT_CAUSAL:   No causal masking (bidirectional attention).
// ---------------------------------------------------------------------------
enum class CausalBehavior { FULL_CAUSAL, SKIP_FUTURE, NOT_CAUSAL };

// ---------------------------------------------------------------------------
// Pre-embedding sequence sharding helper.
//
// Mirrors PyTorch's context_parallel() buffer pre-sharding: takes full-T
// token ids (and optional targets) on the device, returns this rank's
// [B, T/n] slice + matching [1, T/n] position indices.
//
//   load_balance=false: contiguous chunk [rank*T_local, (rank+1)*T_local)
//   load_balance=true : HeadTail permutation (out[2k]=k, out[2k+1]=T-1-k),
//                       then take this rank's contiguous slice.
//
// Note: HeadTail mode produces a runnable pipeline (shapes + grads OK) but
// the loss is NOT numerically correct for causal attention until the ring
// loop's causal mask is upgraded to the 3-mask sub-chunk scheme. Contiguous
// mode is fully correct.
// ---------------------------------------------------------------------------
struct ShardedInputs {
  Tensor idx_local;  // [B, T/n] int64
  Tensor pos_local;  // [1, T/n] int64
  Tensor y_local;    // [B, T/n] int64 (invalid Tensor if y_full was invalid)
};

inline ShardedInputs shard_sequence_pre_embed(
    const Tensor &idx_full,
    const Tensor &y_full,
    int64_t T_full,
    int world_size,
    int rank,
    bool load_balance,
    DeviceIndex device) {
  if (T_full % world_size != 0) {
    throw std::invalid_argument(
        "shard_sequence_pre_embed: T_full must be divisible by world_size");
  }
  int64_t T_local = T_full / world_size;
  int64_t B = idx_full.shape().dims[0];

  // Build per-rank absolute position list on CPU.
  //
  // Two layouts:
  //   load_balance=false: contiguous chunk [rank*T_local, (rank+1)*T_local).
  //   load_balance=true : PyTorch chunk-level HeadTail. Sequence is split into
  //     2*world_size equal chunks of size chunk_sz = T_full/(2*N). Rank r owns
  //     chunks (r, 2N-1-r), concatenated as [head_chunk, tail_chunk]. This
  //     gives each rank a contiguous "early" half and a contiguous "late" half
  //     so the round-robin sub-chunk causal dispatch in forward_cp works
  //     correctly (first half = early global positions, second half = late).
  //     Matches PyTorch _attention.py @ e9ebbd3b _rearrange_seq_for_load_balance.
  //
  // TODO(headtail-consolidation): This CPU-side perm_local construction
  // duplicates the chunk-level math now implemented in
  // tensor/headtail_kernel.cu (HeadTail::loadbalance). Both MUST stay
  // in sync: chunk_sz = T/(2*N), rank r owns chunks (r, 2N-1-r)
  // concatenated as [head_chunk, tail_chunk]. A future cleanup should
  // replace this block with a call to HeadTail::loadbalance followed by
  // make_shards_inplace_axis. Until then, any change to the kernel's
  // permutation semantics must be mirrored here.
  std::vector<int64_t> perm_local(static_cast<size_t>(T_local));
  if (!load_balance) {
    for (int64_t i = 0; i < T_local; ++i) {
      perm_local[static_cast<size_t>(i)] = rank * T_local + i;
    }
  } else {
    if (T_full % (2 * static_cast<int64_t>(world_size)) != 0) {
      throw std::invalid_argument(
          "shard_sequence_pre_embed: HeadTail requires T_full divisible by 2*world_size");
    }
    int64_t chunk_sz = T_full / (2 * static_cast<int64_t>(world_size));
    int64_t head_chunk = rank;
    int64_t tail_chunk = 2 * static_cast<int64_t>(world_size) - 1 - rank;
    for (int64_t i = 0; i < chunk_sz; ++i) {
      perm_local[static_cast<size_t>(i)] = head_chunk * chunk_sz + i;
      perm_local[static_cast<size_t>(chunk_sz + i)] = tail_chunk * chunk_sz + i;
    }
  }

  // pos_local: [1, T_local] int64 on device.
  Tensor pos_cpu(Shape{{1, T_local}}, TensorOptions().with_dtype(Dtype::Int64));
  std::memcpy(pos_cpu.data(), perm_local.data(),
              static_cast<size_t>(T_local) * sizeof(int64_t));
  Tensor pos_local = pos_cpu.to(device);

  ShardedInputs out;
  out.pos_local = pos_local;

  if (!load_balance) {
    // Contiguous: zero-copy view + contiguous materialization.
    auto idx_chunks = const_cast<Tensor &>(idx_full)
                          .make_shards_inplace_axis(
                              static_cast<size_t>(world_size), 1);
    out.idx_local = idx_chunks[static_cast<size_t>(rank)].contiguous();
    if (y_full.is_valid()) {
      auto y_chunks = const_cast<Tensor &>(y_full)
                          .make_shards_inplace_axis(
                              static_cast<size_t>(world_size), 1);
      out.y_local = y_chunks[static_cast<size_t>(rank)].contiguous();
    }
  } else {
    // HeadTail: build [B, T_local] gather index and gather along axis 1.
    Tensor gidx_cpu(Shape{{B, T_local}},
                    TensorOptions().with_dtype(Dtype::Int64));
    int64_t *gp = static_cast<int64_t *>(gidx_cpu.data());
    for (int64_t b = 0; b < B; ++b) {
      std::memcpy(gp + b * T_local, perm_local.data(),
                  static_cast<size_t>(T_local) * sizeof(int64_t));
    }
    Tensor gather_idx = gidx_cpu.to(device);
    out.idx_local = OwnTensor::gather(idx_full, /*dim=*/1, gather_idx);
    if (y_full.is_valid()) {
      out.y_local = OwnTensor::gather(y_full, /*dim=*/1, gather_idx);
    }
  }

  return out;
}

// ---------------------------------------------------------------------------
// Hybrid 3-D sharding (Ulysses-inner + Ring-outer). Composes the ring HeadTail
// zigzag (OUTER, over ring_size) with a contiguous Ulysses sub-slice (INNER,
// over ulysses_size). Rank (ring_rank r, ulysses_rank u) loads
// T/(ring_size*ulysses_size) tokens = contiguous piece u of ring block r's
// [head_half_r | tail_half_r] buffer. After the inner ulysses_combine all-to-all
// reassembles the U pieces in rank order, the ring sees the full T/ring_size
// block in HeadTail layout -- exactly what compute_deltas(N=ring_size) expects.
// Targets (y) use the IDENTICAL composed perm.
//
// Divisibility: T_full % (2*ring_size) == 0 (ring HeadTail) AND
//               (T_full/ring_size) % ulysses_size == 0 (contiguous inner slice).
// ---------------------------------------------------------------------------
inline ShardedInputs shard_sequence_pre_embed_hybrid(
    const Tensor &idx_full,
    const Tensor &y_full,
    int64_t T_full,
    int ring_size,
    int ring_rank,
    int ulysses_size,
    int ulysses_rank,
    DeviceIndex device) {
  const int64_t R = static_cast<int64_t>(ring_size);
  const int64_t U = static_cast<int64_t>(ulysses_size);
  if (T_full % (2 * R) != 0)
    throw std::invalid_argument(
        "shard_sequence_pre_embed_hybrid: T_full must be divisible by 2*ring_size (HeadTail)");
  const int64_t T_ring = T_full / R; // tokens per ring block
  if (T_ring % U != 0)
    throw std::invalid_argument(
        "shard_sequence_pre_embed_hybrid: (T_full/ring_size) must be divisible by ulysses_size");
  const int64_t Tl = T_ring / U; // tokens per rank = T/(R*U)
  const int64_t B = idx_full.shape().dims[0];

  // 1. ring HeadTail perm for ring block `ring_rank` (length T_ring), [head|tail].
  const int64_t chunk_sz = T_full / (2 * R);
  const int64_t head_chunk = static_cast<int64_t>(ring_rank);
  const int64_t tail_chunk = 2 * R - 1 - static_cast<int64_t>(ring_rank);
  std::vector<int64_t> perm_ring(static_cast<size_t>(T_ring));
  for (int64_t i = 0; i < chunk_sz; ++i) {
    perm_ring[static_cast<size_t>(i)] = head_chunk * chunk_sz + i;
    perm_ring[static_cast<size_t>(chunk_sz + i)] = tail_chunk * chunk_sz + i;
  }
  // 2. contiguous ulysses sub-slice: piece `ulysses_rank` of the ring block.
  std::vector<int64_t> perm_local(static_cast<size_t>(Tl));
  for (int64_t i = 0; i < Tl; ++i)
    perm_local[static_cast<size_t>(i)] =
        perm_ring[static_cast<size_t>(ulysses_rank) * Tl + i];

  // pos_local: [1, Tl] int64 on device.
  Tensor pos_cpu(Shape{{1, Tl}}, TensorOptions().with_dtype(Dtype::Int64));
  std::memcpy(pos_cpu.data(), perm_local.data(),
              static_cast<size_t>(Tl) * sizeof(int64_t));
  ShardedInputs out;
  out.pos_local = pos_cpu.to(device);

  // idx_local / y_local via gather along axis 1 with the composed perm.
  Tensor gidx_cpu(Shape{{B, Tl}}, TensorOptions().with_dtype(Dtype::Int64));
  int64_t *gp = static_cast<int64_t *>(gidx_cpu.data());
  for (int64_t b = 0; b < B; ++b)
    std::memcpy(gp + b * Tl, perm_local.data(),
                static_cast<size_t>(Tl) * sizeof(int64_t));
  Tensor gather_idx = gidx_cpu.to(device);
  out.idx_local = OwnTensor::gather(idx_full, /*dim=*/1, gather_idx);
  if (y_full.is_valid())
    out.y_local = OwnTensor::gather(y_full, /*dim=*/1, gather_idx);

  return out;
}

// ---------------------------------------------------------------------------
// ContextParallel : DModule
//
// Implements context parallelism (ring attention) as a distributed module.
//
// Forward:
//   1. Shard the input along the sequence dimension across ranks
//      (with optional load balancing via HeadTail permutation)
//   2. Each rank holds Q for its local chunk; K,V rotate around the ring
//   3. At each ring step: compute local SDPA, merge results via SDPAMerger
//   4. After all steps: unshard (gather) the output back to full sequence
//
// The module does NOT own any parameters -- it wraps the attention
// computation pattern for distributed execution.
// ---------------------------------------------------------------------------
class ContextParallel {
public:
  ContextParallel(const DeviceMesh &mesh, std::shared_ptr<ProcessGroupNCCL> pg,
                  float attn_scale, bool is_causal = true,
                  RotatorType rotator_type = RotatorType::P2P,
                  bool load_balance = true, bool recompute_k = false)
      : mesh_(&mesh), pg_(pg), attn_scale_(attn_scale), is_causal_(is_causal),
        rotator_type_(rotator_type), load_balance_(load_balance),
        recompute_k_(recompute_k),
        world_size_(pg->get_worldsize()), rank_(pg->get_rank()) {}

  // Forward ring staging that can be SHARED across layers. The forward is
  // sequential (layer i finishes before layer i+1 starts), so one persistent
  // rotator + send_buf set can serve all layers instead of one set per layer
  // (the per-layer persistence costs ~25MB*2*n_layers resident). Backward
  // rotators are NOT shared (two are live at once there). When a layer has a
  // shared holder set, forward_cp uses it in place of its own per-instance
  // members; otherwise behavior is unchanged. Gated by CP_SHARE_FWD_ROTATOR
  // in the model builder.
  struct SharedFwdRing {
    std::unique_ptr<RingRotatorBase> rotator;
    Tensor send_buf[2];
    int64_t kv_numel = -1;
  };
  void set_shared_fwd_ring(std::shared_ptr<SharedFwdRing> r) {
    shared_fwd_ring_ = std::move(r);
  }

  // Opt this CP layer into the FUSED RoPE attention path (additive). When set,
  // the forward ring loop derives 4 per-side position deltas and calls
  // sdpa_fused_forward_rope(...) instead of the scalar-offset sdpa_fused_forward.
  // Left unset (default), behavior is byte-identical to the GPT-2/wpe path.
  // cache: [cache_len, hd] cos/sin; q_gamma/k_gamma: [hd] QK-norm scales
  // (pass invalid Tensors to skip QK-norm).
  void enable_rope(const Tensor &cos_sin_cache, const Tensor &q_gamma,
                   const Tensor &k_gamma, float eps = 1e-6f) {
    cos_sin_cache_ = cos_sin_cache;
    q_gamma_ = q_gamma;
    k_gamma_ = k_gamma;
    rope_eps_ = eps;
    use_rope_ = true;
  }

  // Opt this CP layer into the DeepSpeed-style ULYSSES attention path (additive).
  // When set, forward_cp dispatches to forward_ulysses (a single all-to-all layout
  // swap + one full-sequence causal SDPA + all-to-all back) instead of the ring.
  // Left unset (default), behavior is byte-identical to the ring path.
  // v1: MHA only (H % world_size == 0), contiguous sharding (no HeadTail),
  // blocking all-to-all. pre_sharded inputs are rejected (see forward_ulysses).
  void enable_ulysses() { use_ulysses_ = true; }

  // Opt this CP layer into the FUSED GQA Ulysses path (additive; Llama-style):
  // RoPE + QK-norm + GQA are done INSIDE the team's gqa_fused_flash_attn kernel
  // (bf16). Reached only when use_ulysses_ is also set (forward_ulysses dispatches
  // to forward_ulysses_fused first). Gammas are all-or-nothing: pass BOTH q_gamma
  // and k_gamma valid, or BOTH invalid (QK-norm off). cache: [cache_len>=T, hd]
  // fp32 cos/sin from autograd::build_rope_cache.
  void enable_ulysses_fused(const Tensor &cos_sin_cache, const Tensor &q_gamma,
                            const Tensor &k_gamma, float eps = 1e-6f,
                            bool interleaved = false) {
    if (q_gamma.is_valid() != k_gamma.is_valid())
      throw std::runtime_error(
          "enable_ulysses_fused: q_gamma and k_gamma must both be valid or both "
          "invalid (gammas are an all-or-nothing pair)");
    cos_sin_cache_f_ = cos_sin_cache;
    q_gamma_f_ = q_gamma;
    k_gamma_f_ = k_gamma;
    eps_f_ = eps;
    interleaved_f_ = interleaved;
    use_ulysses_fused_ = true;
    use_ulysses_ = true; // route forward_cp -> forward_ulysses -> fused
  }

  // Opt this CP layer into the 3-D HYBRID path (additive; Ulysses-inner nested in
  // Ring-outer). This instance's own pg_ is the RING sub-group (size R); the
  // Ulysses all-to-all runs over the separate ulysses_pg (size U). Attention flow:
  //   seq-shard [B,H,T/(R*U),D]
  //     --combine(ulysses_pg,U)-->  head-shard [B,H/U,T/R,D]
  //     --ring forward_cp (pre_sharded, RoPE)-->  [B,H/U,T/R,D]
  //     --partition(ulysses_pg,U)-->  seq-shard [B,H,T/(R*U),D]
  // The ring stage IS a fused-RoPE ring, so this stores the same cache/gammas as
  // enable_rope() and sets use_rope_. The caller MUST pre-shard the sequence over
  // BOTH CP axes (shard_sequence_pre_embed_hybrid) and construct this instance
  // with load_balance=true (HeadTail on the ring axis) and pg = ring sub-group.
  void enable_hybrid(std::shared_ptr<ProcessGroupNCCL> ulysses_pg, int ulysses_size,
                     const Tensor &cos_sin_cache, const Tensor &q_gamma,
                     const Tensor &k_gamma, float eps = 1e-6f) {
    // Ring stage = fused-RoPE ring: reuse the enable_rope() state exactly.
    cos_sin_cache_ = cos_sin_cache;
    q_gamma_ = q_gamma;
    k_gamma_ = k_gamma;
    rope_eps_ = eps;
    use_rope_ = true;
    // Inner Ulysses all-to-all group.
    ulysses_pg_ = std::move(ulysses_pg);
    ulysses_size_ = ulysses_size;
    use_hybrid_ = true;
  }

  // -----------------------------------------------------------------------
  // forward
  //
  // Input: DTensor containing the full Q, K, V concatenated or separate.
  //        For this implementation, we expect q, k, v as separate Tensors
  //        already shaped as [B, H, T, D] (4D).
  //
  // This overload takes raw Tensors and returns the merged attention output.
  // -----------------------------------------------------------------------
  // unshard: when true (default), all-gather output to full [B,H,T,D].
  //          when false, output stays [B,H,T/n,D]; downstream layers work on
  //          the local sequence chunk with no inter-rank communication.
  // pre_sharded: when true, q/k/v are already [B,H,T/n,D] (e.g. from a
  //              previous layer that returned unshard=false). Phase 1 sharding
  //              is skipped. Must be paired with unshard=false.
  Tensor forward_cp(Tensor &q, // [B, H, T, D] -- full sequence query
                    Tensor &k, // [B, H, T, D] -- full sequence key
                    Tensor &v, // [B, H, T, D] -- full sequence value
                    bool unshard = true,
                    bool pre_sharded = false)
  {
    // ----- Ulysses opt-in gate (additive) -----
    // When enabled, take the all-to-all Ulysses path and bypass the ring
    // entirely. The ring code below is left untouched for the default path.
    if (use_ulysses_)
      return forward_ulysses(q, k, v, unshard, pre_sharded);

    // ----- Hybrid opt-in gate (Ulysses-inner + Ring-outer) -----
    // forward_hybrid wraps the ring body (below) with the inner Ulysses
    // all-to-all. It re-enters forward_cp with in_hybrid_ring_=true so the
    // recursive call runs the RING body here instead of re-dispatching.
    if (use_hybrid_ && !in_hybrid_ring_)
      return forward_hybrid(q, k, v, unshard, pre_sharded);

    // ----- Phase 1: Context Parallel Shard -----
    // Shard Q along dim=2 (sequence dim in 4D [B, H, T, D])
    // K, V also sharded along dim=2
    //
    // Two independent flags now govern the LB pipeline:
    //   external_balanced  : pre-embedding HeadTail was already applied by
    //                        the caller (pre_sharded && load_balance_).
    //                        CP must NOT call its internal HeadTail kernel.
    //   sub_chunk_active   : round-robin Q/K/V sub-chunking + partial-merger
    //                        + pipelined dkv_rotater are active. Driven by
    //                        load_balance_ regardless of is_causal_, matching
    //                        PyTorch _attention.py @ e9ebbd3b.
    //
    // PyTorch dispatch table (enable_load_balance=True, is_causal=True):
    //   i == 0           : IS_CAUSAL,   full Q, full K/V, partial=false
    //   0 < i <= rank    : NOT_CAUSAL,  full Q, K/V[:T/2], partial=false
    //   i >  rank        : NOT_CAUSAL,  Q[T/2:], full K/V, partial=true
    //   never SKIP under LB.
    // HeadTail requires T divisible by 2*world_size (chunk-level layout has
    // chunk_sz = T/(2*N)). Generation paths pass variable, often-small T
    // (e.g. T=10 during incremental token generation) that may not satisfy
    // this. When the internal kernel would be invoked (pre_sharded=false)
    // and divisibility fails, fall back to a non-LB path for this call.
    // PyTorch has the same divisibility constraint.
    const int64_t T_full = q.shape().dims[2];
    const bool lb_div_ok = (T_full % (2 * static_cast<int64_t>(world_size_)) == 0);
    const bool lb_effective = load_balance_ && (pre_sharded || lb_div_ok);

    const bool external_balanced = pre_sharded && lb_effective;
    const bool sub_chunk_active = lb_effective;

    // Q can stay strided: sdpa_fused_forward / sdpa_fused_backward read
    // strides explicitly. Q is never sent across NCCL.
    //
    // K, V are kept contiguous: the ring rotator's cudaMemcpyAsync below
    // does a flat byte copy, and the strided pack-kernel alternative
    // (KVPackKernel) was measured slower than cudaMemcpyAsync on the
    // contig hot path. Keep contig here; pack kernel infrastructure remains
    // available for future use cases where source is genuinely strided.
    Tensor q_work = pre_sharded ? q : autograd::contiguous(q);
    Tensor k_work = autograd::contiguous(k);
    Tensor v_work = autograd::contiguous(v);

    if (lb_effective && !external_balanced) {
      load_balancer_.set_world_size(world_size_);
      load_balancer_.set_chunk_dim(2); // sequence dim in 4D
      load_balancer_.set_stream(0);
      load_balancer_.loadbalance(q_work);
      load_balancer_.loadbalance(k_work);
      load_balancer_.loadbalance(v_work);
    }

    // ----- Phase 1: Shard (skipped when pre_sharded=true) -----
    // When pre_sharded=true, q/k/v are already [B,H,T/n,D] local chunks.
    Tensor local_q, local_k, local_v;
    if (pre_sharded) {
      // q_work is already autograd::contiguous(q) — local_q IS q_work
      local_q = q_work;
      local_k = k_work;
      local_v = v_work;
    } else {
      std::vector<Tensor> q_chunks =
          q_work.make_shards_inplace_axis(static_cast<size_t>(world_size_), 2);
      std::vector<Tensor> k_chunks =
          k_work.make_shards_inplace_axis(static_cast<size_t>(world_size_), 2);
      std::vector<Tensor> v_chunks =
          v_work.make_shards_inplace_axis(static_cast<size_t>(world_size_), 2);

      local_q = autograd::contiguous(q_chunks[rank_]); // [B, H, T/n, D]
      local_k = autograd::contiguous(k_chunks[rank_]); // [B, H, T/n, D]
      local_v = autograd::contiguous(v_chunks[rank_]); // [B, H, T/n, D]
    }

    // DUMP_CP_DEEP_FWD alignment: fire the per-step deep dumps ONLY on the very
    // first forward_cp invocation (block 0, step 0) so two separate runs
    // (overlap vs CP_SYNC_RING) dump the SAME deterministic call with identical
    // loaded-init inputs. Without this gate the dump overwrites every call and
    // two processes capture different last-calls (even different shapes), making
    // the diff meaningless.
    static std::atomic<int> _deep_call_idx{0};
    const int my_call_idx = _deep_call_idx.fetch_add(1);
    const bool deep_first_call = (my_call_idx == 0);

    // ----- Phase 2: Ring Attention Loop -----
    // Rotator for K,V communication. PERSISTENT across forward_cp calls (bound
    // below, once the per-rank KV size is known). Rationale: with a fresh
    // rotator + stack send_buf per call, the ring staging churned through the
    // caching allocator, and under steady-state pressure the per-step SDPA
    // OUTPUT buffer could be handed a block physically overlapping an in-flight
    // ring buffer -> the concurrent ring write clobbered the attention output
    // (diagnosed: lse stayed bit-identical, out did not; only the V/output
    // pathway was corrupted; draining the ring before SDPA fixed it). Persistent
    // ring buffers never share a block with per-step SDPA allocations.
    RingRotatorBase *kv_rotator = nullptr;

    // Initialize merger for accumulating partial attention outputs
    SDPAMerger merger(/*convert_to_f32=*/true);

    // Save K,V chunks, causal flags, partial flags, and per-step LSE for backward
    std::vector<Tensor> saved_k_chunks(world_size_);
    std::vector<Tensor> saved_v_chunks(world_size_);
    std::vector<bool> saved_causal_flags(world_size_, false);
    std::vector<bool> saved_partial_flags(world_size_, false);
    std::vector<Tensor> saved_lse_per_step(world_size_);
    std::vector<Tensor> saved_out_per_step(world_size_);

    // Per-step (out, lse) are consumed by the backward ONLY under
    // USE_PER_STEP_LSE=1; the default backward path uses merged_out/merged_lse.
    // Saving them per ring step otherwise is dead weight retained from forward
    // through backward across every layer (the dominant CP memory overhead at
    // large T). Gate the save on the same env so the default path keeps these
    // vectors empty and frees that memory. (Backward indexes them only when the
    // same env is set, so empty is safe.)
    const bool save_per_step_stats = std::getenv("USE_PER_STEP_LSE") != nullptr;

    // Sequence length of each rank's local chunk
    int64_t T_local_fwd = local_q.shape().dims[2];
    const int seq_dim = 2; // [B, H, T, D]

    // ---- Compute/comm overlap setup ----
    // CP_NO_OVERLAP=1 reverts to the serialized (CPU-blocking) path for A/B.
    // Value-aware: unset OR "0" => overlap ON; any other value => OFF.
    const char *_cp_no_ovl = std::getenv("CP_NO_OVERLAP");
    bool OVERLAP =
        (_cp_no_ovl == nullptr) || (_cp_no_ovl[0] == '0' && _cp_no_ovl[1] == '\0');
    // Independent FORWARD overlap gate (for bisecting fwd-vs-bwd races):
    // CP_NO_OVERLAP_FWD=1 forces the forward ring onto the blocking/CPU path
    // even when overlap is on.
    const char *_no_fwd = std::getenv("CP_NO_OVERLAP_FWD");
    if (_no_fwd != nullptr && !(_no_fwd[0] == '0' && _no_fwd[1] == '\0'))
        OVERLAP = false;
    cudaStream_t compute_stream = OwnTensor::cuda::getCurrentStream();

    int64_t k_numel = local_k.numel();
    int64_t kv_numel = k_numel * 2;
    // [#6] Persistent double-buffered send staging + rotator, keyed on the
    // per-rank KV size. Reused across forward_cp calls; reallocated only when the
    // size changes (e.g. training T vs generation T). Each slot holds K AND V
    // (kv_numel each). Persistence is the fix for the forward-overlap race: it
    // stops the ring buffers churning through the allocator so they can never
    // share a caching-allocator block with the per-step SDPA output.
    // Select the forward ring staging: shared (one set across all layers, when
    // shared_fwd_ring_ is set) or per-instance (one set per layer, default).
    // Both paths keep the buffers PERSISTENT (re-allocated only on a kv_numel
    // change), so neither churns the allocator. The shared path simply makes a
    // single set serve every layer (forward is sequential), saving the
    // ~kv_numel*2 * n_layers resident staging.
    Tensor *send_buf;
    if (shared_fwd_ring_) {
      SharedFwdRing &R = *shared_fwd_ring_;
      if (R.kv_numel != kv_numel) {
        R.send_buf[0] = Tensor::empty(Shape({{kv_numel}}), local_k.opts());
        R.send_buf[1] = Tensor::empty(Shape({{kv_numel}}), local_k.opts());
        R.rotator = create_rotator(); // fresh recv_ slots for new size
        R.kv_numel = kv_numel;
      }
      kv_rotator = R.rotator.get();
      send_buf = R.send_buf;
    } else {
      if (persistent_kv_numel_ != kv_numel) {
        send_buf_[0] = Tensor::empty(Shape({{kv_numel}}), local_k.opts());
        send_buf_[1] = Tensor::empty(Shape({{kv_numel}}), local_k.opts());
        kv_rotator_persistent_ = create_rotator(); // fresh recv_ slots for new size
        persistent_kv_numel_ = kv_numel;
      }
      kv_rotator = kv_rotator_persistent_.get();
      send_buf = send_buf_; // alias so downstream send_buf[s] is unchanged
    }
    // Re-arm the rotator for THIS forward's ring (per layer, per microstep).
    // No-op for P2P/AlltoAll (they re-communicate fresh data every exchange).
    // Critical for AllGather: its rotator caches a one-shot all_gather and would
    // otherwise serve the FIRST step's stale K/V forever (with a drifting
    // (rank - idx_) shard index) across the reused persistent/shared instance.
    // reset() invalidates that cache so the next exchange re-gathers the CURRENT
    // step's K/V, while keeping the buffer storage and the persistent pack-event
    // (no allocator churn, no event-destroy UB). With a SHARED rotator this also
    // re-arms it cleanly for each layer's independent ring.
    kv_rotator->reset();
    std::shared_ptr<Work> exch_work[2] = {nullptr, nullptr};  // [#1] per send slot
    // Pack-ordering events are OWNED by the rotator (persistent, per-slot).
    // Never create/destroy them per call: destroying an event with a pending
    // cudaStreamWaitEvent referencing it is unconditional UB (it was a race at
    // EVERY step; shallow pipelines merely won the timing).

    // Current K, V being processed (starts with local chunk)
    Tensor curr_k = local_k;
    Tensor curr_v = local_v;

    for (int i = 0; i < world_size_; ++i) {
      // Step 1: If not first iteration, get K,V from previous exchange.
      // GPU-side wait (no CPU stall) when overlapping; the recv was posted in
      // step i-1 and overlapped that step's SDPA.
      if (i > 0) {
        Tensor next_kv = OVERLAP
            ? kv_rotator->next_buffer_streamordered(compute_stream)
            : kv_rotator->next_buffer();
        // DIAGNOSTIC (CP_FWD_SYNC_RECV=1): real host-side sync of the ring
        // stream immediately after consuming the received K/V, BEFORE the SDPA
        // reads it. Isolates the recv->SDPA ordering: next_buffer_streamordered
        // uses cudaStreamWaitEvent (GPU-side) to order the recv ahead of the
        // SDPA; if that event ordering is ineffective, the SDPA reads recv_[slot]
        // before the NCCL recv lands. If this flag flattens the norm (with send
        // overlap still on), the broken link is the recv-consume event ordering,
        // and the fix belongs in next_buffer_streamordered -- NOT a blanket ring
        // sync.
        const char *_fwd_sync_recv = std::getenv("CP_FWD_SYNC_RECV");
        if (_fwd_sync_recv && _fwd_sync_recv[0] == '1')
          cudaStreamSynchronize(pg_->cpRingStream());
        Tensor kv_flat = next_kv.flatten();
        curr_k = kv_flat.narrow(0, 0, k_numel).reshape(local_k.shape());
        curr_v = kv_flat.narrow(0, k_numel, k_numel).reshape(local_v.shape());
      }

      // Step 2: Send current K,V to next rank (async; recv lands in the OTHER
      // ping-pong slot so it overlaps this step's compute without aliasing).
      if (i < (world_size_ - 1)) {
        int s = i & 1;
        // [#1] Before reusing send_buf[s] (last used at step i-2), ensure that
        // send has drained reading it. GPU-side; usually already complete.
        if (OVERLAP && exch_work[s]) exch_work[s]->streamWait(compute_stream);
        size_t k_bytes = static_cast<size_t>(k_numel) * sizeof(float);
        cudaMemcpyAsync(send_buf[s].data<float>(), curr_k.data<float>(),
                        k_bytes, cudaMemcpyDeviceToDevice, compute_stream);
        cudaMemcpyAsync(send_buf[s].data<float>() + k_numel,
                        curr_v.data<float>(), k_bytes, cudaMemcpyDeviceToDevice,
                        compute_stream);
        nvtxRangePushA("CP.fwd.ring.exchange_buffers");
        // Explicit overlap flag + the pack (compute) stream. The flag is
        // separate because compute may run on the legacy NULL stream (0),
        // which cannot double as a "no overlap" sentinel.
        exch_work[s] =
            kv_rotator->exchange_buffers(send_buf[s], OVERLAP, compute_stream);
        nvtxRangePop();
      }

      // Step 3: Determine causal behavior for this ring step
      // With load balance: i==0 is causal, all others are NOT_CAUSAL (never SKIP)
      // Without load balance: i==0 is causal, past chunks full attention, future skipped
      bool skip_step = false;
      bool use_causal = false;
      if (is_causal_) {
        if (i == 0) {
          use_causal = true;
        } else if (sub_chunk_active) {
          // Load balance enabled: never skip, never causal on non-diagonal
          use_causal = false;
        } else {
          // No load balance: skip future chunks
          int source_rank =
              ((rank_ - i) % world_size_ + world_size_) % world_size_;
          if (source_rank > rank_) {
            skip_step = true;
          }
          use_causal = false;
        }
      }

      if (skip_step) {
        continue;
      }

      // Step 4: Sub-chunk Q, K, V for load-balanced dispatch
      // Matches PyTorch _templated_ring_attention lines 451-471:
      //   i==0:       full Q, K, V (causal)
      //   i<=rank LB: full Q, K[:T/2], V[:T/2] (not causal, partial=false)
      //   i>rank  LB: Q[T/2:], full K, V (not causal, partial=true)
      Tensor q_use = local_q;
      Tensor k_use = curr_k;
      Tensor v_use = curr_v;
      bool use_partial = false;

      if (sub_chunk_active && i > 0) {
        if (i <= rank_) {
          // Past chunk with LB: full Q, first half of K/V.
          // sdpa_fused_forward reads per-tensor B/M/H strides from Q/K/V, so a
          // strided half-view from make_shards_inplace_axis can be passed
          // directly. Avoids the materialization copy that previously dominated
          // the LB strided_copy_kernel time in nsys.
          std::vector<Tensor> k_halves =
              curr_k.make_shards_inplace_axis(2, seq_dim);
          std::vector<Tensor> v_halves =
              curr_v.make_shards_inplace_axis(2, seq_dim);
          k_use = k_halves[0];
          v_use = v_halves[0];
        } else {
          // Future chunk with LB: second half of Q, full K/V.
          // Q strides are read from the tensor by the kernel — half-view OK.
          std::vector<Tensor> q_halves =
              local_q.make_shards_inplace_axis(2, seq_dim);
          q_use = q_halves[1];
          use_partial = true;
        }
      }

      // Save full (un-chunked) K,V for backward before computing.
      // When recompute_k_=true, only save step 0 (local K,V) as the starting
      // point for backward re-rotation. Other steps are recomputed.
      if (!recompute_k_ || i == 0) {
        saved_k_chunks[i] = curr_k.clone();
        saved_v_chunks[i] = curr_v.clone();
      }
      saved_causal_flags[i] = use_causal;
      saved_partial_flags[i] = use_partial;

      // DIAGNOSTIC (CP_DRAIN_AT=N): localize WHICH step-0 stream-0 op races the
      // concurrent i==0 ring send/recv, by bisecting the position of a real
      // ring-stream drain. Drain at P fixes the explosion iff the racing op runs
      // AFTER P. Endpoints already known: drain-before-everything (CP_SYNC_RING)
      // fixes; drain-after-all-step-0 (CP_FWD_SYNC_RECV) does not; SDPA output
      // ruled out by CP_SELFCHECK. So the victim is the clone (above) or the
      // merger.step (below). Points:
      //   1 = here, AFTER the backward-save clone, BEFORE SDPA
      //   2 = after SDPA, before merger.step
      //   3 = after merger.step
      // Run N=1,2,3; the LARGEST N that flattens the norm => racing op is the
      //   one immediately after that drain. (N=1 fixes => merger; N=1 fails =>
      //   clone is the victim.)
      const char *_drain = std::getenv("CP_DRAIN_AT");
      if (_drain && _drain[0] == '1')
        cudaStreamSynchronize(pg_->cpRingStream());

      // Step 5: Compute fused SDPA
      // With LB sub-chunking: no cross-chunk causal offsets needed (is_causal
      // only on diagonal, sub-chunks handle the rest).
      // Without LB: offsets needed for cross-chunk causal masking.
      int q_off = 0;
      int k_off = 0;
      if (!sub_chunk_active && is_causal_) {
        int source_rank =
            ((rank_ - i) % world_size_ + world_size_) % world_size_;
        q_off = rank_ * static_cast<int>(T_local_fwd);
        k_off = source_rank * static_cast<int>(T_local_fwd);
      }
      // [KERNEL SHAPE CHECK] one-time, first few SDPA calls on rank 0: print the
      // ACTUAL q/k/v shapes fed to the attention kernel. Confirms the per-rank
      // local seqlen (and HeadTail sub-chunks) reaching the kernel, not full T.
      static int kshape_prints = 0;
      if (std::getenv("CP_DEBUG_SHAPES") && rank_ == 0 && kshape_prints < 4) {
        ++kshape_prints;
        std::cout << "[KERNEL SHAPE CHECK] ring_step i=" << i
                  << " q_use=[" << q_use.shape().dims[0] << ","
                  << q_use.shape().dims[1] << "," << q_use.shape().dims[2] << ","
                  << q_use.shape().dims[3] << "]"
                  << " k_use seqlen=" << k_use.shape().dims[2]
                  << " v_use seqlen=" << v_use.shape().dims[2]
                  << " causal=" << (use_causal ? 1 : 0)
                  << " partial=" << (use_partial ? 1 : 0) << "\n";
      }
      // RoPE path (additive, gated): when this CP layer wraps the fused
      // RoPE kernel, derive the 4 per-side deltas and call the RoPE wrapper.
      // Default (use_rope_==false, GPT-2/wpe) takes the EXACT existing call.
      //
      // >>> CONTRACT with the fused CP kernel (GQA_fused_*_sm103_cp.cu): the
      // >>> kernel does causal masking on LOCAL indices only. That is correct
      // >>> ONLY because this sub-chunk slicing (Full / KHeadHalf / QTailHalf)
      // >>> keeps each ring step's tensor LOCALLY MONOTONIC (local order ==
      // >>> relative global order) and sets use_causal per step. If you change
      // >>> how sub-chunks are sliced here, you MUST re-confirm that invariant
      // >>> or the fused kernel's mask becomes SILENTLY wrong. The 4 deltas
      // >>> below drive RoPE cache indexing ONLY, never masking. <<<
      SDPAResult result;
      if (use_rope_) {
        OwnTensor::cp::SubChunk sc =
            (i == 0) ? OwnTensor::cp::SubChunk::Full
                     : (sub_chunk_active
                            ? (i <= rank_ ? OwnTensor::cp::SubChunk::KHeadHalf
                                          : OwnTensor::cp::SubChunk::QTailHalf)
                            : OwnTensor::cp::SubChunk::Full);
        OwnTensor::cp::SdpaDeltas dl = OwnTensor::cp::compute_deltas(
            rank_, i, world_size_, static_cast<int>(T_local_fwd),
            /*lb=*/sub_chunk_active, sc);
        result = sdpa_fused_forward_rope(
            q_use, k_use, v_use, use_causal, attn_scale_,
            dl.q.d0, dl.q.d1, dl.k.d0, dl.k.d1,
            cos_sin_cache_, q_gamma_, k_gamma_, rope_eps_);
      } else {
        result = sdpa_fused_forward(
            q_use, k_use, v_use, use_causal, attn_scale_, q_off, k_off);
      }

      // CP_DRAIN_AT=2: drain AFTER SDPA, before merger.step (see point 1 note).
      {
        const char *_d2 = std::getenv("CP_DRAIN_AT");
        if (_d2 && _d2[0] == '2')
          cudaStreamSynchronize(pg_->cpRingStream());
      }

      // DIAGNOSTIC (CP_SELFCHECK=1): single-run, weight-drift-immune race
      // detector. The SDPA above just read curr_k/curr_v as the overlap
      // scheduling delivered them (possibly before the NCCL recv landed). Now
      // force the ring stream fully done, recompute the SAME SDPA on the SAME
      // buffers (guaranteed-correct data), and diff. If the recv ordering held,
      // the two outputs are bit-identical; any nonzero diff at step i means the
      // first SDPA read not-yet-arrived K -> pinpoints the race in ONE run,
      // immune to cross-run weight drift. Logs the first offending (global call,
      // step) with the max-abs diff. Heavy: use only for diagnosis.
      {
        // NOTE: checks ALL steps including i==0. i==0 is the local causal SDPA
        // that runs CONCURRENTLY with the just-posted i==0 ring send/recv -- the
        // i>0-only earlier version skipped exactly this overlap window.
        const char *_sc = std::getenv("CP_SELFCHECK");
        if (_sc && _sc[0] == '1') {
          // ARTIFACT FIX: snapshot result.out/lse to HOST *before* the recompute.
          // The earlier version compared result.out vs ref.out AFTER ref ran, so
          // ref's own fresh out/lse/workspace allocations could clobber result.out
          // and fake a huge diff (we saw out_diff~1.8, lse_diff=0 -- the signature
          // of the OUTPUT buffer, not the inputs, being overwritten by ref). By
          // copying to host first, ref cannot touch what we compare. h_out/h_lse
          // are the genuine SDPA(i) output as produced under overlap; we then
          // recompute a clean reference (ring drained) and diff. Any nonzero diff
          // now means the SDPA output truly differs with vs without the ring
          // co-executing -> real forward-output corruption, not an artifact.
          cudaStreamSynchronize(0); // finish this step's SDPA on the compute stream
          Tensor h_out = result.out.to_cpu();
          Tensor h_lse = result.lse.to_cpu();
          // Clean reference: ring fully drained, recompute same SDPA.
          cudaStreamSynchronize(pg_->cpRingStream());
          cudaStreamSynchronize(0);
          SDPAResult ref = sdpa_fused_forward(
              q_use, k_use, v_use, use_causal, attn_scale_, q_off, k_off);
          cudaStreamSynchronize(0);
          Tensor h_ref_out = ref.out.to_cpu();
          Tensor h_ref_lse = ref.lse.to_cpu();
          auto maxdiff = [](const Tensor &ha, const Tensor &hb) -> double {
            const float *pa = ha.data<float>();
            const float *pb = hb.data<float>();
            int64_t n = std::min<int64_t>(ha.numel(), hb.numel());
            double mx = 0.0;
            for (int64_t j = 0; j < n; ++j) {
              double d = std::fabs((double)pa[j] - (double)pb[j]);
              if (d > mx) mx = d;
            }
            return mx;
          };
          double mx_out = maxdiff(h_out, h_ref_out);
          double mx_lse = maxdiff(h_lse, h_ref_lse);
          if (mx_out > 0.0 || mx_lse > 0.0) {
            static std::atomic<int> _sc_call{0};
            int c = _sc_call.fetch_add(1);
            std::cerr << "[CP_SELFCHECK] RACE call=" << c << " rank=" << rank_
                      << " step=" << i
                      << " maxabs_out_diff=" << mx_out
                      << " maxabs_lse_diff=" << mx_lse << "\n";
            // Persist the FIRST hit per rank to a file so it can be read without
            // catching a scrolling terminal. Includes magnitudes: a tiny out
            // diff (~1e-6) is kernel FP nondeterminism (noise); a large one
            // (>1e-3) is real corruption. Also records the reference magnitude
            // so the relative size is judgeable.
            static std::atomic<bool> _sc_written{false};
            bool expected = false;
            if (_sc_written.compare_exchange_strong(expected, true)) {
              const float *pro = h_ref_out.data<float>();
              double refmax = 0.0;
              for (int64_t j = 0; j < h_ref_out.numel(); ++j)
                if (std::fabs((double)pro[j]) > refmax) refmax = std::fabs((double)pro[j]);
              std::string path =
                  "/tmp/cp_selfcheck_rank" + std::to_string(rank_) + ".log";
              std::ofstream f(path);
              f << "FIRST RACE HIT\n"
                << "call=" << c << " rank=" << rank_ << " step=" << i << "\n"
                << "maxabs_out_diff=" << mx_out << "\n"
                << "maxabs_lse_diff=" << mx_lse << "\n"
                << "ref_out_maxabs=" << refmax << "\n"
                << "rel_out_diff=" << (mx_out / (refmax + 1e-30)) << "\n";
            }
          }
        }
      }

      // Save per-step LSE and OUT for backward. Backward calls
      // sdpa_fused_backward with these per-step values so the kernel's
      // D = sum(out*dout) and softmax(QK^T - lse) are computed against the
      // SAME K used in this step, not the final merged values (which
      // accumulate contributions from other ranks' K and corrupt the partial
      // step's gradient computation).
      if (save_per_step_stats) {
        saved_lse_per_step[i] = result.lse;
        saved_out_per_step[i] = result.out;
      }

      // Step 6: Merge into accumulator (with partial flag)
      merger.step(result.out, result.lse, use_partial);

      // CP_DRAIN_AT=3: drain AFTER merger.step (see point 1 note).
      {
        const char *_d3 = std::getenv("CP_DRAIN_AT");
        if (_d3 && _d3[0] == '3')
          cudaStreamSynchronize(pg_->cpRingStream());
      }

      // DUMP_CP_DEEP_FWD=1: per-step merged_out/merged_lse + per-step
      // SDPA result (block_out/block_lse). Lets us pin which ring step
      // (i=0 full merge or i>=1 partial-tail merge) introduces drift.
      {
        const char *env = std::getenv("DUMP_CP_DEEP_FWD");
        if (env && env[0] == '1' && deep_first_call) {
          auto [cur_out, cur_lse] = merger.results();
          auto save_bin = [&](const char *label, const Tensor &t) {
            Tensor host = t.to_cpu();
            std::string path = std::string("/tmp/cp_bwd_test/deep/cpp_") +
                               label + "_fwdstep" + std::to_string(i) +
                               "_rank" + std::to_string(rank_) + ".bin";
            std::ofstream fout(path, std::ios::binary);
            fout.write(reinterpret_cast<const char *>(host.data<float>()),
                       host.numel() * sizeof(float));
          };
          save_bin("block_out", result.out);
          save_bin("block_lse", result.lse);
          save_bin("merged_out", cur_out);
          save_bin("merged_lse", cur_lse);
        }
      }
    }

    // cudaEvent_t memcpy_event;
    // cudaEventCreate(&memcpy_event);
    // cudaEventRecord(memcpy_event, compute_stream);
    // cudaStreamWaitEvent(pg_->getStream(), memcpy_event, 0);

    // [#2] Order the compute stream after ALL pending ring sends at loop exit.
    // Coverage audit: WITHIN the loop, each slot's send is guarded before
    // re-pack (exch_work[s]->streamWait above), so at loop exit at most the last
    // two sends (one per slot) can still be in flight — and we order against
    // BOTH slots here, not just the final one.
    //
    // LIFETIME SAFETY IS NOT PROVIDED BY THIS DRAIN. Only the send_buf[2] Tensor
    // HANDLES are stack-local; their device storage is caching-allocator memory
    // (Tensor::empty), not a CPU stack buffer. So the destructor at function
    // return does NOT free GPU memory directly — it routes to
    // CachingCUDAAllocator::deallocate(), which at CudaCachingAllocator.cpp:362
    // checks block->recorded_streams: exchange_buffers() called
    // recordStream(send_buf.data(), cp_ring_stream_) in the overlap path
    // (RingRotator.h), so deallocate takes the DEFERRED branch — insert_events()
    // records an event on cp_ring_stream_ (ordered after the send) and the block
    // is returned to the free pool only once process_events() sees that event
    // complete. A later (e.g. backward) allocation therefore cannot reuse this
    // memory while the send is still reading it.
    // Forward sends and the backward dkv_rotater both ride the SINGLE shared
    // cp_ring_stream_, so they additionally serialize on that one stream. The
    // reuse-clobber race earlier revisions of this comment feared cannot occur
    // given recordStream; this loop is therefore NOT a use-after-free guard.
    //
    // What the streamWait below DOES do: it makes the compute stream wait on the
    // tail sends so subsequent compute-stream work is ordered after them. A
    // CPU-blocking wait() is unnecessary for correctness (recordStream covers
    // buffer lifetime); streamWait keeps the tail GPU-ordered without a CPU stall.
    if (OVERLAP) {
      for (int s = 0; s < 2; ++s)
        if (exch_work[s]) exch_work[s]->streamWait(compute_stream);
    }

    // DIAGNOSTIC (CP_DUMP_KV_CHUNKS=1): cross-rank send/recv content check.
    // Dump this rank's per-step saved K chunks on the FIRST forward_cp call.
    // For ws=2: saved_k_chunks[0] = this rank's LOCAL K (what it sent),
    // saved_k_chunks[1] = the K it RECEIVED from the peer. The transfer is
    // correct iff rank0's received-K equals rank1's sent-K and vice versa:
    //   diff  cpp_kvchunk_k1_rank0.bin  (rank0 received)
    //         cpp_kvchunk_k0_rank1.bin  (rank1 sent)
    //   diff  cpp_kvchunk_k1_rank1.bin  vs  cpp_kvchunk_k0_rank0.bin
    // Settled host snapshot (to_cpu syncs), so this catches CONTENT corruption
    // of the transferred bytes (recv timing already ruled out by CP_FWD_SYNC_RECV).
    {
      // Fire at a configurable call index so we can inspect STEADY STATE (near
      // the ~650-step onset), not just the first call. CP_DUMP_KV_CALL=N => dump
      // once on the first forward_cp call with index >= N (default 0 = first).
      // Cross-rank comparison is valid at ANY step (both ranks are at the same
      // step of the same run), so no weight-drift concern.
      const char *_dkv = std::getenv("CP_DUMP_KV_CHUNKS");
      const char *_dkv_call = std::getenv("CP_DUMP_KV_CALL");
      int dkv_threshold = _dkv_call ? std::atoi(_dkv_call) : 0;
      static std::atomic<bool> _dkv_done{false};
      bool dkv_fire = false;
      if (_dkv && _dkv[0] == '1' && my_call_idx >= dkv_threshold) {
        bool expected = false;
        dkv_fire = _dkv_done.compare_exchange_strong(expected, true);
      }
      if (dkv_fire) {
        std::cerr << "[CP_DUMP_KV_CHUNKS] dumping at call=" << my_call_idx
                  << " rank=" << rank_ << "\n";
        cudaStreamSynchronize(pg_->cpRingStream());
        cudaStreamSynchronize(0);
        for (int s = 0; s < world_size_; ++s) {
          if (!saved_k_chunks[s].is_valid()) continue;
          Tensor h = saved_k_chunks[s].to_cpu();
          std::string path = std::string("/tmp/cp_bwd_test/cpp_kvchunk_k") +
                             std::to_string(s) + "_rank" +
                             std::to_string(rank_) + ".bin";
          std::ofstream f(path, std::ios::binary);
          f.write(reinterpret_cast<const char *>(h.data<float>()),
                  h.numel() * sizeof(float));
        }
      }
    }

    // ----- Phase 3: Get final merged result -----
    auto [merged_out, merged_lse] = merger.results();

    // DUMP_CP_DEEP_FWD=1: save merged_out and merged_lse as .bin files for
    // PT-vs-C++ parity diff of the forward outputs that feed backward.
    {
      const char *env = std::getenv("DUMP_CP_DEEP_FWD");
      if (env && env[0] == '1' && deep_first_call) {
        auto save_bin = [&](const char *label, const Tensor &t) {
          Tensor host = t.to_cpu();
          std::string path = std::string("/tmp/cp_bwd_test/deep/cpp_") + label +
                             "_rank" + std::to_string(rank_) + ".bin";
          std::ofstream fout(path, std::ios::binary);
          fout.write(reinterpret_cast<const char *>(host.data<float>()),
                     host.numel() * sizeof(float));
        };
        save_bin("merged_out", merged_out);
        save_bin("merged_lse", merged_lse);
      }
    }

    // Optional first-call dump for PT parity probe (gated by DUMP_CP_OUT=1).
    // Only fires on the very first forward_cp invocation in this process,
    // which corresponds to block 0, step 0, micro 0.
    {
      static std::atomic<int> _cp_call_idx{0};
      int call_idx = _cp_call_idx.fetch_add(1);
      const char *env = std::getenv("DUMP_CP_OUT");
      if (call_idx == 0 && env && env[0] == '1') {
        Tensor host = merged_out.to_cpu();
        const float *p = host.data<float>();
        int run_idx = 0;
        while (true) {
          std::string probe = "block0_merged_out_rank" + std::to_string(rank_) +
                              "_" + std::to_string(run_idx) + ".md";
          std::ifstream check(probe);
          if (!check.good()) break;
          ++run_idx;
        }
        std::string path = "block0_merged_out_rank" + std::to_string(rank_) +
                           "_" + std::to_string(run_idx) + ".md";
        std::ofstream df(path);
        const auto &dims = merged_out.shape().dims;
        df << "shape=[" << dims[0] << "," << dims[1] << "," << dims[2] << ","
           << dims[3] << "]\n";
        df << "first16: [";
        int64_t n = std::min<int64_t>(16, merged_out.numel());
        for (int64_t i = 0; i < n; ++i) {
          df << p[i] << (i == n - 1 ? "" : ", ");
        }
        df << "]\n";
      }
    }

    // ----- Phase 4: Context Parallel Unshard -----
    // When unshard=false: skip allgather; return merged_out [B,H,T/n,D].
    // Downstream layers (MLP, LayerNorm, loss) work on the local T/n chunk.
    int64_t B = local_q.shape().dims[0];
    int64_t H = local_q.shape().dims[1];
    int64_t T_local = local_q.shape().dims[2];
    int64_t D = local_q.shape().dims[3];

    Tensor output_tensor;
    if (!unshard) {
      // No allgather: output stays [B, H, T/n, D]
      output_tensor = merged_out;
    } else {
      // Gather the output chunks back to full [B, H, T, D]
      size_t local_count = static_cast<size_t>(merged_out.numel());
      size_t total_count = local_count * static_cast<size_t>(world_size_);

      Shape gathered_shape({{static_cast<int64_t>(total_count)}});
      Tensor gathered_flat = Tensor::empty(gathered_shape, merged_out.opts());

      nvtxRangePushA("CP.fwd.unshard.all_gather");
      pg_->all_gather(merged_out.data<float>(), gathered_flat.data<float>(),
                      local_count, merged_out.dtype(),
                      true); // sync
      nvtxRangePop();

      int64_t T_full = T_local * world_size_;
      Shape full_shape({{B, H, T_full, D}});
      Tensor full_output = Tensor::empty(full_shape, merged_out.opts());

      size_t slice_bytes = static_cast<size_t>(T_local * D) * sizeof(float);
      for (int r = 0; r < world_size_; ++r) {
        for (int64_t b = 0; b < B; ++b) {
          for (int64_t h = 0; h < H; ++h) {
            float *src =
                gathered_flat.data<float>() + r * (B * H * T_local * D) +
                b * (H * T_local * D) + h * (T_local * D);
            float *dst = full_output.data<float>() + b * (H * T_full * D) +
                         h * (T_full * D) + r * (T_local * D);
            cudaMemcpyAsync(dst, src, slice_bytes, cudaMemcpyDeviceToDevice, 0);
          }
        }
      }
      cudaStreamSynchronize(0);

      // Undo internal HeadTail only when CP itself applied it. If the caller
      // pre-balanced the inputs externally (external_balanced), unshard output
      // is left in HeadTail order to match the caller's expected layout.
      if (lb_effective && !external_balanced) {
        load_balancer_.unloadbalance(full_output);
      }
      output_tensor = full_output;
    }

    // ----- Register backward node -----
    if (q.requires_grad() || k.requires_grad() || v.requires_grad()) {
      int rot_type = static_cast<int>(rotator_type_);
      // Pass merged_out.detach() to break the cycle:
      // output_tensor -> grad_fn -> merged_out -> output_tensor (when unshard=false
      // output_tensor IS merged_out). Detach severs the grad_fn strong-ptr path.
      auto grad_fn = std::make_shared<ContextParallelBackward>(
          local_q, saved_k_chunks, saved_v_chunks, saved_causal_flags,
          saved_partial_flags, saved_lse_per_step, saved_out_per_step,
          merged_lse, merged_out.detach(), pg_, attn_scale_, is_causal_,
          rot_type, load_balance_, world_size_, rank_, unshard, recompute_k_,
          sub_chunk_active, external_balanced,
          // RoPE opt-in (additive): forwards gammas/cache so the backward can
          // recompute RoPE+QK-norm and emit per-call gamma grads. All default
          // to invalid/false on the GPT-2/wpe path.
          use_rope_, q_gamma_, k_gamma_, cos_sin_cache_, rope_eps_);

      if (q.requires_grad()) {
        Tensor &q_mut = const_cast<Tensor &>(q);
        grad_fn->set_next_edge(0, autograd::get_grad_edge(q_mut));
      }
      if (k.requires_grad()) {
        Tensor &k_mut = const_cast<Tensor &>(k);
        grad_fn->set_next_edge(1, autograd::get_grad_edge(k_mut));
      }
      if (v.requires_grad()) {
        Tensor &v_mut = const_cast<Tensor &>(v);
        grad_fn->set_next_edge(2, autograd::get_grad_edge(v_mut));
      }
      // Gamma edges (3,4) only when RoPE is on AND the gammas are learnable.
      // Routes the node's dq_gamma/dk_gamma (outputs 3,4) to the parameters'
      // grad accumulation; from there the existing CP param all-reduce sums
      // them across ranks. Off by default => GPT-2/wpe node stays 3-output.
      if (use_rope_ && q_gamma_.is_valid() && k_gamma_.is_valid()) {
        if (q_gamma_.requires_grad())
          grad_fn->set_next_edge(3, autograd::get_grad_edge(q_gamma_));
        if (k_gamma_.requires_grad())
          grad_fn->set_next_edge(4, autograd::get_grad_edge(k_gamma_));
      }

      output_tensor.set_grad_fn(grad_fn);
      output_tensor.set_requires_grad(true);
    }

    // DIAGNOSTIC (CP_FWD_SYNC_RETURN=1): force the compute stream (0) to drain
    // before forward_cp returns and the local kv_rotator (with its recv_ slots)
    // destructs. Hypothesis under test: the recv buffers are recordStream'd only
    // on the ring stream (writer), NOT on stream 0 (the SDPA reader), so after
    // recv completes the allocator can recycle a recv block into the NEXT layer's
    // forward_cp while THIS layer's SDPA is still reading it on stream 0 -- a
    // cross-layer write-after-read that only manifests with forward overlap on.
    // If the norm explosion vanishes under this flag (forward overlap still on),
    // the consumer-lifetime gap is confirmed and the real fix is to recordStream
    // recv_[slot] on the compute stream in next_buffer_streamordered.
    {
      const char *_fwd_sync_ret = std::getenv("CP_FWD_SYNC_RETURN");
      if (_fwd_sync_ret && _fwd_sync_ret[0] == '1')
        cudaStreamSynchronize(0);
    }

    return output_tensor;
  }

private:
  // -----------------------------------------------------------------------
  // forward_ulysses (additive; reached only via the use_ulysses_ gate).
  //
  // DeepSpeed-style Ulysses sequence parallelism. Replaces the ring with a
  // single all-to-all layout swap, one full-sequence causal SDPA over the
  // local head group, and an all-to-all back. See UlyssesAttention.h.
  // v1: MHA only, contiguous sharding (no HeadTail), blocking all-to-all.
  // -----------------------------------------------------------------------
  // ---- Hybrid (Ulysses-inner + Ring-outer) helpers ----------------------

  // ulysses_combine with autograd: the adjoint is ulysses_partition (pure
  // permutation, no scaling). Mirrors the ring/ulysses grad-enable guard
  // (requires_grad(), NoGrad-safe) so validation builds no graph.
  Tensor combine_ag(const Tensor &x) {
    Tensor out = OwnTensor::cp::ulysses_combine(ulysses_pg_, x, ulysses_size_);
    if (x.requires_grad()) {
      auto gf = std::make_shared<UlyssesCombineBackward>(ulysses_pg_, ulysses_size_);
      Tensor &xm = const_cast<Tensor &>(x);
      gf->set_next_edge(0, autograd::get_grad_edge(xm));
      out.set_grad_fn(gf);
      out.set_requires_grad(true);
    }
    return out;
  }

  // ulysses_partition with autograd: the adjoint is ulysses_combine.
  Tensor partition_ag(const Tensor &x) {
    Tensor out = OwnTensor::cp::ulysses_partition(ulysses_pg_, x, ulysses_size_);
    if (x.requires_grad()) {
      auto gf = std::make_shared<UlyssesPartitionBackward>(ulysses_pg_, ulysses_size_);
      Tensor &xm = const_cast<Tensor &>(x);
      gf->set_next_edge(0, autograd::get_grad_edge(xm));
      out.set_grad_fn(gf);
      out.set_requires_grad(true);
    }
    return out;
  }

  // Risk C isolated check (CP_SELFCHECK-gated): the inner all-to-all must be an
  // exact inverse pair over ulysses_pg_. We check partition(combine(x)) == x
  // (a round-trip identity). NOTE: this catches all-to-all axis/permute bugs but
  // not a permutation that is consistently wrong in BOTH directions; the global
  // [head|tail] reassembly-order invariant is validated end-to-end by the
  // -np 8 hybrid parity test. Checked for q AND k/v because the KV path differs
  // when U > kv_heads (a kv head is replicated across ulysses ranks).
  void hybrid_selfcheck(const Tensor &x_local, const Tensor &xh, const char *tag) {
    Tensor rt = OwnTensor::cp::ulysses_partition(ulysses_pg_, xh, ulysses_size_);
    Tensor a = x_local.contiguous().to_cpu();
    Tensor b = rt.to_cpu();
    const float *pa = a.data<float>();
    const float *pb = b.data<float>();
    int64_t n = std::min<int64_t>(a.numel(), b.numel());
    float md = 0.0f;
    for (int64_t i = 0; i < n; ++i) md = std::max(md, std::abs(pa[i] - pb[i]));
    if (md > 1e-5f)
      std::cerr << "[CP_SELFCHECK hybrid] " << tag
                << " combine/partition round-trip mismatch maxdiff=" << md
                << " (ulysses_size=" << ulysses_size_ << ")" << std::endl;
  }

  // Turn the hybrid seq-sharded output into the caller's requested layout.
  // unshard=false (TRAINING): return the local [B,H,T/(R*U),D] shard as-is.
  // unshard=true  (inference/debug only, NoGrad): rebuild full [B,H,T,D]
  //   INNER-FIRST -- gather over Ulysses (rank-order concat, NO de-zigzag; rebuilds
  //   the full T/R ring block [head|tail]) THEN over Ring + unloadbalance de-zigzag.
  //   Ring-first would hand unloadbalance a partial T/(R*U) slice it cannot
  //   de-zigzag. Autograd through this gather is NOT supported (guarded below).
  Tensor finalize_hybrid_output(Tensor &out_seq, bool unshard,
                                const Tensor &grad_probe) {
    if (!unshard)
      return out_seq;
    if (out_seq.requires_grad() || grad_probe.requires_grad())
      throw std::runtime_error(
          "ContextParallel hybrid: unshard=true is not supported under autograd "
          "(training uses unshard=false). Wrap inference/validation in a "
          "NoGradGuard.");
    // 1. inner: rebuild the full T/R ring block from its U contiguous pieces.
    Tensor ring_block =
        OwnTensor::cp::ulysses_gather_seq(ulysses_pg_, out_seq, ulysses_size_);
    // 2. outer: gather the R ring blocks (rank-order concat = HeadTail-zigzag
    //    layout), then de-zigzag into global 0..T-1 order.
    Tensor full =
        OwnTensor::cp::ulysses_gather_seq(pg_, ring_block, world_size_);
    if (load_balance_) {
      load_balancer_.set_world_size(world_size_);
      load_balancer_.unloadbalance(full);
    }
    return full;
  }

  // Full hybrid forward: inner Ulysses all-to-all -> outer Ring -> all-to-all back.
  Tensor forward_hybrid(Tensor &q, Tensor &k, Tensor &v, bool unshard,
                        bool pre_sharded) {
    // Caller pre-shards the sequence over BOTH CP axes (the projections run on
    // the [B,H,T/(R*U),D] local shard); pre_sharded is thus expected true here.
    (void)pre_sharded;
    // INNER all-to-all (ulysses_pg_): seq-shard -> head-shard.
    Tensor qh = combine_ag(q); // [B,H/U,T/R,D]
    Tensor kh = combine_ag(k);
    Tensor vh = combine_ag(v);

#ifndef NDEBUG
    if (std::getenv("CP_SELFCHECK")) {
      hybrid_selfcheck(q, qh, "q");
      hybrid_selfcheck(k, kh, "k"); // covers U % kv == 0 replication
      hybrid_selfcheck(v, vh, "v");
    }
#endif

    // OUTER ring over pg_ (== ring sub-group). Re-enter forward_cp with the
    // guard set so the RING body runs (not another hybrid dispatch). Always
    // unshard=false / pre_sharded=true: the ring returns the local [B,H/U,T/R,D]
    // shard; full unshard (if any) is handled by finalize_hybrid_output over
    // both axes.
    in_hybrid_ring_ = true;
    Tensor oh;
    try {
      oh = forward_cp(qh, kh, vh, /*unshard=*/false, /*pre_sharded=*/true);
    } catch (...) {
      in_hybrid_ring_ = false;
      throw;
    }
    in_hybrid_ring_ = false;

    // INNER all-to-all back: head-shard -> seq-shard.
    Tensor out = partition_ag(oh); // [B,H,T/(R*U),D]
    return finalize_hybrid_output(out, unshard, q);
  }

  Tensor forward_ulysses(Tensor &q, Tensor &k, Tensor &v, bool unshard,
                         bool pre_sharded) {
    // Fused RoPE+QK-norm+GQA dispatch (additive gate): when enabled, use the
    // team's GQA-native fused kernel (handles both MHA and GQA). Checked first.
    if (use_ulysses_fused_)
      return forward_ulysses_fused(q, k, v, unshard, pre_sharded);

    // GQA/MQA dispatch (additive gate): when KV has fewer heads than Q, take the
    // separate GQA path. MHA (equal head counts) falls through to the unchanged
    // body below.
    if (k.shape().dims[1] != q.shape().dims[1])
      return forward_ulysses_gqa(q, k, v, unshard, pre_sharded);

    // pre_sharded inputs are accepted ONLY when this layer is NOT load-balanced.
    // Ring callers that load-balance hand in HeadTail/zig-zag-sharded tensors,
    // which would silently mis-order the gathered sequence. With load_balance_
    // off, the caller's shard is contiguous (rank r -> seq block r), exactly the
    // layout Ulysses needs. (Critique pt 1: this is a CHECKABLE guard, not a
    // blanket ban, because the layer knows its own load_balance_ flag.)
    if (pre_sharded && load_balance_)
      throw std::runtime_error(
          "Ulysses: pre_sharded inputs require load_balance=false (HeadTail/"
          "zig-zag shards would mis-order the gathered sequence).");

    const int P = world_size_;
    const auto &qd = q.shape().dims;
    const int64_t H = qd[1];
    if (H % P != 0)
      throw std::runtime_error("Ulysses: H must be divisible by world_size");
    if (k.shape().dims[1] != H || v.shape().dims[1] != H)
      throw std::runtime_error(
          "Ulysses v1: MHA only (q/k/v head counts must match)");

    // 1. Obtain this rank's contiguous local seq shard [B,H,Tl,D].
    Tensor ql, kl, vl;
    if (pre_sharded) {
      // q/k/v are already the local contiguous shard (the caller seq-sharded
      // contiguously). Use directly; do NOT re-shard.
      ql = autograd::contiguous(q);
      kl = autograd::contiguous(k);
      vl = autograd::contiguous(v);
    } else {
      // Full [B,H,T,D]: contiguous narrow this rank's seq block (NO zig-zag).
      const int64_t T = qd[2];
      if (T % P != 0)
        throw std::runtime_error("Ulysses: T must be divisible by world_size");
      const int64_t Tl = T / P;
      Tensor qc = autograd::contiguous(q);
      Tensor kc = autograd::contiguous(k);
      Tensor vc = autograd::contiguous(v);
      ql = qc.narrow(2, static_cast<int64_t>(rank_) * Tl, Tl).contiguous();
      kl = kc.narrow(2, static_cast<int64_t>(rank_) * Tl, Tl).contiguous();
      vl = vc.narrow(2, static_cast<int64_t>(rank_) * Tl, Tl).contiguous();
    }

    // 2. combine: seq-sharded -> head-sharded full sequence.
    Tensor qg = cp::ulysses_combine(pg_, ql, P); // [B,Hl,T,D]
    Tensor kg = cp::ulysses_combine(pg_, kl, P);
    Tensor vg = cp::ulysses_combine(pg_, vl, P);

    // 3. ONE full square causal SDPA -- reuse existing kernel (no offsets/merge).
    SDPAResult res = sdpa_fused_forward(qg, kg, vg, /*is_causal=*/true,
                                        attn_scale_, /*q_offset=*/0,
                                        /*k_offset=*/0);
    Tensor out_g = res.out; // [B,Hl,T,D]

    // 4. partition: head-sharded -> seq-sharded local output.
    Tensor out_l = cp::ulysses_partition(pg_, out_g, P); // [B,H,Tl,D]

    // 5. optional unshard: plain rank-order gather, NO unloadbalance de-zigzag.
    Tensor output_tensor =
        unshard ? cp::ulysses_gather_seq(pg_, out_l, P) // [B,H,T,D]
                : out_l;                                // [B,H,Tl,D]

    // 6. autograd wiring (mirror the ring pattern). The node saves the
    //    head-sharded tensors + lse; backward re-does the all-to-alls in reverse
    //    and branches on the persisted unshard flag.
    if (q.requires_grad() || k.requires_grad() || v.requires_grad()) {
      auto grad_fn = std::make_shared<UlyssesAttentionBackward>(
          qg, kg, vg, out_g, res.lse, pg_, attn_scale_, /*is_causal=*/true,
          world_size_, rank_, unshard, pre_sharded);
      if (q.requires_grad()) {
        Tensor &q_mut = const_cast<Tensor &>(q);
        grad_fn->set_next_edge(0, autograd::get_grad_edge(q_mut));
      }
      if (k.requires_grad()) {
        Tensor &k_mut = const_cast<Tensor &>(k);
        grad_fn->set_next_edge(1, autograd::get_grad_edge(k_mut));
      }
      if (v.requires_grad()) {
        Tensor &v_mut = const_cast<Tensor &>(v);
        grad_fn->set_next_edge(2, autograd::get_grad_edge(v_mut));
      }
      output_tensor.set_grad_fn(grad_fn);
      output_tensor.set_requires_grad(true);
    }

    return output_tensor;
  }

  // -----------------------------------------------------------------------
  // forward_ulysses_gqa (additive; reached only from forward_ulysses when
  // nkv < nq). DeepSpeed Strategy B: the KV all-to-all carries only nkv heads
  // (comm x1); the grouped broadcast is done LOCALLY (our CP SDPA is MHA-only).
  // When nkv < P, KV is partially replicated by rep = P/nkv so each rank gets
  // one KV head -- never expanded all the way to nq (that would be comm x g).
  // -----------------------------------------------------------------------
  Tensor forward_ulysses_gqa(Tensor &q, Tensor &k, Tensor &v, bool unshard,
                             bool pre_sharded) {
    if (pre_sharded && load_balance_)
      throw std::runtime_error(
          "Ulysses GQA: pre_sharded inputs require load_balance=false "
          "(HeadTail/zig-zag shards would mis-order the gathered sequence).");

    const int P = world_size_;
    const int64_t nq = q.shape().dims[1];
    const int64_t nkv = k.shape().dims[1];
    if (v.shape().dims[1] != nkv)
      throw std::runtime_error("Ulysses GQA: k/v head counts must match");
    if (nq % P != 0)
      throw std::runtime_error("Ulysses GQA: nq must be divisible by world_size");
    if (nkv == 0 || nq % nkv != 0)
      throw std::runtime_error("Ulysses GQA: nq must be divisible by nkv");
    if (!(nkv % P == 0 || P % nkv == 0))
      throw std::runtime_error(
          "Ulysses GQA: need (nkv % P == 0) or (P % nkv == 0) (DeepSpeed rule)");

    const int rep = (nkv < P) ? static_cast<int>(P / nkv) : 1;

    // 1. Obtain this rank's contiguous local shards (mirror MHA forward_ulysses).
    Tensor ql, kl, vl;
    if (pre_sharded) {
      ql = autograd::contiguous(q);
      kl = autograd::contiguous(k);
      vl = autograd::contiguous(v);
    } else {
      const int64_t T = q.shape().dims[2];
      if (T % P != 0)
        throw std::runtime_error("Ulysses GQA: T must be divisible by world_size");
      const int64_t Tl = T / P;
      Tensor qc = autograd::contiguous(q);
      Tensor kc = autograd::contiguous(k);
      Tensor vc = autograd::contiguous(v);
      ql = qc.narrow(2, static_cast<int64_t>(rank_) * Tl, Tl).contiguous();
      kl = kc.narrow(2, static_cast<int64_t>(rank_) * Tl, Tl).contiguous();
      vl = vc.narrow(2, static_cast<int64_t>(rank_) * Tl, Tl).contiguous();
    }

    // 2. Optional partial replication of KV BEFORE the all-to-all (only nkv < P).
    Tensor kl_r = cp::head_repeat_interleave(kl, rep); // [B, eff_kv, Tl, D]
    Tensor vl_r = cp::head_repeat_interleave(vl, rep);

    // 3. combine: Q by nq, KV by eff_kv (separate all-to-alls; KV comm x1).
    Tensor qg = cp::ulysses_combine(pg_, ql, P);   // [B, nq/P,     T, D]
    Tensor kg = cp::ulysses_combine(pg_, kl_r, P); // [B, kv_local, T, D]
    Tensor vg = cp::ulysses_combine(pg_, vl_r, P);
    const int64_t nq_local = qg.shape().dims[1];
    const int64_t kv_local = kg.shape().dims[1];
    const int g_local = static_cast<int>(nq_local / kv_local);

    // 4. Local broadcast kv_local -> nq_local (the "enable_gqa" broadcast, done
    //    explicitly), then ONE MHA SDPA on the existing kernel.
    Tensor kg_e = cp::head_repeat_interleave(kg, g_local); // [B, nq/P, T, D]
    Tensor vg_e = cp::head_repeat_interleave(vg, g_local);
    SDPAResult res = sdpa_fused_forward(qg, kg_e, vg_e, /*is_causal=*/true,
                                        attn_scale_, /*q_off=*/0, /*k_off=*/0);
    Tensor out_g = res.out; // [B, nq/P, T, D]

    // 5. partition Q-shaped output back to seq-sharded local.
    Tensor out_l = cp::ulysses_partition(pg_, out_g, P); // [B, nq, Tl, D]

    // 6. optional unshard: plain rank-order gather, NO unloadbalance de-zigzag.
    Tensor output_tensor =
        unshard ? cp::ulysses_gather_seq(pg_, out_l, P) : out_l;

    // 7. autograd wiring. The node saves UN-expanded kg/vg + group/replication
    //    params + all three flags (unshard_, pre_sharded_, is_causal_).
    if (q.requires_grad() || k.requires_grad() || v.requires_grad()) {
      auto grad_fn = std::make_shared<UlyssesGQAAttentionBackward>(
          qg, kg, vg, out_g, res.lse, pg_, attn_scale_, /*is_causal=*/true,
          world_size_, rank_, unshard, pre_sharded, g_local, kv_local, nkv, rep,
          nq);
      if (q.requires_grad()) {
        Tensor &q_mut = const_cast<Tensor &>(q);
        grad_fn->set_next_edge(0, autograd::get_grad_edge(q_mut));
      }
      if (k.requires_grad()) {
        Tensor &k_mut = const_cast<Tensor &>(k);
        grad_fn->set_next_edge(1, autograd::get_grad_edge(k_mut));
      }
      if (v.requires_grad()) {
        Tensor &v_mut = const_cast<Tensor &>(v);
        grad_fn->set_next_edge(2, autograd::get_grad_edge(v_mut));
      }
      output_tensor.set_grad_fn(grad_fn);
      output_tensor.set_requires_grad(true);
    }

    return output_tensor;
  }

  // -----------------------------------------------------------------------
  // forward_ulysses_fused (additive; reached only via enable_ulysses_fused()).
  // Llama-style: RoPE + QK-norm + GQA are fused INSIDE the team's bf16
  // gqa_fused_flash_attn kernel. The kernel is GQA-native (broadcasts KV by
  // hkv=hq/G), so there is NO local head broadcast; only the nkv<P replication
  // (for the all-to-all distribution) remains. After the combine gather each
  // rank holds the full contiguous sequence -> pos_offset = 0.
  // -----------------------------------------------------------------------
  Tensor forward_ulysses_fused(Tensor &q, Tensor &k, Tensor &v, bool unshard,
                               bool pre_sharded) {
    if (pre_sharded && load_balance_)
      throw std::runtime_error(
          "Ulysses fused: pre_sharded requires load_balance=false (zig-zag "
          "would mis-order the gathered sequence).");

    const int P = world_size_;
    const int64_t nq = q.shape().dims[1];
    const int64_t nkv = k.shape().dims[1];
    const int64_t hd = q.shape().dims[3];
    if (hd != 64 && hd != 128)
      throw std::runtime_error("Ulysses fused: fused kernel supports hd 64 or 128 only");
    if (v.shape().dims[1] != nkv)
      throw std::runtime_error("Ulysses fused: k/v head counts must match");
    if (nq % P != 0)
      throw std::runtime_error("Ulysses fused: nq must be divisible by world_size");
    if (nkv == 0 || nq % nkv != 0)
      throw std::runtime_error("Ulysses fused: nq must be divisible by nkv");
    if (!(nkv % P == 0 || P % nkv == 0))
      throw std::runtime_error(
          "Ulysses fused: need (nkv % P == 0) or (P % nkv == 0) (DeepSpeed rule)");

    const int rep = (nkv < P) ? static_cast<int>(P / nkv) : 1;
    const bool has_gamma = q_gamma_f_.is_valid() && k_gamma_f_.is_valid();

    // 1. local contiguous shards.
    Tensor ql, kl, vl;
    if (pre_sharded) {
      ql = autograd::contiguous(q);
      kl = autograd::contiguous(k);
      vl = autograd::contiguous(v);
    } else {
      const int64_t T = q.shape().dims[2];
      if (T % P != 0)
        throw std::runtime_error("Ulysses fused: T must be divisible by world_size");
      const int64_t Tl = T / P;
      Tensor qc = autograd::contiguous(q);
      Tensor kc = autograd::contiguous(k);
      Tensor vc = autograd::contiguous(v);
      ql = qc.narrow(2, static_cast<int64_t>(rank_) * Tl, Tl).contiguous();
      kl = kc.narrow(2, static_cast<int64_t>(rank_) * Tl, Tl).contiguous();
      vl = vc.narrow(2, static_cast<int64_t>(rank_) * Tl, Tl).contiguous();
    }

    // 2. optional partial KV replication (nkv < P) before the all-to-all.
    Tensor kl_r = cp::head_repeat_interleave(kl, rep);
    Tensor vl_r = cp::head_repeat_interleave(vl, rep);

    // 3. combine: Q by nq, KV by eff_kv. No local broadcast (kernel is GQA-native).
    Tensor qg = cp::ulysses_combine(pg_, ql, P);   // [B, nq_local,  T, D]
    Tensor kg = cp::ulysses_combine(pg_, kl_r, P); // [B, kv_local,  T, D]
    Tensor vg = cp::ulysses_combine(pg_, vl_r, P);
    const int64_t nq_local = qg.shape().dims[1];
    const int64_t kv_local = kg.shape().dims[1];
    const int64_t B = qg.shape().dims[0];
    const int64_t T = qg.shape().dims[2];

    // 4. pack + bf16 (mirror sdpa_gqa_fused), run the fused kernel.
    Tensor packed = Tensor::cat({qg.contiguous().as_type(Dtype::Bfloat16).flatten(),
                                 kg.contiguous().as_type(Dtype::Bfloat16).flatten(),
                                 vg.contiguous().as_type(Dtype::Bfloat16).flatten()},
                                0);
    TensorOptions bf = qg.opts().with_dtype(Dtype::Bfloat16).with_req_grad(false);
    TensorOptions fp = qg.opts().with_dtype(Dtype::Float32).with_req_grad(false);
    Tensor out_bf = Tensor::empty(Shape({{B, nq_local, T, hd}}), bf);
    Tensor lse    = Tensor::empty(Shape({{B, nq_local, T}}), fp);
    Tensor q_rstd = Tensor::empty(Shape({{B, nq_local, T}}), fp);
    Tensor k_rstd = Tensor::empty(Shape({{B, kv_local, T}}), fp);
    const int cache_seq_len = static_cast<int>(cos_sin_cache_f_.shape().dims[0]);

    OwnTensor::cuda::gqa_fused_flash_attn_forward(
        packed.data<bfloat16_t>(), cos_sin_cache_f_.data<float>(),
        has_gamma ? q_gamma_f_.data<float>() : nullptr,
        has_gamma ? k_gamma_f_.data<float>() : nullptr,
        out_bf.data<bfloat16_t>(), lse.data<float>(), q_rstd.data<float>(),
        k_rstd.data<float>(), static_cast<int>(B), static_cast<int>(nq_local),
        static_cast<int>(kv_local), static_cast<int>(T), static_cast<int>(hd),
        cache_seq_len, /*pos_offset=*/0, eps_f_, interleaved_f_, /*is_causal=*/true);

    Tensor out_g = out_bf.as_type(Dtype::Float32);       // [B, nq_local, T, hd]

    { // DTYPE_PROBE: the bf16 attention leg (inside the fused kernel), once, rank 0.
      // The C++-level probes straddle forward_cp and only see fp32 in/out; this
      // one witnesses the actual bf16 pack/kernel-output/fp32-recast here.
      static bool once = false;
      if (!once && rank_ == 0) {
        std::cout << "[DTYPE_PROBE] ulysses_fused attn -> qkv_in="
                  << get_dtype_name(qg.dtype())
                  << "  packed(kernel_in)=" << get_dtype_name(packed.dtype())
                  << "  out_bf(kernel_out)=" << get_dtype_name(out_bf.dtype())
                  << "  out_g(recast)=" << get_dtype_name(out_g.dtype())
                  << std::endl;
        once = true;
      }
    }

    // 5. partition Q-shaped output back; 6. optional plain unshard.
    Tensor out_l = cp::ulysses_partition(pg_, out_g, P); // [B, nq, Tl, D]
    Tensor output_tensor =
        unshard ? cp::ulysses_gather_seq(pg_, out_l, P) : out_l;

    // 7. autograd wiring.
    if (q.requires_grad() || k.requires_grad() || v.requires_grad() ||
        (has_gamma && (q_gamma_f_.requires_grad() || k_gamma_f_.requires_grad()))) {
      auto grad_fn = std::make_shared<UlyssesFusedGQAAttentionBackward>(
          packed, out_bf, lse, q_rstd, k_rstd, cos_sin_cache_f_, q_gamma_f_,
          k_gamma_f_, pg_, /*is_causal=*/true, world_size_, rank_, unshard,
          pre_sharded, B, nq_local, kv_local, hd, cache_seq_len, eps_f_,
          interleaved_f_, nkv, rep, nq);
      if (q.requires_grad()) {
        Tensor &q_mut = const_cast<Tensor &>(q);
        grad_fn->set_next_edge(0, autograd::get_grad_edge(q_mut));
      }
      if (k.requires_grad()) {
        Tensor &k_mut = const_cast<Tensor &>(k);
        grad_fn->set_next_edge(1, autograd::get_grad_edge(k_mut));
      }
      if (v.requires_grad()) {
        Tensor &v_mut = const_cast<Tensor &>(v);
        grad_fn->set_next_edge(2, autograd::get_grad_edge(v_mut));
      }
      if (has_gamma) {
        if (q_gamma_f_.requires_grad())
          grad_fn->set_next_edge(3, autograd::get_grad_edge(q_gamma_f_));
        if (k_gamma_f_.requires_grad())
          grad_fn->set_next_edge(4, autograd::get_grad_edge(k_gamma_f_));
      }
      output_tensor.set_grad_fn(grad_fn);
      output_tensor.set_requires_grad(true);
    }

    return output_tensor;
  }

  const DeviceMesh *mesh_;
  std::shared_ptr<ProcessGroupNCCL> pg_;
  float attn_scale_;
  bool is_causal_;
  RotatorType rotator_type_;
  bool load_balance_;
  bool recompute_k_;
  int world_size_;
  int rank_;
  HeadTail load_balancer_;

  // Fused-RoPE opt-in (additive; default off => GPT-2/wpe path unchanged).
  // Set via enable_rope(). When false, none of these are read.
  bool use_rope_ = false;
  Tensor cos_sin_cache_;   // [cache_len, hd] cos/sin cache (resident, no grad)
  Tensor q_gamma_;         // [hd] QK-norm scale (or invalid => skip)
  Tensor k_gamma_;         // [hd] QK-norm scale (or invalid => skip)
  float rope_eps_ = 1e-6f;

  // Ulysses opt-in (additive; default off => ring path unchanged). Set via
  // enable_ulysses(); when true, forward_cp dispatches to forward_ulysses.
  bool use_ulysses_ = false;

  // Fused RoPE+QK-norm+GQA Ulysses opt-in (additive; default off). Set via
  // enable_ulysses_fused(); when true forward_ulysses dispatches to the fused
  // path. Gammas are all-or-nothing (both valid or both invalid).
  bool use_ulysses_fused_ = false;
  Tensor cos_sin_cache_f_; // [cache_len, hd] fp32 cos/sin (resident, no grad)
  Tensor q_gamma_f_;       // [hd] QK-norm scale (both valid or both invalid)
  Tensor k_gamma_f_;
  float eps_f_ = 1e-6f;
  bool interleaved_f_ = false;

  // Hybrid (Ulysses-inner + Ring-outer) opt-in (additive; default off). Set via
  // enable_hybrid(). pg_ is the RING sub-group; ulysses_pg_ carries the inner
  // all-to-all. in_hybrid_ring_ is a re-entrancy guard so the recursive ring
  // forward_cp call bypasses the hybrid dispatch and runs the ring body.
  bool use_hybrid_ = false;
  std::shared_ptr<ProcessGroupNCCL> ulysses_pg_;
  int ulysses_size_ = 1;
  bool in_hybrid_ring_ = false;

  // Persistent forward ring buffers (see Phase 2). Reused across forward_cp
  // calls and reallocated only when the per-rank KV size changes, so the ring
  // staging never churns the caching allocator and never shares a block with a
  // per-step SDPA output allocation (the forward-overlap corruption).
  Tensor send_buf_[2];
  std::unique_ptr<RingRotatorBase> kv_rotator_persistent_;
  int64_t persistent_kv_numel_ = -1;
  // When set, forward_cp uses this shared holder's rotator+send_buf instead of
  // the per-instance members above (one set shared across all layers).
  std::shared_ptr<SharedFwdRing> shared_fwd_ring_;

  std::unique_ptr<RingRotatorBase> create_rotator() const {
    switch (rotator_type_) {
    case RotatorType::P2P:
      return std::make_unique<P2PRingRotator>(pg_);
    case RotatorType::AlltoAll:
      return std::make_unique<AlltoAllRingRotator>(pg_);
    case RotatorType::AllGather:
      return std::make_unique<AllGatherRingRotator>(pg_);
    default:
      throw std::runtime_error("Unknown rotator type");
    }
  }
};

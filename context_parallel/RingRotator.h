#pragma once

#include "ProcessGroupNCCL.h"   // canonical BluTrain PG (via -IBluTrain/dist/communication/include)
#include "core/Tensor.h"
#include "device/CachingCudaAllocator.h"
#include <vector>
#include <memory>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

using namespace OwnTensor;

// ---------------------------------------------------------------------------
// Base class for all ring rotators.
// A rotator shifts KV (or grad) buffers around a ring of ranks.
//
// Usage pattern (per ring iteration):
//   1. exchange_buffers(curr_buffer)   -- initiates async send of curr_buffer
//   2. next_buffer()                   -- blocks until the received buffer is ready
// ---------------------------------------------------------------------------
class RingRotatorBase {
public:
    RingRotatorBase(std::shared_ptr<ProcessGroupNCCL> pg)
        : pg_(pg),
          rank_(pg->get_rank()),
          world_size_(pg->get_worldsize()) {}

    virtual ~RingRotatorBase() = default;

    // Posts the async send of curr_buffer and receives into the rotator's
    // (double-buffered) recv slot. Returns the Work handle so the CALLER can
    // guard reuse of its own send-staging buffer (must outlive the transfer).
    //
    // pack_stream: if non-null, OVERLAP is requested. The rotator records its OWN
    // persistent per-slot event on pack_stream (right after the caller's pack
    // memcpy, which ran on that same stream) and makes the dedicated ring stream
    // wait on it before the NCCL send — so the send never reads a half-packed
    // buffer. The event is owned by the rotator and NEVER destroyed per-call:
    // destroying an event while a pending cudaStreamWaitEvent references it is
    // UNCONDITIONAL UB — the old per-call create/destroy was always a race; it
    // merely appeared safe while the GPU happened to evaluate the wait before
    // the CPU reached destroy (shallow pipeline), and manifested once the ring
    // stream lagged at steady state. Persistent events are the only correct fix
    // (a sync-before-destroy would just narrow the window). nullptr => the
    // legacy blocking path (NCCL on the shared blocking comm stream).
    // NOTE: `overlap` must be an explicit flag — the pack stream CANNOT serve as
    // the sentinel because legitimate compute runs on the legacy NULL stream
    // (cudaStream_t)0, which is indistinguishable from "not provided".
    virtual std::shared_ptr<Work> exchange_buffers(Tensor& curr_buffer,
                                                   bool overlap = false,
                                                   cudaStream_t pack_stream = nullptr) = 0;

    // CPU-blocking consume (legacy path; used when CP_NO_OVERLAP is set).
    virtual Tensor next_buffer() = 0;

    // GPU-side consume: make `compute_stream` wait on the pending recv event
    // (no CPU stall) and return the received buffer. Default falls back to the
    // CPU-blocking next_buffer() (correct, just not overlapped) so rotators that
    // cannot pipeline (e.g. AllGather) remain valid.
    virtual Tensor next_buffer_streamordered(cudaStream_t /*compute_stream*/) {
        return next_buffer();
    }

    // Re-arm the rotator for a NEW ring (called once per forward_cp before the
    // ring loop). P2P/AlltoAll re-communicate fresh data on every exchange, so
    // they need no re-arming -> default no-op. AllGather caches a one-shot gather
    // and must re-gather the current step's KV each forward, so it overrides this
    // to invalidate its cached buffer (without freeing it -- the storage and the
    // persistent pack-event are reused, avoiding allocator churn and the
    // event-destroy UB that recreating the rotator each call would cause).
    virtual void reset() {}

protected:
    std::shared_ptr<ProcessGroupNCCL> pg_;
    int rank_;
    int world_size_;
};


// ---------------------------------------------------------------------------
// P2P Ring Rotator (optimized)
//
// Pre-allocates the receive buffer on first use to avoid per-step allocation.
// Uses ncclSend / ncclRecv point-to-point operations.
// Even ranks send first then receive; odd ranks receive first then send.
// ---------------------------------------------------------------------------
class P2PRingRotator : public RingRotatorBase {
public:
    P2PRingRotator(std::shared_ptr<ProcessGroupNCCL> pg)
        : RingRotatorBase(pg), buffer_allocated_(false), slot_(0) {}

    // Double-buffered ring shift: receive into the slot NOT being read by the
    // current step's compute, so recv(i+1) can be in flight while compute(i)
    // reads recv(i). Returns the Work covering both this step's send and recv.
    ~P2PRingRotator() override {
        for (int s = 0; s < 2; ++s) if (pack_ev_[s]) cudaEventDestroy(pack_ev_[s]);
    }

    std::shared_ptr<Work> exchange_buffers(Tensor& curr_buffer,
                                           bool overlap = false,
                                           cudaStream_t pack_stream = nullptr) override {
        int next_rank = (rank_ + 1) % world_size_;
        int prev_rank = (rank_ - 1 + world_size_) % world_size_;

        size_t count = static_cast<size_t>(curr_buffer.numel());
        Dtype dtype = curr_buffer.dtype();

        // Pre-allocate BOTH receive slots once (ping-pong), reuse across steps.
        if (!buffer_allocated_) {
            for (int s = 0; s < 2; ++s)
                recv_[s] = Tensor::empty(curr_buffer.shape(), curr_buffer.opts());
            buffer_allocated_ = true;
        }

        slot_ ^= 1;  // receive into the free slot
        {
            static bool _once = false;
            if (!_once) { _once = true;
                fprintf(stderr, "[CP ring P2P] %s\n", overlap
                    ? "OVERLAP: dedicated non-blocking stream"
                    : "NO-OVERLAP: shared blocking stream"); }
        }
        if (overlap) {
            // OVERLAP: run the ring on the dedicated NON-BLOCKING stream, ordered
            // after the caller's pack memcpy via a PERSISTENT per-slot event.
            //
            // INVARIANT (why TWO events suffice): events are per recv-slot and a
            // slot is re-recorded only every OTHER exchange (slot_ ^= 1), i.e.
            // in-flight depth is exactly the 2-slot ping-pong / 1-step lookahead.
            // To make this safe INDEPENDENT of caller discipline, we self-guard:
            // before re-recording pack_ev_[slot_], order pack_stream after this
            // slot's prior send (GPU-side). Once that send completed, its
            // cudaStreamWaitEvent on the ring stream has necessarily executed,
            // so re-recording the event cannot alias a pending wait. If the
            // pipeline is ever deepened beyond 1-step lookahead, the slot count
            // (recv_/work_/pack_ev_) must grow with it.
            if (work_[slot_]) work_[slot_]->streamWait(pack_stream);
            if (pack_ev_[slot_] == nullptr)
                cudaEventCreateWithFlags(&pack_ev_[slot_], cudaEventDisableTiming);
            cudaEventRecord(pack_ev_[slot_], pack_stream);
            cudaStream_t rs = pg_->cpRingStream();
            cudaStreamWaitEvent(rs, pack_ev_[slot_], 0);
            // See AlltoAllRingRotator: register the cross-stream use so the
            // caching allocator defers reuse until the ring stream completes.
            OwnTensor::CachingCUDAAllocator::instance().recordStream(curr_buffer.data(), rs);
            OwnTensor::CachingCUDAAllocator::instance().recordStream(recv_[slot_].data(), rs);
            work_[slot_] = pg_->sendrecv_async_stream(
                curr_buffer.data<float>(), recv_[slot_].data<float>(),
                next_rank, prev_rank, count, dtype, rs);
        } else {
            // NO-OVERLAP: original path on the (blocking) shared comm stream.
            work_[slot_] = pg_->sendrecv_async(
                curr_buffer.data<float>(), recv_[slot_].data<float>(),
                next_rank, prev_rank, count, dtype);
        }
        return work_[slot_];
    }

    Tensor next_buffer() override {
        if (work_[slot_]) {
            work_[slot_]->wait();      // CPU-blocking (legacy / CP_NO_OVERLAP path)
            work_[slot_] = nullptr;
        }
        if (!recv_[slot_].is_valid()) {
            throw std::runtime_error("P2PRingRotator::next_buffer: no buffer available");
        }
        return recv_[slot_];
    }

    Tensor next_buffer_streamordered(cudaStream_t compute_stream) override {
        if (work_[slot_]) {
            work_[slot_]->streamWait(compute_stream);  // GPU-side, no CPU stall
        }
        if (!recv_[slot_].is_valid()) {
            throw std::runtime_error(
                "P2PRingRotator::next_buffer_streamordered: no buffer available");
        }
        return recv_[slot_];
    }

private:
    Tensor recv_[2];                       // ping-pong recv slots
    bool buffer_allocated_;
    int  slot_;                            // slot most recently received into
    std::shared_ptr<Work> work_[2];        // pending op per recv slot
    cudaEvent_t pack_ev_[2] = {nullptr, nullptr};  // persistent per-slot pack events
};


// ---------------------------------------------------------------------------
// AlltoAll Ring Rotator
//
// Implements ring rotation via ncclAlltoAll using the shifted permutation
// pattern dsts = [1, 2, ..., n-1, 0].
//
// Each rank builds a send buffer with world_size chunks. Only the chunk
// destined for next_rank contains the actual KV data; all other chunks
// are zeroed. After the collective, we extract the data from the
// prev_rank slot of the receive buffer.
//
// This is deadlock-free and topology-aware (NCCL optimizes the routing).
// ---------------------------------------------------------------------------
class AlltoAllRingRotator : public RingRotatorBase {
public:
    AlltoAllRingRotator(std::shared_ptr<ProcessGroupNCCL> pg)
        : RingRotatorBase(pg), buffer_allocated_(false) {}

    // Sparse alltoall ring shift: send curr_buffer to rank (i+1)%N,
    // receive from rank (i-1)%N. Matches PyTorch's permute_tensor
    // with dsts=[1,2,...,n-1,0] using alltoallv with sparse split sizes.
    // Buffer: 1x per_rank_count (not world_size * per_rank_count).
    // One sparse alltoallv per step (send->next_rank, recv<-prev_rank): exactly
    // ONE in-flight op writing ONE recv destination per step — structurally
    // identical to P2P, so the ping-pong (recv_[2]/slot_) applies. The sparse
    // split arrays are slot-independent (they encode ranks, computed once).
    ~AlltoAllRingRotator() override {
        for (int s = 0; s < 2; ++s) if (pack_ev_[s]) cudaEventDestroy(pack_ev_[s]);
    }

    std::shared_ptr<Work> exchange_buffers(Tensor& curr_buffer,
                                           bool overlap = false,
                                           cudaStream_t pack_stream = nullptr) override {
        int next_rank = (rank_ + 1) % world_size_;
        int prev_rank = (rank_ - 1 + world_size_) % world_size_;

        size_t numel = static_cast<size_t>(curr_buffer.numel());
        Dtype dtype = curr_buffer.dtype();

        if (!buffer_allocated_) {
            for (int s = 0; s < 2; ++s)
                recv_[s] = Tensor::empty(curr_buffer.shape(), curr_buffer.opts());
            sendcounts_.assign(world_size_, 0);
            recvcounts_.assign(world_size_, 0);
            senddispls_.assign(world_size_, 0);
            recvdispls_.assign(world_size_, 0);
            sendcounts_[next_rank] = numel;
            recvcounts_[prev_rank] = numel;
            buffer_allocated_ = true;
        }

        slot_ ^= 1;
        {
            static bool _once = false;
            if (!_once) { _once = true;
                fprintf(stderr, "[CP ring A2A] %s\n", overlap
                    ? "OVERLAP: dedicated non-blocking stream"
                    : "NO-OVERLAP: shared blocking stream"); }
        }
        if (overlap) {
            // Self-guard + 2-event invariant: see P2PRingRotator::exchange_buffers.
            if (work_[slot_]) work_[slot_]->streamWait(pack_stream);
            if (pack_ev_[slot_] == nullptr)
                cudaEventCreateWithFlags(&pack_ev_[slot_], cudaEventDisableTiming);
            cudaEventRecord(pack_ev_[slot_], pack_stream);
            cudaStream_t rs = pg_->cpRingStream();
            cudaStreamWaitEvent(rs, pack_ev_[slot_], 0);
            // CRITICAL: send_buf (curr_buffer) and recv_[slot_] are used on the
            // non-blocking ring stream, but were allocated on (and are owned by)
            // the compute stream. Without recordStream the caching allocator
            // defers their reuse based ONLY on the compute stream, so it can
            // recycle these blocks while the ring stream's collective is still
            // reading/writing them -- corrupting an unrelated concurrent
            // allocation (e.g. the step-0 SDPA workspace). This is the forward
            // overlap race; backward escapes it only because its transfer rides
            // the blocking comm stream (implicitly ordered with stream 0).
            OwnTensor::CachingCUDAAllocator::instance().recordStream(curr_buffer.data(), rs);
            OwnTensor::CachingCUDAAllocator::instance().recordStream(recv_[slot_].data(), rs);
            work_[slot_] = pg_->alltoallv_async_stream(
                curr_buffer.data(), sendcounts_.data(), senddispls_.data(),
                recv_[slot_].data(), recvcounts_.data(), recvdispls_.data(),
                dtype, rs);
        } else {
            work_[slot_] = pg_->alltoallv_async(
                curr_buffer.data(), sendcounts_.data(), senddispls_.data(),
                recv_[slot_].data(), recvcounts_.data(), recvdispls_.data(),
                dtype);
        }
        return work_[slot_];
    }

    Tensor next_buffer() override {
        if (work_[slot_]) { work_[slot_]->wait(); work_[slot_] = nullptr; }
        return recv_[slot_];
    }

    Tensor next_buffer_streamordered(cudaStream_t compute_stream) override {
        if (work_[slot_]) work_[slot_]->streamWait(compute_stream);  // GPU-side, no CPU stall
        return recv_[slot_];
    }

private:
    Tensor recv_[2];
    bool buffer_allocated_;
    int  slot_ = 0;
    std::vector<size_t> sendcounts_;
    std::vector<size_t> recvcounts_;
    std::vector<size_t> senddispls_;
    std::vector<size_t> recvdispls_;
    std::shared_ptr<Work> work_[2];
    cudaEvent_t pack_ev_[2] = {nullptr, nullptr};  // persistent per-slot pack events
};


// ---------------------------------------------------------------------------
// AllGather Ring Rotator
//
// Gathers all ranks' KV in a single all_gather on the first exchange of a ring,
// then indexes into the gathered buffer for each subsequent ring step.
//
// OVERLAP: the gather is issued ASYNC on the dedicated CP ring stream (cp_comm_)
// and the consuming compute stream waits on it GPU-side (streamWait), exactly
// like P2P/AlltoAll. So the gather overlaps step 0's LOCAL attention compute
// instead of CPU-blocking the whole step on the shared comm (the source of the
// erratic-throughput rendezvous jitter). Note: AllGather issues ONE collective,
// so it can only overlap the gather with the first (local) SDPA -- P2P/AlltoAll
// hide a hop behind every step and so overlap more, but this removes the
// CPU-side blocking barrier that made AllGather's step time oscillate.
//
// The instance is PERSISTENT (owned by ContextParallel); reset() re-arms it for
// each forward (re-gather the current step's KV) while reusing the aggregated
// buffer storage and the persistent pack-event -> no per-call allocation churn
// and no event-destroy UB.
// ---------------------------------------------------------------------------
class AllGatherRingRotator : public RingRotatorBase {
public:
    AllGatherRingRotator(std::shared_ptr<ProcessGroupNCCL> pg)
        : RingRotatorBase(pg) {}

    ~AllGatherRingRotator() override {
        if (pack_ev_) cudaEventDestroy(pack_ev_);
    }

    // Re-arm for a new ring: invalidate the cached gather (so the next exchange
    // re-gathers the CURRENT step's KV) and reset the shard index. Keep the
    // aggregated buffer storage and the pack-event for reuse.
    void reset() override {
        gathered_ = false;
        idx_ = 0;
        work_ = nullptr;  // waits on its event were already issued last ring
    }

    std::shared_ptr<Work> exchange_buffers(Tensor& curr_buffer,
                                           bool overlap = false,
                                           cudaStream_t pack_stream = nullptr) override {
        idx_ += 1;                       // tracks ring step for next_buffer indexing
        {
            // One-time confirmation of which forward ring path is active.
            static bool _once = false;
            if (!_once) { _once = true;
                fprintf(stderr, "[CP ring AllGather] %s\n", overlap
                    ? "OVERLAP: async all_gather on dedicated ring stream"
                    : "NO-OVERLAP: blocking all_gather on shared stream"); }
        }
        if (gathered_) return nullptr;   // gather for this ring already issued

        size_t per_rank_count = static_cast<size_t>(curr_buffer.numel());
        size_t total_count = per_rank_count * static_cast<size_t>(world_size_);
        Dtype dtype = curr_buffer.dtype();
        per_rank_numel_ = per_rank_count;

        // Reuse the aggregated buffer across forwards; reallocate only if the
        // per-rank size changed (training T vs generation T).
        if (!aggregated_buffer_.is_valid() ||
            static_cast<size_t>(aggregated_buffer_.numel()) != total_count) {
            Shape agg_shape({{static_cast<int64_t>(total_count)}});
            aggregated_buffer_ = Tensor::empty(agg_shape, curr_buffer.opts());
        }

        Tensor flat_input = curr_buffer.flatten();

        if (overlap) {
            cudaStream_t rs = pg_->cpRingStream();
            // Order the ring stream after the caller's pack memcpy (which ran on
            // pack_stream) so the gather never reads a half-packed send buffer.
            // The same event also orders the gather after the PREVIOUS forward's
            // reads of aggregated_buffer_ (pack_stream is the compute stream,
            // which is FIFO), making the buffer reuse safe. Persistent event:
            // never destroyed per-call (UB to destroy while a wait is pending).
            if (pack_ev_ == nullptr)
                cudaEventCreateWithFlags(&pack_ev_, cudaEventDisableTiming);
            cudaEventRecord(pack_ev_, pack_stream);
            cudaStreamWaitEvent(rs, pack_ev_, 0);
            // send + recv buffers are read/written on the ring stream but owned by
            // the compute stream; recordStream stops the caching allocator from
            // recycling them while the gather is in flight.
            OwnTensor::CachingCUDAAllocator::instance().recordStream(flat_input.data(), rs);
            OwnTensor::CachingCUDAAllocator::instance().recordStream(aggregated_buffer_.data(), rs);
            work_ = pg_->all_gather_async_stream(
                flat_input.data<float>(), aggregated_buffer_.data<float>(),
                per_rank_count, dtype, rs);
            gathered_ = true;
            return work_;
        }

        // Legacy blocking path (CP_NO_OVERLAP): synchronous gather on the shared
        // comm; no pending Work.
        pg_->all_gather(flat_input.data<float>(), aggregated_buffer_.data<float>(),
                        per_rank_count, dtype, /*sync=*/true);
        gathered_ = true;
        return nullptr;
    }

    Tensor next_buffer() override {
        if (!gathered_)
            throw std::runtime_error("AllGatherRingRotator::next_buffer: exchange_buffers not called");
        if (work_) work_->wait();  // CPU-block until the async gather lands
        return index_chunk(/*stream=*/0, /*blocking=*/true);
    }

    Tensor next_buffer_streamordered(cudaStream_t compute_stream) override {
        if (!gathered_)
            throw std::runtime_error("AllGatherRingRotator::next_buffer_streamordered: exchange_buffers not called");
        if (work_) work_->streamWait(compute_stream);  // GPU-side: no CPU stall
        return index_chunk(compute_stream, /*blocking=*/false);
    }

private:
    // Copy this ring step's shard (rank - idx_) out of the gathered buffer.
    Tensor index_chunk(cudaStream_t stream, bool blocking) {
        int source_rank = ((rank_ - idx_) % world_size_ + world_size_) % world_size_;
        int64_t offset = static_cast<int64_t>(source_rank) * static_cast<int64_t>(per_rank_numel_);
        float* base_ptr = aggregated_buffer_.data<float>() + offset;
        Shape chunk_shape({{static_cast<int64_t>(per_rank_numel_)}});
        Tensor chunk = Tensor::empty(chunk_shape, aggregated_buffer_.opts());
        if (blocking) {
            cudaMemcpy(chunk.data<float>(), base_ptr,
                       per_rank_numel_ * sizeof(float), cudaMemcpyDeviceToDevice);
        } else {
            // Ordered after the streamWait above (gather has landed) and before
            // the caller's SDPA, since both run on compute_stream.
            cudaMemcpyAsync(chunk.data<float>(), base_ptr,
                            per_rank_numel_ * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
        }
        return chunk;
    }

    int idx_ = 0;
    bool gathered_ = false;
    Tensor aggregated_buffer_;
    size_t per_rank_numel_ = 0;
    std::shared_ptr<Work> work_;
    cudaEvent_t pack_ev_ = nullptr;  // persistent pack-ordering event
};
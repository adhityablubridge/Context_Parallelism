#pragma once
// ---------------------------------------------------------------------------
// LoadBalancer.h
// -----------------------------------------------------------------------------
// Context-Parallel sequence load-balancing classes, extracted from
// tensor/dtensor.h so all CP-specific code lives under
// context_parallel/. dtensor.h keeps only a forward declaration
// of LoadBalancer (for the DTensor::context_parallel_* method signatures); the
// full class definitions live here, and the non-inline method bodies
// (HeadTail::loadbalance / unloadbalance) live in ContextParallel.cpp.
//
// Types are spelled OwnTensor::Tensor explicitly so this header does not rely
// on a `using namespace OwnTensor` being in scope at the include site.
// ---------------------------------------------------------------------------

#include <cstdint>
#include <cuda_runtime.h>

#include "core/Tensor.h"

// Abstract sequence load-balancer interface used by ContextParallel sharding.
class LoadBalancer {
public:
  bool loadbalancing_enabled;
  int64_t world_size;
  int64_t chunkdim;
  LoadBalancer() = default;
  LoadBalancer(int64_t chunkdim_) { chunkdim = chunkdim_; }
  virtual void loadbalance(OwnTensor::Tensor &tensor) = 0;
  virtual void unloadbalance(OwnTensor::Tensor &tensor) = 0;
  void set_world_size(int64_t world_size_) { world_size = world_size_; }
  void set_load_balancing(bool enable) { loadbalancing_enabled = enable; }
  void set_chunk_dim(int64_t dim) { chunkdim = dim; }
  bool is_load_balancing_enabled() { return loadbalancing_enabled; }
};

// HeadTail permutation (chunk-level, PyTorch round-robin parity). Splits the
// sequence into 2*world_size chunks; rank r owns chunks (r, 2N-1-r). Method
// bodies are in ContextParallel.cpp (they call the headtail CUDA kernels).
class HeadTail : public LoadBalancer {
private:
  cudaStream_t stream_ = nullptr;

public:
  HeadTail() = default;
  HeadTail(int64_t world_size_) { world_size = world_size_; }
  void set_stream(cudaStream_t stream) { stream_ = stream; }
  void loadbalance(OwnTensor::Tensor &tensor) override;
  void unloadbalance(OwnTensor::Tensor &tensor) override;
};

// Placeholder merger (LoadBalancer subclass); retained from the original
// dtensor.cpp. Fully inline.
class SDPA_Merger : public LoadBalancer {
private:
  cudaStream_t stream_ = nullptr;

public:
  SDPA_Merger() = default;
  SDPA_Merger(int64_t world_size_) { world_size = world_size_; }
  void set_stream(cudaStream_t stream) { stream_ = stream; }
  void loadbalance(OwnTensor::Tensor & /*tensor*/) override {}
  void unloadbalance(OwnTensor::Tensor & /*tensor*/) override {}
  void merge(OwnTensor::Tensor & /*q*/, OwnTensor::Tensor & /*k*/,
             OwnTensor::Tensor & /*v*/, OwnTensor::Tensor & /*output*/) {}
};

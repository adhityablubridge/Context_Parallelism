// ---------------------------------------------------------------------------
// ContextParallel.cpp
// -----------------------------------------------------------------------------
// Context-Parallel sequence sharding + HeadTail load-balancing implementations,
// extracted from tensor/dtensor.cpp so all CP-specific code lives under
// gpt2_cp_test/context_parallel/.
//
// Contains:
//   - DTensor::context_parallel_shard / context_parallel_unshard
//   - HeadTail::loadbalance / unloadbalance (call the headtail CUDA kernels)
//
// Declarations: DTensor::context_parallel_* stay in tensor/dtensor.h (member
// functions); LoadBalancer / HeadTail / SDPA_Merger are declared in
// gpt2_cp_test/context_parallel/LoadBalancer.h.
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "TensorLib.h"
// #include "tensor/dtensor.h"
#include "headtail_kernel.cuh"
#include "context_parallel/LoadBalancer.h"

using namespace OwnTensor;

// ----- DTensor context-parallel sharding ----------------------------------
// DEAD / LEGACY: DTensor::context_parallel_shard / unshard have ZERO call sites
// (the live CP path shards via shard_sequence_pre_embed + HeadTail::loadbalance
// + Tensor::make_shards_inplace_axis directly on raw Tensors, never through
// DTensor). Kept commented for reference from the old DTensor-centric design.
// Declarations in tensor/dtensor.h are commented out to match.
/*
void DTensor::context_parallel_shard(std::vector<Tensor> &chunks,
                                    LoadBalancer &load_balancer) {
     load_balancer.set_world_size(world_size_);
    if (tensor_.shape().dims.size() == 4)
      load_balancer.set_chunk_dim(2);
    else if (tensor_.shape().dims.size() == 3)
      load_balancer.set_chunk_dim(1);
    else
      throw std::runtime_error(
          "context_parallel_shard only supports 3D and 4D tensors");

   load_balancer.loadbalance(tensor_);

  if (tensor_.shape().dims.size() == 4)
    chunks = tensor_.make_shards_inplace_axis(world_size_, 2);
  else if (tensor_.shape().dims.size() == 3)
    chunks = tensor_.make_shards_inplace_axis(world_size_, 1);
  else
    throw std::runtime_error(
        "context_parallel_shard only supports 3D and 4D tensors");
}

void DTensor::context_parallel_unshard(std::vector<Tensor> &chunks,
                                     LoadBalancer &load_balancer) {
    if (tensor_.shape().dims.size() == 4)
      chunks = tensor_.make_shards_inplace_axis(world_size_, 2);
    else if (tensor_.shape().dims.size() == 3)
      chunks = tensor_.make_shards_inplace_axis(world_size_, 1);
    else
      throw std::runtime_error(
          "context_parallel_shard only supports 3D and 4D tensors");

    load_balancer.set_world_size(world_size_);
    if (tensor_.shape().dims.size() == 4)
      load_balancer.set_chunk_dim(2);
    else if (tensor_.shape().dims.size() == 3)
      load_balancer.set_chunk_dim(1);
    else
      throw std::runtime_error(
          "context_parallel_shard only supports 3D and 4D tensors");

   load_balancer.unloadbalance(tensor_);
}
*/

// ----- HeadTail load-balancing (chunk-level, PyTorch round-robin parity) ---

void HeadTail::loadbalance(Tensor &tensor) {

  if (chunkdim < 0 || chunkdim >= (int)tensor.shape().dims.size()) {
    throw std::runtime_error("HeadTail::loadbalance: Invalid dimension");
  }

  int64_t seq_len = tensor.shape().dims[chunkdim];

  if (world_size <= 0) {
    throw std::runtime_error(
        "HeadTail::loadbalance: world_size must be > 0 (call set_world_size first)");
  }
  if (seq_len % (2 * world_size) != 0) {
    std::ostringstream oss;
    oss << "HeadTail::loadbalance: Sequence length (" << seq_len
        << ") must be divisible by 2*world_size (2*" << world_size << ")";
    throw std::runtime_error(oss.str());
  }

  // Compute outer_size = product of dims before chunkdim
  // Compute inner_size = product of dims after chunkdim
  auto dims = tensor.shape().dims;
  int64_t outer_size = 1;
  for (int i = 0; i < chunkdim; i++) outer_size *= dims[i];
  int64_t inner_size = 1;
  for (int i = chunkdim + 1; i < (int)dims.size(); i++) inner_size *= dims[i];

  // Allocate temp buffer for the permuted result
  Tensor result = Tensor::empty(tensor.shape(), tensor.opts());

  // HeadTail permutation via CUDA kernel (chunk-level, PyTorch parity):
  //   Splits T into 2*N chunks of size chunk_sz = T/(2*N). Rank r ends up
  //   owning chunks (r, 2N-1-r) concatenated as [head_chunk, tail_chunk]
  //   after a subsequent split-by-N along the seq dim.
  launch_headtail_loadbalance(
      tensor.data<float>(),
      result.data<float>(),
      outer_size, seq_len, inner_size,
      world_size,
      stream_);

  // Copy permuted data back to the original tensor's GPU buffer
  cudaMemcpyAsync(tensor.data<float>(), result.data<float>(),
                  tensor.numel() * sizeof(float), cudaMemcpyDeviceToDevice,
                  stream_);
  cudaStreamSynchronize(stream_);
}

void HeadTail::unloadbalance(Tensor &tensor) {
  if (chunkdim < 0 || chunkdim >= (int)tensor.shape().dims.size()) {
    throw std::runtime_error("HeadTail::unloadbalance: Invalid dimension");
  }

  int64_t seq_len = tensor.shape().dims[chunkdim];

  if (world_size <= 0) {
    throw std::runtime_error(
        "HeadTail::unloadbalance: world_size must be > 0 (call set_world_size first)");
  }
  if (seq_len % (2 * world_size) != 0) {
    std::ostringstream oss;
    oss << "HeadTail::unloadbalance: Sequence length (" << seq_len
        << ") must be divisible by 2*world_size (2*" << world_size << ")";
    throw std::runtime_error(oss.str());
  }

  auto dims = tensor.shape().dims;
  int64_t outer_size = 1;
  for (int i = 0; i < chunkdim; i++) outer_size *= dims[i];
  int64_t inner_size = 1;
  for (int i = chunkdim + 1; i < (int)dims.size(); i++) inner_size *= dims[i];

  Tensor result = Tensor::empty(tensor.shape(), tensor.opts());

  // Inverse HeadTail permutation via CUDA kernel (chunk-level inverse).
  // Restores original sequence order from a loadbalanced layout.
  launch_headtail_unloadbalance(
      tensor.data<float>(),
      result.data<float>(),
      outer_size, seq_len, inner_size,
      world_size,
      stream_);

  cudaMemcpyAsync(tensor.data<float>(), result.data<float>(),
                  tensor.numel() * sizeof(float), cudaMemcpyDeviceToDevice,
                  stream_);
  cudaStreamSynchronize(stream_);
}

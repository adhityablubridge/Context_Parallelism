#include <chrono>
#include <cstdint>
#include <execution>
#include <future>
#include <iostream>
#include <memory.h>
#include <thread>
#include <vector>
// #include "DataParallel.hpp"
#include "process_group/ProcessGroupNCCL.h"

#define TIME_INTERVAL 10

#define RESULT_CHECK(cmd)                                                      \
  do {                                                                         \
    result_t r = (cmd);                                                        \
    if (r != pgSuccess) {                                                      \
      throw std::runtime_error(std::string("Process Group Error"));            \
    }                                                                          \
  } while (0)

#define PG_CHECK_FALSE(condition, str)                                         \
  do {                                                                         \
    if (!condition) {                                                          \
      throw std::runtime_error(std::string("Condition failed"));               \
    }                                                                          \
  } while (0)

ProcessGroupNCCL::ProcessGroupNCCL(int world_size, int rank, ncclUniqueId &id,
                                   std::shared_ptr<Work> work_obj,
                                   cudaStream_t &stream, int root)
    : world_size_(world_size), id_(id), rank_(rank), work_obj_(work_obj),
      communication_stream_(stream), root_(root) {
  // no_of_gpus per node for initialization.

  const char *env = std::getenv("NO_GPUS_PER_NODE");
  if (env) {
    int parsed = std::atoi(env);
    if (parsed > 0)
      gpus_per_node_ = parsed;
  }

  local_rank_ = rank % gpus_per_node_;
  CUDACHECK(cudaSetDevice(local_rank_));

  NCCLCHECK(ncclCommInitRank(&comm_, world_size_, id_, rank_));

  work_obj_->setCommunicator(comm_);
  if (comm_ == (ncclComm_t)NULL) {
    throw std::runtime_error(std::string("NCCL COMMUNICATION NOT INITIALIZED"));
  }

  // Second, DEDICATED communicator for the CP ring overlap path. Keeps ring
  // sendrecv/alltoallv off the comm_ used by loss/param all-reduces so the two
  // never share one NCCL communicator (PyTorch parity). cp_id_ is broadcast the
  // same way as id_ (MPI_COMM_WORLD, root 0) — correct for a 1D CP mesh where
  // the pg spans all of WORLD; a multi-dim mesh would need the per-dim
  // sub-communicator (pre-existing limitation of id handling, not new here).
  if (rank_ == 0) {
    NCCLCHECK(ncclGetUniqueId(&cp_id_));
  }
  MPI_Bcast(&cp_id_, sizeof(cp_id_), MPI_BYTE, 0, MPI_COMM_WORLD);
  NCCLCHECK(ncclCommInitRank(&cp_comm_, world_size_, cp_id_, rank_));
  const char *_sync_ring = std::getenv("CP_SYNC_RING");
  cp_sync_ring_ = (_sync_ring && _sync_ring[0] == '1');
  if (cp_sync_ring_ && rank_ == 0)
    fprintf(stderr, "[CP ring] CP_SYNC_RING=1: host-sync after each exchange "
                    "(diagnostic: overlap path, no concurrency)\n");
}

ProcessGroupNCCL::~ProcessGroupNCCL() {
  NCCLCHECK(ncclCommDestroy(comm_));
  if (cp_comm_) NCCLCHECK(ncclCommDestroy(cp_comm_));
  if (cp_ring_stream_) CUDACHECK(cudaStreamDestroy(cp_ring_stream_));
  if (owns_stream_) {
    CUDACHECK(cudaStreamDestroy(communication_stream_));
  }
}

void ProcessGroupNCCL::start_time() {
  // Create events fresh for each measurement to avoid resource handle errors.
  CUDACHECK(cudaEventCreate(&start_));
  CUDACHECK(cudaEventCreate(&stop_));
  CUDACHECK(cudaEventRecord(start_, communication_stream_));
}

void ProcessGroupNCCL::end_time(float &ms) {
  CUDACHECK(cudaEventRecord(stop_, communication_stream_));
  CUDACHECK(cudaEventSynchronize(stop_));
  CUDACHECK(cudaEventElapsedTime(&ms, start_, stop_));

  // Destroy events after use.
  CUDACHECK(cudaEventDestroy(start_));
  CUDACHECK(cudaEventDestroy(stop_));
  start_ = nullptr;
  stop_ = nullptr;
}

// init commuication
std::shared_ptr<ProcessGroupNCCL> init_process_group(int world_size, int rank,
                                                     cudaStream_t stream) {
  cudaStream_t communication_stream;
  bool stream_created = false;
  if (stream == 0) {
    cudaSetDevice(rank);
    CUDACHECK(cudaStreamCreateWithFlags(
        &communication_stream,
        cudaStreamNonBlocking)); // better than normal cudaStreamCreate
    stream_created = true;
  } else {
    communication_stream = stream;
  }
  ncclUniqueId id;
  if (rank == 0) {
    NCCLCHECK(ncclGetUniqueId(&id));
    // to communicate the id to all the processes.
  }
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

  std::shared_ptr<Work> work_obj =
      std::make_shared<Work>(communication_stream, nullptr);
  auto pg = std::make_shared<ProcessGroupNCCL>(world_size, rank, id, work_obj,
                                               communication_stream);
  pg->set_owns_stream(stream_created);
  return pg;
}

bool ProcessGroupNCCL::blockStreamEvent() {
  // stream synchronize is for the entire stream. but event synchronize is only
  // upto a point
  work_obj_->wait();
  if (work_obj_->is_success() == false) {
    return false;
  }
  return true;
}

bool ProcessGroupNCCL::blockStream() {
  CUDACHECK(cudaStreamSynchronize(communication_stream_));
  return true;
}

// result_t ToNcclError(bool status, std::shared_ptr<Work> work_obj){
//     if(status == true){
//         return pgSuccess;
//     }else{
//         if(work)
//     }
// }

// collectives

// all reduce
result_t ProcessGroupNCCL::all_reduce(const void *sendbuff, void *recvbuff,
                                      size_t count, OwnTensor::Dtype dtype,
                                      op_t operation, bool sync) {
  auto work_obj =
      all_reduce_async(sendbuff, recvbuff, count, dtype, operation, sync);

  if (work_obj->is_success())
    return pgSuccess;
  else {
    if (work_obj->get_lastCudaError() != cudaSuccess)
      return pgCudaError;
    else if (work_obj->get_ncclStatus() != ncclSuccess)
      return pgNcclError;
    else if (time_limit_exceeded.load())
      return pgTimeout;
  }
  return pgInternalError;
}

//
result_t ProcessGroupNCCL::reduce_scatter(const void *sendbuff, void *recvbuff,
                                          size_t recv_count,
                                          OwnTensor::Dtype dtype,
                                          op_t operation, bool sync) {
  auto work_obj = reduce_scatter_async(sendbuff, recvbuff, recv_count, dtype,
                                       operation, sync);

  if (work_obj->is_success())
    return pgSuccess;
  else {
    if (work_obj->get_lastCudaError() != cudaSuccess)
      return pgCudaError;
    else if (work_obj->get_ncclStatus() != ncclSuccess)
      return pgNcclError;
    else if (time_limit_exceeded.load())
      return pgTimeout;
  }
  return pgInternalError;
}

result_t ProcessGroupNCCL::all_gather(const void *sendbuff, void *recvbuff,
                                      size_t sendcount, OwnTensor::Dtype dtype,
                                      bool sync) {
  auto work_obj = all_gather_async(sendbuff, recvbuff, sendcount, dtype, sync);

  if (work_obj->is_success())
    return pgSuccess;
  else {
    if (work_obj->get_lastCudaError() != cudaSuccess)
      return pgCudaError;
    else if (work_obj->get_ncclStatus() != ncclSuccess)
      return pgNcclError;
    else if (time_limit_exceeded.load())
      return pgTimeout;
  }
  return pgInternalError;
}

result_t ProcessGroupNCCL::gather(const void *sendbuff, void *recvbuff,
                                  size_t count, OwnTensor::Dtype dtype,
                                  int root, bool sync) {
  auto work_obj = gather_async(sendbuff, recvbuff, count, dtype, root, sync);

  if (work_obj->is_success())
    return pgSuccess;
  else {
    if (work_obj->get_lastCudaError() != cudaSuccess)
      return pgCudaError;
    else if (work_obj->get_ncclStatus() != ncclSuccess)
      return pgNcclError;
    else if (time_limit_exceeded.load())
      return pgTimeout;
  }
  return pgInternalError;
}

result_t ProcessGroupNCCL::reduce(const void *sendbuff, void *recvbuff,
                                  size_t count, OwnTensor::Dtype dtype, op_t op,
                                  int root, bool sync) {
  auto work_obj =
      reduce_async(sendbuff, recvbuff, count, dtype, op, root, sync);

  if (work_obj->is_success())
    return pgSuccess;
  else {
    if (work_obj->get_lastCudaError() != cudaSuccess)
      return pgCudaError;
    else if (work_obj->get_ncclStatus() != ncclSuccess)
      return pgNcclError;
    else if (time_limit_exceeded.load())
      return pgTimeout;
  }
  return pgInternalError;
}

result_t ProcessGroupNCCL::scatter(const void *sendbuff, void *recvbuff,
                                   size_t count, OwnTensor::Dtype dtype,
                                   int root, bool sync) {
  auto work_obj = scatter_async(sendbuff, recvbuff, count, dtype, root, sync);

  if (work_obj->is_success())
    return pgSuccess;
  else {
    if (work_obj->get_lastCudaError() != cudaSuccess)
      return pgCudaError;
    else if (work_obj->get_ncclStatus() != ncclSuccess)
      return pgNcclError;
    else if (time_limit_exceeded.load())
      return pgTimeout;
  }
  return pgInternalError;
}

result_t ProcessGroupNCCL::broadcast(const void *sendbuff, void *recvbuff,
                                     size_t count, OwnTensor::Dtype dtype,
                                     int root, bool sync) {
  cudaSetDevice(local_rank_);
  auto work_obj = broadcast_async(sendbuff, recvbuff, count, dtype, root, sync);

  if (work_obj->is_success())
    return pgSuccess;
  else {
    if (work_obj->get_lastCudaError() != cudaSuccess)
      return pgCudaError;
    else if (work_obj->get_ncclStatus() != ncclSuccess)
      return pgNcclError;
    else if (time_limit_exceeded.load())
      return pgTimeout;
  }
  return pgInternalError;
}

result_t ProcessGroupNCCL::alltoall(const void *sendbuff, void *recvbuff,
                                    size_t count, OwnTensor::Dtype dtype,
                                    bool sync) {
  auto work_obj = alltoall_async(sendbuff, recvbuff, count, dtype, sync);

  if (work_obj->is_success())
    return pgSuccess;
  else {
    if (work_obj->get_lastCudaError() != cudaSuccess)
      return pgCudaError;
    else if (work_obj->get_ncclStatus() != ncclSuccess)
      return pgNcclError;
    else if (time_limit_exceeded.load())
      return pgTimeout;
  }
  return pgInternalError;
}

result_t ProcessGroupNCCL::sendrecv(const void *sendbuff, void *recvbuff,
                                    int send_rank, int recv_rank, size_t count,
                                    OwnTensor::Dtype dtype, bool sync) {
  auto work_obj = sendrecv_async(sendbuff, recvbuff, send_rank, recv_rank,
                                 count, dtype, sync);

  if (work_obj->is_success())
    return pgSuccess;
  else {
    if (work_obj->get_lastCudaError() != cudaSuccess)
      return pgCudaError;
    else if (work_obj->get_ncclStatus() != ncclSuccess)
      return pgNcclError;
    else if (time_limit_exceeded.load())
      return pgTimeout;
  }
  return pgInternalError;
}

// broadcast_coalesced

result_t ProcessGroupNCCL::broadcast_coalesced(
    const std::vector<OwnTensor::Tensor> &tensor_list,
    OwnTensor::Tensor &output_tensor, size_t buffer_size /*in MB*/,
    int rank /*default root rank is 0 unless specified*/) {
  cudaSetDevice(local_rank_);
  if (tensor_list.empty()) {
    return pgInternalError;
  }

  output_tensor.copy_(OwnTensor::Tensor::flatten_concat(tensor_list));

  std::shared_ptr<Work> work =
      ProcessGroupNCCL::broadcast_inplace(output_tensor, rank, true);

  return pgSuccess;
}

// all reduce
std::shared_ptr<Work>
ProcessGroupNCCL::all_reduce_async(const void *sendbuff, void *recvbuff,
                                   size_t count, OwnTensor::Dtype dtype,
                                   op_t operation, bool sync_) {
  return launch_work_collectives(
      communication_stream_,
      [&]() -> ncclResult_t {
        return ncclAllReduce(
            sendbuff, recvbuff, count, ncclTypeConversion(dtype),
            ncclOperationConversion(operation), comm_, communication_stream_);
      },
      sync_);
}

// reduce
std::shared_ptr<Work>
ProcessGroupNCCL::reduce_async(const void *sendbuff, void *recvbuff,
                               size_t count, OwnTensor::Dtype dtype, op_t op,
                               int root, bool sync_) {

  return launch_work_collectives(
      communication_stream_,
      [&]() -> ncclResult_t {
        return ncclReduce(sendbuff, recvbuff, count, ncclTypeConversion(dtype),
                          ncclOperationConversion(op), root, comm_,
                          communication_stream_);
      },
      sync_);
}

// reduce scatter

std::shared_ptr<Work> ProcessGroupNCCL::reduce_scatter_async(
    const void *sendbuff, void *recvbuff, size_t recv_count,
    OwnTensor::Dtype dtype, op_t operation, bool sync_) {

  return launch_work_collectives(
      communication_stream_,
      [&]() -> ncclResult_t {
        return ncclReduceScatter(
            sendbuff, recvbuff, recv_count, ncclTypeConversion(dtype),
            ncclOperationConversion(operation), comm_, communication_stream_);
      },
      sync_);
}

// all gather

std::shared_ptr<Work> ProcessGroupNCCL::all_gather_async(const void *sendbuff,
                                                         void *recvbuff,
                                                         size_t sendcount,
                                                         OwnTensor::Dtype dtype,
                                                         bool sync_) {
  size_t type_size = OwnTensor::Tensor::dtype_size(dtype);
  const void *actual_send =
      static_cast<const char *>(sendbuff) + rank_ * sendcount * type_size;
  return launch_work_collectives(
      communication_stream_,
      [&]() -> ncclResult_t {
        return ncclAllGather(sendbuff, recvbuff, sendcount,
                             ncclTypeConversion(dtype), comm_,
                             communication_stream_);
      },
      sync_);
}

// gather

std::shared_ptr<Work> ProcessGroupNCCL::gather_async(const void *sendbuff,
                                                     void *recvbuff,
                                                     size_t count,
                                                     OwnTensor::Dtype dtype,
                                                     int root, bool sync_) {
  return launch_work_collectives(
      communication_stream_,
      [&]() -> ncclResult_t {
        return ncclGather(sendbuff, recvbuff, count, ncclTypeConversion(dtype),
                          root, comm_, communication_stream_);
      },
      sync_);
}

// scatter

std::shared_ptr<Work> ProcessGroupNCCL::scatter_async(const void *sendbuff,
                                                      void *recvbuff,
                                                      size_t count,
                                                      OwnTensor::Dtype dtype,
                                                      int root, bool sync_) {
  return launch_work_collectives(
      communication_stream_,
      [&]() -> ncclResult_t {
        return ncclScatter(sendbuff, recvbuff, count, ncclTypeConversion(dtype),
                           root, comm_, communication_stream_);
      },
      sync_);
}

// broadcast

std::shared_ptr<Work> ProcessGroupNCCL::broadcast_async(const void *sendbuff,
                                                        void *recvbuff,
                                                        size_t count,
                                                        OwnTensor::Dtype dtype,
                                                        int root, bool sync_) {
  cudaSetDevice(local_rank_);
  return launch_work_collectives(
      communication_stream_,
      [&]() -> ncclResult_t {
        return ncclBroadcast(sendbuff, recvbuff, count,
                             ncclTypeConversion(dtype), root, comm_,
                             communication_stream_);
      },
      sync_);
}

// all to all

std::shared_ptr<Work> ProcessGroupNCCL::alltoall_async(const void *sendbuff,
                                                       void *recvbuff,
                                                       size_t count,
                                                       OwnTensor::Dtype dtype,
                                                       bool sync_) {
  return launch_work_collectives(
      communication_stream_,
      [&]() -> ncclResult_t {
        return ncclAlltoAll(sendbuff, recvbuff, count,
                            ncclTypeConversion(dtype), comm_,
                            communication_stream_);
      },
      sync_);
}

// Sparse alltoall via ncclGroupStart + ncclSend/ncclRecv (matches PyTorch all2all_single_unequal_split)
std::shared_ptr<Work> ProcessGroupNCCL::alltoallv_async(
    const void *sendbuff, const size_t *sendcounts, const size_t *senddispls,
    void *recvbuff, const size_t *recvcounts, const size_t *recvdispls,
    OwnTensor::Dtype dtype, bool sync_) {
  return launch_work_collectives(
      communication_stream_,
      [&]() -> ncclResult_t {
        ncclDataType_t nccl_type = ncclTypeConversion(dtype);
        size_t elem_size = OwnTensor::Tensor::dtype_size(dtype);
        ncclResult_t res = ncclGroupStart();
        if (res != ncclSuccess) return res;
        for (int r = 0; r < world_size_; ++r) {
          if (sendcounts[r] > 0) {
            res = ncclSend(
                static_cast<const char*>(sendbuff) + senddispls[r] * elem_size,
                sendcounts[r], nccl_type, r, comm_, communication_stream_);
            if (res != ncclSuccess) return res;
          }
          if (recvcounts[r] > 0) {
            res = ncclRecv(
                static_cast<char*>(recvbuff) + recvdispls[r] * elem_size,
                recvcounts[r], nccl_type, r, comm_, communication_stream_);
            if (res != ncclSuccess) return res;
          }
        }
        return ncclGroupEnd();
      },
      sync_);
}

// void process_parallel(std::vector<int>& executable, int start, int end){
//     for(int st = start; st <= end; st++){
//         sendrecv_async(sendbuff, executable[st], )
//     }
// }

result_t ProcessGroupNCCL::send_recv_ranks(const void *sendbuff, void *recvbuff,
                                           int send_rank,
                                           std::vector<int> &ranks,
                                           size_t count, OwnTensor::Dtype dtype,
                                           bool sync_) {

  for (auto i = 0; i < ranks.size(); i++) {
    sendrecv_async(sendbuff, recvbuff, send_rank, ranks[i], count, dtype,
                   sync_);
  }
  return pgSuccess;
}

result_t ProcessGroupNCCL::send_recv_ranks(
    const void *sendbuff, std::vector<void *> &recvbuffs, int send_rank,
    std::vector<int> &ranks, size_t count, OwnTensor::Dtype dtype, bool sync_) {

  for (auto i = 0; i < ranks.size(); i++) {
    sendrecv_async(sendbuff, recvbuffs[i], send_rank, ranks[i], count, dtype,
                   sync_);
  }

  // std::for_each(std::execution::par, ranks.begin(), ranks.end)
  return pgSuccess;
}

// point to point communications

std::shared_ptr<Work>
ProcessGroupNCCL::sendrecv_async(const void *sendbuff, void *recvbuff,
                                 int send_rank, int recv_rank, size_t count,
                                 OwnTensor::Dtype dtype, bool sync_) {
  return launch_work_collectives(
      communication_stream_,
      [&]() -> ncclResult_t {
        ncclGroupStart();
        ncclResult_t nccl_send_status =
            ncclSend(sendbuff, count, ncclTypeConversion(dtype), recv_rank,
                     comm_, communication_stream_);
        ncclResult_t nccl_recv_status =
            ncclRecv(recvbuff, count, ncclTypeConversion(dtype), send_rank,
                     comm_, communication_stream_);
        ncclGroupEnd();
        return nccl_send_status;
      },
      sync_);
}

// Lazily create the dedicated non-blocking CP-ring stream.
cudaStream_t ProcessGroupNCCL::cpRingStream() {
  if (cp_ring_stream_ == nullptr) {
    // DIAGNOSTIC: CP_RING_BLOCKING=1 creates a BLOCKING stream, which implicitly
    // orders with the legacy default stream (stream 0). If this is stable while
    // the non-blocking stream explodes, the corruption is the non-blocking
    // concurrency with stream-0 compute; if it still explodes, the NCCL op
    // itself is at fault, not stream-0 interplay.
    if (std::getenv("CP_RING_BLOCKING")) {
      CUDACHECK(cudaStreamCreate(&cp_ring_stream_));
      if (rank_ == 0) fprintf(stderr, "[CP ring] CP_RING_BLOCKING=1: blocking ring stream\n");
    } else {
      CUDACHECK(cudaStreamCreateWithFlags(&cp_ring_stream_, cudaStreamNonBlocking));
    }
  }
  return cp_ring_stream_;
}

// sendrecv on an EXPLICIT stream (the dedicated CP-ring stream), not the shared
// communication_stream_. The Work's event is recorded (in launch_work_collectives)
// on this stream AFTER the grouped send+recv -> waiting on it waits for the recv.
std::shared_ptr<Work> ProcessGroupNCCL::sendrecv_async_stream(
    const void *sendbuff, void *recvbuff, int send_rank, int recv_rank,
    size_t count, OwnTensor::Dtype dtype, cudaStream_t stream) {
  // Runs on the DEDICATED cp_comm_ (not comm_) so the ring never shares a
  // communicator with the loss/param all-reduces.
  auto w = launch_work_collectives(
      stream,
      [&]() -> ncclResult_t {
        ncclGroupStart();
        ncclResult_t s = ncclSend(sendbuff, count, ncclTypeConversion(dtype),
                                  recv_rank, cp_comm_, stream);
        ncclRecv(recvbuff, count, ncclTypeConversion(dtype), send_rank, cp_comm_,
                 stream);
        ncclGroupEnd();
        return s;
      },
      false, cp_comm_);
  // DIAGNOSTIC: CP_SYNC_RING=1 forces a full host-side sync of the ring stream
  // after every exchange, keeping the overlap CODE PATH but removing all
  // concurrency. If divergence vanishes under this -> the bug is a missing
  // synchronization (a race); if it persists -> the bug is logic, not timing.
  if (cp_sync_ring_) CUDACHECK(cudaStreamSynchronize(stream));
  return w;
}

std::shared_ptr<Work> ProcessGroupNCCL::alltoallv_async_stream(
    const void *sendbuff, const size_t *sendcounts, const size_t *senddispls,
    void *recvbuff, const size_t *recvcounts, const size_t *recvdispls,
    OwnTensor::Dtype dtype, cudaStream_t stream) {
  ncclDataType_t nccl_type = ncclTypeConversion(dtype);
  size_t elem_size = OwnTensor::Tensor::dtype_size(dtype);
  auto w = launch_work_collectives(
      stream,
      [&]() -> ncclResult_t {
        ncclGroupStart();
        for (int r = 0; r < world_size_; ++r) {
          if (sendcounts[r] > 0)
            ncclSend(static_cast<const char *>(sendbuff) + senddispls[r] * elem_size,
                     sendcounts[r], nccl_type, r, cp_comm_, stream);
          if (recvcounts[r] > 0)
            ncclRecv(static_cast<char *>(recvbuff) + recvdispls[r] * elem_size,
                     recvcounts[r], nccl_type, r, cp_comm_, stream);
        }
        return ncclGroupEnd();
      },
      false, cp_comm_);
  // DIAGNOSTIC: see sendrecv_async_stream above (CP_SYNC_RING=1).
  if (cp_sync_ring_) CUDACHECK(cudaStreamSynchronize(stream));
  return w;
}

std::shared_ptr<Work> ProcessGroupNCCL::all_gather_async_stream(
    const void *sendbuff, void *recvbuff, size_t sendcount,
    OwnTensor::Dtype dtype, cudaStream_t stream) {
  ncclDataType_t nccl_type = ncclTypeConversion(dtype);
  auto w = launch_work_collectives(
      stream,
      [&]() -> ncclResult_t {
        return ncclAllGather(sendbuff, recvbuff, sendcount, nccl_type, cp_comm_,
                             stream);
      },
      false, cp_comm_);
  // DIAGNOSTIC: see sendrecv_async_stream / alltoallv_async_stream (CP_SYNC_RING=1).
  if (cp_sync_ring_) CUDACHECK(cudaStreamSynchronize(stream));
  return w;
}

std::shared_ptr<Work> ProcessGroupNCCL::send_async(const void *sendbuff,
                                                   size_t count,
                                                   OwnTensor::Dtype dtype,
                                                   int recv_rank, bool sync_) {
  return launch_work_collectives(
      communication_stream_,
      [&]() -> ncclResult_t {
        return ncclSend(sendbuff, count, ncclTypeConversion(dtype), recv_rank,
                        comm_, communication_stream_);
      },
      sync_);
}

std::shared_ptr<Work> ProcessGroupNCCL::recieve_async(void *recvbuff,
                                                      size_t count,
                                                      OwnTensor::Dtype dtype,
                                                      int send_rank,
                                                      bool sync_) {
  return launch_work_collectives(
      communication_stream_,
      [&]() -> ncclResult_t {
        return ncclRecv(recvbuff, count, ncclTypeConversion(dtype), send_rank,
                        comm_, communication_stream_);
      },
      sync_);
}

std::shared_ptr<Work>
ProcessGroupNCCL::broadcast_inplace(OwnTensor::Tensor &sendbuff, int rank,
                                    bool sync_) {
  cudaSetDevice(local_rank_);
  if (sendbuff.numel() == 0) {
    throw std::runtime_error("empty tensor....");
  }

  // std::cout << (sendbuff.is_cuda() ? "CUDA: " : "CPU: ")  <<
  // sendbuff.device().index << std::endl; sendbuff.display();
  auto work_ = broadcast_async(sendbuff.data(), sendbuff.data(),
                               sendbuff.numel(), sendbuff.dtype(), rank, sync_);

  return work_;
}

template <typename NCCLFUNC>
std::shared_ptr<Work>
ProcessGroupNCCL::launch_work_collectives(cudaStream_t stream, NCCLFUNC nccl_op,
                                          bool to_sync, ncclComm_t work_comm) {
  // create the work collective (Work uses the op's comm for async-error checks)
  auto work_collectives =
      std::make_shared<Work>(stream, work_comm ? work_comm : comm_);

  PG_CHECK_FALSE(work_collectives->begin(),
                 "Error in work_collective->begin()");

  ncclResult_t nccl_status = nccl_op();

  work_collectives->setNcclStatus(nccl_status);

  work_collectives->event_record();

  if (to_sync) {
    bool status = work_collectives->wait();

    if (!status) {
      cudaError_t err = work_collectives->get_lastCudaError();
      throw std::runtime_error(
          std::string("Synchronization failed at line 484 ") +
          ncclGetErrorString(work_collectives->get_ncclStatus()) + " " +
          cudaGetErrorString(work_collectives->get_lastCudaError()));
    }
  } else {
    // std::future<bool> collective_output = std::async(std::launch::async,
    // [&,this](std::shared_ptr<Work> work_collectives)->bool{
    //     const auto start_time = std::chrono::system_clock::now();
    //     double max_duration = 120.0f;
    //     while(true){
    //         // to check the query
    //         std::future<bool> progress = std::async(std::launch::async,
    //         [&,this](std::shared_ptr<Work> work_collectives)->bool{
    //             return work_collectives->query();
    //         }, work_collectives);
    //         if(progress.get()) return true;

    //         //to check if the time is exceeded
    //         std::chrono::duration<double> elapsed_seconds =
    //         std::chrono::system_clock::now() - start_time;
    //         if(elapsed_seconds.count() > max_duration) {
    //             time_limit_exceeded.store(true);
    //             return false;
    //         }
    //         std::this_thread::sleep_for(std::chrono::seconds(TIME_INTERVAL));
    //     }
    // }, work_collectives);

    // for now this method. will use the future status once have a better
    // understanding
  }
  return work_collectives;
}

#pragma once

#include <iostream>
#include <nccl.h>
#include <cuda_runtime.h>
#include <mutex>
#include <thread>
#include <chrono>
#include <mpi.h>
#include <memory.h>
#include <unordered_map>
#include <cstdint>
#include <atomic>
#include "TensorLib.h"
#include "CpuSync_fixed.hpp"
// #include "ProcessGroupMain.h"


#define NCCLCHECK(cmd)                                                                       \
    do{                                                                                      \
        ncclResult_t r = (cmd);                                                              \
        if(r != ncclSuccess) {                                                               \
            throw std::runtime_error(std::string("CUDA error: ") + ncclGetErrorString(r));   \
        }                                                                                    \
    } while(0)                                                                               

#define CUDACHECK(cmd)                                                                       \
    do{                                                                                      \
        cudaError_t r = (cmd);                                                               \
        if(r != cudaSuccess) {                                                               \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(r));   \
        }                                                                                    \
    } while(0)                                                                               


// class Work;


typedef enum {
    pgSuccess = 0,
    pgTimeout = 1,
    pgCudaError = 2,
    pgNcclError = 3,
    pgCommunicationError = 4,
    pgInternalError = 5

} result_t;

typedef enum{
    sum = 0,
    max = 1,
    min = 2,
    avg = 3,
    mul = 4
} op_t;


inline std::string pgGetError(result_t error){
    switch(error){
        case pgTimeout:
            return "Process Group timeout";
        case pgCudaError:
            return "Cuda Error";
        case pgNcclError:
            return "NCCL Error";
        case pgCommunicationError:
            return "Internal Communication Error";
        case pgInternalError:
            return "Internal Code Error";
        default:
            return "Success"; 
    }
}


inline constexpr ncclDataType_t ncclTypeConversion(OwnTensor::Dtype type) {
    switch(type){
        // case OwnTensor::Dtype::Int8:
        //     return ncclInt8;
        case OwnTensor::Dtype::Int32:
            return ncclInt32;
        case OwnTensor::Dtype::Int64:
            return ncclInt64;
        case OwnTensor::Dtype::Float16:
            return ncclFloat16;
        case OwnTensor::Dtype::Bfloat16:
            return ncclBfloat16;
        case OwnTensor::Dtype::Float32:
            return ncclFloat32;
        case OwnTensor::Dtype::Float64:
            return ncclFloat64;
        default:
            throw std::runtime_error("No such Datatype matching");
    }
}

inline constexpr ncclRedOp_t ncclOperationConversion(op_t op){
    switch(op){
        // case OwnTensor::Dtype::Int8:
        //     return ncclInt8;
        case sum:
            return ncclSum;
        case avg:
            return ncclAvg;
        case max:
            return ncclMax;
        case min:
            return ncclMin;
        case mul:
            return ncclProd;
        default:
            throw std::runtime_error("No such Datatype matching");
    }
}



class ProcessGroupNCCL{

public: 
    ProcessGroupNCCL(int world_size, int rank, ncclUniqueId& id, std::shared_ptr<Work> work_obj, cudaStream_t& stream, int root = 0 );
    ProcessGroupNCCL()= default;
    ~ProcessGroupNCCL();


    template<typename NCCLFUNC>
    std::shared_ptr<Work> launch_work_collectives( cudaStream_t stream, NCCLFUNC nccl_op, bool to_sync = false, ncclComm_t work_comm = nullptr);

    //collectves

    result_t all_reduce(const void* sendbuff, void* recvbuff,size_t count, OwnTensor::Dtype dtype, op_t operation, bool sync = false);

    result_t reduce_scatter(const void* sendbuff, void* recvbuff, size_t recv_count, OwnTensor::Dtype dtype, op_t operation, bool sync = false);

    result_t all_gather(const void* sendbuff, void* recvbuff, size_t sendcount, OwnTensor::Dtype dtype, bool sync = false);

    result_t gather(const void* sendbuff, void* recvbuff, size_t count, OwnTensor::Dtype dtype, int root, bool sync = false);

    result_t reduce(const void* sendbuff, void* recvbuff, size_t count, OwnTensor::Dtype dtype, op_t op, int root, bool sync = false);

    result_t scatter(const void* sendbuff, void* recvbuff, size_t count, OwnTensor::Dtype dtype, int root, bool sync = false);

    result_t broadcast(const void* sendbuff, void* recvbuff, size_t count, OwnTensor::Dtype dtype, int root, bool sync = false);

    result_t alltoall(const void* sendbuff, void* recvbuff, size_t count, OwnTensor::Dtype dtype, bool sync = false);

    result_t sendrecv(const void* sendbuff, void* recvbuff, int send_rank, int recv_rank, size_t count, OwnTensor::Dtype dtype, bool sync = false);

    result_t broadcast_coalesced(const std::vector<OwnTensor::Tensor>& tensor_list, OwnTensor::Tensor& output_tensor, size_t buffer_size = 25 , int rank = 0);






    //async calls
    std::shared_ptr<Work> all_reduce_async(const void* sendbuff, void* recvbuff,size_t count, OwnTensor::Dtype dtype, op_t operation, bool sync_ = false);
    std::shared_ptr<Work> reduce_scatter_async(const void* sendbuff, void* recvbuff, size_t recv_count, OwnTensor::Dtype dtype, op_t operation, bool sync_ = false);
    std::shared_ptr<Work> all_gather_async(const void* sendbuff, void* recvbuff, size_t sendcount, OwnTensor::Dtype dtype, bool sync_ = false);
    std::shared_ptr<Work> gather_async(const void* sendbuff, void* recvbuff, size_t count, OwnTensor::Dtype dtype, int root, bool sync_ = false);
    std::shared_ptr<Work> reduce_async(const void* sendbuff, void* recvbuff, size_t count, OwnTensor::Dtype dtype, op_t op, int root, bool sync_ = false);
    std::shared_ptr<Work> scatter_async(const void* sendbuff, void* recvbuff, size_t count, OwnTensor::Dtype dtype, int root, bool sync_ = false);
    std::shared_ptr<Work> broadcast_async(const void* sendbuff, void* recvbuff, size_t count, OwnTensor::Dtype dtype, int root, bool sync_ = false);
    std::shared_ptr<Work> alltoall_async(const void* sendbuff, void* recvbuff, size_t count, OwnTensor::Dtype dtype, bool sync_ = false);
    // Sparse alltoall: per-rank send/recv counts and byte offsets (like ncclAllToAllv / PyTorch permute_tensor)
    std::shared_ptr<Work> alltoallv_async(const void* sendbuff, const size_t* sendcounts, const size_t* senddispls,
                                          void* recvbuff, const size_t* recvcounts, const size_t* recvdispls,
                                          OwnTensor::Dtype dtype, bool sync_ = false);
    std::shared_ptr<Work> send_async(const void* sendbuff, size_t count, OwnTensor::Dtype dtype, int recv_rank, bool sync_ = false);
    std::shared_ptr<Work> recieve_async(void* recvbuff, size_t count, OwnTensor::Dtype dtype, int send_rank, bool sync_ = false);
    std::shared_ptr<Work> sendrecv_async(const void* sendbuff, void* recvbuff, int send_rank, int recv_rank, size_t count, OwnTensor::Dtype dtype, bool sync_ = false);
    std::shared_ptr<Work> broadcast_inplace(OwnTensor::Tensor& sendbuff, int root, bool sync_ = false);
    result_t send_recv_ranks(const void* sendbuff, void* recvbuff, int send_rank, std::vector<int>& ranks, size_t count, OwnTensor::Dtype dtype, bool sync_ = false);
    result_t send_recv_ranks(const void* sendbuff, std::vector<void*>& recvbuffs, int send_rank, std::vector<int>& ranks, size_t count, OwnTensor::Dtype dtype, bool stnc = false);
    //nvshmem

    //return collectives
    inline int get_rank() noexcept{ return rank_; }
    inline int get_worldsize() noexcept{ return world_size_; }
    inline int get_local_rank() noexcept{ return local_rank_; }
    void set_owns_stream(const bool state){ owns_stream_ = state; }
    bool is_owns_stream(){ return owns_stream_; }  
    std::shared_ptr<Work> get_work_obj(){ return work_obj_; }
    cudaStream_t getStream(){return communication_stream_;}

    // Dedicated NON-BLOCKING stream for the CP ring K/V rotation ONLY (lazily
    // created). The shared communication_stream_ stays BLOCKING so DP/TP
    // collectives keep their implicit FIFO ordering; only the CP ring overlaps
    // local attention compute. Mirrors torchtitan's cp_comm_stream model.
    cudaStream_t cpRingStream();
    std::shared_ptr<Work> sendrecv_async_stream(
        const void* sendbuff, void* recvbuff, int send_rank, int recv_rank,
        size_t count, OwnTensor::Dtype dtype, cudaStream_t stream);
    std::shared_ptr<Work> alltoallv_async_stream(
        const void* sendbuff, const size_t* sendcounts, const size_t* senddispls,
        void* recvbuff, const size_t* recvcounts, const size_t* recvdispls,
        OwnTensor::Dtype dtype, cudaStream_t stream);
    // Async all_gather on an explicit stream using the dedicated CP communicator
    // (cp_comm_), returning a Work (no CPU sync). Mirrors alltoallv_async_stream
    // so the AllGather rotator can overlap the gather with local attention compute
    // instead of CPU-blocking on the shared comm.
    std::shared_ptr<Work> all_gather_async_stream(
        const void* sendbuff, void* recvbuff, size_t sendcount,
        OwnTensor::Dtype dtype, cudaStream_t stream);

    //synchronization using cudaEvent_t
    bool blockStreamEvent();

    bool blockStream();

    void start_time();
    void end_time(float& ms);
    void set_tle(bool status){
        time_limit_exceeded.store(status);
    }
    static std::shared_ptr<ProcessGroupNCCL> init_process_group(int world_size, int rank, cudaStream_t stream = 0);

private:
    int rank_ = 0;
    int world_size_ = 1;
    int local_rank_ = 0;
    int node_ = 1;
    int gpus_per_node_ = 8;
    int root_ = 0;  
    ncclUniqueId id_;
    ncclUniqueId cp_id_;  // unique id for the dedicated CP-ring communicator
    std::shared_ptr<Work> work_obj_;
    std::mutex mutex_lock_; //to access the work_obj
    cudaStream_t communication_stream_ = 0;
    cudaStream_t cp_ring_stream_ = nullptr;  // dedicated non-blocking CP-ring stream
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
    ncclComm_t comm_ = nullptr;
    // Dedicated communicator for the CP ring (overlap path) — isolates ring
    // traffic from the loss/param all-reduces on comm_, matching PyTorch's
    // separate-comm-per-parallelism-dimension design. Driven exclusively by the
    // cp_ring_stream_, so ring collectives serialize on one stream and are never
    // concurrent on cp_comm_ (NCCL-safe, incl. the AlltoAll variant).
    ncclComm_t cp_comm_ = nullptr;
    // DIAGNOSTIC: CP_SYNC_RING=1 -> host-sync the ring stream after each ring
    // exchange (overlap code path kept, concurrency removed). Race vs logic test.
    bool cp_sync_ring_ = false;
    bool owns_stream_ = false;

    std::string cuda_error_in_nccl = ""; 
    std::atomic<bool> time_limit_exceeded {false};  
    std::uint64_t flag_for_polling_ = 0;
};


// function for the initialization of the process group should be done outside the process group class.
// because, if inside the class, creating the constructor requires creating an object which reduces the aim.


std::shared_ptr<ProcessGroupNCCL> init_process_group(int world_size, int rank, cudaStream_t stream = 0);
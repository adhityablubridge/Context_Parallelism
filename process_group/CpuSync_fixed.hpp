#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <mutex>
#include <cuda_runtime.h>
#include <nccl.h>
#include <memory>
#include <thread>
#include "TensorLib.h"


// CpuSync_fixed.hpp - Fixed version with lazy CUDA event creation
// This fixes the "invalid resource handle" error when the event is created
// on a different CUDA device than the stream.

class Work{

public:

    //will only get the stream as argument.
    explicit Work(cudaStream_t stream, ncclComm_t comm = nullptr) : stream_(stream), comm_(comm), completed_(true), success_(false), event_(nullptr) {
        // Event will be created lazily in event_record() to ensure it's on the correct device
        // std::cout << "Using CpuSync_fixed.hpp Work class" << std::endl;
    }

    ~Work() {
        // event may be destroyed after use; only if it was created
        if (event_ != nullptr) {
            cudaError_t err = cudaEventDestroy(event_);
            if (err != cudaSuccess) {
                std::cerr << "CpuSync_fixed: Work::~Work: cudaEventDestroy failed: " << cudaGetErrorString(err) << std::endl;
            }
        }
    }

    // will create the event with a cudaEventDisableTimings flag.
    // to avoid unnecessary time recording by the cuda for events.
    //uses lock()
    bool begin(){
        std::lock_guard<std::mutex> lk(mutex_);
        completed_ = false;
        success_ = false;
        return true;
    }

    // will stop the recoding. will put a asynchronous record on the kernel execution to note that the task is completed.
    // will set the completed and success as true if the execution is completed properly.
    // throws error if the status is not equals to cudaSuccess.
    //uses unique_lock()
    bool event_record(){
        // Lazy event creation - create on the current device context
        if (event_ == nullptr) {
            cudaError_t create_err = cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
            if (create_err != cudaSuccess) {
                std::cerr << "[Work::event_record] cudaEventCreate FAILED: " << cudaGetErrorString(create_err) << std::endl;
                std::lock_guard<std::mutex> lk(mutex_);
                last_err = create_err;
                success_ = false;
                completed_ = true;
                return false;
            }
        }

        cudaError_t err = cudaEventRecord(event_, stream_);
        if (err != cudaSuccess) {
            std::cerr << "[Work::event_record] cudaEventRecord FAILED: " << cudaGetErrorString(err) << std::endl;
            std::lock_guard<std::mutex> lk(mutex_);
            last_err = err;
            success_ = false;
            completed_ = true;
            return false;
        }
        return true;
    }

    void setCommunicator(ncclComm_t comm){ comm_ = comm; }

    // returns if completed is true
    bool is_completed(){
        // std::unique_lock<std::mutex> lock(mutex_);
        std::lock_guard<std::mutex> lock(mutex_);
        return completed_;
    }

    // will synchronize the cpu until the stream is fully executed.
    // only when the kernels in the stream is completed, the getStatus() will be called function is returned
    bool wait(){
        if (event_ == nullptr) {
            // No event was ever recorded, nothing to wait for
            std::lock_guard<std::mutex> lock(mutex_);
            completed_ = true;
            success_ = true;
            return true;
        }
        cudaError_t err = cudaStreamWaitEvent(OwnTensor::cuda::getCurrentStream(), event_, 0);
        // cudaError_t err = cudaStreamWaitEvent(OwnTensor::cuda::getCurrentStream(), event_, 0);
        ncclResult_t async_error;
        ncclCommGetAsyncError(comm_, &async_error);
        std::lock_guard<std::mutex> lock(mutex_);
        completed_ = true;
        if (async_error != ncclSuccess) {
            nccl_status = async_error;
            success_ = false;
            return false;
        }
        if( err != cudaSuccess || nccl_status != ncclSuccess ){
            success_ = false;
            last_err = err;
            return false;
        }

        success_ = true;
        return true;
    }

    //will return the success.
    bool is_success(){
        // std::unique_lock<std::mutex> lock(mutex_);
        std::lock_guard<std::mutex> lock(mutex_);
        return success_;
    }

    bool query() {
        if (event_ == nullptr) {
            // No event was ever recorded
            return true;
        }
        cudaError_t err = cudaEventQuery(event_);
        if(err  == cudaErrorNotReady){ return false; }
        ncclResult_t async_error;
        ncclCommGetAsyncError(comm_, &async_error);
        std::lock_guard<std::mutex> lock(mutex_);

        completed_ = true;
        if (async_error != ncclSuccess) {
            nccl_status = async_error;
            success_ = false;
        }
        if(err != cudaSuccess){
            success_ = false;
            last_err = err;
            cuda_error = cudaGetErrorString(last_err);
            return true;
        }

        success_ = (nccl_status == ncclSuccess);
        return true;
    }

    void setNcclStatus(ncclResult_t status){
        // std::unique_lock<std::mutex> lock(mutex_);
        std::lock_guard<std::mutex> lock(mutex_);
        nccl_status = status;
        // lock.unlock();
        return;
    }

    ncclResult_t get_ncclStatus() {
        // std::unique_lock<std::mutex> lock(mutex_);
        std::lock_guard<std::mutex> lock(mutex_);
        return nccl_status;
    }

    cudaError_t get_lastCudaError() {
        // std::unique_lock<std::mutex> lock(mutex_);
        std::lock_guard<std::mutex> lock(mutex_);
        return last_err;
    }

    // Makes the given stream wait for this work's event to complete (GPU-side sync)
    // This is non-blocking to the CPU - the stream will wait on the GPU
    bool streamWait(cudaStream_t waitingStream) {
        if (event_ == nullptr) {
            // No event was ever recorded, nothing to wait for
            return true;
        }
        cudaError_t err = cudaStreamWaitEvent(waitingStream, event_, 0);
        if (err != cudaSuccess) {
            std::cerr << "[Work::streamWait] cudaStreamWaitEvent FAILED: " << cudaGetErrorString(err) << std::endl;
            std::lock_guard<std::mutex> lock(mutex_);
            last_err = err;
            return false;
        }
        return true;
    }

    // Get the underlying CUDA event (for advanced use cases)
    cudaEvent_t getEvent() const {
        return event_;
    }


private:
    std::mutex mutex_;
    cudaEvent_t event_;
    bool completed_ = true;
    bool success_ = false;
    cudaStream_t stream_;
    ncclResult_t nccl_status{ncclSuccess};
    cudaError_t last_err{cudaSuccess};
    std::string cuda_error;
    ncclComm_t comm_;


};

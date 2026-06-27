#pragma once
#include <vector>
#include <memory>
#include <string>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <numeric>
#include "process_group/ProcessGroupNCCL.h"
#include <mpi.h>
#include <nccl.h>



class DeviceMesh {
public:

    DeviceMesh(const std::vector<int>& mesh_shape, 
                       const std::vector<int>& device_ids);
    
    // Disable copying - DeviceMesh owns MPI_Comm handles that must not be double-freed
    DeviceMesh(const DeviceMesh&) = delete;
    DeviceMesh& operator=(const DeviceMesh&) = delete;
    
    // Allow move semantics
    DeviceMesh(DeviceMesh&&) = default;
    DeviceMesh& operator=(DeviceMesh&&) = default;
    
    ~DeviceMesh();
    
    std::vector<int64_t> get_coordinate(int rank) const;
    
    
    int get_rank(const std::vector<int64_t>& coordinate) const;

 
    std::shared_ptr<ProcessGroupNCCL> get_process_group(int64_t mesh_dim);
 
    std::vector<int> get_group_ranks(int64_t mesh_dim) const;
    
    
    int get_dim_rank(int64_t mesh_dim) const ;
    
    
    int64_t ndim() const { return mesh_shape_.size(); }
    const std::vector<int>& shape() const { return mesh_shape_; }
    int64_t size() const;  // Total number of devices
    int rank() const { return global_rank_; }
    int world_size() const { return total_devices_; }

    void describe() const;
    
private:
    std::vector<int> mesh_shape_;      
    std::vector<int> device_ids_;      
    int64_t total_devices_;              
    int global_rank_;                  
    std::vector<int64_t> my_coordinate_;   
    
    std::vector<std::shared_ptr<ProcessGroupNCCL>> process_groups_;
    
    
    std::vector<MPI_Comm> mpi_comms_;

    void initialize_process_groups();
    
    ncclUniqueId create_nccl_id(int root_rank, MPI_Comm comm);
};

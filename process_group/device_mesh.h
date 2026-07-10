#pragma once
#include <vector>
#include <memory>
#include <string>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <numeric>
#include "ProcessGroupNCCL.h"   // canonical (BluTrain dist/communication) PG — via -I
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

    // The WORLD (all-ranks) PG built once as the ncclCommSplit parent. Reuse this
    // for global collectives (e.g. DataParallel) instead of a 2nd init_process_group
    // (BluTrain's registry throws if init_process_group is called twice).
    std::shared_ptr<ProcessGroupNCCL> world_pg() const { return world_pg_; }
 
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

    // Parent (world) PG whose comm_ is the ncclCommSplit parent for all axes.
    std::shared_ptr<ProcessGroupNCCL> world_pg_;

    std::vector<MPI_Comm> mpi_comms_;   // unused now (kept for ABI/dtor); ncclCommSplit replaces MPI split

    void initialize_process_groups();

    ncclUniqueId create_nccl_id(int root_rank, MPI_Comm comm);
};

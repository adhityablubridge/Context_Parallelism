#include "process_group/device_mesh.h"
#include <algorithm>
#include <cassert>
#include <cstdio>


DeviceMesh::DeviceMesh(const std::vector<int>& mesh_shape, 
                       const std::vector<int>& device_ids)
    : mesh_shape_(mesh_shape) {
    
    if (mesh_shape_.empty()) {
        throw std::runtime_error("DeviceMesh: mesh_shape cannot be empty");
    }

    total_devices_ = std::accumulate(mesh_shape_.begin(), mesh_shape_.end(), 
                                     1, std::multiplies<int64_t>());
    

    MPI_Comm_rank(MPI_COMM_WORLD, &global_rank_);
    
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    if (total_devices_ != world_size) {
        throw std::runtime_error("DeviceMesh: mesh size (" + 
                                std::to_string(total_devices_) + 
                                ") must match MPI world size (" + 
                                std::to_string(world_size) + ")");
    }

    if (device_ids.empty()) {
        
        device_ids_.resize(total_devices_);
        std::iota(device_ids_.begin(), device_ids_.end(), 0);
    } else {
        if ((int)device_ids.size() != total_devices_) {
            throw std::runtime_error("DeviceMesh: device_ids size must match total devices");
        }
        device_ids_ = device_ids;
    }
    

    my_coordinate_ = get_coordinate(global_rank_);
    

    initialize_process_groups();
}

DeviceMesh::~DeviceMesh() {
    // Check if MPI is still active before freeing communicators
    int finalized = 0;
    MPI_Finalized(&finalized);
    
    if (!finalized) {
        for (auto& comm : mpi_comms_) {
            if (comm != MPI_COMM_NULL) {
                MPI_Comm_free(&comm);
            }
        }
    }
}


std::vector<int64_t> DeviceMesh::get_coordinate(int rank) const {
    std::vector<int64_t> coord(ndim());
    int64_t remaining = rank;
    
    // Row-major ordering (x,y) y changes faster
    for (int64_t dim = ndim() - 1; dim >= 0; --dim) {
        coord[dim] = remaining % mesh_shape_[dim];
        remaining /= mesh_shape_[dim];
    }
    
    return coord;
}


int DeviceMesh::get_rank(const std::vector<int64_t>& coordinate) const {
    if ((int64_t)coordinate.size() != ndim()) {
        throw std::runtime_error("DeviceMesh: coordinate size must match mesh ndim");
    }
    
    int rank = 0;
    int64_t stride = 1;
    
    // Row-major ordering
    for (int64_t dim = ndim() - 1; dim >= 0; --dim) {
        if (coordinate[dim] < 0 || coordinate[dim] >= mesh_shape_[dim]) {
            throw std::runtime_error("DeviceMesh: coordinate out of bounds");
        }
        rank += coordinate[dim] * stride;
        stride *= mesh_shape_[dim];
    }
    
    return rank;
}


std::vector<int> DeviceMesh::get_group_ranks(int64_t mesh_dim) const {
    if (mesh_dim < 0 || mesh_dim >= ndim()) {
        throw std::runtime_error("DeviceMesh: invalid mesh_dim");
    }
    
    std::vector<int> ranks;

    std::vector<int64_t> base_coord = my_coordinate_;
    
    for (int64_t i = 0; i < mesh_shape_[mesh_dim]; ++i) {
        base_coord[mesh_dim] = i;
        ranks.push_back(get_rank(base_coord));
    }
    
    return ranks;
}

int DeviceMesh::get_dim_rank(int64_t mesh_dim) const {
    if (mesh_dim < 0 || mesh_dim >= ndim()) {
        throw std::runtime_error("DeviceMesh: invalid mesh_dim");
    }
    return my_coordinate_[mesh_dim];
}

void DeviceMesh::initialize_process_groups() {
    process_groups_.resize(ndim());

    // CUDA device MUST be set before any comm/stream creation.
    int gpus_per_node = 1;
    cudaGetDeviceCount(&gpus_per_node);
    if (const char* env = std::getenv("NO_GPUS_PER_NODE")) {
        int parsed = std::atoi(env); if (parsed > 0) gpus_per_node = parsed;
    }
    int local_rank = global_rank_ % gpus_per_node;
    cudaSetDevice(local_rank);

    // 1) Build the WORLD process group (one MPI_Bcast of the ncclUniqueId inside
    //    init_process_group). Its comm_ is the ncclCommSplit parent for all axes.
    world_pg_ = init_process_group(total_devices_, global_rank_);
    ncclComm_t world_comm = world_pg_->get_comm();

    // 2) Per axis: ncclCommSplit(world_comm, color=other-coords, key=axis-coord).
    //    This is COLLECTIVE over world_comm (all ranks call it each iteration).
    //    Replaces MPI_Comm_split + per-axis MPI id-bcast, and is correct for N-D.
    for (int64_t mesh_dim = 0; mesh_dim < ndim(); ++mesh_dim) {
        // Re-assert the GLOBAL physical device each iteration: the previous axis's
        // PG ctor may have left a different device current, which would otherwise
        // create this axis's comm_stream on the wrong device (stream/comm device
        // mismatch -> NCCL "invalid resource handle").
        cudaSetDevice(local_rank);
        std::vector<int> group_ranks = get_group_ranks(mesh_dim);
        int64_t group_size = group_ranks.size();
        auto it = std::find(group_ranks.begin(), group_ranks.end(), global_rank_);
        int64_t my_group_rank = std::distance(group_ranks.begin(), it);

        int color = 0, multiplier = 1;
        for (int64_t d = ndim() - 1; d >= 0; --d) {
            if (d != mesh_dim) { color += (int)my_coordinate_[d] * multiplier; multiplier *= mesh_shape_[d]; }
        }
        int key = (int)my_coordinate_[mesh_dim];

        ncclComm_t sub_comm = nullptr;
        ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
        ncclResult_t sr = ncclCommSplit(world_comm, color, key, &sub_comm, &config);
        if (sr != ncclSuccess || sub_comm == nullptr) {
            throw std::runtime_error("DeviceMesh: ncclCommSplit failed for mesh_dim " +
                                     std::to_string(mesh_dim));
        }

        // Dedicated BLOCKING comm stream per axis PG (ring overlap uses a separate
        // non-blocking cpRingStream() with explicit event ordering).
        cudaStream_t comm_stream;
        cudaStreamCreate(&comm_stream);
        auto work_obj = std::make_shared<Work>(comm_stream, sub_comm);
        process_groups_[mesh_dim] = std::make_shared<ProcessGroupNCCL>(
            (int)group_size, (int)my_group_rank, sub_comm, work_obj, comm_stream);
        process_groups_[mesh_dim]->set_owns_stream(true);
    }
}

// ncclUniqueId DeviceMesh::create_nccl_id(int root_rank, MPI_Comm comm) {
//     ncclUniqueId nccl_id;

//     int my_rank_in_comm;
//     MPI_Comm_rank(comm, &my_rank_in_comm);
    
//     if (my_rank_in_comm == root_rank) {
//         ncclGetUniqueId(&nccl_id);
//     }

//     MPI_Bcast(&nccl_id, sizeof(ncclUniqueId), MPI_BYTE, root_rank, comm);
    
//     return nccl_id;
// }

std::shared_ptr<ProcessGroupNCCL> DeviceMesh::get_process_group(int64_t mesh_dim) {
    if (mesh_dim < 0 || mesh_dim >= ndim()) {
        throw std::runtime_error("DeviceMesh: invalid mesh_dim");
    }
    return process_groups_[mesh_dim];
}


int64_t DeviceMesh::size() const {
    return total_devices_;
}


void DeviceMesh::describe() const {
    std::ostringstream oss;
    oss << "[DeviceMesh] Rank " << global_rank_ << "/" << total_devices_ << " | Shape: [";
    for (size_t i = 0; i < mesh_shape_.size(); ++i) {
        oss << mesh_shape_[i];
        if (i < mesh_shape_.size() - 1) oss << ", ";
    }   
    oss << "] | Coordinate: [";
    for (size_t i = 0; i < my_coordinate_.size(); ++i) {
        oss << my_coordinate_[i];
        if (i < my_coordinate_.size() - 1) oss << ", ";
    }
    oss << "]";
    
    std::cout << oss.str() << std::endl;
}



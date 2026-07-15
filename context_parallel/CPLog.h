#pragma once
#include <cstdlib>

// Global-rank gate for the CP stack's one-time stderr notices (RoPE/ring/overlap
// banners) so they print once (on MPI global rank 0) instead of once per rank.
// Returns true on rank 0 per the launcher env, or true when the launcher rank is
// unknown, so a notice is never silently lost under an unrecognized launcher.
// std::cerr / fprintf(stderr) for genuine errors on other ranks stay untouched.
// Namespace 'cplog' (NOT 'cp' — the CP stack already has OwnTensor::cp, which
// would make every cp:: reference ambiguous).
namespace cplog {
inline bool log_rank() {
    const char* r = std::getenv("OMPI_COMM_WORLD_RANK");   // OpenMPI
    if (!r) r = std::getenv("PMI_RANK");                    // MPICH / Hydra
    return !r || (r[0] == '0' && r[1] == '\0');
}
}  // namespace cplog

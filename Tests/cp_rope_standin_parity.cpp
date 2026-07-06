// =============================================================================
// cp_rope_standin_parity.cpp — Phase 2 DE-FUSED RoPE CP oracle.
//
// Validates the CP ring (forward + backward) under RoPE WITHOUT the team's
// fused kernel, by applying QK-norm + RoPE as standalone ops on the FULL
// sequence BEFORE sharding, then running the existing pure-SDPA CP ring.
// Compares CP (world_size=N) against a single-GPU reference on identical
// rotated inputs — forward output and dQ/dK/dV/dq_gamma/dk_gamma.
//
// De-fused building blocks (all pre-existing):
//   QK-norm : autograd::rms_norm(x, gamma, head_dim, eps)
//   RoPE    : autograd::rope(x, cos_sin_cache, interleaved=false, offset=0)
//   cache   : autograd::build_rope_cache(T, head_dim, theta, device)
//   attn    : ContextParallel::forward_cp  (pure-SDPA sm89 kernel; use_rope_=false)
//
// SCOPE: this exercises the CP ring orchestration (shard/merge/sub-chunk/dKV)
// and that gamma grads flow through the de-fused rms_norm — it does NOT
// exercise compute_deltas or the manual per-step gamma accumulation (those are
// the FUSED path, covered by the Phase-1a/1b CPU unit tests and Phase-3 parity).
//
// Build: make cp-rope-standin           (see Makefile target)
// Run:   mpirun -np 2 ./bin/cp_rope_standin_parity_exec
// =============================================================================

#include <mpi.h>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>

#include "TensorLib.h"
#include "autograd/AutogradOps.h"
#include "autograd/operations/RoPEOps.h"
#include "autograd/operations/NormalizationOps.h"
#include "context_parallel/ContextParallel.h"

using namespace OwnTensor;

static float max_abs_diff(const Tensor& a, const Tensor& b) {
  const float* pa = a.data<float>(); const float* pb = b.data<float>();
  int64_t n = std::min<int64_t>(a.numel(), b.numel());
  float m = 0.0f;
  for (int64_t i = 0; i < n; ++i) m = std::max(m, std::abs(pa[i] - pb[i]));
  return m;
}
static double cosine(const Tensor& a, const Tensor& b) {
  const float* pa = a.data<float>(); const float* pb = b.data<float>();
  int64_t n = std::min<int64_t>(a.numel(), b.numel());
  double dot = 0, na = 0, nb = 0;
  for (int64_t i = 0; i < n; ++i) { dot += (double)pa[i]*pb[i]; na += (double)pa[i]*pa[i]; nb += (double)pb[i]*pb[i]; }
  if (na == 0 || nb == 0) return 1.0;
  return dot / (std::sqrt(na) * std::sqrt(nb));
}

// Single-GPU reference attention (matches the CP kernel's causal SDPA).
static Tensor ref_sdpa(Tensor& q, Tensor& k, Tensor& v, float scale) {
  Shape s({{1}});
  TensorOptions o = TensorOptions().with_dtype(q.dtype()).with_device(q.device());
  Tensor st = Tensor::full(s, o, scale);
  Tensor qs = autograd::mul(q, st);
  Tensor kt = autograd::transpose(k, -2, -1);
  Tensor scores = autograd::matmul(qs, kt);
  float ninf = -std::numeric_limits<float>::infinity();
  Tensor masked = autograd::tril(scores, 0, ninf);
  Tensor probs = autograd::softmax(masked);
  return autograd::matmul(probs, v);
}

// De-fused QK-norm + RoPE on the FULL sequence (offset 0 over positions 0..T-1).
static Tensor qk_rope(const Tensor& x, const Tensor& gamma, const Tensor& cache,
                      int64_t head_dim, float eps) {
  Tensor n = autograd::rms_norm(x, gamma, static_cast<int>(head_dim), eps);
  return autograd::rope(n, cache, /*interleaved=*/false, /*position_offset=*/0);
}

static int g_fail = 0;
static void gate(bool ok, const char* tag, double cos_v, float maxd) {
  std::cout << (ok ? "  PASS " : "  FAIL ") << std::left << std::setw(22) << tag
            << " cos=" << std::fixed << std::setprecision(7) << cos_v
            << " maxdiff=" << std::scientific << maxd << std::endl;
  if (!ok) ++g_fail;
}

static void run_lb_mode(const DeviceMesh& mesh, std::shared_ptr<ProcessGroupNCCL> pg,
                        int rank, bool load_balance, DeviceIndex device,
                        int64_t B, int64_t H, int64_t T, int64_t D, float scale,
                        const Tensor& cache, float eps) {
  if (rank == 0)
    std::cout << "\n=== Stand-in parity  (load_balance=" << (load_balance ? "ON/HeadTail" : "OFF/contiguous")
              << ") ===" << std::endl;

  Shape qkv({{B, H, T, D}}), gsh({{D}});
  TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32).with_device(device).with_req_grad(true);

  // Identical inputs + gammas on every rank (same seeds).
  auto mk = [&](int seed){ return Tensor::randn<float>(qkv, opts, seed, 0.5f); };
  Tensor qg = Tensor::randn<float>(gsh, opts, 11, 0.1f);  // q_gamma
  Tensor kg = Tensor::randn<float>(gsh, opts, 22, 0.1f);  // k_gamma

  // ---------------- FORWARD ----------------
  Tensor ref_out;
  if (rank == 0) {
    Tensor q = mk(100), k = mk(200), v = mk(300);
    Tensor qr = qk_rope(q, qg.clone(), cache, D, eps);
    Tensor kr = qk_rope(k, kg.clone(), cache, D, eps);
    ref_out = ref_sdpa(qr, kr, v, scale);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  Tensor q = mk(100), k = mk(200), v = mk(300);
  Tensor qr = qk_rope(q, qg, cache, D, eps);
  Tensor kr = qk_rope(k, kg, cache, D, eps);
  ContextParallel cp(mesh, pg, scale, /*is_causal=*/true, RotatorType::AlltoAll, load_balance);
  Tensor cp_out = cp.forward_cp(qr, kr, v);   // unshard=true => full [B,H,T,D] on all ranks

  if (rank == 0) {
    Tensor a = ref_out.to_cpu(), b = cp_out.to_cpu();
    double c = cosine(a, b); float m = max_abs_diff(a, b);
    gate(c > 0.99999 && m < 1e-3f, "forward", c, m);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // ---------------- BACKWARD ----------------
  // CP path: grads w.r.t. q,k,v and q_gamma,k_gamma on all ranks.
  Tensor ones_cp = Tensor::full(cp_out.shape(),
      TensorOptions().with_dtype(Dtype::Float32).with_device(device), 1.0f);
  cp_out.backward(&ones_cp);

  if (rank == 0) {
    Tensor q2 = mk(100), k2 = mk(200), v2 = mk(300);
    Tensor qg2 = qg.clone(); qg2.set_requires_grad(true);
    Tensor kg2 = kg.clone(); kg2.set_requires_grad(true);
    Tensor qr2 = qk_rope(q2, qg2, cache, D, eps);
    Tensor kr2 = qk_rope(k2, kg2, cache, D, eps);
    Tensor ro = ref_sdpa(qr2, kr2, v2, scale);
    Tensor ones = Tensor::full(ro.shape(),
        TensorOptions().with_dtype(Dtype::Float32).with_device(device), 1.0f);
    ro.backward(&ones);

    auto cmp = [&](Tensor& cp_src, Tensor& ref_src, const char* tag){
      if (!cp_src.has_grad() || !ref_src.has_grad()) {
        std::cout << "  FAIL " << tag << " (missing grad)\n"; ++g_fail; return;
      }
      Tensor a = ref_src.grad_view().to_cpu(), b = cp_src.grad_view().to_cpu();
      double c = cosine(a, b); float m = max_abs_diff(a, b);
      gate(c > 0.9999 && m < 1e-2f, tag, c, m);
    };
    cmp(q,  q2,  "dQ");
    cmp(k,  k2,  "dK");
    cmp(v,  v2,  "dV");
    cmp(qg, qg2, "dq_gamma");
    cmp(kg, kg2, "dk_gamma");
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  // Tolerate oversubscription (e.g. ws=4 on 2 GPUs) so the ring's deep-pipeline
  // path (world_size>=3, per-step slot reuse) can be validated on fewer GPUs.
  int _ndev = 0; cudaGetDeviceCount(&_ndev);
  cudaSetDevice(_ndev > 0 ? (rank % _ndev) : 0);

  std::vector<int> ranks(world_size);
  for (int i = 0; i < world_size; ++i) ranks[i] = i;
  DeviceMesh mesh({world_size}, ranks);
  auto pg = mesh.get_process_group(0);
  DeviceIndex device(Device::CUDA, rank);

  const int64_t B = 2, H = 4, D = 64;
  const int64_t T = 64;                 // divisible by 2*world_size for HeadTail
  const float scale = 1.0f / std::sqrt((float)D);
  const float eps = 1e-6f;
  Tensor cache = autograd::build_rope_cache(T, D, 10000.0f, device);

  if (rank == 0)
    std::cout << "=== CP de-fused RoPE stand-in parity (ws=" << world_size
              << ", B=" << B << " H=" << H << " T=" << T << " D=" << D << ") ===" << std::endl;

  run_lb_mode(mesh, pg, rank, /*load_balance=*/false, device, B, H, T, D, scale, cache, eps);
  run_lb_mode(mesh, pg, rank, /*load_balance=*/true,  device, B, H, T, D, scale, cache, eps);

  int total_fail = g_fail;
  MPI_Allreduce(MPI_IN_PLACE, &total_fail, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    if (total_fail == 0) std::cout << "\nALL cp_rope_standin parity gates PASSED.\n";
    else std::cout << "\n" << total_fail << " parity gate(s) FAILED.\n";
  }
  MPI_Finalize();
  return total_fail == 0 ? 0 : 1;
}

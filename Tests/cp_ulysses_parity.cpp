// =============================================================================
// cp_ulysses_parity.cpp — DeepSpeed-style Ulysses sequence-parallel attention
// parity oracle.
//
// Validates ContextParallel::enable_ulysses() (the all-to-all path: combine ->
// full causal SDPA -> partition) against a single-GPU reference on identical
// inputs — forward output and dQ/dK/dV.
//
// Run at BOTH ws=2 AND ws=4 (the all-to-all P-axis permute has only one possible
// swap at P=2, so a transposed-axis bug can hide there and only surface at P>=4):
//   make run-cp-ulysses NP=2
//   make run-cp-ulysses NP=4
// Config (B=2, H=8, T=64, D=64) satisfies H%4==0 and T%4==0 for the ws=4 run.
//
// Both unshard=true (full [B,H,T,D] return; exercises the backward narrow branch)
// and unshard=false (local [B,H,Tl,D] return; exercises the no-narrow branch with
// a slice-masked reference) are exercised.
// =============================================================================

#include <mpi.h>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>
#include <vector>

#include "TensorLib.h"
#include "autograd/AutogradOps.h"
#include "context_parallel/ContextParallel.h"

using namespace OwnTensor;

static int g_fail = 0;

static float max_abs_diff(const Tensor &a, const Tensor &b) {
  const float *pa = a.data<float>();
  const float *pb = b.data<float>();
  int64_t n = std::min<int64_t>(a.numel(), b.numel());
  float m = 0.0f;
  for (int64_t i = 0; i < n; ++i) m = std::max(m, std::abs(pa[i] - pb[i]));
  return m;
}
static double cosine(const Tensor &a, const Tensor &b) {
  const float *pa = a.data<float>();
  const float *pb = b.data<float>();
  int64_t n = std::min<int64_t>(a.numel(), b.numel());
  double dot = 0, na = 0, nb = 0;
  for (int64_t i = 0; i < n; ++i) {
    dot += (double)pa[i] * pb[i];
    na += (double)pa[i] * pa[i];
    nb += (double)pb[i] * pb[i];
  }
  if (na == 0 || nb == 0) return 1.0;
  return dot / (std::sqrt(na) * std::sqrt(nb));
}

// Single-GPU reference causal SDPA (matches the CP kernel's masked softmax).
static Tensor ref_sdpa(Tensor &q, Tensor &k, Tensor &v, float scale) {
  Shape s({{1}});
  TensorOptions o =
      TensorOptions().with_dtype(q.dtype()).with_device(q.device());
  Tensor st = Tensor::full(s, o, scale);
  Tensor qs = autograd::mul(q, st);
  Tensor kt = autograd::transpose(k, -2, -1);
  Tensor scores = autograd::matmul(qs, kt);
  float ninf = -std::numeric_limits<float>::infinity();
  Tensor masked = autograd::tril(scores, 0, ninf);
  Tensor probs = autograd::softmax(masked);
  return autograd::matmul(probs, v);
}

static void gate(bool ok, const char *tag, double cos_v, float maxd) {
  std::cout << (ok ? "  PASS " : "  FAIL ") << std::left << std::setw(20) << tag
            << " cos=" << std::fixed << std::setprecision(7) << cos_v
            << " maxdiff=" << std::scientific << maxd << std::endl;
  if (!ok) ++g_fail;
}

static void run_mode(const DeviceMesh &mesh,
                     std::shared_ptr<ProcessGroupNCCL> pg, int rank,
                     int world_size, bool unshard, DeviceIndex device, int64_t B,
                     int64_t H, int64_t T, int64_t D, float scale) {
  if (rank == 0)
    std::cout << "\n=== Ulysses parity  (unshard=" << (unshard ? "true" : "false")
              << ") ===" << std::endl;

  const int64_t Tl = T / world_size;
  Shape qkv({{B, H, T, D}});
  TensorOptions opts = TensorOptions()
                           .with_dtype(Dtype::Float32)
                           .with_device(device)
                           .with_req_grad(true);
  auto mk = [&](int seed) { return Tensor::randn<float>(qkv, opts, seed, 0.5f); };

  // ---------------- FORWARD (CP) ----------------
  Tensor q = mk(100), k = mk(200), v = mk(300);
  ContextParallel cp(mesh, pg, scale, /*is_causal=*/true, RotatorType::AlltoAll,
                     /*load_balance=*/false);
  cp.enable_ulysses();
  Tensor cp_out = cp.forward_cp(q, k, v, unshard);

  // ---------------- FORWARD (ref, rank 0) ----------------
  if (rank == 0) {
    Tensor qr = mk(100), kr = mk(200), vr = mk(300);
    Tensor ro = ref_sdpa(qr, kr, vr, scale);
    Tensor a = ro.to_cpu();
    if (!unshard) a = a.narrow(2, 0, Tl).contiguous(); // compare rank 0's slice
    Tensor b = cp_out.to_cpu();
    double c = cosine(a, b);
    float m = max_abs_diff(a, b);
    gate(c > 0.99999 && m < 1e-3f, "forward", c, m);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // ---------------- BACKWARD (CP) ----------------
  Tensor ones_cp = Tensor::full(
      cp_out.shape(),
      TensorOptions().with_dtype(Dtype::Float32).with_device(device), 1.0f);
  cp_out.backward(&ones_cp);

  // ---------------- BACKWARD (ref, rank 0) ----------------
  // NOTE: the CP backward's combine all-to-all is COLLECTIVE — every rank's
  // local-slice upstream grad is gathered, so the total loss is the sum over
  // ALL ranks' slices == the full-sequence output sum, identical for both
  // unshard modes. Hence the reference uses the full ones-grad in BOTH cases;
  // each rank's returned dQ/dK/dV is the full-loss gradient.
  if (rank == 0) {
    Tensor q2 = mk(100), k2 = mk(200), v2 = mk(300);
    Tensor ro = ref_sdpa(q2, k2, v2, scale);
    Tensor ref_grad = Tensor::full(
        ro.shape(),
        TensorOptions().with_dtype(Dtype::Float32).with_device(device), 1.0f);
    ro.backward(&ref_grad);

    auto cmp = [&](Tensor &cp_src, Tensor &ref_src, const char *tag) {
      if (!cp_src.has_grad() || !ref_src.has_grad()) {
        std::cout << "  FAIL " << tag << " (missing grad)\n";
        ++g_fail;
        return;
      }
      Tensor a = ref_src.grad_view().to_cpu();
      Tensor b = cp_src.grad_view().to_cpu();
      double c = cosine(a, b);
      float m = max_abs_diff(a, b);
      gate(c > 0.9999 && m < 1e-2f, tag, c, m);
    };
    cmp(q, q2, "dQ");
    cmp(k, k2, "dK");
    cmp(v, v2, "dV");
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

// GQA/MQA parity: q has nq heads, k/v have nkv < nq heads (group g = nq/nkv).
// Reference expands KV to nq heads (repeat_interleave) and runs plain MHA; the
// reference dK/dV are group-reduced back to nkv heads (the same group-sum the CP
// GQA path applies) before comparison.
static void run_gqa_mode(const DeviceMesh &mesh,
                         std::shared_ptr<ProcessGroupNCCL> pg, int rank,
                         int world_size, int64_t nkv, DeviceIndex device,
                         int64_t B, int64_t nq, int64_t T, int64_t D,
                         float scale) {
  const int g = static_cast<int>(nq / nkv);
  if (rank == 0)
    std::cout << "\n=== Ulysses GQA parity  (nq=" << nq << " nkv=" << nkv
              << " g=" << g << (nkv < world_size ? ", MQA/replicate" : "")
              << ") ===" << std::endl;

  TensorOptions opts = TensorOptions()
                           .with_dtype(Dtype::Float32)
                           .with_device(device)
                           .with_req_grad(true);
  auto mkq = [&](int seed) {
    return Tensor::randn<float>(Shape({{B, nq, T, D}}), opts, seed, 0.5f);
  };
  auto mkkv = [&](int seed) {
    return Tensor::randn<float>(Shape({{B, nkv, T, D}}), opts, seed, 0.5f);
  };

  // ---- CP (Ulysses GQA) ----
  Tensor q = mkq(100), k = mkkv(200), v = mkkv(300);
  ContextParallel cp(mesh, pg, scale, /*is_causal=*/true, RotatorType::AlltoAll,
                     /*load_balance=*/false);
  cp.enable_ulysses();
  Tensor cp_out = cp.forward_cp(q, k, v, /*unshard=*/true); // [B,nq,T,D] full
  Tensor ones_cp = Tensor::full(
      cp_out.shape(),
      TensorOptions().with_dtype(Dtype::Float32).with_device(device), 1.0f);
  cp_out.backward(&ones_cp);

  // ---- reference (rank 0): expand KV to nq, plain MHA, reduce KV grads ----
  if (rank == 0) {
    // KV built WITHOUT grad (same seeds/values as CP's k,v), expanded via the raw
    // helper, THEN marked as leaves so grad flows cleanly to the expanded heads
    // (expanding a requires_grad tensor would yield a non-leaf and a bogus grad).
    TensorOptions optsng = TensorOptions()
                               .with_dtype(Dtype::Float32)
                               .with_device(device)
                               .with_req_grad(false);
    Tensor q2 = mkq(100);
    Tensor k2 = Tensor::randn<float>(Shape({{B, nkv, T, D}}), optsng, 200, 0.5f);
    Tensor v2 = Tensor::randn<float>(Shape({{B, nkv, T, D}}), optsng, 300, 0.5f);
    Tensor k2e = OwnTensor::cp::head_repeat_interleave(k2, g); // [B,nq,T,D]
    Tensor v2e = OwnTensor::cp::head_repeat_interleave(v2, g);
    k2e.set_requires_grad(true);
    v2e.set_requires_grad(true);
    Tensor ro = ref_sdpa(q2, k2e, v2e, scale);

    // forward compare
    {
      Tensor a = ro.to_cpu(), b = cp_out.to_cpu();
      double c = cosine(a, b);
      float m = max_abs_diff(a, b);
      gate(c > 0.99999 && m < 1e-3f, "gqa forward", c, m);
    }

    Tensor ones = Tensor::full(
        ro.shape(),
        TensorOptions().with_dtype(Dtype::Float32).with_device(device), 1.0f);
    ro.backward(&ones);

    // dQ direct; dK/dV group-reduced from expanded heads back to nkv.
    auto cmp = [&](const Tensor &ref_grad, Tensor &cp_src, const char *tag) {
      if (!cp_src.has_grad()) {
        std::cout << "  FAIL " << tag << " (missing grad)\n";
        ++g_fail;
        return;
      }
      Tensor a = ref_grad.to_cpu();
      Tensor b = cp_src.grad_view().to_cpu();
      double c = cosine(a, b);
      float m = max_abs_diff(a, b);
      gate(c > 0.9999 && m < 1e-2f, tag, c, m);
    };
    Tensor dk_ref = OwnTensor::cp::head_group_reduce(k2e.grad_view(), nkv, g);
    Tensor dv_ref = OwnTensor::cp::head_group_reduce(v2e.grad_view(), nkv, g);
    cmp(q2.grad_view(), q, "gqa dQ");
    cmp(dk_ref, k, "gqa dK");
    cmp(dv_ref, v, "gqa dV");
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  // Map ranks to devices; tolerate oversubscription (e.g. ws=4 on 2 GPUs) so the
  // mandatory ws=4 axis-coverage run can execute on fewer physical GPUs.
  int ndev = 0;
  cudaGetDeviceCount(&ndev);
  cudaSetDevice(ndev > 0 ? (rank % ndev) : 0);

  std::vector<int> ranks(world_size);
  for (int i = 0; i < world_size; ++i) ranks[i] = i;
  DeviceMesh mesh({world_size}, ranks);
  auto pg = mesh.get_process_group(0);
  DeviceIndex device(Device::CUDA, rank);

  const int64_t B = 2, H = 8, D = 64;
  const int64_t T = 64; // H%ws==0 and T%ws==0 for ws in {2,4}
  const float scale = 1.0f / std::sqrt((float)D);

  if (H % world_size != 0 || T % world_size != 0) {
    if (rank == 0)
      std::cout << "SKIP: H=" << H << " T=" << T
                << " not divisible by world_size=" << world_size << std::endl;
    MPI_Finalize();
    return 0;
  }

  if (rank == 0)
    std::cout << "=== CP Ulysses parity (ws=" << world_size << ", B=" << B
              << " H=" << H << " T=" << T << " D=" << D << ") ===" << std::endl;

  run_mode(mesh, pg, rank, world_size, /*unshard=*/true, device, B, H, T, D, scale);
  run_mode(mesh, pg, rank, world_size, /*unshard=*/false, device, B, H, T, D, scale);

  // GQA/MQA cases (nq = H = 8). Each runs only when the DeepSpeed divisibility
  // rule holds for this world_size: (nkv % P == 0) or (P % nkv == 0).
  //   nkv=4 : nkv>=P at ws in {2,4} (no replication, kv_local>=1)
  //   nkv=2 : nkv>=P at ws=2; nkv<P (rep=2) at ws=4
  //   nkv=1 : MQA, nkv<P (rep=P) at ws in {2,4}
  for (int64_t nkv : {(int64_t)4, (int64_t)2, (int64_t)1}) {
    if (H % nkv != 0) continue;
    if (!((nkv % world_size == 0) || (world_size % nkv == 0))) {
      if (rank == 0)
        std::cout << "\n[skip GQA nkv=" << nkv << "] divisibility not met for ws="
                  << world_size << std::endl;
      continue;
    }
    run_gqa_mode(mesh, pg, rank, world_size, nkv, device, B, /*nq=*/H, T, D, scale);
  }

  int total_fail = g_fail;
  MPI_Allreduce(MPI_IN_PLACE, &total_fail, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) {
    if (total_fail == 0)
      std::cout << "\nALL cp_ulysses parity gates PASSED.\n";
    else
      std::cout << "\n" << total_fail << " parity gate(s) FAILED.\n";
  }
  MPI_Finalize();
  return total_fail == 0 ? 0 : 1;
}

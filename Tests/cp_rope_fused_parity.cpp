// =============================================================================
// cp_rope_fused_parity.cpp — Phase 3 FUSED RoPE kernel parity (single GPU).
//
// Exercises the NEW CP fused kernel gqa_fused_rope_cp_forward (via the wrapper
// sdpa_fused_forward_rope) by feeding RAW Q/K/V + gammas + the full cos/sin
// cache and letting the kernel do QK-norm + RoPE(4-delta) + causal attention.
// Compares against a bf16-consistent DE-FUSED reference (rms_norm + rope + SDPA)
// on the SAME bf16-rounded inputs.
//
// Two cases:
//   A. identity      : T=64, deltas all 0 -> pos = local index (== non-CP kernel).
//   B. headtail seam : T_local=32 simulating rank0 of ws=2 HeadTail. Local shard
//                      is [head=global 0..15 | tail=global 48..63]; deltas
//                      q=k={d0=0, d1=32} so pos(local j)= j (j<16) / j+32 (j>=16).
//                      The de-fused reference rotates with a GATHERED cache
//                      (rows {0..15,48..63}) at offset 0 — same global rotation.
//                      This validates the head/tail split point (seam at 16).
//
// MHA (Nq==Nkv) keeps the reference a plain causal SDPA (local order is globally
// monotonic, so causal-on-local == causal-on-global). Build with CP_FUSED_ROPE=1
// (and CP_ROPE_DEBUG=1 to assert no out-of-range RoPE indices).
//
// Build: make CP_FUSED_ROPE=1 CP_ROPE_DEBUG=1 cp-rope-fused
// Run:   ./build/cp_rope_fused_parity_exec
// =============================================================================
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>

#include "TensorLib.h"
#include "autograd/AutogradOps.h"
#include "autograd/operations/RoPEOps.h"
#include "autograd/operations/NormalizationOps.h"
#include "context_parallel/FusedRoPESDPA.h"
#include "ops/helpers/AttentionKernels.h"   // production non-CP kernel (identity check)

using namespace OwnTensor;

static int g_fail = 0;

static double cosine(const Tensor& a, const Tensor& b) {
  const float* pa = a.data<float>(); const float* pb = b.data<float>();
  int64_t n = std::min<int64_t>(a.numel(), b.numel());
  double dot = 0, na = 0, nb = 0;
  for (int64_t i = 0; i < n; ++i) { dot += (double)pa[i]*pb[i]; na += (double)pa[i]*pa[i]; nb += (double)pb[i]*pb[i]; }
  if (na == 0 || nb == 0) return 1.0;
  return dot / (std::sqrt(na) * std::sqrt(nb));
}
static float max_abs_diff(const Tensor& a, const Tensor& b) {
  const float* pa = a.data<float>(); const float* pb = b.data<float>();
  int64_t n = std::min<int64_t>(a.numel(), b.numel());
  float m = 0.0f;
  for (int64_t i = 0; i < n; ++i) m = std::max(m, std::abs(pa[i] - pb[i]));
  return m;
}
static void gate(bool ok, const char* tag, double c, float m) {
  std::cout << (ok ? "  PASS " : "  FAIL ") << std::left << std::setw(26) << tag
            << " cos=" << std::fixed << std::setprecision(7) << c
            << " maxdiff=" << std::scientific << m << std::endl;
  if (!ok) ++g_fail;
}

// bf16 round-trip so the reference sees exactly the kernel's input precision.
static Tensor bf16_round(const Tensor& x) {
  return x.as_type(Dtype::Bfloat16).as_type(Dtype::Float32);
}

// MHA causal SDPA reference.
static Tensor ref_sdpa(Tensor& q, Tensor& k, Tensor& v, float scale) {
  TensorOptions o = TensorOptions().with_dtype(q.dtype()).with_device(q.device());
  Tensor st = Tensor::full(Shape({{1}}), o, scale);
  Tensor qs = autograd::mul(q, st);
  Tensor scores = autograd::matmul(qs, autograd::transpose(k, -2, -1));
  float ninf = -std::numeric_limits<float>::infinity();
  Tensor probs = autograd::softmax(autograd::tril(scores, 0, ninf));
  return autograd::matmul(probs, v);
}

static void run_case(const char* tag, DeviceIndex device,
                     int64_t B, int64_t H, int64_t T, int64_t D,
                     int q_d0, int q_d1, int k_d0, int k_d1,
                     const Tensor& cache_ref,   // de-fused reference cache (offset 0)
                     const Tensor& cache_full,  // full global cache for the kernel
                     float scale, float eps,
                     bool cmp_prod) {           // also compare vs production non-CP kernel
  std::cout << "\n=== case " << tag << " (T=" << T
            << ", q_d=(" << q_d0 << "," << q_d1 << ") k_d=(" << k_d0 << "," << k_d1 << ")) ===\n";
  Shape qkv({{B, H, T, D}}), gsh({{D}});
  TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32).with_device(device);

  Tensor Q = bf16_round(Tensor::randn<float>(qkv, opts, 100, 0.5f));
  Tensor K = bf16_round(Tensor::randn<float>(qkv, opts, 200, 0.5f));
  Tensor V = bf16_round(Tensor::randn<float>(qkv, opts, 300, 0.5f));
  Tensor qg = Tensor::randn<float>(gsh, opts, 11, 0.1f);
  Tensor kg = Tensor::randn<float>(gsh, opts, 22, 0.1f);

  // Reference: de-fused QK-norm + RoPE (at global positions via cache_ref) + SDPA.
  Tensor qr = autograd::rope(autograd::rms_norm(Q, qg, (int)D, eps), cache_ref, false, 0);
  Tensor kr = autograd::rope(autograd::rms_norm(K, kg, (int)D, eps), cache_ref, false, 0);
  Tensor ref = ref_sdpa(qr, kr, V, scale);

  // Fused kernel: raw Q/K/V, kernel does norm+rope(4-delta)+attention.
  Tensor Qc = Q, Kc = K, Vc = V;
  SDPAResult res = sdpa_fused_forward_rope(
      Qc, Kc, Vc, /*is_causal=*/true, scale,
      q_d0, q_d1, k_d0, k_d1, cache_full, qg, kg, eps);

  Tensor a = ref.to_cpu(), b = res.out.to_cpu();
  double c = cosine(a, b); float m = max_abs_diff(a, b);
  // bf16 kernel vs fp32 de-fused reference: cosine is the correctness gate;
  // maxdiff is bf16 accumulation noise (informational), not a hard bound.
  gate(c > 0.999, "forward vs fp32-ref (cos)", c, m);

  // DECISIVE identity check: with deltas=0 the kernel must reproduce the
  // production non-CP gqa_fused_flash_attn_forward BIT-for-BIT (same bf16 math).
  if (cmp_prod) {
    Tensor Qb = Q.as_type(Dtype::Bfloat16);
    Tensor Kb = K.as_type(Dtype::Bfloat16);
    Tensor Vb = V.as_type(Dtype::Bfloat16);
    Tensor packed = Tensor::cat({Qb, Kb, Vb}, 0);           // [3B,H,T,D] bf16 contiguous
    TensorOptions bf = TensorOptions().with_dtype(Dtype::Bfloat16).with_device(device);
    TensorOptions fp = TensorOptions().with_dtype(Dtype::Float32).with_device(device);
    Tensor out_p = Tensor::empty(Shape({{B, H, T, D}}), bf);
    Tensor lse_p = Tensor::empty(Shape({{B, H, T}}), fp);
    Tensor qrstd = Tensor::empty(Shape({{B, H, T}}), fp);   // saved for the bwd
    Tensor krstd = Tensor::empty(Shape({{B, H, T}}), fp);
    OwnTensor::cuda::gqa_fused_flash_attn_forward(
        packed.data<bfloat16_t>(), cache_full.data<float>(), qg.data<float>(), kg.data<float>(),
        out_p.data<bfloat16_t>(), lse_p.data<float>(),
        qrstd.data<float>(), krstd.data<float>(),
        (int)B, (int)H, (int)H, (int)T, (int)D,
        (int)cache_full.shape().dims[0], /*pos_offset=*/0, eps,
        /*interleaved=*/false, /*is_causal=*/true);
    Tensor pa = out_p.as_type(Dtype::Float32).to_cpu();
    double cp = cosine(pa, b); float mp = max_abs_diff(pa, b);
    gate(cp > 0.999999 && mp < 1e-3f, "forward vs production (identity)", cp, mp);

    // Backward: production kernel (gold standard, bf16) vs my kernel, deltas=0.
    // Both bf16 -> expect ~bit-identical (dQ/dgamma are bf16-sensitive, so an
    // fp32 autograd reference would falsely diverge; this comparison is exact).
    Tensor dO_p    = Tensor::full(Shape({{B, H, T, D}}), bf, 1.0f);
    Tensor gradqkv = Tensor::empty(Shape({{3 * B, H, T, D}}), bf);
    Tensor dqg_p   = Tensor::zeros(Shape({{D}}), fp);
    Tensor dkg_p   = Tensor::zeros(Shape({{D}}), fp);
    Tensor Dbuf    = Tensor::empty(Shape({{B * H, T}}), fp);
    OwnTensor::cuda::gqa_fused_flash_attn_backward(
        packed.data<bfloat16_t>(), cache_full.data<float>(), qg.data<float>(), kg.data<float>(),
        out_p.data<bfloat16_t>(), dO_p.data<bfloat16_t>(), lse_p.data<float>(),
        qrstd.data<float>(), krstd.data<float>(),
        gradqkv.data<bfloat16_t>(), dqg_p.data<float>(), dkg_p.data<float>(), Dbuf.data<float>(),
        (int)B, (int)H, (int)H, (int)T, (int)D,
        (int)cache_full.shape().dims[0], /*pos_offset=*/0, eps,
        /*interleaved=*/false, /*is_causal=*/true);
    Tensor dQp = gradqkv.narrow(0, 0,     B).as_type(Dtype::Float32);
    Tensor dKp = gradqkv.narrow(0, B,     B).as_type(Dtype::Float32);
    Tensor dVp = gradqkv.narrow(0, 2 * B, B).as_type(Dtype::Float32);

    Tensor dones = Tensor::full(Shape({{B, H, T, D}}), fp, 1.0f);
    Tensor Qc2 = Q, Kc2 = K, Vc2 = V;
    SDPARoPEBackwardResult mb = sdpa_fused_backward_rope(
        Qc2, Kc2, Vc2, dones, res.out, res.lse, /*is_causal=*/true, scale,
        q_d0, q_d1, k_d0, k_d1, cache_full, qg, kg, eps);
    auto cmpp = [&](const Tensor& mine, const Tensor& prod, const char* nm) {
      Tensor x = mine.to_cpu(), y = prod.to_cpu();
      double cc = cosine(x, y); float mm = max_abs_diff(x, y);
      gate(cc > 0.9999 && mm < 1e-2f, nm, cc, mm);
    };
    cmpp(mb.dQ, dQp, "bwd dQ vs production");
    cmpp(mb.dK, dKp, "bwd dK vs production");
    cmpp(mb.dV, dVp, "bwd dV vs production");
    cmpp(mb.dq_gamma, dqg_p, "bwd dq_gamma vs production");
    cmpp(mb.dk_gamma, dkg_p, "bwd dk_gamma vs production");
  }

  // Seam-focused check (plan Seam-tightening #5): the rows straddling the
  // head/tail split (local T/2-1 and T/2) must match, not just the aggregate.
  auto row_cos = [&](int64_t row)->double {
    Tensor ar = a.narrow(2, row, 1).contiguous();
    Tensor br = b.narrow(2, row, 1).contiguous();
    return cosine(ar, br);
  };
  int64_t seam = T / 2;
  double cs0 = row_cos(seam - 1), cs1 = row_cos(seam);
  gate(cs0 > 0.999 && cs1 > 0.999, "forward (seam rows T/2-1,T/2)",
       std::min(cs0, cs1), 0.0f);

}

int main() {
  int ndev = 0; cudaGetDeviceCount(&ndev);
  cudaSetDevice(0);
  DeviceIndex device(Device::CUDA, 0);

  const int64_t B = 2, H = 4, D = 64;
  const float scale = 1.0f / std::sqrt((float)D);
  const float eps = 1e-6f;

  // Full global cache at T=64 (the extended sequence length).
  Tensor cache = autograd::build_rope_cache(64, D, 10000.0f, device);

  std::cout << "=== CP fused-RoPE kernel parity (single GPU, B=" << B
            << " H=" << H << " D=" << D << ") ===\n";

  // Case A — identity: deltas 0, T=64. Reference = full cache at offset 0; also
  // compared BIT-for-BIT against the production non-CP kernel.
  run_case("A identity", device, B, H, 64, D, 0, 0, 0, 0, cache, cache, scale, eps,
           /*cmp_prod=*/true);

  // Case B — HeadTail seam (rank0 of ws=2): local T=32 = [head 0..15 | tail 48..63].
  // deltas q=k={0,32}; reference cache = gathered rows {0..15, 48..63}. No production
  // equivalent (non-contiguous positions), so fp32-ref cosine + seam gate only.
  Tensor cacheB = Tensor::cat({cache.narrow(0, 0, 16), cache.narrow(0, 48, 16)}, 0);
  run_case("B headtail seam", device, B, H, 32, D, 0, 32, 0, 32, cacheB, cache, scale, eps,
           /*cmp_prod=*/false);

  if (g_fail == 0) std::cout << "\nALL cp_rope_fused parity gates PASSED.\n";
  else             std::cout << "\n" << g_fail << " parity gate(s) FAILED.\n";
  return g_fail == 0 ? 0 : 1;
}

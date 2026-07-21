// =============================================================================
// cream_cache_gather_parity.cpp -- CREAM single-GPU no-CUDA-change proof.
//
// The CREAM trick: at CP_SIZE=1 the fused kernel computes pos = local_idx (all
// deltas 0) and indexes cos_sin_cache[pos]. So feeding a cache whose row j already
// holds the cos/sin for CREAM label L[j] makes the UNMODIFIED kernel rotate every
// token at its label. This test proves that end to end:
//
//   CASE 1  gather mechanics + kernel identity:
//     Gd = gather(full_cache, dim=0, broadcast(L))  must equal rows full[L[j]],
//     and sdpa_fused_forward_rope(cache=Gd, deltas=0) must match a de-fused
//     reference (rms_norm + rope@Gd + SDPA) -- i.e. the kernel indexes the
//     gathered cache correctly at pos=local_idx.
//
//   CASE 2  YaRN spot-check:
//     the full cache CREAM gathers from is genuinely YaRN-scaled (not plain
//     extrapolated): YARN_SCALE=1 vs >1 agree at pos 0 and DIFFER at high pos.
//
// Build: make CP_FUSED_ROPE=1 cream-cache-parity
// Run:   ./build/cream_cache_gather_parity_exec
// =============================================================================
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <limits>
#include <vector>

#include "TensorLib.h"
#include "autograd/AutogradOps.h"
#include "autograd/operations/RoPEOps.h"
#include "autograd/operations/NormalizationOps.h"
#include "context_parallel/FusedRoPESDPA.h"
#include "context_parallel/CreamPositions.h"
#include "ops/IndexingOps.h"

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
  std::cout << (ok ? "  PASS " : "  FAIL ") << std::left << std::setw(34) << tag
            << " cos=" << std::fixed << std::setprecision(7) << c
            << " maxdiff=" << std::scientific << m << std::endl;
  if (!ok) ++g_fail;
}
static Tensor bf16_round(const Tensor& x) {
  return x.as_type(Dtype::Bfloat16).as_type(Dtype::Float32);
}
static Tensor ref_sdpa(Tensor& q, Tensor& k, Tensor& v, float scale) {
  TensorOptions o = TensorOptions().with_dtype(q.dtype()).with_device(q.device());
  Tensor st = Tensor::full(Shape({{1}}), o, scale);
  Tensor qs = autograd::mul(q, st);
  Tensor scores = autograd::matmul(qs, autograd::transpose(k, -2, -1));
  float ninf = -std::numeric_limits<float>::infinity();
  Tensor probs = autograd::softmax(autograd::tril(scores, 0, ninf));
  return autograd::matmul(probs, v);
}

// Build a [T, D] int64 gather index broadcasting L across the D columns, on device.
static Tensor build_gather_index(const std::vector<int64_t>& L, int64_t D, DeviceIndex dev) {
  const int64_t T = static_cast<int64_t>(L.size());
  Tensor idx_cpu(Shape{{T, D}}, TensorOptions().with_dtype(Dtype::Int64));
  int64_t* p = static_cast<int64_t*>(idx_cpu.data());
  for (int64_t i = 0; i < T; ++i)
    for (int64_t j = 0; j < D; ++j) p[i * D + j] = L[static_cast<size_t>(i)];
  return idx_cpu.to(dev);
}

int main() {
  int ndev = 0; cudaGetDeviceCount(&ndev);
  if (ndev < 1) { std::cout << "no CUDA device; skipping\n"; return 0; }
  cudaSetDevice(0);
  DeviceIndex device(Device::CUDA, 0);

  const int64_t B = 2, H = 4, D = 64, T = 64;
  const int     factor = 4;
  const int64_t scaled_max = factor * T;          // 256
  const float scale = 1.0f / std::sqrt((float)D);
  const float eps = 1e-6f;

  std::cout << "=== CREAM cache-gather parity (single GPU, T=" << T
            << " scaled_max=" << scaled_max << ") ===\n";

  // ---- CASE 1: gather mechanics + kernel identity on a CREAM-relabeled cache --
  Tensor full = autograd::build_rope_cache(scaled_max, D, 10000.0f, device);  // plain (scale=1)

  // A representative CREAM label vector (monotonic head<middle<tail).
  std::vector<int64_t> L =
      OwnTensor::cp::cream::generate_cream_positions((int)T, factor, /*sigma=*/3.0, /*seed=*/42);
  if ((int64_t)L.size() != T) { std::cout << "bad L size\n"; return 1; }

  Tensor idx = build_gather_index(L, D, device);
  Tensor Gd  = OwnTensor::gather(full, /*dim=*/0, idx);   // [T, D] relabeled cache

  // (a) gather correctness: Gd row j == full row L[j], exactly.
  {
    Tensor Gd_cpu = Gd.to_cpu(), full_cpu = full.to_cpu();
    const float* g = Gd_cpu.data<float>(); const float* f = full_cpu.data<float>();
    float m = 0.0f;
    for (int64_t j = 0; j < T; ++j)
      for (int64_t c = 0; c < D; ++c)
        m = std::max(m, std::abs(g[j*D + c] - f[L[(size_t)j]*D + c]));
    gate(m == 0.0f, "gather rows == full[L[j]] (exact)", 1.0, m);
  }

  // (b) kernel (deltas=0, cache=Gd) == de-fused rope@Gd + SDPA.
  {
    Shape qkv({{B, H, T, D}}), gsh({{D}});
    TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32).with_device(device);
    Tensor Q = bf16_round(Tensor::randn<float>(qkv, opts, 100, 0.5f));
    Tensor K = bf16_round(Tensor::randn<float>(qkv, opts, 200, 0.5f));
    Tensor V = bf16_round(Tensor::randn<float>(qkv, opts, 300, 0.5f));
    Tensor qg = Tensor::randn<float>(gsh, opts, 11, 0.1f);
    Tensor kg = Tensor::randn<float>(gsh, opts, 22, 0.1f);

    // Reference rotates row j with Gd[j] == full[L[j]] (RoPE at CREAM label L[j]).
    Tensor qr = autograd::rope(autograd::rms_norm(Q, qg, (int)D, eps), Gd, false, 0);
    Tensor kr = autograd::rope(autograd::rms_norm(K, kg, (int)D, eps), Gd, false, 0);
    Tensor ref = ref_sdpa(qr, kr, V, scale);

    Tensor Qc = Q, Kc = K, Vc = V;
    SDPAResult res = sdpa_fused_forward_rope(
        Qc, Kc, Vc, /*is_causal=*/true, scale,
        /*q_d0=*/0, /*q_d1=*/0, /*k_d0=*/0, /*k_d1=*/0, Gd, qg, kg, eps);

    Tensor a = ref.to_cpu(), b = res.out.to_cpu();
    double c = cosine(a, b); float m = max_abs_diff(a, b);
    gate(c > 0.999, "kernel(cache=Gd,delta=0) vs ref", c, m);
  }

  // ---- CASE 2: YaRN spot-check -- the gather SOURCE is truly YaRN-scaled --------
  {
    // Plain (scale=1) already built as `full`. Build a YaRN-scaled cache for the
    // same length: scale=4, orig_maxpos=T so scaled_max==factor*T (no warn).
    setenv("YARN_SCALE", "4", 1);
    setenv("YARN_ORIG_MAXPOS", std::to_string(T).c_str(), 1);
    Tensor yarn = autograd::build_rope_cache(scaled_max, D, 10000.0f, device);
    unsetenv("YARN_SCALE"); unsetenv("YARN_ORIG_MAXPOS");

    Tensor y_cpu = yarn.to_cpu(), p_cpu = full.to_cpu();
    const float* y = y_cpu.data<float>(); const float* p = p_cpu.data<float>();
    // This YaRN impl bakes the attention-temperature scalar m into the cache, so
    // YaRN and plain differ in MAGNITUDE even at pos 0. The meaningful, temperature-
    // independent signal is DIRECTION (row-vector cosine): at pos 0 every angle is 0
    // for both -> direction identical (cos==1); at a high pos NTK-by-parts changes
    // the per-dim angles -> direction diverges (cos<1). That divergence is the proof
    // the gathered cache is YaRN-interpolated, not plain-extrapolated.
    auto row_cos = [&](int64_t r)->double {
      double dot = 0, na = 0, nb = 0;
      for (int64_t c = 0; c < D; ++c) {
        dot += (double)y[r*D+c]*p[r*D+c]; na += (double)y[r*D+c]*y[r*D+c];
        nb += (double)p[r*D+c]*p[r*D+c];
      }
      return dot / (std::sqrt(na) * std::sqrt(nb));
    };
    const double c0 = row_cos(0);
    const double ch = row_cos(scaled_max - 1);
    gate(c0 > 0.999999, "YaRN dir==plain at pos 0", c0, 0.0f);
    gate(ch < 0.999,    "YaRN dir!=plain at high pos (active)", ch, 0.0f);
  }

  if (g_fail == 0) std::cout << "\nALL cream_cache_gather parity gates PASSED.\n";
  else             std::cout << "\n" << g_fail << " parity gate(s) FAILED.\n";
  return g_fail == 0 ? 0 : 1;
}

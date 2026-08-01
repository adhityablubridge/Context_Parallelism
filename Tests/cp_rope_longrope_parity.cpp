// =============================================================================
// cp_rope_longrope_parity.cpp -- unit test for build_rope_cache_longrope.
//
// CPU-only value checks on the cache tensor (no kernel, no GPU needed):
//   A. identity : lambda=1, n_hat=0  ==  build_rope_cache at YARN_SCALE=1 (plain RoPE).
//   B. PI       : lambda=s, n_hat=0  ==  hand-computed pos*base_freq/s (divide-by-lambda sign).
//   C. n_hat    : lambda=2, n_hat=K  -> rows < K are plain RoPE, rows >= K are /2.
//   D. guards   : non-monotone lambda and wrong-size lambda throw.
//
// Build:  make CP_FUSED_ROPE=1 cp-rope-longrope   (links the CP/libtensor objects)
// Run:    ./build/cp_rope_longrope_parity_exec
// =============================================================================
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <vector>

#include "TensorLib.h"
#include "autograd/operations/RoPEOps.h"          // build_rope_cache
#include "context_parallel/LongRoPEOps.h"         // build_rope_cache_longrope

using namespace OwnTensor;

static int g_fail = 0;

static void gate(bool ok, const char* tag, double maxdiff) {
    std::printf("%s %-34s maxdiff=%.3e\n", ok ? "  PASS" : "  FAIL", tag, maxdiff);
    if (!ok) ++g_fail;
}

// Direct reference for LongRoPE row `pos`, dim `i` (NeoX half-split).
static void ref_row(double base, int64_t head_dim, int64_t pos, int64_t i,
                    double lambda_i, int64_t n_hat, double& c, double& s_) {
    const int64_t half = head_dim / 2;
    (void)half;
    double base_freq = 1.0 / std::pow(base, (2.0 * i) / head_dim);
    double scale = (pos < n_hat) ? 1.0 : lambda_i;
    double ang = (double)pos * base_freq / scale;
    c = std::cos(ang); s_ = std::sin(ang);
}

int main() {
    const int64_t T = 32, HD = 64;
    const float   BASE = 10000.0f;
    const int64_t half = HD / 2;
    DeviceIndex cpu(Device::CPU);

    // ---- A. identity: lambda=1, n_hat=0 == build_rope_cache at YARN_SCALE=1 ----
    setenv("YARN_SCALE", "1", 1);   // ensure plain RoPE from the YaRN builder
    {
        std::vector<float> lam(half, 1.0f);
        Tensor lr  = autograd::build_rope_cache_longrope(T, HD, BASE, lam, /*n_hat=*/0, cpu);
        Tensor yar = autograd::build_rope_cache(T, HD, BASE, cpu);
        const float* a = lr.data<float>(); const float* b = yar.data<float>();
        double m = 0.0;
        for (int64_t k = 0; k < T * HD; ++k) m = std::max(m, (double)std::abs(a[k] - b[k]));
        gate(m < 1e-6, "identity lambda=1 == plain RoPE", m);
    }
    unsetenv("YARN_SCALE");

    // ---- B. PI: lambda=s, n_hat=0 == pos*base_freq/s (hand-computed) ----
    {
        const double s = 4.0;
        std::vector<float> lam(half, (float)s);
        Tensor lr = autograd::build_rope_cache_longrope(T, HD, BASE, lam, /*n_hat=*/0, cpu);
        const float* a = lr.data<float>();
        double m = 0.0;
        for (int64_t pos = 0; pos < T; ++pos)
            for (int64_t i = 0; i < half; ++i) {
                double c, si; ref_row(BASE, HD, pos, i, s, /*n_hat=*/0, c, si);
                m = std::max(m, (double)std::abs(a[pos*HD + i]        - c));
                m = std::max(m, (double)std::abs(a[pos*HD + i + half] - si));
            }
        gate(m < 1e-5, "PI lambda=s == pos*base_freq/s", m);
    }

    // ---- C. n_hat threshold: lambda=2, n_hat=8; rows<8 plain, rows>=8 halved ----
    {
        const double lamv = 2.0; const int64_t NHAT = 8;
        std::vector<float> lam(half, (float)lamv);
        Tensor lr = autograd::build_rope_cache_longrope(T, HD, BASE, lam, NHAT, cpu);
        const float* a = lr.data<float>();
        double m = 0.0;
        for (int64_t pos : {0, 3, 7, 8, 9, 20}) {   // straddle the threshold
            for (int64_t i = 0; i < half; ++i) {
                double c, si; ref_row(BASE, HD, pos, i, lamv, NHAT, c, si);
                m = std::max(m, (double)std::abs(a[pos*HD + i]        - c));
                m = std::max(m, (double)std::abs(a[pos*HD + i + half] - si));
            }
        }
        gate(m < 1e-5, "n_hat threshold (rows<K plain, >=K /2)", m);
    }

    // ---- D. guards throw ----
    {
        auto throws = [](auto fn) { try { fn(); return false; } catch (...) { return true; } };
        bool t1 = throws([&]{ std::vector<float> l(half, 1.0f); l[5]=2.0f; l[6]=1.5f;   // non-monotone
                              autograd::build_rope_cache_longrope(T, HD, BASE, l, 0, DeviceIndex(Device::CPU)); });
        bool t2 = throws([&]{ std::vector<float> l(half-1, 1.0f);                        // wrong size
                              autograd::build_rope_cache_longrope(T, HD, BASE, l, 0, DeviceIndex(Device::CPU)); });
        bool t3 = throws([&]{ std::vector<float> l(half, 0.5f);                          // lambda<1
                              autograd::build_rope_cache_longrope(T, HD, BASE, l, 0, DeviceIndex(Device::CPU)); });
        gate(t1 && t2 && t3, "guards throw (mono / size / lambda>=1)", 0.0);
    }

    if (g_fail == 0) { std::printf("\nALL cp_rope_longrope parity tests passed.\n"); return 0; }
    std::printf("\n%d cp_rope_longrope test(s) FAILED.\n", g_fail);
    return 1;
}

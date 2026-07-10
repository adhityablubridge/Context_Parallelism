
#include "autograd/operations/RoPEOps.h"
#include "autograd/backward/RoPEBackward.h"
#include "autograd/AutogradContext.h"
#include "autograd/Variable.h"
#include "autograd/GraphRecorder.h"
#include "checkpointing/GradMode.h"
#include "ops/helpers/RoPEKernels.h"
#include "autograd/ops_template.h"

#include <stdexcept>
#include <cmath>
#include <cstdlib>   // std::getenv, std::atof
#include <cstdio>    // std::fprintf
#include <string>    // std::to_string
#include <algorithm> // std::max, std::min

namespace OwnTensor {
namespace autograd {

// ============================================================================
// build_rope_cache — YaRN (NTK-by-parts + attention temperature) cos/sin cache
//
// Canonical HF/Qwen3 YaRN. Extends context (e.g. 1024 -> 8192) by blending
// per-dimension inverse frequencies (extrapolate high-freq, interpolate low-freq
// via a dimension-index linear ramp between correction dims) and baking the
// attention temperature m = 0.1*ln(s)+1 into BOTH cos and sin. RoPE hits both Q
// and K, so m appears twice in each QK logit (m^2 = 1/t), matching YaRN's
// softmax temperature; the fused kernel's 1/sqrt(hd) scale is left untouched.
//
// Params are hardcoded defaults, each overridable by an env var. Default
// YARN_SCALE=1.0 reduces this to standard RoPE bit-identically (base_freq/s ==
// base_freq for all i, and m==1), preserving all existing callers.
//
// The cache is always built on CPU (fp32) then copied to the target device: the
// device kernel rope_build_cache_cuda has no per-dimension frequency hook, so we
// bypass it. Cost is negligible (built once at model init).
// ============================================================================
namespace {
inline float yarn_env_f(const char* key, float dflt) {
    const char* v = std::getenv(key);
    return v ? static_cast<float>(std::atof(v)) : dflt;
}
} // anonymous namespace

Tensor build_rope_cache(int64_t seq_len, int64_t head_dim, float base, DeviceIndex device)
{
    if (head_dim % 2 != 0) {
        throw std::invalid_argument("build_rope_cache: head_dim must be even");
    }

    // --- YaRN config (env-overridable; scale=1.0 => standard RoPE) ---
    const float s      = yarn_env_f("YARN_SCALE",       1.0f);   // target_ctx / original_ctx
    const float L      = yarn_env_f("YARN_ORIG_MAXPOS", 1024.0f);// original pretrain context
    const float b_fast = yarn_env_f("YARN_BETA_FAST",   32.0f);  // "beta"  (high-rotation bound)
    const float b_slow = yarn_env_f("YARN_BETA_SLOW",   1.0f);   // "alpha" (low-rotation bound)

    // Attention temperature m (s==1 -> m==1 -> no scaling).
    const float m = (s > 1.0f) ? (0.1f * std::log(s) + 1.0f) : 1.0f;

    const int64_t half = head_dim / 2;

    // Canonical correction range: linear ramp over dimension index between the
    // dims where the number of rotations equals b_fast (high-freq end) and
    // b_slow (low-freq end). Matches HF find_correction_dim/find_correction_range.
    auto find_dim = [&](float num_rot) {
        return (static_cast<float>(head_dim)
                * std::log(L / (num_rot * 2.0f * static_cast<float>(M_PI))))
               / (2.0f * std::log(base));
    };
    int64_t low  = static_cast<int64_t>(std::floor(find_dim(b_fast)));
    int64_t high = static_cast<int64_t>(std::ceil (find_dim(b_slow)));
    low  = std::max<int64_t>(low, 0);
    high = std::min<int64_t>(high, half - 1);   // clamp to half-1: loop indexes [0,half) only;
                                                // deliberate deviation from HF head_dim-1 -- do not "fix"
    float denom = static_cast<float>(high - low);
    if (denom <= 0.0f) denom = 1e-3f;           // avoid /0 when low==high

    // (R5) Guard degenerate env input: L/(num_rot*2pi) <= 0 -> log(neg) = NaN.
    if (s > 1.0f && (!std::isfinite(find_dim(b_fast)) || !std::isfinite(find_dim(b_slow)) || low > high)) {
        throw std::runtime_error(
            "build_rope_cache(YaRN): bad correction range -- check "
            "YARN_BETA_FAST/YARN_BETA_SLOW/YARN_ORIG_MAXPOS (got low="
            + std::to_string(low) + " high=" + std::to_string(high) + ")");
    }

    // (R2) scale <-> seq_len consistency (warn, don't hard-fail).
    if (s > 1.0f && std::abs(static_cast<float>(seq_len) - s * L) > 1.0f) {
        std::fprintf(stderr,
            "[YaRN][warn] seq_len(%lld) != scale(%.3f)*orig_maxpos(%.1f)=%.1f "
            "-- cache scaled for a different context than it is sized for\n",
            (long long)seq_len, s, L, s * L);
    }

    // (R3) Init log: makes the active config + the YARNOps-symbol-wins link
    // dependency visible. Missing line / scale=1 => silent revert to std RoPE.
    std::fprintf(stderr,
        "[YaRN] build_rope_cache: seq_len=%lld hd=%lld scale=%.3f low=%lld high=%lld m=%.4f\n",
        (long long)seq_len, (long long)head_dim, s, (long long)low, (long long)high, m);

    // --- Build on CPU, fill, then copy to device (bypass the CUDA kernel). ---
    TensorOptions cpu_opts = TensorOptions().with_dtype(Dtype::Float32)
                                            .with_device(DeviceIndex(Device::CPU));
    Tensor cpu(Shape{{seq_len, head_dim}}, cpu_opts);
    float* p = cpu.data<float>();

    for (int64_t pos = 0; pos < seq_len; ++pos) {
        for (int64_t i = 0; i < half; ++i) {
            float exponent  = (2.0f * static_cast<float>(i)) / static_cast<float>(head_dim);
            float base_freq = 1.0f / std::pow(base, exponent);   // extrapolation freq

            float ramp = (static_cast<float>(i) - static_cast<float>(low)) / denom;
            ramp = ramp < 0.0f ? 0.0f : (ramp > 1.0f ? 1.0f : ramp);
            // ramp=0 (i<low, high-freq) -> keep base_freq (extrapolate)
            // ramp=1 (i>high, low-freq) -> base_freq/s   (interpolate)
            float inv_freq = (base_freq / s) * ramp + base_freq * (1.0f - ramp);

            float angle = static_cast<float>(pos) * inv_freq;
            p[pos * head_dim + i]        = std::cos(angle) * m;  // bake temperature m
            p[pos * head_dim + i + half] = std::sin(angle) * m;
        }
    }

    cpu.set_requires_grad(false);
    if (device.device == Device::CUDA) {
        Tensor cache = cpu.to(device);
        cache.set_requires_grad(false);
        return cache;
    }
    return cpu;
}

// ============================================================================
// rope — separate-tensor forward
// Operates on x of shape [B, T, H, D]; returns rotated tensor (out-of-place).
// ============================================================================
Tensor rope(const Tensor& x,
            const Tensor& cos_sin_cache,
            bool interleaved,
            int64_t position_offset)
{
    GraphRecordMode::record_forward("ROPE: rope");

    if (x.ndim() != 4) {
        throw std::invalid_argument("rope: input must be 4-D [B, T, H, D]");
    }
    if (x.dtype() != Dtype::Float32 || cos_sin_cache.dtype() != Dtype::Float32) {
        throw std::invalid_argument("rope: only fp32 is supported in v1");
    }

    int64_t B  = x.shape().dims[0];
    int64_t T  = x.shape().dims[1];
    int64_t H  = x.shape().dims[2];
    int64_t D  = x.shape().dims[3];
    if (D % 2 != 0) throw std::invalid_argument("rope: head_dim must be even");

    // Cache is [seq_len, head_dim]; every token's pos = t + position_offset must
    // index a valid row. Validate before launch so we fail with a clear message
    // instead of an out-of-bounds device read.
    int64_t cache_seq_len = cos_sin_cache.shape().dims[0];
    if (T + position_offset > cache_seq_len) {
        throw std::invalid_argument(
            "rope: T + position_offset (" + std::to_string(T + position_offset) +
            ") exceeds cos_sin_cache seq_len (" + std::to_string(cache_seq_len) +
            "); rebuild cache with a larger seq_len");
    }

    // Out-of-place: clone input, rotate the clone in place. Clone is
    // contiguous so strides are simple.
    Tensor out = x.contiguous().clone();

    int64_t token_stride = H * D;   // contiguous [B,T,H,D]
    int64_t head_stride  = D;

    if (x.device().device == Device::CUDA) {
        cuda::rope_forward_cuda(
            out.data<float>(),
            /*k=*/nullptr,
            cos_sin_cache.data<float>(),
            B, T,
            cache_seq_len,
            /*nh_q=*/H, /*nh_k=*/0, D,
            token_stride, head_stride,
            /*k_token_stride=*/0, /*k_head_stride=*/0,
            position_offset,
            interleaved,
            /*inverse=*/false);
    } else {
        throw std::runtime_error("rope: CPU path not yet wired (see RoPE.cpp)");
    }

    if (GradMode::is_enabled() && x.requires_grad()) {
        auto grad_fn = std::make_shared<RoPEBackward>(
            cos_sin_cache, interleaved, position_offset);
        grad_fn->set_next_edge(0, get_grad_edge(x));
        out.set_grad_fn(grad_fn);
        out.set_requires_grad(true);
    }

    if (autograd::g_shape_debug) GraphRecordMode::attach_forward_shape(out.shape(), out.dtype());
    return out;
}

// ============================================================================
// rope_packed_qk — packed-QKV forward
// Rotates Q (heads [0..nh_q)) and K (heads [nh_q..nh_q+nh_k)) of a packed
// [B, T, (nh_q+nh_k+nh_v)*D] tensor. V slice is copied through untouched.
// ============================================================================
Tensor rope_packed_qk(const Tensor& qkv,
                      int64_t nh_q, int64_t nh_k, int64_t nh_v,
                      int64_t head_dim,
                      const Tensor& cos_sin_cache,
                      bool interleaved,
                      int64_t position_offset)
{
    GraphRecordMode::record_forward("ROPE: rope_packed_qk");

    if (qkv.ndim() != 3) {
        throw std::invalid_argument("rope_packed_qk: qkv must be 3-D [B, T, C]");
    }
    if (qkv.dtype() != Dtype::Float32 || cos_sin_cache.dtype() != Dtype::Float32) {
        throw std::invalid_argument("rope_packed_qk: only fp32 is supported in v1");
    }
    if (head_dim % 2 != 0) throw std::invalid_argument("rope_packed_qk: head_dim must be even");

    int64_t B = qkv.shape().dims[0];
    int64_t T = qkv.shape().dims[1];
    int64_t C = qkv.shape().dims[2];
    int64_t expected_C = (nh_q + nh_k + nh_v) * head_dim;
    if (C != expected_C) {
        throw std::invalid_argument("rope_packed_qk: last dim does not match (nh_q+nh_k+nh_v)*head_dim");
    }

    // Cache is [seq_len, head_dim]; pos = t + position_offset must index a valid
    // row. Validate before launch (clear error instead of an OOB device read).
    int64_t cache_seq_len = cos_sin_cache.shape().dims[0];
    if (T + position_offset > cache_seq_len) {
        throw std::invalid_argument(
            "rope_packed_qk: T + position_offset (" + std::to_string(T + position_offset) +
            ") exceeds cos_sin_cache seq_len (" + std::to_string(cache_seq_len) +
            "); rebuild cache with a larger seq_len");
    }

    Tensor out = qkv.contiguous().clone();

    // In packed layout: token stride spans the full Q|K|V row; head stride is D.
    // Q starts at offset 0; K starts at offset nh_q * head_dim within each token.
    int64_t token_stride = (nh_q + nh_k + nh_v) * head_dim;
    int64_t head_stride  = head_dim;

    if (qkv.device().device == Device::CUDA) {
        float* base = out.data<float>();
        float* q_ptr = base;
        float* k_ptr = base + nh_q * head_dim;

        cuda::rope_forward_cuda(
            q_ptr, k_ptr,
            cos_sin_cache.data<float>(),
            B, T,
            cache_seq_len,
            nh_q, nh_k, head_dim,
            token_stride, head_stride,
            token_stride, head_stride,
            position_offset,
            interleaved,
            /*inverse=*/false);
    } else {
        throw std::runtime_error("rope_packed_qk: CPU path not yet wired (see RoPE.cpp)");
    }

    if (GradMode::is_enabled() && qkv.requires_grad()) {
        auto grad_fn = std::make_shared<RoPEPackedBackward>(
            cos_sin_cache, nh_q, nh_k, nh_v, head_dim, interleaved, position_offset);
        grad_fn->set_next_edge(0, get_grad_edge(qkv));
        out.set_grad_fn(grad_fn);
        out.set_requires_grad(true);
    }

    if (autograd::g_shape_debug) GraphRecordMode::attach_forward_shape(out.shape(), out.dtype());
    return out;
}

} // namespace autograd
} // namespace OwnTensor

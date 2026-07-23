// =============================================================================
// bluscriptCP.cpp — CONTEXT-PARALLEL Llama-style decoder LM training.
//
// Context-parallel port of bluscript.cpp: the SEQUENCE is split across GPUs
// (not the batch), so we can train at long context. Attention runs through the
// CP layer's FUSED paths (RoPE + QK-norm + GQA happen INSIDE ContextParallel):
//   CP_ATTN_MODE=ring    (default) -> enable_rope       (ring; new fused-RoPE CP
//                                     kernels; HeadTail load-balance; needs
//                                     building with CP_FUSED_ROPE=1)
//   CP_ATTN_MODE=ulysses           -> enable_ulysses_fused (all-to-all; GQA;
//                                     contiguous shard; default build)
//
// Topology: CP_SIZE (env, default = world_size => CP-only). dp_size =
// world_size/CP_SIZE. A 2-D DeviceMesh({dp_size, cp_size}) gives CP-only
// (dp_size==1) or 2-D DP x CP. cp_pg (mesh axis 1) drives the attention ring;
// dp_rank (axis 0) selects which batch each rank loads. DataParallel is KEPT
// over the GLOBAL pg -> init-weight broadcast + param-grad all-reduce over all
// dp*cp ranks (the correct group, since params replicate globally). The ring
// only adds the attention-activation (dQ/dK/dV) grad sum over cp_pg (disjoint
// from DDP's leaf-param reduction — no double-count).
//
// Launch (ring, 2 GPUs, CP-only):
//   make CP_FUSED_ROPE=1 bluscript-cp && make CP_FUSED_ROPE=1 run-bluscript-cp NP=2
// =============================================================================

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <map>
#include <random>

#include <mpi.h>
#include <cuda_runtime.h>

// Tensor library
#include "TensorLib.h"
#include "autograd/AutogradOps.h"               // matmul, add, reshape, transpose, swiglu, ...
#include "autograd/operations/RoPEOps.h"        // build_rope_cache
#include "autograd/operations/LossOps.h"        // sparse_cross_entropy_loss
#include "autograd/operations/EmbeddingOps.h"
#include "nn/NN.h"
#include "nn/optimizer/Optim.h"
#include "checkpointing/GradMode.h"
#include "checkpointing/Checkpointing.h"       // CheckpointManager (save / load_latest)
#include "device/CachingCudaAllocator.h"

// Data + distributed
#include "DataLoader.h"
// NOTE: include DataParallel.hpp DIRECTLY (not the dist/distributed.h umbrella,
// which pulls the TP stack incl. a different `class DeviceMesh` that collides
// with the CP one). init_process_group comes from the canonical ProcessGroupNCCL.h
// (pulled in via device_mesh.h / ContextParallel.h).
#include "DataParallel.hpp"                     // DataParallel, DDP_Options

// Context parallelism
#include "process_group/device_mesh.h"          // DeviceMesh (canonical PG, ncclCommSplit)
#include "context_parallel/ContextParallel.h"   // ContextParallel, shard_sequence_pre_embed, RotatorType
#include "context_parallel/CreamPositions.h"     // CREAM/PoSE/RandPos position-label generators
#include "ops/IndexingOps.h"                      // gather (cache-row select for CREAM)

using namespace OwnTensor;

// =============================================================================
// CudaTimer — event-based per-phase timer (ported from gpt2_cp_test.cpp so the
// per-step [TIMING] breakdown matches that script's terminal output).
// =============================================================================
struct CudaTimer {
    cudaEvent_t start, stop;
    CudaTimer()  { cudaEventCreate(&start); cudaEventCreate(&stop); }
    ~CudaTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void start_timer() { cudaEventRecord(start); }
    float get_elapsed_ms() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0; cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
    double get_elapsed_seconds() { return get_elapsed_ms() / 1000.0; }
};

// Discards all output — used to temporarily silence the DataLoader's per-shard
// debug prints during the resume fast-forward (skip_batches), which otherwise
// spam one line per shard crossing, per rank.
struct NullBuf : std::streambuf {
    int overflow(int c) override { return c; }
};

// =============================================================================
// Configuration
// =============================================================================
struct ModelConfig {
    // ---- model ----
    int64_t d_model        = 384;
    int64_t n_layers       = 6;
    int64_t q_heads        = 6;
    int64_t kv_heads       = 2;                          // grouped-query: G = q/kv = 3
    int64_t head_dim       = 384 / 12;                   // = 64  (fused kernel needs 64 or 128)
    int64_t ffn_hidden     = (8 * 384) / 3;              // 8/3 * d_model = 2048 (SwiGLU inner dim)
    int64_t context_length = 4096;                       // = T (multiple of 2*CP_SIZE, RoPE cache length)
    int     rope_theta     = 500000;
    int64_t vocab_size     = 50304;                      // 50257 (tiktoken) padded for GPU efficiency
    bool    weight_tying   = false;

    float   rms_eps        = 1e-5f;                      // block RMSNorm epsilon
    float   qk_eps         = 1e-6f;                      // QK-norm (fused) epsilon
    bool    rope_interleaved = false;                    // false = NeoX half-split (Llama style)

    // ---- optimization / data ----
    int64_t B            = 2;                            // micro-batch per rank
    int64_t T            = 4096;                         // sequence length (== context_length)
    int64_t global_batch = 524288;                       // tokens per optimizer step (all ranks)
    float   grad_clip    = 1.0f;
    float   weight_decay = 0.1f;

    float   max_lr       = 6e-4f;
    float   min_lr       = 6e-5f;                        // 10% of max_lr
    int     warmup_steps = 91;
    int     max_steps    = 917;                        // ~10B tokens at 0.5M tokens/step

    int     val_freq       = 250;
    int     val_loss_steps = 20;

    // ---- context parallel (env-resolved in main) ----
    int         cp_size    = 2;                          // sequence-parallel group size (CP_SIZE)
    std::string attn_mode  = "ulysses";                     // "ring" | "ulysses" | "hybrid" (CP_ATTN_MODE)
    RotatorType rotator    = RotatorType::P2P;      // CP_ROTATOR
    bool        ring_mode  = true;                       // (attn_mode == "ring") -> HeadTail LB
    // ---- hybrid 3-D CP (Ulysses-inner + Ring-outer); active when attn_mode=="hybrid" ----
    int         ulysses_size = 1;                        // CP_ULYSSES_SIZE (inner all-to-all degree)
    bool        hybrid_mode  = false;                    // (attn_mode == "hybrid")

    // ---- CREAM context-extension position relabeling (additive; env-resolved) ----
    // "off" (default) => byte-identical to prior behavior. "cream"|"pose"|"randpos"
    // feed manipulated RoPE position LABELS via a gathered cos/sin cache. Single-GPU
    // (CP_SIZE=1) only for now. factor(=YARN_SCALE) and scaled_max(=factor*T) derived.
    std::string cream_mode      = "off";                 // CP_CREAM_MODE
    double      cream_sigma     = 3.0;                    // CP_CREAM_SIGMA (truncated-Gaussian)
    uint64_t    cream_seed      = 0;                      // CP_CREAM_SEED base (0 => step-derived)
    int         cream_factor    = 0;                      // = YARN_SCALE (resolved in main)
    int64_t     cream_scaled_max = 0;                     // = cream_factor * T (resolved in main)

    // ---- checkpointing ----
    bool        checkpointing = true;
    int         ckpt_freq     = 50;
    int         ckpt_keep     = 6;
    std::string ckpt_dir      = "checkpoints_bluscriptcp";
    std::string ckpt_prefix   = "blumodelcp";
};

// =============================================================================
// Shared advancing RNG for custom weight init (unchanged from bluscript.cpp).
// =============================================================================
struct InitRng {
    std::mt19937 gen;
    explicit InitRng(uint64_t seed) : gen(static_cast<uint32_t>(seed)) {}
    Tensor normal(const Shape& shape, float std) {
        Tensor t(shape, TensorOptions().with_dtype(Dtype::Float32));
        std::normal_distribution<float> dist(0.0f, std);
        float* d = t.data<float>();
        const size_t n = t.numel();
        for (size_t i = 0; i < n; ++i) d[i] = dist(gen);
        return t;
    }
};

static void copy_into_param(Tensor& dst, const Tensor& src) {
    const bool was = dst.requires_grad();
    if (was) dst.set_requires_grad(false);
    dst.copy_(src);
    if (was) dst.set_requires_grad(true);
}
static void init_linear(nn::Linear& layer, float std, InitRng& rng) {
    copy_into_param(layer.weight, rng.normal(layer.weight.shape(), std));
    if (layer.bias.is_valid()) {
        copy_into_param(layer.bias,
                        Tensor::zeros(layer.bias.shape(), TensorOptions().with_dtype(Dtype::Float32)));
    }
}
static void init_embedding(nn::Embedding& emb, float std, InitRng& rng) {
    copy_into_param(emb.weight, rng.normal(emb.weight.shape(), std));
}

// =============================================================================
// Transformer block: RMSNorm -> CP fused (QK-norm + RoPE + causal GQA) -> Wo -> +x
// The attention (RoPE + QK-norm + GQA + causal) runs INSIDE ContextParallel via
// the fused ring (enable_rope) or Ulysses (enable_ulysses_fused) path. x is the
// LOCAL sequence shard [B, T_local, d_model]; forward_cp keeps the output local.
// =============================================================================
class CausalGQA : public nn::Module {
public:
    nn::RMSNorm norm;
    nn::Linear  wQ, wK, wV, wO;
    Tensor      q_gamma, k_gamma;
    Tensor      cos_sin_cache;
    ModelConfig cfg;
    std::shared_ptr<ContextParallel> cp_;

    // cp_pg is the CP process group. In hybrid mode it is the RING sub-group and
    // ulysses_pg carries the inner all-to-all; otherwise ulysses_pg is null.
    CausalGQA(const ModelConfig& c, DeviceIndex dev, const Tensor& cache, InitRng& rng,
              const DeviceMesh& mesh, std::shared_ptr<ProcessGroupNCCL> cp_pg,
              std::shared_ptr<ProcessGroupNCCL> ulysses_pg = nullptr)
        : norm(c.d_model, c.rms_eps),
          wQ(c.d_model, c.q_heads  * c.head_dim, /*bias=*/false),
          wK(c.d_model, c.kv_heads * c.head_dim, /*bias=*/false),
          wV(c.d_model, c.kv_heads * c.head_dim, /*bias=*/false),
          wO(c.q_heads * c.head_dim, c.d_model,  /*bias=*/false),
          cos_sin_cache(cache),
          cfg(c)
    {
        const float resid_scale = 1.0f / std::sqrt(2.0f * static_cast<float>(c.n_layers));
        init_linear(wQ, 0.02f, rng);
        init_linear(wK, 0.02f, rng);
        init_linear(wV, 0.02f, rng);
        init_linear(wO, 0.02f * resid_scale, rng);

        auto gopts = TensorOptions().with_dtype(Dtype::Float32).with_device(dev);
        q_gamma = Tensor::full(Shape{{c.head_dim}}, gopts, 1.0f);
        k_gamma = Tensor::full(Shape{{c.head_dim}}, gopts, 1.0f);
        q_gamma.set_requires_grad(true);
        k_gamma.set_requires_grad(true);

        norm.to(dev); wQ.to(dev); wK.to(dev); wV.to(dev); wO.to(dev);

        register_module(norm);
        register_module(wQ); register_module(wK); register_module(wV);
        register_module(wO);
        register_parameter(q_gamma);
        register_parameter(k_gamma);

        // ---- Context-parallel attention layer ----
        // ring   => HeadTail load-balance (matches shard_sequence_pre_embed's perm);
        // ulysses=> contiguous shard;
        // hybrid => Ulysses-inner (ulysses_pg) nested in Ring-outer (cp_pg==ring pg),
        //           HeadTail on the ring axis. RoPE/QK-norm/GQA applied INSIDE CP.
        const float attn_scale = 1.0f / std::sqrt(static_cast<float>(c.head_dim));
        const bool  lb = (c.ring_mode || c.hybrid_mode);
        cp_ = std::make_shared<ContextParallel>(
            mesh, cp_pg, attn_scale, /*is_causal=*/true, c.rotator,
            /*load_balance=*/lb);
        if (c.hybrid_mode) {
            cp_->enable_hybrid(ulysses_pg, c.ulysses_size, cos_sin_cache,
                               q_gamma, k_gamma, c.qk_eps);
        } else if (c.ring_mode) {
            cp_->enable_rope(cos_sin_cache, q_gamma, k_gamma, c.qk_eps);
        } else {
            cp_->enable_ulysses_fused(cos_sin_cache, q_gamma, k_gamma, c.qk_eps,
                                      c.rope_interleaved);
        }
    }

    // x: LOCAL sequence shard [B, T_local, d_model]
    Tensor forward(const Tensor& x) override {
        const int64_t B  = x.shape().dims[0];
        const int64_t Tl = x.shape().dims[1];

        Tensor h = norm.forward(x);
        Tensor q = wQ.forward(h);                        // [B, Tl, Nq*hd]
        Tensor k = wK.forward(h);                        // [B, Tl, Nkv*hd]
        Tensor v = wV.forward(h);                        // [B, Tl, Nkv*hd]

        // -> head-major [B, heads, T_local, hd]
        q = autograd::transpose(autograd::reshape(q, Shape{{B, Tl, cfg.q_heads,  cfg.head_dim}}), 1, 2);
        k = autograd::transpose(autograd::reshape(k, Shape{{B, Tl, cfg.kv_heads, cfg.head_dim}}), 1, 2);
        v = autograd::transpose(autograd::reshape(v, Shape{{B, Tl, cfg.kv_heads, cfg.head_dim}}), 1, 2);

        // CP fused attention: RoPE + QK-norm + causal GQA inside the ring/ulysses
        // path. pre_sharded=true (q/k/v are already the local shard); unshard=false
        // (output stays [B, Nq, T_local, hd], no all-gather).
        Tensor attn = cp_->forward_cp(q, k, v, /*unshard=*/false, /*pre_sharded=*/true);

        Tensor merged = autograd::reshape(autograd::transpose(attn, 1, 2),
                                          Shape{{B, Tl, cfg.q_heads * cfg.head_dim}});
        Tensor proj = wO.forward(merged);
        return autograd::add(x, proj);                   // residual
    }
};

// =============================================================================
// SwiGLU feed-forward (unchanged from bluscript.cpp; operates on the local shard)
// =============================================================================
class SwiGLUMLP : public nn::Module {
public:
    nn::RMSNorm norm;
    nn::Linear  gate_up;
    nn::Linear  down;

    SwiGLUMLP(const ModelConfig& c, DeviceIndex dev, InitRng& rng)
        : norm(c.d_model, c.rms_eps),
          gate_up(c.d_model, 2 * c.ffn_hidden, /*bias=*/false),
          down(c.ffn_hidden, c.d_model,        /*bias=*/false)
    {
        const float resid_scale = 1.0f / std::sqrt(2.0f * static_cast<float>(c.n_layers));
        init_linear(gate_up, 0.02f, rng);
        init_linear(down,    0.02f * resid_scale, rng);
        norm.to(dev); gate_up.to(dev); down.to(dev);
        register_module(norm);
        register_module(gate_up);
        register_module(down);
    }

    Tensor forward(const Tensor& x) override {
        Tensor h   = norm.forward(x);
        Tensor gu  = gate_up.forward(h);
        Tensor act = autograd::swiglu(gu);
        Tensor out = down.forward(act);
        return autograd::add(x, out);
    }
};

// =============================================================================
// Full model. GPT::forward shards the SEQUENCE pre-embed (over the CP axis),
// then runs all blocks on the local [B, T_local] shard. No wpe (RoPE is applied
// inside CP). Weight-tied lm_head.
// =============================================================================
class GPT : public nn::Module {
public:
    ModelConfig cfg;
private:
    InitRng init_rng_;
public:
    nn::Embedding wte;
    Tensor        cos_sin_cache;
    // ADDITIVE (CREAM): full YaRN-scaled cos/sin cache of length scaled_max, built
    // ONLY when cream_mode != "off". Used solely as the gather SOURCE for per-step
    // relabeled caches; never passed to the kernel directly. Invalid/empty when off.
    Tensor        cream_full_cache_;
    std::vector<std::shared_ptr<CausalGQA>> attn;
    std::vector<std::shared_ptr<SwiGLUMLP>> mlp;
    nn::RMSNorm   norm_f;
    std::shared_ptr<nn::Linear> lm_head;
    // CP sharding coordinate (over the CP mesh axis). In hybrid mode the CP axis
    // is factored into (ring_rank_, ulysses_rank_); cp_size_/cp_rank_ hold the
    // composite CP degree/coord and drive the non-hybrid single-axis sharder.
    int cp_size_, cp_rank_;
    int ring_rank_, ulysses_rank_;

    GPT(const ModelConfig& c, DeviceIndex dev, const DeviceMesh& mesh,
        std::shared_ptr<ProcessGroupNCCL> cp_pg, int cp_size, int cp_rank,
        std::shared_ptr<ProcessGroupNCCL> ulysses_pg = nullptr,
        int ring_rank = 0, int ulysses_rank = 0,
        uint64_t seed = 1234)
        : cfg(c),
          init_rng_(seed),
          wte(static_cast<int>(c.vocab_size), static_cast<int>(c.d_model)),
          cos_sin_cache(autograd::build_rope_cache(c.context_length, c.head_dim,
                                                   static_cast<float>(c.rope_theta), dev)),
          norm_f(c.d_model, c.rms_eps),
          cp_size_(cp_size), cp_rank_(cp_rank),
          ring_rank_(ring_rank), ulysses_rank_(ulysses_rank)
    {
        init_embedding(wte, 0.02f, init_rng_);
        wte.to(dev);
        norm_f.to(dev);

        // CREAM (additive): build the full YaRN-scaled cache (length scaled_max) as
        // the gather source. build_rope_cache reads YARN_SCALE from env, so this row
        // `pos` holds the YaRN-interpolated cos/sin for absolute position `pos`. Only
        // built when CREAM is on; the base cos_sin_cache above is left untouched.
        if (c.cream_mode != "off") {
            cream_full_cache_ = autograd::build_rope_cache(
                c.cream_scaled_max, c.head_dim,
                static_cast<float>(c.rope_theta), dev);
        }

        for (int i = 0; i < c.n_layers; ++i) {
            auto a = std::make_shared<CausalGQA>(c, dev, cos_sin_cache, init_rng_, mesh, cp_pg, ulysses_pg);
            auto m = std::make_shared<SwiGLUMLP>(c, dev, init_rng_);
            attn.push_back(a);
            mlp.push_back(m);
            register_module(a.get());
            register_module(m.get());
        }

        if (!c.weight_tying) {
            lm_head = std::make_shared<nn::Linear>(static_cast<int>(c.d_model),
                                                   static_cast<int>(c.vocab_size), /*bias=*/false);
            init_linear(*lm_head, 0.02f, init_rng_);
            lm_head->to(dev);
        }

        register_module(wte);
        register_module(norm_f);
        if (!c.weight_tying && lm_head) register_module(lm_head.get());
    }

    // idx [B, T] (full, UInt16/Int) -> logits [B, T_local, vocab] on the CP shard.
    Tensor forward(const Tensor& idx) override {
        // Shard the sequence over the CP axis (HeadTail for ring, contiguous for
        // ulysses, composed HeadTail-over-ring + contiguous-over-ulysses for
        // hybrid). Targets are sharded with the SAME perm in the loss (main).
        Tensor empty_y;
        ShardedInputs sh;
        if (cfg.hybrid_mode) {
            const int ring_size = cfg.cp_size / cfg.ulysses_size;
            sh = shard_sequence_pre_embed_hybrid(
                idx, empty_y, cfg.T, ring_size, ring_rank_,
                cfg.ulysses_size, ulysses_rank_, idx.device());
        } else {
            sh = shard_sequence_pre_embed(
                idx, empty_y, cfg.T, cp_size_, cp_rank_, /*load_balance=*/cfg.ring_mode,
                idx.device());
        }

        Tensor x = wte.forward(sh.idx_local);            // [B, T_local, d_model] (NO wpe)
        for (int i = 0; i < cfg.n_layers; ++i) {
            x = attn[i]->forward(x);
            x = mlp[i]->forward(x);
        }
        x = norm_f.forward(x);

        if (cfg.weight_tying) {
            Tensor w_t = autograd::transpose(wte.weight, 0, 1);   // [d_model, vocab]
            return autograd::matmul(x, w_t);                      // [B, T_local, vocab]
        }
        return lm_head->forward(x);
    }

    // CREAM (additive): generate manipulated RoPE position LABELS for this step and
    // install a gathered cos/sin cache on EVERY CP layer. The gather is done ONCE
    // here and fanned out to all layers -- do NOT move it inside a per-layer loop
    // (it would cost n_layers x for nothing). No-op when cream_mode=="off"; call
    // before forward() when CREAM is active. Single-GPU (CP_SIZE=1): local_idx==pos,
    // so cream_cache[local_idx] rotates each token at its label L[local_idx].
    void install_cream_positions(uint64_t seed) {
        if (cfg.cream_mode == "off") return;
        const int T = static_cast<int>(cfg.T);
        const int factor = cfg.cream_factor;
        std::vector<int64_t> L;
        if (cfg.cream_mode == "cream")
            L = OwnTensor::cp::cream::generate_cream_positions(T, factor, cfg.cream_sigma, seed);
        else if (cfg.cream_mode == "pose")
            L = OwnTensor::cp::cream::generate_pose_positions(T, factor, seed);
        else if (cfg.cream_mode == "randpos")
            L = OwnTensor::cp::cream::generate_randpos_positions(T, factor, seed);
        else
            throw std::runtime_error("unknown CP_CREAM_MODE: " + cfg.cream_mode);

        // [T, hd] gather index: row i = L[i] broadcast across the hd columns so
        // gather(dim=0) selects whole cache rows: out[i][j] = full_cache[L[i]][j].
        const int64_t hd = cfg.head_dim;
        Tensor idx_cpu(Shape{{static_cast<int64_t>(T), hd}},
                       TensorOptions().with_dtype(Dtype::Int64));
        int64_t* ip = static_cast<int64_t*>(idx_cpu.data());
        for (int i = 0; i < T; ++i)
            for (int64_t j = 0; j < hd; ++j)
                ip[static_cast<int64_t>(i) * hd + j] = L[static_cast<size_t>(i)];
        Tensor idx_dev = idx_cpu.to(cream_full_cache_.device());

        Tensor cream_cache = OwnTensor::gather(cream_full_cache_, /*dim=*/0, idx_dev);
        for (auto& a : attn) a->cp_->set_rope_cache(cream_cache);
    }

    // CREAM diagnostics: the middle-block START position sampled for a seed, so the
    // training loop can log its distribution (verification step 4 -- confirm the
    // truncated-Gaussian shape). head==tail, so head = the leading run [0,1,2,...];
    // the middle block starts at the first label that breaks that run. -1 if not
    // cream mode.
    int64_t cream_middle_start(uint64_t seed) const {
        if (cfg.cream_mode != "cream") return -1;
        auto L = OwnTensor::cp::cream::generate_cream_positions(
            static_cast<int>(cfg.T), cfg.cream_factor, cfg.cream_sigma, seed);
        size_t head = 0;
        while (head < L.size() && L[head] == static_cast<int64_t>(head)) ++head;
        return (head < L.size()) ? L[head] : -1;
    }
};

// =============================================================================
// Cosine LR schedule (unchanged).
// =============================================================================
// Cosine LR with warmup. `warmup_start` anchors the warmup ramp: 0 for a fresh
// run, or the resume step for a 2-stage re-warmup on context extension (ramp
// [warmup_start, warmup_start+warmup), then cosine-decay to min_lr by max_steps).
// `max_lr` is the (possibly reduced) peak reached at the top of the ramp.
static float get_lr(int step, float max_lr, float min_lr, int warmup, int max_steps,
                    int warmup_start = 0) {
    if (max_lr < min_lr) max_lr = min_lr;                    // guard: absurdly low peak -> constant min
    const int local = step - warmup_start;                  // steps since the (re)warmup anchor
    if (warmup > 0 && local < warmup)
        return max_lr * static_cast<float>(local + 1) / static_cast<float>(warmup);
    if (step >= max_steps) return min_lr;
    const int   decay_start = warmup_start + warmup;
    float denom = static_cast<float>(max_steps - decay_start);
    if (denom <= 0.0f) denom = 1.0f;                         // guard: warmup window past max_steps
    const float decay = static_cast<float>(step - decay_start) / denom;
    const float coeff = 0.5f * (1.0f + std::cos(M_PI * decay));
    return min_lr + coeff * (max_lr - min_lr);
}

static int64_t env_i64(const char* k, int64_t d) {
    const char* v = std::getenv(k); return v ? std::atoll(v) : d;
}

// =============================================================================
// Training
// =============================================================================
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    const bool is_master = (rank == 0);

    // Only rank 0 writes to stdout. Redirect std::cout on every other rank to a
    // discard buffer so un-gated library chatter (CheckpointManager auto-load, CP
    // ring/overlap notices, etc.) is emitted once instead of world_size times.
    // std::cerr is left intact so genuine errors/warnings from any rank still show.
    static NullBuf global_null_buf;
    if (!is_master) std::cout.rdbuf(&global_null_buf);

    if (is_master)
        std::cout << "=== Llama Context Parallel Training Script (bluscriptCP) ==="
                  << std::endl;

    cudaSetDevice(rank);
    DeviceIndex device(Device::CUDA, rank);

    ModelConfig cfg;

    // ---- context-parallel topology from env ----
    cfg.cp_size = static_cast<int>(env_i64("CP_SIZE", world_size));  // default CP-only
    if (const char* m = std::getenv("CP_ATTN_MODE")) cfg.attn_mode = m;
    cfg.ring_mode   = (cfg.attn_mode == "ring");
    cfg.hybrid_mode = (cfg.attn_mode == "hybrid");
    cfg.ulysses_size = static_cast<int>(env_i64("CP_ULYSSES_SIZE", 1));
    if (is_master) {
        if (cfg.hybrid_mode)
            std::cout << "[CP attn mode] HYBRID (Ulysses-inner all-to-all + Ring-outer "
                         "P2P); HeadTail load-balance on the ring axis\n";
        else if (!cfg.ring_mode)
            std::cout << "[CP attn mode] ULYSSES (all-to-all); load_balancing forced "
                         "off (contiguous sharding)\n";
    }
    std::string rotator_label = "p2p";
    if (const char* e = std::getenv("CP_ROTATOR")) {
        std::string rv = e;
        if      (rv == "p2p"       || rv == "P2P")       { cfg.rotator = RotatorType::P2P;       rotator_label = "p2p"; }
        else if (rv == "allgather" || rv == "AllGather") { cfg.rotator = RotatorType::AllGather; rotator_label = "allgather"; }
        else                                             { cfg.rotator = RotatorType::AlltoAll; rotator_label = "alltoall"; }
    }
    // Under pure Ulysses the ring rotator is unused (all-to-all attention); tag the
    // snapshot with "ulysses" so it matches the sweep harness's filename. Hybrid
    // DOES use the ring rotator (outer axis), so it keeps the rotator label.
    if (!cfg.ring_mode && !cfg.hybrid_mode) rotator_label = "ulysses";

    // ---- memory-occupancy probe (mirrors gpt2_cp_test.cpp) ----
    //   CP_MEM_PROBE=1     : run CP_MEM_PROBE_STEPS steps (grad_accum forced to 1),
    //                        skip validation + checkpointing, snapshot nvidia-smi +
    //                        cudaMemGetInfo + allocator stats after the last step,
    //                        then exit. Used by Tests/bluscriptcp/mem_scaling_sweep.sh.
    //   CP_MEM_PROBE_STEPS : steps in probe mode (default 2; must be >=2 so the
    //                        optimizer-state allocation is counted).
    //   CP_MODEL_LABEL     : free-text label embedded in the snapshot (e.g. "48M").
    const bool mem_probe = std::getenv("CP_MEM_PROBE") != nullptr &&
                           std::atoi(std::getenv("CP_MEM_PROBE")) != 0;

    // ---- long-context PPL-vs-position eval mode (additive, forward-only) ----
    //   CP_EVAL_PPL=1  : load the checkpoint (CP_CKPT_RESUME=<run>), teacher-force
    //                    over CP_EVAL_WINDOWS windows of length T from the given
    //                    split, write pos,count,mean_nll,ppl to CP_EVAL_OUT, exit.
    //                    Requires NP=1 / CP_SIZE=1 so T_local==T (positions are
    //                    global -> no cross-rank NLL gather). Skips training,
    //                    checkpoint saving, and training-log/config writes.
    //   Build the base-vs-extended comparison by running it per checkpoint+YaRN:
    //     base ckpt + YARN_SCALE=1              (naive extrapolation @ T)
    //     extended ckpt + YARN_SCALE=4 ORIG=... (YaRN @ T)
    const bool eval_ppl_mode = std::getenv("CP_EVAL_PPL") != nullptr &&
                               std::atoi(std::getenv("CP_EVAL_PPL")) != 0;
    const int mem_probe_steps =
        std::getenv("CP_MEM_PROBE_STEPS") ? std::atoi(std::getenv("CP_MEM_PROBE_STEPS")) : 2;
    const std::string mem_label =
        std::getenv("CP_MODEL_LABEL") ? std::getenv("CP_MODEL_LABEL") : "bluscriptcp";
    // Optional config overrides (smoke tests / scaling).
    cfg.T = cfg.context_length = env_i64("CP_T", cfg.context_length);
    cfg.d_model   = env_i64("CP_N_EMBD",  cfg.d_model);
    cfg.n_layers  = env_i64("CP_N_LAYER", cfg.n_layers);
    cfg.q_heads   = env_i64("CP_N_HEAD",  cfg.q_heads);
    cfg.kv_heads  = env_i64("CP_N_KVHEAD", cfg.kv_heads);
    cfg.head_dim  = cfg.d_model / cfg.q_heads;
    // SwiGLU inner dim: honor CP_FFN, else scale with d_model (8/3*d_model). Was
    // previously frozen at the d=384 default (1024) regardless of CP_N_EMBD, which
    // silently undersized the FFN for any wider model in the scaling sweeps.
    cfg.ffn_hidden = env_i64("CP_FFN", (8 * cfg.d_model) / 3);
    if (const char* e = std::getenv("CP_WEIGHT_TYING")) cfg.weight_tying = (e[0] == '1');
    cfg.B            = env_i64("CP_B", cfg.B);
    cfg.global_batch = env_i64("CP_GLOBAL_BATCH", cfg.global_batch);
    cfg.max_steps    = static_cast<int>(env_i64("CP_MAX_STEPS", cfg.max_steps));
    cfg.warmup_steps = static_cast<int>(env_i64("CP_WARMUP", cfg.warmup_steps));

    // ---- checkpoint / logging run controls (env) ----
    if (const char* e = std::getenv("CP_CKPT"))      cfg.checkpointing = (e[0] == '1');
    if (const char* e = std::getenv("CP_CKPT_FREQ")) cfg.ckpt_freq = std::atoi(e);
    // Force a brand-new run number even if resumable checkpoints exist.
    const bool ckpt_new_run =
        std::getenv("CP_CKPT_NEW_RUN") && std::getenv("CP_CKPT_NEW_RUN")[0] == '1';
    // Resume a specific run number (>=0); -1 = auto-resume latest incomplete run.
    int ckpt_resume_run = -1;
    if (const char* e = std::getenv("CP_CKPT_RESUME")) ckpt_resume_run = std::atoi(e);

    // 2-stage re-warmup for context extension: on a RESUMING run, re-anchor a fresh
    // warmup ramp at the resume step (the base-context schedule's warmup is absolute
    // from step 0, so it cannot ramp again). CP_REWARMUP=<steps> (0 = off; only
    // applies when start_step>0). CP_REWARMUP_PEAK=<fraction of max_lr> gives a
    // reduced peak for the long-context phase (default 1.0 = same peak).
    const int rewarmup_steps = static_cast<int>(env_i64("CP_REWARMUP", 0));
    float rewarmup_peak_frac = 1.0f;
    if (const char* e = std::getenv("CP_REWARMUP_PEAK")) rewarmup_peak_frac = std::atof(e);

    // ---- LOUD invariant asserts (silent-failure guards) ----
    auto die = [&](const std::string& msg) {
        if (is_master) std::cerr << "ERROR: " << msg << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    };
    if (world_size % cfg.cp_size != 0)
        die("world_size=" + std::to_string(world_size) + " % CP_SIZE=" + std::to_string(cfg.cp_size) + " != 0");
    const int cp_size = cfg.cp_size;
    const int dp_size = world_size / cp_size;

    // ---- CREAM context-extension (additive; env-resolved; single-GPU only) ----
    if (const char* e = std::getenv("CP_CREAM_MODE"))  cfg.cream_mode  = e;
    if (const char* e = std::getenv("CP_CREAM_SIGMA")) cfg.cream_sigma = std::atof(e);
    if (const char* e = std::getenv("CP_CREAM_SEED"))
        cfg.cream_seed = static_cast<uint64_t>(std::strtoull(e, nullptr, 10));
    const char* cream_log_path = std::getenv("CP_CREAM_LOG");  // optional dist-check CSV
    const bool cream_on = (cfg.cream_mode != "off");
    if (cream_on) {
        // factor == YARN_SCALE (locked); scaled_max == factor * T. YaRN must be ON.
        const double yarn_scale =
            std::getenv("YARN_SCALE") ? std::atof(std::getenv("YARN_SCALE")) : 1.0;
        cfg.cream_factor = static_cast<int>(yarn_scale + 0.5);
        cfg.cream_scaled_max = static_cast<int64_t>(cfg.cream_factor) * cfg.T;
        if (cfg.cream_mode != "cream" && cfg.cream_mode != "pose" &&
            cfg.cream_mode != "randpos")
            die("CP_CREAM_MODE must be off|cream|pose|randpos, got '" + cfg.cream_mode + "'");
        if (cp_size != 1)
            die("CREAM is single-GPU only for now: requires CP_SIZE=1 (got " +
                std::to_string(cp_size) + ")");
        if (yarn_scale <= 1.0)
            die("CREAM requires YARN_SCALE > 1 (cache must be YaRN-scaled, got " +
                std::to_string(yarn_scale) + ")");
        if (cfg.cream_factor < 2)
            die("CREAM requires integer YARN_SCALE >= 2 (got factor=" +
                std::to_string(cfg.cream_factor) + ")");
        if (cfg.cream_scaled_max % cfg.T != 0)
            die("CREAM: scaled_max must be a multiple of T");
        if (cfg.cream_scaled_max < cfg.T)
            die("CREAM: scaled_max < T");
    }

    // ---- CP_DUMP_ROPE: dump the YaRN rope cache for the parity gate (plan 0.3) ----
    // Writes autograd::build_rope_cache(seq_len, head_dim, rope_theta) as raw float32
    // [seq_len, head_dim] (cols [0:half]=cos*mscale, [half:head_dim]=sin*mscale, the
    // YARNOps layout) so scripts/convert_ckpt/rope_parity.py --cp-cache can diff it
    // against HF Qwen3 YaRN. YARN_SCALE / YARN_ORIG_MAXPOS are read by build_rope_cache
    // from env. Needs no model or checkpoint; forward-only, exits. Placed BEFORE the
    // T!=context_length guard so CP_T need not be set for a dump.
    if (const char* rope_path = std::getenv("CP_DUMP_ROPE")) {
        const int64_t seqlen = env_i64("CP_DUMP_ROPE_SEQLEN", 8192);
        Tensor cache = autograd::build_rope_cache(seqlen, cfg.head_dim,
                                                  static_cast<float>(cfg.rope_theta), device);
        if (is_master) {
            Tensor c = cache.as_type(Dtype::Float32).to_cpu();
            std::ofstream rf(rope_path, std::ios::binary);
            rf.write(reinterpret_cast<const char*>(c.data<float>()),
                     static_cast<std::streamsize>(c.numel()) * sizeof(float));
            std::cout << "[dump-rope] wrote YaRN cache [" << seqlen << "," << cfg.head_dim
                      << "] raw float32 (YARN_SCALE via env) -> " << rope_path << std::endl;
        }
        MPI_Finalize();
        return 0;
    }

    // The base RoPE cache is built at context_length. With CREAM the kernel-facing
    // cache is a per-step gather of the full scaled cache, so T is legitimately
    // decoupled from context_length -- but ONLY under CREAM. The original guard is
    // preserved verbatim for the default (CREAM-off) path.
    if (!cream_on) {
        if (cfg.T != cfg.context_length)
            die("T != context_length (RoPE cache is built at context_length)");
    }
    // Hybrid: CP_SIZE = ring_size * ulysses_size. ring_size is the OUTER (P2P
    // rotation) degree, ulysses_size the INNER (all-to-all) degree.
    int ring_size = cp_size;                 // non-hybrid: ring axis == whole CP
    if (cfg.hybrid_mode) {
        if (cfg.ulysses_size < 1)
            die("CP_ULYSSES_SIZE must be >= 1");
        if (cp_size % cfg.ulysses_size != 0)
            die("CP_SIZE=" + std::to_string(cp_size) + " % CP_ULYSSES_SIZE=" +
                std::to_string(cfg.ulysses_size) + " != 0");
        ring_size = cp_size / cfg.ulysses_size;
        // Ring HeadTail on the ring axis: T divisible by 2*ring_size.
        if (cfg.T % (2 * ring_size) != 0)
            die("T=" + std::to_string(cfg.T) + " not divisible by 2*ring_size=" +
                std::to_string(2 * ring_size) + " (hybrid HeadTail)");
        // Contiguous inner Ulysses slice of each ring block.
        if ((cfg.T / ring_size) % cfg.ulysses_size != 0)
            die("(T/ring_size)=" + std::to_string(cfg.T / ring_size) +
                " not divisible by CP_ULYSSES_SIZE=" + std::to_string(cfg.ulysses_size) +
                " (hybrid inner slice)");
        // Fused-RoPE kernel tile: the HeadTail sub-chunk reaching the kernel is
        // T/(2*ring_size); it MUST be a multiple of 32 or gqa_fused_rope_cp aborts.
        if ((cfg.T / (2 * ring_size)) % 32 != 0)
            die("T/(2*ring_size)=" + std::to_string(cfg.T / (2 * ring_size)) +
                " not a multiple of 32 (fused-RoPE tile); increase CP_T so "
                "T % (64*ring_size) == 0");
        // Ulysses head split needs both q and kv heads divisible by U. The U>kv
        // (kv-replication) path is NOT supported in hybrid v1.
        if (cfg.q_heads % cfg.ulysses_size != 0)
            die("q_heads=" + std::to_string(cfg.q_heads) +
                " not divisible by CP_ULYSSES_SIZE=" + std::to_string(cfg.ulysses_size));
        if (cfg.kv_heads % cfg.ulysses_size != 0)
            die("kv_heads=" + std::to_string(cfg.kv_heads) +
                " not divisible by CP_ULYSSES_SIZE=" + std::to_string(cfg.ulysses_size) +
                " (hybrid does not yet support U > kv_heads replication)");
        // Advisory fast-domain guard (perf, not correctness): warn if a Ulysses
        // group would straddle a fast-interconnect island. F defaults to
        // NO_GPUS_PER_NODE / cudaGetDeviceCount; set it to a NUMA/NVLink-island
        // size to place Ulysses intra-island and Ring across it. NOT fatal.
        int fast_domain = 0;
        if (const char* e = std::getenv("NO_GPUS_PER_NODE")) fast_domain = std::atoi(e);
        if (fast_domain <= 0) { int gc = 1; cudaGetDeviceCount(&gc); fast_domain = gc; }
        if (fast_domain > 0 && (fast_domain % cfg.ulysses_size) != 0 && is_master)
            std::cerr << "[CP WARN] fast-domain size (" << fast_domain
                      << ") % CP_ULYSSES_SIZE (" << cfg.ulysses_size
                      << ") != 0: a Ulysses group straddles a fast-interconnect "
                         "boundary; the intra-node all-to-all may land on a slower "
                         "link. This is a performance hint, not a correctness error."
                      << std::endl;
    } else if (cfg.ring_mode) {
        if (cfg.T % (2 * cp_size) != 0)
            die("T=" + std::to_string(cfg.T) + " not divisible by 2*CP_SIZE=" + std::to_string(2*cp_size) + " (HeadTail)");
    } else {
        if (cfg.T % cp_size != 0)
            die("T=" + std::to_string(cfg.T) + " not divisible by CP_SIZE=" + std::to_string(cp_size) + " (contiguous)");
    }
    if (cfg.head_dim != 64 && cfg.head_dim != 128)
        die("head_dim must be 64 or 128 (fused kernel)");
    if (cfg.q_heads % cfg.kv_heads != 0)
        die("q_heads must be a multiple of kv_heads");

    // Probe mode: one micro-batch of B*T*dp_size tokens (grad_accum=1) captures the
    // per-microstep peak faithfully (the accum loop is sequential), runs fast, and
    // skips checkpointing/validation. max_steps>=2 so optimizer state is allocated.
    if (mem_probe) {
        cfg.global_batch  = cfg.B * cfg.T * static_cast<int64_t>(dp_size);
        cfg.max_steps     = mem_probe_steps;
        cfg.warmup_steps  = 1;
        cfg.checkpointing = false;
    }

    // Tokens per optimizer step: CP shares the batch (no data factor); DP multiplies.
    const int64_t tokens_per_micro = cfg.B * cfg.T * static_cast<int64_t>(dp_size);
    if (cfg.global_batch % tokens_per_micro != 0)
        die("global_batch not divisible by B*T*dp_size (" + std::to_string(tokens_per_micro) + ")");
    const int GRAD_ACCUM = static_cast<int>(cfg.global_batch / tokens_per_micro);

    // ---- process groups: the mesh builds the WORLD PG (ncclCommSplit parent)
    // + per-axis sub-groups. Reuse the mesh's world PG for DataParallel — a second
    // init_process_group would throw (BluTrain's PG registry is a singleton).
    //   non-hybrid: 2-D {dp, cp}  -> cp_pg = axis 1.
    //   hybrid (default, dp outer): 3-D {dp, ring, ulysses}. Ulysses is the fastest
    //     axis (U consecutive ranks); RING stride = U; DP stride = ring*U. Good when
    //     you want DP to cross the slow boundary (e.g. inter-node) and ring+ulysses
    //     to stay on the fast domain.
    //   hybrid + CP_RING_OUTER=1: 3-D {ring, dp, ulysses}. RING stride = dp*U so a
    //     ring group spans across the coarse boundary while Ulysses AND DDP stay on
    //     the fast domain. Use this to put the RING (P2P, overlaps) across NUMA/nodes
    //     and keep the bulk all-to-all + all-reduce intra-domain. Grouping is by rank
    //     index; rank r binds to GPU (r % gpus_per_node), so on one node the logical
    //     groups land on physical GPUs in order. ----
    const bool ring_outer =
        cfg.hybrid_mode && std::getenv("CP_RING_OUTER") &&
        std::getenv("CP_RING_OUTER")[0] == '1';
    std::vector<int> ranks_vec(world_size);
    for (int i = 0; i < world_size; ++i) ranks_vec[i] = i;
    // Axis indices into the mesh (depend on the chosen ordering).
    int dp_axis = 0, ring_axis = 1, uly_axis = 2;
    std::vector<int> mesh_shape;
    if (!cfg.hybrid_mode) {
        mesh_shape = {dp_size, cp_size};                    // {dp, cp}
    } else if (ring_outer) {
        mesh_shape = {ring_size, dp_size, cfg.ulysses_size}; // {ring, dp, ulysses}
        ring_axis = 0; dp_axis = 1; uly_axis = 2;
    } else {
        mesh_shape = {dp_size, ring_size, cfg.ulysses_size}; // {dp, ring, ulysses}
        dp_axis = 0; ring_axis = 1; uly_axis = 2;
    }
    DeviceMesh mesh(mesh_shape, ranks_vec);
    std::shared_ptr<ProcessGroupNCCL> pg = mesh.world_pg();  // global (all ranks), for DDP
    std::shared_ptr<ProcessGroupNCCL> cp_pg, ulysses_pg;
    int cp_rank = 0, dp_rank = mesh.get_dim_rank(dp_axis);
    int ring_rank = 0, ulysses_rank = 0;
    if (cfg.hybrid_mode) {
        cp_pg        = mesh.get_process_group(ring_axis);   // RING sub-group (ContextParallel pg_)
        ulysses_pg   = mesh.get_process_group(uly_axis);    // Ulysses sub-group (inner all-to-all)
        ring_rank    = mesh.get_dim_rank(ring_axis);
        ulysses_rank = mesh.get_dim_rank(uly_axis);
        cp_rank      = ring_rank * cfg.ulysses_size + ulysses_rank;  // composite CP coord
        if (is_master)
            std::cout << "[CP hybrid mesh] order="
                      << (ring_outer ? "{ring,dp,ulysses} (RING outer -> crosses coarse boundary)"
                                     : "{dp,ring,ulysses} (DP outer)")
                      << "  ring=" << ring_size << " dp=" << dp_size
                      << " ulysses=" << cfg.ulysses_size << std::endl;
    } else {
        cp_pg     = mesh.get_process_group(1);     // CP ring/ulysses sub-group
        cp_rank   = mesh.get_dim_rank(1);
        ring_rank = cp_rank;                        // whole CP axis is the ring axis
    }

    if (is_master) {
        std::cout << "================ Configuration (bluscriptCP) ================\n"
                  << "  d_model/layers  : " << cfg.d_model << " / " << cfg.n_layers << "\n"
                  << "  q/kv heads      : " << cfg.q_heads << " / " << cfg.kv_heads
                  << "  (G=" << (cfg.q_heads / cfg.kv_heads) << ", hd=" << cfg.head_dim << ")\n"
                  << "  ffn_hidden      : " << cfg.ffn_hidden << " (SwiGLU)\n"
                  << "  context_length  : " << cfg.context_length << "\n"
                  << "  vocab_size      : " << cfg.vocab_size << "  tying=" << cfg.weight_tying << "\n"
                  << "  B x T           : " << cfg.B << " x " << cfg.T << "\n"
                  << "  world_size      : " << world_size
                  << "  (dp=" << dp_size << " x cp=" << cp_size
                  << (cfg.hybrid_mode ? " [ring=" + std::to_string(ring_size) +
                                        " x ulysses=" + std::to_string(cfg.ulysses_size) + "]"
                                      : "")
                  << ")\n"
                  << "  CP_ATTN_MODE    : " << cfg.attn_mode
                  << "  (load_balance=" << ((cfg.ring_mode || cfg.hybrid_mode) ? "HeadTail" : "contiguous") << ")\n"
                  << "  global_batch    : " << cfg.global_batch << "  (grad_accum=" << GRAD_ACCUM << ")\n"
                  << "  max_lr/min_lr   : " << cfg.max_lr << " / " << cfg.min_lr << "\n"
                  << "  warmup/steps    : " << cfg.warmup_steps << " / " << cfg.max_steps << "\n"
                  << "  checkpointing   : " << (cfg.checkpointing ? "true" : "false")
                  << "  (ckpt_freq=" << cfg.ckpt_freq << ", dir=" << cfg.ckpt_dir << ")\n"
                  << "=============================================================" << std::endl;
    }

    if (is_master) std::cout << "Initializing model on CUDA device " << rank << "..." << std::endl;
    GPT model(cfg, device, mesh, cp_pg, cp_size, cp_rank,
              ulysses_pg, ring_rank, ulysses_rank, /*seed=*/1234);

    auto params = model.parameters();
    int64_t num_params = 0;
    for (auto& p : params) num_params += p.numel();
    if (is_master) {
        std::cout << "Parameters: " << num_params << "\n";
        std::cout << "Parameters per GPU: " << num_params << "\n";
        std::cout << "max_steps: " << cfg.max_steps << "\n";
        std::cout << "warmup_steps: " << cfg.warmup_steps << std::endl;
    }

    // ---- DDP wrapper KEPT over the GLOBAL pg: init-broadcast + param-grad
    // all-reduce over ALL dp*cp ranks (correct group for globally-replicated
    // params). The CP ring adds only the attention-activation grad sum over
    // cp_pg (disjoint from DDP's leaf-param reduction). ----
    std::unique_ptr<DataParallel> ddp_model;
    if (world_size > 1) {
        DDP_Options ddp_opts;
        ddp_opts = ddp_opts
            .with_process_group(pg)
            .with_world_size(world_size)
            .with_bucket_data(/*bucket=*/true, /*bucket_size_bytes=*/25 * 1024 * 1024)
            .with_grad_view(true);
        ddp_model = std::make_unique<DataParallel>(&model, ddp_opts, /*init_sync=*/true);
    }

    // ---- optimizer (single AdamW, decoupled wd applied manually to dim>=2) ----
    std::vector<Tensor> decay_params;
    for (const auto& p : params) if (p.shape().dims.size() >= 2) decay_params.push_back(p);
    nn::AdamW optimizer(params, cfg.max_lr, 0.9f, 0.95f, 1e-8f, /*weight_decay=*/0.0f);

    // ---- data: shard the BATCH over the DP axis; all CP ranks in a group share it ----
    const std::string data_root = std::getenv("CP_DATA_ROOT")
        ? std::getenv("CP_DATA_ROOT") : "Data_Loader/Data";
    DataLoaderLite train_loader(static_cast<int>(cfg.B), static_cast<int>(cfg.T),
                                dp_rank, dp_size, "train", data_root, is_master, 100000000, rank);
    DataLoaderLite val_loader(static_cast<int>(cfg.B), static_cast<int>(cfg.T),
                              dp_rank, dp_size, "val", data_root, is_master, 100000000, rank);

    Tensor grad_scale = Tensor::full(Shape{{1}}, TensorOptions().with_device(device),
                                     1.0f / static_cast<float>(GRAD_ACCUM));
    Tensor loss_accum_gpu = Tensor::zeros(Shape{{1}}, TensorOptions().with_device(device));
    float last_val_loss = -1.0f;

    // ── Run-numbered logging + checkpoints (mirrors gpt2_cp_test.cpp) ─────────
    // run_number governs BOTH the CSV log index (CP_BluScreipt_Training_logs/CP_Training_log
    // <N>.csv) AND the checkpoint filename prefix (blumodelcp_run<N>_step_<S>.ckpt).
    // Resolved on rank 0 (needs filesystem scans), then broadcast so every rank
    // builds the same prefix. When resuming, run_number is kept (log is appended).
    std::string log_filename, config_filename;
    std::ofstream log_file;
    int  run_number    = 0;
    bool ckpt_resuming = false;   // rank-0 decision; broadcast below

    if (is_master) {
        std::filesystem::create_directories("CP_BluScreipt_Training_logs");
        int next_free_log = 1;
        while (std::filesystem::exists("CP_BluScreipt_Training_logs/CP_Training_log" +
                                       std::to_string(next_free_log) + ".csv"))
            next_free_log++;

        // Scan the checkpoint dir for <prefix>_run<K>_step_<S>.ckpt, tracking each
        // run's highest step; a run is "complete" once its top step >= max_steps.
        int resume_run = -1, resume_top_step = -1;
        if (cfg.checkpointing && !ckpt_new_run) {
            std::map<int, int> run_top_step;  // run K -> highest step S
            if (std::filesystem::exists(cfg.ckpt_dir)) {
                for (const auto& de : std::filesystem::directory_iterator(cfg.ckpt_dir)) {
                    if (!de.is_regular_file()) continue;
                    const std::string fn = de.path().filename().string();
                    const std::string p1 = cfg.ckpt_prefix + "_run", p2 = "_step_", p3 = ".ckpt";
                    if (fn.rfind(p1, 0) != 0) continue;
                    size_t spos = fn.find(p2);
                    if (spos == std::string::npos) continue;
                    if (fn.size() < p3.size() ||
                        fn.compare(fn.size() - p3.size(), p3.size(), p3) != 0)
                        continue;
                    try {
                        int k = std::stoi(fn.substr(p1.size(), spos - p1.size()));
                        int s = std::stoi(fn.substr(spos + p2.size(),
                                                    fn.size() - p3.size() - (spos + p2.size())));
                        auto it = run_top_step.find(k);
                        if (it == run_top_step.end() || s > it->second) run_top_step[k] = s;
                    } catch (const std::exception&) { continue; }
                }
            }
            if (ckpt_resume_run >= 0) {
                auto it = run_top_step.find(ckpt_resume_run);
                if (it != run_top_step.end()) { resume_run = ckpt_resume_run; resume_top_step = it->second; }
                else std::cerr << "[Resume] CP_CKPT_RESUME=" << ckpt_resume_run
                               << " has no checkpoint; starting a new run." << std::endl;
            } else if (!run_top_step.empty()) {
                auto last = std::prev(run_top_step.end());  // highest run key
                if (last->second < cfg.max_steps) { resume_run = last->first; resume_top_step = last->second; }
                else std::cout << "[Resume] latest run " << last->first
                               << " already complete at step " << last->second
                               << "; starting a new run." << std::endl;
            }
        }
        if (resume_run >= 0) { run_number = resume_run; ckpt_resuming = true; }
        else                 { run_number = next_free_log; }

        log_filename = "CP_BluScreipt_Training_logs/CP_Training_log" + std::to_string(run_number) + ".csv";
        const bool log_exists = std::filesystem::exists(log_filename);
        std::cout << "Saving logs to: " << log_filename
                  << (ckpt_resuming ? " (resume run " : " (run ") << run_number << ")\n";

        config_filename = "CP_BluScreipt_Training_logs/CP_Training_log" + std::to_string(run_number) + "_config.txt";
        std::ofstream config_file(eval_ppl_mode ? "/dev/null" : config_filename.c_str(),
                                  (ckpt_resuming && log_exists) ? std::ios::app : std::ios::out);
        if (ckpt_resuming && log_exists)
            config_file << "\n# resumed run " << run_number << " from step " << resume_top_step << "\n";
        config_file << "Configuration (bluscriptCP):\n";
        config_file << "  d_model: " << cfg.d_model << "\n";
        config_file << "  n_layers: " << cfg.n_layers << "\n";
        config_file << "  q_heads: " << cfg.q_heads << "\n";
        config_file << "  kv_heads: " << cfg.kv_heads << "  (G=" << (cfg.q_heads / cfg.kv_heads)
                    << ", head_dim=" << cfg.head_dim << ")\n";
        config_file << "  ffn_hidden: " << cfg.ffn_hidden << "\n";
        config_file << "  context_length: " << cfg.context_length << "\n";
        config_file << "  vocab_size: " << cfg.vocab_size << "  (tying=" << cfg.weight_tying << ")\n";
        config_file << "  B: " << cfg.B << "\n";
        config_file << "  T: " << cfg.T << "\n";
        config_file << "  world_size: " << world_size << "  (dp=" << dp_size << " x cp=" << cp_size << ")\n";
        if (cfg.hybrid_mode)
            config_file << "  hybrid: ring_size=" << ring_size
                        << " ulysses_size=" << cfg.ulysses_size << "\n";
        config_file << "  CP_ATTN_MODE: " << cfg.attn_mode
                    << "  (load_balance=" << ((cfg.ring_mode || cfg.hybrid_mode) ? "HeadTail" : "contiguous") << ")\n";
        config_file << "  global_batch: " << cfg.global_batch << "\n";
        config_file << "  grad_accum_steps: " << GRAD_ACCUM << "\n";
        config_file << "  Parameters: " << num_params << "\n";
        config_file << "  Parameters per GPU: " << num_params << "\n";
        config_file << "  Max Learning Rate: " << cfg.max_lr << "\n";
        config_file << "  Min Learning Rate: " << cfg.min_lr << "\n";
        config_file << "  max_steps: " << cfg.max_steps << "\n";
        config_file << "  warmup_steps: " << cfg.warmup_steps << "\n";
        config_file << "  checkpointing: " << (cfg.checkpointing ? 1 : 0) << "\n";
        if (cfg.checkpointing) config_file << "  ckpt_freq: " << cfg.ckpt_freq << "\n";
        config_file << "  run: " << run_number << "\n";
        size_t free_mem = 0, total_mem = 0; cudaMemGetInfo(&free_mem, &total_mem);
        double used_mb  = static_cast<double>(total_mem - free_mem) / (1024.0 * 1024.0);
        double total_mb = static_cast<double>(total_mem) / (1024.0 * 1024.0);
        config_file << "  GPU Memory Used (rank 0): " << std::fixed << std::setprecision(1)
                    << used_mb << " MB / " << total_mb << " MB\n";
        config_file.close();

        // Resume -> append to the same run's CSV; fresh run -> truncate + header.
        // Eval mode must not touch the training CSV -> write to /dev/null.
        const bool append_log = ckpt_resuming && log_exists;
        log_file.open(eval_ppl_mode ? "/dev/null" : log_filename.c_str(),
                      append_log ? std::ios::app : std::ios::out);
        if (!log_file.is_open()) {
            std::cerr << "ERROR: Could not open log file " << log_filename << "\n";
            std::exit(1);
        }
        if (append_log)
            log_file << "# resumed run " << run_number << " from step " << resume_top_step << "\n";
        else
            log_file << "step,train_loss,val_loss,lr,grad_norm,dt_ms,tok_per_sec,"
                        "timer_data,timer_fwd,timer_loss,timer_bwd,timer_clip,timer_optim,mem_gpu_mb\n";
        log_file << std::fixed << std::setprecision(6);
    }

    // Broadcast the run number so every CheckpointManager builds the same prefix.
    int run_number_bc = is_master ? run_number : 0;
    MPI_Bcast(&run_number_bc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    run_number = run_number_bc;

    // ---- checkpointing (master save, all-rank load; params replicated) ----
    const std::string run_prefix = cfg.ckpt_prefix + "_run" + std::to_string(run_number);
    int start_step = 0;
    CheckpointManager ckpt(cfg.ckpt_dir, run_prefix, cfg.ckpt_keep, rank,
                           /*shard_dir=*/false, /*use_async=*/false);
    if (is_master) {
        if (cfg.checkpointing)
            std::cout << "Checkpointing: ON -> " << cfg.ckpt_dir << "/" << run_prefix
                      << "_step_<N>.ckpt (every " << cfg.ckpt_freq << " steps, keep "
                      << cfg.ckpt_keep << ")\n";
        else
            std::cout << "Checkpointing: OFF\n";
    }
    if (cfg.checkpointing) {
        float resume_loss = 0.0f;
        if (ckpt.load_latest(model, optimizer, start_step, resume_loss)) {
            if (is_master)
                std::cout << "[Resume] run " << run_number << " from step " << start_step
                          << " (loss " << resume_loss << ")" << std::endl;
            const size_t batches_to_skip =
                static_cast<size_t>(start_step) * static_cast<size_t>(GRAD_ACCUM);
            if (batches_to_skip > 0) {
                // Silence the per-shard-crossing debug spam during the replay, then
                // emit a single line summarizing where the loader was fast-forwarded to.
                static NullBuf null_buf;
                std::streambuf* prev = std::cout.rdbuf(&null_buf);
                train_loader.skip_batches(batches_to_skip);
                std::cout.rdbuf(prev);
                if (is_master)
                    std::cout << "[resume] fast-forwarded data loader " << batches_to_skip
                              << " batches (" << (static_cast<size_t>(start_step) *
                                 static_cast<size_t>(cfg.global_batch))
                              << " tokens)" << std::endl;
            }
        }
    }

    // Effective LR schedule: on a resuming run with CP_REWARMUP>0, ramp a fresh
    // warmup to a (possibly reduced) peak, then cosine-decay to min_lr by
    // max_steps. Otherwise the original absolute-from-0 schedule.
    //
    // The ramp anchor MUST be a fixed absolute step (the step the extension
    // warmup first began at), NOT the current resume step -- otherwise every
    // resume of the extension re-anchors a fresh ramp and warmup restarts.
    // CP_REWARMUP_ANCHOR=<step> pins it; default (-1) falls back to start_step
    // (correct only on the very first extension resume). Read it once, then
    // reuse the SAME value on every subsequent resume of the extension run.
    const int rewarmup_anchor_env = static_cast<int>(env_i64("CP_REWARMUP_ANCHOR", -1));
    const bool  rewarmup_active = (rewarmup_steps > 0 && start_step > 0);
    const int   lr_warmup_start = rewarmup_active
                                    ? (rewarmup_anchor_env >= 0 ? rewarmup_anchor_env : start_step)
                                    : 0;
    const int   lr_warmup       = rewarmup_active ? rewarmup_steps    : cfg.warmup_steps;
    const float lr_peak         = rewarmup_active ? (rewarmup_peak_frac * cfg.max_lr) : cfg.max_lr;
    if (is_master && rewarmup_active) {
        const int steps_into_ramp = start_step - lr_warmup_start;
        std::cout << "[re-warmup] anchored at step " << lr_warmup_start << " for " << rewarmup_steps
                  << " steps -> peak " << lr_peak << " (" << rewarmup_peak_frac
                  << " x max_lr), then cosine-decay to " << cfg.min_lr
                  << " | resuming at step " << start_step << " ("
                  << (steps_into_ramp >= rewarmup_steps ? "past ramp, in cosine-decay"
                      : (steps_into_ramp < 0 ? "BEFORE anchor -- check CP_REWARMUP_ANCHOR"
                                             : "mid-ramp"))
                  << ")" << std::endl;
    }

    // Per-phase event timers (data/fwd/loss/bwd/clip/optim) + whole-step timer,
    // to produce gpt2_cp_test-style per-step [TIMING] breakdown lines.
    CudaTimer timer_step, timer_data, timer_fwd, timer_loss, timer_bwd, timer_clip, timer_optim;

    // Shard the targets with the SAME perm GPT::forward applies to the inputs
    // (single-axis for ring/ulysses, composed HeadTail-over-ring + contiguous-
    // over-ulysses for hybrid).
    auto shard_targets = [&](const Tensor& in, const Tensor& tgt) -> ShardedInputs {
        if (cfg.hybrid_mode)
            return shard_sequence_pre_embed_hybrid(
                in, tgt, cfg.T, ring_size, ring_rank, cfg.ulysses_size, ulysses_rank, device);
        return shard_sequence_pre_embed(
            in, tgt, cfg.T, cp_size, cp_rank, cfg.ring_mode, device);
    };

    // ─────────────────────────────────────────────────────────────────────────
    // CP_DUMP_NAMED: identity manifest for the ckpt2hf converter gate.
    // Writes a value fingerprint of EACH weight read from its LIVE NAMED MEMBER
    // (model.attn[i]->wK.weight, model.mlp[i]->gate_up.weight, ...), NOT by
    // re-walking model.parameters()/the serialization order. The converter
    // (scripts/convert_ckpt/ckpt2hf.py --verify-manifest) independently
    // fingerprints each raw positional .ckpt slot under the same name and
    // cross-checks, so the C++ ACTUAL parameter order is verified against the
    // Python ASSUMED order -- this is the only check that catches a same-shape
    // permutation (q/k_gamma, wK/wV, attn/mlp norm) that shape-validation can't.
    // Requires the base checkpoint to be loaded (CP_CKPT_RESUME). Forward-only;
    // exits after writing the manifest. Names are CP-native (layers.i.wK, ...) so
    // they map 1:1 to positional slots on both sides.
    // ─────────────────────────────────────────────────────────────────────────
    if (const char* dump_path = std::getenv("CP_DUMP_NAMED")) {
        auto write_fp = [](std::ostream& os, const Tensor& t) {
            const Tensor c = t.as_type(Dtype::Float32).to_cpu();
            const float* p = c.data<float>();
            const size_t n = c.numel();
            double sum = 0.0, sumsq = 0.0;
            for (size_t i = 0; i < n; ++i) { const double v = p[i]; sum += v; sumsq += v * v; }
            // Fields (must match the header): n,sum,sumsq,e_first,e_mid,e_last.
            os << n << "," << std::setprecision(17)
               << sum << "," << sumsq << ","
               << (n ? static_cast<double>(p[0])       : 0.0) << ","
               << (n ? static_cast<double>(p[n / 2])   : 0.0) << ","
               << (n ? static_cast<double>(p[n - 1])   : 0.0);
        };
        if (is_master) {
            std::ofstream mf(dump_path);
            mf << "name,n,sum,sumsq,e_first,e_mid,e_last\n";
            for (int i = 0; i < cfg.n_layers; ++i) {
                const std::string pre = "layers." + std::to_string(i) + ".";
                mf << pre << "q_gamma,";   write_fp(mf, model.attn[i]->q_gamma);      mf << "\n";
                mf << pre << "k_gamma,";   write_fp(mf, model.attn[i]->k_gamma);      mf << "\n";
                mf << pre << "attn_norm,"; write_fp(mf, model.attn[i]->norm.weight);  mf << "\n";
                mf << pre << "wQ,";        write_fp(mf, model.attn[i]->wQ.weight);    mf << "\n";
                mf << pre << "wK,";        write_fp(mf, model.attn[i]->wK.weight);    mf << "\n";
                mf << pre << "wV,";        write_fp(mf, model.attn[i]->wV.weight);    mf << "\n";
                mf << pre << "wO,";        write_fp(mf, model.attn[i]->wO.weight);    mf << "\n";
                mf << pre << "mlp_norm,";  write_fp(mf, model.mlp[i]->norm.weight);   mf << "\n";
                mf << pre << "gate_up,";   write_fp(mf, model.mlp[i]->gate_up.weight); mf << "\n";
                mf << pre << "down,";      write_fp(mf, model.mlp[i]->down.weight);   mf << "\n";
            }
            mf << "wte,";    write_fp(mf, model.wte.weight);    mf << "\n";
            mf << "norm_f,"; write_fp(mf, model.norm_f.weight); mf << "\n";
            if (!cfg.weight_tying && model.lm_head) { mf << "lm_head,"; write_fp(mf, model.lm_head->weight); mf << "\n"; }
            std::cout << "[dump-named] wrote identity manifest (live named members) -> "
                      << dump_path << std::endl;
        }
        MPI_Finalize();
        return 0;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Long-context PPL-vs-position eval (forward-only; exits before training).
    // Requires cp_size==1 so model.forward returns full [B,T,vocab] and each
    // position t maps directly to the global index (no cross-rank NLL gather).
    // Teacher-forces CP_EVAL_WINDOWS contiguous windows of length T from the
    // chosen split and accumulates per-position NLL; PPL(t) = exp(mean_nll(t)).
    // ─────────────────────────────────────────────────────────────────────────
    if (eval_ppl_mode) {
        if (cp_size != 1)
            throw std::runtime_error(
                "CP_EVAL_PPL requires NP=1 / CP_SIZE=1 (per-position NLL is computed "
                "without cross-rank gather).");

        int               n_windows = static_cast<int>(env_i64("CP_EVAL_WINDOWS", 10));
        const std::string out_csv   = std::getenv("CP_EVAL_OUT")
                                        ? std::getenv("CP_EVAL_OUT") : "ppl_vs_pos.csv";
        const int     Tpos = static_cast<int>(cfg.T);
        const int     Bloc = static_cast<int>(cfg.B);
        const int64_t V    = cfg.vocab_size;

        // ─── Eval-faithfulness: optional shared token stream ────────────────────
        // When CP_EVAL_TOKENS_BIN is set we bypass val_loader and window a raw
        // int32 little-endian token file directly (input=toks[w*T .. w*T+T-1],
        // target=toks[w*T+1 .. w*T+T]) with B forced to 1, so the exact same
        // [pos] windows are scored here AND by the HF scorer (scripts/eval/
        // ppl_vs_pos.py, which also uses B=1). This makes the CP-vs-HF PPL
        // comparison read byte-identical token IDs. When UNSET, behavior below is
        // byte-identical to the previous val_loader path (additive change).
        const char*          tok_bin = std::getenv("CP_EVAL_TOKENS_BIN");
        std::vector<int32_t> eval_toks;
        const bool           bypass  = (tok_bin != nullptr);
        if (bypass) {
            std::ifstream tf(tok_bin, std::ios::binary | std::ios::ate);
            if (!tf) throw std::runtime_error(std::string("CP_EVAL_TOKENS_BIN not readable: ") + tok_bin);
            const std::streamsize nbytes = tf.tellg();
            tf.seekg(0);
            if (nbytes < 0 || (nbytes % 4) != 0)
                throw std::runtime_error("CP_EVAL_TOKENS_BIN size not a multiple of 4 (expect int32 tokens)");
            eval_toks.resize(static_cast<size_t>(nbytes) / 4);
            tf.read(reinterpret_cast<char*>(eval_toks.data()), nbytes);
            const long long max_w = (static_cast<long long>(eval_toks.size()) - 1) / Tpos;
            if (max_w <= 0)
                throw std::runtime_error("CP_EVAL_TOKENS_BIN too short for one window at T=CP_T");
            if (n_windows > max_w) {
                if (is_master)
                    std::cout << "[eval] clamping windows " << n_windows << " -> " << max_w
                              << " (stream has " << eval_toks.size() << " tokens at T=" << Tpos << ")\n";
                n_windows = static_cast<int>(max_w);
            }
            if (is_master)
                std::cout << "[eval] shared token stream: " << eval_toks.size() << " int32 tokens from "
                          << tok_bin << " (B forced to 1 for HF-scorer parity)" << std::endl;
        }

        if (is_master)
            std::cout << "[eval] PPL-vs-position: " << n_windows << " windows x B="
                      << (bypass ? 1 : Bloc) << " T=" << Tpos << " (YARN_SCALE via env) -> "
                      << out_csv << std::endl;

        std::vector<double>    sum_nll(static_cast<size_t>(Tpos), 0.0);
        std::vector<long long> cnt(static_cast<size_t>(Tpos), 0);

        autograd::NoGradGuard no_grad;
        if (!bypass) val_loader.reset();
        // CREAM (additive): install ONE fixed labeling for the whole eval so the
        // per-physical-position NLL accumulation is consistent across windows. No-op
        // when off (base cache is used, byte-identical to prior eval behavior).
        if (cream_on) model.install_cream_positions(cfg.cream_seed);
        for (int w = 0; w < n_windows; ++w) {
            // Produce (input, y_local) either from the shared token stream (B=1)
            // or from val_loader (B=cfg.B) — accumulation below uses the actual
            // per-window batch size Bcur so both paths share the same NLL code.
            Tensor input, y_local;
            int    Bcur;
            if (bypass) {
                Bcur = 1;
                std::vector<uint16_t> xh(static_cast<size_t>(Tpos)), yh(static_cast<size_t>(Tpos));
                const size_t base = static_cast<size_t>(w) * static_cast<size_t>(Tpos);
                for (int t = 0; t < Tpos; ++t) {
                    const int32_t xi = eval_toks[base + t];
                    const int32_t yi = eval_toks[base + t + 1];
                    if (xi < 0 || xi >= V)
                        throw std::runtime_error("CP_EVAL_TOKENS_BIN: input token out of [0,vocab)");
                    if (yi < 0 || yi >= 65536)
                        throw std::runtime_error("CP_EVAL_TOKENS_BIN: target token out of uint16 range");
                    xh[static_cast<size_t>(t)] = static_cast<uint16_t>(xi);
                    yh[static_cast<size_t>(t)] = static_cast<uint16_t>(yi);
                }
                input = Tensor(Shape{{1, static_cast<int64_t>(Tpos)}},
                               TensorOptions().with_dtype(Dtype::UInt16).with_device(device));
                Tensor tgt(Shape{{1, static_cast<int64_t>(Tpos)}},
                           TensorOptions().with_dtype(Dtype::UInt16).with_device(device));
                cudaMemcpy(input.data(), xh.data(), static_cast<size_t>(Tpos) * sizeof(uint16_t),
                           cudaMemcpyHostToDevice);
                cudaMemcpy(tgt.data(),   yh.data(), static_cast<size_t>(Tpos) * sizeof(uint16_t),
                           cudaMemcpyHostToDevice);
                ShardedInputs shy = shard_targets(input, tgt);      // cp_size==1 => no-op reshard
                y_local = shy.y_local;
            } else {
                Batch b = val_loader.next_batch();
                Bcur = Bloc;
                input = b.input;
                ShardedInputs shy = shard_targets(b.input, b.target);   // cp_size==1 => full [B,T]
                y_local = shy.y_local;
            }
            Tensor logits = model.forward(input);                   // [Bcur,T,V]
            // Logits: force fp32 host copy (attention returns fp32, but guard bf16).
            Tensor logits_cpu = logits.as_type(Dtype::Float32).to_cpu();
            const float* lp = logits_cpu.data<float>();

            // Targets: y_local inherits the loader dtype (UInt16), NOT int64 --
            // read it in its actual dtype or indexing walks off the buffer.
            Tensor y_cpu = y_local.to_cpu();                         // [Bcur,T]
            const Dtype ydt = y_cpu.dtype();
            const uint16_t* yp16 = (ydt == Dtype::UInt16) ? y_cpu.data<uint16_t>() : nullptr;
            const int32_t*  yp32 = (ydt == Dtype::Int32)  ? y_cpu.data<int32_t>()  : nullptr;
            const int64_t*  yp64 = (ydt == Dtype::Int64)  ? y_cpu.data<int64_t>()  : nullptr;
            if (!yp16 && !yp32 && !yp64)
                throw std::runtime_error("CP_EVAL_PPL: unexpected target dtype for y_local");

            // Per-position NLL[t] = logsumexp(logits[b,t]) - logits[b,t, target].
            // Each position t is written by one iteration only (safe to OMP over t
            // if -fopenmp is enabled; harmless no-op otherwise).
            #pragma omp parallel for schedule(dynamic)
            for (int t = 0; t < Tpos; ++t) {
                for (int bi = 0; bi < Bcur; ++bi) {
                    const size_t idx = static_cast<size_t>(bi) * Tpos + t;
                    const float* row = lp + idx * V;
                    float mx = row[0];
                    for (int64_t v = 1; v < V; ++v) if (row[v] > mx) mx = row[v];
                    double se = 0.0;
                    for (int64_t v = 0; v < V; ++v) se += std::exp(static_cast<double>(row[v] - mx));
                    const double lse = static_cast<double>(mx) + std::log(se);
                    const int64_t tgt = yp16 ? static_cast<int64_t>(yp16[idx])
                                       : yp64 ? yp64[idx]
                                              : static_cast<int64_t>(yp32[idx]);
                    if (tgt < 0 || tgt >= V) continue;              // skip padding/ignore ids
                    sum_nll[static_cast<size_t>(t)] += lse - static_cast<double>(row[tgt]);
                    cnt[static_cast<size_t>(t)]     += 1;
                }
            }
            if (is_master)
                std::cout << "[eval] window " << (w + 1) << "/" << n_windows << " done" << std::endl;
        }

        if (is_master) {
            std::ofstream f(out_csv);
            f << "pos,count,mean_nll,ppl\n";
            f << std::fixed << std::setprecision(6);
            for (int t = 0; t < Tpos; ++t) {
                const double mn = cnt[static_cast<size_t>(t)] > 0
                    ? sum_nll[static_cast<size_t>(t)] / static_cast<double>(cnt[static_cast<size_t>(t)])
                    : 0.0;
                f << t << "," << cnt[static_cast<size_t>(t)] << "," << mn << "," << std::exp(mn) << "\n";
            }
            std::cout << "[eval] wrote " << out_csv << " (" << Tpos << " positions, "
                      << n_windows << " windows)" << std::endl;
        }
        MPI_Finalize();
        return 0;
    }

    if (is_master) std::cout << "\nStarting training...\n" << std::endl;

    for (int step = start_step; step < cfg.max_steps; ++step) {
        timer_step.start_timer();

        // ---------- periodic validation (local shard loss + global avg) ----------
        if (!mem_probe && (step % cfg.val_freq == 0 || step == cfg.max_steps - 1)) {
            autograd::NoGradGuard no_grad;
            val_loader.reset();
            float val_accum = 0.0f;
            for (int v = 0; v < cfg.val_loss_steps; ++v) {
                Batch b = val_loader.next_batch();
                ShardedInputs shy = shard_targets(b.input, b.target);
                Tensor logits = model.forward(b.input);            // [B, T_local, vocab]
                Tensor loss = autograd::sparse_cross_entropy_loss(logits, shy.y_local);
                val_accum += loss.to_cpu().data<float>()[0] / static_cast<float>(cfg.val_loss_steps);
            }
            float global_val = val_accum;
            MPI_Allreduce(&val_accum, &global_val, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            global_val /= world_size;   // avg over all dp*cp ranks
            last_val_loss = global_val;
            if (is_master)
                std::cout << "validation loss: " << std::fixed
                          << std::setprecision(4) << global_val << std::endl;
        }

        // ---------- training step ----------
        double time_data = 0, time_forward = 0, time_loss = 0;
        double time_backward = 0, time_clip = 0, time_optim = 0;
        optimizer.zero_grad();
        loss_accum_gpu *= 0.0f;

        // CREAM (additive): resample position labels once per optimizer step (shared
        // across micro-batches) and install the gathered cache on all layers. No-op
        // when off. Optional CP_CREAM_LOG=<csv> records the middle-block start per
        // step for the truncated-Gaussian distribution sanity check.
        if (cream_on) {
            const uint64_t step_seed = cfg.cream_seed + static_cast<uint64_t>(step);
            model.install_cream_positions(step_seed);
            if (is_master && cream_log_path) {
                std::ofstream cf(cream_log_path, std::ios::app);
                cf << step << "," << model.cream_middle_start(step_seed) << "\n";
            }
        }

        // no_sync()/sync() gate ONLY DDP's param-grad hooks; the CP ring's cp_pg
        // comm runs every microstep (inside model.forward/backward) regardless.
        if (ddp_model) ddp_model->no_sync();
        for (int micro = 0; micro < GRAD_ACCUM; ++micro) {
            timer_data.start_timer();
            Batch batch = train_loader.next_batch();
            // Shard targets with the SAME perm GPT::forward uses on the inputs.
            ShardedInputs shy = shard_targets(batch.input, batch.target);
            time_data += timer_data.get_elapsed_seconds();

            timer_fwd.start_timer();
            Tensor logits = ddp_model ? ddp_model->forward(batch.input)
                                      : model.forward(batch.input);   // [B, T_local, vocab]
            time_forward += timer_fwd.get_elapsed_seconds();

            timer_loss.start_timer();
            Tensor loss = autograd::sparse_cross_entropy_loss(logits, shy.y_local);
            loss_accum_gpu += loss.detach();
            time_loss += timer_loss.get_elapsed_seconds();

            if (ddp_model && micro == GRAD_ACCUM - 1) ddp_model->sync();  // DDP all-reduce on last micro

            timer_bwd.start_timer();
            loss.backward(&grad_scale);
            time_backward += timer_bwd.get_elapsed_seconds();
        }

        // Logged loss: local mean over micros, then avg over all dp*cp ranks.
        float loss_accum;
        {
            float local = loss_accum_gpu.to_cpu().data<float>()[0] / static_cast<float>(GRAD_ACCUM);
            MPI_Allreduce(&local, &loss_accum, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            loss_accum /= world_size;
        }
        if (std::isnan(loss_accum) || std::isinf(loss_accum)) {
            if (is_master) { std::cerr << "ERROR: NaN/Inf loss at step " << step << std::endl; log_file.close(); }
            MPI_Abort(MPI_COMM_WORLD, 2);
        }

        timer_clip.start_timer();
        const float grad_norm = nn::clip_grad_norm_(params, cfg.grad_clip);
        time_clip = timer_clip.get_elapsed_seconds();

        const float lr = get_lr(step, lr_peak, cfg.min_lr, lr_warmup, cfg.max_steps, lr_warmup_start);
        optimizer.set_lr(lr);
        timer_optim.start_timer();
        const float wd_factor = 1.0f - lr * cfg.weight_decay;
        for (auto& p : decay_params) p *= wd_factor;
        optimizer.step();
        time_optim = timer_optim.get_elapsed_seconds();

        const double dt = timer_step.get_elapsed_seconds();
        const int64_t tokens = cfg.B * cfg.T * GRAD_ACCUM * static_cast<int64_t>(dp_size);
        const double tok_per_sec = static_cast<double>(tokens) / dt;

        // Throughput + time left + GPU memory (gpt2_cp_test-style per-step report).
        long long total_sec = static_cast<long long>((cfg.max_steps - step) * dt);
        int h = static_cast<int>(total_sec / 3600);
        int m = static_cast<int>((total_sec % 3600) / 60);
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);
        const double used_mb = static_cast<double>(total_mem - free_mem) / (1024.0 * 1024.0);

        if (is_master) {
            std::cout << "step " << std::setw(5) << step
                      << " | loss: " << std::fixed << std::setprecision(6) << loss_accum
                      << " | lr " << std::scientific << std::setprecision(4) << lr
                      << " | norm: " << std::fixed << std::setprecision(4) << grad_norm
                      << " | dt: " << std::fixed << std::setprecision(2) << (dt * 1000.0) << "ms"
                      << " | tok/sec: " << std::fixed << std::setprecision(1) << tok_per_sec
                      << " | mem: " << std::fixed << std::setprecision(0) << used_mb << "MB"
                      << " | Time Left: " << std::setfill('0') << std::setw(2) << h << " hrs : "
                      << std::setw(2) << m << " mins" << std::setfill(' ') << "\n";

            std::cout << "  [TIMING] data: " << std::fixed << std::setprecision(1)
                      << (time_data * 1000.0) << "ms"
                      << " | fwd: " << (time_forward * 1000.0) << "ms"
                      << " | loss: " << (time_loss * 1000.0) << "ms"
                      << " | bwd: " << (time_backward * 1000.0) << "ms"
                      << " | clip: " << (time_clip * 1000.0) << "ms"
                      << " | optim: " << (time_optim * 1000.0) << "ms\n";

            log_file << step << "," << loss_accum << "," << last_val_loss << "," << lr << ","
                     << grad_norm << "," << (dt * 1000.0) << "," << tok_per_sec << ","
                     << (time_data * 1000.0) << "," << (time_forward * 1000.0) << ","
                     << (time_loss * 1000.0) << "," << (time_backward * 1000.0) << ","
                     << (time_clip * 1000.0) << "," << (time_optim * 1000.0) << ","
                     << used_mb << "\n";
            log_file.flush();
        }

        // ── Memory-probe snapshot: after the final probe step, capture nvidia-smi
        //    + cudaMemGetInfo + allocator stats (live, before teardown), then stop.
        //    Same snapshot format as gpt2_cp_test.cpp so mem_scaling_table.py parses
        //    both. tag = CPP_<label>_<rotator>_T<T>_ws<ws>.txt. ────────────────────
        if (mem_probe && step == cfg.max_steps - 1) {
            cudaDeviceSynchronize();
            size_t pf = 0, pt = 0;
            cudaMemGetInfo(&pf, &pt);
            const double probe_used_mb = static_cast<double>(pt - pf) / (1024.0 * 1024.0);
            auto mstats = CachingCUDAAllocator::instance().get_stats(rank);
            const double MB = 1024.0 * 1024.0;
            const double st_reserved_mb  = mstats.reserved_current / MB;
            const double st_active_mb    = mstats.active_current / MB;
            const double st_requested_mb = mstats.allocated_current / MB;
            const double st_frag_pct     = mstats.fragmentation_ratio();
            const double st_active_peak_mb   = mstats.active_peak / MB;
            const double st_reserved_peak_mb = mstats.reserved_peak / MB;
            const double st_cached_free_mb =
                (mstats.reserved_current > mstats.active_current)
                    ? (mstats.reserved_current - mstats.active_current) / MB : 0.0;
            const std::string tag = "CPP_" + mem_label + "_" + rotator_label + "_T" +
                                    std::to_string(cfg.T) + "_ws" + std::to_string(world_size);
            std::cout << "[MEM_PROBE rank=" << rank << "] tag=" << tag
                      << " used_mb=" << std::fixed << std::setprecision(1)
                      << probe_used_mb << "\n";
            if (is_master) {
                const char* sd = std::getenv("MEM_SNAPSHOT_DIR");
                const std::string snap_dir = sd ? sd : "mem_scaling_runs";
                std::filesystem::create_directories(snap_dir);
                const std::string snap_path = snap_dir + "/" + tag + ".txt";
                std::ofstream sf(snap_path);
                sf << "# MEM PROBE SNAPSHOT  tag=" << tag << "\n";
                sf << "# impl=Cpp label=" << mem_label << " rotator=" << rotator_label
                   << " n_embd=" << cfg.d_model << " n_layer=" << cfg.n_layers
                   << " n_head=" << cfg.q_heads
                   << " weight_tying=" << (cfg.weight_tying ? 1 : 0) << "\n";
                sf << "# B=" << cfg.B << " T=" << cfg.T
                   << " cp_world_size=" << world_size << " params=" << num_params << "\n";
                sf << "# cudaMemGetInfo used_mb(rank0)=" << std::fixed
                   << std::setprecision(1) << probe_used_mb << "\n";
                sf << "# ALLOC reserved_mb=" << std::fixed << std::setprecision(1)
                   << st_reserved_mb << " active_mb=" << st_active_mb
                   << " requested_mb=" << st_requested_mb
                   << " cached_free_mb=" << st_cached_free_mb
                   << " internal_frag_pct=" << std::setprecision(2) << st_frag_pct << "\n";
                sf << "# ALLOC_PEAK active_peak_mb=" << std::fixed << std::setprecision(1)
                   << st_active_peak_mb << " reserved_peak_mb=" << st_reserved_peak_mb << "\n";
                // Machine-readable live per-GPU used MiB (parsed by the table gen).
                // nvidia-smi ignores CUDA_VISIBLE_DEVICES; keep only this run's GPUs.
                {
                    std::string smi_used;
                    std::vector<std::string> vis;
                    if (const char* cvd = std::getenv("CUDA_VISIBLE_DEVICES")) {
                        std::string s(cvd), cur;
                        for (char c : s) {
                            if (c == ',') { if (!cur.empty()) vis.push_back(cur); cur.clear(); }
                            else if (c != ' ') cur += c;
                        }
                        if (!cur.empty()) vis.push_back(cur);
                    }
                    auto allowed = [&](const std::string& idx) {
                        if (vis.empty()) return true;
                        for (auto& v : vis) if (v == idx) return true;
                        return false;
                    };
                    FILE* qp = popen("nvidia-smi --query-gpu=index,memory.used "
                                     "--format=csv,noheader,nounits 2>/dev/null", "r");
                    if (qp) {
                        char buf[256];
                        while (fgets(buf, sizeof(buf), qp)) {
                            std::string ln(buf);
                            while (!ln.empty() && (ln.back() == '\n' || ln.back() == '\r')) ln.pop_back();
                            std::string cleaned;
                            for (char c : ln) if (c != ' ') cleaned += c;
                            if (cleaned.empty()) continue;
                            const std::string idx = cleaned.substr(0, cleaned.find(','));
                            if (!allowed(idx)) continue;
                            if (!smi_used.empty()) smi_used += ";";
                            smi_used += cleaned;
                        }
                        pclose(qp);
                    }
                    sf << "# SMI_USED_MB_PER_GPU=" << smi_used << "\n";
                }
                sf << "# ---- nvidia-smi ----\n";
                sf.close();
                const std::string cmd = "nvidia-smi >> '" + snap_path + "' 2>&1";
                int rc = std::system(cmd.c_str());
                (void)rc;
                std::cout << "[MEM_PROBE] wrote snapshot " << snap_path << "\n";
            }
            MPI_Barrier(MPI_COMM_WORLD);
            break;
        }

        if (cfg.checkpointing && is_master &&
            ((step + 1) % cfg.ckpt_freq == 0 || step == cfg.max_steps - 1)) {
            try { ckpt.save(step + 1, model, optimizer, loss_accum); }
            catch (const std::exception& e) {
                std::cerr << "[WARN] checkpoint save at step " << (step + 1) << " failed: " << e.what() << std::endl;
            }
        }
        if (step == 0 || step == 1) CachingCUDAAllocator::instance().empty_cache();
    }

    if (cfg.checkpointing) {
        try { ckpt.wait_for_completion(); }
        catch (const std::exception& e) {
            if (is_master) std::cerr << "[WARN] checkpoint flush: " << e.what() << std::endl;
        }
    }
    if (is_master) { log_file.close(); std::cout << "\n=== Training complete ===" << std::endl; }

    MPI_Finalize();
    return 0;
}

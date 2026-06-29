#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <nvtx3/nvToolsExt.h>

// Tensor library includes
#include "TensorLib.h"

// NVTX wrapper (TensorLib already includes nvToolsExt.h internally)
class EmitNVTX {
public:
  EmitNVTX(const char *name) { nvtxRangePushA(name); }
  ~EmitNVTX() { nvtxRangePop(); }
};
#define emit_nvtx(name) EmitNVTX nvtx_tmp_##__LINE__(name)
#include "autograd/AutogradOps.h"
#include "autograd/operations/EmbeddingOps.h"
#include "autograd/operations/LossOps.h"
#include "checkpointing/GradMode.h"
// #include "dnn/DistributedNN.h"
#include "mlp/activation.h"
#include "nn/NN.h"
#include "nn/optimizer/Optim.h"
#include "process_group/ProcessGroupNCCL.h"
#include "process_group/device_mesh.h"
#include "profiler/AllocationTracker.h"
// DataLoader (same path as gpt2_tp_test)
#include "Data_Loader/dl_test.cpp"

// Context Parallel
#include "context_parallel/ContextParallel.h"

// #include "dnn/FusedLayerNormOp.h"

using namespace OwnTensor;

// =============================================================================
// CudaTimer
// =============================================================================

struct CudaTimer {
  cudaEvent_t start, stop;
  CudaTimer() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }
  ~CudaTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
  void start_timer() { cudaEventRecord(start); }
  float get_elapsed_ms() {
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
  }
  double get_elapsed_seconds() { return get_elapsed_ms() / 1000.0; }
};

// =============================================================================
// Configuration
// =============================================================================

struct GPTConfig {
  int64_t batch_size = 4;
  int64_t context_length = 1024;
  int64_t vocab_size = 50304;
  int64_t n_embd = 384;
  int64_t n_layers = 3;
  int64_t n_heads = 1;
  bool weight_tying = true;
  bool load_balancing = true;
  // When false: CP layers keep output local [B,T/n,C]; loss requires allreduce.
  // When true:  CP layers allgather output to full [B,T,C]; loss is scalar-identical
  //             across ranks, no allreduce needed.
  bool cp_unshard = false;
  // Ring rotator used by ContextParallel (P2P / AlltoAll / AllGather).
  RotatorType rotator = RotatorType::AlltoAll;
  // Recompute K in the ring backward instead of storing all ring-step K
  // buffers (trades compute for memory; PyTorch CP recomputes by default).
  bool recompute_k = false;
};

// =============================================================================
// Learning Rate Scheduler
// =============================================================================

float get_lr(int step, float max_lr, float min_lr, int warmup_steps,
             int max_steps) {
  if (step < warmup_steps) {
    return max_lr * static_cast<float>(step + 1) /
           static_cast<float>(warmup_steps);
  }
  if (step > max_steps) {
    return min_lr;
  }
  float decay_ratio = static_cast<float>(step - warmup_steps) /
                      static_cast<float>(max_steps - warmup_steps);
  float coeff =
      0.5f * (1.0f + std::cos(static_cast<float>(M_PI) * decay_ratio));
  return min_lr + coeff * (max_lr - min_lr);
}

// =============================================================================
// Tiktoken Decoder (via Python popen)
// =============================================================================

std::string decode_tokens_tiktoken(const std::vector<int64_t> &tokens) {
  std::string cmd =
      "python3 -c \"import tiktoken; enc = tiktoken.get_encoding('gpt2'); "
      "print(enc.decode([";
  for (size_t i = 0; i < tokens.size(); ++i) {
    if (i > 0)
      cmd += ",";
    cmd += std::to_string(tokens[i]);
  }
  cmd += "]))\" 2>/dev/null";

  std::string result;
  FILE *pipe = popen(cmd.c_str(), "r");
  if (pipe) {
    char buffer[512];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr)
      result += buffer;
    pclose(pipe);
  }
  if (!result.empty() && result.back() == '\n')
    result.pop_back();
  return result;
}

// =============================================================================
// Embedding Layer
// =============================================================================

class Embedding : public nn::Module {
public:
  Tensor weight;
  Embedding() = default;
  Embedding(int64_t vocab_size, int64_t embed_dim, DeviceIndex device,
            uint64_t seed = 1234)
      : vocab_size_(vocab_size), embed_dim_(embed_dim) {
    TensorOptions opts = TensorOptions()
                             .with_dtype(Dtype::Float32)
                             .with_device(device)
                             .with_req_grad(true);
    weight =
        Tensor::randn<float>(Shape{{vocab_size, embed_dim}}, opts, seed, 0.02f);
    register_parameter(weight);
  }

  Tensor forward(const Tensor &indices) override {
    return autograd::embedding(weight, indices);
  }

private:
  int64_t vocab_size_;
  int64_t embed_dim_;
};

// =============================================================================
// Helper: init linear weights
// =============================================================================

void init_linear_gpt2(nn::Linear &layer, float std_val = 0.02f,
                      uint64_t seed = 1234, bool req_grad = true) {
  auto shape = layer.weight.shape();
  TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32);
  Tensor init_data = Tensor::randn<float>(shape, opts, seed, std_val);
  layer.weight.copy_(init_data);
  layer.weight.set_requires_grad(req_grad);

  if (layer.bias.is_valid()) {
    Tensor bias_init = Tensor::zeros(layer.bias.shape(), opts);
    layer.bias.copy_(bias_init);
    layer.bias.set_requires_grad(req_grad);
  }
}

// =============================================================================
// Context Parallel Attention Block
// =============================================================================

class CPAttention : public nn::Module {
public:
  nn::LayerNorm ln;
  nn::Linear c_attn;
  nn::Linear c_proj;
  CudaTimer timer_attn;
  double t_attn = 0.0;

  // unshard: when false, CP output stays [B,H,T/n,D]; downstream layers get
  //          [B,T/n,C]. No AllGather is issued.
  // x is always pre-sharded [B,T/n,C] by GPT::forward before entering any layer.
  CPAttention(int64_t n_embd, int64_t n_heads, int64_t n_layers,
              DeviceIndex device, std::shared_ptr<ProcessGroupNCCL> pg,
              const DeviceMesh &mesh, uint64_t seed = 1234,
              bool load_balancing = false, bool unshard = true,
              RotatorType rotator = RotatorType::AlltoAll,
              bool recompute_k = false,
              std::shared_ptr<ContextParallel::SharedFwdRing> shared_fwd_ring =
                  nullptr)
      : ln(n_embd), c_attn(n_embd, 3 * n_embd, true),
        c_proj(n_embd, n_embd, true), n_embd_(n_embd), n_heads_(n_heads),
        head_dim_(n_embd / n_heads), unshard_(unshard),
        rank_(pg->get_rank()), world_size_(pg->get_worldsize()) {
    init_linear_gpt2(c_attn, 0.02f, seed);
    float proj_std = 0.02f / std::sqrt(2.0f * static_cast<float>(n_layers));
    init_linear_gpt2(c_proj, proj_std, seed + 1);

    float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    // load_balance=false: HeadTail permutation breaks causal masking because
    // interleaved Q positions make the simple tril mask incorrect.
    // With load_balance=false, contiguous chunks allow correct causal behavior:
    //   self-chunk: tril causal mask
    //   past chunks (source_rank < rank_): full attention
    //   future chunks (source_rank > rank_): skipped
    cp_ = std::make_shared<ContextParallel>(
        mesh, pg, attn_scale, /*is_causal=*/true, rotator,
        /*load_balance=*/load_balancing, /*recompute_k=*/recompute_k);
    // Optionally share one forward ring rotator + staging across all layers
    // (CP_SHARE_FWD_ROTATOR). Backward rotators stay per-call/independent.
    if (shared_fwd_ring) cp_->set_shared_fwd_ring(shared_fwd_ring);

    ln.to(device);
    c_attn.to(device);
    c_proj.to(device);

    register_module(ln);
    register_module(c_attn);
    register_module(c_proj);
  }

  Tensor forward(const Tensor &x) override {
    emit_nvtx("CPAttention");
    int64_t B = x.shape().dims[0];
    int64_t T = x.shape().dims[1];
    int64_t C = x.shape().dims[2];

    // Training: x is pre-sharded [B,T/n,C]. pre_sharded=true.
    //   T_out = T (T/n) when unshard_=false.
    //   T_out = T*world_size_ (full T) when unshard_=true.
    // Generation: x is full [B,T,C]. pre_sharded=false — CP shards q/k/v
    //   internally and unshards to full T. T_out = T (same as input).
    bool pre_sharded = !generation_mode_;
    int64_t T_out;
    if (generation_mode_) {
      T_out = T; // CP shards internally; output = same T as input
    } else {
      T_out = unshard_ ? T * world_size_ : T;
    }

    Tensor h = autograd::layer_norm(x, ln.weight, ln.bias,
                                     static_cast<int>(x.shape().dims[2]), ln.eps);

    Tensor qkv = c_attn.forward(h);
    std::vector<Tensor> inp = qkv.make_shards_inplace_axis(3, 2);
    Tensor q = inp[0];
    Tensor k = inp[1];
    Tensor v = inp[2];

    // Reshape to [B, H, T, D] (T is local T/n in training, full T in generation)
    q = autograd::transpose(
        autograd::reshape(q, Shape({{B, T, n_heads_, head_dim_}})), 1, 2);
    k = autograd::transpose(
        autograd::reshape(k, Shape({{B, T, n_heads_, head_dim_}})), 1, 2);
    v = autograd::transpose(
        autograd::reshape(v, Shape({{B, T, n_heads_, head_dim_}})), 1, 2);

    timer_attn.start_timer();
    // Generation: unshard=true always (need full output), pre_sharded=false.
    // Training: unshard=unshard_, pre_sharded=true.
    bool cp_unshard = generation_mode_ ? true : unshard_;
    Tensor attn_out = cp_->forward_cp(q, k, v, cp_unshard, pre_sharded);
    t_attn += timer_attn.get_elapsed_seconds();

    Tensor merged = autograd::reshape(autograd::transpose(attn_out, 1, 2),
                                      Shape({{B, T_out, C}}));

    // Raw SDPA output (block 0), before c_proj + residual — isolates the
    // attention operator from the projection. Dumps once when DUMP_FWD set.
    static bool raw_dumped = false;
    if (std::getenv("DUMP_FWD") && !raw_dumped) {
      Tensor host = merged.to_cpu().contiguous();
      std::ofstream bf("fwd_sdpa_cpp.bin", std::ios::binary);
      bf.write(reinterpret_cast<const char *>(host.data<float>()),
               host.numel() * sizeof(float));
      bf.close();
      raw_dumped = true;
      std::cout << "[DUMP_FWD] saved fwd_sdpa_cpp.bin (raw attn output, pre-c_proj)\n";
    }

    Tensor proj = c_proj.forward(merged);
    return autograd::add(x, proj);
  }

  void reset_t_attn() { t_attn = 0.0; }

  void set_unshard(bool unshard) { unshard_ = unshard; }
  void set_generation_mode(bool gen) { generation_mode_ = gen; }

private:
  int64_t n_embd_;
  int64_t n_heads_;
  int64_t head_dim_;
  bool unshard_;
  bool generation_mode_ = false;
  int rank_;
  int world_size_;
  std::shared_ptr<ContextParallel> cp_;
};

// =============================================================================
// MLP Block
// =============================================================================

class MLP : public nn::Module {
public:
  nn::LayerNorm ln;
  nn::Linear fc_up;
  nn::Linear fc_down;

  MLP(int64_t n_embd, int64_t n_layers, DeviceIndex device,
      uint64_t seed = 1234)
      : ln(n_embd), fc_up(n_embd, 4 * n_embd, true),
        fc_down(4 * n_embd, n_embd, true) {
    init_linear_gpt2(fc_up, 0.02f, seed);
    float proj_std = 0.02f / std::sqrt(2.0f * static_cast<float>(n_layers));
    init_linear_gpt2(fc_down, proj_std, seed + 1);

    ln.to(device);
    fc_up.to(device);
    fc_down.to(device);

    register_module(ln);
    register_module(fc_up);
    register_module(fc_down);
  }

  Tensor forward(const Tensor &x) override {
    emit_nvtx("MLP");
    Tensor h = autograd::layer_norm(x, ln.weight, ln.bias,
                                     static_cast<int>(x.shape().dims.back()), ln.eps);
    h = fc_up.forward(h);
    h = autograd::gelu(h);
    h = fc_down.forward(h);
    return autograd::add(x, h);
  }
};

// =============================================================================
// GPT Model with Context Parallel Attention
// =============================================================================

class GPT : public nn::Module {
public:
  GPTConfig config;
  Embedding wte;
  Embedding wpe;
  std::vector<std::shared_ptr<CPAttention>> attn_blocks;
  std::vector<std::shared_ptr<MLP>> mlp_blocks;
  nn::LayerNorm ln_f;
  std::shared_ptr<nn::Linear> lm_head;

  // Component timers
  double t_tok_emb = 0, t_pos_emb = 0, t_attn = 0;
  double t_mlp = 0, t_ln_f = 0, t_lm_head = 0;
  CudaTimer timer_tok_emb, timer_pos_emb, timer_attn_block;
  CudaTimer timer_mlp, timer_ln_f, timer_lm_head;

  GPT(GPTConfig cfg, DeviceIndex device, std::shared_ptr<ProcessGroupNCCL> pg,
      const DeviceMesh &mesh, uint64_t seed = 1234)
      : config(cfg), wte(cfg.vocab_size, cfg.n_embd, device, seed),
        wpe(cfg.context_length, cfg.n_embd, device, seed + 100),
        ln_f(cfg.n_embd),
        rank_(pg->get_rank()), world_size_(pg->get_worldsize()) {
    ln_f.to(device);

    // One forward ring rotator + staging shared by ALL layers (forward is
    // sequential), instead of one persistent set per layer. Saves
    // ~kv_numel*2 * n_layers resident at no throughput cost, parity-exact.
    // ON BY DEFAULT (verified: -1100MB @124M/T2048, throughput-neutral, 160-step
    // grad-norm stable, all rotators). Set CP_SHARE_FWD_ROTATOR=0 to disable
    // (per-layer rotators) for A/B.
    std::shared_ptr<ContextParallel::SharedFwdRing> shared_fwd_ring;
    {
      const char *e = std::getenv("CP_SHARE_FWD_ROTATOR");
      const bool share = (e == nullptr) || !(e[0] == '0' && e[1] == '\0');
      if (share)
        shared_fwd_ring = std::make_shared<ContextParallel::SharedFwdRing>();
    }

    for (int i = 0; i < cfg.n_layers; ++i) {
      auto a = std::make_shared<CPAttention>(
          cfg.n_embd, cfg.n_heads, cfg.n_layers, device, pg, mesh,
          seed + 200 + i * 10,
          cfg.load_balancing, cfg.cp_unshard, cfg.rotator, cfg.recompute_k,
          shared_fwd_ring);
      auto m = std::make_shared<MLP>(cfg.n_embd, cfg.n_layers, device,
                                     seed + 200 + i * 10);
      // seed + 300 + static_cast<uint64_t>(i) * 10);
      attn_blocks.push_back(a);
      mlp_blocks.push_back(m);
      register_module(a.get());
      register_module(m.get());
    }

    if (cfg.weight_tying) {
      lm_head = std::make_shared<nn::Linear>();
      {
        autograd::NoGradGuard no_grad;
        lm_head->weight = wte.weight.transpose(0, 1);
      }
      lm_head->weight.set_requires_grad(true);
    } else {
      lm_head = std::make_shared<nn::Linear>(cfg.n_embd, cfg.vocab_size, false);
      init_linear_gpt2(*lm_head, 0.02f, seed + 1000, true);
      lm_head->to(device);
      register_module(lm_head.get());
    }

    Tensor pos_cpu(Shape{{1, cfg.context_length}},
                   TensorOptions().with_dtype(Dtype::Int64));
    int64_t *pos_data = pos_cpu.data<int64_t>();
    for (int64_t i = 0; i < cfg.context_length; ++i)
      pos_data[i] = i;
    cached_pos_ = pos_cpu.to(device);

    register_module(wte);
    register_module(wpe);
    register_module(ln_f);
  }

  // Switch all CP attention layers to unshard=true (for generation) or restore
  // training mode (unshard=false, seq_is_local per original construction).
  void set_generation_mode(bool gen) {
    for (auto &a : attn_blocks) {
      a->set_unshard(gen ? true : config.cp_unshard);
      a->set_generation_mode(gen);
    }
    is_in_generation_mode_ = gen;
  }

  void reset_timing() {
    t_tok_emb = t_pos_emb = t_attn = t_mlp = t_ln_f = t_lm_head = 0.0;
    for (auto &a : attn_blocks)
      a->reset_t_attn();
  }

  void collect_attn_timing() {
    t_attn = 0.0;
    for (auto &a : attn_blocks)
      t_attn += a->t_attn;
  }

  void print_timing(int rank) const {
    if (rank == 0) {
      std::cout << "  [LAYER] tok_emb: " << std::fixed << std::setprecision(1)
                << (t_tok_emb * 1000.0) << "ms"
                << " | pos_emb: " << (t_pos_emb * 1000.0) << "ms"
                << " | attn_cp: " << (t_attn * 1000.0) << "ms"
                << " | mlp: " << (t_mlp * 1000.0) << "ms"
                << " | ln_f: " << (t_ln_f * 1000.0) << "ms"
                << " | lm_head: " << (t_lm_head * 1000.0) << "ms" << std::endl;
    }
  }

  Tensor forward(const Tensor &idx) override {
    emit_nvtx("GPT_Forward");
    int64_t T = idx.shape().dims[1];

    // Pre-embedding sequence shard (training path only).
    // After this branch: idx_to_embed is [B, T/n] and pos_idx is [1, T/n].
    // For generation / cp_unshard: keep full T and 0..T-1 position range.
    Tensor idx_to_embed = idx;
    Tensor pos_idx;
    if (!config.cp_unshard && !is_in_generation_mode_) {
      Tensor empty_y;
      ShardedInputs sh = shard_sequence_pre_embed(
          idx, empty_y, T, world_size_, rank_, config.load_balancing,
          idx.device());
      idx_to_embed = sh.idx_local;
      pos_idx = sh.pos_local;
    } else {
      Tensor pos_flat =
          autograd::reshape(cached_pos_, Shape({{config.context_length}}));
      Tensor pos_sliced = pos_flat.slice(0, T);
      pos_idx = autograd::reshape(pos_sliced, Shape({{1, T}}));
    }

    // [SHARD CHECK] one-time: confirm the local sequence length fed into the
    // embeddings/blocks. Expect T/world_size (sharded) unless cp_unshard=true.
    static bool printed_shard_check = false;
    if (std::getenv("CP_DEBUG_SHAPES") && rank_ == 0 && !printed_shard_check &&
        !is_in_generation_mode_) {
      printed_shard_check = true;
      std::cout << "[SHARD CHECK] global T=" << T
                << " world_size=" << world_size_
                << " cp_unshard=" << (config.cp_unshard ? "true" : "false")
                << " -> local seqlen fed to blocks = "
                << idx_to_embed.shape().dims[1]
                << " (expect " << (config.cp_unshard ? T : T / world_size_)
                << ")\n";
    }

    // Token embedding
    timer_tok_emb.start_timer();
    Tensor tok_emb = wte.forward(idx_to_embed);
    t_tok_emb += timer_tok_emb.get_elapsed_seconds();

    // Position embedding
    timer_pos_emb.start_timer();
    Tensor pos_emb = wpe.forward(pos_idx);
    t_pos_emb += timer_pos_emb.get_elapsed_seconds();

    Tensor x = autograd::add(tok_emb, pos_emb); // [B, T/n, C] or [B, T, C]

    // Env-gated forward-parity dump (ws=1 base-model check). Captures ONCE on
    // rank 0. Raw [B,T,C] row-major binary, matches PT's .npy layout.
    static bool fwd_dumped = false;
    bool dump_fwd = std::getenv("DUMP_FWD") && !fwd_dumped && rank_ == 0 &&
                    !is_in_generation_mode_;
    auto dump_act = [](const std::string &fn, const Tensor &t) {
      Tensor host = t.to_cpu().contiguous();
      std::ofstream bf(fn, std::ios::binary);
      bf.write(reinterpret_cast<const char *>(host.data<float>()),
               host.numel() * sizeof(float));
      bf.close();
    };
    if (dump_fwd) {
      dump_act("fwd_emb_cpp.bin", x);
      Tensor ie = idx_to_embed.to_cpu();
      Tensor pe = pos_idx.to_cpu();
      const int64_t *iep = ie.data<int64_t>();
      const int64_t *pep = pe.data<int64_t>();
      std::cout << "[DUMP_FWD] embedded idx[:8]=";
      for (int i = 0; i < 8; ++i) std::cout << iep[i] << " ";
      std::cout << " pos[:8]=";
      for (int i = 0; i < 8 && i < pe.numel(); ++i) std::cout << pep[i] << " ";
      std::cout << "\n";
    }

    // Attention + MLP blocks
    for (int i = 0; i < config.n_layers; ++i) {
      x = attn_blocks[i]->forward(x);
      // attn timing tracked inside CPAttention; collect at end
      if (dump_fwd && i == 0)
        dump_act("fwd_blk0attn_cpp.bin", x);  // post-attention, pre-MLP

      timer_mlp.start_timer();
      x = mlp_blocks[i]->forward(x);
      t_mlp += timer_mlp.get_elapsed_seconds();
      if (dump_fwd && i == 0)
        dump_act("fwd_blk0_cpp.bin", x);
    }

    // Final LayerNorm
    timer_ln_f.start_timer();
    x = autograd::layer_norm(x, ln_f.weight, ln_f.bias, config.n_embd, ln_f.eps);
    if (dump_fwd) {
      dump_act("fwd_lnf_cpp.bin", x);
      fwd_dumped = true;
      std::cout << "[DUMP_FWD] saved fwd_emb_cpp.bin, fwd_blk0_cpp.bin, "
                   "fwd_lnf_cpp.bin\n";
    }
    t_ln_f += timer_ln_f.get_elapsed_seconds();

    // LM Head
    timer_lm_head.start_timer();
    Tensor logits = lm_head->forward(x);
    t_lm_head += timer_lm_head.get_elapsed_seconds();

    return logits;
  }

private:
  Tensor cached_pos_;
  int rank_;
  int world_size_;
  bool is_in_generation_mode_ = false;
};

// =============================================================================
// Main Training Loop
// =============================================================================

int main(int argc, char **argv) {
  // ISOLATION TEST (CP_ALLOC_LEGACY=1): reproduce the OLD tracker init — before
  // MPI_Init, shared "1step.csv" for all ranks. Used to test whether the
  // tracker relocation caused the peak-memory change.
  // const bool legacy_tracker = std::getenv("CP_ALLOC_LEGACY") != nullptr;
  // if (legacy_tracker) {
  //   OwnTensor::AllocationTracker::instance().init("1step.csv");
  // }
  MPI_Init(&argc, &argv);

  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Rank-aware allocation-trace filename so multiple ranks don't clobber one
  // shared file. Override base via CP_ALLOC_CSV (default "1step"); rank suffix
  // is always appended -> e.g. 1step_rank0.csv.
  // if (!legacy_tracker) {
  //   const char *base = std::getenv("CP_ALLOC_CSV");
  //   std::string csv = std::string(base ? base : "1step") + "_rank" +
  //                     std::to_string(rank) + ".csv";
  //   OwnTensor::AllocationTracker::instance().init(csv.c_str());
  // }

  if (rank == 0) {
    std::cout << "=== GPT-2 Context Parallel Training Script ===" << std::endl;
  }

  bool nsys_report = false;
  //nsys profile -t cuda -o my_report ./path/to/your_executable
  //nsys stats --report cuda_gpu_kern_sum:base --format csv -o my_custom_report /path/to/my_report.nsys-rep
  //nsys profile -t cuda -o my_report ./your_executable && nsys stats --report cuda_gpu_kern_sum:base --format csv -o my_custom_report my_report.nsys-rep
  try {

    bool fourtyfour = false;
    // Env override to select the small (~44M) config for fast overlap testing
    // (matches PT MODEL_44M). Load init_weights_named_44M.bin to match.
    if (std::getenv("CP_MODEL_44M")) fourtyfour = true;

    // Configuration
    GPTConfig config;
    config.batch_size = 4;
    config.context_length = 1024;
    config.vocab_size = 50304;
    config.n_embd = fourtyfour?384:768;
    config.n_layers = fourtyfour?3:12;
    config.n_heads = fourtyfour?6:12;
    config.weight_tying = false;
    config.load_balancing = true;

    // ── Memory-scaling sweep overrides (env-driven) ────────────────────────
    // Optional; defaults preserve original behavior. Mirror the PyTorch probe
    // knobs so the mem_scaling harness can drive identical sweeps.
    //   CP_T            : sequence length (also sets the wpe/pos table size)
    //   CP_N_EMBD/CP_N_LAYER/CP_N_HEAD : override model dims
    //   CP_WEIGHT_TYING : 1/0 to tie lm_head to wte
    //   CP_MEM_PROBE    : 1 = run CP_MEM_PROBE_STEPS steps (grad_accum=1), skip
    //                     val/gen, snapshot nvidia-smi + cudaMemGetInfo, exit.
    //   CP_MEM_PROBE_STEPS : steps in probe mode (default 2).
    //   CP_MODEL_LABEL  : free-text label embedded in snapshots.
    if (const char *e = std::getenv("CP_T"))         config.context_length = atoll(e);
    if (const char *e = std::getenv("CP_N_EMBD"))    config.n_embd = atoll(e);
    if (const char *e = std::getenv("CP_N_LAYER"))   config.n_layers = atoll(e);
    if (const char *e = std::getenv("CP_N_HEAD"))    config.n_heads = atoll(e);
    if (const char *e = std::getenv("CP_WEIGHT_TYING"))
      config.weight_tying = (atoi(e) != 0);
    const bool mem_probe = std::getenv("CP_MEM_PROBE") != nullptr &&
                           atoi(std::getenv("CP_MEM_PROBE")) != 0;
    const int mem_probe_steps =
        std::getenv("CP_MEM_PROBE_STEPS") ? atoi(std::getenv("CP_MEM_PROBE_STEPS")) : 2;
    const std::string mem_label =
        std::getenv("CP_MODEL_LABEL") ? std::getenv("CP_MODEL_LABEL") : "cpp";
    // Rotator selection: CP_ROTATOR = p2p | alltoall | allgather (default a2a).
    std::string rotator_label = "alltoall";
    if (const char *e = std::getenv("CP_ROTATOR")) {
      std::string rv = e;
      if (rv == "p2p" || rv == "P2P") {
        config.rotator = RotatorType::P2P; rotator_label = "p2p";
      } else if (rv == "allgather" || rv == "AllGather") {
        config.rotator = RotatorType::AllGather; rotator_label = "allgather";
      } else {
        config.rotator = RotatorType::AlltoAll; rotator_label = "alltoall";
      }
    }
    // Recompute K in ring backward (CP_RECOMPUTE_K=1) to lower peak memory.
    if (const char *e = std::getenv("CP_RECOMPUTE_K")) {
      config.recompute_k = (e[0] == '1');
    }

    const int B = static_cast<int>(config.batch_size);
    const int T = static_cast<int>(config.context_length);
    int global_batch = fourtyfour?65536:524288;
    // Test-only override: shrink the global batch (=> smaller grad_accum =>
    // faster steps) for quick overlap-bisection runs. Default = original.
    if (const char *e = std::getenv("CP_GLOBAL_BATCH")) global_batch = atoi(e);
    // Probe mode: one micro-batch (B*T tokens) -> grad_accum=1. Activation peak
    // is identical to a full grad-accum loop (sequential), so this is faithful.
    if (mem_probe) global_batch = B * T;
    const int grad_accum_steps = global_batch / (B * T);

    const float max_lr = 6e-4f;
    const float min_lr = max_lr * 0.1f;
    const int VAL_FREQ = 100;
    const int TOK_GEN_FREQ = 100;

    // int max_steps    = (static_cast<int>(num_params) / global_batch ) * 5;
    int max_steps = fourtyfour?6768:1555;
    int warmup_steps = max_steps / 10;
    // Test-only overrides for fast overlap-bisection runs (default = original).
    if (const char *e = std::getenv("CP_MAX_STEPS")) max_steps = atoi(e);
    if (const char *e = std::getenv("CP_WARMUP"))    warmup_steps = atoi(e);
    if (nsys_report && !mem_probe)
    {
      max_steps = 1;
      warmup_steps = 0;
    }
    // mem_probe takes precedence over nsys_report so memory/throughput probes
    // always run the requested number of steps (need >=2 for steady-state).
    if (mem_probe) { max_steps = mem_probe_steps; warmup_steps = 1; }
    // const int max_steps    = 1;

    if (rank == 0) {
      std::cout << "Configuration:\n";
      std::cout << "  vocab_size: " << config.vocab_size << "\n";
      std::cout << "  context_length: " << config.context_length << "\n";
      std::cout << "  n_embd: " << config.n_embd << "\n";
      std::cout << "  n_layers: " << config.n_layers << "\n";
      std::cout << "  n_heads: " << config.n_heads << "\n";
      std::cout << "  B=" << B << ", T=" << T << "\n";
      std::cout << "  world_size: " << world_size << "\n";
    }

    // Device + Process Group
    DeviceIndex device(Device::CUDA, rank);
    cudaSetDevice(rank);

    std::vector<int> ranks_vec(world_size);

    for (int i = 0; i < world_size; i++)
      ranks_vec[i] = i;
    DeviceMesh mesh({world_size}, ranks_vec);
    auto pg = mesh.get_process_group(0);

    if (rank == 0) {
      std::cout << "\nInitializing model on CUDA device " << rank << "...\n";
    }

    // Model
    GPT model(config, device, pg, mesh, /*seed=*/1234);

    // Parameter count
    auto params = model.parameters();
    int64_t num_params = 0;
    int64_t num_params_gpu = 0;
    for (auto &p : params) {
      num_params += p.numel();
      num_params_gpu += p.numel();
    }

    // ----- INIT WEIGHT LOAD (PT parity, opt-in via LOAD_INIT_WEIGHTS env) -----
    // Loads init_weights.bin written by gpt2_cp_headtail_fp32.py so the C++
    // model starts from identical weights as the PyTorch baseline.
    if (const char *init_path = std::getenv("LOAD_INIT_WEIGHTS")) {
      if (rank == 0) {
        std::cout << "Loading init weights from " << init_path << " ...\n";
      }
      std::ifstream wf(init_path, std::ios::binary);
      if (!wf) {
        throw std::runtime_error(std::string("cannot open ") + init_path);
      }
      TensorOptions cpu_opts = TensorOptions().with_dtype(Dtype::Float32);
      for (size_t i = 0; i < params.size(); ++i) {
        Tensor &p = params[i];
        int64_t n = p.numel();
        Tensor host_t = Tensor::empty(p.shape(), cpu_opts);
        wf.read(reinterpret_cast<char *>(host_t.data<float>()),
                static_cast<std::streamsize>(n) * sizeof(float));
        if (!wf) {
          throw std::runtime_error("short read on init_weights.bin at param " +
                                   std::to_string(i));
        }
        Tensor dev_t = host_t.to(device);
        p.copy_(dev_t);
      }
      if (rank == 0) {
        std::cout << "Loaded init weights from " << init_path << "\n";
      }
    }
    // ----- END INIT WEIGHT LOAD -----

    // ----- NAME-AWARE INIT WEIGHT LOAD (LOAD_INIT_NAMED) -----
    // The positional LOAD_INIT_WEIGHTS above SCRAMBLES weights: C++
    // model.parameters() order (blocks first, then wte/wpe/ln_f) differs from
    // PT named_parameters() order (wte/wpe/ln_f first). This loader matches by
    // NAME and transposes Linear weights (PT [out,in] -> C++ [in,out]).
    if (const char *named_path = std::getenv("LOAD_INIT_NAMED")) {
      if (rank == 0)
        std::cout << "Loading NAMED init weights from " << named_path << " ...\n";
      std::ifstream wf(named_path, std::ios::binary);
      if (!wf)
        throw std::runtime_error(std::string("cannot open ") + named_path);
      struct Rec { std::vector<int64_t> dims; std::vector<float> data; };
      std::vector<std::pair<std::string, Rec>> recs;
      while (wf.peek() != EOF) {
        int32_t nlen = 0;
        wf.read(reinterpret_cast<char *>(&nlen), 4);
        if (!wf) break;
        std::string name(static_cast<size_t>(nlen), '\0');
        wf.read(&name[0], nlen);
        int32_t ndim = 0;
        wf.read(reinterpret_cast<char *>(&ndim), 4);
        Rec rec;
        int64_t numel = 1;
        for (int d = 0; d < ndim; ++d) {
          int64_t dd = 0;
          wf.read(reinterpret_cast<char *>(&dd), 8);
          rec.dims.push_back(dd);
          numel *= dd;
        }
        rec.data.resize(static_cast<size_t>(numel));
        wf.read(reinterpret_cast<char *>(rec.data.data()),
                numel * static_cast<int64_t>(sizeof(float)));
        recs.emplace_back(std::move(name), std::move(rec));
      }
      TensorOptions cpu_opts = TensorOptions().with_dtype(Dtype::Float32);
      // needs_transpose MUST be decided by ROLE, not by shape: C++ Linear weights
      // are [in,out], PyTorch's are [out,in], so every Linear weight needs a
      // transpose on load. Shape-based detection silently FAILS for SQUARE Linear
      // weights (e.g. attn.c_proj 768x768), where [out,in]==[in,out] dims match
      // and a direct copy loads W instead of W^T. That single bug made attn.c_proj
      // use W (not W^T), corrupting every block and producing the orthogonal
      // step-0 gradients. Embeddings/LayerNorm/biases are NOT transposed.
      auto load = [&](const std::string &name, Tensor &target,
                      bool needs_transpose) {
        Rec *r = nullptr;
        for (auto &pr : recs)
          if (pr.first == name) { r = &pr.second; break; }
        if (!r) throw std::runtime_error("named init missing: " + name);
        auto ts = target.shape().dims;
        Tensor host = Tensor::empty(target.shape(), cpu_opts);
        float *hp = host.data<float>();
        if (needs_transpose) {
          if (ts.size() != 2 || r->dims.size() != 2 ||
              r->dims[0] != ts[1] || r->dims[1] != ts[0])
            throw std::runtime_error("transpose shape mismatch for " + name);
          int64_t a = r->dims[0], b = r->dims[1]; // PT [out,in] -> C++ [in,out]
          for (int64_t i = 0; i < a; ++i)
            for (int64_t j = 0; j < b; ++j)
              hp[j * a + i] = r->data[i * b + j];
        } else {
          if (r->dims.size() != ts.size())
            throw std::runtime_error("shape mismatch for " + name);
          for (size_t k = 0; k < ts.size(); ++k)
            if (r->dims[k] != ts[k])
              throw std::runtime_error("shape mismatch for " + name);
          std::memcpy(hp, r->data.data(), r->data.size() * sizeof(float));
        }
        Tensor dev = host.to(device);
        target.copy_(dev);
      };
      load("transformer.wte.weight", model.wte.weight, false);
      load("transformer.wpe.weight", model.wpe.weight, false);
      load("transformer.ln_f.weight", model.ln_f.weight, false);
      load("transformer.ln_f.bias", model.ln_f.bias, false);
      for (int i = 0; i < config.n_layers; ++i) {
        std::string b = "transformer.h." + std::to_string(i) + ".";
        auto &a = *model.attn_blocks[i];
        auto &m = *model.mlp_blocks[i];
        load(b + "attn.ln.weight", a.ln.weight, false);
        load(b + "attn.ln.bias", a.ln.bias, false);
        load(b + "attn.c_attn.weight", a.c_attn.weight, true);
        load(b + "attn.c_attn.bias", a.c_attn.bias, false);
        load(b + "attn.c_proj.weight", a.c_proj.weight, true);
        load(b + "attn.c_proj.bias", a.c_proj.bias, false);
        load(b + "mlp.ln.weight", m.ln.weight, false);
        load(b + "mlp.ln.bias", m.ln.bias, false);
        load(b + "mlp.c_fc.weight", m.fc_up.weight, true);
        load(b + "mlp.c_fc.bias", m.fc_up.bias, false);
        load(b + "mlp.c_proj.weight", m.fc_down.weight, true);
        load(b + "mlp.c_proj.bias", m.fc_down.bias, false);
      }
      load("lm_head.weight", model.lm_head->weight, true);
      if (rank == 0)
        std::cout << "Loaded NAMED init weights (" << recs.size()
                  << " records)\n";
    }
    // ----- END NAME-AWARE INIT WEIGHT LOAD -----


    if (rank == 0) {
      std::cout << "Parameters: " << num_params << "\n";
      std::cout << "Parameters per GPU: " << num_params_gpu << "\n";
      std::cout << "max_steps: " << max_steps << "\n";
      std::cout << "warmup_steps: " << warmup_steps << "\n";
    }

    // Optimizer
    nn::AdamW optimizer(params, max_lr, 0.9f, 0.95f, 1e-8f, 0.1f);

    // Data loaders (same path as gpt2_tp_test)
    std::string data_root =
        "/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/Data_Loader/Data/";
    DataLoaderLite train_loader(B, T, 0, 1, "train", data_root, true, 100000000,
                                rank);
    DataLoaderLite val_loader(B, T, 0, 1, "val", data_root, true, 100000000,
                              rank);

    // Restore CUDA device context after DataLoader init
    // (DataLoader constructor may switch active device via tensor allocation)
    cudaSetDevice(rank);

    // Step timers
    CudaTimer timer_step, timer_data, timer_fwd, timer_loss, timer_bwd;
    CudaTimer timer_clip, timer_optim;

    if (rank == 0) {
      std::cout << "\nStarting training...\n";
    }

    // CSV log + config file setup
    std::string log_filename, config_filename;
    std::ofstream log_file;

    if (rank == 0) {
      std::filesystem::create_directories("CP_Training_logs");
      int log_idx = 1;
      while (true) {
        log_filename = "CP_Training_logs/CP_Training_log" +
                       std::to_string(log_idx) + ".csv";
        if (!std::filesystem::exists(log_filename))
          break;
        log_idx++;
      }
      std::cout << "Saving logs to: " << log_filename << "\n";

      config_filename = "CP_Training_logs/CP_Training_log" +
                        std::to_string(log_idx) + "_config.txt";
      std::ofstream config_file(config_filename);
      config_file << "Configuration:\n";
      config_file << "  Batch_size: " << B << "\n";
      config_file << "  context_length: " << config.context_length << "\n";
      config_file << "  n_embd: " << config.n_embd << "\n";
      config_file << "  n_heads: " << config.n_heads << "\n";
      config_file << "  vocab_size: " << config.vocab_size << "\n";
      config_file << "  n_layers: " << config.n_layers << "\n";
      config_file << "  global_batch: " << global_batch << "\n";
      config_file << "  grad_accum_steps: " << grad_accum_steps << "\n";
      config_file << "  world_size: " << world_size << "\n";
      config_file << "  Parameters: " << num_params << "\n";
      config_file << "  Parameters per GPU: " << num_params_gpu << "\n";
      config_file << "  Max Learning Rate: " << max_lr << "\n";
      config_file << "  Min Learning Rate: " << min_lr << "\n";
      config_file << "  max_steps: " << max_steps << "\n";
      config_file << "  warmup_steps: " << warmup_steps << "\n";

      // Initial GPU memory
      size_t free_mem = 0, total_mem = 0;
      cudaMemGetInfo(&free_mem, &total_mem);
      double used_mb =
          static_cast<double>(total_mem - free_mem) / (1024.0 * 1024.0);
      double total_mb = static_cast<double>(total_mem) / (1024.0 * 1024.0);
      config_file << "  GPU Memory Used (rank 0): " << std::fixed
                  << std::setprecision(1) << used_mb << " MB / " << total_mb
                  << " MB\n";
      config_file.close();

      log_file.open(log_filename);
      if (!log_file.is_open()) {
        std::cerr << "ERROR: Could not open log file " << log_filename << "\n";
        std::exit(1);
      }
      log_file << "step,loss,val_loss,lr,grad_norm,dt_ms,tok_per_sec,"
                  "timer_data,timer_fwd,timer_loss,timer_bwd,timer_clip,"
                  "timer_optim,timer_tok_emb,timer_pos_emb,timer_attn_cp,"
                  "timer_mlp,timer_ln_f,timer_lm_head,mem_gpu_mb\n";
      log_file << std::fixed << std::setprecision(6);
    }

    float val_loss_log = -1.0f;

    // Auto-increment run index for the per-step debug dump file: scan existing
    // debug_rank_lb{rank}_N.md files and pick the next free N. Each rank picks
    // its own; aligns per-rank across reruns without overwriting.
    int debug_run_idx = 0;
    while (true) {
      std::string probe = "debug_rank_lb" + std::to_string(rank) + "_" +
                          std::to_string(debug_run_idx) + ".md";
      std::ifstream check(probe);
      if (!check.good()) break;
      ++debug_run_idx;
    }
    std::string debug_path = "debug_rank_lb" + std::to_string(rank) + "_" +
                             std::to_string(debug_run_idx) + ".md";
    if (rank == 0) {
      std::cout << "Debug dump file: " << debug_path << "\n";
    }

    for (int step = 0; step < max_steps; ++step) {
      try {
        timer_step.start_timer();

        // ---- Validation every VAL_FREQ steps ---- (skipped in probe mode)
        if (!mem_probe && (step % VAL_FREQ == 0 || step == max_steps - 1)) {
          val_loader.reset();
          float val_loss_accum = 0.0f;
          const int val_steps = 5;

          for (int v = 0; v < val_steps; ++v) {
            Batch vbatch = val_loader.next_batch();
            Tensor vx = vbatch.input.to(device).as_type(Dtype::Int64);
            Tensor vy = vbatch.target.to(device).as_type(Dtype::Int64);
            autograd::NoGradGuard no_grad;
            Tensor vlogits = model.forward(vx);
            Tensor vloss;
            if (!config.cp_unshard) {
              // Sharded: vlogits is [B, T/n, vocab]; produce matching vy slice
              // using the same pre-embedding shard (contiguous or HeadTail).
              ShardedInputs vsh = shard_sequence_pre_embed(
                  vx, vy, T, world_size, rank, config.load_balancing, device);
              Tensor vy_local = vsh.y_local;
              vloss = autograd::sparse_cross_entropy_loss(vlogits, vy_local);
              pg->all_reduce(vloss.data<float>(), vloss.data<float>(), 1,
                             Dtype::Float32, op_t::sum, true);
              float vloss_val = vloss.to_cpu().data<float>()[0] / static_cast<float>(world_size);
              val_loss_accum += vloss_val / static_cast<float>(val_steps);
            } else {
              // Unsharded: vlogits is [B, T, vocab]; all ranks identical.
              vloss = autograd::sparse_cross_entropy_loss(vlogits, vy);
              val_loss_accum += vloss.to_cpu().data<float>()[0] / static_cast<float>(val_steps);
            }
          }

          if (rank == 0) {
            std::cout << "validation loss: " << std::fixed
                      << std::setprecision(4) << val_loss_accum << "\n";
          }
          val_loss_log = val_loss_accum;
        }

        // ---- Token generation every TOK_GEN_FREQ steps ---- (skip in probe)
        if (!mem_probe && step > 0 && (step % TOK_GEN_FREQ == 0 || step == max_steps - 1)) {
          if (rank == 0) {
            std::cout << "--- Generating tokens at step " << step << " ---\n";
          }
          // Switch to unshard=true so generation gets full [B,T,C] logits.
          model.set_generation_mode(true);
          autograd::NoGradGuard no_grad_gen;

          const int num_return_seq = 2;
          const int max_length = 32;

          Tensor xgen(
              Shape({{num_return_seq, 8}}),
              TensorOptions().with_dtype(Dtype::Int64).with_device(device));
          std::vector<int64_t> xgen_tokens = {
              15496, 11, 314, 1101, 281, 9552, 2746, 11,
              15496, 11, 314, 1101, 281, 9552, 2746, 11};
          xgen.set_data(xgen_tokens);

          while (xgen.shape().dims[1] < max_length) {
            // Pad sequence to multiple of world_size for CP
            int64_t seq_len = xgen.shape().dims[1];
            int64_t padded_len =
                ((seq_len + world_size - 1) / world_size) * world_size;
            Tensor xgen_fwd = xgen;
            if (padded_len != seq_len) {
              int64_t Bg_pad = xgen.shape().dims[0];
              Tensor pad = Tensor::zeros(
                  Shape({{Bg_pad, padded_len - seq_len}}), xgen.opts());
              xgen_fwd = Tensor::cat({xgen, pad}, 1);
            }
            Tensor gen_logits = model.forward(xgen_fwd);

            int64_t Bg = gen_logits.shape().dims[0];
            int64_t Tg = gen_logits.shape().dims[1];
            int64_t Vg = gen_logits.shape().dims[2];

            // Use original seq_len-1 as last real token position
            Tensor gather_idx = Tensor::full(
                Shape({{Bg, 1, Vg}}),
                TensorOptions().with_dtype(Dtype::Int64).with_device(device),
                static_cast<float>(seq_len - 1));

            Tensor last_logits = OwnTensor::gather(gen_logits, 1, gather_idx);
            Tensor probs = autograd::softmax(last_logits, -1);

            auto topk_res = probs.topk(50, -1);
            Tensor topk_probs = topk_res.first;
            Tensor topk_idx = topk_res.second;

            Tensor topk_probs_2d =
                autograd::reshape(topk_probs, Shape({{Bg, 50}}));

            Tensor ix;
            if (rank == 0) {
              try {
                ix = Tensor::multinomial(topk_probs_2d, 1);
              } catch (...) {
                ix = Tensor::zeros(Shape({{Bg, 1}}),
                                   TensorOptions()
                                       .with_dtype(Dtype::Int64)
                                       .with_device(device));
              }
            } else {
              ix = Tensor::zeros(
                  Shape({{Bg, 1}}),
                  TensorOptions().with_dtype(Dtype::Int64).with_device(device));
            }

            if (world_size > 1) {
              ix = ix.contiguous();
              pg->broadcast(ix.data<int64_t>(), ix.data<int64_t>(), ix.numel(),
                            Dtype::Int64, 0, true);
            }

            Tensor topk_idx_2d = autograd::reshape(topk_idx, Shape({{Bg, 50}}));
            Tensor next_token = OwnTensor::gather(topk_idx_2d, 1, ix);
            xgen = Tensor::cat({xgen, next_token}, 1);
          }

          if (rank == 0) {
            Tensor xgen_cpu = xgen.to_cpu();
            int64_t *data = xgen_cpu.data<int64_t>();
            int64_t Bg = xgen.shape().dims[0];
            int64_t Tg = xgen.shape().dims[1];

            std::ofstream cfg_app(config_filename, std::ios::app);
            cfg_app << "\n--- Generated Tokens at Step " << step << " ---\n";

            for (int i = 0; i < static_cast<int>(Bg); ++i) {
              std::vector<int64_t> sample_tokens;
              std::string token_str;
              for (int j = 0; j < static_cast<int>(Tg); ++j) {
                sample_tokens.push_back(data[i * Tg + j]);
                if (j > 0)
                  token_str += " ";
                token_str += std::to_string(data[i * Tg + j]);
              }
              std::string decoded = decode_tokens_tiktoken(sample_tokens);
              std::cout << "Sample " << i << " [IDs]: " << token_str << "\n";
              std::cout << "Sample " << i << " [Text]: " << decoded << "\n";
              cfg_app << "Sample " << i << " [IDs]: " << token_str << "\n";
              cfg_app << "Sample " << i << " [Text]: " << decoded << "\n";
            }
            cfg_app.close();
          }
          // Restore training mode (unshard=false, sharded sequence).
          model.set_generation_mode(false);
        }

        // ---- Training step ----
        double time_data = 0, time_forward = 0, time_loss = 0;
        double time_backward = 0, time_clip = 0, time_optim = 0;

        optimizer.zero_grad();
        if (model.config.weight_tying && model.lm_head->weight.has_grad()) {
          model.lm_head->weight.zero_grad();
        }
        model.reset_timing();

        Tensor loss_accum_gpu =
            Tensor::zeros(Shape{{1}}, TensorOptions().with_device(device));

        static Tensor grad_scale =
            Tensor::full(Shape{{1}}, TensorOptions().with_device(device),
                         1.0f / static_cast<float>(grad_accum_steps));

        // ── Step 0 gradient dump (for PyTorch parity check) — FULLY GATED ──
        // Runs ONE micro-batch UNSCALED (grad_scale=1.0), dumps NAMED per-param
        // grad L2 (transpose-invariant) to step0_grads_cpp.txt, then zeroes
        // grads and resets the loader so the real accumulation below is
        // byte-identical to a normal run. Only fires when DUMP_STEP0_GRADS set.
        // Mirrors the exact loss path (incl. avg all_reduce) used in training.
        if (step == 0 && std::getenv("DUMP_STEP0_GRADS")) {
          Batch db = train_loader.next_batch();
          Tensor dx = db.input.to(device).as_type(Dtype::Int64);
          Tensor dy = db.target.to(device).as_type(Dtype::Int64);
          if (rank == 0) {
            Tensor dx_cpu = dx.to_cpu();
            const int64_t *xp = dx_cpu.data<int64_t>();
            std::cout << "[DUMP] first 8 token ids:";
            for (int i = 0; i < 8; ++i) std::cout << " " << xp[i];
            std::cout << "\n";
          }
          Tensor dlogits = model.forward(dx);
          Tensor dloss;
          if (!config.cp_unshard) {
            ShardedInputs dsh = shard_sequence_pre_embed(
                dx, dy, T, world_size, rank, config.load_balancing, device);
            dloss = autograd::sparse_cross_entropy_loss(dlogits, dsh.y_local);
            pg->all_reduce(dloss.data<float>(), dloss.data<float>(), 1,
                           Dtype::Float32, op_t::avg, true);
          } else {
            dloss = autograd::sparse_cross_entropy_loss(dlogits, dy);
          }
          Tensor one_scale =
              Tensor::full(Shape{{1}}, TensorOptions().with_device(device), 1.0f);
          dloss.backward(&one_scale);

          // AVG-all-reduce PARAM grads across ranks -> full-batch gradient
          // (sum over all tokens), independent of how each impl splits tokens
          // across ranks. Collective: ALL ranks participate. Required for a
          // valid per-element compare vs PT (rank0 local shards differ).
          for (auto &p : params) {
            if (p.has_grad()) {
              Tensor g = p.grad_view();
              pg->all_reduce(g.data<float>(), g.data<float>(), g.numel(),
                             Dtype::Float32, op_t::avg, true);
            }
          }

          if (rank == 0) {
            auto l2_of = [](const Tensor &t) -> double {
              Tensor host = t.to_cpu();
              const float *p = host.data<float>();
              double s = 0.0;
              for (int64_t i = 0; i < t.numel(); ++i)
                s += static_cast<double>(p[i]) * static_cast<double>(p[i]);
              return std::sqrt(s);
            };
            std::ofstream gf("step0_grads_cpp.txt");
            // Full per-element raw blob (ALL params, PT [out,in] layout) for a
            // complete backward cosine check. Linear weights are transposed from
            // C++ [in,out] -> PT [out,in]; embeddings/LN/biases written as-is.
            // Written in the SAME ORDER as the txt lines; compare reads names+
            // numels from the txt and slices this blob sequentially.
            std::ofstream gfr("step0_grads_cpp_raw.bin", std::ios::binary);
            double total_l2 = 0.0;
            auto emit_t = [&](const std::string &name, const Tensor &w,
                              bool needs_transpose) {
              if (w.has_grad()) {
                Tensor g = w.grad_view();
                double l2 = l2_of(g);
                total_l2 += l2 * l2;
                gf << name << " " << g.numel() << " " << std::scientific
                   << std::setprecision(8) << l2 << "\n";
                Tensor gc = needs_transpose ? g.transpose(0, 1).contiguous() : g;
                Tensor host = gc.to_cpu().contiguous();
                gfr.write(reinterpret_cast<const char *>(host.data<float>()),
                          host.numel() * sizeof(float));
              }
            };
            emit_t("transformer.wte.weight", model.wte.weight, false);
            emit_t("transformer.wpe.weight", model.wpe.weight, false);
            emit_t("transformer.ln_f.weight", model.ln_f.weight, false);
            emit_t("transformer.ln_f.bias", model.ln_f.bias, false);
            for (int i = 0; i < config.n_layers; ++i) {
              std::string b = "transformer.h." + std::to_string(i) + ".";
              auto &a = *model.attn_blocks[i];
              auto &m = *model.mlp_blocks[i];
              emit_t(b + "attn.ln.weight", a.ln.weight, false);
              emit_t(b + "attn.ln.bias", a.ln.bias, false);
              emit_t(b + "attn.c_attn.weight", a.c_attn.weight, true);
              emit_t(b + "attn.c_attn.bias", a.c_attn.bias, false);
              emit_t(b + "attn.c_proj.weight", a.c_proj.weight, true);
              emit_t(b + "attn.c_proj.bias", a.c_proj.bias, false);
              emit_t(b + "mlp.ln.weight", m.ln.weight, false);
              emit_t(b + "mlp.ln.bias", m.ln.bias, false);
              emit_t(b + "mlp.c_fc.weight", m.fc_up.weight, true);    // PT name: c_fc
              emit_t(b + "mlp.c_fc.bias", m.fc_up.bias, false);
              emit_t(b + "mlp.c_proj.weight", m.fc_down.weight, true); // PT name: c_proj
              emit_t(b + "mlp.c_proj.bias", m.fc_down.bias, false);
            }
            emit_t("lm_head.weight", model.lm_head->weight, true);
            gf << "# total_L2 " << std::scientific << std::setprecision(8)
               << std::sqrt(total_l2) << "\n";
            gf.close();
            gfr.close();

            // Raw grad arrays for cosine check. C++ stores Linear weight as
            // [in,out]; transpose to PT's [out,in] before writing so element
            // order matches PT's .npy. ln_f.weight is 1D -> no transpose.
            auto write_raw = [](const std::string &fn, const Tensor &t) {
              Tensor host = t.to_cpu().contiguous();
              std::ofstream bf(fn, std::ios::binary);
              bf.write(reinterpret_cast<const char *>(host.data<float>()),
                       host.numel() * sizeof(float));
              bf.close();
            };
            if (model.ln_f.weight.has_grad())
              write_raw("step0_raw_lnf_cpp.bin", model.ln_f.weight.grad_view());
            if (model.attn_blocks[0]->c_attn.weight.has_grad()) {
              Tensor cg = model.attn_blocks[0]->c_attn.weight.grad_view();
              Tensor cgT = cg.transpose(0, 1).contiguous(); // [in,out]->[out,in]
              write_raw("step0_raw_c_attn_cpp.bin", cgT);
            }
            std::cout << "[STEP 0 GRAD DUMP] wrote step0_grads_cpp.txt total_L2="
                      << std::scientific << std::setprecision(8)
                      << std::sqrt(total_l2) << "\n";
          }

          // Restore state so the real accumulation below is unaffected.
          optimizer.zero_grad();
          if (model.config.weight_tying && model.lm_head->weight.has_grad())
            model.lm_head->weight.zero_grad();
          train_loader.reset();
          model.reset_timing();
        }

        for (int micro = 0; micro < grad_accum_steps; ++micro) {
          // Data
          timer_data.start_timer();
          Batch batch = train_loader.next_batch();
          Tensor x_in = batch.input.to(device).as_type(Dtype::Int64);
          Tensor y_in = batch.target.to(device).as_type(Dtype::Int64);
          time_data += timer_data.get_elapsed_seconds();

          // Record autograd graph on step 0, micro 0, rank 0
          // std::unique_ptr<autograd::GraphRecordGuard> graph_guard;
          // if (step == 0 && micro == 0 && rank == 0) {
          //     graph_guard =
          //     std::make_unique<autograd::GraphRecordGuard>(true);
          // }

          // Forward
          timer_fwd.start_timer();
          Tensor logits;
          {
            emit_nvtx("Model_Forward");
            logits = model.forward(x_in);
          }
          time_forward += timer_fwd.get_elapsed_seconds();

          // Loss
          // When unshard=false, logits is [B, T/n, vocab] (local chunk only).
          // Slice y_in to the matching local token range [rank*T/n, (rank+1)*T/n).
          timer_loss.start_timer();
          Tensor loss;
          {
            emit_nvtx("Loss_Computation");
            if (!config.cp_unshard) {
              // Sharded mode: logits is [B, T/n, vocab]; produce matching y
              // slice via same pre-embedding shard (contiguous or HeadTail).
              ShardedInputs sh_y = shard_sequence_pre_embed(
                  x_in, y_in, T, world_size, rank, config.load_balancing,
                  device);
              Tensor y_local = sh_y.y_local;
              loss = autograd::sparse_cross_entropy_loss(logits, y_local);
              // NOTE: do NOT scale loss by 1/world_size here. The scaling
              // shrinks every C++ gradient to 0.5x PT (for N=2), masking
              // any LB-specific divergence and acting like an implicit half-LR.
              // For the logged scalar we AVG (not SUM) the per-rank losses to
              // recover the global mean loss.
              pg->all_reduce(loss.data<float>(), loss.data<float>(), 1,
                             Dtype::Float32, op_t::avg, true);
            } else {
              // Unsharded mode: logits is [B, T, vocab]; all ranks identical.
              loss = autograd::sparse_cross_entropy_loss(logits, y_in);
            }
            loss_accum_gpu = loss_accum_gpu + loss.detach();
          }
          time_loss += timer_loss.get_elapsed_seconds();

          // Backward
          timer_bwd.start_timer();
          {
            emit_nvtx("Backward_Pass");
            loss.backward(&grad_scale);
          }
          time_backward += timer_bwd.get_elapsed_seconds();

          // Guard destructor auto-prints forward + backward sequences
          // graph_guard.reset();
        }

        // Weight tying: accumulate lm_head grad into wte
        if (model.config.weight_tying && model.lm_head->weight.has_grad()) {
          Tensor lm_grad_T =
              model.lm_head->weight.grad_view().transpose(0, 1).contiguous();
          Tensor wte_grad = model.wte.weight.grad_view();
          wte_grad += lm_grad_T;
        }

        // Collect attention timing from all layers
        model.collect_attn_timing();

        // ---- Parameter-gradient All-Reduce across CP ranks ----
        // In Context Parallelism each rank computes param grads from ONLY its
        // sequence shard (like data parallelism); the correct full-batch grad is
        // the AVERAGE across ranks. The ring sums only the attention tensor grads
        // (dQ/dK/dV) — the Linear/LayerNorm/embedding weight grads must be synced
        // here, or the ranks become divergent replicas (grad-norm explosion).
        //
        // AVG (not SUM): the active loss is NOT scaled by 1/world_size, so each
        // rank's mean-reduced shard loss -> AVG gives the global mean gradient.
        // (Empirically verified: step-0 AVG-allreduce == single-GPU full-batch
        // grad at cosine 1.0.) The old commented code used SUM under a stale
        // "loss scaled by 1/world_size via rank_scale" assumption that the
        // current code contradicts -> SUM would be world_size x too large.
        //
        // CP_NO_GRAD_ALLREDUCE=1 disables this to reproduce the (broken)
        // divergent-replica run for A/B comparison.
        const char *_cp_no_gar = std::getenv("CP_NO_GRAD_ALLREDUCE");
        const bool _grad_ar_on =
            (_cp_no_gar == nullptr) || (_cp_no_gar[0] == '0' && _cp_no_gar[1] == '\0');
        if (world_size > 1 && _grad_ar_on) {
          for (auto &p : params) {
            if (p.has_grad()) {
              Tensor g = p.grad_view();
              pg->all_reduce(g.data<float>(), g.data<float>(), g.numel(),
                             Dtype::Float32, op_t::avg, true);
            }
          }
        }

        // ---- Post-AR debug: dump param and grad values to rank-specific .md ----
        // Mirrors PyTorch dump in Pytorch/gpt2_cp_headtail_fp32.py: writes the
        // first 4 elements of the [:2, :2] block of block-0 c_attn.weight,
        // c_attn.grad, fc_up.weight, fc_up.grad each step for cross-impl diff.
        {
          auto dump4 = [&](std::ofstream &f, const char *label, const Tensor &t) {
            // C++ stores Linear weight as [in, out] (transpose of PT's [out, in]).
            // To make this dump diffable line-by-line against PyTorch's
            // (which prints [0,0], [0,1], [1,0], [1,1] of the [out, in] tensor),
            // we read p[0], p[W], p[1], p[W+1] of the C++ [in, out] tensor —
            // those are the same four scalars in PT's print order.
            Tensor host = t.to_cpu();
            const float *p = host.data<float>();
            int64_t W = t.shape().dims[1]; // out_features (C++ layout)
            f << "**" << label << "** (first 4): ["
              << p[0] << ", " << p[W] << ", " << p[1] << ", " << p[W + 1] << "]\n";
          };
          std::ofstream df(debug_path, std::ios::app);
          if (df) {
            df << "\n## Step " << step << "\n";
            auto &c_attn = model.attn_blocks[0]->c_attn;
            auto &fc_up  = model.mlp_blocks[0]->fc_up;
            dump4(df, "c_attn.weight", c_attn.weight);
            if (c_attn.weight.has_grad())
              dump4(df, "c_attn.grad", c_attn.weight.grad_view());
            dump4(df, "c_fc.weight", fc_up.weight);
            if (fc_up.weight.has_grad())
              dump4(df, "c_fc.grad", fc_up.weight.grad_view());
          }
        }

        // TEMP DEBUGGING: Print parameter gradients to check cross-rank sync
        // if ((grad_accum_steps + 1) % 1 == 0 ) {
        //   for (int r = 0; r < world_size; ++r) {
        //     if (rank == r) {
        //       std::cout << "\n=== DEBUG: Parameter Gradients at Step " << step
        //                 << " [Rank " << rank << "] ===" << std::endl;
        //       for (auto &p : params) {
        //         if (p.has_grad()) {
        //           std::cout << "\nParam size: " << p.numel() << std::endl;
        //           try {
        //             // p.grad_view().display();
        //             p.display();
        //           } catch (const std::exception &e) {
        //             std::cout << "  Error displaying grad: " << e.what()
        //                       << std::endl;
        //           }
        //         } else {
        //           std::cout << "Param | NO GRADIENT" << std::endl;
        //         }
        //       }
        //       std::cout << "=================================================\n"
        //                 << std::endl;
        //     }
        //   }
        // }
        // MPI_Barrier(MPI_COMM_WORLD);
        // Loss to CPU
        Tensor loss_cpu = loss_accum_gpu.to_cpu();
        float loss_scalar =
            loss_cpu.data<float>()[0] / static_cast<float>(grad_accum_steps);

        if (std::isnan(loss_scalar) || std::isinf(loss_scalar)) {
          std::cerr << "ERROR: NaN/Inf at step " << step << "\n";
          if (rank == 0)
            log_file.close();
          MPI_Finalize();
          return 1;
        }

        // Gradient clipping
        timer_clip.start_timer();
        float norm = nn::clip_grad_norm_(params, 1.0f);
        time_clip = timer_clip.get_elapsed_seconds();

        // LR schedule + optimizer step
        float lr = get_lr(step, max_lr, min_lr, warmup_steps, max_steps);
        optimizer.set_lr(lr);

        if (rank == 0 && (step == 0 || step == 1 || step == 100)) {
          int64_t total_numel = 0;
          int n_fp32 = 0, n_other = 0, n_contig = 0, n_noncontig = 0;
          std::cout << "=== STEP " << step << " PARAM AUDIT ===\n";
          std::cout << "params.size() = " << params.size() << "\n";
          for (size_t i = 0; i < params.size(); ++i) {
            auto& p = params[i];
            if (!p.requires_grad() || !p.has_grad()) continue;
            Tensor g = p.grad_view();
            total_numel += g.numel();
            if (g.dtype() == Dtype::Float32) n_fp32++; else n_other++;
            if (g.is_contiguous()) n_contig++; else n_noncontig++;
            if (i < 5) {
              std::cout << "  param[" << i << "]: numel=" << p.numel()
                        << " grad.dtype=" << (int)g.dtype()
                        << " grad.contig=" << g.is_contiguous()
                        << " grad.ptr=" << (void*)g.data<float>() << "\n";
            }
          }
          std::cout << "  total_grad_numel=" << total_numel
                    << " n_fp32=" << n_fp32 << " n_other=" << n_other
                    << " n_contig=" << n_contig << " n_noncontig=" << n_noncontig << "\n";
        }
        timer_optim.start_timer();
        optimizer.step();
        time_optim = timer_optim.get_elapsed_seconds();

        double dt = timer_step.get_elapsed_seconds();

        // Throughput + time left
        int64_t tokens_processed =
            static_cast<int64_t>(B) * T * grad_accum_steps;
        double tokens_per_sec = static_cast<double>(tokens_processed) / dt;
        long long total_sec = static_cast<long long>((max_steps - step) * dt);
        int h = static_cast<int>(total_sec / 3600);
        int m = static_cast<int>((total_sec % 3600) / 60);

        // GPU memory
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);
        double used_mb =
            static_cast<double>(total_mem - free_mem) / (1024.0 * 1024.0);

        if (rank == 0) {
          std::cout << "step " << std::setw(5) << step
                    << " | loss: " << std::fixed << std::setprecision(6)
                    << loss_scalar << " | lr " << std::scientific
                    << std::setprecision(4) << lr << " | norm: " << std::fixed
                    << std::setprecision(4) << norm << " | dt: " << std::fixed
                    << std::setprecision(2) << (dt * 1000.0) << "ms"
                    << " | tok/sec: " << std::fixed << std::setprecision(1)
                    << tokens_per_sec << " | mem: " << std::fixed
                    << std::setprecision(0) << used_mb << "MB"
                    << " | Time Left: " << std::setfill('0') << std::setw(2)
                    << h << " hrs : " << std::setw(2) << m << " mins"
                    << std::setfill(' ') << "\n";

          std::cout << "  [TIMING] data: " << std::fixed << std::setprecision(1)
                    << (time_data * 1000.0) << "ms"
                    << " | fwd: " << (time_forward * 1000.0) << "ms"
                    << " | loss: " << (time_loss * 1000.0) << "ms"
                    << " | bwd: " << (time_backward * 1000.0) << "ms"
                    << " | clip: " << (time_clip * 1000.0) << "ms"
                    << " | optim: " << (time_optim * 1000.0) << "ms\n";

          model.print_timing(rank);

          // CSV
          log_file << step << "," << loss_scalar << "," << val_loss_log << ","
                   << lr << "," << norm << "," << (dt * 1000.0) << ","
                   << tokens_per_sec << "," << (time_data * 1000.0) << ","
                   << (time_forward * 1000.0) << "," << (time_loss * 1000.0)
                   << "," << (time_backward * 1000.0) << ","
                   << (time_clip * 1000.0) << "," << (time_optim * 1000.0)
                   << "," << (model.t_tok_emb * 1000.0) << ","
                   << (model.t_pos_emb * 1000.0) << ","
                   << (model.t_attn * 1000.0) << "," << (model.t_mlp * 1000.0)
                   << "," << (model.t_ln_f * 1000.0) << ","
                   << (model.t_lm_head * 1000.0) << "," << used_mb << "\n";
          log_file.flush();
        }

        val_loss_log = -1.0f;

        // ── Memory-probe snapshot: after final probe step, capture nvidia-smi
        //    + cudaMemGetInfo (live, before teardown), then stop. ───────────
        if (mem_probe && step == max_steps - 1) {
          cudaDeviceSynchronize();
          size_t pf = 0, pt = 0;
          cudaMemGetInfo(&pf, &pt);
          double probe_used_mb = static_cast<double>(pt - pf) / (1024.0 * 1024.0);
          // Caching-allocator breakdown: reserved vs active vs requested lets us
          // see how much of the footprint is size-class rounding (internal frag)
          // vs cached-but-free blocks vs genuine live memory.
          auto mstats =
              OwnTensor::CachingCUDAAllocator::instance().get_stats(rank);
          const double MB = 1024.0 * 1024.0;
          double st_reserved_mb  = mstats.reserved_current / MB;
          double st_active_mb    = mstats.active_current / MB;
          double st_requested_mb = mstats.allocated_current / MB;
          double st_frag_pct     = mstats.fragmentation_ratio();
          // Running-max peaks over the whole probe (true max occupancy during
          // fwd+bwd), the apples-to-apples analog of PyTorch's torch.peak_*.
          double st_active_peak_mb   = mstats.active_peak / MB;
          double st_reserved_peak_mb = mstats.reserved_peak / MB;
          double st_cached_free_mb =
              (mstats.reserved_current > mstats.active_current)
                  ? (mstats.reserved_current - mstats.active_current) / MB
                  : 0.0;
          if (rank == 0) {
            OwnTensor::CachingCUDAAllocator::instance().print_memory_summary();
          }
          std::string tag = "CPP_" + mem_label + "_" + rotator_label + "_T" +
                            std::to_string(config.context_length) + "_ws" +
                            std::to_string(world_size);
          std::cout << "[MEM_PROBE rank=" << rank << "] tag=" << tag
                    << " used_mb=" << std::fixed << std::setprecision(1)
                    << probe_used_mb << "\n";
          if (rank == 0) {
            const char *sd = std::getenv("MEM_SNAPSHOT_DIR");
            std::string snap_dir = sd ? sd : "mem_scaling_runs";
            std::filesystem::create_directories(snap_dir);
            std::string snap_path = snap_dir + "/" + tag + ".txt";
            std::ofstream sf(snap_path);
            sf << "# MEM PROBE SNAPSHOT  tag=" << tag << "\n";
            sf << "# impl=Cpp label=" << mem_label << " rotator=" << rotator_label
               << " recompute_k=" << (config.recompute_k ? 1 : 0)
               << " n_embd=" << config.n_embd << " n_layer=" << config.n_layers
               << " n_head=" << config.n_heads
               << " weight_tying=" << (config.weight_tying ? 1 : 0) << "\n";
            sf << "# B=" << B << " T=" << config.context_length
               << " cp_world_size=" << world_size << " params=" << num_params << "\n";
            sf << "# cudaMemGetInfo used_mb(rank0)=" << std::fixed
               << std::setprecision(1) << probe_used_mb << "\n";
            // Allocator breakdown (rank0): reserved = active + cached-free;
            // active = requested + internal-fragmentation (size-class rounding).
            sf << "# ALLOC reserved_mb=" << std::fixed << std::setprecision(1)
               << st_reserved_mb << " active_mb=" << st_active_mb
               << " requested_mb=" << st_requested_mb
               << " cached_free_mb=" << st_cached_free_mb
               << " internal_frag_pct=" << std::setprecision(2) << st_frag_pct
               << "\n";
            // Peak (running-max) occupancy — true max during fwd+bwd, matches
            // PyTorch's torch.peak_alloc/peak_reserved semantics.
            sf << "# ALLOC_PEAK active_peak_mb=" << std::fixed << std::setprecision(1)
               << st_active_peak_mb << " reserved_peak_mb=" << st_reserved_peak_mb
               << "\n";
            // Machine-readable live per-GPU used MiB (parsed by the table gen).
            // nvidia-smi ignores CUDA_VISIBLE_DEVICES and lists ALL physical
            // GPUs; on a shared box we keep only the GPUs this run uses (the
            // physical indices in CUDA_VISIBLE_DEVICES) so a co-tenant's memory
            // does not pollute the reported number.
            {
              std::string smi_used;
              // Build the set of allowed physical indices from the env var.
              std::vector<std::string> vis;
              if (const char *cvd = std::getenv("CUDA_VISIBLE_DEVICES")) {
                std::string s(cvd), cur;
                for (char c : s) {
                  if (c == ',') { if (!cur.empty()) vis.push_back(cur); cur.clear(); }
                  else if (c != ' ') cur += c;
                }
                if (!cur.empty()) vis.push_back(cur);
              }
              auto allowed = [&](const std::string &idx) {
                if (vis.empty()) return true;
                for (auto &v : vis) if (v == idx) return true;
                return false;
              };
              FILE *qp = popen("nvidia-smi --query-gpu=index,memory.used "
                               "--format=csv,noheader,nounits 2>/dev/null", "r");
              if (qp) {
                char buf[256];
                while (fgets(buf, sizeof(buf), qp)) {
                  std::string ln(buf);
                  while (!ln.empty() && (ln.back() == '\n' || ln.back() == '\r'))
                    ln.pop_back();
                  std::string cleaned;
                  for (char c : ln) if (c != ' ') cleaned += c;
                  if (cleaned.empty()) continue;
                  std::string idx = cleaned.substr(0, cleaned.find(','));
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
            std::string cmd = "nvidia-smi >> '" + snap_path + "' 2>&1";
            int rc = std::system(cmd.c_str());
            (void)rc;
            std::cout << "[MEM_PROBE] wrote snapshot " << snap_path << "\n";
          }
          MPI_Barrier(MPI_COMM_WORLD);
          break;
        }

        if(step==0){
        OwnTensor::CachingCUDAAllocator::instance().empty_cache();
        }

      } catch (const std::exception &e) {
        std::cerr << "EXCEPTION RANK " << rank << " STEP " << step << ": "
                  << e.what() << "\n";
        std::exit(1);
      }
    }

    if (rank == 0) {
      log_file.close();
      std::cout << "\nTraining log saved to: " << log_filename << "\n";
      std::cout << "\n=== Context Parallel Training Complete ===\n";
    }

  } catch (const std::exception &e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    MPI_Finalize();
    return 1;
  }

  MPI_Finalize();
  return 0;
}
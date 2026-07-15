# CP — Context-Parallel GPT-2 Training

Context-parallel (ring-attention) GPT-2 training built on the
[Tensor-Implementations](Tensor-Implementations) tensor library (the `OwnTensor`
runtime, included here as a git submodule).

The training entrypoint is [Scripts/BluTrain/gpt2_cp_test.cpp](Scripts/BluTrain/gpt2_cp_test.cpp).
It runs one process per GPU under MPI; each rank holds a sequence shard and the
ring rotator exchanges K/V (and dK/dV in the backward pass) across ranks.

There is also a **Llama-style** context-parallel trainer,
[Scripts/BluTrain/bluscriptCP.cpp](Scripts/BluTrain/bluscriptCP.cpp) (fused QK-norm + RoPE + GQA,
SwiGLU, RMSNorm), supporting both **Ring** and **Ulysses** attention and composing CP with
DataParallel — see [Context-parallel Llama training (bluscriptCP)](#context-parallel-llama-training-bluscriptcp).

## Repository layout

| Path | Contents |
|------|----------|
| `context_parallel/` | Ring-attention forward/backward, fused SDPA kernels, KV-pack, rotators |
| `process_group/` | `ProcessGroupNCCL` (NCCL collectives + async ring stream), `DeviceMesh` |
| `Data_Loader/` | mmap shard `DataLoader`, plus `Data/` token shards |
| `Scripts/BluTrain/` | `gpt2_cp_test.cpp` (GPT-2 CP trainer) + `bluscriptCP.cpp` (Llama-style CP trainer: Ring/Ulysses, fused RoPE+QK-norm+GQA) |
| `Scripts/Pytorch/` | `gpt2_cp_attnstyle_fp32.py` — PyTorch parity reference |
| `Tensor-Implementations/` | `OwnTensor` tensor library (git submodule → `libtensor`) |

## Requirements

- CUDA toolkit (tested with CUDA 13; `nvcc` on `PATH`)
- An MPI implementation exposing `mpic++` / `mpirun` (tested with the local OpenMPI)
- NCCL
- One or more NVIDIA GPUs. Compute capability is auto-detected (this box: 2x RTX 3060, `sm_86`)

## 1. Get the submodule

```bash
git submodule update --init --recursive
```

## 2. Build the tensor library

CP links against `Tensor-Implementations/lib/libtensor.a`. Build it once (it
delegates to the submodule's own Makefile and forwards the detected GPU arch):

```bash
make libtensor
```

## 3. Build the training binary

```bash
make            # -> build/gpt2_cp_test_exec
```

This compiles the `context_parallel/` and `process_group/` sources, then links
the binary against `libtensor`. To target a specific GPU arch explicitly:

```bash
make SM_ARCH=86      # e.g. RTX 3060 / 3090 ; use 89 for Ada (RTX 6000 Ada / 4090)
```

> Note: the main training translation unit is large and built at `-O3`. A highly
> parallel/keep-going build (`-j -k`) can get the compile OOM-killed, leaving the
> link to fail with "could not open ... .o". Build serially (the default) or use a
> small `-j` if you hit that.

## 4. Before the first run: set the data path

The token-shard directory is currently **hardcoded** in
[Scripts/BluTrain/gpt2_cp_test.cpp:890](Scripts/BluTrain/gpt2_cp_test.cpp#L890):

```cpp
std::string data_root =
    "/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/Data_Loader/Data/";
```

Point it at this repo's shards (or wherever your `.bin` token shards live), e.g.:

```cpp
std::string data_root = "Data_Loader/Data/";
```

Then rebuild (`make`). The loader expects `train`/`val` shards named `*.bin`
(uint16 tokens) in that directory.

## 5. Run

```bash
make run NP=2                 # mpirun -np 2 ./build/gpt2_cp_test_exec
make run NP=2 ARGS="..."      # pass extra args through to the binary
```

`NP` is the number of ranks / GPUs (default 2). Equivalent manual invocation:

```bash
LD_LIBRARY_PATH=Tensor-Implementations/lib:$LD_LIBRARY_PATH \
    mpirun -np 2 ./build/gpt2_cp_test_exec
```

## Runtime configuration (environment variables)

The binary is configured at runtime via env vars (no CLI flags). Common ones:

### Model / data shape
| Var | Effect |
|-----|--------|
| `CP_MODEL_44M=1` | Use the 44M preset (n_embd 384 / n_layer 3 / n_head 6) |
| `CP_N_EMBD`, `CP_N_LAYER`, `CP_N_HEAD` | Override individual model dims |
| `CP_T` | Sequence length |
| `CP_WEIGHT_TYING=0/1` | Tie/untie wte and lm_head |
| `CP_GLOBAL_BATCH` | Global batch size (tokens) |
| `CP_MAX_STEPS`, `CP_WARMUP` | Training step count / LR warmup steps |

### Context-parallel behavior
| Var | Effect |
|-----|--------|
| `CP_ROTATOR` | Ring rotator type (e.g. P2P / AllToAll / AllGather) |
| `CP_SHARE_FWD_ROTATOR=1` | Share one forward rotator across all layers (lower memory) |
| `CP_RECOMPUTE_K` | Recompute-K behavior in backward |
| `CP_NO_GRAD_ALLREDUCE=1` | Disable param-gradient all-reduce (A/B debugging) |

### Checkpoint / init
| Var | Effect |
|-----|--------|
| `LOAD_INIT_NAMED=<file>` | Load a config-tagged named-init weight dump |
| `LOAD_INIT_WEIGHTS=<file>` | Load a raw init-weights file |

### Memory / debug
| Var | Effect |
|-----|--------|
| `CP_MEM_PROBE=1`, `CP_MEM_PROBE_STEPS=<n>` | Run a short memory-scaling probe instead of full training |
| `CP_ALLOC_CSV=<file>`, `MEM_SNAPSHOT_DIR=<dir>` | Allocator breakdown / snapshot dumps |
| `CP_DEBUG_SHAPES=1`, `DUMP_FWD=1`, `DUMP_STEP0_GRADS=1` | Shape / activation / gradient dumps for parity work |

Example:

```bash
make run NP=2 ARGS="" \
  && CP_MODEL_44M=1 CP_T=2048 CP_MAX_STEPS=100 \
     LD_LIBRARY_PATH=Tensor-Implementations/lib:$LD_LIBRARY_PATH \
     mpirun -np 2 ./build/gpt2_cp_test_exec
```

(env vars must precede `mpirun`; `make run` does not forward them, so set them on
a direct `mpirun` invocation as above.)

## Make targets

```
make [all]                 Build the CP training binary (build/gpt2_cp_test_exec)
make libtensor             Build libtensor in the Tensor-Implementations submodule
make run [NP=2] [ARGS=...] mpirun the training binary on NP ranks
make CP_FUSED_ROPE=1 bluscript-cp        Build the Llama CP trainer (build/bluscriptCP_exec)
make CP_FUSED_ROPE=1 run-bluscript-cp [NP=2]  mpirun bluscriptCP on NP ranks (ring; drop the flag + CP_ATTN_MODE=ulysses for Ulysses)
make run-snippet FILE=<f>  Compile + run a single .cpp against libtensor
make run-folder  FOLDER=<d> Compile + run every .cpp in a folder
make rebuild               clean + all
make clean                 Remove CP build artifacts (not libtensor)
make print_sm              Print the detected GPU compute capability
make help                  List all targets and flags
```

## Context-parallel Llama training (bluscriptCP)

[Scripts/BluTrain/bluscriptCP.cpp](Scripts/BluTrain/bluscriptCP.cpp) is a second, **Llama-style**
trainer (distinct from `gpt2_cp_test`): fp32 model with bf16 **fused QK-norm + RoPE + grouped-query
attention**, SwiGLU MLP, RMSNorm, tied lm_head. RoPE / QK-norm / GQA run **inside** the
context-parallel layer (fused in the CP kernels), and it composes **CP with DataParallel** on the
unified `ProcessGroupNCCL`.

Two attention modes, selected by `CP_ATTN_MODE`:

- **`ring`** (default) — RingAttention via the fused-RoPE CP kernels (GQA + HeadTail load-balance).
  **Requires building with `CP_FUSED_ROPE=1`.**
- **`ulysses`** — DeepSpeed-style all-to-all sequence parallelism (GQA, contiguous shard). Works in
  the default build.

**Topology.** `CP_SIZE` (default = world size) sets the context-parallel group size;
`dp_size = world_size / CP_SIZE`. So `CP_SIZE = world_size` is pure CP (sequence split across all
ranks); a proper divisor gives 2-D **DP×CP** (data-parallel replicas each running a CP group). The
sequence is split across the CP group; DataParallel replicates params and reduces gradients across
all ranks. `world_size % CP_SIZE` must be 0.

### Build + run

```bash
# --- Ring (default). Pass CP_FUSED_ROPE=1 to BOTH build and run. ---
make CP_FUSED_ROPE=1 bluscript-cp
CP_DATA_ROOT=/path/to/shards \
    make CP_FUSED_ROPE=1 run-bluscript-cp NP=2

# --- Ulysses (default build). ---
make bluscript-cp
CP_ATTN_MODE=ulysses CP_DATA_ROOT=/path/to/shards \
    make run-bluscript-cp NP=2
```

`NP` = ranks/GPUs. **Env vars set before `make ... run-bluscript-cp` ARE forwarded to the ranks**
(OpenMPI forwards the launcher's environment); the run target sets `LD_LIBRARY_PATH` for you.

> **`CP_FUSED_ROPE` must be passed to BOTH the build and the run target for ring mode.** Ring and
> Ulysses share objects via this build flag (tracked by a flag-stamp, so no `make clean` is needed
> when you flip it). A bare `make run-bluscript-cp` re-evaluates with the flag off and rebuilds to
> the default (Ulysses-only) kernel — which then throws in ring mode.

> **Checkpoints auto-resume** from `checkpoints_bluscriptcp/`. For a fresh smoke run,
> `rm -rf checkpoints_bluscriptcp` or bump `CP_MAX_STEPS` (a completed run at the same step count
> short-circuits a rerun).

### Environment variables (bluscriptCP)

**Topology / attention**

| Var | Default | Effect |
|-----|---------|--------|
| `CP_SIZE` | world size | CP group size. `= world_size` ⇒ CP-only; divisor ⇒ 2-D DP×CP. |
| `CP_ATTN_MODE` | `ring` | `ring` (fused-RoPE ring; needs `CP_FUSED_ROPE=1` build) or `ulysses` (all-to-all). |
| `CP_ROTATOR` | `alltoall` | Ring comm strategy: `p2p` / `alltoall` / `allgather`. |

**Model / data / optimization**

| Var | Default | Effect |
|-----|---------|--------|
| `CP_N_EMBD` | 768 | `d_model`. |
| `CP_N_LAYER` | 12 | Transformer blocks. |
| `CP_N_HEAD` | 12 | Query heads. |
| `CP_N_KVHEAD` | 4 | KV heads (GQA). `q_heads % kv_heads == 0`; `head_dim = d_model/q_heads` must be **64 or 128**. |
| `CP_T` | 4096 | Sequence length (`== context_length`). Must be even; ring needs `T % (2·CP_SIZE) == 0`, ulysses `T % CP_SIZE == 0`. |
| `CP_B` | 8 | Micro-batch per rank. |
| `CP_GLOBAL_BATCH` | 524288 | Tokens/step. Must be divisible by `B·T·dp_size`; `grad_accum = CP_GLOBAL_BATCH / (B·T·dp_size)`. |
| `CP_MAX_STEPS` | 19073 | Training steps. |
| `CP_WARMUP` | 715 | LR warmup steps. |
| `CP_DATA_ROOT` | (hardcoded path) | Token-shard directory (train/val `*.bin`, uint16). |

**Checkpointing / resume / LR re-warmup** — checkpoints go to `checkpoints_bluscriptcp/blumodelcp_run<N>_step_<S>.ckpt`; the run number `N` ties the CSV log (`CP_Training_logs/CP_Training_log<N>.csv`) and the checkpoint prefix, and is echoed at startup (`Checkpointing: ON -> …`).

| Var | Default | Effect |
|-----|---------|--------|
| `CP_CKPT` | 1 | Enable periodic checkpointing (`0` to disable). |
| `CP_CKPT_FREQ` | 250 | Save every N steps (plus the final step). **First save is at step N** — with the default 250 nothing is written before then. |
| `CP_CKPT_NEW_RUN=1` | – | Force a fresh run number even if a resumable checkpoint exists. |
| `CP_CKPT_RESUME=<N>` | auto | Resume run number `N` explicitly (honored regardless of completeness); otherwise auto-resume the latest **incomplete** run (top step `< CP_MAX_STEPS`). |
| `CP_REWARMUP` | 0 | **Resume-only** (`start_step > 0`): re-anchor a fresh LR warmup of this many steps at the resume step. `0` keeps the continuous schedule (the base warmup is absolute from step 0 and cannot re-ramp). |
| `CP_REWARMUP_PEAK` | 1.0 | Peak of the re-warmup as a fraction of the run's `max_lr` (6e-4). E.g. `0.3` for a reduced long-context peak. |

**RoPE / YaRN context extension** (the shared `build_rope_cache`; `YARN_SCALE=1.0` ⇒ plain RoPE)

| Var | Default | Effect |
|-----|---------|--------|
| `YARN_SCALE` | 1.0 | Context-extension factor; set to `target_ctx / orig_ctx`. |
| `YARN_ORIG_MAXPOS` | 1024 | Original pretrain context length. |
| `YARN_BETA_FAST` / `YARN_BETA_SLOW` | 32 / 1 | NTK-by-parts correction bounds. |

`build_rope_cache` prints `[YaRN] … scale=<s>` at model init — `scale=1.000` means YaRN is **off**. Keep `CP_T == YARN_SCALE · YARN_ORIG_MAXPOS` or it warns that the cache is sized for a different context than it is scaled for.

**2-stage context extension (YaRN + re-warmup).** To extend a model from `orig_ctx` to a longer context, run two stages rather than changing `CP_T` at peak LR (which meets the context-shift perplexity spike with full-strength updates):

1. **Base stage** — train at `CP_T=orig_ctx` and let the LR **decay** (don't stop at peak); checkpoint via `CP_CKPT_FREQ`.
2. **Long-context stage** — resume with the extended `CP_T`, YaRN on, and a fresh **re-warmup to a reduced peak** so the spike is met with a gentle, rising LR:

   ```bash
   CP_T=4096 YARN_SCALE=4 YARN_ORIG_MAXPOS=1024 \
   CP_CKPT_RESUME=<run> CP_MAX_STEPS=<new_target> \
   CP_REWARMUP=<W> CP_REWARMUP_PEAK=0.3 \
       make CP_FUSED_ROPE=1 run-bluscript-cp NP=<n>
   ```

   Model params are sequence-length- and world-size-independent (RoPE positions are computed, not learned), so the extended `CP_T` — and even a different world size — loads the same checkpoint; the RoPE cache is rebuilt from the YaRN env. Keep the architecture (`d_model` / heads / kv / layers / vocab) identical. Set `CP_MAX_STEPS` above the base-stage step (or pass `CP_CKPT_RESUME`) so the run is treated as incomplete — otherwise a completed run starts fresh instead of resuming.

**Diagnostics (ring) + device mapping**

| Var | Effect |
|-----|--------|
| `CP_NO_OVERLAP=1` | Serialize fwd+bwd ring comm (A/B for overlap races). |
| `CP_NO_OVERLAP_FWD=1` / `CP_NO_OVERLAP_BWD=1` | Serialize only the forward / backward ring. |
| `CP_RING_BLOCKING=1` | Make the dedicated ring stream blocking. |
| `CP_SYNC_RING=1` | Host-sync after each ring exchange. |
| `NO_GPUS_PER_NODE=<n>` | Override GPUs-per-node for device mapping (e.g. oversubscribing ranks). |

**Build flags** (compile-time; pass to `make`, not the runtime env)

| Flag | Effect |
|------|--------|
| `CP_FUSED_ROPE=1` | Compile the fused-RoPE CP kernels. **REQUIRED for `CP_ATTN_MODE=ring`.** |
| `CP_ROPE_DEBUG=1` | Add the out-of-range RoPE-index counter (bring-up only; adds a per-call sync). |

### Example — full-scale (114M), 2 GPUs, CP-only ring

```bash
make CP_FUSED_ROPE=1 bluscript-cp
CP_DATA_ROOT=/path/to/shards CP_SIZE=2 CP_T=4096 CP_B=1 CP_GLOBAL_BATCH=4096 \
    make CP_FUSED_ROPE=1 run-bluscript-cp NP=2
# d_model 768 / 12 layers / q12-kv4 (hd 64) / T 4096, sequence split 2048/rank.
```

## PyTorch parity reference

[Scripts/Pytorch/gpt2_cp_attnstyle_fp32.py](Scripts/Pytorch/gpt2_cp_attnstyle_fp32.py)
is the reference implementation used to validate the C++ context-parallel path
(step-0 loss / gradient parity). Run it under `torchrun` with a matching config.

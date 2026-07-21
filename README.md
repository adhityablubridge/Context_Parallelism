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

Three attention modes, selected by `CP_ATTN_MODE`:

- **`ring`** (default) — RingAttention via the fused-RoPE CP kernels (GQA + HeadTail load-balance).
  **Requires building with `CP_FUSED_ROPE=1`.**
- **`ulysses`** — DeepSpeed-style all-to-all sequence parallelism (GQA, contiguous shard). Works in
  the default build.
- **`hybrid`** — 3-D **Ulysses-inner + Ring-outer** (Unified Sequence Parallelism: USP / LoongTrain).
  Ulysses (all-to-all) runs on the **inner** axis (place it intra-node / on NVLink) nested inside a
  Ring (P2P rotation) on the **outer** axis (place it inter-node). **Requires `CP_FUSED_ROPE=1`**
  (the ring stage uses the fused kernel). `CP_ULYSSES_SIZE` sets the inner degree `U`; the ring
  degree is `R = CP_SIZE / U`. v1 requires `q_heads % U == 0` **and** `kv_heads % U == 0`
  (the `U > kv_heads` kv-replication layout is not yet supported).

**Topology.** `CP_SIZE` (default = world size) sets the context-parallel group size;
`dp_size = world_size / CP_SIZE`. So `CP_SIZE = world_size` is pure CP (sequence split across all
ranks); a proper divisor gives 2-D **DP×CP** (data-parallel replicas each running a CP group). The
sequence is split across the CP group; DataParallel replicates params and reduces gradients across
all ranks. `world_size % CP_SIZE` must be 0.

**Hybrid topology (3-D `{DP, Ring, Ulysses}`).** With `CP_ATTN_MODE=hybrid`, the CP group is
factored `CP_SIZE = R × U` (`R = CP_SIZE / CP_ULYSSES_SIZE`), and the mesh is built
`{dp_size, ring_size, ulysses_size}`. Grouping is purely by **rank index** (Ulysses is the fastest
axis, so a Ulysses group is `U` consecutive ranks; Ring strides by `U`); correctness never depends
on physical placement. Which axis lands on which interconnect is the **launcher's** job:

- Use **node-major** rank placement so each Ulysses group sits on one node/NVLink island:
  `mpirun --map-by ppr:${GPUS_PER_NODE}:node` (OpenMPI) or `srun --ntasks-per-node=${GPUS_PER_NODE}`
  (SLURM). Ring then strides across nodes — Ulysses (bulk all-to-all) on the fast link, Ring (P2P,
  overlaps with compute) across the slow link. This is the recommended USP/LoongTrain placement.
- An **advisory** fast-domain guard warns (does not abort) if `NO_GPUS_PER_NODE % CP_ULYSSES_SIZE`
  is nonzero. Set `NO_GPUS_PER_NODE` to a NUMA/NVLink-**island** size (smaller than a node) to place
  Ulysses intra-island and run **Ring across NUMA** within a single node. It is a performance hint,
  not a correctness constraint.

Worked example — `DP=2, Ring=2, Ulysses=2` (8 ranks): `CP_SIZE=4`, `CP_ULYSSES_SIZE=2`,
`dp_size = 8/4 = 2`. Ulysses groups are `{0,1},{2,3},{4,5},{6,7}` (co-locate each on one island);
Ring groups stride by 2 (`{0,2},{1,3},…`); DP strides by 4.

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

# --- Hybrid (Ulysses-inner + Ring-outer). Needs CP_FUSED_ROPE=1 + node-major launch. ---
# DP=2, Ring=2, Ulysses=2 on 8 ranks (2 nodes x 4 GPUs). CP_SIZE = R*U = 4.
make CP_FUSED_ROPE=1 bluscript-cp
CP_SIZE=4 CP_ULYSSES_SIZE=2 CP_ATTN_MODE=hybrid NO_GPUS_PER_NODE=4 \
CP_DATA_ROOT=/path/to/shards \
    make CP_FUSED_ROPE=1 run-bluscript-cp NP=8
# (For explicit node-major placement, launch mpirun/srun directly with
#  --map-by ppr:4:node / --ntasks-per-node=4; see Tests/hybrid_cp_smoke.sh.)
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
| `CP_SIZE` | world size | CP group size. `= world_size` ⇒ CP-only; divisor ⇒ 2-D DP×CP. In `hybrid`, `CP_SIZE = ring_size × ulysses_size`. |
| `CP_ATTN_MODE` | `ring` | `ring` (fused-RoPE ring; needs `CP_FUSED_ROPE=1`), `ulysses` (all-to-all), or `hybrid` (Ulysses-inner + Ring-outer; needs `CP_FUSED_ROPE=1`). |
| `CP_ULYSSES_SIZE` | `1` | **hybrid only** — inner Ulysses (all-to-all) degree `U`. Ring degree `R = CP_SIZE / U`. Requires `CP_SIZE % U == 0`, `q_heads % U == 0`, `kv_heads % U == 0`. |
| `CP_RING_OUTER` | `0` | **hybrid only** — `1` builds the mesh `{ring, dp, ulysses}` (ring outermost) instead of the default `{dp, ring, ulysses}`. Makes the ring stride cross the coarse boundary so the **Ring** (P2P, overlaps) spans NUMA/nodes while **Ulysses AND DDP** stay on the fast domain. Use with DDP + ring-across-NUMA layouts. |
| `CP_ROTATOR` | `alltoall` | Ring comm strategy: `p2p` / `alltoall` / `allgather` (used by `ring` and the outer axis of `hybrid`). |
| `NO_GPUS_PER_NODE` | GPU count | Fast-domain (node / NUMA-island) size. In `hybrid`, an **advisory** guard warns if it is not a multiple of `CP_ULYSSES_SIZE`; also used for `cudaSetDevice` mapping. |

**Model / data / optimization**

| Var | Default | Effect |
|-----|---------|--------|
| `CP_N_EMBD` | 768 | `d_model`. |
| `CP_N_LAYER` | 12 | Transformer blocks. |
| `CP_N_HEAD` | 12 | Query heads. |
| `CP_N_KVHEAD` | 4 | KV heads (GQA). `q_heads % kv_heads == 0`; `head_dim = d_model/q_heads` must be **64 or 128**. In `hybrid`, also `kv_heads % CP_ULYSSES_SIZE == 0`. |
| `CP_T` | 4096 | Sequence length (`== context_length`). Must be even; ring needs `T % (2·CP_SIZE) == 0`, ulysses `T % CP_SIZE == 0`, hybrid `T % (2·ring_size) == 0`, `(T/ring_size) % ulysses_size == 0`, **and** the fused-RoPE tile `T/(2·ring_size) % 32 == 0` (i.e. `T % (64·ring_size) == 0`). |
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

**CREAM position-relabeling context extension** (single-GPU / `CP_SIZE=1` only for now)

CREAM (bigai-nlco/CREAM, *Continuity-Relativity indexing with Gaussian Middle*) extends the context window by **fine-tuning at the physical `CP_T` while feeding manipulated RoPE position labels** drawn from the larger `scaled_max = YARN_SCALE · CP_T` range: a head block `[0, head)`, a truncated-Gaussian-sampled contiguous middle block, and a tail block ending at `scaled_max`. The attention matrix stays `CP_T × CP_T` (memory cost unchanged) while the model learns rotations at large positions. It composes with YaRN: the cos/sin cache CREAM gathers from is built at `scaled_max` with YaRN active, so `YARN_SCALE > 1` is **required**.

Mechanism (no CUDA change): at `CP_SIZE=1` the fused kernel indexes `cos_sin_cache[local_idx]`, so per step CREAM gathers the cache rows for its labels (`cache[L[j]]`) and installs that `[T, hd]` cache on every layer — see [context_parallel/CreamPositions.h](context_parallel/CreamPositions.h). `CP_CREAM_MODE=off` (default) is byte-identical to prior behavior.

| Var | Default | Effect |
|-----|---------|--------|
| `CP_CREAM_MODE` | `off` | `off` / `cream` (Gaussian middle) / `pose` (2-chunk skip) / `randpos` (sorted random). Requires `CP_SIZE=1` and `YARN_SCALE > 1`. |
| `CP_CREAM_SIGMA` | 3.0 | Truncated-Gaussian width for the middle-block sampling (`cream` only). |
| `CP_CREAM_SEED` | 0 | Base seed; the per-step label seed is `CP_CREAM_SEED + step` (training) and the fixed `CP_CREAM_SEED` (eval). |
| `CP_CREAM_LOG` | – | Optional CSV path; appends `step,middle_start` per step for the truncated-Gaussian distribution sanity check. |

`factor` and `scaled_max` are **derived** (`factor = YARN_SCALE`, `scaled_max = factor · CP_T`) — there is no separate scaled-max knob, so they cannot drift. Startup asserts `YARN_SCALE > 1`, integer `factor ≥ 2`, and `scaled_max % CP_T == 0`. Note the v1 cache is shared across the batch (one labeling per step, resampled each step) and relabels the natural contiguous token chunk; per-row labels and multi-GPU CP need an explicit per-token position array in the kernel (deferred). Example:

```bash
CP_SIZE=1 CP_T=256 \
CP_CREAM_MODE=cream CP_CREAM_SIGMA=3.0 CP_CREAM_LOG=cream_mid.csv \
YARN_SCALE=8 YARN_ORIG_MAXPOS=256 \
CP_CKPT_RESUME=<base_run> CP_MAX_STEPS=<target> CP_REWARMUP=<W> CP_REWARMUP_PEAK=0.3 \
    make CP_FUSED_ROPE=1 run-bluscript-cp NP=1
```

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

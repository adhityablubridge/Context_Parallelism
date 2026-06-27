# CP — Context-Parallel GPT-2 Training

Context-parallel (ring-attention) GPT-2 training built on the
[Tensor-Implementations](Tensor-Implementations) tensor library (the `OwnTensor`
runtime, included here as a git submodule).

The training entrypoint is [Scripts/BluTrain/gpt2_cp_test.cpp](Scripts/BluTrain/gpt2_cp_test.cpp).
It runs one process per GPU under MPI; each rank holds a sequence shard and the
ring rotator exchanges K/V (and dK/dV in the backward pass) across ranks.

## Repository layout

| Path | Contents |
|------|----------|
| `context_parallel/` | Ring-attention forward/backward, fused SDPA kernels, KV-pack, rotators |
| `process_group/` | `ProcessGroupNCCL` (NCCL collectives + async ring stream), `DeviceMesh` |
| `Data_Loader/` | mmap shard `DataLoader`, plus `Data/` token shards |
| `Scripts/BluTrain/` | `gpt2_cp_test.cpp` — the training binary source |
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
make run-snippet FILE=<f>  Compile + run a single .cpp against libtensor
make run-folder  FOLDER=<d> Compile + run every .cpp in a folder
make rebuild               clean + all
make clean                 Remove CP build artifacts (not libtensor)
make print_sm              Print the detected GPU compute capability
make help                  List all targets and flags
```

## PyTorch parity reference

[Scripts/Pytorch/gpt2_cp_attnstyle_fp32.py](Scripts/Pytorch/gpt2_cp_attnstyle_fp32.py)
is the reference implementation used to validate the C++ context-parallel path
(step-0 loss / gradient parity). Run it under `torchrun` with a matching config.

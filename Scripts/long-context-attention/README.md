# USP (yunchang) hybrid Ulysses x Ring — comparison harness vs bluscriptCP

USP / yunchang (github.com/feifeibear/long-context-attention) is the one external
framework that implements the **same 2-D Ulysses x Ring hybrid** context
parallelism as bluscriptCP. This folder wraps it in a faithful transformer
block stack (same dims as our 48M / 114M configs) so we can compare **throughput,
peak memory, and max-context-before-OOM** apples-to-apples. LlamaFactory and
DeepSpeed are Ulysses-only and have no hybrid counterpart, so they are not here.

## Files
| file | purpose |
|---|---|
| `usp_model.py` | RMSNorm + GQA(USP attn) + SwiGLU block stack; sequence sharding via `EXTRACT_FUNC_DICT` |
| `usp_probe.py` | 2-step memory snapshot (writes the `.txt` format `mem_scaling_table.py` parses) |
| `train_usp_compare.py` | logged training run (CSV: step, loss, dt_ms, tok/sec, mem) for throughput comparison |

## Install (in a venv/conda with CUDA torch)
```bash
pip install yunchang flash-attn --no-build-isolation
python -c "import yunchang, flash_attn; print('ok')"
```
`flash-attn` must match your torch/CUDA; USP uses FlashAttention-2 (`AttnType.FA`).

## Systems sweep (bluscriptCP hybrid vs USP hybrid)
Driven by `../../Tests/bluscriptcp/mem_scaling_sweep_hybrid_compare.sh`:
```bash
cd ../../Tests/bluscriptcp
CLEAN=1 BUILD_CPP=1 FINE_GRAINED=1 \
RUN_CPP=1 RUN_USP=1 USP_PYTHON=python RING_IMPL=zigzag \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./mem_scaling_sweep_hybrid_compare.sh
python3 mem_scaling_table.py mem_scaling_runs_hybrid_compare
```
bluscriptCP runs on the full world (dp folding); USP runs dp=1 on the first
`cp_size` GPUs. Per-rank memory at a given `(ring, ulysses, T)` is the comparison.

## Single training-run comparison (throughput / step-time)
dp=1, launch with `nproc_per_node = ring * ulysses`:
```bash
# USP hybrid 2x2 (CP=4), 114M dims, 16k context, 200 steps
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 \
  train_usp_compare.py --label 114M_r2u2 --ring 2 --ulysses 2 --ring-impl zigzag \
  --d-model 768 --n-layer 12 --q-heads 12 --kv-heads 4 --head-dim 64 --ffn 2048 \
  --B 2 --T 16384 --steps 200 --csv logs/usp_114M_r2u2_T16384.csv
```
Compare the printed STEADY-STATE median tok/sec + peak_mem against a bluscriptCP
training run at the same `--label` / topology / `T`.

## Constraints (match bluscriptCP)
- `ulysses` must divide BOTH `q_heads` and `kv_heads` (head split). 48M kv=2 caps
  ulysses at 2; 114M kv=4 allows ulysses in {1,2,4}.
- `ring * ulysses` must divide the GPU count.
- zigzag needs `T % (2 * ring * ulysses) == 0`.

## Caveat
Random-init weights + surrogate loss: this measures SYSTEMS behavior (throughput,
memory, scaling), NOT convergence. Quality comparison lives in Track 1.

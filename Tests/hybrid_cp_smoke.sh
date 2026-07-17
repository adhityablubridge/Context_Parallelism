#!/usr/bin/env bash
# =============================================================================
# hybrid_cp_smoke.sh -- multi-node integration smoke test for the 3-D HYBRID
# context-parallel path (Ulysses-inner all-to-all + Ring-outer P2P rotation) in
# bluscriptCP.
#
# THIS SCRIPT CANNOT RUN ON A 1-2 GPU BOX. It needs >= 4 GPUs (ideally across 2
# nodes) so that ring_size * ulysses_size * dp_size = world_size is meaningful and
# the intra-node Ulysses / inter-node Ring split is actually exercised.
#
# What it does:
#   1. Builds bluscriptCP with CP_FUSED_ROPE=1 (the ring stage needs the fused
#      RoPE kernel).
#   2. Launches it with NODE-MAJOR rank placement so that each Ulysses group
#      (ulysses_size consecutive ranks) lands on ONE node/NVLink island and the
#      Ring axis strides across nodes:
#        mpirun --map-by ppr:${GPUS_PER_NODE}:node    (OpenMPI)
#        srun   --ntasks-per-node=${GPUS_PER_NODE}    (SLURM, default block)
#   3. Runs a handful of steps and checks: the run started, loss is finite, and
#      the advisory fast-domain guard did not (unexpectedly) warn.
#
# Topology is set via env:
#   CP_SIZE           total CP degree = ring_size * ulysses_size
#   CP_ULYSSES_SIZE   inner all-to-all degree (should divide GPUS_PER_NODE)
#   CP_ATTN_MODE      = hybrid
#   NO_GPUS_PER_NODE  fast-domain (NUMA/NVLink-island) size for the guard; set to
#                     the island size for ring-across-NUMA on a single node.
# ring_size is inferred as CP_SIZE / CP_ULYSSES_SIZE; dp_size = WORLD / CP_SIZE.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---- knobs (override via env) ----
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"     # GPUs per node / fast-interconnect island
NNODES="${NNODES:-1}"                   # number of nodes
WORLD_SIZE="${WORLD_SIZE:-$((GPUS_PER_NODE * NNODES))}"
CP_SIZE="${CP_SIZE:-4}"                 # ring_size * ulysses_size
CP_ULYSSES_SIZE="${CP_ULYSSES_SIZE:-2}" # intra-node all-to-all degree
STEPS="${STEPS:-5}"
# small model + short context so the smoke run is quick; T must satisfy
# T % (2*ring_size) == 0 and (T/ring_size) % ulysses_size == 0.
CP_T="${CP_T:-2048}"
CP_N_EMBD="${CP_N_EMBD:-384}"
CP_N_LAYER="${CP_N_LAYER:-4}"
CP_N_HEAD="${CP_N_HEAD:-6}"
CP_N_KVHEAD="${CP_N_KVHEAD:-2}"         # NOTE: must be divisible by CP_ULYSSES_SIZE
CP_B="${CP_B:-1}"

RING_SIZE=$(( CP_SIZE / CP_ULYSSES_SIZE ))
DP_SIZE=$(( WORLD_SIZE / CP_SIZE ))

echo "=== hybrid_cp_smoke ==="
echo "  world_size      : ${WORLD_SIZE}  (nodes=${NNODES} x gpus/node=${GPUS_PER_NODE})"
echo "  CP_SIZE         : ${CP_SIZE}  (ring=${RING_SIZE} x ulysses=${CP_ULYSSES_SIZE})"
echo "  dp_size         : ${DP_SIZE}"
echo "  model           : d=${CP_N_EMBD} L=${CP_N_LAYER} q=${CP_N_HEAD} kv=${CP_N_KVHEAD} T=${CP_T} B=${CP_B}"
if (( WORLD_SIZE < 4 )); then
  echo "ERROR: hybrid smoke needs WORLD_SIZE >= 4 (have ${WORLD_SIZE}); run on a cluster." >&2
  exit 2
fi
if (( GPUS_PER_NODE % CP_ULYSSES_SIZE != 0 )); then
  echo "WARN: GPUS_PER_NODE (${GPUS_PER_NODE}) % CP_ULYSSES_SIZE (${CP_ULYSSES_SIZE}) != 0:" \
       "a Ulysses group would straddle a node/island boundary (perf, not correctness)." >&2
fi

# ---- build ----
echo "--- building bluscriptCP (CP_FUSED_ROPE=1) ---"
make CP_FUSED_ROPE=1 bluscript-cp

EXE="${REPO_ROOT}/build/bluscriptCP_exec"
[[ -x "$EXE" ]] || { echo "ERROR: $EXE not built" >&2; exit 3; }

TENSOR_LIBDIR="${REPO_ROOT}/BluTrain/Tensor-Implementations/lib"
PROFILER_LIBDIR="${REPO_ROOT}/BluTrain/Profiler/lib"

# ---- launch (node-major placement) ----
LOG="$(mktemp)"
COMMON_ENV=(
  "CP_SIZE=${CP_SIZE}" "CP_ULYSSES_SIZE=${CP_ULYSSES_SIZE}" "CP_ATTN_MODE=hybrid"
  "NO_GPUS_PER_NODE=${GPUS_PER_NODE}"
  "CP_T=${CP_T}" "CP_N_EMBD=${CP_N_EMBD}" "CP_N_LAYER=${CP_N_LAYER}"
  "CP_N_HEAD=${CP_N_HEAD}" "CP_N_KVHEAD=${CP_N_KVHEAD}" "CP_B=${CP_B}"
  "CP_MAX_STEPS=${STEPS}" "CP_WARMUP=2" "CP_CKPT=0"
)

echo "--- launching ${WORLD_SIZE} ranks (node-major) ---"
if command -v srun >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
  # SLURM: default block distribution is node-major; pin ntasks-per-node.
  srun --ntasks="${WORLD_SIZE}" --ntasks-per-node="${GPUS_PER_NODE}" \
    bash -lc "export ${COMMON_ENV[*]} LD_LIBRARY_PATH=${TENSOR_LIBDIR}:${PROFILER_LIBDIR}:\${LD_LIBRARY_PATH:-}; exec '${EXE}'" \
    2>&1 | tee "$LOG"
else
  # OpenMPI: ppr:<gpus>:node keeps consecutive ranks on one node (node-major).
  ENV_ARGS=(); for kv in "${COMMON_ENV[@]}"; do ENV_ARGS+=(-x "$kv"); done
  LD_LIBRARY_PATH="${TENSOR_LIBDIR}:${PROFILER_LIBDIR}:${LD_LIBRARY_PATH:-}" \
  mpirun -np "${WORLD_SIZE}" --map-by "ppr:${GPUS_PER_NODE}:node" -x LD_LIBRARY_PATH \
    "${ENV_ARGS[@]}" "${EXE}" 2>&1 | tee "$LOG"
fi

# ---- checks ----
echo "--- checking smoke output ---"
rc=0
if ! grep -q "HYBRID (Ulysses-inner" "$LOG"; then
  echo "FAIL: hybrid mode banner not found (did CP_ATTN_MODE=hybrid take effect?)" >&2; rc=1
fi
if grep -Eq "NaN|Inf|ERROR:" "$LOG"; then
  echo "FAIL: NaN/Inf/ERROR encountered in the run" >&2; rc=1
fi
if ! grep -Eq "step +[0-9]+ .* loss:" "$LOG"; then
  echo "FAIL: no training-step lines with a loss value" >&2; rc=1
fi
if grep -q "\[CP WARN\] fast-domain" "$LOG"; then
  echo "NOTE: fast-domain guard warned -- Ulysses group straddles an island (perf hint)." >&2
fi
if (( rc == 0 )); then echo "PASS: hybrid smoke run started, produced finite losses over ${STEPS} steps."; fi
rm -f "$LOG"
exit "$rc"

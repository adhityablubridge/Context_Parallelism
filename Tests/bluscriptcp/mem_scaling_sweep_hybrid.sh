#!/usr/bin/env bash
# =============================================================================
# mem_scaling_sweep_hybrid.sh   (bluscriptCP  HYBRID  Ring x Ulysses 2-D CP)
# -----------------------------------------------------------------------------
# Memory-occupancy scaling sweep for the 3-D hybrid context-parallel path
# (CP_ATTN_MODE=hybrid: Ulysses-inner all-to-all nested in Ring-outer P2P). It
# sweeps sequence length T (auto-doubling until OOM) across a MATRIX of
# (ring_size x ulysses_size) factorizations, so you can see how peak GPU memory
# and max-T-before-OOM change as you trade ring degree for ulysses degree at a
# fixed GPU count.
#
# Unlike mem_scaling_sweep.sh (which pairs pure-Ulysses bluscriptCP against
# DeepSpeed-Ulysses), this script is bluscriptCP-only and drives the fused-RoPE
# ring, so it MUST build with CP_FUSED_ROPE=1.
#
# TOPOLOGY. For a given GPU count (world_size = #CUDA_VISIBLE_DEVICES):
#   CP_SIZE = ring_size * ulysses_size      dp_size = world_size / CP_SIZE
# Each TOPOLOGIES row is "RING ULYSSES RING_OUTER". Rows that don't divide the
# world size, or whose ulysses degree doesn't divide the model's q/kv heads, are
# skipped with a message. RING_OUTER=1 builds the {ring,dp,ulysses} mesh so the
# ring crosses the coarse (NUMA/node) boundary while ulysses+DDP stay local.
#
# 4-GPU and 8-GPU: run twice, changing CUDA_VISIBLE_DEVICES (the TOPOLOGIES list
# below already contains rows for both; invalid ones self-skip):
#   CUDA_VISIBLE_DEVICES=0,1,2,3        ./mem_scaling_sweep_hybrid.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./mem_scaling_sweep_hybrid.sh
#
# Each probe is a short 2-step bluscriptCP run (CP_MEM_PROBE=1: grad_accum=1, no
# validation/checkpoints) that writes CPP_<label>_<rotator>_T<T>_ws<ws>.txt
# itself (cudaMemGetInfo used + caching-allocator peak + nvidia-smi). The
# topology is baked into <label> (e.g. 48M_r2u2 / 48M_r2u2ro) so the table keeps
# each factorization on its own row.
#
# Build the table afterwards:
#   python3 mem_scaling_table.py mem_scaling_runs_hybrid
# =============================================================================
set -u  # (no -e: we WANT to continue after an OOM run fails)

# ============================ EDIT THIS BLOCK ================================

# Which GPUs to use. world_size = number of devices listed here.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# Build the C++ binary once before sweeping (1=yes). Set 0 if already built.
BUILD_CPP="${BUILD_CPP:-1}"

# Repo paths. This script sits in CP/Tests/bluscriptcp/, so REPO_ROOT is two up.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CPP_EXEC="${REPO_ROOT}/build/bluscriptCP_exec"
TENSOR_LIBDIR="${REPO_ROOT}/BluTrain/Tensor-Implementations/lib"
PROFILER_LIBDIR="${REPO_ROOT}/BluTrain/Profiler/lib"
DATA_ROOT="${CP_DATA_ROOT:-${REPO_ROOT}/Data_Loader/Data}"

# Fast-domain (NUMA/NVLink-island) size for the advisory guard AND rank->GPU
# binding. On a single node this is just the GPU count; set it to the NUMA-island
# size only in combination with RING_OUTER rows for ring-across-NUMA layouts.
NO_GPUS_PER_NODE_ENV="${NO_GPUS_PER_NODE:-}"

# Output: per-run snapshots + the results CSV.
OUT_DIR="${SCRIPT_DIR}/mem_scaling_runs_hybrid"

# T sweep: start, doubling, with a hard cap so it cannot loop forever.
T_START="${T_START:-2048}"
T_MAX="${T_MAX:-262144}"

# Fine-grained boundary search after the coarse doubling finds [last_OK, OOM].
FINE_GRAINED="${FINE_GRAINED:-0}"
FINE_STEP="${FINE_STEP:-256}"

# Per-run wall-clock timeout (seconds). Kills a hung run and treats it as fail.
RUN_TIMEOUT="${RUN_TIMEOUT:-1200}"

# Probe step count. >=2 so optimizer state is allocated/counted.
PROBE_STEPS="${PROBE_STEPS:-2}"

# Micro-batch per rank (B).
B="${B:-2}"

# Ring rotator for the outer axis (p2p|alltoall|allgather). p2p overlaps best.
CP_ROTATOR_ENV="${CP_ROTATOR:-p2p}"

# Topology matrix: "RING ULYSSES RING_OUTER".  CP_SIZE = RING*ULYSSES.
# Rows are auto-skipped when RING*ULYSSES does not divide world_size, or when
# ULYSSES does not divide the model's q_heads/kv_heads.
TOPOLOGIES=(
  "1 2 0"   # pure Ulysses (CP_SIZE=2)
  "2 1 0"   # pure Ring    (CP_SIZE=2)
  "2 2 0"   # hybrid 2x2   (CP_SIZE=4)   [4GPU: dp1 | 8GPU: dp2]
  "2 2 1"   # hybrid 2x2, RING-OUTER (ring across NUMA)  [8GPU: dp2]
  "2 4 0"   # hybrid 2x4   (CP_SIZE=8)   [8GPU: dp1]  (needs ulysses|q,kv -> wide-head cfg)
  "4 2 0"   # hybrid 4x2   (CP_SIZE=8)   [8GPU: dp1]  (ring degree 4)
  "4 1 0"   # pure Ring 4  (CP_SIZE=4)
  "8 1 0"   # pure Ring 8  (CP_SIZE=8)
)

# Model config matrix: "LABEL D_MODEL N_LAYER Q_HEADS KV_HEADS FFN TYING".
# head_dim = D_MODEL/Q_HEADS must be 64 or 128. FFN is honored via CP_FFN (auto
# = 8/3*D_MODEL if left as -1). For ULYSSES=4 the model needs q_heads%4==0 AND
# kv_heads%4==0, so include a wide-head row (114M: q12/kv4).
CONFIGS=(
  "48M    384   6    6    2    1024   0"
  "114M   768   12   12   4    2048   1"
)

# ============================ END EDIT BLOCK ================================

WORLD_SIZE="$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')"
# Fast-domain default = world size (single node) unless the user set one.
FAST_DOMAIN="${NO_GPUS_PER_NODE_ENV:-$WORLD_SIZE}"

# ---- Stale-output guard ----------------------------------------------------
if [[ -d "$OUT_DIR" ]]; then
  stale_count=$(find "$OUT_DIR" -maxdepth 1 -name '*.txt' 2>/dev/null | wc -l)
  if (( stale_count > 0 )); then
    if [[ "${CLEAN:-0}" == "1" ]]; then
      echo "[GUARD] CLEAN=1 -> removing $stale_count stale snapshot(s) in $OUT_DIR"
      rm -f "$OUT_DIR"/*.txt "$OUT_DIR"/mem_scaling_results.csv \
            "$OUT_DIR"/mem_scaling_table.csv "$OUT_DIR"/mem_scaling_table.md \
            "$OUT_DIR"/mem_scaling_limits.csv "$OUT_DIR"/mem_scaling_limits.md
    elif [[ "${FORCE:-0}" == "1" ]]; then
      echo "[GUARD] FORCE=1 -> proceeding into non-empty $OUT_DIR ($stale_count stale snapshots)"
    else
      echo "ERROR: $OUT_DIR already has $stale_count snapshot(s) from a prior sweep." >&2
      echo "       Re-run with CLEAN=1 (wipe) or FORCE=1 (append), or change OUT_DIR." >&2
      exit 1
    fi
  fi
fi

mkdir -p "$OUT_DIR"
RESULTS_CSV="${OUT_DIR}/mem_scaling_results.csv"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "$LOG_DIR"
echo "impl,label,rotator,n_embd,n_layer,n_head,weight_tying,T,world_size,status,snapshot" \
  > "$RESULTS_CSV"

echo "=============================================================="
echo " HYBRID (Ring x Ulysses) memory scaling sweep -- bluscriptCP"
echo "   CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  (world_size=$WORLD_SIZE, fast_domain=$FAST_DOMAIN)"
echo "   T: $T_START -> doubling -> <=$T_MAX   PROBE_STEPS=$PROBE_STEPS  B=$B  rotator=$CP_ROTATOR_ENV"
echo "   topologies: ${#TOPOLOGIES[@]}   configs: ${#CONFIGS[@]}"
echo "   DATA_ROOT=$DATA_ROOT"
echo "   OUT_DIR=$OUT_DIR"
echo "=============================================================="

# ---- Build C++ once (config/T/topology are runtime env) --------------------
# Hybrid uses the fused-RoPE ring kernel -> MUST build with CP_FUSED_ROPE=1.
if [[ "$BUILD_CPP" == "1" ]]; then
  echo "[BUILD] make CP_FUSED_ROPE=1 bluscript-cp (once) -> $CPP_EXEC ..."
  ( cd "$REPO_ROOT" && make CP_FUSED_ROPE=1 bluscript-cp ) 2>&1 | tee "${LOG_DIR}/build_cpp.log" | tail -8
fi
if [[ ! -x "$CPP_EXEC" ]]; then
  echo "[BUILD] ERROR: $CPP_EXEC not found. Build first (make CP_FUSED_ROPE=1 bluscript-cp)." >&2
  exit 1
fi
if [[ ! -d "$DATA_ROOT" ]]; then
  echo "ERROR: DATA_ROOT $DATA_ROOT not found. Set CP_DATA_ROOT." >&2
  exit 1
fi

# is_fail <logfile> <exit_code> <snapshot_path>  -> 0 (true) if failed/OOM.
is_fail() {
  local log="$1" rc="$2" snap="$3"
  if [[ "$rc" != "0" ]]; then return 0; fi
  if [[ ! -s "$snap" ]]; then return 0; fi
  if grep -qiE "out of memory|cuda(.{0,8})?out of memory|cudaErrorMemoryAllocation|bad_alloc|CUDA error|RuntimeError|terminate called" "$log"; then
    return 0
  fi
  return 1
}

# run_one_hybrid  label d l qh kvh ffn ty  ring uly ring_outer  T
run_one_hybrid() {
  local label="$1" d="$2" l="$3" qh="$4" kvh="$5" ffn="$6" ty="$7"
  local ring="$8" uly="$9" ro="${10}" T="${11}"
  local cp_size=$(( ring * uly ))
  local rolab=""; [[ "$ro" == "1" ]] && rolab="ro"
  # Topology baked into the model label so each factorization is a distinct row.
  local mlabel="${label}_r${ring}u${uly}${rolab}"
  local tag="CPP_${mlabel}_${CP_ROTATOR_ENV}_T${T}_ws${WORLD_SIZE}"
  local snap="${OUT_DIR}/${tag}.txt" log="${LOG_DIR}/${tag}.log"
  rm -f "$snap"
  echo "  -> [HYB] $mlabel (d=$d L=$l qh=$qh kvh=$kvh ffn=$ffn tie=$ty) ring=$ring uly=$uly ro=$ro dp=$(( WORLD_SIZE / cp_size ))  T=$T"
  CP_MEM_PROBE=1 CP_MEM_PROBE_STEPS="$PROBE_STEPS" CP_MODEL_LABEL="$mlabel" \
  CP_ATTN_MODE=hybrid CP_SIZE="$cp_size" CP_ULYSSES_SIZE="$uly" CP_RING_OUTER="$ro" \
  CP_ROTATOR="$CP_ROTATOR_ENV" NO_GPUS_PER_NODE="$FAST_DOMAIN" CP_B="$B" \
  CP_N_EMBD="$d" CP_N_LAYER="$l" CP_N_HEAD="$qh" CP_N_KVHEAD="$kvh" \
  CP_FFN="$ffn" CP_WEIGHT_TYING="$ty" CP_T="$T" \
  CP_CKPT=0 CP_DATA_ROOT="$DATA_ROOT" MEM_SNAPSHOT_DIR="$OUT_DIR" \
  LD_LIBRARY_PATH="${TENSOR_LIBDIR}:${PROFILER_LIBDIR}:${LD_LIBRARY_PATH:-}" \
  timeout "$RUN_TIMEOUT" mpirun -x LD_LIBRARY_PATH -np "$WORLD_SIZE" "$CPP_EXEC" \
    > "$log" 2>&1
  local rc=$?
  if is_fail "$log" "$rc" "$snap"; then
    echo "     FAIL/OOM (rc=$rc) -- see $log"
    echo "CPP,${mlabel},${CP_ROTATOR_ENV},${d},${l},${qh},${ty},${T},${WORLD_SIZE},OOM,${snap}" >> "$RESULTS_CSV"
    return 1
  fi
  echo "     OK -- snapshot $snap"
  echo "CPP,${mlabel},${CP_ROTATOR_ENV},${d},${l},${qh},${ty},${T},${WORLD_SIZE},OK,${snap}" >> "$RESULTS_CSV"
  return 0
}

# T feasibility for a (ring,uly): HeadTail (2*ring), inner slice ((T/ring)%uly),
# and the fused-RoPE tile (T/(2*ring) multiple of 32 => T % (64*ring) == 0).
t_ok() {  # ring uly T
  local ring="$1" uly="$2" T="$3"
  (( T % (2 * ring) == 0 )) || return 1
  (( (T / ring) % uly == 0 )) || return 1
  (( T % (64 * ring) == 0 )) || return 1
  return 0
}

sweep_topo() {  # label d l qh kvh ffn ty  ring uly ring_outer
  local label="$1" d="$2" l="$3" qh="$4" kvh="$5" ffn="$6" ty="$7"
  local ring="$8" uly="$9" ro="${10}"
  local cp_size=$(( ring * uly ))
  echo ""
  echo "==== $label  ring=$ring x uly=$uly (CP_SIZE=$cp_size, dp=$(( WORLD_SIZE / cp_size ))) ring_outer=$ro ===="
  # Topology feasibility vs GPU count + model heads.
  if (( WORLD_SIZE % cp_size != 0 )); then
    echo "  !! SKIP: CP_SIZE=$cp_size does not divide world_size=$WORLD_SIZE"; return
  fi
  if (( qh % uly != 0 )) || (( kvh % uly != 0 )); then
    echo "  !! SKIP: ulysses=$uly must divide q_heads=$qh AND kv_heads=$kvh"; return
  fi
  local last_ok=0 oom_T=0 T=$T_START
  while (( T <= T_MAX )); do
    if ! t_ok "$ring" "$uly" "$T"; then
      echo "  (skip T=$T: fails 2*ring / (T/ring)%uly / 64*ring tile)"; T=$(( T * 2 )); continue
    fi
    if run_one_hybrid "$label" "$d" "$l" "$qh" "$kvh" "$ffn" "$ty" "$ring" "$uly" "$ro" "$T"; then
      last_ok=$T; T=$(( T * 2 ))
    else
      oom_T=$T
      echo "  -> coarse OOM for $label ring=$ring uly=$uly at T=$oom_T (last OK=$last_ok)"; break
    fi
  done

  if [[ "$FINE_GRAINED" == "1" ]] && (( last_ok > 0 && oom_T > last_ok )); then
    echo "  ~~ fine search in ($last_ok, $oom_T), step=$FINE_STEP"
    local lo=$last_ok hi=$oom_T mid step=$(( 64 * ring ))   # keep tile-aligned
    while (( hi - lo > FINE_STEP )); do
      mid=$(( (lo + hi) / 2 )); mid=$(( (mid / step) * step ))
      if (( mid <= lo || mid >= hi )) || ! t_ok "$ring" "$uly" "$mid"; then break; fi
      if run_one_hybrid "$label" "$d" "$l" "$qh" "$kvh" "$ffn" "$ty" "$ring" "$uly" "$ro" "$mid"; then lo=$mid; else hi=$mid; fi
    done
    echo "  == TRUE max-T for $label ring=$ring uly=$uly: $lo OK, OOM by $hi"
  fi
}

for row in "${CONFIGS[@]}"; do
  read -r label d l qh kvh ffn ty <<< "$row"
  echo ""
  echo "######## CONFIG: $label (d_model=$d n_layer=$l q_heads=$qh kv_heads=$kvh ffn=$ffn tie=$ty) ########"
  for topo in "${TOPOLOGIES[@]}"; do
    read -r ring uly ro <<< "$topo"
    sweep_topo "$label" "$d" "$l" "$qh" "$kvh" "$ffn" "$ty" "$ring" "$uly" "$ro"
  done
done

echo ""
echo "=============================================================="
echo " Hybrid sweep complete."
echo "   Snapshots : $OUT_DIR/*.txt"
echo "   Results   : $RESULTS_CSV"
echo "   Build a table:  python3 ${SCRIPT_DIR}/mem_scaling_table.py $OUT_DIR"
echo "=============================================================="

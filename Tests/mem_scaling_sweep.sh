#!/usr/bin/env bash
# =============================================================================
# mem_scaling_sweep.sh
# -----------------------------------------------------------------------------
# Memory-occupancy scaling sweep for the Context-Parallel GPT-2 models.
# Sweeps sequence length T (auto-doubling until OOM) across several parameter
# configs, for BOTH the C++ (BluTrain) and PyTorch implementations, running a
# 2-step probe per run and saving a labeled nvidia-smi snapshot for each.
#
# IMPORTANT: the model config and T are passed via ENVIRONMENT VARIABLES, which
# both scripts read at runtime. The C++ binary is therefore built ONCE and
# reused for every config/T -- no rebuild per run.
#
# Usage:
#   ./mem_scaling_sweep.sh
# Then build the table:
#   python3 mem_scaling_table.py mem_scaling_runs
# =============================================================================
set -u  # (no -e: we WANT to continue after an OOM run fails)

# ============================ EDIT THIS BLOCK ================================

# Which GPUs to use. world_size = number of devices listed here.
export CUDA_VISIBLE_DEVICES="4,5"

# Implementations to run (1=yes, 0=no). They never run concurrently.
RUN_PT=1
RUN_CPP=1

# Build the C++ binary once before sweeping (1=yes). Set 0 if already built.
BUILD_CPP=1

# Repo paths. This script sits in CP/Tests/, so the repo root is one level up.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PT_SCRIPT="${REPO_ROOT}/Scripts/Pytorch/gpt2_cp_attnstyle_fp32.py"
CPP_EXEC="${REPO_ROOT}/build/gpt2_cp_test_exec"

# Output: per-run snapshots + the results CSV.
OUT_DIR="${SCRIPT_DIR}/mem_scaling_runs_fine"

# T sweep: start, doubling, with a hard cap so it cannot loop forever.
T_START=1024
T_MAX=131072

# Fine-grained boundary search. After the coarse doubling finds the failing
# octave [last_OK, OOM], binary-search between them to pin the TRUE max-T cutoff
# (not just the power-of-2 step). FINE_STEP = resolution in tokens; the search
# stops when the OK/OOM bracket is within FINE_STEP. Values are rounded to a
# multiple of world_size. Set FINE_GRAINED=0 to keep only the coarse doubling.
FINE_GRAINED=1
FINE_STEP=128

# RESUME (fine-only) mode. When RESUME_FINE_ONLY=1, SKIP the coarse doubling
# pass entirely and jump straight to the binary search using a KNOWN bracket
# (last-OK, first-OOM) taken from a previous sweep's limits table. Provide the
# bracket as two extra columns on each CONFIGS row: ... LO HI.
#   - LO must be a T you already know runs OK.
#   - HI must be a T you already know OOMs.
#   - Brackets are hardware-specific (edit per machine).
# With RESUME_FINE_ONLY=0 the LO/HI columns are ignored and the normal coarse
# doubling runs first.
# RESUME_FINE_ONLY=0

# Per-run wall-clock timeout (seconds). Kills a hung run and treats it as fail.
RUN_TIMEOUT=900

# # Config matrix:  "LABEL  N_EMBD  N_LAYER  N_HEAD  WEIGHT_TYING(1/0)  [LO  HI]"
# # (vocab is fixed at 50304 in both scripts; weight tying just drops the lm_head.)
# # The optional LO/HI columns are ONLY used when RESUME_FINE_ONLY=1.
# CONFIGS=(
#   "25M   384  3   6   1"
#   "44M   384  3   6   0"
#   "124M  768  12  12  1"
#   "163M  768  12  12  0"
#   # "350M  1024 24  16  0"   # uncomment to push further toward OOM on big GPUs
# )

RESUME_FINE_ONLY=0
CONFIGS=(
  "25M   384  3   6   1   4096  8192"
  "44M   384  3   6   0   4096  8192"
  "124M  768  12  12  1   2048  4096"
  "163M  768  12  12  0   2048  4096"
)



# Rotators to sweep per implementation.
#   PyTorch CP supports 2:  alltoall, allgather
#   C++ BluTrain supports 3: p2p, alltoall, allgather
PT_ROTATORS=(alltoall allgather)
CPP_ROTATORS=(p2p alltoall allgather)

# Launchers (override if your cluster needs different flags).
TORCHRUN="torchrun --standalone --nnodes=1"
MPIRUN="mpirun"

# ============================ END EDIT BLOCK ================================

WORLD_SIZE="$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')"

# ---- Stale-output guard ----------------------------------------------------
# Re-running into a folder that already holds snapshots from a previous sweep
# mixes stale (possibly differently-measured) rows into the table and produces
# physically impossible non-monotonic OK/OOM results. Refuse unless the folder
# is empty of prior snapshots, or the user explicitly opts in:
#   FORCE=1     -> proceed anyway (append/overwrite into the existing folder)
#   CLEAN=1     -> wipe the snapshot files first, then proceed
if [[ -d "$OUT_DIR" ]]; then
  stale_count=$(find "$OUT_DIR" -maxdepth 1 -name '*.txt' 2>/dev/null | wc -l)
  if (( stale_count > 0 )); then
    if [[ "${CLEAN:-0}" == "1" ]]; then
      echo "[GUARD] CLEAN=1 -> removing $stale_count stale snapshot(s) in $OUT_DIR"
      rm -f "$OUT_DIR"/*.txt "$OUT_DIR"/mem_scaling_results.csv \
            "$OUT_DIR"/mem_scaling_table.csv "$OUT_DIR"/mem_scaling_table.md
    elif [[ "${FORCE:-0}" == "1" ]]; then
      echo "[GUARD] FORCE=1 -> proceeding into non-empty $OUT_DIR ($stale_count stale snapshots; results may be mixed)"
    else
      echo "ERROR: $OUT_DIR already has $stale_count snapshot(s) from a prior sweep." >&2
      echo "       Mixing them into a new run yields invalid (non-monotonic) tables." >&2
      echo "       Re-run with one of:" >&2
      echo "         CLEAN=1 ./mem_scaling_sweep.sh   # wipe old snapshots, then sweep" >&2
      echo "         FORCE=1 ./mem_scaling_sweep.sh   # proceed anyway (not recommended)" >&2
      echo "       Or point OUT_DIR at a fresh folder." >&2
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
echo " Memory scaling sweep"
echo "   CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  (world_size=$WORLD_SIZE)"
echo "   RUN_PT=$RUN_PT  RUN_CPP=$RUN_CPP  BUILD_CPP=$BUILD_CPP"
echo "   T: $T_START -> doubling -> <=$T_MAX"
echo "   OUT_DIR=$OUT_DIR"
echo "=============================================================="

# ---- Build C++ once (config/T are runtime env, so one build serves all) ----
if [[ "$RUN_CPP" == "1" && "$BUILD_CPP" == "1" ]]; then
  echo "[BUILD] make (once) -> build/gpt2_cp_test_exec ..."
  ( cd "$REPO_ROOT" && make ) 2>&1 | tee "${LOG_DIR}/build_cpp.log" | tail -5
  if [[ ! -x "$CPP_EXEC" ]]; then
    echo "[BUILD] ERROR: $CPP_EXEC not found after build. Disabling C++ runs."
    RUN_CPP=0
  fi
fi

# is_oom <logfile> <exit_code> <snapshot_path>
# Returns 0 (true=OOM/fail) if the run failed or produced no snapshot.
is_fail() {
  local log="$1" rc="$2" snap="$3"
  if [[ "$rc" != "0" ]]; then return 0; fi
  if [[ ! -s "$snap" ]]; then return 0; fi
  if grep -qiE "out of memory|cuda(.{0,8})?out of memory|cudaErrorMemoryAllocation|bad_alloc|CUDA error|RuntimeError|terminate called" "$log"; then
    return 0
  fi
  return 1
}

run_one() {  # impl rotator label e l h ty T
  local impl="$1" rot="$2" label="$3" e="$4" l="$5" h="$6" ty="$7" T="$8"
  local tag log rc snap
  if [[ "$impl" == "PT" ]]; then
    tag="PT_${label}_${rot}_T${T}_ws${WORLD_SIZE}"
  else
    tag="CPP_${label}_${rot}_T${T}_ws${WORLD_SIZE}"
  fi
  snap="${OUT_DIR}/${tag}.txt"
  log="${LOG_DIR}/${tag}.log"
  rm -f "$snap"
  echo "  -> [$impl] $label  rotator=$rot  T=$T  (tag=$tag)"

  if [[ "$impl" == "PT" ]]; then
    MEM_PROBE=1 MEM_PROBE_STEPS=2 MODEL_LABEL="$label" ROTATE_METHOD="$rot" \
    N_EMBD="$e" N_LAYER="$l" N_HEAD="$h" WEIGHT_TYING="$ty" T="$T" \
    MEM_SNAPSHOT_DIR="$OUT_DIR" \
    timeout "$RUN_TIMEOUT" $TORCHRUN --nproc-per-node="$WORLD_SIZE" "$PT_SCRIPT" \
      > "$log" 2>&1
    rc=$?
  else
    CP_MEM_PROBE=1 CP_MEM_PROBE_STEPS=2 CP_MODEL_LABEL="$label" CP_ROTATOR="$rot" \
    CP_N_EMBD="$e" CP_N_LAYER="$l" CP_N_HEAD="$h" CP_WEIGHT_TYING="$ty" CP_T="$T" \
    MEM_SNAPSHOT_DIR="$OUT_DIR" \
    timeout "$RUN_TIMEOUT" $MPIRUN -np "$WORLD_SIZE" "$CPP_EXEC" \
      > "$log" 2>&1
    rc=$?
  fi

  if is_fail "$log" "$rc" "$snap"; then
    echo "     FAIL/OOM (rc=$rc) -- see $log"
    echo "${impl},${label},${rot},${e},${l},${h},${ty},${T},${WORLD_SIZE},OOM,${snap}" >> "$RESULTS_CSV"
    return 1
  fi
  echo "     OK -- snapshot $snap"
  echo "${impl},${label},${rot},${e},${l},${h},${ty},${T},${WORLD_SIZE},OK,${snap}" >> "$RESULTS_CSV"
  return 0
}

sweep_impl() {  # impl  rotator...
  local impl="$1"; shift
  local rotators=("$@")
  local rot row label e l h ty T
  echo ""
  echo "######## IMPLEMENTATION: $impl ########"
  for rot in "${rotators[@]}"; do
    echo ""
    echo "++++++++ rotator: $rot ++++++++"
    for row in "${CONFIGS[@]}"; do
      read -r label e l h ty lo_in hi_in <<< "$row"
      echo ""
      echo "==== $impl/$rot config $label (n_embd=$e n_layer=$l n_head=$h tying=$ty) ===="
      local last_ok=0 oom_T=0
      if [[ "$RESUME_FINE_ONLY" == "1" ]]; then
        # ---- Resume: skip coarse, seed the bracket from the CONFIGS LO/HI ----
        if [[ -z "$lo_in" || -z "$hi_in" ]]; then
          echo "  !! RESUME_FINE_ONLY=1 but no LO/HI on row '$row' -- skipping"
          continue
        fi
        last_ok=$lo_in
        oom_T=$hi_in
        echo "  >> resume: seeded bracket (last OK=$last_ok, OOM=$oom_T) from table"
      else
        # ---- Coarse pass: double T until OOM, tracking the OK/OOM bracket ----
        T=$T_START
        while (( T <= T_MAX )); do
          if (( T % WORLD_SIZE != 0 )); then
            echo "  (skip T=$T: not divisible by world_size=$WORLD_SIZE)"
            T=$(( T * 2 )); continue
          fi
          if run_one "$impl" "$rot" "$label" "$e" "$l" "$h" "$ty" "$T"; then
            last_ok=$T
            T=$(( T * 2 ))
          else
            oom_T=$T
            echo "  -> coarse OOM for $impl/$rot/$label at T=$oom_T (last OK=$last_ok)"
            break
          fi
        done
      fi

      # ---- Fine pass: binary-search the true cutoff in (last_ok, oom_T) ----
      # Only when we have a real bracket (the config OOM'd above some OK T).
      if [[ "$FINE_GRAINED" == "1" ]] && (( last_ok > 0 && oom_T > last_ok )); then
        echo "  ~~ fine search for $impl/$rot/$label in ($last_ok, $oom_T), step=$FINE_STEP"
        local lo=$last_ok hi=$oom_T mid
        while (( hi - lo > FINE_STEP )); do
          mid=$(( (lo + hi) / 2 ))
          mid=$(( (mid / WORLD_SIZE) * WORLD_SIZE ))   # snap to world_size multiple
          if (( mid <= lo || mid >= hi )); then break; fi
          if run_one "$impl" "$rot" "$label" "$e" "$l" "$h" "$ty" "$mid"; then
            lo=$mid
          else
            hi=$mid
          fi
        done
        echo "  == TRUE max-T for $impl/$rot/$label: $lo OK, OOM by $hi (res ${FINE_STEP})"
      fi
    done
  done
}

[[ "$RUN_PT"  == "1" ]] && sweep_impl "PT"  "${PT_ROTATORS[@]}"
[[ "$RUN_CPP" == "1" ]] && sweep_impl "CPP" "${CPP_ROTATORS[@]}"

echo ""
echo "=============================================================="
echo " Sweep complete."
echo "   Snapshots : $OUT_DIR/*.txt"
echo "   Results   : $RESULTS_CSV"
echo "   Build a table:  python3 ${SCRIPT_DIR}/mem_scaling_table.py $OUT_DIR"
echo "=============================================================="

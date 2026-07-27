#!/usr/bin/env bash
# =============================================================================
# mem_scaling_sweep_hybrid_compare.sh
#   bluscriptCP HYBRID (Ring x Ulysses)  vs  USP / yunchang HYBRID (Ring x Ulysses)
# -----------------------------------------------------------------------------
# The ONLY external framework that implements the same 2-D Ulysses x Ring hybrid
# context parallelism as bluscriptCP is USP (github.com/feifeibear/long-context-
# attention, pip package `yunchang`). LlamaFactory/DeepSpeed are Ulysses-only, so
# they have no row here -- this sweep is bluscriptCP vs USP only.
#
# For each (config, ring x ulysses) it T-doubles to OOM (+ optional fine search)
# and writes snapshots mem_scaling_table.py parses:
#   bluscriptCP -> CPP_<label>_<rotator>_T<T>_ws<ws>.txt   (impl=Cpp, full world, dp folding)
#   USP         -> USP_<label>_<rotator>_T<T>_ws<ws>.txt   (impl=Usp, dp=1, world=cp_size)
# Same (label, rotator) on both so the table lines up CPP vs USP per topology.
#
# NOTE ON GPU USAGE: bluscriptCP runs on the full world (dp=world/cp); USP runs
# dp=1 on the first cp_size visible GPUs. PER-RANK memory at a given (ring,uly,T)
# is what we compare -- dp replicas don't change per-rank attention memory.
#
# Build the table:  python3 mem_scaling_table.py mem_scaling_runs_hybrid_compare
# =============================================================================
set -u

# ============================ EDIT THIS BLOCK ================================
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
RUN_CPP="${RUN_CPP:-1}"
RUN_USP="${RUN_USP:-1}"
BUILD_CPP="${BUILD_CPP:-1}"           # bluscriptCP hybrid needs CP_FUSED_ROPE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CPP_EXEC="${REPO_ROOT}/build/bluscriptCP_exec"
TENSOR_LIBDIR="${REPO_ROOT}/BluTrain/Tensor-Implementations/lib"
PROFILER_LIBDIR="${REPO_ROOT}/BluTrain/Profiler/lib"
DATA_ROOT="${CP_DATA_ROOT:-${REPO_ROOT}/Data_Loader/Data}"

# USP arm: python with yunchang + flash-attn, and the script dir holding usp_probe.py.
USP_PYTHON="${USP_PYTHON:-python}"
USP_DIR="${USP_DIR:-${REPO_ROOT}/Scripts/long-context-attention}"
RING_IMPL="${RING_IMPL:-zigzag}"      # basic|zigzag|stripe

OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/mem_scaling_runs_hybrid_compare}"
T_START="${T_START:-2048}"
T_MAX="${T_MAX:-262144}"
FINE_GRAINED="${FINE_GRAINED:-0}"
FINE_STEP="${FINE_STEP:-256}"
RUN_TIMEOUT="${RUN_TIMEOUT:-1200}"
PROBE_STEPS="${PROBE_STEPS:-2}"
B="${B:-2}"
CP_ROTATOR_ENV="${CP_ROTATOR:-p2p}"
NO_GPUS_PER_NODE_ENV="${NO_GPUS_PER_NODE:-}"

# Topology matrix: "RING ULYSSES RING_OUTER".  CP_SIZE = RING*ULYSSES.
TOPOLOGIES=(
  "1 2 0"   # pure Ulysses (CP=2)
  "2 1 0"   # pure Ring    (CP=2)
  "2 2 0"   # hybrid 2x2   (CP=4)
  "2 2 1"   # hybrid 2x2 ring-outer (CP=4)
  "2 4 0"   # hybrid 2x4   (CP=8)  (needs uly|q,kv -> 114M)
  "4 2 0"   # hybrid 4x2   (CP=8)
  "4 1 0"   # pure Ring 4  (CP=4)
  "8 1 0"   # pure Ring 8  (CP=8)
)

# Model config matrix: "LABEL D_MODEL N_LAYER Q_HEADS KV_HEADS FFN TYING". head_dim=D/Q.
CONFIGS=(
  "48M    384   6    6    2    1024   0"
  "114M   768   12   12   4    2048   1"
)
# ============================ END EDIT BLOCK ================================

WORLD_SIZE="$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')"
FAST_DOMAIN="${NO_GPUS_PER_NODE_ENV:-$WORLD_SIZE}"
IFS=',' read -r -a GPU_ARR <<< "$CUDA_VISIBLE_DEVICES"

# ---- Stale-output guard ----
if [[ -d "$OUT_DIR" ]]; then
  stale_count=$(find "$OUT_DIR" -maxdepth 1 -name '*.txt' 2>/dev/null | wc -l)
  if (( stale_count > 0 )); then
    if [[ "${CLEAN:-0}" == "1" ]]; then
      echo "[GUARD] CLEAN=1 -> removing $stale_count stale snapshot(s) in $OUT_DIR"
      rm -f "$OUT_DIR"/*.txt "$OUT_DIR"/mem_scaling_results.csv \
            "$OUT_DIR"/mem_scaling_table.csv "$OUT_DIR"/mem_scaling_table.md \
            "$OUT_DIR"/mem_scaling_limits.csv "$OUT_DIR"/mem_scaling_limits.md
    elif [[ "${FORCE:-0}" == "1" ]]; then
      echo "[GUARD] FORCE=1 -> proceeding into non-empty $OUT_DIR"
    else
      echo "ERROR: $OUT_DIR already has $stale_count snapshot(s). Use CLEAN=1 or FORCE=1." >&2
      exit 1
    fi
  fi
fi
mkdir -p "$OUT_DIR"; LOG_DIR="${OUT_DIR}/logs"; mkdir -p "$LOG_DIR"
RESULTS_CSV="${OUT_DIR}/mem_scaling_results.csv"
echo "impl,label,rotator,n_embd,n_layer,n_head,weight_tying,T,world_size,status,snapshot" > "$RESULTS_CSV"

echo "=============================================================="
echo " HYBRID compare:  bluscriptCP  vs  USP/yunchang  (Ring x Ulysses)"
echo "   CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (world_size=$WORLD_SIZE)"
echo "   RUN_CPP=$RUN_CPP  RUN_USP=$RUN_USP (ring_impl=$RING_IMPL)"
echo "   T: $T_START -> x2 -> <=$T_MAX  PROBE_STEPS=$PROBE_STEPS  B=$B"
echo "   OUT_DIR=$OUT_DIR"
echo "=============================================================="

# ---- Build bluscriptCP once (fused-RoPE ring) ----
if [[ "$RUN_CPP" == "1" && "$BUILD_CPP" == "1" ]]; then
  echo "[BUILD] make CP_FUSED_ROPE=1 bluscript-cp ..."
  ( cd "$REPO_ROOT" && make CP_FUSED_ROPE=1 bluscript-cp ) 2>&1 | tee "${LOG_DIR}/build_cpp.log" | tail -8
fi
if [[ "$RUN_CPP" == "1" && ! -x "$CPP_EXEC" ]]; then
  echo "[CPP] ERROR: $CPP_EXEC missing. Disabling CPP."; RUN_CPP=0
fi
# ---- USP preflight ----
if [[ "$RUN_USP" == "1" ]]; then
  if ! command -v "$USP_PYTHON" >/dev/null 2>&1; then
    echo "[USP] ERROR: USP_PYTHON ('$USP_PYTHON') not found. Disabling USP."; RUN_USP=0
  elif ! "$USP_PYTHON" -c "import yunchang, flash_attn" 2>/dev/null; then
    echo "[USP] ERROR: '$USP_PYTHON' cannot import yunchang+flash_attn. pip install yunchang flash-attn. Disabling USP."; RUN_USP=0
  elif [[ ! -f "$USP_DIR/usp_probe.py" ]]; then
    echo "[USP] ERROR: $USP_DIR/usp_probe.py missing. Disabling USP."; RUN_USP=0
  fi
fi

is_fail() {  # log rc snap
  local log="$1" rc="$2" snap="$3"
  [[ "$rc" != "0" ]] && return 0
  [[ ! -s "$snap" ]] && return 0
  grep -qiE "out of memory|cudaErrorMemoryAllocation|bad_alloc|CUDA error|RuntimeError|terminate called" "$log" && return 0
  return 1
}

# t_ok ring uly T : bluscriptCP fused-RoPE tile + USP zigzag both satisfied by T % (64*ring)==0 and (T/ring)%uly==0.
t_ok() {
  local ring="$1" uly="$2" T="$3"
  (( T % (2 * ring) == 0 )) || return 1
  (( (T / ring) % uly == 0 )) || return 1
  (( T % (64 * ring) == 0 )) || return 1
  local cp=$(( ring * uly ))
  (( T % (2 * cp) == 0 )) || return 1   # USP zigzag needs seqlen % (2*cp)==0
  return 0
}

run_cpp() {  # label d l qh kvh ffn ty ring uly ro T
  local label="$1" d="$2" l="$3" qh="$4" kvh="$5" ffn="$6" ty="$7" ring="$8" uly="$9" ro="${10}" T="${11}"
  local cp=$(( ring * uly )) rolab=""; [[ "$ro" == "1" ]] && rolab="ro"
  local mlabel="${label}_r${ring}u${uly}${rolab}"
  local tag="CPP_${mlabel}_${CP_ROTATOR_ENV}_T${T}_ws${WORLD_SIZE}"
  local snap="${OUT_DIR}/${tag}.txt" log="${LOG_DIR}/${tag}.log"; rm -f "$snap"
  echo "  -> [CPP] $mlabel ring=$ring uly=$uly ro=$ro dp=$(( WORLD_SIZE / cp )) T=$T"
  CP_MEM_PROBE=1 CP_MEM_PROBE_STEPS="$PROBE_STEPS" CP_MODEL_LABEL="$mlabel" \
  CP_ATTN_MODE=hybrid CP_SIZE="$cp" CP_ULYSSES_SIZE="$uly" CP_RING_OUTER="$ro" \
  CP_ROTATOR="$CP_ROTATOR_ENV" NO_GPUS_PER_NODE="$FAST_DOMAIN" CP_B="$B" \
  CP_N_EMBD="$d" CP_N_LAYER="$l" CP_N_HEAD="$qh" CP_N_KVHEAD="$kvh" \
  CP_FFN="$ffn" CP_WEIGHT_TYING="$ty" CP_T="$T" \
  CP_CKPT=0 CP_DATA_ROOT="$DATA_ROOT" MEM_SNAPSHOT_DIR="$OUT_DIR" \
  LD_LIBRARY_PATH="${TENSOR_LIBDIR}:${PROFILER_LIBDIR}:${LD_LIBRARY_PATH:-}" \
  timeout "$RUN_TIMEOUT" mpirun -x LD_LIBRARY_PATH -np "$WORLD_SIZE" "$CPP_EXEC" > "$log" 2>&1
  local rc=$?
  if is_fail "$log" "$rc" "$snap"; then
    echo "CPP,${mlabel},${CP_ROTATOR_ENV},${d},${l},${qh},${ty},${T},${WORLD_SIZE},OOM,${snap}" >> "$RESULTS_CSV"; return 1
  fi
  echo "CPP,${mlabel},${CP_ROTATOR_ENV},${d},${l},${qh},${ty},${T},${WORLD_SIZE},OK,${snap}" >> "$RESULTS_CSV"; return 0
}

run_usp() {  # label d l qh kvh ffn ty ring uly ro T
  local label="$1" d="$2" l="$3" qh="$4" kvh="$5" ffn="$6" ty="$7" ring="$8" uly="$9" ro="${10}" T="${11}"
  local cp=$(( ring * uly )) rolab=""; [[ "$ro" == "1" ]] && rolab="ro"
  local mlabel="${label}_r${ring}u${uly}${rolab}"
  local hd=$(( d / qh ))
  local tag="USP_${mlabel}_${CP_ROTATOR_ENV}_T${T}_ws${cp}"
  local snap="${OUT_DIR}/${tag}.txt" log="${LOG_DIR}/${tag}.log"; rm -f "$snap"
  # USP runs dp=1 on the first cp visible GPUs.
  local usp_gpus; usp_gpus="$(IFS=,; echo "${GPU_ARR[*]:0:cp}")"
  echo "  -> [USP] $mlabel ring=$ring uly=$uly (dp=1, gpus=$usp_gpus, impl=$RING_IMPL) T=$T"
  ( cd "$USP_DIR" && CUDA_VISIBLE_DEVICES="$usp_gpus" \
    timeout "$RUN_TIMEOUT" "$USP_PYTHON" -m torch.distributed.run --nproc_per_node="$cp" \
      --master_port="$(( 29500 + RANDOM % 2000 ))" usp_probe.py \
      --label "$mlabel" --ring "$ring" --ulysses "$uly" --ring-impl "$RING_IMPL" \
      --d-model "$d" --n-layer "$l" --q-heads "$qh" --kv-heads "$kvh" --head-dim "$hd" \
      --ffn "$ffn" --tie "$ty" --B "$B" --T "$T" --steps "$PROBE_STEPS" \
      --rotator "$CP_ROTATOR_ENV" --out-dir "$OUT_DIR" ) > "$log" 2>&1
  local rc=$?
  if is_fail "$log" "$rc" "$snap"; then
    echo "USP,${mlabel},${CP_ROTATOR_ENV},${d},${l},${qh},${ty},${T},${cp},OOM,${snap}" >> "$RESULTS_CSV"; return 1
  fi
  echo "USP,${mlabel},${CP_ROTATOR_ENV},${d},${l},${qh},${ty},${T},${cp},OK,${snap}" >> "$RESULTS_CSV"; return 0
}

sweep_arm() {  # which(cpp|usp) label d l qh kvh ffn ty ring uly ro
  local which="$1" label="$2" d="$3" l="$4" qh="$5" kvh="$6" ffn="$7" ty="$8" ring="$9" uly="${10}" ro="${11}"
  local cp=$(( ring * uly ))
  if (( WORLD_SIZE % cp != 0 )); then echo "  !! SKIP $which/$label r$ring u$uly: CP=$cp does not divide world=$WORLD_SIZE"; return; fi
  if (( qh % uly != 0 )) || (( kvh % uly != 0 )); then echo "  !! SKIP $which/$label r$ring u$uly: uly=$uly must divide q=$qh AND kv=$kvh"; return; fi
  local last_ok=0 oom_T=0 T=$T_START runner="run_${which}"
  while (( T <= T_MAX )); do
    if ! t_ok "$ring" "$uly" "$T"; then echo "  (skip T=$T: fails tile/shard feasibility)"; T=$(( T * 2 )); continue; fi
    if "$runner" "$label" "$d" "$l" "$qh" "$kvh" "$ffn" "$ty" "$ring" "$uly" "$ro" "$T"; then
      last_ok=$T; T=$(( T * 2 ))
    else
      oom_T=$T; echo "  -> coarse OOM $which/$label r$ring u$uly at T=$oom_T (last OK=$last_ok)"; break
    fi
  done
  if [[ "$FINE_GRAINED" == "1" ]] && (( last_ok > 0 && oom_T > last_ok )); then
    echo "  ~~ fine search $which/$label r$ring u$uly in ($last_ok, $oom_T)"
    local lo=$last_ok hi=$oom_T mid step=$(( 64 * ring ))
    while (( hi - lo > FINE_STEP )); do
      mid=$(( (lo + hi) / 2 )); mid=$(( (mid / step) * step ))
      (( mid <= lo || mid >= hi )) && break
      t_ok "$ring" "$uly" "$mid" || break
      if "$runner" "$label" "$d" "$l" "$qh" "$kvh" "$ffn" "$ty" "$ring" "$uly" "$ro" "$mid"; then lo=$mid; else hi=$mid; fi
    done
    echo "  == TRUE max-T $which/$label r$ring u$uly: $lo OK, OOM by $hi"
  fi
}

for row in "${CONFIGS[@]}"; do
  read -r label d l qh kvh ffn ty <<< "$row"
  echo ""; echo "######## CONFIG: $label (d=$d L=$l q=$qh kv=$kvh ffn=$ffn tie=$ty) ########"
  for topo in "${TOPOLOGIES[@]}"; do
    read -r ring uly ro <<< "$topo"
    echo ""; echo "==== $label  ring=$ring x uly=$uly (CP=$(( ring * uly ))) ring_outer=$ro ===="
    [[ "$RUN_CPP" == "1" ]] && sweep_arm cpp "$label" "$d" "$l" "$qh" "$kvh" "$ffn" "$ty" "$ring" "$uly" "$ro"
    [[ "$RUN_USP" == "1" ]] && sweep_arm usp "$label" "$d" "$l" "$qh" "$kvh" "$ffn" "$ty" "$ring" "$uly" "$ro"
  done
done

echo ""; echo "=============================================================="
echo " Hybrid compare sweep complete."
echo "   Snapshots : $OUT_DIR/*.txt   Results: $RESULTS_CSV"
echo "   Table     : python3 ${SCRIPT_DIR}/mem_scaling_table.py $OUT_DIR"
echo "=============================================================="

#!/usr/bin/env bash
# =============================================================================
# run_longrope_pipeline.sh -- CP-enabled LongRoPE / YaRN context-extension pipeline.
#   1. wait for the base checkpoint (step >= BASE_MAX_STEPS) + free CP GPUs
#   2. (LongRoPE arm only) evolutionary search -> longrope_best_s<S>_<mvar>.txt
#   3. fine-tune the base with the arm's cache (COMPOSE + ARM select how)
#   4. eval the arm at the target length on HELD-OUT windows -> ppl_<arm>_x<S>_<compose>_<mvar>.csv
#
# Multi-GPU (Context Parallelism): put CP_SIZE=N in ARCH and pass GPUS=0,1,..,N-1.
# run_bin launches `mpirun -np N` with all CP GPUs visible (cudaSetDevice(rank)
# binds rank->GPU). Requires a SINGLE DP replica (world_size == CP_SIZE).
#
#   ARM=longrope MSCALE={0|1}   -> searched cache; MSCALE=1 bakes YaRN's temperature
#   ARM=yarn     YARN_M={m|nom} -> formula cache (skips search); m=published, nom=no temperature
#
# Launch (SEQUENTIAL -- each run uses all CP GPUs; never background two at once):
#   ARM=longrope MSCALE=0 S=32 GPUS=0,1,2,3 CP_ATTN_MODE=ring \
#   ARCH="CP_SIZE=4 CP_ATTN_MODE=ring CP_N_EMBD=384 CP_N_LAYER=6 CP_N_HEAD=6 CP_N_KVHEAD=2 CP_FFN=1024 CP_WEIGHT_TYING=0" \
#   ... bash Tests/bluscriptcp/run_longrope_pipeline.sh 2>&1 | tee lr_s32.log
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> repo root (CP/)

# ---- config (edit or override via env) --------------------------------------
CKPT_DIR=${CKPT_DIR:-checkpoints_bluscriptcp}
PREFIX=${PREFIX:-blumodelcp}
ARCH=${ARCH:-"CP_SIZE=1 CP_N_EMBD=384 CP_N_LAYER=6 CP_N_HEAD=6 CP_N_KVHEAD=2 CP_FFN=1024 CP_WEIGHT_TYING=0"}
GBATCH=${GBATCH:-131072}          # tokens/step. s=64 (T=262144) REQUIRES GBATCH=262144.
BASE_MAX_STEPS=${BASE_MAX_STEPS:-1000}
PHYS_T=${PHYS_T:-4096}            # physical/base context length
S=${S:-4}                        # extension factor. integer >= 2
HEAD_DIM=${HEAD_DIM:-64}
ORIG_MAXPOS=${ORIG_MAXPOS:-$PHYS_T}
DATA_ROOT=${DATA_ROOT:-/home/blu-bridge25/CP/Data_Loader/Data}
GPUS=${GPUS:-${GPU:-0}}          # comma list of CP GPUs (e.g. 0,1,2,3). #ranks derived from CP_SIZE.
GPU=${GPUS%%,*}                  # first GPU (single-index nvidia-smi wait)
FT_RUN=${FT_RUN:-80}             # starting run number (auto-increments to first free)
FT_STEPS=${FT_STEPS:-300}        # fine-tune steps. Halve at s=64 to token-match s=32.
FT_B=${FT_B:-1}
EVAL_WINDOWS=${EVAL_WINDOWS:-32}
COMPOSE=${COMPOSE:-alone}        # alone | cream | none
CP_REWARMUP=${CP_REWARMUP:-100}      # NOW overridable (halve to 50 at s=64 to match warmup fraction)
CP_REWARMUP_PEAK=${CP_REWARMUP_PEAK:-0.3}
MSCALE=${MSCALE:-0}              # LongRoPE: 1 bakes YaRN's temperature m
ARM=${ARM:-longrope}            # longrope | yarn
[ "$MSCALE" = "1" ] && ARCH="$ARCH CP_LONGROPE_MSCALE=1"

# search knobs (right-sized; see plan for paper-scale). Overridable.
POP=${POP:-16}; N1=${N1:-8}; N2=${N2:-8}; ITERS=${ITERS:-20}; CALIB=${CALIB:-3}
PATIENCE=${PATIENCE:-4}; TOL=${TOL:-0.05}
SEARCH_BUDGET_SEC=${SEARCH_BUDGET_SEC:-30600}

# ---- derive #ranks from CP_SIZE in ARCH -------------------------------------
NP=1; for kv in $ARCH; do case "$kv" in CP_SIZE=*) NP=${kv#CP_SIZE=};; esac; done

# ---- arm / m-variant tag + YaRN no-m gate -----------------------------------
if [ "$ARM" = "yarn" ]; then
  YARN_M=${YARN_M:-m}                                   # m (published) | nom
  [ "$YARN_M" = "nom" ] && YNM="YARN_NO_MSCALE=1" || YNM=""
  MVAR=$YARN_M
else
  YNM=""
  [ "$MSCALE" = "1" ] && MVAR=m || MVAR=nom
fi
BEST=longrope_best_s${S}_${MVAR}.txt                    # LongRoPE search output (per S + m-variant)
OUT=${OUT:-ppl_${ARM}_x${S}_${COMPOSE}_${MVAR}.csv}     # arm + m-variant tagged (no cross-arm overwrite)

TARGET_T=$(( PHYS_T * S ))
export LD_LIBRARY_PATH=BluTrain/Tensor-Implementations/lib:BluTrain/Profiler/lib:${LD_LIBRARY_PATH:-}
EXE=./build/bluscriptCP_exec

log(){ echo "[$(date '+%H:%M:%S')] $*"; }
run_bin(){ env "$@" CUDA_VISIBLE_DEVICES=$GPUS mpirun -np "$NP" "$EXE"; }
wait_gpu_free(){  # block until EVERY CP GPU has >= 20 GB free
  local ids="${GPUS//,/ }"
  while :; do
    local ok=1
    for g in $ids; do
      local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" | tr -d ' ')
      [ "${free:-0}" -lt 20000 ] && ok=0
    done
    [ "$ok" = "1" ] && break
    log "waiting for CP GPUs [$GPUS] to free (>=20GB each) ..."; sleep 20
  done
}

# ---- 1. wait for base checkpoint at step >= BASE_MAX_STEPS -------------------
BASE_RUN=${BASE_RUN:-}
if [ -n "$BASE_RUN" ]; then
  log "using explicit BASE_RUN=$BASE_RUN"
  while :; do
    BASE_CKPT=$(ls -1t "$CKPT_DIR"/${PREFIX}_run${BASE_RUN}_step_*.ckpt 2>/dev/null | head -1 || true)
    [ -n "$BASE_CKPT" ] && break
    log "waiting for run $BASE_RUN checkpoint ..."; sleep 30
  done
else
  log "waiting for base checkpoint (step >= $BASE_MAX_STEPS) in $CKPT_DIR ..."
  BASE_CKPT=""
  while :; do
    latest=$(ls -1t "$CKPT_DIR"/${PREFIX}_run*_step_*.ckpt 2>/dev/null | head -1 || true)
    if [ -n "$latest" ]; then
      step=$(sed -E 's/.*_step_([0-9]+)\.ckpt/\1/' <<<"$latest")
      if [ "${step:-0}" -ge "$BASE_MAX_STEPS" ]; then BASE_CKPT="$latest"; break; fi
    fi
    sleep 30
  done
  BASE_RUN=$(sed -E "s#.*/${PREFIX}_run([0-9]+)_step_.*#\1#" <<<"$BASE_CKPT")
fi
log "base done: run $BASE_RUN  ($BASE_CKPT)  [arm=$ARM mvar=$MVAR NP=$NP GPUS=$GPUS]"
wait_gpu_free

# ---- 2. LongRoPE search (LongRoPE arm only; YaRN uses the formula cache) -----
if [ "$ARM" = "longrope" ]; then
  log "LongRoPE search: target=$TARGET_T s=$S NP=$NP (POP=$POP ITERS=$ITERS CALIB=$CALIB) -> $BEST"
  python3 Tests/bluscriptcp/longrope_search.py \
      --exec "$EXE" --ckpt-run "$BASE_RUN" \
      --target-t "$TARGET_T" --s "$S" --head-dim "$HEAD_DIM" --orig-maxpos "$ORIG_MAXPOS" \
      --data-root "$DATA_ROOT" --cuda-devices "$GPUS" --arch "$ARCH" \
      --pop "$POP" --n1 "$N1" --n2 "$N2" --iters "$ITERS" --calib-windows "$CALIB" \
      --patience "$PATIENCE" --tol "$TOL" \
      --time-budget-sec "$SEARCH_BUDGET_SEC" --out "$BEST"
  log "search done -> $BEST"; wait_gpu_free
else
  log "ARM=yarn: skipping search (formula cache via YARN_SCALE=$S)"
fi

# ---- 3. fine-tune (COMPOSE + ARM select how) --------------------------------
if [ "$COMPOSE" = "none" ]; then
  log "COMPOSE=none: skipping fine-tune -- eval the BASE ckpt (run $BASE_RUN) with the arm's cache"
  EVAL_RUN=$BASE_RUN
else
  while ls "$CKPT_DIR"/${PREFIX}_run${FT_RUN}_* >/dev/null 2>&1 || [ "$FT_RUN" = "$BASE_RUN" ]; do
    FT_RUN=$(( FT_RUN + 1 ))
  done
  log "fine-tuned checkpoint will be run $FT_RUN (first free >= start)"
  log "branching base run $BASE_RUN -> $FT_RUN"
  for f in "$CKPT_DIR"/${PREFIX}_run${BASE_RUN}_*; do cp "$f" "${f/run${BASE_RUN}/run${FT_RUN}}"; done
  FT_MAX=$(( BASE_MAX_STEPS + FT_STEPS ))
  EVAL_RUN=$FT_RUN
  if [ "$ARM" = "yarn" ]; then
    log "fine-tune YaRN ($MVAR) at T=$TARGET_T run $FT_RUN -> step $FT_MAX"
    run_bin $ARCH $YNM CP_T=$TARGET_T CP_B=$FT_B CP_GLOBAL_BATCH=$GBATCH \
        YARN_SCALE=$S YARN_ORIG_MAXPOS=$ORIG_MAXPOS \
        CP_CKPT=1 CP_CKPT_RESUME=$FT_RUN CP_MAX_STEPS=$FT_MAX \
        CP_REWARMUP=$CP_REWARMUP CP_REWARMUP_PEAK=$CP_REWARMUP_PEAK CP_DATA_ROOT=$DATA_ROOT
  elif [ "$COMPOSE" = "alone" ]; then
    log "fine-tune LongRoPE-ALONE ($MVAR) at full T=$TARGET_T run $FT_RUN -> step $FT_MAX"
    run_bin $ARCH CP_T=$TARGET_T CP_B=$FT_B CP_GLOBAL_BATCH=$GBATCH \
        CP_LONGROPE_FACTORS=$BEST \
        CP_CKPT=1 CP_CKPT_RESUME=$FT_RUN CP_MAX_STEPS=$FT_MAX \
        CP_REWARMUP=$CP_REWARMUP CP_REWARMUP_PEAK=$CP_REWARMUP_PEAK CP_DATA_ROOT=$DATA_ROOT
  else  # cream (single-GPU composition; NP must be 1)
    log "fine-tune LongRoPE+CREAM at 4k physical run $FT_RUN -> step $FT_MAX"
    run_bin $ARCH CP_T=$PHYS_T CP_B=2 CP_GLOBAL_BATCH=$GBATCH \
        CP_CREAM_MODE=cream CP_CREAM_SIGMA=3.0 \
        YARN_SCALE=$S YARN_ORIG_MAXPOS=$ORIG_MAXPOS CP_LONGROPE_FACTORS=$BEST \
        CP_CKPT=1 CP_CKPT_RESUME=$FT_RUN CP_MAX_STEPS=$FT_MAX \
        CP_REWARMUP=$CP_REWARMUP CP_REWARMUP_PEAK=$CP_REWARMUP_PEAK CP_DATA_ROOT=$DATA_ROOT
  fi
  log "fine-tune done (run $FT_RUN)"; wait_gpu_free
fi

# ---- 4. eval on HELD-OUT windows [CALIB, CALIB+EVAL_WINDOWS) -----------------
# Both arms score the SAME held-out windows (skip = the search's CALIB) so the
# search's calibration windows are never in the reported PPL.
log "eval PPL-vs-position at T=$TARGET_T (run $EVAL_RUN, arm=$ARM, $MVAR, skip=$CALIB) -> $OUT"
if [ "$ARM" = "yarn" ]; then
  run_bin $ARCH $YNM CP_EVAL_PPL=1 CP_T=$TARGET_T CP_B=1 CP_EVAL_WINDOWS=$EVAL_WINDOWS \
      CP_EVAL_SKIP_WINDOWS=$CALIB YARN_SCALE=$S YARN_ORIG_MAXPOS=$ORIG_MAXPOS \
      CP_CKPT_RESUME=$EVAL_RUN CP_EVAL_OUT=$OUT CP_DATA_ROOT=$DATA_ROOT
else
  run_bin $ARCH CP_EVAL_PPL=1 CP_T=$TARGET_T CP_B=1 CP_EVAL_WINDOWS=$EVAL_WINDOWS \
      CP_EVAL_SKIP_WINDOWS=$CALIB CP_LONGROPE_FACTORS=$BEST \
      CP_CKPT_RESUME=$EVAL_RUN CP_EVAL_OUT=$OUT CP_DATA_ROOT=$DATA_ROOT
fi
log "DONE. wrote $OUT  (overlay with the 4k-tail reference in graph_bluscript_cp.ipynb)"

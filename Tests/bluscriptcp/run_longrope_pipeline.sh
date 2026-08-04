#!/usr/bin/env bash
# =============================================================================
# run_longrope_pipeline.sh -- wait for the base run to finish, then run the full
# LongRoPE pipeline automatically:
#   1. wait for the base checkpoint (step >= BASE_MAX_STEPS) + a free GPU
#   2. LongRoPE evolutionary search  -> longrope_best.txt
#   3. branch base -> FT_RUN, CREAM-composed fine-tune (cheap 4k-physical)
#   4. eval the LongRoPE arm at the target length -> ppl_longrope_x<S>.csv
#
# Launch it NOW (while the base trains); it blocks on step 1 until the base is done.
#   nohup bash Tests/bluscriptcp/run_longrope_pipeline.sh > longrope_pipeline.log 2>&1 &
#   tail -f longrope_pipeline.log
#
# Override any UPPER_CASE var via env, e.g.:  S=8 POP=32 ITERS=20 bash ...run_longrope_pipeline.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> repo root (CP/)

# ---- config (edit or override via env) --------------------------------------
CKPT_DIR=${CKPT_DIR:-checkpoints_bluscriptcp}
PREFIX=${PREFIX:-blumodelcp}
ARCH=${ARCH:-"CP_SIZE=1 CP_N_EMBD=384 CP_N_LAYER=6 CP_N_HEAD=6 CP_N_KVHEAD=2 CP_FFN=1024 CP_WEIGHT_TYING=0"}
GBATCH=${GBATCH:-131072}          # tokens/step (keep same as base for comparability)
BASE_MAX_STEPS=${BASE_MAX_STEPS:-1000}
PHYS_T=${PHYS_T:-4096}            # physical training length (base context)
S=${S:-4}                        # extension factor (4k -> 16k). MUST be integer >= 2
HEAD_DIM=${HEAD_DIM:-64}
ORIG_MAXPOS=${ORIG_MAXPOS:-$PHYS_T}
DATA_ROOT=${DATA_ROOT:-/home/blu-bridge25/CP/Data_Loader/Data}   # clean shard dir (15 train + val)
GPU=${GPU:-0}
FT_RUN=${FT_RUN:-80}             # start numbering new fine-tuned checkpoints here (auto-increments if taken)
FT_STEPS=${FT_STEPS:-300}        # fine-tune steps on top of base
FT_B=${FT_B:-1}                  # micro-batch for the fine-tune (1 for long-physical alone)
EVAL_WINDOWS=${EVAL_WINDOWS:-32}
# COMPOSE = how the fine-tune uses the searched cache:
#   alone : LongRoPE-alone, fine-tune at the FULL target length (faithful; needs memory)
#   cream : LongRoPE cache as CREAM gather source, fine-tune at 4k physical (cheap)
#   none  : NO fine-tune -- eval the base ckpt directly with the searched cache (search-only)
COMPOSE=${COMPOSE:-alone}
# MSCALE=1 bakes YaRN's attention temperature (m=0.1*ln(s)+1) into the LongRoPE
# cache -> a LongRoPE+m hybrid for an apples-to-apples fight vs YaRN. 0 = paper-
# faithful (no temperature). Applied consistently across search + fine-tune + eval.
MSCALE=${MSCALE:-0}
[ "$MSCALE" = "1" ] && ARCH="$ARCH CP_LONGROPE_MSCALE=1"
MTAG=""; [ "$MSCALE" = "1" ] && MTAG="_m"
# search knobs -- RIGHT-SIZED for this model, not the paper's 7B/512x scale.
# head_dim/2 search dims (32 here) + PI/NTK/YaRN seeds converge fast, so a small
# GA with EARLY-STOPPING runs only as much as the model needs. PATIENCE stops when
# best PPL stalls; SEARCH_BUDGET_SEC is a hard wall-clock ceiling as a backstop.
# (For a paper-scale search set POP=64 ITERS=40 N1=16 N2=16 PATIENCE=0.)
POP=${POP:-16}; N1=${N1:-8}; N2=${N2:-8}; ITERS=${ITERS:-20}; CALIB=${CALIB:-3}
PATIENCE=${PATIENCE:-4}; TOL=${TOL:-0.05}
SEARCH_BUDGET_SEC=${SEARCH_BUDGET_SEC:-30600}   # 8.5h hard ceiling (early-stop usually hits first)

TARGET_T=$(( PHYS_T * S ))       # search/eval target length (= factor*T for the composed path)
export LD_LIBRARY_PATH=BluTrain/Tensor-Implementations/lib:BluTrain/Profiler/lib:${LD_LIBRARY_PATH:-}
EXE=./build/bluscriptCP_exec

log(){ echo "[$(date '+%H:%M:%S')] $*"; }
run_bin(){ env "$@" CUDA_VISIBLE_DEVICES=$GPU mpirun -np 1 "$EXE"; }
wait_gpu_free(){  # block until GPU has >= 4 GB free
  while :; do
    local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
    [ "${free:-0}" -ge 4000 ] && break
    log "waiting for GPU $GPU to free (free=${free}MB) ..."; sleep 20
  done
}

# ---- 1. wait for base checkpoint at step >= BASE_MAX_STEPS -------------------
# BASE_RUN can be set explicitly (e.g. BASE_RUN=30); otherwise auto-detect the
# newest checkpoint whose top step >= BASE_MAX_STEPS.
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
log "base done: run $BASE_RUN  ($BASE_CKPT)"
wait_gpu_free

# ---- 2. LongRoPE search -----------------------------------------------------
log "LongRoPE search: target=$TARGET_T s=$S  (POP=$POP ITERS=$ITERS CALIB=$CALIB)"
python3 Tests/bluscriptcp/longrope_search.py \
    --exec "$EXE" --ckpt-run "$BASE_RUN" \
    --target-t "$TARGET_T" --s "$S" --head-dim "$HEAD_DIM" --orig-maxpos "$ORIG_MAXPOS" \
    --data-root "$DATA_ROOT" --cuda-devices "$GPU" --arch "$ARCH" \
    --pop "$POP" --n1 "$N1" --n2 "$N2" --iters "$ITERS" --calib-windows "$CALIB" \
    --patience "$PATIENCE" --tol "$TOL" \
    --time-budget-sec "$SEARCH_BUDGET_SEC" --out longrope_best.txt
log "search done -> longrope_best.txt"; wait_gpu_free

# ---- 3. fine-tune with the searched cache (COMPOSE selects how) --------------
if [ "$COMPOSE" = "none" ]; then
  log "COMPOSE=none: skipping fine-tune -- will eval the BASE ckpt (run $BASE_RUN) with the searched cache"
  EVAL_RUN=$BASE_RUN
else
  # never overwrite an existing checkpoint: bump FT_RUN to the first free run number
  # (>= the starting value, default 80). Also avoids colliding with the base run.
  while ls "$CKPT_DIR"/${PREFIX}_run${FT_RUN}_* >/dev/null 2>&1 || [ "$FT_RUN" = "$BASE_RUN" ]; do
    FT_RUN=$(( FT_RUN + 1 ))
  done
  log "fine-tuned checkpoint will be run $FT_RUN (first free >= start)"
  log "branching base run $BASE_RUN -> $FT_RUN"
  for f in "$CKPT_DIR"/${PREFIX}_run${BASE_RUN}_*; do cp "$f" "${f/run${BASE_RUN}/run${FT_RUN}}"; done
  FT_MAX=$(( BASE_MAX_STEPS + FT_STEPS ))
  EVAL_RUN=$FT_RUN
  if [ "$COMPOSE" = "alone" ]; then
    log "fine-tune LongRoPE-ALONE at full length T=$TARGET_T (no CREAM) run $FT_RUN -> step $FT_MAX"
    run_bin $ARCH CP_T=$TARGET_T CP_B=$FT_B CP_GLOBAL_BATCH=$GBATCH \
        CP_LONGROPE_FACTORS=longrope_best.txt \
        CP_CKPT=1 CP_CKPT_RESUME=$FT_RUN CP_MAX_STEPS=$FT_MAX \
        CP_REWARMUP=100 CP_REWARMUP_PEAK=0.3 CP_DATA_ROOT=$DATA_ROOT
  else  # cream
    log "fine-tune LongRoPE+CREAM at 4k physical run $FT_RUN -> step $FT_MAX"
    run_bin $ARCH CP_T=$PHYS_T CP_B=2 CP_GLOBAL_BATCH=$GBATCH \
        CP_CREAM_MODE=cream CP_CREAM_SIGMA=3.0 \
        YARN_SCALE=$S YARN_ORIG_MAXPOS=$ORIG_MAXPOS CP_LONGROPE_FACTORS=longrope_best.txt \
        CP_CKPT=1 CP_CKPT_RESUME=$FT_RUN CP_MAX_STEPS=$FT_MAX \
        CP_REWARMUP=100 CP_REWARMUP_PEAK=0.3 CP_DATA_ROOT=$DATA_ROOT
  fi
  log "fine-tune done (run $FT_RUN)"; wait_gpu_free
fi

# ---- 4. eval the LongRoPE arm at the target length --------------------------
OUT=ppl_longrope_x${S}_${COMPOSE}${MTAG}.csv
log "eval PPL-vs-position at T=$TARGET_T (run $EVAL_RUN) -> $OUT"
run_bin $ARCH CP_EVAL_PPL=1 CP_T=$TARGET_T CP_B=1 CP_EVAL_WINDOWS=$EVAL_WINDOWS \
    CP_LONGROPE_FACTORS=longrope_best.txt CP_CKPT_RESUME=$EVAL_RUN \
    CP_EVAL_OUT=$OUT CP_DATA_ROOT=$DATA_ROOT
log "DONE. wrote $OUT  (overlay with your CREAM/YaRN curves in graph_bluscript_cp.ipynb)"

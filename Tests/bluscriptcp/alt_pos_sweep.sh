#!/bin/bash
# =============================================================================
# alt_pos_sweep.sh -- PoSE / RandPos position-extension arms on the SAME rig as
# the CREAM sweep. Only difference vs CREAM is CP_CREAM_MODE. Reads BASE + DATA
# from the ENVIRONMENT so you never edit the file:
#
#   BASE=30 DATA=/path/to/shards ./alt_pos_sweep.sh pose    train
#   BASE=30 DATA=/path/to/shards ./alt_pos_sweep.sh pose    eval
#   BASE=30 DATA=/path/to/shards ./alt_pos_sweep.sh randpos train
#   BASE=30 DATA=/path/to/shards ./alt_pos_sweep.sh randpos eval
#
# Run from the repo root (where the Makefile + checkpoints_bluscriptcp/ live).
# Optional env overrides: GPU GB BSZ STEPS EW  (defaults match the CREAM arms).
# =============================================================================
set -euo pipefail
: "${BASE:?set BASE=<run topping at 917>}"
: "${DATA:?set DATA=<CP_DATA_ROOT>}"
GPU="${GPU:-3}"; GB="${GB:-131072}"; BSZ="${BSZ:-2}"; STEPS="${STEPS:-1222}"; EW="${EW:-32}"

MODE="${1:?mode: pose|randpos}"; PHASE="${2:?phase: train|eval}"
[[ "$MODE" == "pose" || "$MODE" == "randpos" ]] || { echo "mode must be pose|randpos"; exit 1; }
[[ "$PHASE" == "train" || "$PHASE" == "eval" ]] || { echo "phase must be train|eval"; exit 1; }

# fresh run numbers (avoid the existing 40/41/42)
declare -A RUN_pose=(    [2]=61 [4]=62 [8]=63 [16]=64 )
declare -A RUN_randpos=( [2]=71 [4]=72 [8]=73 [16]=74 )
declare -n RUN="RUN_${MODE}"

ARCH=( CUDA_VISIBLE_DEVICES=$GPU CP_SIZE=1
       CP_N_EMBD=384 CP_N_LAYER=6 CP_N_HEAD=6 CP_N_KVHEAD=2 CP_FFN=1024 CP_WEIGHT_TYING=0
       CP_DATA_ROOT="$DATA" )

for F in 2 4 8 16; do
  R="${RUN[$F]}"; T=$((4096 * F))
  if [[ "$PHASE" == "train" ]]; then
    echo "=== TRAIN $MODE x$F -> run $R (4k physical, scaled_max=$T) ==="
    ( cd checkpoints_bluscriptcp && for f in blumodelcp_run${BASE}_*; do
        cp "$f" "${f/run${BASE}/run${R}}"; done )
    env "${ARCH[@]}" CP_GLOBAL_BATCH=$GB CP_T=4096 CP_B=$BSZ \
        CP_CREAM_MODE=$MODE YARN_SCALE=$F YARN_ORIG_MAXPOS=4096 \
        CP_CKPT=1 CP_CKPT_RESUME=$R CP_MAX_STEPS=$STEPS \
        CP_REWARMUP=100 CP_REWARMUP_PEAK=0.3 \
        make CP_FUSED_ROPE=1 run-bluscript-cp NP=1
  else
    OUT="ppl_${MODE}_x${F}.csv"
    echo "=== EVAL $MODE x$F (run $R) at T=$T -> $OUT ==="
    env "${ARCH[@]}" CP_EVAL_PPL=1 CP_CREAM_MODE=off CP_T=$T CP_B=1 \
        CP_EVAL_WINDOWS=$EW YARN_SCALE=$F YARN_ORIG_MAXPOS=4096 \
        CP_CKPT_RESUME=$R CP_EVAL_OUT="$OUT" \
        make CP_FUSED_ROPE=1 run-bluscript-cp NP=1
  fi
done
echo "done: $MODE $PHASE"

#!/usr/bin/env bash
# =============================================================================
# longrope_progressive.sh -- progressive LongRoPE context extension, RESUMABLE.
#
# For each target length (STAGES, as extension factors from the 4k base) it runs:
#   1. SEARCH   : forward-only GA on the CURRENT checkpoint -> best cache (lambda,n_hat)
#   2. FINE-TUNE: adapt the weights at that length WITH the searched cache installed
#   3. EVAL     : PPL-vs-position at that length (held-out) -> ppl_prog_s<S>.csv
# The fine-tuned checkpoint of each stage becomes the base for the next stage
# (that's why 128k works: the search at stage N runs on an already-adapted model,
# not the 4k base, so it isn't in the worse-than-random zone).
#
# RESUME: safe to Ctrl-C / kill and re-run the SAME command -- it continues where
# it left off:
#   * completed sub-steps are marked in $STATE_DIR and skipped;
#   * a half-done FINE-TUNE resumes from its latest checkpoint (saved every
#     CP_CKPT_FREQ steps -> a kill loses at most that many steps);
#   * a half-done SEARCH resumes from its saved population + memo -- every evaluated
#     candidate is cached to disk, so restarting NEVER repeats a forward pass, and
#     the time budget counts cumulatively across restarts.
#
# GPUS: choose which GPUs (and how many) AT LAUNCH via the GPUS list. Its count sets
#   CP_SIZE automatically (CP_SIZE in ARCH is ignored/overwritten). Keep the topology
#   (CP_ATTN_MODE / CP_ULYSSES_SIZE) in ARCH. Don't change GPUS mid search/fine-tune.
#
# FASTER SEARCH (candidate parallelism): set SEARCH_CP=1 (or 2) so each candidate is
#   scored on that many GPUs and POOL=NP/SEARCH_CP candidates run AT ONCE -> ~POOL x
#   faster search, result-identical to serial. e.g. SEARCH_CP=1 on 8 GPUs = 8x. The
#   fine-tune still uses all NP GPUs. Default SEARCH_CP=NP POOL=1 (original behavior).
#
# Launch (example: 4k -> 64k -> 128k, 8 GPUs, hybrid ulysses=4 x ring=2):
#   (no base step count needed -- auto-detected from the checkpoint filename)
#   STAGES="16 32" BASE_RUN=1 FT_RUN_BASE=100 \
#   GPUS=0,1,2,3,4,5,6,7 \
#   ARCH="CP_ATTN_MODE=hybrid CP_ULYSSES_SIZE=4 CP_N_EMBD=768 CP_N_LAYER=12 CP_N_HEAD=12 CP_N_KVHEAD=4 CP_FFN=2048 CP_WEIGHT_TYING=1" \
#   DATA_ROOT=/mnt/.../Data_Loader/Data \
#     nohup bash Tests/bluscriptcp/longrope_progressive.sh > lr_prog.log 2>&1 &
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> repo root

# ---- config (override via env) ----------------------------------------------
CKPT_DIR=${CKPT_DIR:-checkpoints_bluscriptcp}
PREFIX=${PREFIX:-blumodelcp}
ARCH=${ARCH:-"CP_SIZE=8 CP_ATTN_MODE=hybrid CP_ULYSSES_SIZE=4 CP_N_EMBD=768 CP_N_LAYER=12 CP_N_HEAD=12 CP_N_KVHEAD=4 CP_FFN=2048 CP_WEIGHT_TYING=1"}
STAGES=${STAGES:-"16 32"}         # extension factors from the 4k base (16=64k, 32=128k, 64=256k)
BASE_RUN=${BASE_RUN:-1}           # starting checkpoint (the trained 4k base)
PHYS_T=${PHYS_T:-4096}            # base/pretrain context length
ORIG_MAXPOS=${ORIG_MAXPOS:-$PHYS_T}   # YaRN "original" length -- stays the true base (4k) throughout
HEAD_DIM=${HEAD_DIM:-64}
DATA_ROOT=${DATA_ROOT:-/mnt/volgrp03/3rd_floor/Adhitya/CP/Context_Parallelism/Data_Loader/Data}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
FT_RUN_BASE=${FT_RUN_BASE:-100}   # stage i uses run (FT_RUN_BASE+i). MUST be a free range (resume targets it).
FT_STEPS=${FT_STEPS:-300}         # fine-tune steps added per stage
FT_B=${FT_B:-1}
CP_REWARMUP=${CP_REWARMUP:-100}
CP_CKPT_FREQ=${CP_CKPT_FREQ:-25}  # fine-tune saves every N steps -> a kill loses <= N steps of training
MSCALE=${MSCALE:-0}               # 1 = bake YaRN temperature into the LongRoPE cache
EVAL_WINDOWS=${EVAL_WINDOWS:-32}
STATE_DIR=${STATE_DIR:-longrope_prog_state}

# search knobs (per stage; same for all). Watch the [precheck] line per stage.
POP=${POP:-48}; N1=${N1:-15}; N2=${N2:-15}; ITERS=${ITERS:-30}; CALIB=${CALIB:-4}
PATIENCE=${PATIENCE:-6}; TOL=${TOL:-0.01}
SEARCH_BUDGET_SEC=${SEARCH_BUDGET_SEC:-10800}

[ "$MSCALE" = "1" ] && ARCH="$ARCH CP_LONGROPE_MSCALE=1"
# GPUs are chosen AT LAUNCH via GPUS (which GPUs) -- their count sets NP and CP_SIZE.
# You keep the topology (CP_ATTN_MODE / CP_ULYSSES_SIZE) in ARCH; CP_SIZE is forced to
# match the GPU count so the two never drift. (Don't change GPUS mid search/fine-tune.)
NP=$(awk -F, '{print NF}' <<<"$GPUS")
ARCH="CP_SIZE=$NP $(echo "$ARCH" | sed -E 's/(^| )CP_SIZE=[0-9]+//g')"

# ---- candidate-parallel SEARCH ----------------------------------------------
# The SEARCH is forward-only and often fits on fewer GPUs than the fine-tune, so
# we score POOL candidates at once, each on SEARCH_CP GPUs (POOL = NP / SEARCH_CP).
# SEARCH_CP=1 POOL=NP gives ~NP x search speedup, RESULT-IDENTICAL to serial
# (proved by Tests/bluscriptcp/test_pool_equivalence.sh). Default SEARCH_CP=NP
# POOL=1 == the original single-candidate search. The FINE-TUNE always uses all NP
# GPUs (a long-context training step needs the full CP group). Bump SEARCH_CP if a
# single candidate doesn't fit on 1 GPU (e.g. 256k forward-only -> SEARCH_CP=2).
SEARCH_CP=${SEARCH_CP:-$NP}
if [ $(( NP % SEARCH_CP )) -ne 0 ]; then
  echo "ERROR: NP=$NP not divisible by SEARCH_CP=$SEARCH_CP"; exit 1
fi
if [ "$SEARCH_CP" = "$NP" ]; then
  # POOL=1: search each candidate at the FULL CP (== the original single-candidate
  # behavior); reuse the fine-tune topology (hybrid etc.) verbatim.
  POOL=1; SEARCH_ARCH="$ARCH"
else
  # candidate-parallel: SEARCH_CP must be 1 (single GPU) or 2 (verified 2-rank ring);
  # pure ring>=4 is unverified, so refuse it here.
  if [ "$SEARCH_CP" -ge 3 ]; then
    echo "ERROR: candidate-parallel search needs SEARCH_CP=1 or 2 (pure ring>=4 unverified). "\
         "Use SEARCH_CP=1/2, or SEARCH_CP=$NP for a full-CP single-candidate search."; exit 1
  fi
  POOL=$(( NP / SEARCH_CP ))
  # rebuild a clean per-candidate topology: strip CP_SIZE/ATTN/ULYSSES, set CP_SIZE,
  # and (for CP>=2) the verified 2-rank ring. CP_SIZE=1 needs no attn topology.
  SEARCH_ARCH="CP_SIZE=$SEARCH_CP $(echo "$ARCH" | sed -E 's/(^| )(CP_SIZE|CP_ATTN_MODE|CP_ULYSSES_SIZE)=[^ ]*//g')"
  [ "$SEARCH_CP" -ge 2 ] && SEARCH_ARCH="$SEARCH_ARCH CP_ATTN_MODE=ring"
fi
export LD_LIBRARY_PATH=BluTrain/Tensor-Implementations/lib:BluTrain/Profiler/lib:${LD_LIBRARY_PATH:-}
EXE=./build/bluscriptCP_exec
mkdir -p "$STATE_DIR"

log(){ echo "[$(date '+%H:%M:%S')] $*"; }
run_bin(){ env "$@" CUDA_VISIBLE_DEVICES=$GPUS mpirun -np "$NP" "$EXE"; }
ckpt_exists(){ ls "$CKPT_DIR/${PREFIX}_run$1_step_"*.ckpt >/dev/null 2>&1; }
latest_step(){ ls -1 "$CKPT_DIR/${PREFIX}_run$1_step_"*.ckpt 2>/dev/null \
                 | sed -E 's/.*_step_([0-9]+)\.ckpt/\1/' | sort -n | tail -1; }

# ---- sanity ----
ckpt_exists "$BASE_RUN" || { echo "ERROR: base run $BASE_RUN has no checkpoint in $CKPT_DIR"; exit 1; }
log "progressive LongRoPE: stages=[$STAGES] base=run$BASE_RUN NP=$NP GPUS=$GPUS state=$STATE_DIR"

prev_run="$BASE_RUN"
i=0
for S in $STAGES; do
  T=$(( PHYS_T * S ))
  ft_run=$(( FT_RUN_BASE + i ))
  best="longrope_best_prog_s${S}.txt"
  m_search="$STATE_DIR/s${S}.search.done"
  m_ft="$STATE_DIR/s${S}.ft.done"
  m_eval="$STATE_DIR/s${S}.eval.done"
  OUT="ppl_prog_s${S}.csv"
  log "===== STAGE $i : S=$S  T=$T  from run$prev_run  -> ft run$ft_run ====="

  # ---- 1. SEARCH (forward-only, on the CURRENT/adapted checkpoint) ----
  if [ -f "$m_search" ]; then
    log "  [skip] search already done -> $best"
  else
    log "  search: target=$T on run$prev_run (POP=$POP CALIB=$CALIB pool=$POOL x CP=$SEARCH_CP budget=${SEARCH_BUDGET_SEC}s)"
    python3 Tests/bluscriptcp/longrope_search.py \
        --exec "$EXE" --ckpt-run "$prev_run" \
        --target-t "$T" --s "$S" --head-dim "$HEAD_DIM" --orig-maxpos "$ORIG_MAXPOS" \
        --data-root "$DATA_ROOT" --cuda-devices "$GPUS" --arch "$SEARCH_ARCH" --pool "$POOL" \
        --pop "$POP" --n1 "$N1" --n2 "$N2" --iters "$ITERS" --calib-windows "$CALIB" \
        --patience "$PATIENCE" --tol "$TOL" \
        --time-budget-sec "$SEARCH_BUDGET_SEC" --out "$best" \
        --state "$STATE_DIR/s${S}.search"
    touch "$m_search"
    log "  search done -> $best"
  fi

  # ---- 2. FINE-TUNE (adapt weights at T with the searched cache; resumable) ----
  if [ -f "$m_ft" ]; then
    log "  [skip] fine-tune already done -> run$ft_run"
  else
    prev_step=$(latest_step "$prev_run")
    [ -n "$prev_step" ] || { echo "ERROR: no checkpoint step for run$prev_run"; exit 1; }
    ft_max=$(( prev_step + FT_STEPS ))
    gbatch=$T   # tokens/step = B*T*dp = 1*T*1 = T (dp=1); grad_accum=1
    # Branch prev -> ft ONLY on first entry. On resume, run$ft_run already has a
    # partial checkpoint -> DO NOT re-branch (that would wipe progress); training
    # auto-loads the latest run$ft_run checkpoint and continues to ft_max.
    if ckpt_exists "$ft_run"; then
      log "  fine-tune RESUME: run$ft_run exists (latest step $(latest_step "$ft_run")) -> continue to $ft_max"
    else
      log "  branching run$prev_run (step $prev_step) -> run$ft_run, then fine-tune to $ft_max"
      cp "$CKPT_DIR/${PREFIX}_run${prev_run}_step_${prev_step}.ckpt" \
         "$CKPT_DIR/${PREFIX}_run${ft_run}_step_${prev_step}.ckpt"
    fi
    run_bin $ARCH CP_T=$T CP_B=$FT_B CP_GLOBAL_BATCH=$gbatch \
        CP_LONGROPE_FACTORS=$best \
        CP_CKPT=1 CP_CKPT_RESUME=$ft_run CP_MAX_STEPS=$ft_max CP_CKPT_FREQ=$CP_CKPT_FREQ \
        CP_REWARMUP=$CP_REWARMUP CP_REWARMUP_PEAK=0.3 CP_DATA_ROOT=$DATA_ROOT
    touch "$m_ft"
    log "  fine-tune done -> run$ft_run (step ~$ft_max)"
  fi

  # ---- 3. EVAL (held-out PPL-vs-position at T) ----
  if [ -f "$m_eval" ]; then
    log "  [skip] eval already done -> $OUT"
  else
    log "  eval: T=$T on run$ft_run (skip=$CALIB windows) -> $OUT"
    run_bin $ARCH CP_EVAL_PPL=1 CP_T=$T CP_B=1 CP_EVAL_WINDOWS=$EVAL_WINDOWS \
        CP_EVAL_SKIP_WINDOWS=$CALIB CP_LONGROPE_FACTORS=$best CP_CKPT_RESUME=$ft_run \
        CP_EVAL_OUT=$OUT CP_DATA_ROOT=$DATA_ROOT
    touch "$m_eval"
    log "  eval done -> $OUT"
  fi

  prev_run="$ft_run"   # next stage extends from THIS fine-tuned checkpoint
  i=$(( i + 1 ))
done

log "ALL STAGES DONE. final ckpt=run$prev_run. curves: ppl_prog_s*.csv ; state: $STATE_DIR/"

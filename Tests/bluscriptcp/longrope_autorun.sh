#!/usr/bin/env bash
# =============================================================================
# longrope_autorun.sh -- wait for the GPUs to free up, then run the progressive
# LongRoPE chain unattended: stage-1 fine-tune + eval, then stage-2 (128k)
# search + fine-tune + eval.
#
# It does NOT kill anyone else's job -- it polls until every requested GPU is
# genuinely idle (>= MIN_FREE_MB free) for STABLE_CHECKS consecutive polls, then
# hands off to longrope_progressive.sh. Completed sub-steps are skipped via the
# markers in STATE_DIR, so an already-finished stage-1 search is not redone.
#
# Typical use (stage 1 already searched on 4 GPUs; finish it and do stage 2 on 8):
#   STAGES="16 32" GPUS=0,1,2,3,4,5,6,7 \
#   S2_TOPK=24 S2_PMUT=0.1 S2_BUDGET=21600 \
#     nohup bash Tests/bluscriptcp/longrope_autorun.sh > lr_auto.log 2>&1 &
#
# Watch it:   tail -f lr_auto.log
# Stop it:    pkill -f longrope_autorun ; pkill -f longrope_progressive
#             (everything is resumable -- just relaunch the same command)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> repo root

# ---- what to run ------------------------------------------------------------
STAGES=${STAGES:-"16 32"}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
BASE_RUN=${BASE_RUN:-1}
FT_RUN_BASE=${FT_RUN_BASE:-100}
ARCH=${ARCH:-"CP_ATTN_MODE=hybrid CP_ULYSSES_SIZE=4 CP_N_EMBD=768 CP_N_LAYER=12 CP_N_HEAD=12 CP_N_KVHEAD=4 CP_FFN=2048 CP_WEIGHT_TYING=1"}
DATA_ROOT=${DATA_ROOT:-/mnt/volgrp03/3rd_floor/Adhitya/CP/Context_Parallelism/Data_Loader/Data}
SEARCH_CP=${SEARCH_CP:-1}         # 1 GPU per candidate -> POOL = #GPUS (fastest search)
FT_STEPS=${FT_STEPS:-300}
CALIB=${CALIB:-4}
CP_CKPT_FREQ=${CP_CKPT_FREQ:-25}
STATE_DIR=${STATE_DIR:-longrope_prog_state}

# stage-2 search knobs. The 16x search showed the default operator (TOPK=32 of 48,
# PMUT=0.3) cannot beat its own NTK seed -- it resamples ~10 of 32 dims per child,
# jumping away from a smooth optimum. 24/0.1 restores the paper's 50% selection
# ratio and refines locally instead. These apply to whichever searches still run.
S2_TOPK=${S2_TOPK:-24}
S2_PMUT=${S2_PMUT:-0.1}
S2_BUDGET=${S2_BUDGET:-21600}     # seconds; CUMULATIVE across restarts

# ---- GPU-wait policy --------------------------------------------------------
MIN_FREE_MB=${MIN_FREE_MB:-40000} # per GPU; 40 GB => a 48 GB card is genuinely idle
POLL_SEC=${POLL_SEC:-60}
STABLE_CHECKS=${STABLE_CHECKS:-3} # consecutive good polls before we commit
MAX_WAIT_SEC=${MAX_WAIT_SEC:-0}   # 0 = wait forever
SKIP_WAIT=${SKIP_WAIT:-0}         # 1 = start immediately (GPUs known free)

log(){ echo "[$(date '+%F %H:%M:%S')] $*"; }

gpu_free_mb(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '; }

wait_for_gpus(){
  local ids="${GPUS//,/ }" t0 streak=0
  t0=$(date +%s)
  log "waiting for GPUs [$GPUS] -- need >= ${MIN_FREE_MB}MB free on each, ${STABLE_CHECKS}x in a row"
  while :; do
    local ok=1 report=""
    for g in $ids; do
      local f; f=$(gpu_free_mb "$g")
      report+=" g${g}=${f:-?}"
      [ "${f:-0}" -lt "$MIN_FREE_MB" ] && ok=0
    done
    if [ "$ok" = "1" ]; then
      streak=$(( streak + 1 ))
      log "GPUs look free ($streak/$STABLE_CHECKS):$report"
      [ "$streak" -ge "$STABLE_CHECKS" ] && { log "GPUs confirmed free -- starting"; return 0; }
    else
      [ "$streak" -gt 0 ] && log "GPU became busy again -- resetting streak"
      streak=0
      log "busy, waiting:$report"
    fi
    if [ "$MAX_WAIT_SEC" -gt 0 ] && [ $(( $(date +%s) - t0 )) -ge "$MAX_WAIT_SEC" ]; then
      log "ERROR: GPUs still busy after ${MAX_WAIT_SEC}s -- giving up"; return 1
    fi
    sleep "$POLL_SEC"
  done
}

# ---- pre-flight -------------------------------------------------------------
[ -x ./build/bluscriptCP_exec ] || { log "ERROR: ./build/bluscriptCP_exec missing -- build first"; exit 1; }
if pgrep -f bluscriptCP_exec >/dev/null 2>&1; then
  log "WARNING: bluscriptCP_exec already running. If those are stale workers from a"
  log "         stopped search they will contend for GPUs -- check: pgrep -fa bluscriptCP_exec"
fi
log "plan: stages=[$STAGES] GPUS=$GPUS SEARCH_CP=$SEARCH_CP (pool=$(( $(awk -F, '{print NF}' <<<"$GPUS") / SEARCH_CP )))"
log "      search knobs for any remaining search: TOPK=$S2_TOPK PMUT=$S2_PMUT budget=${S2_BUDGET}s (cumulative)"
log "      completed sub-steps in $STATE_DIR are skipped (nothing is recomputed)"

[ "$SKIP_WAIT" = "1" ] || wait_for_gpus

# ---- hand off to the progressive chain --------------------------------------
# One invocation runs every stage in STAGES end-to-end: for each, search (if not
# already done) -> fine-tune -> eval, with each stage starting from the previous
# stage's fine-tuned checkpoint.
log "launching longrope_progressive.sh"
STAGES="$STAGES" GPUS="$GPUS" BASE_RUN="$BASE_RUN" FT_RUN_BASE="$FT_RUN_BASE" \
ARCH="$ARCH" DATA_ROOT="$DATA_ROOT" SEARCH_CP="$SEARCH_CP" \
FT_STEPS="$FT_STEPS" CALIB="$CALIB" CP_CKPT_FREQ="$CP_CKPT_FREQ" STATE_DIR="$STATE_DIR" \
TOPK="$S2_TOPK" PMUT="$S2_PMUT" SEARCH_BUDGET_SEC="$S2_BUDGET" \
  bash Tests/bluscriptcp/longrope_progressive.sh

log "AUTORUN COMPLETE. curves: ppl_prog_s*.csv ; caches: longrope_best_prog_s*.txt"

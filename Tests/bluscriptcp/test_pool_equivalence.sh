#!/usr/bin/env bash
# =============================================================================
# test_pool_equivalence.sh -- PROVE the candidate-parallel search is identical to
# the serial search. Runs the SAME tiny search twice with the SAME --seed:
#   (1) --pool 1  (serial, 1 GPU)
#   (2) --pool N  (parallel, N GPUs)
# then requires BIT-IDENTICAL best file + per-generation best-PPL sequence.
# On a bit-identical failure it falls back to selection-equivalence (same winner
# within a tiny tolerance) and reports the max difference for diagnosis.
#
# Deterministic by construction: patience OFF, tol 0, NO time budget -> the search
# runs a fixed number of generations with a fixed candidate set, so pool size can
# only change *how fast*, never *what* is scored.
#
# Run:  bash Tests/bluscriptcp/test_pool_equivalence.sh
#   override: CKPT_RUN=30 T=8192 GPUS_SERIAL=0 GPUS_POOL=0,1 POOL=2 bash ...
# Must run where a checkpoint (CKPT_RUN) and the built exe exist (local 2-GPU box
# is enough for POOL=2 with CP_SIZE=1).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."
export LD_LIBRARY_PATH=BluTrain/Tensor-Implementations/lib:BluTrain/Profiler/lib:${LD_LIBRARY_PATH:-}

EXE=${EXE:-./build/bluscriptCP_exec}
CKPT_RUN=${CKPT_RUN:-30}
S=${S:-2}                                   # extension factor; T = 4096*S (S=2 -> 8k)
T=${T:-$((4096 * S))}
HEAD_DIM=${HEAD_DIM:-64}
DATA_ROOT=${DATA_ROOT:-/home/blu-bridge25/CP/Data_Loader/Data}
ARCH=${ARCH:-"CP_SIZE=1 CP_N_EMBD=384 CP_N_LAYER=6 CP_N_HEAD=6 CP_N_KVHEAD=2 CP_FFN=1024 CP_WEIGHT_TYING=0"}
POOL=${POOL:-2}
GPUS_SERIAL=${GPUS_SERIAL:-0}               # 1 GPU for pool=1
GPUS_POOL=${GPUS_POOL:-0,1}                 # POOL*CP_SIZE GPUs for pool=N
SEED=${SEED:-0}
POP=${POP:-8}; N1=${N1:-3}; N2=${N2:-3}; ITERS=${ITERS:-3}; CALIB=${CALIB:-2}
OUTD=${OUTD:-/tmp/pool_equiv}; mkdir -p "$OUTD"; rm -f "$OUTD"/best_*.txt "$OUTD"/log_*.txt "$OUTD"/ppl_*.txt

run(){  # $1=pool  $2=devices  $3=tag
  python3 Tests/bluscriptcp/longrope_search.py \
    --exec "$EXE" --ckpt-run "$CKPT_RUN" --target-t "$T" --s "$S" \
    --head-dim "$HEAD_DIM" --orig-maxpos 4096 --data-root "$DATA_ROOT" \
    --arch "$ARCH" --cuda-devices "$2" --pool "$1" \
    --pop "$POP" --n1 "$N1" --n2 "$N2" --iters "$ITERS" --calib-windows "$CALIB" \
    --patience 0 --tol 0.0 --time-budget-sec 0 --seed "$SEED" \
    --out "$OUTD/best_$3.txt" > "$OUTD/log_$3.txt" 2>&1
}

echo "== serial   (pool=1,      GPU  $GPUS_SERIAL) =="; run 1      "$GPUS_SERIAL" serial
echo "== parallel (pool=$POOL, GPUs $GPUS_POOL) =="; run "$POOL" "$GPUS_POOL"   pool

bf=0; pf=0
# 1. bit-identical best file (the searched lambda + n_hat)
if diff -q "$OUTD/best_serial.txt" "$OUTD/best_pool.txt" >/dev/null 2>&1; then
  echo "  best file        : IDENTICAL"; bf=1
else
  echo "  best file        : DIFFERS"; diff "$OUTD/best_serial.txt" "$OUTD/best_pool.txt" || true
fi
# 2. bit-identical per-generation best-PPL sequence
grep -oE "best PPL [0-9.]+" "$OUTD/log_serial.txt" > "$OUTD/ppl_serial.txt" 2>/dev/null || true
grep -oE "best PPL [0-9.]+" "$OUTD/log_pool.txt"   > "$OUTD/ppl_pool.txt"   2>/dev/null || true
if diff -q "$OUTD/ppl_serial.txt" "$OUTD/ppl_pool.txt" >/dev/null 2>&1; then
  echo "  per-gen best PPL : IDENTICAL"; pf=1
else
  echo "  per-gen best PPL : DIFFERS"; paste "$OUTD/ppl_serial.txt" "$OUTD/ppl_pool.txt" | head
fi

if [ "$bf" = 1 ] && [ "$pf" = 1 ]; then
  echo "EQUIVALENCE GATE: PASS (bit-identical)"; exit 0
fi

echo "EQUIVALENCE GATE: bit-identical FAILED -> selection-equivalence fallback"
sp=$(grep -oE "BEST PPL [0-9.]+" "$OUTD/log_serial.txt" | tail -1 | awk '{print $3}')
pp=$(grep -oE "BEST PPL [0-9.]+" "$OUTD/log_pool.txt"   | tail -1 | awk '{print $3}')
echo "  serial BEST=$sp   pool BEST=$pp"
awk -v a="$sp" -v b="$pp" 'BEGIN{
  if(a==""||b==""){print "  FAIL: could not parse BEST PPL (see logs in '"$OUTD"')"; exit 1}
  d=a-b; if(d<0)d=-d;
  if(d<1e-3){print "  selection-equivalent (|dBEST| =",d,"< 1e-3): PASS (fp noise only)"; exit 0}
  else      {print "  FAIL: BEST PPL differs by",d,"-- investigate cross-GPU nondeterminism"; exit 1}}'

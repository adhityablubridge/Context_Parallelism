#!/usr/bin/env bash
# =============================================================================
# cp_parity.sh -- CP correctness gate for the LongRoPE/YaRN PPL eval + search.
# Proves the per-position map + cross-rank reduction are correct by requiring
# CP=1 == CP=2 == CP=4 (within fp tolerance) on the SAME base ckpt:
#   1. CSV parity, HeadTail (ring): CP=1 vs 2 vs 4     [the live map for CP=4]
#   2. CSV parity, contiguous (ulysses): CP=1 vs 2     [exercises the lb=false branch]
#   3. scalar search-evaluator parity: identity candidate, CP=1 vs 4
#   4. swap-takes-effect: identity vs aggressive candidate under CP => PPL differs
#
# Run (from repo root):  bash Tests/bluscriptcp/cp_parity.sh
#   override: DATA_ROOT=... CKPT_RUN=30 T=8192 WIN=4 GPUS4=0,1,2,3 bash ...
# T must be divisible by 2*4=8 (HeadTail at CP=4).  Absolute PPL is irrelevant
# here (base trained at 4k, evaluated at T with plain RoPE) -- only CP-invariance.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."
export LD_LIBRARY_PATH=BluTrain/Tensor-Implementations/lib:BluTrain/Profiler/lib:${LD_LIBRARY_PATH:-}

EXE=./build/bluscriptCP_exec
DATA_ROOT=${DATA_ROOT:-/mnt/volgrp03/3rd_floor/Adhitya/CP/Context_Parallelism/Data_Loader/Data}
CKPT_RUN=${CKPT_RUN:-30}
T=${T:-8192}
WIN=${WIN:-4}
TOL=${TOL:-1e-3}                         # relative tolerance on the ppl column
GPUS4=${GPUS4:-0,1,2,3}
ARCH="CP_N_EMBD=384 CP_N_LAYER=6 CP_N_HEAD=6 CP_N_KVHEAD=2 CP_FFN=1024 CP_WEIGHT_TYING=0"
OUTD=${OUTD:-/tmp/cp_parity}; mkdir -p "$OUTD"
g2=$(echo "$GPUS4" | cut -d, -f1-2); g1=$(echo "$GPUS4" | cut -d, -f1)
fail=0

run_csv(){  # $1=cpsize $2=devs $3=attn $4=outfile
  env CP_SIZE=$1 CP_ATTN_MODE=$3 $ARCH CP_EVAL_PPL=1 CP_T=$T CP_B=1 CP_EVAL_WINDOWS=$WIN \
      CP_CKPT_RESUME=$CKPT_RUN CP_EVAL_OUT=$4 CP_DATA_ROOT=$DATA_ROOT \
      CUDA_VISIBLE_DEVICES=$2 mpirun -np $1 "$EXE" > "$4.log" 2>&1
}
cmp_csv(){  # $1=ref $2=test $3=label ; compares the ppl column (field 4) with rel tol
  awk -F, -v tol="$TOL" -v lab="$3" '
    NR==FNR { if (FNR>1) a[$1]=$4+0; next }
    FNR>1 { d=a[$1]-($4+0); if(d<0)d=-d; b=(a[$1]<0?-a[$1]:a[$1]); if(b<1e-9)b=1e-9;
            if (d/b>tol){ if(++bad<=3) print "    MISMATCH pos="$1" ref="a[$1]" test="$4 } }
    END { if(bad){ print "  "lab": FAIL ("bad" rows > tol)"; exit 1 } else print "  "lab": PASS" }
  ' "$1" "$2"
}

echo "== 1. HeadTail (ring) CSV parity: CP=1 vs 2 vs 4 at T=$T =="
run_csv 1 "$g1"    ring "$OUTD/cp1.csv"
run_csv 2 "$g2"    ring "$OUTD/cp2.csv"
run_csv 4 "$GPUS4" ring "$OUTD/cp4.csv"
cmp_csv "$OUTD/cp1.csv" "$OUTD/cp2.csv" "CP1-vs-CP2 (ring/HeadTail)" || fail=1
cmp_csv "$OUTD/cp1.csv" "$OUTD/cp4.csv" "CP1-vs-CP4 (ring/HeadTail)" || fail=1

echo "== 2. Contiguous (ulysses) CSV parity: CP=1 vs 2 =="
run_csv 2 "$g2" ulysses "$OUTD/cp2_ul.csv"
cmp_csv "$OUTD/cp1.csv" "$OUTD/cp2_ul.csv" "CP1-vs-CP2 (ulysses/contiguous)" || fail=1

echo "== 3+4. Scalar search-evaluator parity + swap-takes-effect =="
# identity candidate (lambda=1 x32, n_hat=0) and an aggressive one (ramp to ~T/4k).
half=32
idf="$OUTD/ident.txt"; agf="$OUTD/aggr.txt"
{ echo "n_hat 0"; echo "s 1"; echo "S_search $T"; printf "lambda"; for i in $(seq $half); do printf " 1.00"; done; echo; } > "$idf"
{ echo "n_hat 0"; echo "s $((T/4096))"; echo "S_search $T"; printf "lambda"; for i in $(seq $half); do printf " %.2f" "$((T/4096))"; done; echo; } > "$agf"
scalar(){  # $1=cpsize $2=devs ; feeds identity THEN aggressive, prints two PPLs
  printf '%s\n%s\n' "$idf" "$agf" | \
    env CP_SIZE=$1 CP_ATTN_MODE=ring $ARCH CP_LONGROPE_SEARCH=1 CP_T=$T CP_B=1 CP_EVAL_WINDOWS=$WIN \
        CP_CKPT_RESUME=$CKPT_RUN CP_DATA_ROOT=$DATA_ROOT \
        CUDA_VISIBLE_DEVICES=$2 mpirun -np $1 "$EXE" 2>/dev/null | awk '/^PPL /{print $2}'
}
read -r p1_id p1_ag < <(scalar 1 "$g1"  | paste -sd' ')
read -r p4_id p4_ag < <(scalar 4 "$GPUS4" | paste -sd' ')
echo "  identity  PPL: CP=1=$p1_id  CP=4=$p4_id"
echo "  aggressive PPL: CP=1=$p1_ag  CP=4=$p4_ag"
awk -v a="$p1_id" -v b="$p4_id" -v tol="$TOL" 'BEGIN{d=a-b;if(d<0)d=-d;base=(a<0?-a:a);if(base<1e-9)base=1e-9;
  if(d/base>tol){print "  scalar parity (identity, CP1 vs CP4): FAIL";exit 1}else print "  scalar parity (identity, CP1 vs CP4): PASS"}' || fail=1
awk -v a="$p4_id" -v b="$p4_ag" 'BEGIN{d=a-b;if(d<0)d=-d;
  if(d<1e-4){print "  swap-takes-effect (CP=4): FAIL (identity==aggressive => cache not swapping)";exit 1}
  else print "  swap-takes-effect (CP=4): PASS (identity != aggressive)"}' || fail=1

echo
[ "$fail" = "0" ] && echo "ALL CP PARITY CHECKS PASSED" || { echo "SOME CP PARITY CHECKS FAILED (see above / $OUTD/*.log)"; exit 1; }

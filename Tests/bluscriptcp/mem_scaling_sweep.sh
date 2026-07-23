#!/usr/bin/env bash
# =============================================================================
# mem_scaling_sweep.sh  (bluscriptCP  vs  DeepSpeed-Ulysses Qwen3)
# -----------------------------------------------------------------------------
# Memory-occupancy scaling sweep pairing OUR production context-parallel model
# (Scripts/Blutrain/bluscriptCP.cpp -- Llama-style GQA + RoPE + QK-norm + SwiGLU)
# against the standalone DeepSpeed-Ulysses reference that mirrors it
# (LlamaFactory/ds_ulysses/qwen3_ds_ulysses.py -- same d384/L6/qh6/kvh2 arch,
# official deepspeed.sequence.layer.DistributedAttention all-to-all).
#
# For each sequence length T (auto-doubling until OOM) it runs a short 2-step
# probe per implementation and writes a labeled snapshot that mem_scaling_table.py
# parses into a side-by-side table + a max-T-before-OOM limits summary.
#
#   CPP (bluscriptCP): built ONCE; driven by CP_* env. CP_MEM_PROBE=1 makes the
#     binary run 2 steps (grad_accum forced to 1), skip validation/checkpoints,
#     and write CPP_<label>_ulysses_T<T>_ws<ws>.txt itself (cudaMemGetInfo +
#     caching-allocator peak + nvidia-smi).
#   DS  (Qwen3 DS-Ulysses): a real 2-step `deepspeed` run per T with grad_accum=1
#     (global_batch_tokens = B*T). GPU memory comes from TWO sources, both written
#     into the snapshot: torch.cuda.max_memory_reserved (the script's mem_gpu_mb
#     CSV column -> torch.peak_reserved_mb) and a live nvidia-smi peak sampled
#     while it trains (-> SMI_USED_MB_PER_GPU). No edits to the python script.
#
# The two are architecturally matched (both all-to-all Ulysses attention).
#
# PORTABILITY (2x RTX 3060  ->  Nx RTX 6000 Ada): set CUDA_VISIBLE_DEVICES for the
# target GPUs. world_size = number of devices listed. Both sides run PURE CP
# (dp_size=1): CP_SIZE=world_size (C++), sequence_parallel_size=world_size (DS).
#   NB: Ulysses splits attention heads across ranks, so world_size must divide
#   BOTH q_heads(6) AND kv_heads(2). kv_heads=2 => world_size in {1,2} for this
#   48M config. Larger GPU counts need a config with more kv_heads (edit CONFIGS
#   for C++, and pass matching --kv_heads via DS_EXTRA_ARGS for the DS side).
#
# Usage:
#   ./mem_scaling_sweep.sh
#   python3 mem_scaling_table.py mem_scaling_runs_bluscriptcp
# =============================================================================
set -u  # (no -e: we WANT to continue after an OOM run fails)

# ============================ EDIT THIS BLOCK ================================

# Which GPUs to use. world_size = number of devices listed here.
export CUDA_VISIBLE_DEVICES="0,1"

# Implementations to run (1=yes, 0=no). They never run concurrently. Env-overridable.
RUN_CPP="${RUN_CPP:-1}"
RUN_DS="${RUN_DS:-1}"
# LlamaFactory v1 trainer (FSDP2 + Ulysses) as a THIRD systems arm (the actual LF
# framework, not the standalone DeepSpeed script). Off by default so the existing
# CPP-vs-DS sweep is unchanged; set RUN_LF=1 to add the LF-Qwen3 rows.
RUN_LF="${RUN_LF:-0}"

# Build the C++ binary once before sweeping (1=yes). Set 0 if already built.
BUILD_CPP="${BUILD_CPP:-1}"

# Repo paths. This script sits in CP/Tests/bluscriptcp/, so REPO_ROOT is two up.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CPP_EXEC="${REPO_ROOT}/build/bluscriptCP_exec"
TENSOR_LIBDIR="${REPO_ROOT}/BluTrain/Tensor-Implementations/lib"
PROFILER_LIBDIR="${REPO_ROOT}/BluTrain/Profiler/lib"
DATA_ROOT="${CP_DATA_ROOT:-${REPO_ROOT}/Data_Loader/Data}"

# DeepSpeed-Ulysses reference (separate repo). Override LF_ROOT if it moved.
LF_ROOT="${LF_ROOT:-/home/blu-bridge25/LlamaFactory}"
DS_DIR="${LF_ROOT}/ds_ulysses"
# Python interpreter that has DeepSpeed. Default is the LF venv, but that venv's
# `python` can be a BROKEN symlink on a different machine (uv-managed venvs point
# at a per-host interpreter path). Override DS_PYTHON with any interpreter that
# can `import deepspeed`, e.g.  DS_PYTHON=/usr/bin/python3 ./mem_scaling_sweep.sh
DS_PYTHON="${DS_PYTHON:-${LF_ROOT}/.venv/bin/python}"

# WHICH DeepSpeed reference(s) to sweep alongside bluscriptCP:
#   qwen3  -> qwen3_ds_ulysses.py  (d384/L6/qh6/kvh2 -- the bluscriptCP match)
#   gpt2   -> gpt2_ds_ulysses.py   (n_embd384/L3/H6  -- the gpt2_cp_test match)
#   both   -> run BOTH as separate impl rows (DS-Qwen3 + DS-GPT2) in one table
DS_MODEL="${DS_MODEL:-qwen3}"

# Extra args passed verbatim to the DS script (e.g. "--kv_heads 4" for wider CP).
DS_EXTRA_ARGS="${DS_EXTRA_ARGS:-}"

# ---- LlamaFactory v1 arm (RUN_LF=1) ----
# The LF v1 trainer reads a fixed model dir (arch is baked in config.json), so the
# LF arm only sweeps the config row whose arch matches LF_MODEL_DIR. Default is the
# 48M dir; override both to benchmark the 114M config.
LF_MODEL_DIR="${LF_MODEL_DIR:-${LF_ROOT}/qwen3_48m}"   # config-only dir (from-scratch init_on_meta)
LF_LABEL="${LF_LABEL:-48M}"                            # CONFIGS row this arm runs (arch match)
# Python/venv that has llamafactory + flash-attn (Ulysses patches flash-attn fwd).
LF_PYTHON="${LF_PYTHON:-${LF_ROOT}/.venv/bin/python}"
LF_CLI="${LF_CLI:-${LF_ROOT}/.venv/bin/llamafactory-cli}"
# Helper that writes a throwaway fixed-length input_ids parquet at block=T (probe
# data; systems metrics do not depend on token values).
LF_PROBE_MK="${LF_ROOT}/scripts/make_probe_parquet.py"

# Output: per-run snapshots + the results CSV.
OUT_DIR="${SCRIPT_DIR}/mem_scaling_runs_bluscriptcp"

# T sweep: start, doubling, with a hard cap so it cannot loop forever.
T_START=2048
T_MAX=262144

# Fine-grained boundary search after the coarse doubling finds [last_OK, OOM].
# Default OFF: the DS probe runs are real (slow) training launches. Enable to pin
# the exact max-T cutoff. Env-overridable, e.g. FINE_GRAINED=1 ./mem_scaling_sweep.sh
FINE_GRAINED="${FINE_GRAINED:-0}"
FINE_STEP="${FINE_STEP:-256}"

# Per-run wall-clock timeout (seconds). Kills a hung run and treats it as fail.
RUN_TIMEOUT=1200

# Probe step count (both impls). >=2 so optimizer state is allocated/counted.
PROBE_STEPS=2

# Micro-batch per rank (B). Kept equal across both impls (bluscriptCP default B=2).
B=2

# Model config matrix:  "LABEL  D_MODEL  N_LAYER  Q_HEADS  KV_HEADS  FFN  TYING"
# Both impls are driven from these rows (the DS side is passed matching model args,
# with head_dim forced to 64 to match bluscriptCP -- see run_one_ds). head_dim =
# D_MODEL/Q_HEADS must be 64 or 128 (fused kernel). FFN = SwiGLU inner (8/3*D_MODEL).
# TYING = 1 shares lm_head with the token embedding (drops vocab*d_model params).
# These two rows are the real bluscriptCP configs (48M untied, 114M tied).
CONFIGS=(
  "48M    384   6    6    2    1024   0"
  "114M   768   12   12   4    2048   1"
)

# ============================ END EDIT BLOCK ================================

WORLD_SIZE="$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')"

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
      echo "[GUARD] FORCE=1 -> proceeding into non-empty $OUT_DIR ($stale_count stale snapshots; results may be mixed)"
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
echo " Memory scaling sweep: bluscriptCP vs DeepSpeed-Ulysses Qwen3"
echo "   CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  (world_size=$WORLD_SIZE)"
echo "   RUN_CPP=$RUN_CPP  RUN_DS=$RUN_DS (DS_MODEL=$DS_MODEL)  BUILD_CPP=$BUILD_CPP"
echo "   T: $T_START -> doubling -> <=$T_MAX   PROBE_STEPS=$PROBE_STEPS  B=$B"
echo "   DATA_ROOT=$DATA_ROOT"
echo "   OUT_DIR=$OUT_DIR"
echo "=============================================================="

# ---- Build C++ once (config/T are runtime env, so one build serves all) ----
# Ulysses attention is the DEFAULT build (the CP_FUSED_ROPE flag is only for the
# ring fused-RoPE path). make bluscript-cp -> build/bluscriptCP_exec.
if [[ "$RUN_CPP" == "1" && "$BUILD_CPP" == "1" ]]; then
  echo "[BUILD] make bluscript-cp (once) -> $CPP_EXEC ..."
  ( cd "$REPO_ROOT" && make bluscript-cp ) 2>&1 | tee "${LOG_DIR}/build_cpp.log" | tail -8
  if [[ ! -x "$CPP_EXEC" ]]; then
    echo "[BUILD] ERROR: $CPP_EXEC not found after build. Disabling C++ runs."
    RUN_CPP=0
  fi
fi

# Resolve DS_MODEL -> the list of DS impl codes to sweep (DSQ=qwen3, DSG=gpt2).
DS_IMPLS=()
case "$DS_MODEL" in
  qwen3) DS_IMPLS=(DSQ) ;;
  gpt2)  DS_IMPLS=(DSG) ;;
  both)  DS_IMPLS=(DSQ DSG) ;;
  *) echo "[DS] ERROR: DS_MODEL='$DS_MODEL' invalid (qwen3|gpt2|both). Disabling DS."; RUN_DS=0 ;;
esac

# Map an impl code to its python script.
ds_script_for() { case "$1" in DSQ) echo "${DS_DIR}/qwen3_ds_ulysses.py";;
                               DSG) echo "${DS_DIR}/gpt2_ds_ulysses.py";; esac; }

if [[ "$RUN_DS" == "1" ]]; then
  # One test covers everything: can this interpreter import deepspeed? (A broken
  # symlink or a missing module both fail here, with a clear message.)
  if ! "$DS_PYTHON" -c "import deepspeed" 2>/dev/null; then
    echo "[DS] ERROR: '$DS_PYTHON' cannot 'import deepspeed'"
    echo "     (broken/missing interpreter or deepspeed not installed there)."
    echo "     Set DS_PYTHON=/path/to/python-with-deepspeed and re-run. Disabling DS."; RUN_DS=0
  elif [[ ! -d "$DATA_ROOT" ]]; then
    echo "[DS] ERROR: DATA_ROOT $DATA_ROOT not found. Disabling DS runs."; RUN_DS=0
  else
    for _m in "${DS_IMPLS[@]}"; do
      _s="$(ds_script_for "$_m")"
      [[ -f "$_s" ]] || { echo "[DS] ERROR: $_s not found. Disabling DS runs."; RUN_DS=0; }
    done
  fi
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

# Sample nvidia-smi peak per (visible) GPU while PID is alive; write the peak as
# "idx,used;idx,used;..." to $2 on exit.
sample_smi_peak() {  # pid outfile
  local pid="$1" outfile="$2"
  local -A peak=()
  local vis="${CUDA_VISIBLE_DEVICES:-}"
  while kill -0 "$pid" 2>/dev/null; do
    while IFS=',' read -r idx used; do
      idx="${idx// /}"; used="${used// /}"
      [[ -z "$idx" || -z "$used" ]] && continue
      if [[ -n "$vis" ]]; then
        case ",$vis," in *,"$idx",*) : ;; *) continue ;; esac
      fi
      if [[ -z "${peak[$idx]:-}" || "$used" -gt "${peak[$idx]}" ]]; then peak[$idx]="$used"; fi
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null)
    sleep 1
  done
  local out="" k
  for k in $(echo "${!peak[@]}" | tr ' ' '\n' | sort -n); do
    [[ -n "$out" ]] && out="${out};"
    out="${out}${k},${peak[$k]}"
  done
  echo "$out" > "$outfile"
}

run_one_cpp() {  # label d l qh kvh ffn ty T
  local label="$1" d="$2" l="$3" qh="$4" kvh="$5" ffn="$6" ty="$7" T="$8"
  local tag="CPP_${label}_ulysses_T${T}_ws${WORLD_SIZE}"
  local snap="${OUT_DIR}/${tag}.txt" log="${LOG_DIR}/${tag}.log"
  rm -f "$snap"
  echo "  -> [CPP] $label  T=$T  (tag=$tag)"
  CP_MEM_PROBE=1 CP_MEM_PROBE_STEPS="$PROBE_STEPS" CP_MODEL_LABEL="$label" \
  CP_ATTN_MODE=ulysses CP_SIZE="$WORLD_SIZE" CP_B="$B" \
  CP_N_EMBD="$d" CP_N_LAYER="$l" CP_N_HEAD="$qh" CP_N_KVHEAD="$kvh" \
  CP_FFN="$ffn" CP_WEIGHT_TYING="$ty" CP_T="$T" \
  CP_CKPT=0 CP_DATA_ROOT="$DATA_ROOT" MEM_SNAPSHOT_DIR="$OUT_DIR" \
  LD_LIBRARY_PATH="${TENSOR_LIBDIR}:${PROFILER_LIBDIR}:${LD_LIBRARY_PATH:-}" \
  timeout "$RUN_TIMEOUT" mpirun -x LD_LIBRARY_PATH -np "$WORLD_SIZE" "$CPP_EXEC" \
    > "$log" 2>&1
  local rc=$?
  if is_fail "$log" "$rc" "$snap"; then
    echo "     FAIL/OOM (rc=$rc) -- see $log"
    echo "CPP,${label},ulysses,${d},${l},${qh},${ty},${T},${WORLD_SIZE},OOM,${snap}" >> "$RESULTS_CSV"
    return 1
  fi
  echo "     OK -- snapshot $snap"
  echo "CPP,${label},ulysses,${d},${l},${qh},${ty},${T},${WORLD_SIZE},OK,${snap}" >> "$RESULTS_CSV"
  return 0
}

run_one_ds() {  # impl_code(DSQ|DSG) label d l qh kvh ffn ty T
  local ic="$1" label="$2" d="$3" l="$4" qh="$5" kvh="$6" ffn="$7" ty="$8" T="$9"
  local script mtag hd=$(( d / qh ))       # head_dim = d_model/q_heads (=64), matches bluscriptCP
  local -a model_args
  if [[ "$ic" == "DSQ" ]]; then
    script="${DS_DIR}/qwen3_ds_ulysses.py"; mtag="qwen3"
    # GQA + decoupled head_dim: pass the full spec so DS matches bluscriptCP exactly.
    model_args=(--n_embd "$d" --n_layer "$l" --q_heads "$qh" --kv_heads "$kvh"
                --head_dim "$hd" --intermediate "$ffn")
  else
    script="${DS_DIR}/gpt2_ds_ulysses.py";  mtag="gpt2"
    # GPT-2 is MHA (no kv_heads/ffn/head_dim args); head_dim = n_embd/n_head.
    model_args=(--n_embd "$d" --n_layer "$l" --n_head "$qh")
  fi
  [[ "$ty" == "1" ]] && model_args+=(--tie_weights)
  local tag="${ic}_${label}_ulysses_T${T}_ws${WORLD_SIZE}"
  local snap="${OUT_DIR}/${tag}.txt" log="${LOG_DIR}/${tag}.log"
  local csv="${LOG_DIR}/${tag}.csv" smi="${LOG_DIR}/${tag}.smi"
  rm -f "$snap" "$csv"
  echo "  -> [${ic}] $mtag/$label (d=$d L=$l qh=$qh kvh=$kvh hd=$hd ffn=$ffn tie=$ty)  T=$T"

  # grad_accum=1: global_batch_tokens = B*T. max_steps=PROBE_STEPS, warmup=1.
  local gbt=$(( B * T ))
  (
    cd "$LF_ROOT" || exit 97
    export PATH="$(dirname "$DS_PYTHON"):$PATH"
    # Launch DeepSpeed via the chosen PYTHON + the launcher MODULE, not the bare
    # `deepspeed` shim: on some boxes `deepspeed` resolves to a stale copy in
    # ~/.local/bin whose shebang points at the wrong interpreter (missing the
    # deepspeed module). `python -m deepspeed.launcher.runner` uses THIS
    # interpreter's real DeepSpeed regardless of PATH/shim state.
    LF_CSV_LOG="$csv" \
    timeout "$RUN_TIMEOUT" "$DS_PYTHON" -m deepspeed.launcher.runner \
      --num_gpus "$WORLD_SIZE" "$script" \
      --data_root "$DATA_ROOT" --block_size "$T" --micro_batch_size "$B" \
      --global_batch_tokens "$gbt" --max_steps "$PROBE_STEPS" --warmup_steps 1 \
      --logging_steps 1 "${model_args[@]}" $DS_EXTRA_ARGS
  ) > "$log" 2>&1 &
  local ds_pid=$!
  sample_smi_peak "$ds_pid" "$smi" &
  local smi_pid=$!
  wait "$ds_pid"; local rc=$?
  wait "$smi_pid" 2>/dev/null

  local smi_used="" ; [[ -s "$smi" ]] && smi_used="$(cat "$smi")"
  # torch peak reserved = max mem_gpu_mb (last CSV column) over the probe rows.
  local torch_peak=""
  if [[ -s "$csv" ]]; then
    torch_peak="$(awk -F',' 'NR>1 && $NF ~ /^[0-9.]+$/ {if($NF+0>m)m=$NF+0} END{if(m>0)printf "%.1f", m}' "$csv")"
  fi

  local failed=0
  if [[ "$rc" != "0" ]] || [[ -z "$smi_used" && -z "$torch_peak" ]] || \
     grep -qiE "out of memory|CUDA error|RuntimeError|torch.OutOfMemory" "$log"; then
    failed=1
  fi
  if [[ "$failed" == "1" ]]; then
    echo "     FAIL/OOM (rc=$rc) -- see $log"
    echo "${ic},${label},ulysses,${d},${l},${qh},${ty},${T},${WORLD_SIZE},OOM,${snap}" >> "$RESULTS_CSV"
    return 1
  fi
  {
    echo "# MEM PROBE SNAPSHOT  tag=${tag}"
    echo "# impl=${ic} label=${label} rotator=ulysses n_embd=${d} n_layer=${l} n_head=${qh} weight_tying=${ty}"
    echo "# B=${B} T=${T} cp_world_size=${WORLD_SIZE} params="
    [[ -n "$torch_peak" ]] && echo "# torch.peak_alloc_mb(rank0)= torch.peak_reserved_mb(rank0)=${torch_peak}"
    echo "# SMI_USED_MB_PER_GPU=${smi_used}"
    echo "# ---- deepspeed-ulysses ${mtag} (mem_gpu_mb=torch.max_memory_reserved; smi peak sampled) ----"
  } > "$snap"
  echo "     OK -- snapshot $snap (torch_peak=${torch_peak:-NA}MB smi peak: ${smi_used:-NA})"
  echo "${ic},${label},ulysses,${d},${l},${qh},${ty},${T},${WORLD_SIZE},OK,${snap}" >> "$RESULTS_CSV"
  return 0
}

run_one_lf() {  # label d l qh kvh ffn ty T   (LlamaFactory v1 FSDP2+Ulysses, arch = LF_MODEL_DIR)
  local label="$1" d="$2" l="$3" qh="$4" kvh="$5" ffn="$6" ty="$7" T="$8"
  local tag="LFQ_${label}_ulysses_T${T}_ws${WORLD_SIZE}"
  local snap="${OUT_DIR}/${tag}.txt" log="${LOG_DIR}/${tag}.log"
  local csv="${LOG_DIR}/${tag}.csv" smi="${LOG_DIR}/${tag}.smi"
  local ycfg="${LOG_DIR}/${tag}.yaml" dcfg="${LOG_DIR}/${tag}.dataset.yaml"
  local pq="${LOG_DIR}/${tag}.parquet"
  rm -f "$snap" "$csv"
  echo "  -> [LFQ] LlamaFactory-v1 $label  T=$T  (cp_size=$WORLD_SIZE)"

  # Throwaway probe data at block=T (systems metrics are value-independent).
  if ! "$LF_PYTHON" "$LF_PROBE_MK" "$pq" --block "$T" \
        --rows $(( B * PROBE_STEPS + 2 )) > "$log" 2>&1; then
    echo "     FAIL: could not build probe parquet -- see $log"
    echo "LFQ,${label},ulysses,${d},${l},${qh},${ty},${T},${WORLD_SIZE},OOM,${snap}" >> "$RESULTS_CSV"; return 1
  fi
  cat > "$dcfg" <<EOF
probe_packed_${T}:
  path: ${pq}
  source: local
  converter: pretrain_packed
EOF
  # grad_accum=1 at dp=1: global_batch_size = micro (=B); tokens/step = B*T (== DS arm).
  cat > "$ycfg" <<EOF
model: ${LF_MODEL_DIR}
model_class: llm
trust_remote_code: true
flash_attn: flash_attention_2
init_config:
  name: init_on_meta
custom_chat_template: "{% for message in messages %}{{ message['content'] }}{% endfor %}"
dist_config:
  name: fsdp2
  dcp_path: null
  cp_mode: ulysses
  cp_size: ${WORLD_SIZE}
  mp_replicate_size: ${WORLD_SIZE}
  mp_shard_size: 1
train_dataset: ${dcfg}
output_dir: ${LOG_DIR}/${tag}_out
micro_batch_size: ${B}
cutoff_len: ${T}
global_batch_size: ${B}
max_grad_norm: 1.0
max_steps: ${PROBE_STEPS}
bf16: true
enable_activation_checkpointing: false
seed: 42
logging_steps: 1
save_steps: 100000
optim_config:
  name: adamw
  learning_rate: 6.0e-4
  weight_decay: 0.1
  betas: [0.9, 0.95]
  eps: 1.0e-8
lr_scheduler_config:
  name: cosine_with_warmup
  warmup_steps: 1
  min_lr_ratio: 0.1
EOF

  (
    cd "$LF_ROOT" || exit 97
    USE_V1=1 FORCE_TORCHRUN=1 LF_CSV_LOG="$csv" \
    timeout "$RUN_TIMEOUT" "$LF_CLI" train "$ycfg"
  ) >> "$log" 2>&1 &
  local lf_pid=$!
  sample_smi_peak "$lf_pid" "$smi" &
  local smi_pid=$!
  wait "$lf_pid"; local rc=$?
  wait "$smi_pid" 2>/dev/null

  local smi_used="" ; [[ -s "$smi" ]] && smi_used="$(cat "$smi")"
  # torch peak reserved = max mem_gpu_mb (last CSV column) over the probe rows.
  local torch_peak=""
  if [[ -s "$csv" ]]; then
    torch_peak="$(awk -F',' 'NR>1 && $NF ~ /^[0-9.]+$/ {if($NF+0>m)m=$NF+0} END{if(m>0)printf "%.1f", m}' "$csv")"
  fi

  if [[ "$rc" != "0" ]] || [[ -z "$smi_used" && -z "$torch_peak" ]] || \
     grep -qiE "out of memory|CUDA error|RuntimeError|torch.OutOfMemory" "$log"; then
    echo "     FAIL/OOM (rc=$rc) -- see $log"
    echo "LFQ,${label},ulysses,${d},${l},${qh},${ty},${T},${WORLD_SIZE},OOM,${snap}" >> "$RESULTS_CSV"
    rm -f "$pq"; return 1
  fi
  {
    echo "# MEM PROBE SNAPSHOT  tag=${tag}"
    echo "# impl=LFQ label=${label} rotator=ulysses n_embd=${d} n_layer=${l} n_head=${qh} weight_tying=${ty}"
    echo "# B=${B} T=${T} cp_world_size=${WORLD_SIZE} params="
    [[ -n "$torch_peak" ]] && echo "# torch.peak_alloc_mb(rank0)= torch.peak_reserved_mb(rank0)=${torch_peak}"
    echo "# SMI_USED_MB_PER_GPU=${smi_used}"
    echo "# ---- llamafactory-v1 fsdp2+ulysses qwen3 (mem_gpu_mb=torch.max_memory_reserved; smi peak sampled) ----"
  } > "$snap"
  echo "     OK -- snapshot $snap (torch_peak=${torch_peak:-NA}MB smi peak: ${smi_used:-NA})"
  echo "LFQ,${label},ulysses,${d},${l},${qh},${ty},${T},${WORLD_SIZE},OK,${snap}" >> "$RESULTS_CSV"
  rm -f "$pq"; return 0
}

run_probe() {  # impl label d l qh kvh ffn ty T
  local impl="$1"
  case "$impl" in
    CPP)      run_one_cpp "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" ;;
    DSQ|DSG)  run_one_ds  "$impl" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" ;;
    LFQ)      run_one_lf  "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" ;;
  esac
}

sweep_impl() {  # impl
  local impl="$1" row label d l qh kvh ffn ty T
  echo ""
  echo "######## IMPLEMENTATION: $impl ########"
  for row in "${CONFIGS[@]}"; do
    read -r label d l qh kvh ffn ty <<< "$row"
    echo ""
    echo "==== $impl config $label (d_model=$d n_layer=$l q_heads=$qh kv_heads=$kvh ffn=$ffn tie=$ty) ===="
    # LF arm reads a fixed model dir (arch baked in config.json) -> only its matching row.
    if [[ "$impl" == "LFQ" && "$label" != "$LF_LABEL" ]]; then
      echo "  (skip LFQ for $label: LF arm arch = $LF_LABEL / $LF_MODEL_DIR)"; continue
    fi
    # Ulysses head-split feasibility: world_size must divide q_heads (and kv_heads
    # for the GQA impls: bluscriptCP + DSQ). DSG (GPT-2 MHA) only needs q_heads.
    local need_kv=1; [[ "$impl" == "DSG" ]] && need_kv=0
    if (( qh % WORLD_SIZE != 0 )) || { (( need_kv == 1 )) && (( kvh % WORLD_SIZE != 0 )); }; then
      echo "  !! SKIP: world_size=$WORLD_SIZE does not divide q_heads=$qh$([[ $need_kv == 1 ]] && echo " / kv_heads=$kvh") (Ulysses head split)."
      continue
    fi
    local last_ok=0 oom_T=0 T=$T_START
    while (( T <= T_MAX )); do
      if (( T % WORLD_SIZE != 0 )); then
        echo "  (skip T=$T: not divisible by world_size=$WORLD_SIZE)"; T=$(( T * 2 )); continue
      fi
      if run_probe "$impl" "$label" "$d" "$l" "$qh" "$kvh" "$ffn" "$ty" "$T"; then
        last_ok=$T; T=$(( T * 2 ))
      else
        oom_T=$T
        echo "  -> coarse OOM for $impl/$label at T=$oom_T (last OK=$last_ok)"; break
      fi
    done

    if [[ "$FINE_GRAINED" == "1" ]] && (( last_ok > 0 && oom_T > last_ok )); then
      echo "  ~~ fine search for $impl/$label in ($last_ok, $oom_T), step=$FINE_STEP"
      local lo=$last_ok hi=$oom_T mid
      while (( hi - lo > FINE_STEP )); do
        mid=$(( (lo + hi) / 2 )); mid=$(( (mid / WORLD_SIZE) * WORLD_SIZE ))
        if (( mid <= lo || mid >= hi )); then break; fi
        if run_probe "$impl" "$label" "$d" "$l" "$qh" "$kvh" "$ffn" "$ty" "$mid"; then lo=$mid; else hi=$mid; fi
      done
      echo "  == TRUE max-T for $impl/$label: $lo OK, OOM by $hi (res ${FINE_STEP})"
    fi
  done
}

# Run order: RUN_ORDER=cpp_first (default) or ds_first. Override per-run, e.g.
#   RUN_ORDER=ds_first ./mem_scaling_sweep.sh
# LF preflight: needs a llamafactory-cli, an interpreter, the model dir, and the probe helper.
if [[ "$RUN_LF" == "1" ]]; then
  if [[ ! -x "$LF_CLI" ]] && ! command -v llamafactory-cli >/dev/null 2>&1; then
    echo "[LF] ERROR: llamafactory-cli not found ($LF_CLI). Set LF_CLI=... Disabling LF."; RUN_LF=0
  elif [[ ! -f "$LF_PROBE_MK" ]]; then
    echo "[LF] ERROR: probe helper $LF_PROBE_MK missing. Disabling LF."; RUN_LF=0
  elif [[ ! -d "$LF_MODEL_DIR" ]]; then
    echo "[LF] ERROR: LF_MODEL_DIR $LF_MODEL_DIR not found. Disabling LF."; RUN_LF=0
  fi
  # Fall back to a PATH-resolved CLI if the venv path is absent but one is on PATH.
  [[ ! -x "$LF_CLI" ]] && command -v llamafactory-cli >/dev/null 2>&1 && LF_CLI="$(command -v llamafactory-cli)"
fi

RUN_ORDER="${RUN_ORDER:-cpp_first}"
run_cpp_block() { [[ "$RUN_CPP" == "1" ]] && sweep_impl "CPP"; }
run_ds_block()  { [[ "$RUN_DS"  == "1" ]] && { for _m in "${DS_IMPLS[@]}"; do sweep_impl "$_m"; done; }; }
run_lf_block()  { [[ "$RUN_LF"  == "1" ]] && sweep_impl "LFQ"; }

if [[ "$RUN_ORDER" == "ds_first" ]]; then
  run_ds_block
  run_lf_block
  run_cpp_block
else
  run_cpp_block
  run_ds_block
  run_lf_block
fi

echo ""
echo "=============================================================="
echo " Sweep complete."
echo "   Snapshots : $OUT_DIR/*.txt"
echo "   Results   : $RESULTS_CSV"
echo "   Build a table:  python3 ${SCRIPT_DIR}/mem_scaling_table.py $OUT_DIR"
echo "=============================================================="

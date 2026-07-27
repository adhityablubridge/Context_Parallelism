#!/usr/bin/env python3
"""recover_ws2.py -- rebuild the ws2 table from surviving leftovers.

The LF 48M ws2 .txt snapshots AND its results.csv were wiped/overwritten by the
ws4/114M run (shared-dir naming bug). The ONLY surviving LF ws2 truth is the
per-run CSV logs (logs/LFQ_48M_ulysses_T<T>_ws2.csv), each carrying per-step
mem_gpu_mb (torch reserved) + tok_per_sec. This script:

  1. Reconstructs LFQ_48M_..._ws2.txt snapshots from those CSV logs:
       peak_mb   = max(mem_gpu_mb)                 (torch reserved -- LF's native peak)
       tok/sec   = median(tok_per_sec, steps>=2)   (skip step-1 warmup)
       status    = OK  if the run logged a step>=2 (survived past first fwd/bwd)
                   OOM if it only reached step 1 or logged nothing (crashed early)
  2. Copies the 3 surviving CPP ws2 .txt snapshots in as-is.
  3. Writes a merged results.csv (LF inferred rows) so mem_scaling_table.py can
     fold the OOM boundary into the limits summary.

Caveats printed at the end -- read them. smi (cross-impl) mem is unrecoverable for
LF (was .txt-only); CPP ws2 only has 3 T points (that sweep was truncated at 8192).
"""
import csv
import glob
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LF_DIR = os.path.join(HERE, "mem_scaling_runs_bluscriptcp_LF_ws2")
CPP_DIR = os.path.join(HERE, "mem_scaling_runs_bluscriptcp_ws2")
OUT_DIR = os.path.join(HERE, "mem_scaling_runs_ws2_recovered")

# 48M arch (d384/L6/qh6/kvh2/hd64/ffn1024, untied).
ARCH = dict(n_embd=384, n_layer=6, n_head=6, weight_tying=0, params=48076416)


def read_csv_log(path):
    steps = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                step = int(r["step"])
                mem = float(r["mem_gpu_mb"])
                tps = float(r["tok_per_sec"])
            except (KeyError, TypeError, ValueError):
                continue
            steps.append((step, mem, tps))
    return steps


def main():
    if not os.path.isdir(LF_DIR):
        sys.exit(f"ERROR: {LF_DIR} not found")
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. reconstruct LF snapshots from csv logs
    lf_rows = []  # (T, peak_mb, tok_sec, n_steps, status)
    for csvp in sorted(glob.glob(os.path.join(LF_DIR, "logs", "LFQ_48M_ulysses_T*_ws2.csv"))):
        base = os.path.basename(csvp)
        T = int(base.split("_T")[1].split("_ws")[0])
        steps = read_csv_log(csvp)
        max_step = max((s for s, _, _ in steps), default=0)
        status = "OK" if max_step >= 2 else "OOM"
        peak = max((m for _, m, _ in steps), default=float("nan"))
        tail = [t for s, _, t in steps if s >= 2]
        tps = sorted(tail)[len(tail) // 2] if tail else float("nan")
        lf_rows.append((T, peak, tps, len(steps), status))

        if status == "OK":
            snap = os.path.join(OUT_DIR, f"LFQ_48M_ulysses_T{T}_ws2.txt")
            with open(snap, "w") as f:
                f.write(f"# MEM PROBE SNAPSHOT (RECOVERED FROM CSV)  tag=LFQ_48M_ulysses_T{T}_ws2\n")
                f.write(f"# impl=Lfq label=48M rotator=ulysses n_embd={ARCH['n_embd']} "
                        f"n_layer={ARCH['n_layer']} n_head={ARCH['n_head']} "
                        f"weight_tying={ARCH['weight_tying']}\n")
                f.write(f"# B=1 T={T} cp_world_size=2 params={ARCH['params']}\n")
                f.write(f"# torch.peak_reserved_mb(rank0)={peak:.1f}\n")
                f.write(f"# RECOVERED tok_per_sec_median={tps:.1f} (smi unavailable -- csv-only)\n")

    # 2. copy CPP ws2 snapshots
    cpp_n = 0
    for txt in sorted(glob.glob(os.path.join(CPP_DIR, "CPP_*_ws2.txt"))):
        shutil.copy2(txt, OUT_DIR)
        cpp_n += 1

    # 3. write results.csv so table.py folds the OOM boundary
    res = os.path.join(OUT_DIR, "mem_scaling_results.csv")
    with open(res, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["impl", "label", "rotator", "n_embd", "n_layer", "n_head",
                    "weight_tying", "T", "world_size", "status", "snapshot"])
        for T, peak, tps, n, status in sorted(lf_rows):
            w.writerow(["LFQ", "48M", "ulysses", ARCH["n_embd"], ARCH["n_layer"],
                        ARCH["n_head"], ARCH["weight_tying"], T, 2, status, ""])

    # summary
    lf_ok = [T for T, _, _, _, s in lf_rows if s == "OK"]
    lf_oom = [T for T, _, _, _, s in lf_rows if s == "OOM"]
    print("=== LF 48M ws2 recovered (T, peak_mb, tok/sec, n_steps, status) ===")
    for T, peak, tps, n, status in sorted(lf_rows):
        print(f"  T={T:<7} peak={peak:>9.1f}MB  tok/s={tps:>10.1f}  steps={n}  {status}")
    print(f"\nLF max_T_ok = {max(lf_ok) if lf_ok else 'NA'}   "
          f"first_T_oom = {min(lf_oom) if lf_oom else 'NA (no OOM in recovered logs)'}")
    print(f"CPP ws2 snapshots copied: {cpp_n} (T points limited -- that sweep was truncated)")
    print(f"\nCombined dir: {OUT_DIR}")
    print(f"Now run:  python3 {os.path.join(HERE, 'mem_scaling_table.py')} {OUT_DIR}")
    print("\nCAVEATS: LF peak_mb = torch reserved (smi cross-impl metric unrecoverable, was .txt-only). "
          "LF OK/OOM inferred from csv step-count. CPP ws2 covers only T<=8192.")


if __name__ == "__main__":
    main()

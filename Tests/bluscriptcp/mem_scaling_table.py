#!/usr/bin/env python3
"""
mem_scaling_table.py — aggregate memory-scaling snapshots into a table.

Pairs OUR bluscriptCP (C++ CP-Ulysses) against the DeepSpeed-Ulysses Qwen3
reference. Reads the per-run snapshot files written by mem_scaling_sweep.sh
(CPP_*.txt / DS_*.txt) plus mem_scaling_results.csv (for OOM rows), and emits a
tidy CSV + Markdown comparison table and a max-T-before-OOM limits summary.

Usage:
    python3 mem_scaling_table.py [OUT_DIR]   # default OUT_DIR=mem_scaling_runs_bluscriptcp
"""
import os
import re
import sys
import csv

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "mem_scaling_runs_bluscriptcp"


def parse_snapshot(path):
    """Pull the header fields from one snapshot file into a dict."""
    rec = {
        "impl": "", "label": "", "rotator": "", "n_embd": "", "n_layer": "",
        "n_head": "", "weight_tying": "", "B": "", "T": "", "world_size": "",
        "params": "", "torch_peak_reserved_mb": "", "torch_peak_alloc_mb": "",
        "cuda_used_mb": "", "smi_max_used_mb": "", "status": "OK",
    }
    with open(path) as f:
        for line in f:
            if line.startswith("# impl="):
                for key in ("impl", "label", "rotator", "n_embd", "n_layer",
                            "n_head", "weight_tying"):
                    m = re.search(rf"{key}=(\S+)", line)
                    if m:
                        rec[key] = m.group(1)
            elif line.startswith("# B="):
                for key in ("B", "T", "params"):
                    m = re.search(rf"{key}=(\S+)", line)
                    if m:
                        rec[key] = m.group(1)
                m = re.search(r"cp_world_size=(\S+)", line)
                if m:
                    rec["world_size"] = m.group(1)
            elif "torch.peak_reserved_mb" in line:
                m = re.search(r"torch\.peak_reserved_mb\(rank0\)=([\d.]+)", line)
                if m:
                    rec["torch_peak_reserved_mb"] = m.group(1)
                m = re.search(r"torch\.peak_alloc_mb\(rank0\)=([\d.]+)", line)
                if m:
                    rec["torch_peak_alloc_mb"] = m.group(1)
            elif "cudaMemGetInfo used_mb" in line:
                m = re.search(r"used_mb\(rank0\)=([\d.]+)", line)
                if m:
                    rec["cuda_used_mb"] = m.group(1)
            elif line.startswith("# SMI_USED_MB_PER_GPU="):
                payload = line.split("=", 1)[1].strip()
                # format: "idx,used;idx,used;..."
                vals = []
                for grp in payload.split(";"):
                    parts = grp.split(",")
                    if len(parts) >= 2 and parts[1].strip().isdigit():
                        vals.append(int(parts[1].strip()))
                if vals:
                    rec["smi_max_used_mb"] = str(max(vals))
    return rec


def main():
    if not os.path.isdir(OUT_DIR):
        print(f"ERROR: {OUT_DIR} not found", file=sys.stderr)
        sys.exit(1)

    records = []
    for fn in sorted(os.listdir(OUT_DIR)):
        if fn.endswith(".txt") and (fn.startswith("CPP_") or
                                    fn.startswith("DSQ_") or fn.startswith("DSG_")):
            records.append(parse_snapshot(os.path.join(OUT_DIR, fn)))

    # Fold in OOM rows from results.csv (runs that produced no snapshot).
    res_csv = os.path.join(OUT_DIR, "mem_scaling_results.csv")
    have = {(r["impl"], r["label"], r["rotator"], r["T"]) for r in records}
    if os.path.isfile(res_csv):
        with open(res_csv) as f:
            for row in csv.DictReader(f):
                if row.get("status") == "OOM":
                    key = (row["impl"], row["label"], row.get("rotator", ""),
                           row["T"])
                    if key not in have:
                        records.append({
                            "impl": row["impl"], "label": row["label"],
                            "rotator": row.get("rotator", ""),
                            "n_embd": row["n_embd"], "n_layer": row["n_layer"],
                            "n_head": row["n_head"],
                            "weight_tying": row["weight_tying"],
                            "B": "", "T": row["T"],
                            "world_size": row["world_size"], "params": "",
                            "torch_peak_reserved_mb": "",
                            "torch_peak_alloc_mb": "", "cuda_used_mb": "",
                            "smi_max_used_mb": "", "status": "OOM",
                        })

    # Collapse each impl's native peak metric into a single peak_mb column:
    # bluscriptCP -> cudaMemGetInfo used; DS-Ulysses -> torch peak reserved.
    for r in records:
        if r.get("torch_peak_reserved_mb"):
            r["peak_mb"] = r["torch_peak_reserved_mb"]
            r["peak_src"] = "torch_reserved"
        elif r.get("cuda_used_mb"):
            r["peak_mb"] = r["cuda_used_mb"]
            r["peak_src"] = "cudaMemGetInfo"
        else:
            r["peak_mb"] = ""
            r["peak_src"] = ""

    # Drop any stray record with no rotator (e.g. a pre-rotator smoke test).
    records = [r for r in records if r.get("rotator") or r.get("status") == "OOM"]

    # Normalize impl names so OK snapshots and OOM CSV rows for the same config
    # merge. Snapshots + OOM rows both use CPP / DSQ / DSG codes:
    #   c*   -> bluscriptCP      (cudaMemGetInfo peak)
    #   dsq  -> DS-Qwen3         (torch reserved peak)
    #   dsg  -> DS-GPT2          (torch reserved peak)
    #   else -> DS-Ulysses       (fallback)
    def canon_impl(x):
        x = x.lower()
        if x.startswith("c"):
            return "bluscriptCP"
        if x == "dsq":
            return "DS-Qwen3"
        if x == "dsg":
            return "DS-GPT2"
        return "DS-Ulysses"
    for r in records:
        r["impl"] = canon_impl(r["impl"])

    def sort_key(r):
        try:
            t = int(r["T"])
        except Exception:
            t = 0
        return (r["impl"], r["label"], r["rotator"], t)

    records.sort(key=sort_key)

    cols = ["impl", "label", "params", "rotator", "n_embd", "n_layer", "n_head",
            "weight_tying", "T", "world_size", "smi_max_used_mb",
            "peak_mb", "peak_src", "status"]

    out_csv = os.path.join(OUT_DIR, "mem_scaling_table.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)

    out_md = os.path.join(OUT_DIR, "mem_scaling_table.md")
    with open(out_md, "w") as f:
        f.write("# Memory Scaling Results — bluscriptCP vs DeepSpeed-Ulysses Qwen3\n\n")
        f.write("`smi_max_used_mb` = max per-GPU MiB from live nvidia-smi "
                "(includes CUDA context/NCCL), filtered to the run's GPUs — the "
                "apples-to-apples cross-impl metric. `peak_mb` = each impl's "
                "native peak: bluscriptCP=cudaMemGetInfo used, "
                "DS-Ulysses=torch max_reserved (see `peak_src`). Blank rows = OOM "
                "(no snapshot produced).\n\n")
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for r in records:
            f.write("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n")

    # ---- Limits summary: per (impl,label,rotator) max OK T and first OOM T ----
    from collections import defaultdict
    ok_by = defaultdict(list)
    oom_by = defaultdict(list)
    for r in records:
        try:
            t = int(r["T"])
        except Exception:
            continue
        key = (r["impl"], r["label"], r["rotator"])
        if r["status"] == "OK":
            ok_by[key].append(t)
        elif r["status"] == "OOM":
            oom_by[key].append(t)

    keys = sorted(set(list(ok_by) + list(oom_by)))
    lim_cols = ["impl", "label", "rotator", "max_T_ok", "first_T_oom"]
    out_lim_csv = os.path.join(OUT_DIR, "mem_scaling_limits.csv")
    out_lim_md = os.path.join(OUT_DIR, "mem_scaling_limits.md")
    with open(out_lim_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(lim_cols)
        for k in keys:
            max_ok = max(ok_by[k]) if ok_by[k] else ""
            first_oom = min(oom_by[k]) if oom_by[k] else ""
            w.writerow([k[0], k[1], k[2], max_ok, first_oom])
    with open(out_lim_md, "w") as f:
        f.write("# Scaling Limits (max T that ran OK / first T that OOM'd)\n\n")
        f.write("| " + " | ".join(lim_cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(lim_cols)) + "|\n")
        for k in keys:
            max_ok = max(ok_by[k]) if ok_by[k] else ""
            first_oom = min(oom_by[k]) if oom_by[k] else ""
            f.write(f"| {k[0]} | {k[1]} | {k[2]} | {max_ok} | {first_oom} |\n")

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")
    print(f"Wrote {out_lim_csv}")
    print(f"Wrote {out_lim_md}")
    print(f"Rows: {len(records)}")


if __name__ == "__main__":
    main()

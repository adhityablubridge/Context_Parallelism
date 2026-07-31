#!/usr/bin/env python3
# =============================================================================
# plot_ppl_sweep.py -- headless PPL-vs-position plots for the CREAM/PoSE/RandPos/
# naive sweep. Same logic as the graph_bluscript_cp.ipynb cell, but runs from a
# terminal (no Jupyter) and auto-detects every ppl_<method>_x<F>.csv in the CWD.
#
#   python3 plot_ppl_sweep.py            # one comparison PNG per factor
#
# Writes ppl_x<F>.png per factor + prints the verdict table (ratio ~1 == flat).
# =============================================================================
import glob, os, re
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")                       # headless (works over SSH)
import matplotlib.pyplot as plt

ORIG_CTX = 4096
METHODS = {"cream": "CREAM", "pose": "PoSE", "rand": "RandPos", "naive": "naive"}

def smooth(y, k=64):
    if k <= 1 or len(y) < k:
        return y
    return np.convolve(y, np.ones(k) / k, mode="same")

files = glob.glob("ppl_*_x*.csv")
factors = sorted({int(m.group(1)) for f in files
                  if (m := re.search(r"_x(\d+)\.csv$", f))})
if not factors:
    raise SystemExit("no ppl_<method>_x<F>.csv files found in this directory")
print("factors found:", factors)

print(f'\n{"curve":24s}  PPL<{ORIG_CTX}  PPL>={ORIG_CTX}  ratio')
for F in factors:
    plt.figure(figsize=(11, 6))
    plotted = False
    for key, label in METHODS.items():
        path = f"ppl_{key}_x{F}.csv"
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path)
        d = d[d["count"] > 0]
        if d.empty:
            continue
        plt.plot(d["pos"], smooth(d["ppl"].values), label=f"{label} x{F}", linewidth=1.3)
        plotted = True
        before = d[d["pos"] < ORIG_CTX]["ppl"].mean()
        after = d[d["pos"] >= ORIG_CTX]["ppl"].mean()
        ratio = (after / before) if before == before else float("nan")
        print(f'{label+" x"+str(F):24s}  {before:7.2f}  {after:8.2f}  {ratio:5.2f}')
    if not plotted:
        plt.close()
        continue
    plt.axvline(ORIG_CTX, color="gray", ls="--", alpha=0.7, label=f"orig ctx = {ORIG_CTX}")
    plt.yscale("log")
    plt.xlabel("token position")
    plt.ylabel("perplexity (smoothed, log)")
    plt.title(f"PPL-vs-position (48M) x{F} ({4096*F//1024}k): methods")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    out = f"ppl_x{F}.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  -> wrote {out}\n")

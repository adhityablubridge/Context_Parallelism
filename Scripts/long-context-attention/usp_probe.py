"""usp_probe.py -- 2-step memory/throughput probe for USP (yunchang) hybrid CP.

Mirrors bluscriptCP's CP_MEM_PROBE: builds the block-stack model at the given
dims + (ring, ulysses) topology, runs PROBE_STEPS forward+backward+optimizer
steps at sequence length T, then writes a snapshot .txt in the SAME format
mem_scaling_table.py parses (impl=Usp -> canon "USP"). Launched by the sweep via
  torchrun --nproc_per_node=<cp_size> usp_probe.py ...
with dp=1 (world_size == ring*ulysses), so per-rank memory is directly
comparable to the bluscriptCP hybrid rows at the same (ring, ulysses, T).
"""
import argparse
import os
import subprocess

import torch
import torch.distributed as dist

from yunchang import set_seq_parallel_pg

from usp_model import USPModel, shard_sequence, count_params


def smi_used_per_gpu():
    """'idx,used;idx,used;...' from nvidia-smi, honoring CUDA_VISIBLE_DEVICES."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            encoding="utf-8",
        )
        return ";".join(",".join(p.strip() for p in line.split(",")) for line in out.strip().splitlines())
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)          # e.g. 48M_r2u2
    ap.add_argument("--ring", type=int, required=True)
    ap.add_argument("--ulysses", type=int, required=True)
    ap.add_argument("--ring-impl", default="zigzag")   # basic|zigzag|stripe
    ap.add_argument("--d-model", type=int, required=True)
    ap.add_argument("--n-layer", type=int, required=True)
    ap.add_argument("--q-heads", type=int, required=True)
    ap.add_argument("--kv-heads", type=int, required=True)
    ap.add_argument("--head-dim", type=int, required=True)
    ap.add_argument("--ffn", type=int, required=True)
    ap.add_argument("--tie", type=int, default=0)
    ap.add_argument("--B", type=int, default=2)
    ap.add_argument("--T", type=int, required=True)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--rotator", default="p2p")        # label only (matches CPP grouping)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    cp_size = args.ring * args.ulysses
    assert world_size == cp_size, (
        f"usp_probe runs dp=1: launch with nproc_per_node={cp_size} (got world_size={world_size})"
    )
    # USP process groups: (ulysses_degree, ring_degree, rank, world_size).
    set_seq_parallel_pg(args.ulysses, args.ring, rank, world_size)

    torch.manual_seed(0)
    model = USPModel(args.d_model, args.n_layer, args.q_heads, args.kv_heads,
                     args.head_dim, args.ffn, args.ring_impl).to(device).to(torch.bfloat16)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    n_params = count_params(model)

    torch.cuda.reset_peak_memory_stats(device)
    ok = True
    err = ""
    try:
        for _ in range(args.steps):
            # Global hidden -> local shard (same layout USP expects for this ring impl).
            g = torch.randn(args.B, args.T, args.d_model, device=device, dtype=torch.bfloat16)
            dist.broadcast(g, src=0)
            local = shard_sequence(g, rank, world_size, args.ring, args.ulysses, args.ring_impl)
            local.requires_grad_(True)
            out = model(local)
            loss = out.float().pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        torch.cuda.synchronize(device)
    except torch.cuda.OutOfMemoryError as e:
        ok = False
        err = f"CUDA out of memory: {e}"
    except RuntimeError as e:
        ok = False
        err = str(e)
        if "out of memory" not in err.lower():
            raise

    if rank == 0:
        peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 * 1024)
        peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        free_b, total_b = torch.cuda.mem_get_info(device)
        used_mb = (total_b - free_b) / (1024 * 1024)
        if not ok:
            print(f"[usp_probe] OOM/FAIL at T={args.T}: {err[:200]}", flush=True)
        else:
            tag = f"USP_{args.label}_{args.rotator}_T{args.T}_ws{world_size}"
            os.makedirs(args.out_dir, exist_ok=True)
            snap = os.path.join(args.out_dir, f"{tag}.txt")
            with open(snap, "w") as f:
                f.write(f"# MEM PROBE SNAPSHOT (USP / yunchang)  tag={tag}\n")
                f.write(f"# impl=Usp label={args.label} rotator={args.rotator} "
                        f"n_embd={args.d_model} n_layer={args.n_layer} n_head={args.q_heads} "
                        f"weight_tying={args.tie}\n")
                f.write(f"# B={args.B} T={args.T} cp_world_size={world_size} params={n_params}\n")
                f.write(f"# ring_impl={args.ring_impl} ring={args.ring} ulysses={args.ulysses}\n")
                f.write(f"# torch.peak_reserved_mb(rank0)={peak_reserved:.1f} "
                        f"torch.peak_alloc_mb(rank0)={peak_alloc:.1f}\n")
                f.write(f"# cudaMemGetInfo used_mb(rank0)={used_mb:.1f}\n")
                f.write(f"# SMI_USED_MB_PER_GPU={smi_used_per_gpu()}\n")
            print(f"[usp_probe] OK T={args.T} peak_reserved={peak_reserved:.1f}MB "
                  f"used={used_mb:.1f}MB -> {snap}", flush=True)

    dist.barrier()
    dist.destroy_process_group()
    # Non-zero exit on OOM so the sweep's is_fail() catches it.
    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

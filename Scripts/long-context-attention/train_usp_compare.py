"""train_usp_compare.py -- a logged training run of the USP (yunchang) hybrid-CP
block stack, so its throughput / step-time / memory can be compared against a
bluscriptCP training run at the same dims, topology, and sequence length.

This is a SYSTEMS comparison harness, not a convergence study: the model is
random-init and the "data" is random hidden states with an autoregressive-shift
MSE surrogate loss, so per-step FLOPs and communication match a real block stack
while staying tokenizer/dataset-free. Each step logs to a CSV with the same
columns bluscriptCP emits (step, loss, lr, dt_ms, tok_per_sec, mem_gpu_mb) so the
existing plotting/extract tooling reads it directly.

Launch (dp=1, world_size == ring*ulysses):
  torchrun --nproc_per_node=<cp_size> train_usp_compare.py \
      --label 114M_r2u2 --ring 2 --ulysses 2 --ring-impl zigzag \
      --d-model 768 --n-layer 12 --q-heads 12 --kv-heads 4 --head-dim 64 \
      --ffn 2048 --B 2 --T 16384 --steps 200 --csv logs/usp_114M_r2u2_T16384.csv
"""
import argparse
import csv
import os
import time

import torch
import torch.distributed as dist

from yunchang import set_seq_parallel_pg

from usp_model import USPModel, shard_sequence, count_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--ring", type=int, required=True)
    ap.add_argument("--ulysses", type=int, required=True)
    ap.add_argument("--ring-impl", default="zigzag")
    ap.add_argument("--d-model", type=int, required=True)
    ap.add_argument("--n-layer", type=int, required=True)
    ap.add_argument("--q-heads", type=int, required=True)
    ap.add_argument("--kv-heads", type=int, required=True)
    ap.add_argument("--head-dim", type=int, required=True)
    ap.add_argument("--ffn", type=int, required=True)
    ap.add_argument("--tie", type=int, default=0)
    ap.add_argument("--vocab", type=int, default=50304)
    ap.add_argument("--B", type=int, default=2)
    ap.add_argument("--T", type=int, required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    cp_size = args.ring * args.ulysses
    assert world_size == cp_size, f"launch with nproc_per_node={cp_size} (dp=1)"
    set_seq_parallel_pg(args.ulysses, args.ring, rank, world_size)

    torch.manual_seed(0)
    model = USPModel(args.d_model, args.n_layer, args.q_heads, args.kv_heads,
                     args.head_dim, args.ffn, args.ring_impl, args.vocab,
                     tie=bool(args.tie)).to(device).to(torch.bfloat16)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    n_params = count_params(model)
    global_tokens = args.B * args.T  # tokens advanced per optimizer step (whole global seq)

    writer = None
    fcsv = None
    if rank == 0:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        fcsv = open(args.csv, "w", newline="")
        writer = csv.writer(fcsv)
        writer.writerow(["step", "loss", "lr", "dt_ms", "tok_per_sec", "mem_gpu_mb", "params",
                         "ring", "ulysses", "ring_impl", "T", "world_size"])
        print(f"[train_usp] {args.label} params={n_params/1e6:.1f}M ring={args.ring} "
              f"uly={args.ulysses} impl={args.ring_impl} T={args.T} B={args.B} ws={world_size}",
              flush=True)

    torch.cuda.reset_peak_memory_stats(device)
    for step in range(1, args.steps + 1):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        ids = torch.randint(0, args.vocab, (args.B, args.T), device=device)
        dist.broadcast(ids, src=0)
        loss = model(ids, rank, world_size, args.ring, args.ulysses)  # full LM forward + CE
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        torch.cuda.synchronize(device)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if rank == 0:
            mem_mb = torch.cuda.max_memory_reserved(device) / (1024 * 1024)
            tok_s = global_tokens / (dt_ms / 1000.0)
            writer.writerow([step, f"{loss.item():.6f}", f"{args.lr:.3e}", f"{dt_ms:.3f}",
                             f"{tok_s:.1f}", f"{mem_mb:.1f}", n_params, args.ring, args.ulysses,
                             args.ring_impl, args.T, world_size])
            fcsv.flush()
            if step <= 3 or step % 20 == 0:
                print(f"[train_usp] step {step:>4} loss={loss.item():.4f} "
                      f"dt={dt_ms:.1f}ms tok/sec={tok_s:.1f} mem={mem_mb:.0f}MB", flush=True)

    if rank == 0:
        # steady-state summary (skip warmup)
        fcsv.close()
        import statistics
        with open(args.csv) as f:
            rows = list(csv.DictReader(f))
        tail = rows[args.warmup:] if len(rows) > args.warmup else rows
        med_tok = statistics.median(float(r["tok_per_sec"]) for r in tail)
        peak_mem = max(float(r["mem_gpu_mb"]) for r in rows)
        print(f"[train_usp] STEADY-STATE median tok/sec={med_tok:.1f}  peak_mem={peak_mem:.0f}MB "
              f"(over {len(tail)} steps after {args.warmup} warmup) -> {args.csv}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

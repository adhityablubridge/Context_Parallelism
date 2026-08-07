#!/usr/bin/env python3
# =============================================================================
# longrope_search.py -- LongRoPE evolutionary search driver (offline, no CUDA).
#
# Searches per-dimension rescale factors lambda (len head_dim/2) + n_hat that
# minimize overall perplexity, using bluscriptCP's resident evaluator
# (CP_LONGROPE_SEARCH=1) as the fitness function -- the model is loaded ONCE and
# each candidate only rebuilds the (cheap CPU) LongRoPE cache.
#
# Faithful to Ding et al. 2024 (arXiv:2402.13753), Table 4 / Alg. 1:
#   grid lambda_i in [1, s*1.25] step 0.01; n_hat in a fixed set; monotone
#   lambda_0<=...; seed PI/NTK/YaRN; mutation = per-dim RESAMPLE at p=0.3 (not a
#   jitter); crossover = per-dim pick from either parent; repair = running-max
#   (NEVER sort). Memoize on discretized grid indices.
#
# Usage (example):
#   python3 Tests/bluscriptcp/longrope_search.py \
#       --exec ./build/bluscriptCP_exec --ckpt-run 31 --target-t 16384 --s 4 \
#       --head-dim 64 --orig-maxpos 4096 --data-root $HOME/Downloads \
#       --arch "CP_SIZE=1 CP_N_EMBD=384 CP_N_LAYER=6 CP_N_HEAD=6 CP_N_KVHEAD=2 CP_FFN=1024 CP_WEIGHT_TYING=0" \
#       --pop 64 --iters 40 --calib-windows 5 --out longrope_best.txt
#   (use small --pop/--iters for a smoke run)
# =============================================================================
import argparse, json, math, os, queue, random, subprocess, sys, tempfile, threading, time

def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exec", default="./build/bluscriptCP_exec")
    ap.add_argument("--ckpt-run", type=int, required=True)
    ap.add_argument("--target-t", type=int, required=True, help="search target length (= factor*T if composed)")
    ap.add_argument("--s", type=int, required=True, help="extension ratio (= YARN_SCALE)")
    ap.add_argument("--head-dim", type=int, required=True)
    ap.add_argument("--base", type=float, default=500000.0, help="rope_theta")
    ap.add_argument("--orig-maxpos", type=float, default=4096.0)
    ap.add_argument("--beta-fast", type=float, default=32.0)
    ap.add_argument("--beta-slow", type=float, default=1.0)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--arch", required=True, help="space-separated CP_* env for the model shape")
    ap.add_argument("--calib-windows", type=int, default=5)
    ap.add_argument("--pop", type=int, default=64)
    ap.add_argument("--n1", type=int, default=16, help="mutation children/gen")
    ap.add_argument("--n2", type=int, default=16, help="crossover children/gen")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--pmut", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=0,
                    help="early-stop if best PPL has not improved by >--tol for this many "
                         "consecutive generations (0=off). Right-sizes the search to the model.")
    ap.add_argument("--tol", type=float, default=0.05, help="min PPL improvement to reset patience")
    ap.add_argument("--time-budget-sec", type=float, default=0.0,
                    help="0=unlimited; else stop the GA after this many wall-clock seconds "
                         "(writes best-so-far). Guarantees the search fits a time budget.")
    ap.add_argument("--out", default="longrope_best.txt")
    ap.add_argument("--cuda-devices", default=None, help="CUDA_VISIBLE_DEVICES for the evaluator")
    ap.add_argument("--state", default=None,
                    help="resume base path. Writes <state>.memo.jsonl (every evaluated candidate, "
                         "so restarts never repeat a forward pass) and <state>.state.json (population "
                         "+ generation, saved each gen). Re-running with the same --state resumes.")
    ap.add_argument("--pool", type=int, default=1,
                    help="candidate-parallel workers. Scores that many candidates at once, each on "
                         "its own GPU subset (needs --cuda-devices with pool*CP_SIZE GPUs). Default 1 "
                         "== the serial evaluator (byte-identical). Result is identical to serial for "
                         "any pool size (see Tests/bluscriptcp/test_pool_equivalence.sh).")
    return ap.parse_args()

# ---- grid ------------------------------------------------------------------
LAM_MIN = 1.0
STEP = 0.01
N_HAT_SET = [0, 1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 64, 128, 256]

def make_grid(s):
    lam_max = s * 1.25
    n = int(round((lam_max - LAM_MIN) / STEP)) + 1
    return [round(LAM_MIN + k * STEP, 2) for k in range(n)], lam_max

def snap(x, lam_max):
    x = min(max(x, LAM_MIN), lam_max)
    return round(round((x - LAM_MIN) / STEP) * STEP + LAM_MIN, 2)

def repair_monotone(lam, lam_max):
    # forward running-max clamp -- preserves each dim's value except where it
    # violates order. NEVER sort (a sort scrambles dim->factor correspondence).
    out, m = [], LAM_MIN
    for v in lam:
        m = max(m, v)
        out.append(snap(m, lam_max))
    return out

def grid_key(lam, nhat):
    return (tuple(int(round((v - LAM_MIN) / STEP)) for v in lam), nhat)

# ---- seeds: port of YARNOps.cpp find_correction_range + inv_freq ------------
def yarn_ramps(head_dim, base, s, L, b_fast, b_slow):
    half = head_dim // 2
    def find_dim(num_rot):
        return (head_dim * math.log(L / (num_rot * 2.0 * math.pi))) / (2.0 * math.log(base))
    low = max(0, math.floor(find_dim(b_fast)))
    high = min(half - 1, math.ceil(find_dim(b_slow)))
    denom = float(high - low)
    if denom <= 0.0:
        denom = 1e-3
    ramps = []
    for i in range(half):
        r = (i - low) / denom
        ramps.append(min(1.0, max(0.0, r)))
    return ramps

def seed_PI(half, s, lam_max):
    return [snap(float(s), lam_max)] * half

def seed_YaRN(head_dim, s, L, b_fast, b_slow, base, lam_max):
    half = head_dim // 2
    ramps = yarn_ramps(head_dim, base, s, L, b_fast, b_slow)
    # lambda_YaRN[i] = 1 / ((ramp/s) + (1-ramp)); clamp>=1 then snap+repair
    lam = []
    for r in ramps:
        denom = (r / s) + (1.0 - r)
        lam.append(max(1.0, 1.0 / denom))
    return repair_monotone([snap(v, lam_max) for v in lam], lam_max)

def seed_NTK(head_dim, s, lam_max):
    # lambda_NTK[i] = s^(2i/(d-2))  (== base_freq_i / ntk_inv_freq_i)
    half = head_dim // 2
    d = head_dim
    lam = [max(1.0, s ** (2.0 * i / (d - 2))) for i in range(half)]
    return repair_monotone([snap(v, lam_max) for v in lam], lam_max)

# ---- factor file -----------------------------------------------------------
def write_factors(path, lam, nhat, s, search_len):
    with open(path, "w") as f:
        f.write(f"n_hat {nhat}\n")
        f.write(f"s {s}\n")
        f.write(f"S_search {search_len}\n")
        f.write("lambda " + " ".join(f"{v:.2f}" for v in lam) + "\n")

# ---- resident evaluator ----------------------------------------------------
def arch_cp_size(arch):
    """CP degree per candidate = CP_SIZE in --arch (default 1)."""
    cp = 1
    for kv in arch.split():
        k, v = kv.split("=", 1)
        if k == "CP_SIZE":
            cp = int(v)
    return cp

class Evaluator:
    # `devices` overrides CUDA_VISIBLE_DEVICES for THIS worker (a pool gives each
    # worker its own GPU subset); None falls back to args.cuda_devices (serial path).
    def __init__(self, args, devices=None):
        env = dict(os.environ)
        for kv in args.arch.split():
            k, v = kv.split("=", 1)
            env[k] = v
        env["CP_FUSED_ROPE"] = "1"
        env["CP_LONGROPE_SEARCH"] = "1"
        env["CP_T"] = str(args.target_t)
        env["CP_B"] = "1"
        env["CP_EVAL_WINDOWS"] = str(args.calib_windows)
        env["CP_CKPT_RESUME"] = str(args.ckpt_run)
        env["CP_DATA_ROOT"] = args.data_root
        env["YARN_SCALE"] = str(args.s)
        env["YARN_ORIG_MAXPOS"] = str(int(args.orig_maxpos))
        dev = devices if devices is not None else args.cuda_devices
        if dev is not None:
            env["CUDA_VISIBLE_DEVICES"] = dev
        ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ("BluTrain/Tensor-Implementations/lib:"
                                  "BluTrain/Profiler/lib:" + ld)
        # Launch one rank per CP shard (derived from CP_SIZE in --arch). The C++
        # resident evaluator broadcasts each candidate genome from rank 0 to all
        # ranks and reduces the fitness, so the GA still feeds ONE stdin and reads
        # ONE "PPL <v>" line back (non-master ranks print nothing). devices must
        # expose all CP GPUs (comma list) so cudaSetDevice(rank) binds rank->GPU.
        self.cp_size = arch_cp_size(args.arch)
        self.devices = dev
        cmd = ["mpirun", "-np", str(self.cp_size), args.exec]
        self.tmp = tempfile.mkdtemp(prefix="lrsearch_")
        # capture the evaluator's stderr so we can surface the REAL error on death
        self.errpath = os.path.join(self.tmp, "evaluator.stderr")
        self.errfile = open(self.errpath, "w+")
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=self.errfile, env=env, text=True, bufsize=1)
        self._n = 0

    def _stderr_tail(self, n=30):
        try:
            self.errfile.flush()
            with open(self.errpath) as f:
                return "".join(f.readlines()[-n:])
        except Exception:
            return "(could not read evaluator stderr)"

    def eval(self, lam, nhat, s, search_len):
        path = os.path.join(self.tmp, f"cand_{self._n}.txt")
        self._n += 1
        write_factors(path, lam, nhat, s, search_len)
        self.p.stdin.write(path + "\n")
        self.p.stdin.flush()
        # read until a "PPL <v>" or "ERR ..." line (skip evaluator chatter)
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("resident evaluator died. stderr tail:\n" + self._stderr_tail())
            line = line.strip()
            if line.startswith("PPL "):
                return float(line.split()[1])
            if line.startswith("ERR "):
                raise RuntimeError("evaluator: " + line)

    def close(self):
        try:
            self.p.stdin.close(); self.p.wait(timeout=10)
        except Exception:
            self.p.kill()

# ---- candidate-parallel pool ----------------------------------------------
class EvaluatorPool:
    """Scores a generation's candidates concurrently across P worker evaluators,
    each on its own GPU subset. RESULT-IDENTICAL to serial: fitness is a pure
    function of (lam,nhat); the RNG is never touched here; and memo/memo.jsonl are
    mutated only on the caller thread AFTER a barrier (single writer -> no races,
    no duplicate lines, deterministic order). P==1 == the serial evaluator.
    Threads (not processes) suffice: eval() blocks on subprocess I/O, releasing the
    GIL, and each worker owns its own subprocess (no shared pipe)."""
    def __init__(self, args, memo, memo_fh):
        self.args, self.memo, self.memo_fh = args, memo, memo_fh
        cp = arch_cp_size(args.arch)
        P = max(1, args.pool)
        if args.cuda_devices:
            devs = args.cuda_devices.split(",")
            if len(devs) != P * cp:
                raise SystemExit(f"--pool {P} x CP_SIZE {cp} = {P*cp} GPUs, but --cuda-devices "
                                 f"lists {len(devs)} ({args.cuda_devices}). They must match.")
            subsets = [",".join(devs[i*cp:(i+1)*cp]) for i in range(P)]
        else:
            subsets = [None] * P
        self.workers = [Evaluator(args, devices=s) for s in subsets]
        self.P = P
        print(f"[pool] {P} worker(s) x CP={cp} GPU(s) each", flush=True)

    def fitness_batch(self, cands):
        """cands: list of (lam, nhat). Returns PPLs in the SAME input order.
        Memo hits are free; unique uncached candidates are scored across workers."""
        keys = [grid_key(lam, nh) for (lam, nh) in cands]
        # unique + uncached, first-occurrence order (dedup within the batch so a
        # repeated candidate is scored ONCE -> matches serial's memo behavior).
        todo, seen = [], set()
        for (lam, nh), k in zip(cands, keys):
            if k in self.memo or k in seen:
                continue
            seen.add(k); todo.append((k, lam, nh))

        results, results_lock = {}, threading.Lock()
        live, live_lock = set(self.workers), threading.Lock()
        errors = []
        q = queue.Queue()
        for item in todo:
            q.put(item)

        def work(w):
            while True:
                try:
                    k, lam, nh = q.get(timeout=1.0)
                except queue.Empty:
                    with results_lock:
                        if len(results) >= len(todo):   # everything scored -> done
                            return
                    continue                            # in-flight items remain; keep waiting
                try:
                    ppl = w.eval(lam, nh, self.args.s, self.args.target_t)
                    with results_lock:
                        results[k] = ppl
                except Exception as ex:                 # worker died
                    with live_lock:
                        live.discard(w)
                        remaining = len(live)
                        errors.append(str(ex))
                    if remaining > 0:
                        q.put((k, lam, nh))             # re-dispatch to a live worker
                    return

        threads = [threading.Thread(target=work, args=(w,), daemon=True) for w in self.workers]
        for t in threads: t.start()
        for t in threads: t.join()

        if len(results) != len(todo):
            tail = ("; last error: " + errors[-1]) if errors else ""
            raise RuntimeError(f"pool scored {len(results)}/{len(todo)} candidates -- "
                               f"{len(self.workers)-len(live)} worker(s) died{tail}")

        # Single-writer memo + jsonl on THIS thread, in first-occurrence order:
        # deterministic, no races, identical memo contents to the serial path.
        for (k, lam, nh) in todo:
            ppl = results[k]
            self.memo[k] = ppl
            if self.memo_fh:
                self.memo_fh.write(json.dumps({"lam": list(lam), "nhat": nh, "ppl": ppl}) + "\n")
                self.memo_fh.flush()
        return [self.memo[k] for k in keys]

    def close(self):
        for w in self.workers:
            w.close()

# ---- resume helpers --------------------------------------------------------
def _rng_to_json(st):
    return {"v": st[0], "s": list(st[1]), "g": st[2]}

def _rng_from_json(d):
    return (d["v"], tuple(d["s"]), d["g"])

def save_ga_state(path, gen, scored, best_prev, no_improve, spent_sec):
    # Atomic write (tmp + rename) so a kill mid-save can't corrupt the state file.
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"gen": gen,
                   "scored": [[p, list(lam), nh] for (p, lam, nh) in scored],
                   "best_prev": best_prev, "no_improve": no_improve,
                   "spent_sec": spent_sec, "rng": _rng_to_json(random.getstate())}, f)
    os.replace(tmp, path)

# ---- GA --------------------------------------------------------------------
def main():
    args = build_args()
    random.seed(args.seed)
    half = args.head_dim // 2
    grid, lam_max = make_grid(args.s)

    memo = {}
    # ---- resume: warm the memo from every previously-evaluated candidate so a
    # restart NEVER repeats a forward pass (the expensive part). ----
    memo_path  = (args.state + ".memo.jsonl") if args.state else None
    state_path = (args.state + ".state.json") if args.state else None
    if memo_path and os.path.exists(memo_path):
        nload = 0
        with open(memo_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                memo[grid_key(d["lam"], d["nhat"])] = d["ppl"]; nload += 1
        print(f"[resume] warmed memo with {nload} cached evaluations from {memo_path}", flush=True)
    memo_fh = open(memo_path, "a") if memo_path else None

    # Candidate-parallel pool (pool=1 == the serial evaluator). fitness_batch(cands)
    # scores a whole generation at once and returns PPLs in INPUT order. The
    # RNG-driven child construction stays serial (below), so the candidate set/order
    # is unchanged and the result is identical for any pool size.
    pool = EvaluatorPool(args, memo, memo_fh)

    def mutate(lam, nhat):
        child = [snap(random.choice(grid), lam_max) if random.random() < args.pmut else lam[i]
                 for i in range(half)]
        nh = random.choice(N_HAT_SET) if random.random() < args.pmut else nhat
        return repair_monotone(child, lam_max), nh

    def crossover(a, b):
        lam = [a[0][i] if random.random() < 0.5 else b[0][i] for i in range(half)]
        nh = a[1] if random.random() < 0.5 else b[1]
        return repair_monotone(lam, lam_max), nh

    # ---- init: PI / NTK / YaRN + mutated fill ----
    seeds = [
        (seed_PI(half, args.s, lam_max), 0),
        (seed_NTK(args.head_dim, args.s, lam_max), 0),
        (seed_YaRN(args.head_dim, args.s, args.orig_maxpos, args.beta_fast,
                   args.beta_slow, args.base, lam_max), 0),
    ]
    # ---- resume the GA structure from a saved generation, if present ----
    resumed = False
    if state_path and os.path.exists(state_path):
        with open(state_path) as f:
            st = json.load(f)
        scored     = [(p, lam, nh) for (p, lam, nh) in st["scored"]]
        best_prev  = st["best_prev"]; no_improve = st["no_improve"]
        start_it   = st["gen"]; spent0 = float(st["spent_sec"])
        try:
            random.setstate(_rng_from_json(st["rng"]))
        except Exception:
            pass
        run_start = time.time()
        resumed = True
        print(f"[resume] GA state: gen={start_it} pop={len(scored)} best={scored[0][0]:.4f} "
              f"spent={spent0/3600:.2f}h -- continuing", flush=True)

    if not resumed:
        pop = list(seeds)
        while len(pop) < args.pop:
            base = random.choice(seeds)
            pop.append(mutate(base[0], base[1]))
        run_start = time.time(); spent0 = 0.0
        # Budget precondition floor: one FULL generation past the seeds. A search that
        # only reaches the seeds (PI/NTK/YaRN) cannot demonstrate searched-vs-formula.
        floor = args.pop + 2 * (args.n1 + args.n2)
        # Score candidate #1 alone (to TIME one forward), run the pool-aware precheck,
        # then score the rest of the population as ONE parallel batch. The floor
        # guarantees the whole population fits, so gen 0 is never truncated. Results
        # are assembled in pop order -> identical to serial for any pool size.
        t1 = time.time()
        first_ppl = pool.fitness_batch([pop[0]])[0]
        per1 = time.time() - t1
        if args.time_budget_sec:
            fittable = int(args.time_budget_sec * pool.P / per1) if per1 > 0 else 10 ** 9
            print(f"[precheck] candidate #1 took {per1:.1f}s; ~{fittable} fit in "
                  f"{args.time_budget_sec:.0f}s across {pool.P} worker(s) "
                  f"(floor={floor}, one generation past seeds)", flush=True)
            if fittable < floor:
                raise RuntimeError(
                    f"budget precondition FAILED: ~{fittable} candidates fit but the floor is {floor}. "
                    f"A seeds-only search proves nothing -- cut CALIB or the search length T, raise "
                    f"--time-budget-sec, or add more --pool workers.")
        rest = pool.fitness_batch(pop[1:]) if len(pop) > 1 else []
        ppls = [first_ppl] + rest
        scored = [(ppls[i], pop[i][0], pop[i][1]) for i in range(len(pop))]
        scored.sort(key=lambda x: x[0])
        print(f"[gen 0] seeds -> best PPL {scored[0][0]:.4f} "
              f"(PI={memo.get(grid_key(seeds[0][0],0),'?')}, "
              f"NTK={memo.get(grid_key(seeds[1][0],0),'?')}, "
              f"YaRN={memo.get(grid_key(seeds[2][0],0),'?')})", flush=True)
        best_prev = scored[0][0]; no_improve = 0; start_it = 0
        if state_path:
            save_ga_state(state_path, 0, scored, best_prev, no_improve, time.time() - run_start)

    # cumulative wall-clock across restarts (spent0 carries prior runs' time)
    def elapsed():
        return spent0 + (time.time() - run_start)

    for it in range(start_it, args.iters):
        if args.time_budget_sec and elapsed() > args.time_budget_sec:
            print(f"[gen {it}] time budget ({args.time_budget_sec:.0f}s) reached -- stopping early", flush=True)
            break
        parents = scored[:args.topk]
        children = []
        for _ in range(args.n1):
            _, lam, nh = random.choice(parents)
            children.append(mutate(lam, nh))
        for _ in range(args.n2):
            pa, pb = random.sample(parents, 2)
            children.append(crossover((pa[1], pa[2]), (pb[1], pb[2])))
        # Score the whole generation in parallel (results in child order). The budget
        # is checked at the top of the loop, so a generation is atomic -- pool size
        # never changes WHICH candidates are scored, only how fast.
        ppls = pool.fitness_batch(children)
        for (lam, nh), ppl in zip(children, ppls):
            scored.append((ppl, lam, nh))
        scored = sorted(scored, key=lambda x: x[0])[:args.pop]
        # early-stopping on convergence (right-sizes the search to the model)
        if scored[0][0] < best_prev - args.tol:
            best_prev = scored[0][0]; no_improve = 0
        else:
            no_improve += 1
        print(f"[gen {it+1}] best PPL {scored[0][0]:.4f}  n_hat={scored[0][2]}  "
              f"evals={len(memo)}  no_improve={no_improve}  elapsed={elapsed()/3600:.2f}h", flush=True)
        # persist the GA structure at each generation boundary (resume point).
        if state_path:
            save_ga_state(state_path, it + 1, scored, best_prev, no_improve, elapsed())
        if args.patience and no_improve >= args.patience:
            print(f"[gen {it+1}] converged: no PPL improvement > {args.tol} for {args.patience} gens "
                  f"-- stopping", flush=True)
            break

    pool.close()
    if memo_fh:
        memo_fh.close()
    best_ppl, best_lam, best_nhat = scored[0]
    write_factors(args.out, best_lam, best_nhat, args.s, args.target_t)
    print(f"\nBEST PPL {best_ppl:.4f}  n_hat={best_nhat}  -> {args.out}", flush=True)

if __name__ == "__main__":
    main()

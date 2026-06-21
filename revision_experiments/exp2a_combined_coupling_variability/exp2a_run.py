"""Exp 2a runner: single-DGP repeats, 2-way parallel, time-budgeted, checkpointed.
Conditions:
  combined : sim(conn>0,std>0) -> fit warp-ON and warp-OFF on same data
  ref      : sim(conn>0,std=0) -> plain fit (clean true-coupling reference)
  nocoup   : sim(conn=0,std>0) -> fit warp-ON and warp-OFF (false-positive control)
Each record stores recovered source->target coupling filter summary + LRT p.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import sys, time, pickle, argparse
sys.path.insert(0, "/sessions/gifted-cool-babbage/mnt/MultiNeuronGLM")
os.chdir("/sessions/gifted-cool-babbage/mnt/MultiNeuronGLM")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from scipy.stats import chi2
import GLM

STD1, CORR1, STD2, CORR2 = 10.0, 0.5, 25.0, 0.9
CONN = 0.008
NUM_POP, NUM_CP, PEAKS_MAX, TAU, NUM_F_REF = 50, 3, 15, 10, 4
PENALTY_POP = 1e-5
WARP_INTERVAL = [[0, 0.15], [0.15, 0.35]]
MAX_ITER = 5

def build(ntrial, nt, src, tgt, with_src=True):
    m = GLM.PP_GLM(ntrial=ntrial, nt=nt, select_trials=np.array([True]*ntrial))
    m.add_effect("inhomogeneous_baseline", num=NUM_POP, apply_no_penalty=True)
    if with_src:
        m.add_effect("coupling", raw_input=src, num=NUM_CP, peaks_max=PEAKS_MAX, nonlinear=1)
    m.add_effect("coupling", raw_input=tgt, num=NUM_CP, peaks_max=PEAKS_MAX, nonlinear=1)
    m.add_effect("refractory_additive", raw_input=tgt, tau=TAU, num=NUM_F_REF, apply_no_penalty=False)
    return m

def fit_model(m, tgt, warp):
    if warp:
        m.fit_time_warping_baseline(target=tgt, max_iter=MAX_ITER, warp_interval=WARP_INTERVAL,
                                    method="mine", penalty=PENALTY_POP, verbose=False)
    else:
        m.fit(target=tgt, method="mine", penalty=PENALTY_POP, verbose=False)
    return m

def summ(m):
    f = np.asarray(m.get_filter()[1])
    i = int(np.argmax(np.abs(f)))
    return {"integral": float(np.sum(f)), "peak": float(np.max(np.abs(f))),
            "peak_signed": float(f[i]), "peak_lag": i, "filter": f.tolist()}

def fit_pair(ntrial, nt, src, tgt, warp):
    full = fit_model(build(ntrial, nt, src, tgt, True), tgt, warp)
    nest = fit_model(build(ntrial, nt, src, tgt, False), tgt, warp)
    df = full.predictors.shape[1] - nest.predictors.shape[1]
    stat = 2.0*(nest.nll - full.nll)
    out = summ(full)
    out["lrt_stat"] = float(stat); out["df"] = int(df)
    out["lrt_p"] = float(chi2.sf(stat, df)) if df > 0 else float("nan")
    return out

def pool_spikes(sp):
    nt, nn, ntr = sp.shape
    h = nn//2
    return nt, ntr, sp[:, :h, :].sum(1), sp[:, h:, :].sum(1)

def worker(task):
    cond, ntrial, seed = task
    np.random.seed(seed)
    t0 = time.time()
    rec = {"cond": cond, "seed": seed, "ntrial": ntrial}
    if cond == "combined":
        sp = GLM.EIF_simulator(STD1, CORR1, STD2, CORR2, ntrial, CONN)
        nt, ntr, src, tgt = pool_spikes(sp)
        rec["warp"] = fit_pair(ntr, nt, src, tgt, True)
        rec["nowarp"] = fit_pair(ntr, nt, src, tgt, False)
    elif cond == "ref":
        sp = GLM.EIF_simulator(0.0, 0.0, 0.0, 0.0, ntrial, CONN)
        nt, ntr, src, tgt = pool_spikes(sp)
        rec["plain"] = fit_pair(ntr, nt, src, tgt, False)
    elif cond == "nocoup":
        sp = GLM.EIF_simulator(STD1, CORR1, STD2, CORR2, ntrial, 0.0)
        nt, ntr, src, tgt = pool_spikes(sp)
        rec["warp"] = fit_pair(ntr, nt, src, tgt, True)
        rec["nowarp"] = fit_pair(ntr, nt, src, tgt, False)
    rec["secs"] = time.time() - t0
    return rec

def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/sessions/gifted-cool-babbage/mnt/outputs/exp2a_results.pkl")
    ap.add_argument("--cond", required=True, choices=["combined", "ref", "nocoup"])
    ap.add_argument("--ntrial", type=int, default=100)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--wave", type=int, default=2)
    ap.add_argument("--budget", type=float, default=38.0)
    args = ap.parse_args()
    t_start = time.time()
    results = []
    if os.path.exists(args.out):
        with open(args.out, "rb") as fh:
            results = pickle.load(fh)
    done = set((r["cond"], r["seed"]) for r in results)
    pending = [(args.cond, args.ntrial, args.seed0 + i) for i in range(args.reps)
               if (args.cond, args.seed0 + i) not in done]
    if not pending:
        print("NOTHING TODO for", args.cond, "; have",
              sum(1 for r in results if r["cond"] == args.cond), flush=True); return
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=args.wave)
    try:
        i = 0
        while i < len(pending) and (time.time() - t_start) < args.budget:
            batch = pending[i:i + args.wave]
            i += args.wave
            for rec in pool.map(worker, batch):
                results.append(rec)
                with open(args.out, "wb") as fh:
                    pickle.dump(results, fh)
                key = "plain" if rec["cond"] == "ref" else "warp"
                msg = rec.get("warp", rec.get("plain"))
                extra = ""
                if "nowarp" in rec:
                    extra = " nowarp_int=%.3f nowarp_p=%.3g" % (rec["nowarp"]["integral"], rec["nowarp"]["lrt_p"])
                print("%s seed=%d %.1fs int=%.3f peak=%.3f p=%.3g%s"
                      % (rec["cond"], rec["seed"], rec["secs"], msg["integral"], msg["peak"], msg["lrt_p"], extra), flush=True)
    finally:
        pool.close(); pool.join()
    n = sum(1 for r in results if r["cond"] == args.cond)
    print("BATCH DONE; %s now has %d repeats" % (args.cond, n), flush=True)

if __name__ == "__main__":
    main()

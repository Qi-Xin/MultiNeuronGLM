"""Exp 2b (Reviewer optional comment 2b): are coupling estimates explained by trial gain?

Reviewer asks whether reduced V1->LM coupling during locomotion could be a by-product of
the accompanying gain (firing-rate) change, suggesting stratifying trials by gain and
comparing coupling in high- vs low-gain trials.

Synthetic validation (the real-data version, runnable on the lab server, is in
exp2b_realdata_TEMPLATE.py): simulate EIF populations with TRUE coupling held CONSTANT
across trials while trials vary in response amplitude (gain) due to peak-timing jitter and
Poisson variability. If pop-GLM's coupling estimate is gain-invariant here, then a coupling
difference measured between behavioural conditions cannot be a mere artifact of a gain
difference -> the method dissociates gain from coupling.

Per repeat: simulate combined (conn>0, std>0); define empirical per-trial gain = total target
population spike count; split trials at the median into HIGH- and LOW-gain halves; fit the
warped pop-GLM separately in each half; record V1->LM coupling integral + LRT p.
"""
import os
os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"; os.environ["MKL_NUM_THREADS"]="1"
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
WARP_INTERVAL = [[0,0.15],[0.15,0.35]]
MAX_ITER = 5

def build(ntrial, nt, src, tgt, with_src=True):
    m = GLM.PP_GLM(ntrial=ntrial, nt=nt, select_trials=np.array([True]*ntrial))
    m.add_effect("inhomogeneous_baseline", num=NUM_POP, apply_no_penalty=True)
    if with_src:
        m.add_effect("coupling", raw_input=src, num=NUM_CP, peaks_max=PEAKS_MAX, nonlinear=1)
    m.add_effect("coupling", raw_input=tgt, num=NUM_CP, peaks_max=PEAKS_MAX, nonlinear=1)
    m.add_effect("refractory_additive", raw_input=tgt, tau=TAU, num=NUM_F_REF, apply_no_penalty=False)
    return m

def fit_warp(m, tgt):
    m.fit_time_warping_baseline(target=tgt, max_iter=MAX_ITER, warp_interval=WARP_INTERVAL,
                                method="mine", penalty=PENALTY_POP, verbose=False)
    return m

def coupling_and_p(nt, src, tgt):
    ntrial = src.shape[1]
    full = fit_warp(build(ntrial, nt, src, tgt, True), tgt)
    nest = fit_warp(build(ntrial, nt, src, tgt, False), tgt)
    f = np.asarray(full.get_filter()[1])
    df = full.predictors.shape[1] - nest.predictors.shape[1]
    stat = 2.0*(nest.nll - full.nll)
    p = float(chi2.sf(stat, df)) if df>0 else float("nan")
    return float(np.sum(f)), float(np.max(np.abs(f))), p

def worker(task):
    ntrial, seed = task
    np.random.seed(seed); t0=time.time()
    sp = GLM.EIF_simulator(STD1, CORR1, STD2, CORR2, ntrial, CONN)
    nt, nn, ntr = sp.shape
    h = nn//2
    src = sp[:, :h, :].sum(1); tgt = sp[:, h:, :].sum(1)
    gain = tgt.sum(axis=0)                       # empirical per-trial gain (total target spikes)
    med = np.median(gain)
    hi = gain >= med; lo = gain < med
    rec = {"seed": seed, "ntrial": ntrial, "gain_hi_mean": float(gain[hi].mean()),
           "gain_lo_mean": float(gain[lo].mean())}
    int_hi, pk_hi, p_hi = coupling_and_p(nt, src[:, hi], tgt[:, hi])
    int_lo, pk_lo, p_lo = coupling_and_p(nt, src[:, lo], tgt[:, lo])
    rec["hi"] = {"integral": int_hi, "peak": pk_hi, "lrt_p": p_hi, "ntrial": int(hi.sum())}
    rec["lo"] = {"integral": int_lo, "peak": pk_lo, "lrt_p": p_lo, "ntrial": int(lo.sum())}
    rec["secs"] = time.time()-t0
    return rec

def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/sessions/gifted-cool-babbage/mnt/outputs/exp2b_results.pkl")
    ap.add_argument("--ntrial", type=int, default=120)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=5000)
    ap.add_argument("--wave", type=int, default=2)
    ap.add_argument("--budget", type=float, default=40.0)
    args = ap.parse_args()
    t_start=time.time()
    results = []
    if os.path.exists(args.out):
        results = pickle.load(open(args.out,"rb"))
    done = set(r["seed"] for r in results)
    pending = [(args.ntrial, args.seed0+i) for i in range(args.reps) if (args.seed0+i) not in done]
    if not pending:
        print("NOTHING TODO; have", len(results)); return
    ctx = mp.get_context("spawn"); pool = ctx.Pool(processes=args.wave)
    try:
        i=0
        while i < len(pending) and (time.time()-t_start) < args.budget:
            batch = pending[i:i+args.wave]; i += args.wave
            for rec in pool.map(worker, batch):
                results.append(rec)
                pickle.dump(results, open(args.out,"wb"))
                print("seed=%d %.1fs  HIgain n=%d int=%.3f p=%.3g | LOgain n=%d int=%.3f p=%.3g"
                      % (rec["seed"], rec["secs"], rec["hi"]["ntrial"], rec["hi"]["integral"], rec["hi"]["lrt_p"],
                         rec["lo"]["ntrial"], rec["lo"]["integral"], rec["lo"]["lrt_p"]), flush=True)
    finally:
        pool.close(); pool.join()
    print("BATCH DONE; total", len(results), flush=True)

if __name__ == "__main__":
    main()

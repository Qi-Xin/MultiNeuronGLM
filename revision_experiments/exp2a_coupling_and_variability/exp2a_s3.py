"""Exp 2a — Scenario 3: coupling AND trial-to-trial baseline variation BOTH present.

The paper already shows:
  * Sensitivity  : coupling exists, NO trial-to-trial variation -> fitted filter should be nonzero
  * Time warping : NO coupling, trial-to-trial variation exists -> fitted filter should be zero
Reviewer 2a asks for the combined case. We show:
  1. pop-GLM (with time warping) recovers a NONZERO coupling filter
  2. pop-GLM WITHOUT the time-warping component gives a LARGER (inflated) filter
  3. single-neuron GLM filters are noisy but scattered AROUND the pop-GLM filter

Hyperparameters exactly as in 'STAR Synthetic dataset; Pillow et al.ipynb' (Sensitivity section).
"""
import os; os.environ["OMP_NUM_THREADS"]="2"
import sys, time, pickle, argparse
sys.path.insert(0,"/sessions/gifted-cool-babbage/mnt/MultiNeuronGLM")
os.chdir("/sessions/gifted-cool-babbage/mnt/MultiNeuronGLM")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from scipy.stats import chi2
import GLM

# --- paper's hyperparameters (Sensitivity section) ---
NUM=20; NUM_CP=3; PEAKS_MAX=15; NONLIN=1
TAU=10; NUM_FREF=4
PENALTY=1e0; PENALTY_POP=1e0
NTRIAL=100; CONN=0.0075
WARP=[[0,0.15],[0.15,0.35]]; MAXIT=5
# trial-to-trial baseline variation (from the Time-warping section)
STD1,CORR1,STD2,CORR2 = 10, 0.5, 25, 0.9
OUT="/sessions/gifted-cool-babbage/mnt/outputs/exp2a_s3.pkl"

def simulate(std_on, seed):
    np.random.seed(seed)
    s1,c1,s2,c2 = (STD1,CORR1,STD2,CORR2) if std_on else (0,0.0,0,0.0)
    sp = GLM.EIF_simulator(s1,c1,s2,c2,NTRIAL,CONN)      # (nt, 20, ntrial)
    nt,nn,ntr = sp.shape
    src=[sp[:,i,:] for i in range(nn//2)]                # 10 source neurons
    tgt=[sp[:,i,:] for i in range(nn//2,nn)]             # 10 target neurons
    return nt,ntr,src,tgt

def pop_fit(nt,ntr,src,tgt,warp):
    """pop-GLM on the pooled trains; returns source->target coupling filter + LRT p."""
    S=sum(src); T=sum(tgt)
    def build(with_src):
        m=GLM.PP_GLM(ntrial=ntr, nt=nt, select_trials=np.array([True]*ntr))
        m.add_effect("inhomogeneous_baseline", num=NUM, apply_no_penalty=True)
        if with_src:
            m.add_effect("coupling", raw_input=S, num=NUM_CP, peaks_max=PEAKS_MAX, nonlinear=NONLIN)
        m.add_effect("coupling", raw_input=T, num=NUM_CP, peaks_max=PEAKS_MAX, nonlinear=NONLIN)
        m.add_effect("refractory_additive", raw_input=T, tau=TAU, num=NUM_FREF, apply_no_penalty=False)
        if warp:
            m.fit_time_warping_baseline(target=T, max_iter=MAXIT, warp_interval=WARP,
                                        method="mine", penalty=PENALTY_POP, verbose=False)
        else:
            m.fit(target=T, method="mine", penalty=PENALTY_POP, verbose=False)
        return m
    full=build(True); nest=build(False)
    f=np.asarray(full.get_filter()[1])
    df=full.predictors.shape[1]-nest.predictors.shape[1]
    p=float(chi2.sf(2*(nest.nll-full.nll), df)) if df>0 else float("nan")
    return f.tolist(), p

def single_fits(nt,ntr,src,tgt):
    """single-neuron GLMs: for each target neuron, coupling from all 10 sources + all 10 targets.
    Returns the 10 source->target filters per target neuron (get_filter()[1:11])."""
    out=[]
    for T in tgt:
        m=GLM.PP_GLM(ntrial=ntr, nt=nt, select_trials=np.array([True]*ntr))
        m.add_effect("inhomogeneous_baseline", num=NUM, apply_no_penalty=True)
        for S in src: m.add_effect("coupling", raw_input=S, num=NUM_CP, peaks_max=PEAKS_MAX, nonlinear=NONLIN)
        for S in tgt: m.add_effect("coupling", raw_input=S, num=NUM_CP, peaks_max=PEAKS_MAX, nonlinear=NONLIN)
        m.fit(target=T, method="mine", penalty=PENALTY, verbose=False)
        out += [np.asarray(f).tolist() for f in m.get_filter()[1:11]]
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cond",required=True,choices=["both","ref"])   # both = coupling+variation; ref = coupling only (no variation)
    ap.add_argument("--reps",type=int,default=5); ap.add_argument("--budget",type=float,default=38.0)
    a=ap.parse_args(); t0=time.time()
    R=pickle.load(open(OUT,"rb")) if os.path.exists(OUT) else {}
    R.setdefault(a.cond,[])
    while len(R[a.cond])<a.reps and time.time()-t0<a.budget:
        seed=100+len(R[a.cond]); t=time.time()
        nt,ntr,src,tgt = simulate(std_on=(a.cond=="both"), seed=seed)
        rec={"seed":seed}
        fw,pw = pop_fit(nt,ntr,src,tgt,warp=True);  rec["pop_warp"]=fw;   rec["p_warp"]=pw
        if a.cond=="both":
            fn,pn = pop_fit(nt,ntr,src,tgt,warp=False); rec["pop_nowarp"]=fn; rec["p_nowarp"]=pn
            rec["single"]=single_fits(nt,ntr,src,tgt)
        R[a.cond].append(rec); pickle.dump(R,open(OUT,"wb"))
        msg="[%s] rep %d (%.0fs)  pop-warp int=%.4f (p=%.1e)"%(a.cond,len(R[a.cond]),time.time()-t,np.sum(fw),pw)
        if a.cond=="both":
            msg+="  | pop-nowarp int=%.4f  | single n=%d"%(np.sum(fn),len(rec["single"]))
        print(msg,flush=True)
    print("progress: both=%d ref=%d"%(len(R.get("both",[])),len(R.get("ref",[]))),flush=True)
if __name__=="__main__": main()

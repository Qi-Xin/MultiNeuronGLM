"""Exp 2b — Reviewer's test: stratify trials by GAIN (not by locomotion) and compare V1->LM coupling.

  "fitting models in subsets of trials stratified by gain ... whether high-gain trials show weaker
   V1->LM coupling than low-gain trials.  If high-gain trials show weaker coupling, this would be
   consistent with the idea that locomotion introduces shared global drive rather than stronger
   feedforward propagation."

All 210 trials (running + stationary together) are split at the median V1 firing rate, and the
EXACT main-results pop-GLM is fitted in each half:
  target = LM (probeD); coupling from ALL six areas (V1->LM conditioned on AL/RL/AM/PM);
  refractory_additive (f_damp); trial_coef (trial-wise gain); time-warped baseline;
  num_basis_baseline=20, penalty=5e-1, max_iter=10, coupling={'peaks_max':20.2,'num':3,'nonlinear':0.5}.
The per-trial gain and warping terms absorb the baseline differences between running and stationary
trials that coexist inside each half.

Significance: permutation test that randomly reassigns the high/low labels (design matrix built once
with the warping held fixed; only rows / trial_coef columns are sliced per permutation).

--split rate   : split on raw V1 firing rate  (the reviewer's test)
--split resid  : split on V1 rate residualised within stimulus condition (controls for which of the
                 14 gratings was shown, isolating trial-to-trial gain rather than stimulus preference)
"""
import os; os.environ["OMP_NUM_THREADS"]="2"
import sys, time, pickle, argparse
sys.path.insert(0,"/sessions/gifted-cool-babbage/mnt/MultiNeuronGLM")
sys.path.insert(0,"/sessions/gifted-cool-babbage/mnt/outputs")
os.chdir("/sessions/gifted-cool-babbage/mnt/MultiNeuronGLM")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import GLM, utility_functions as utils
from popglm_data import Dataset, load_membership

PROBES=['probeA','probeB','probeC','probeD','probeE','probeF']
CP={'peaks_max':20.2,'num':3,'nonlinear':0.5}
NUM_BASE=20; PEN=5e-1; MAXIT=10; TAU=15; NFREF=4
ds=Dataset(); MEMB,CIDS=load_membership()
POOL={p:utils.pooling_pop(MEMB,CIDS,ds,p,0) for p in PROBES}
TGT=POOL['probeD']; NT=ds.nt; NTR=ds.ntrial

def build(sel):
    m=GLM.PP_GLM(dataset=ds, select_trials=sel, membership=MEMB, condition_ids=CIDS)
    m.add_effect('inhomogeneous_baseline', num=NUM_BASE, apply_no_penalty=True)
    for p in PROBES: m.add_effect('coupling', POOL[p], apply_no_penalty=True, **CP)
    m.add_effect('refractory_additive', TGT, tau=TAU, num=NFREF, apply_no_penalty=True)
    m.add_effect('trial_coef')
    return m

POOL_ALL={p:utils.pooling_pop(MEMB,CIDS,ds,p,0,use_all=True) for p in PROBES}
def gain_vector(kind):
    # global population gain: total firing rate of ALL neurons across ALL six areas per trial
    rate = sum(POOL_ALL[p][ds.npadding:,:].sum(0) for p in PROBES)
    if kind=="rate": return rate
    cond = ds.presentation_table['stimulus_condition_id'].values
    r = rate.astype(float).copy()
    for k in np.unique(cond):                            # residualise within stimulus condition
        m = cond==k; r[m] -= rate[m].mean()
    return r

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--split",default="rate",choices=["rate","resid"])
    ap.add_argument("--nperm",type=int,default=1000); ap.add_argument("--budget",type=float,default=37.0)
    a=ap.parse_args(); t0=time.time()
    OUT="/sessions/gifted-cool-babbage/mnt/outputs/exp2b_gainALL_%s.pkl"%a.split
    SH ="/sessions/gifted-cool-babbage/mnt/outputs/exp2b_gain_shifts.npy"

    allsel=np.full(NTR,True)
    if os.path.exists(SH): shifts=np.load(SH)
    else:
        m=build(allsel); m.fit_time_warping_baseline(TGT,verbose=False,max_iter=MAXIT,penalty=PEN)
        shifts=m.shifts; np.save(SH,shifts)

    # design matrix once, warping fixed
    m=build(allsel); m.fit_time_warping_baseline(TGT,verbose=False,max_iter=MAXIT,penalty=PEN,fix_shifts=shifts)
    X=m.predictors; y=m.response
    widths=[e.shape[1] for e in m.effect_list]; ncf=int(np.sum(widths[:-1]))
    v1s=int(np.sum(widths[:3])); v1e=v1s+widths[3]          # effect 3 = coupling from V1
    v1_basis=m.basis_list[3]; nopen=m.no_penalty
    full_int=float(np.sum(np.asarray(m.get_filter()[3])))

    def filt(ks):
        rows=np.concatenate([np.arange(k*NT,(k+1)*NT) for k in ks])
        cols=np.concatenate([np.arange(ncf), ncf+np.asarray(ks)])
        res=GLM.poisson_regression(y[rows], X[np.ix_(rows,cols)], L2_pen=PEN, no_penalty=nopen)
        return v1_basis @ res.params[v1s:v1e]
    integ=lambda ks: float(np.sum(filt(ks)))

    g=gain_vector(a.split)
    n_hi=int(ds.running_trial_index.sum())          # match the locomotion split (81 running)
    order=np.argsort(g)[::-1]                        # highest gain first
    hi=order[:n_hi]; lo=order[n_hi:]                 # top 81 vs bottom 129
    st=pickle.load(open(OUT,"rb")) if os.path.exists(OUT) else {"perm":[]}
    if "D_obs" not in st:
        rate=POOL['probeC'][ds.npadding:,:].sum(0)
        run=ds.running_trial_index
        st.update(split=a.split, n_hi=len(hi), n_lo=len(lo),
                  rate_hi=float(rate[hi].mean()), rate_lo=float(rate[lo].mean()),
                  frac_run_hi=float(run[hi].mean()), frac_run_lo=float(run[lo].mean()),
                  f_hi=filt(hi).tolist(), f_lo=filt(lo).tolist(),
                  int_hi=integ(hi), int_lo=integ(lo), full_int=full_int)
        st["D_obs"]=st["int_hi"]-st["int_lo"]
        pickle.dump(st,open(OUT,"wb"))
        print("[%s] hi n=%d (V1 rate %.0f, %.0f%% running) | lo n=%d (V1 rate %.0f, %.0f%% running)"
              %(a.split,st["n_hi"],st["rate_hi"],100*st["frac_run_hi"],st["n_lo"],st["rate_lo"],100*st["frac_run_lo"]),flush=True)
        print("      V1->LM integral:  high-gain=%.3f   low-gain=%.3f   D_obs=%+.4f   (all trials %.3f)"
              %(st["int_hi"],st["int_lo"],st["D_obs"],full_int),flush=True)
    rng=np.random.RandomState(7+len(st["perm"]))
    while len(st["perm"])<a.nperm and time.time()-t0<a.budget:
        p=rng.permutation(NTR); A=p[:st["n_hi"]]; B=p[st["n_hi"]:]
        st["perm"].append(integ(A)-integ(B))
        if len(st["perm"])%25==0: pickle.dump(st,open(OUT,"wb"))
    pickle.dump(st,open(OUT,"wb"))
    null=np.array(st["perm"]); pv=float(np.mean(np.abs(null)>=abs(st["D_obs"])))
    print("[%s] perms=%d  D_obs=%+.4f  null|D|=%.4f  p=%.4f  (%.0fs)"
          %(a.split,len(null),st["D_obs"],np.abs(null).mean(),pv,time.time()-t0),flush=True)
if __name__=="__main__": main()

import os; os.environ["OMP_NUM_THREADS"]="2"
import sys, pickle, numpy as np
sys.path.insert(0,"/sessions/gifted-cool-babbage/mnt/MultiNeuronGLM")
sys.path.insert(0,"/sessions/gifted-cool-babbage/mnt/outputs")
os.chdir("/sessions/gifted-cool-babbage/mnt/MultiNeuronGLM")
import warnings; warnings.filterwarnings("ignore")
import GLM, utility_functions as utils
from popglm_data import Dataset, load_membership

PROBES=['probeA','probeB','probeC','probeD','probeE','probeF']
CP={'peaks_max':20.2,'num':3,'nonlinear':0.5}
NUM_BASE=20; PEN=5e-1; MAXIT=10; TAU=15; NFREF=4
ds=Dataset(); MEMB,CIDS=load_membership()
POOL={p:utils.pooling_pop(MEMB,CIDS,ds,p,0) for p in PROBES}
POOL_ALL={p:utils.pooling_pop(MEMB,CIDS,ds,p,0,use_all=True) for p in PROBES}
TGT=POOL['probeD']; NT=ds.nt; NTR=ds.ntrial

def build(sel):
    m=GLM.PP_GLM(dataset=ds, select_trials=sel, membership=MEMB, condition_ids=CIDS)
    m.add_effect('inhomogeneous_baseline', num=NUM_BASE, apply_no_penalty=True)
    for p in PROBES: m.add_effect('coupling', POOL[p], apply_no_penalty=True, **CP)
    m.add_effect('refractory_additive', TGT, tau=TAU, num=NFREF, apply_no_penalty=True)
    m.add_effect('trial_coef')
    return m

allsel=np.full(NTR,True)
shifts=np.load("/sessions/gifted-cool-babbage/mnt/outputs/exp2b_gain_shifts.npy")
m=build(allsel); m.fit_time_warping_baseline(TGT,verbose=False,max_iter=MAXIT,penalty=PEN,fix_shifts=shifts)
X=m.predictors; y=m.response
widths=[e.shape[1] for e in m.effect_list]; ncf=int(np.sum(widths[:-1]))
v1s=int(np.sum(widths[:3])); v1e=v1s+widths[3]
v1_basis=m.basis_list[3]; nopen=m.no_penalty

def filt_ci(ks):
    rows=np.concatenate([np.arange(k*NT,(k+1)*NT) for k in ks])
    cols=np.concatenate([np.arange(ncf), ncf+np.asarray(ks)])
    res=GLM.poisson_regression(y[rows], X[np.ix_(rows,cols)], L2_pen=PEN, no_penalty=nopen)
    f = v1_basis @ res.params[v1s:v1e]
    ih = res.inv_hessian[v1s:v1e, v1s:v1e]
    ci = np.sqrt(np.diag(v1_basis @ ih @ v1_basis.T))
    return np.asarray(f), np.asarray(ci)

# global population gain = total firing of ALL neurons across 6 areas per trial
rate = sum(POOL_ALL[p][ds.npadding:,:].sum(0) for p in PROBES)
n_hi=int(ds.running_trial_index.sum())
order=np.argsort(rate)[::-1]; hi=order[:n_hi]; lo=order[n_hi:]
run=ds.running_trial_index
f_hi,ci_hi=filt_ci(hi); f_lo,ci_lo=filt_ci(lo)
out=dict(f_hi=f_hi, ci_hi=ci_hi, f_lo=f_lo, ci_lo=ci_lo,
         int_hi=float(f_hi.sum()), int_lo=float(f_lo.sum()),
         n_hi=len(hi), n_lo=len(lo),
         frac_run_hi=float(run[hi].mean()), frac_run_lo=float(run[lo].mean()))
pickle.dump(out, open("/sessions/gifted-cool-babbage/mnt/outputs/exp2b_gainALL_rate_ci.pkl","wb"))
print("int_hi=%.3f int_lo=%.3f  frac_run_hi=%.2f frac_run_lo=%.2f  ci_hi[0]=%.4f"
      %(out['int_hi'],out['int_lo'],out['frac_run_hi'],out['frac_run_lo'],ci_hi[0]))

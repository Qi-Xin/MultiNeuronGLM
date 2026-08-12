"""Refit the exp2c variants storing filter + one-sigma CI for the 6 coupling effects."""
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
AREA={'probeA':'AM','probeB':'PM','probeC':'V1','probeD':'LM','probeE':'AL','probeF':'RL'}
CP={'peaks_max':20.2,'num':3,'nonlinear':0.5}
NUM_BASE=20; PEN=5e-1; MAXIT=10; TAU=15; NFREF=4
OUT="/sessions/gifted-cool-babbage/mnt/outputs/exp2c_variants_ci.pkl"

ds=Dataset(); M,C=load_membership()
POOL_SEL={p:utils.pooling_pop(M,C,ds,p,0,use_all=False) for p in PROBES}
POOL_ALL={p:utils.pooling_pop(M,C,ds,p,0,use_all=True)  for p in PROBES}

def fit_one(model, cond, target):
    pool = POOL_ALL if model=="fullpop" else POOL_SEL
    sel  = ds.running_trial_index if cond=="running" else ds.stationary_trial_index
    m=GLM.PP_GLM(dataset=ds, select_trials=sel, membership=M, condition_ids=C)
    m.add_effect('inhomogeneous_baseline', num=NUM_BASE, apply_no_penalty=True)
    for p in PROBES: m.add_effect('coupling', pool[p], apply_no_penalty=True, **CP)
    if model!="linhist":
        m.add_effect('refractory_additive', pool[target], tau=TAU, num=NFREF, apply_no_penalty=True)
    m.add_effect('trial_coef')
    m.fit_time_warping_baseline(pool[target], verbose=False, max_iter=MAXIT, penalty=PEN)
    F=m.get_filter(ci=True)          # [ [base_f,base_ci], [coupA_f,coupA_ci], ... ]
    out={}
    for j in range(6):
        f, ci = F[1+j]
        out[PROBES[j]] = {"f": np.asarray(f).tolist(), "ci": np.asarray(ci).tolist()}
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--budget",type=float,default=38.0)
    a=ap.parse_args(); t0=time.time()
    R=pickle.load(open(OUT,"rb")) if os.path.exists(OUT) else {}
    todo=[(mo,co,tg) for mo in ["main","fullpop","linhist"]
                     for co in ["running","stationary"] for tg in PROBES
                     if (mo,co,tg) not in R]
    if not todo: print("ALL DONE (%d fits)"%len(R)); return
    for key in todo:
        if time.time()-t0 > a.budget: break
        t=time.time(); R[key]=fit_one(*key); pickle.dump(R,open(OUT,"wb"))
        print("fitted %-8s %-10s target=%-3s (%.0fs) [%d/36]"%(key[0],key[1],AREA[key[2]],time.time()-t,len(R)),flush=True)
    print("progress: %d/36"%len(R),flush=True)
if __name__=="__main__": main()

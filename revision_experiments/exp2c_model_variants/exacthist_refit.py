import os; os.environ["OMP_NUM_THREADS"]="2"
import sys,numpy as np,warnings,pickle,time,argparse; warnings.filterwarnings("ignore")
sys.path.insert(0,"/sessions/gifted-cool-babbage/mnt/MultiNeuronGLM")
sys.path.insert(0,"/sessions/gifted-cool-babbage/mnt/outputs")
os.chdir("/sessions/gifted-cool-babbage/mnt/MultiNeuronGLM")
import GLM, utility_functions as utils
from popglm_data import Dataset, load_membership
ds=Dataset(); M,C=load_membership()
acc=pickle.load(open('/sessions/gifted-cool-babbage/mnt/outputs/exacthist_accumulated.pkl','rb'))
PROBES=['probeA','probeB','probeC','probeD','probeE','probeF']
AREA={'probeA':'AM','probeB':'PM','probeC':'V1','probeD':'LM','probeE':'AL','probeF':'RL'}
CP={'peaks_max':20.2,'num':3,'nonlinear':0.5}; NUM_BASE=20; PEN=5e-1; MAXIT=10
POOL={p:utils.pooling_pop(M,C,ds,p,0,use_all=False) for p in PROBES}
OUT='/sessions/gifted-cool-babbage/mnt/outputs/exacthist_variants_ci.pkl'
def fit_one(cond,target):
    sel=ds.running_trial_index if cond=="running" else ds.stationary_trial_index
    m=GLM.PP_GLM(dataset=ds, select_trials=sel, membership=M, condition_ids=C)
    m.add_effect('inhomogeneous_baseline', num=NUM_BASE, apply_no_penalty=True)
    for p in PROBES: m.add_effect('coupling', POOL[p], apply_no_penalty=True, **CP)
    m.add_effect('trial_coef')
    m.fit_time_warping_baseline(POOL[target], verbose=False, max_iter=MAXIT, penalty=PEN, offset=acc[target])
    F=m.get_filter(ci=True)
    return {PROBES[j]: {"f":np.asarray(F[1+j][0]).tolist(),"ci":np.asarray(F[1+j][1]).tolist()} for j in range(6)}
ap=argparse.ArgumentParser(); ap.add_argument("--budget",type=float,default=33.0); a=ap.parse_args()
R=pickle.load(open(OUT,'rb')) if os.path.exists(OUT) else {}
todo=[(co,tg) for co in ["running","stationary"] for tg in PROBES if ("exacthist",co,tg) not in R]
t0=time.time()
for co,tg in todo:
    if time.time()-t0>a.budget: break
    t=time.time(); R[("exacthist",co,tg)]=fit_one(co,tg); pickle.dump(R,open(OUT,'wb'))
    print("fitted exacthist %-10s target=%-3s (%.0fs) [%d/12]"%(co,AREA[tg],time.time()-t,len(R)),flush=True)
print("progress %d/12"%len(R))

"""Exp 2b on REAL Allen data (session 757216464): is the V1->LM coupling estimate explained
by trial gain? Stratify running trials by empirical gain (total LM-pop spikes) and refit
V1->LM coupling within each half."""
import os
os.environ["OMP_NUM_THREADS"]="2"
import sys, time, pickle
sys.path.insert(0,"/sessions/gifted-cool-babbage/mnt/MultiNeuronGLM")
os.chdir("/sessions/gifted-cool-babbage/mnt/MultiNeuronGLM")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from scipy.stats import chi2
import GLM

D=np.load("/sessions/gifted-cool-babbage/mnt/outputs/allen_757216464_VISp_VISl.npz", allow_pickle=True)
run=D["running"]
v1=D["v1_all"].astype(float); lm=D["lm_all"].astype(float)   # (500, 630)
NUM_POP,NUM_CP,PEAKS_MAX,TAU,NUM_F_REF=50,3,15,10,4
PEN=1e-5; WARP=[[0,0.15],[0.15,0.35]]

def build(nt,ntr,src,tgt,with_src):
    m=GLM.PP_GLM(ntrial=ntr,nt=nt,select_trials=np.array([True]*ntr))
    m.add_effect("inhomogeneous_baseline",num=NUM_POP,apply_no_penalty=True)
    if with_src: m.add_effect("coupling",raw_input=src,num=NUM_CP,peaks_max=PEAKS_MAX,nonlinear=1)
    m.add_effect("coupling",raw_input=tgt,num=NUM_CP,peaks_max=PEAKS_MAX,nonlinear=1)
    m.add_effect("refractory_additive",raw_input=tgt,tau=TAU,num=NUM_F_REF,apply_no_penalty=False)
    return m
def fit(m,tgt):
    m.fit_time_warping_baseline(target=tgt,max_iter=5,warp_interval=WARP,method="mine",penalty=PEN,verbose=False); return m
def coupling(src,tgt):
    nt,ntr=src.shape
    full=fit(build(nt,ntr,src,tgt,True),tgt); nest=fit(build(nt,ntr,src,tgt,False),tgt)
    f=np.asarray(full.get_filter()[1]); df=full.predictors.shape[1]-nest.predictors.shape[1]
    p=float(chi2.sf(2*(nest.nll-full.nll),df)) if df>0 else float("nan")
    return float(np.sum(f)), float(np.max(np.abs(f))), p, f.tolist()

src=v1[:,run]; tgt=lm[:,run]                       # running trials only
gain=tgt.sum(axis=0); med=np.median(gain)
hi=gain>=med; lo=gain<med
t0=time.time()
res={}
for name,msk in [("all_running",np.ones(run.sum(),bool)),("hi_gain",hi),("lo_gain",lo)]:
    ti=time.time(); i,pk,p,filt=coupling(src[:,msk],tgt[:,msk])
    res[name]={"integral":i,"peak":pk,"lrt_p":p,"ntrial":int(msk.sum()),"filter":filt,"secs":time.time()-ti}
    print("%-12s n=%3d  V1->LM integral=%.3f peak=%.3f p=%.3g (%.1fs)"%(name,msk.sum(),i,pk,p,time.time()-ti),flush=True)
res["gain_hi_mean"]=float(gain[hi].mean()); res["gain_lo_mean"]=float(gain[lo].mean())
pickle.dump(res,open("/sessions/gifted-cool-babbage/mnt/outputs/exp2b_realdata_results.pkl","wb"))
print("gain hi=%.1f lo=%.1f (%.2fx)  total %.1fs"%(gain[hi].mean(),gain[lo].mean(),gain[hi].mean()/gain[lo].mean(),time.time()-t0),flush=True)

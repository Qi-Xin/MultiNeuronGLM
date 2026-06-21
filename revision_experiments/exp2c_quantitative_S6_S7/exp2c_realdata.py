"""Exp 2c on REAL Allen data (session 757216464): quantitative comparison of the V1->LM
coupling filter under (S6) selected vs full population, and (S7) f_damp vs no-damp self-history.
Fit on stationary trials (the main reference condition)."""
import os; os.environ["OMP_NUM_THREADS"]="2"
import sys,time,pickle
sys.path.insert(0,"/sessions/gifted-cool-babbage/mnt/MultiNeuronGLM")
sys.path.insert(0,"/sessions/gifted-cool-babbage/mnt/MultiNeuronGLM/revision_experiments/exp2c_quantitative_S6_S7")
os.chdir("/sessions/gifted-cool-babbage/mnt/MultiNeuronGLM")
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import GLM
from exp2c_quantify import compare_pair

D=np.load("/sessions/gifted-cool-babbage/mnt/outputs/allen_757216464_VISp_VISl.npz",allow_pickle=True)
run=D["running"]; stat=~run
NUM_POP,NUM_CP,PEAKS_MAX,TAU,NUM_F_REF=50,3,15,10,4
PEN=1e-5; WARP=[[0,0.15],[0.15,0.35]]
which=sys.argv[1] if len(sys.argv)>1 else "S6"

def fit_filter(src,tgt,damp=True):
    nt,ntr=src.shape
    m=GLM.PP_GLM(ntrial=ntr,nt=nt,select_trials=np.array([True]*ntr))
    m.add_effect("inhomogeneous_baseline",num=NUM_POP,apply_no_penalty=True)
    m.add_effect("coupling",raw_input=src,num=NUM_CP,peaks_max=PEAKS_MAX,nonlinear=1)
    m.add_effect("coupling",raw_input=tgt,num=NUM_CP,peaks_max=PEAKS_MAX,nonlinear=1)
    if damp: m.add_effect("refractory_additive",raw_input=tgt,tau=TAU,num=NUM_F_REF,apply_no_penalty=False)
    m.fit_time_warping_baseline(target=tgt,max_iter=5,warp_interval=WARP,method="mine",penalty=PEN,verbose=False)
    return np.asarray(m.get_filter()[1])

t0=time.time()
if which=="S6":   # selected vs full population (stationary trials)
    fs=fit_filter(D["v1_sel"].astype(float)[:,stat], D["lm_sel"].astype(float)[:,stat])
    ff=fit_filter(D["v1_all"].astype(float)[:,stat], D["lm_all"].astype(float)[:,stat])
    c=compare_pair(fs,ff,names=("selected","full"))
    print("S6 selected vs full population, V1->LM (stationary):")
    print("  selected: integral=%.3f peak=%.3f latency=%dms"%(c["selected"]["integral"],c["selected"]["peak_amp"],c["selected"]["peak_latency"]))
    print("  full    : integral=%.3f peak=%.3f latency=%dms"%(c["full"]["integral"],c["full"]["peak_amp"],c["full"]["peak_latency"]))
    print("  correlation=%.3f  d_integral=%.1f%%  d_peak=%.1f%%  d_latency=%dms"%(c["correlation"],c["pct_diff_integral"],c["pct_diff_peak"],c["diff_peak_latency_ms"]))
    pickle.dump({"selected_filter":fs.tolist(),"full_filter":ff.tolist(),"compare":c},open("/sessions/gifted-cool-babbage/mnt/outputs/exp2c_S6_realdata.pkl","wb"))
else:             # S7: f_damp vs no-damp self-history (full population, stationary)
    src=D["v1_all"].astype(float)[:,stat]; tgt=D["lm_all"].astype(float)[:,stat]
    fd=fit_filter(src,tgt,damp=True); fn=fit_filter(src,tgt,damp=False)
    c=compare_pair(fd,fn,names=("f_damp","no_damp"))
    print("S7 f_damp vs linear-only self-history, V1->LM (stationary):")
    print("  f_damp : integral=%.3f peak=%.3f latency=%dms"%(c["f_damp"]["integral"],c["f_damp"]["peak_amp"],c["f_damp"]["peak_latency"]))
    print("  no_damp: integral=%.3f peak=%.3f latency=%dms"%(c["no_damp"]["integral"],c["no_damp"]["peak_amp"],c["no_damp"]["peak_latency"]))
    print("  correlation=%.3f  d_integral=%.1f%%  d_peak=%.1f%%  d_latency=%dms"%(c["correlation"],c["pct_diff_integral"],c["pct_diff_peak"],c["diff_peak_latency_ms"]))
    pickle.dump({"fdamp_filter":fd.tolist(),"nodamp_filter":fn.tolist(),"compare":c},open("/sessions/gifted-cool-babbage/mnt/outputs/exp2c_S7_realdata.pkl","wb"))
print("done %.1fs"%(time.time()-t0))

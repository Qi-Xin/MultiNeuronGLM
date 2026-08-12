import os; os.environ["OMP_NUM_THREADS"]="2"
import sys, numpy as np
sys.path.insert(0,"/sessions/gifted-cool-babbage/mnt/MultiNeuronGLM"); sys.path.insert(0,"/sessions/gifted-cool-babbage/mnt/outputs")
os.chdir("/sessions/gifted-cool-babbage/mnt/MultiNeuronGLM")
import warnings; warnings.filterwarnings("ignore")
import GLM
from exp2b_gain import build, TGT, MAXIT, PEN, ds
shifts=np.load("/sessions/gifted-cool-babbage/mnt/outputs/exp2b_gain_shifts.npy")
m=build(np.full(ds.ntrial,True))
m.fit_time_warping_baseline(TGT, verbose=False, max_iter=MAXIT, penalty=PEN, fix_shifts=shifts)
w=[e.shape[1] for e in m.effect_list]; ncf=int(np.sum(w[:-1]))
v1s=int(np.sum(w[:3])); v1e=v1s+w[3]
np.savez("/sessions/gifted-cool-babbage/mnt/outputs/exp2b_design.npz",
         X=m.predictors.astype(np.float32), y=m.response, ncf=ncf, v1s=v1s, v1e=v1e,
         v1_basis=m.basis_list[3], nopen=np.array(m.no_penalty,dtype=object))
print("cached design:", m.predictors.shape, "| V1 cols", v1s, v1e)

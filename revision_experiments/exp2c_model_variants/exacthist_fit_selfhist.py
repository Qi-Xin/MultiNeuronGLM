import os; os.environ["OMP_NUM_THREADS"]="2"
import sys,numpy as np,warnings,pickle,time; warnings.filterwarnings("ignore")
sys.path.insert(0,"/sessions/gifted-cool-babbage/mnt/MultiNeuronGLM")
sys.path.insert(0,"/sessions/gifted-cool-babbage/mnt/outputs")
os.chdir("/sessions/gifted-cool-babbage/mnt/MultiNeuronGLM")
import GLM
from popglm_data import Dataset, load_membership
ds=Dataset(); M,C=load_membership()
pre=np.load('/sessions/gifted-cool-babbage/mnt/outputs/exacthist_pre_trains.npz')
info=pickle.load(open('/sessions/gifted-cool-babbage/mnt/outputs/exacthist_crosspop.pkl','rb'))
alln=info['alln']
CP={'peaks_max':20.2,'num':3,'nonlinear':0.5}
OUT='/sessions/gifted-cool-babbage/mnt/outputs/exacthist_selfhist_filters.pkl'
D=pickle.load(open(OUT,'rb')) if os.path.exists(OUT) else {}
t0=time.time()
def fit(arr, sel):
    m=GLM.PP_GLM(dataset=ds, select_trials=sel, membership=M, condition_ids=C)
    m.add_effect('inhomogeneous_baseline', num=5)
    m.add_effect('coupling', arr, **CP)
    m.fit(arr, method='mine', verbose=False, penalty=1e2)
    return np.asarray(m.get_filter()[1])
for n in alln:
    if (n,'stationary') in D and (n,'running') in D: continue
    if time.time()-t0>36: break
    arr=pre[str(n)]
    D[(n,'stationary')]=fit(arr, ds.stationary_trial_index)
    D[(n,'running')]=fit(arr, ds.running_trial_index)
    pickle.dump(D,open(OUT,'wb'))
done=len(set(k[0] for k in D))
print("neurons done: %d / %d  (%.0fs)"%(done,len(alln),time.time()-t0))

import os; os.environ["OMP_NUM_THREADS"]="2"
import sys,numpy as np,warnings,pickle,time; warnings.filterwarnings("ignore")
sys.path.insert(0,"/sessions/gifted-cool-babbage/mnt/MultiNeuronGLM")
sys.path.insert(0,"/sessions/gifted-cool-babbage/mnt/outputs")
os.chdir("/sessions/gifted-cool-babbage/mnt/MultiNeuronGLM")
import GLM
from popglm_data import Dataset, load_membership
ds=Dataset(); M,C=load_membership()
D=pickle.load(open('/sessions/gifted-cool-babbage/mnt/outputs/exacthist_selfhist_filters.pkl','rb'))
cross=pickle.load(open('/sessions/gifted-cool-babbage/mnt/outputs/exacthist_crosspop.pkl','rb'))['cross']
PROBES=['probeA','probeB','probeC','probeD','probeE','probeF']
id2col={int(u):i for i,u in enumerate(ds.unit_ids)}
cond=ds.presentation_table['stimulus_condition_id'].values
NT,NTR=ds.nt,ds.ntrial
acc_hist={}; empties=0
t0=time.time()
for tp in PROBES:
    expout={}
    for n in cross[tp]:
        spk=ds.spike_train[:,id2col[int(n)],:]
        for state in ['stationary','running']:
            o=np.asarray(GLM.conv(spk, D[(n,state)], npadding=ds.npadding)).reshape(NT,NTR,order='F')
            expout[(n,state)]=np.exp(o)
    acc=np.zeros((NT,NTR))
    for it in range(NTR):
        state='stationary' if ds.stationary_trial_index[it] else 'running'
        mem=M[int(np.where(C==cond[it])[0][0])]
        idx=[int(n) for n in mem[(mem['probe']==tp)&(mem['group_id']==0)].index.values if (int(n),state) in expout]
        if len(idx)==0:
            empties+=1
            # fallback: all cross-pop neurons of this probe
            idx=[int(n) for n in cross[tp]]
        s=np.zeros(NT)
        for n in idx: s+=expout[(n,state)][:,it]
        acc[:,it]=np.log(s)-np.sqrt(len(idx))
    acc_hist[tp]=acc
    print("  %s done: acc mean %.3f range[%.2f,%.2f]"%(tp,acc.mean(),acc.min(),acc.max()),flush=True)
pickle.dump(acc_hist, open('/sessions/gifted-cool-babbage/mnt/outputs/exacthist_accumulated.pkl','wb'))
print("empties(fallback trials):",empties,"| %.0fs"%(time.time()-t0))

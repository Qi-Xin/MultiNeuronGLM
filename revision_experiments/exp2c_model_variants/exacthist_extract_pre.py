import sys, csv, pickle, numpy as np, h5py
sys.path.insert(0,"/sessions/gifted-cool-babbage/mnt/MultiNeuronGLM"); sys.path.insert(0,".")
import utility_functions as utils
C="/sessions/gifted-cool-babbage/mnt/ecephys_cache_dir"; SID=757216464
NWB=f"{C}/session_{SID}/session_{SID}.nwb"
START,END,PAD,FPS=-0.5,0.0,0.3,1000
NT=int(round((END-START)*FPS)); NPAD=int(round(PAD*FPS)); NTOT=NT+NPAD   # 500+300=800
alln=pickle.load(open('exacthist_crosspop.pkl','rb'))['alln']
d=np.load('popglm_dataset_757216464.npz', allow_pickle=True); t0=d['t0']
ntrial=len(t0); print("ntrial",ntrial,"neurons",len(alln))
f=h5py.File(NWB,"r"); u=f["units"]
uid=u["id"][:]
id2row={int(x):i for i,x in enumerate(uid)}
st=f["units/spike_times"]; sti=f["units/spike_times_index"][:]
# extraction window relative to t0: [START-PAD, END] = [-0.8, 0.0]
edges=np.arange(START-PAD, END+1e-9, 1.0/FPS)   # 801 edges -> 800 bins
assert len(edges)-1==NTOT, (len(edges)-1, NTOT)
pre={}
for n in alln:
    i=id2row[int(n)]
    a=0 if i==0 else int(sti[i-1]); b=int(sti[i]); sp=st[a:b]
    arr=np.zeros((NTOT,ntrial),dtype=np.float32)
    for m,tt in enumerate(t0):
        seg=sp[np.searchsorted(sp,tt+START-PAD):np.searchsorted(sp,tt+END)]-tt
        if seg.size:
            h,_=np.histogram(seg,bins=edges); arr[:,m]=h
    pre[int(n)]=arr
f.close()
np.savez_compressed('exacthist_pre_trains.npz', **{str(k):v for k,v in pre.items()})
tot=np.mean([v[NPAD:,:].mean() for v in pre.values()])
print("saved exacthist_pre_trains.npz ; mean pre-stim rate/bin %.4f"%tot)

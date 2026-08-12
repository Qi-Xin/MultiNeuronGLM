"""Reproduce Allen_dataset (session 757216464, 14 conditions, 6 probes, start=0 end=0.5 pad=0.3)
directly from the NWB with h5py, so the real GLM.PP_GLM can be driven without allensdk's slow loader."""
import sys, csv, pickle, numpy as np, h5py
sys.path.insert(0,"/sessions/gifted-cool-babbage/mnt/MultiNeuronGLM")
import utility_functions as utils
C="/sessions/gifted-cool-babbage/mnt/ecephys_cache_dir"; SID=757216464
NWB=f"{C}/session_{SID}/session_{SID}.nwb"
G="/sessions/gifted-cool-babbage/mnt/Ghigorah/MultiNeuronGLM"
COND14=[275,277,246,255,272,248,283,266,274,276,286,271,268,270]
START,END,PAD,FPS=0.0,0.50,0.3,1000
NT=int(round((END-START)*FPS)); NPAD=int(round(PAD*FPS)); NTOT=NT+NPAD   # 500 + 300 = 800
PROBES=['probeA','probeB','probeC','probeD','probeE','probeF']

# probe id -> name ; channel -> (probe, area)
pid2name={r['id']:r['name'] for r in csv.DictReader(open(C+"/probes.csv")) if r['ecephys_session_id']==str(SID)}
chan={}
for r in csv.DictReader(open(C+"/channels.csv")):
    if r['ecephys_probe_id'] in pid2name:
        chan[int(r['id'])]=(pid2name[r['ecephys_probe_id']], r['ecephys_structure_acronym'])

f=h5py.File(NWB,"r"); u=f["units"]
uid=u["id"][:]; pk=u["peak_channel_id"][:]
qual=np.array([q.decode() for q in u["quality"][:]])
isi,amp,pres=u["isi_violations"][:],u["amplitude_cutoff"][:],u["presence_ratio"][:]
probe=np.array([chan.get(int(c),("",""))[0] for c in pk])
area =np.array([chan.get(int(c),("",""))[1] for c in pk])
good=(qual=="good")&(isi<=0.5)&(amp<=0.1)&(pres>=0.9)&np.isin(area,utils.VISUAL_AREA)
# DataLoader order: concatenate per probe A..F
order=[i for p in PROBES for i in np.where(good&(probe==p))[0]]
unit_ids=uid[order]; unit_probe=probe[order]
print("units per probe:", {p:int((unit_probe==p).sum()) for p in PROBES}, "| total", len(unit_ids))

# membership check (group_id_all_a_c)
memb=pickle.load(open(G+"/group_id_all_a_c/membership.pickle","rb"))
cids=pickle.load(open(G+"/group_id_all_a_c/condition_ids.pickle","rb"))
mem_units=set().union(*[set(m.index.values) for m in memb])
missing=mem_units-set(unit_ids.tolist())
print("membership units:",len(mem_units),"| missing from dataset:",len(missing))

# trials: the 14 conditions
sel=pickle.load(open("/sessions/gifted-cool-babbage/mnt/outputs/selected14_conditions_757216464.pkl","rb"))
sel=sel.sort_values("start_time").reset_index(drop=True)
t0=sel["start_time"].values; cond=sel["stimulus_condition_id"].values
ntrial=len(t0); print("trials:",ntrial)

# spike_train (NTOT, nneuron, ntrial) over [t0-PAD, t0+END]
st=f["units/spike_times"]; sti=f["units/spike_times_index"][:]
spike_train=np.zeros((NTOT,len(unit_ids),ntrial),dtype=np.float32)
edges=np.arange(-PAD, END+1e-9, 1.0/FPS)          # 801 edges -> 800 bins
for k,i in enumerate(order):
    a=0 if i==0 else int(sti[i-1]); b=int(sti[i]); sp=st[a:b]
    for m,tt in enumerate(t0):
        seg=sp[np.searchsorted(sp,tt-PAD):np.searchsorted(sp,tt+END)]-tt
        if seg.size:
            h,_=np.histogram(seg,bins=edges); spike_train[:,k,m]=h
print("spike_train:",spike_train.shape,"mean/bin %.4f"%spike_train.mean())

# running: mean speed over [t0, t0+END] >= 1  (get_running(method='mine'))
rs=f["processing/running/running_speed/data"][:]; rt=f["processing/running/running_speed/timestamps"][:]
mean_speed=np.array([rs[np.searchsorted(rt,tt+START):np.searchsorted(rt,tt+END)].mean() for tt in t0])
running=mean_speed>=1.0; stationary=mean_speed<1.0
print("running:",int(running.sum()),"stationary:",int(stationary.sum()),"(paper: 81 / 129)")
f.close()
np.savez("/sessions/gifted-cool-babbage/mnt/outputs/popglm_dataset_757216464.npz",
         spike_train=spike_train, unit_ids=unit_ids, unit_probe=unit_probe,
         cond=cond, t0=t0, mean_speed=mean_speed, running=running, stationary=stationary,
         nt=NT, npadding=NPAD, fps=FPS)
print("saved popglm_dataset_757216464.npz")

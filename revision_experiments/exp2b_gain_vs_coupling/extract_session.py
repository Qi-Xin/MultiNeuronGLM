"""Extract pooled V1 (VISp) and LM (VISl) population spike trains for drifting-grating trials
directly from the Allen session NWB via h5py (bypasses the slow allensdk session loader).
Caches everything to an npz for fast downstream fitting."""
import sys, os, csv, time
import numpy as np, h5py
C="/sessions/gifted-cool-babbage/mnt/ecephys_cache_dir"
SID=int(sys.argv[1]) if len(sys.argv)>1 else 757216464
NWB="%s/session_%d/session_%d.nwb"%(C,SID,SID)
OUT="/sessions/gifted-cool-babbage/mnt/outputs/allen_%d_VISp_VISl.npz"%SID
DUR=0.5; DT=0.001; NB=int(round(DUR/DT))   # 500 ms, 1 ms bins
t0=time.time()

# 1. channel -> area
chan2area={}
with open(C+"/channels.csv") as fh:
    r=csv.DictReader(fh)
    for row in r: chan2area[int(row["id"])]=row["ecephys_structure_acronym"]

f=h5py.File(NWB,"r")
u=f["units"]
uid=u["id"][:]; pkch=u["peak_channel_id"][:]
qual=np.array([q.decode() if isinstance(q,bytes) else q for q in u["quality"][:]])
isi=u["isi_violations"][:]; amp=u["amplitude_cutoff"][:]; pres=u["presence_ratio"][:]
area=np.array([chan2area.get(int(c),"") for c in pkch])
# allensdk default quality filter
good=(qual=="good")&(isi<=0.5)&(amp<=0.1)&(pres>=0.9)
sel_v1=good&(area=="VISp"); sel_lm=good&(area=="VISl")
print("good VISp units: %d | good VISl units: %d"%(sel_v1.sum(), sel_lm.sum()), flush=True)

# spike_times per unit via index
st=f["units/spike_times"]; sti=f["units/spike_times_index"][:]
def unit_spikes(i):
    a=0 if i==0 else int(sti[i-1]); b=int(sti[i])
    return st[a:b]

# 2. drifting gratings trials
dg=f["intervals/drifting_gratings_presentations"]
start=dg["start_time"][:]; ori=dg["orientation"][:]
tf=dg["temporal_frequency"][:]
# valid trials: orientation is a real number (exclude 'null'/nan blanks)
def isnum(x):
    try: float(x); return True
    except: return False
ori_str=np.array([o.decode() if isinstance(o,bytes) else str(o) for o in ori])
valid=np.array([isnum(o) for o in ori_str])
start_v=start[valid]
cond=np.array([ "%s_%s"%(ori_str[k], (tf[k].decode() if isinstance(tf[k],bytes) else tf[k])) for k in range(len(start)) ])[valid]
ntrial=len(start_v)
print("valid drifting-grating trials: %d"%ntrial, flush=True)

# 3. running speed per trial (mean over the 0-0.5s window)
rs=f["processing/running/running_speed/data"][:]
rt=f["processing/running/running_speed/timestamps"][:]
def trial_speed(t0_):
    i0=np.searchsorted(rt,t0_); i1=np.searchsorted(rt,t0_+DUR)
    return rs[i0:i1].mean() if i1>i0 else 0.0
speed=np.array([trial_speed(t) for t in start_v])
running=speed>=1.0   # paper threshold 1 cm/s
print("running trials: %d | stationary: %d"%(running.sum(), (~running).sum()), flush=True)

# 4. build pooled spike trains (NB, ntrial) for a set of unit indices
def pooled(unit_idx):
    out=np.zeros((NB,ntrial),dtype=np.float32)
    for i in unit_idx:
        sp=unit_spikes(i)
        for m,t in enumerate(start_v):
            seg=sp[np.searchsorted(sp,t):np.searchsorted(sp,t+DUR)]-t
            if seg.size:
                b=np.clip((seg/DT).astype(int),0,NB-1)
                np.add.at(out[:,m],b,1.0)
    return out

v1_all=pooled(np.where(sel_v1)[0])
lm_all=pooled(np.where(sel_lm)[0])
print("built pooled trains in %.1fs"%(time.time()-t0), flush=True)

np.savez(OUT, v1_all=v1_all, lm_all=lm_all, running=running, speed=speed,
         cond=cond, start=start_v, n_v1=int(sel_v1.sum()), n_lm=int(sel_lm.sum()),
         uid_v1=uid[sel_v1], uid_lm=uid[sel_lm])
f.close()
print("saved", OUT, "shapes v1_all", v1_all.shape, flush=True)

# ---- appended: also build the strongly-evoked "selected" subpopulation (top 20% by evoked amplitude) ----
def per_unit_evoked(unit_idx):
    """trial-averaged peak evoked rate per unit, for ranking."""
    strength={}
    for i in unit_idx:
        sp=unit_spikes(i); psth=np.zeros(NB)
        for t in start_v:
            seg=sp[np.searchsorted(sp,t):np.searchsorted(sp,t+DUR)]-t
            if seg.size:
                b=np.clip((seg/DT).astype(int),0,NB-1); np.add.at(psth,b,1.0)
        psth/=ntrial
        # evoked amplitude = peak of smoothed psth minus its pre-onset-ish floor
        k=np.ones(20)/20.0; sm=np.convolve(psth,k,"same")
        strength[i]=float(sm.max()-np.median(sm))
    return strength

def selected_pooled(sel_mask):
    idx=np.where(sel_mask)[0]
    strength=per_unit_evoked(idx)
    order=sorted(idx,key=lambda i:strength[i],reverse=True)
    ktop=max(1,int(np.ceil(0.20*len(idx))))
    top=order[:ktop]
    out=np.zeros((NB,ntrial),dtype=np.float32)
    for i in top:
        sp=unit_spikes(i)
        for m,t in enumerate(start_v):
            seg=sp[np.searchsorted(sp,t):np.searchsorted(sp,t+DUR)]-t
            if seg.size:
                b=np.clip((seg/DT).astype(int),0,NB-1); np.add.at(out[:,m],b,1.0)
    return out, ktop

f2=h5py.File(NWB,"r"); u=f2["units"]; st=f2["units/spike_times"]; sti=f2["units/spike_times_index"][:]
def unit_spikes(i):
    a=0 if i==0 else int(sti[i-1]); b=int(sti[i]); return st[a:b]
v1_sel,k1=selected_pooled(sel_v1); lm_sel,k2=selected_pooled(sel_lm)
print("selected (top 20%%): V1 %d units, LM %d units"%(k1,k2),flush=True)
d=dict(np.load(OUT,allow_pickle=True))
d.update(v1_sel=v1_sel, lm_sel=lm_sel, n_v1_sel=k1, n_lm_sel=k2)
np.savez(OUT, **d)
f2.close()
print("updated npz with selected population", flush=True)

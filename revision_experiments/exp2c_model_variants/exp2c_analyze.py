"""Exp 2c analysis: compare model variants to the main model across all coupling filters.
amplitude = filter integral ; delay = peak latency (argmax |filter|, ms)."""
import pickle, numpy as np
PROBES=['probeA','probeB','probeC','probeD','probeE','probeF']
AREA={'probeA':'AM','probeB':'PM','probeC':'V1','probeD':'LM','probeE':'AL','probeF':'RL'}
R=pickle.load(open("/sessions/gifted-cool-babbage/mnt/outputs/exp2c_variants.pkl","rb"))

def amp(f):  f=np.asarray(f); return float(f.sum())
def delay(f):f=np.asarray(f); return float(np.argmax(np.abs(f)))

rows={}
for mo in ["main","fullpop","linhist"]:
    A,D,keys=[],[],[]
    for co in ["running","stationary"]:
        for tg in PROBES:
            for src in PROBES:
                if src==tg: continue            # cross-area pairs only (30 per condition)
                f=R[(mo,co,tg)][src]
                A.append(amp(f)); D.append(delay(f)); keys.append((co,AREA[tg],AREA[src]))
    rows[mo]=(np.array(A),np.array(D),keys)

print("="*76)
print("EXP 2c — variant robustness of the coupling filters (exact 6-area pop-GLM)")
print("60 cross-area coupling filters (30 pairs x running/stationary)")
print("="*76)
Am,Dm,keys=rows["main"]
for mo,label in [("fullpop","FULL neurons (S6)"),("linhist","LINEAR self-history (S7)")]:
    Av,Dv,_=rows[mo]
    ra=np.corrcoef(Am,Av)[0,1]; rd=np.corrcoef(Dm,Dv)[0,1]
    slope=np.polyfit(Am,Av,1)[0]
    print("\n%s vs MAIN:"%label)
    print("   amplitude correlation (filter integral) : r = %.3f"%ra)
    print("   delay correlation (peak latency, ms)    : r = %.3f"%rd)
    print("   amplitude scale (variant = slope x main): %.2f"%slope)
    print("   mean |delay| difference                 : %.1f ms"%np.mean(np.abs(Dm-Dv)))

# the headline V1->LM filter
print("\nV1->LM coupling-filter integral:")
print("  %-22s %10s %10s"%("model","running","stationary"))
for mo,label in [("main","main"),("fullpop","full neurons"),("linhist","linear self-hist")]:
    r=amp(R[(mo,"running","probeD")]["probeC"]); s=amp(R[(mo,"stationary","probeD")]["probeC"])
    print("  %-22s %10.3f %10.3f"%(label,r,s))
np.savez("/sessions/gifted-cool-babbage/mnt/outputs/exp2c_summary.npz",
         Am=Am,Dm=Dm,Af=rows["fullpop"][0],Df=rows["fullpop"][1],
         Al=rows["linhist"][0],Dl=rows["linhist"][1])

import pickle, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
O="/sessions/gifted-cool-babbage/mnt/outputs/"
R=pickle.load(open(O+"exp2a_s3.pkl","rb"))
BOTH=R["both"]; REF=R["ref"]
W=np.array([r["pop_warp"]   for r in BOTH])      # pop-GLM, time warping ON
N=np.array([r["pop_nowarp"] for r in BOTH])      # pop-GLM, NO time warping
S=np.array([f for r in BOTH for f in r["single"]])  # single-neuron GLM filters
RF=np.array([r["pop_warp"] for r in REF])        # reference: coupling only, no variation
L=40; lag=np.arange(L)
def cut(A): return A[:, :L]
W,N,S,RF = cut(W),cut(N),cut(S),cut(RF)
pw=[r["p_warp"] for r in BOTH]; pn=[r["p_nowarp"] for r in BOTH]

print("="*74)
print("Scenario 3 — coupling AND trial-to-trial baseline variation both present")
print("  (EIF, conn=0.0075, std1=10 corr1=0.5 std2=25 corr2=0.9, ntrial=100, %d sims)"%len(BOTH))
print("="*74)
print("coupling-filter integral (mean +/- sd):")
print("  reference: coupling, NO variation (paper's Sensitivity)  : %.3f +/- %.3f"%(RF.sum(1).mean(),RF.sum(1).std(ddof=1)))
print("  pop-GLM, time warping ON                                 : %.3f +/- %.3f"%(W.sum(1).mean(),W.sum(1).std(ddof=1)))
print("  pop-GLM, NO time warping                                 : %.3f +/- %.3f"%(N.sum(1).mean(),N.sum(1).std(ddof=1)))
print("  single-neuron GLM (n=%d filters)                        : %.3f +/- %.3f"%(len(S),S.sum(1).mean(),S.sum(1).std(ddof=1)))
print("\npop-GLM (warping) detects nonzero coupling in %d/%d sims (LRT p<0.05)"%(sum(p<.05 for p in pw),len(pw)))
print("no-warp / warp integral ratio = %.2f  (no-warp is inflated)"%(N.sum(1).mean()/W.sum(1).mean()))
print("single-neuron mean vs pop-GLM(warp) mean: %.3f vs %.3f"%(S.sum(1).mean(),W.sum(1).mean()))

plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False})
fig,ax=plt.subplots(1,3,figsize=(11,3.4),sharey=True)
# A: pop-GLM with warping
for f in W: ax[0].plot(lag,f,color="grey",lw=.6)
ax[0].plot(lag,W.mean(0),color="k",lw=2,label="mean")
ax[0].plot(lag,RF.mean(0),color="#2ca02c",lw=1.6,ls="--",label="no-variation reference")
ax[0].set_title("pop-GLM (time warping)\nnonzero coupling recovered",fontsize=10,fontweight="bold")
ax[0].legend(fontsize=7,frameon=False)
# B: pop-GLM without warping
for f in N: ax[1].plot(lag,f,color="grey",lw=.6)
ax[1].plot(lag,N.mean(0),color="#d62728",lw=2,label="mean (no warping)")
ax[1].plot(lag,W.mean(0),color="k",lw=1.6,ls="--",label="pop-GLM (warping)")
ax[1].set_title("pop-GLM without time warping\ninflated coupling",fontsize=10,fontweight="bold")
ax[1].legend(fontsize=7,frameon=False)
# C: single-neuron
idx=np.random.RandomState(0).choice(len(S), size=min(200,len(S)), replace=False)
for f in S[idx]: ax[2].plot(lag,f,color="grey",lw=.3,alpha=.5)
ax[2].plot(lag,S.mean(0),color="#1f77b4",lw=2,label="mean (single-neuron)")
ax[2].plot(lag,W.mean(0),color="k",lw=1.6,ls="--",label="pop-GLM (warping)")
ax[2].plot(lag,N.mean(0),color="#d62728",lw=1.6,ls=":",label="pop-GLM (no warping)")
ax[2].set_title("single-neuron GLM\nnoisy AND inflated (no warping possible)",fontsize=10,fontweight="bold")
ax[2].legend(fontsize=7,frameon=False)
for a in ax:
    a.axhline(0,color="k",lw=.5); a.set_xlabel("lag (ms)"); a.set_xlim(0,L-1)
ax[0].set_ylabel("coupling filter (log rate)")
ax[0].set_ylim(-0.06,0.12)
fig.suptitle("Scenario 3: coupling and trial-to-trial baseline variation both present",fontsize=11,y=1.03)
fig.tight_layout()
for e in ["pdf","png"]: fig.savefig(O+"FigS_exp2a_scenario3.%s"%e,dpi=160,bbox_inches="tight")
print("\nsaved FigS_exp2a_scenario3")

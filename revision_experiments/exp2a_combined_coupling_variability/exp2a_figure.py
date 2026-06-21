import pickle, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, wilcoxon

R = pickle.load(open("/sessions/gifted-cool-babbage/mnt/outputs/exp2a_results.pkl","rb"))
g = lambda c: [r for r in R if r["cond"]==c]
comb, ref, noc = g("combined"), g("ref"), g("nocoup")
I  = lambda recs,k: np.array([r[k]["integral"] for r in recs])
P  = lambda recs,k: np.array([r[k]["lrt_p"]   for r in recs])
F  = lambda recs,k: np.array([r[k]["filter"]  for r in recs])

ref_i=I(ref,"plain"); cw_i=I(comb,"warp"); cn_i=I(comb,"nowarp"); nw_i=I(noc,"warp"); nn_i=I(noc,"nowarp")
ref_p=P(ref,"plain"); cw_p=P(comb,"warp"); cn_p=P(comb,"nowarp"); nw_p=P(noc,"warp"); nn_p=P(noc,"nowarp")

# paired within-data contrast (same combined sims): warp vs no-warp
W_pair = wilcoxon(cw_i, cn_i).pvalue
W_noc  = wilcoxon(nw_i, nn_i).pvalue
print("Paired Wilcoxon combined warp vs nowarp: p=%.2e"%W_pair)
print("Paired Wilcoxon nocoup   warp vs nowarp: p=%.2e"%W_noc)

# ---------------- FIGURE ----------------
plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False})
fig = plt.figure(figsize=(13,4.2))
C_REF="#444444"; C_WARP="#1f77b4"; C_NOWARP="#d62728"

# Panel A: recovered coupling integral
axA = fig.add_subplot(1,3,1)
groups = [("Reference\n(no var.)",ref_i,C_REF),
          ("Warp-ON",cw_i,C_WARP),("Warp-OFF",cn_i,C_NOWARP),
          ("Warp-ON",nw_i,C_WARP),("Warp-OFF",nn_i,C_NOWARP)]
xs=[0,1.6,2.7,4.5,5.6]
for x,(lab,dat,col) in zip(xs,groups):
    axA.scatter(np.random.normal(x,0.05,len(dat)),dat,s=22,color=col,alpha=.6,zorder=3,edgecolor="none")
    axA.plot([x-0.22,x+0.22],[dat.mean()]*2,color=col,lw=3,zorder=4)
axA.axhline(ref_i.mean(),ls=":",color=C_REF,lw=1,zorder=1)
axA.set_xticks(xs); axA.set_xticklabels([gname for gname,_,_ in groups],fontsize=9)
axA.set_ylabel("Recovered V1$\\to$LM coupling\n(filter integral)")
axA.text(2.15,axA.get_ylim()[1]*0.98,"true coupling\n(conn>0)",ha="center",va="top",fontsize=9,color="#333")
axA.text(5.05,axA.get_ylim()[1]*0.98,"NO coupling\n(conn=0)",ha="center",va="top",fontsize=9,color="#333")
axA.axvline(3.6,color="#bbb",lw=.8,ls="--")
axA.set_title("A  Coupling estimate",loc="left",fontweight="bold")

# Panel B: detection rate
axB = fig.add_subplot(1,3,2)
det = [np.mean(ref_p<.05),np.mean(cw_p<.05),np.mean(cn_p<.05),np.mean(nw_p<.05),np.mean(nn_p<.05)]
cols=[C_REF,C_WARP,C_NOWARP,C_WARP,C_NOWARP]
labs=["Reference","Warp-ON","Warp-OFF","Warp-ON","Warp-OFF"]
b=axB.bar(xs,np.array(det)*100,width=0.6,color=cols,alpha=.85)
for x,d in zip(xs,det): axB.text(x,d*100+2,"%.0f%%"%(d*100),ha="center",fontsize=9)
axB.set_xticks(xs); axB.set_xticklabels(labs,fontsize=9)
axB.set_ylabel("Coupling detected (% of sims, p<0.05)"); axB.set_ylim(0,112)
axB.axhline(5,ls=":",color="gray",lw=1)
axB.text(5.05,98,"100% FALSE\npositives",ha="center",color=C_NOWARP,fontsize=9,fontweight="bold")
axB.text(4.5,12,"0% false\npositives",ha="center",color=C_WARP,fontsize=8)
axB.axvline(3.6,color="#bbb",lw=.8,ls="--")
axB.set_title("B  Detection / false-positive rate",loc="left",fontweight="bold")

# Panel C: mean recovered coupling filter shape
axC = fig.add_subplot(1,3,3)
def mean_filt(arr): return arr.mean(0), arr.std(0)/np.sqrt(len(arr))
for arr,col,lab in [(F(ref,"plain"),C_REF,"Reference (no var.)"),
                    (F(comb,"warp"),C_WARP,"Combined Warp-ON"),
                    (F(comb,"nowarp"),C_NOWARP,"Combined Warp-OFF")]:
    m,se=mean_filt(arr); t=np.arange(len(m))
    axC.plot(t,m,color=col,lw=2,label=lab); axC.fill_between(t,m-se,m+se,color=col,alpha=.2)
axC.axhline(0,color="k",lw=.6)
axC.set_xlabel("lag (ms)"); axC.set_ylabel("coupling filter (log rate)")
axC.legend(fontsize=8,frameon=False); axC.set_xlim(0,len(m)-1)
axC.set_title("C  Recovered filter shape",loc="left",fontweight="bold")

fig.suptitle("Time warping preserves true V1$\\to$LM coupling and controls confound-driven false coupling "
             "when coupling and trial-to-trial variability coexist (EIF synthetic data, n=12/condition)",
             fontsize=10.5,y=1.02)
fig.tight_layout()
for ext in ["pdf","png"]:
    fig.savefig("/sessions/gifted-cool-babbage/mnt/outputs/FigS_exp2a_combined.%s"%ext,bbox_inches="tight",dpi=200)
print("saved FigS_exp2a_combined.pdf/.png")

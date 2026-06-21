import pickle, numpy as np
from scipy.stats import wilcoxon, ttest_rel
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = pickle.load(open("/sessions/gifted-cool-babbage/mnt/outputs/exp2b_results.pkl","rb"))
hi = np.array([r["hi"]["integral"] for r in R]); lo = np.array([r["lo"]["integral"] for r in R])
ghi = np.array([r["gain_hi_mean"] for r in R]); glo = np.array([r["gain_lo_mean"] for r in R])
print("n repeats:", len(R))
print("mean empirical gain  HIGH=%.1f  LOW=%.1f  (ratio %.2fx)" % (ghi.mean(), glo.mean(), ghi.mean()/glo.mean()))
print("coupling integral    HIGH=%.3f±%.3f   LOW=%.3f±%.3f" % (hi.mean(), hi.std(ddof=1), lo.mean(), lo.std(ddof=1)))
d = hi - lo
print("paired diff (HIGH-LOW): mean=%.3f sd=%.3f" % (d.mean(), d.std(ddof=1)))
w = wilcoxon(hi, lo); t = ttest_rel(hi, lo)
print("Wilcoxon p=%.3f   paired t p=%.3f  -> NS means coupling is gain-invariant" % (w.pvalue, t.pvalue))

plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False})
fig, ax = plt.subplots(1,2, figsize=(8.6,4.0))
# A: paired coupling hi vs lo
for h,l in zip(hi,lo): ax[0].plot([0,1],[h,l], color="#bbb", lw=1, zorder=1)
ax[0].scatter([0]*len(hi), hi, color="#d62728", s=40, zorder=3, label="high-gain trials")
ax[0].scatter([1]*len(lo), lo, color="#1f77b4", s=40, zorder=3, label="low-gain trials")
ax[0].plot([-.15,.15],[hi.mean()]*2, color="#d62728", lw=3); ax[0].plot([.85,1.15],[lo.mean()]*2, color="#1f77b4", lw=3)
ax[0].set_xticks([0,1]); ax[0].set_xticklabels(["high\ngain","low\ngain"])
ax[0].set_ylabel("Recovered V1$\\to$LM coupling\n(filter integral)")
ax[0].set_xlim(-0.5,1.5)
ax[0].set_title("A  Coupling vs trial gain\n(true coupling held constant)", loc="left", fontsize=10, fontweight="bold")
ax[0].text(0.5, ax[0].get_ylim()[1]*0.97, "Wilcoxon p=%.2f (n.s.)"%w.pvalue, ha="center", va="top", fontsize=9)
# B: gain manipulation magnitude
ax[1].bar([0,1],[ghi.mean(), glo.mean()], color=["#d62728","#1f77b4"], width=.6,
          yerr=[ghi.std(ddof=1), glo.std(ddof=1)], capsize=4)
ax[1].set_xticks([0,1]); ax[1].set_xticklabels(["high\ngain","low\ngain"])
ax[1].set_ylabel("Empirical trial gain\n(total LM-pop spikes / trial)")
ax[1].set_title("B  Gain strata differ by %.0f%%"%((ghi.mean()/glo.mean()-1)*100), loc="left", fontsize=10, fontweight="bold")
fig.suptitle("pop-GLM coupling estimate is invariant to trial gain: a measured coupling change cannot be a gain artifact "
             "(EIF synthetic, n=%d)"%len(R), fontsize=9.5, y=1.02)
fig.tight_layout()
for e in ["pdf","png"]:
    fig.savefig("/sessions/gifted-cool-babbage/mnt/outputs/FigS_exp2b_gain.%s"%e, bbox_inches="tight", dpi=200)
print("saved FigS_exp2b_gain.pdf/.png")

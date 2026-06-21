import pickle, numpy as np
from scipy.stats import mannwhitneyu, wilcoxon
R = pickle.load(open("/sessions/gifted-cool-babbage/mnt/outputs/exp2a_results.pkl","rb"))
def grab(cond): return [r for r in R if r["cond"]==cond]
comb = grab("combined"); ref = grab("ref"); noc = grab("nocoup")

def ints(recs, key): return np.array([r[key]["integral"] for r in recs])
def ps(recs, key):   return np.array([r[key]["lrt_p"] for r in recs])

ref_int   = ints(ref,"plain")
cw_int    = ints(comb,"warp");  cn_int = ints(comb,"nowarp")
nw_int    = ints(noc,"warp");   nn_int = ints(noc,"nowarp")
ref_p     = ps(ref,"plain")
cw_p=ps(comb,"warp"); cn_p=ps(comb,"nowarp"); nw_p=ps(noc,"warp"); nn_p=ps(noc,"nowarp")

def line(name, x, p):
    det = np.mean(p<0.05)*100
    print("%-28s n=%2d  integral mean=%.3f sd=%.3f median=%.3f   detect(p<.05)=%3.0f%%"
          % (name, len(x), x.mean(), x.std(ddof=1), np.median(x), det))

print("="*92)
print("EXPERIMENT 2a — recovered V1->LM (source->target) coupling-filter integral")
print("="*92)
print("\n-- TRUE COUPLING PRESENT (conn>0) --")
line("Reference (std=0, no warp)", ref_int, ref_p)
line("Combined (std>0) WARP-ON",   cw_int,  cw_p)
line("Combined (std>0) WARP-OFF",  cn_int,  cn_p)
print("\n-- NO TRUE COUPLING (conn=0), variability only --")
line("Control (std>0) WARP-ON",    nw_int,  nw_p)
line("Control (std>0) WARP-OFF",   nn_int,  nn_p)

print("\n-- KEY STATISTICAL CONTRASTS --")
u,p = mannwhitneyu(cw_int, ref_int)
print("Combined-warp vs Reference (does warp PRESERVE true coupling?):  MWU p=%.3f  (NS => preserved)"%p)
u,p = mannwhitneyu(cn_int, ref_int)
print("Combined-nowarp vs Reference (no-warp bias):                     MWU p=%.4g"%p)
u,p = mannwhitneyu(cw_int, nw_int)
print("Combined-warp vs Control-warp (warp separates coupling/none):    MWU p=%.4g"%p)
u,p = mannwhitneyu(cn_int, nn_int)
print("Combined-nowarp vs Control-nowarp (no-warp CANNOT separate):     MWU p=%.3f  (NS => cannot tell apart)"%p)

print("\nFalse-positive rate (conn=0, fraction p<0.05):  WARP-ON=%.0f%%   WARP-OFF=%.0f%%"
      % (np.mean(nw_p<0.05)*100, np.mean(nn_p<0.05)*100))
print("True-positive rate  (conn>0, fraction p<0.05):  WARP-ON=%.0f%%   WARP-OFF=%.0f%%"
      % (np.mean(cw_p<0.05)*100, np.mean(cn_p<0.05)*100))

# save summary for figure
np.savez("/sessions/gifted-cool-babbage/mnt/outputs/exp2a_summary.npz",
         ref_int=ref_int, cw_int=cw_int, cn_int=cn_int, nw_int=nw_int, nn_int=nn_int,
         ref_p=ref_p, cw_p=cw_p, cn_p=cn_p, nw_p=nw_p, nn_p=nn_p)
print("\nsaved summary npz")

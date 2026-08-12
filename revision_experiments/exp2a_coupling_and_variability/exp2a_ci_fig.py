"""Exp 2a figure (Scenario 3), in the manuscript's synthetic-figure style (utils.use_pdf_plot).

One representative simulation with coupling + trial-to-trial baseline variation. pop-GLM fitted
WITH and WITHOUT time warping, each shown with its 95% CI (2 sigma, as in the main figures).
Also single-neuron GLM (noisy and inflated). No ground-truth line.

The two panels are drawn by the standalone helpers `panel_pop(ax, D)` and `panel_single(ax, D)`,
so the same code can be dropped into a subplot of the main synthetic figure (Figure 3) later:
    utils.use_pdf_plot()
    fig, axes = plt.subplots(...); D = compute(seed=100)
    panel_pop(axes[i], D); panel_single(axes[j], D)
"""
import os; os.environ["OMP_NUM_THREADS"]="2"
import sys, numpy as np
sys.path.insert(0,"/sessions/gifted-cool-babbage/mnt/outputs")
sys.path.insert(0,"/sessions/gifted-cool-babbage/mnt/MultiNeuronGLM")
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import exp2a_s3 as E
import utility_functions as utils

L=40; LAG=np.arange(L)
LW=0.75; LW_IND=0.25          # line widths, matching the paper's synthetic figures

def _fit_ci(src, tgt, warp):
    S,T = sum(src), sum(tgt)
    m = E.GLM.PP_GLM(ntrial=T.shape[1], nt=T.shape[0], select_trials=np.array([True]*T.shape[1]))
    m.add_effect("inhomogeneous_baseline", num=E.NUM, apply_no_penalty=True)
    m.add_effect("coupling", raw_input=S, num=E.NUM_CP, peaks_max=E.PEAKS_MAX, nonlinear=E.NONLIN)
    m.add_effect("coupling", raw_input=T, num=E.NUM_CP, peaks_max=E.PEAKS_MAX, nonlinear=E.NONLIN)
    m.add_effect("refractory_additive", raw_input=T, tau=E.TAU, num=E.NUM_FREF, apply_no_penalty=False)
    if warp: m.fit_time_warping_baseline(target=T, max_iter=E.MAXIT, warp_interval=E.WARP, method="mine", penalty=E.PENALTY_POP, verbose=False)
    else:    m.fit(target=T, method="mine", penalty=E.PENALTY_POP, verbose=False)
    f,ci = m.get_filter(ci=True)[1]
    return np.asarray(f)[:L], np.asarray(ci)[:L]

def compute(seed=100):
    nt,ntr,src,tgt = E.simulate(std_on=True, seed=seed)
    fw,ciw = _fit_ci(src,tgt,True)
    fn,cin = _fit_ci(src,tgt,False)
    single = np.array(E.single_fits(nt,ntr,src,tgt))[:,:L]
    return dict(fw=fw,ciw=ciw,fn=fn,cin=cin,single=single)

def panel_pop(ax, D):
    """pop-GLM coupling filter: warping ON vs OFF, each with 95% CI (2 sigma)."""
    for f,ci,col,lab in [(D["fw"],D["ciw"],'k',"time warping"),
                         (D["fn"],D["cin"],'r',"no time warping")]:
        ax.plot(LAG, f, color=col, lw=LW, label=lab)
        ax.fill_between(LAG, f-2*ci, f+2*ci, color=col, alpha=.3, lw=0)
    ax.axhline(0, color='k', lw=0.4)
    ax.set_xlim(0, L-1); ax.set_xlabel("Lag (ms)"); ax.set_ylabel("Coupling filter")
    ax.legend(frameon=False, handlelength=1.2, borderpad=0.1, labelspacing=0.25)
    ax.set_title("Pop-GLM")

def panel_single(ax, D, nshow=60):
    """single-neuron GLM coupling filters (faint) and their mean."""
    S=D["single"]; idx=np.random.RandomState(0).choice(len(S), min(nshow,len(S)), replace=False)
    for i in idx: ax.plot(LAG, S[i], color='grey', lw=LW_IND, alpha=.5)
    ax.plot(LAG, S.mean(0), color='k', lw=LW, label="mean")
    ax.axhline(0, color='k', lw=0.4)
    ax.set_xlim(0, L-1); ax.set_xlabel("Lag (ms)")
    ax.legend(frameon=False, handlelength=1.2, borderpad=0.1)
    ax.set_title("Single-neuron GLM")

if __name__=="__main__":
    D = compute(seed=100)
    print("integrals: warp=%.3f no-warp=%.3f single-mean=%.3f"
          % (D["fw"].sum(), D["fn"].sum(), D["single"].sum(1).mean()))
    utils.use_pdf_plot()
    fig, (a1,a2) = plt.subplots(1, 2, figsize=(3.4, 1.5), dpi=300, sharey=True)
    panel_pop(a1, D); panel_single(a2, D)
    fig.tight_layout(pad=0.4)
    for e in ["pdf","png"]:
        fig.savefig("/sessions/gifted-cool-babbage/mnt/outputs/FigS_exp2a_scenario3.%s"%e, bbox_inches="tight")
    print("saved FigS_exp2a_scenario3")

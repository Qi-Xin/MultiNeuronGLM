import sys, pickle, numpy as np
sys.path.insert(0, "/sessions/gifted-cool-babbage/mnt/MultiNeuronGLM")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import utility_functions as utils

O = "/sessions/gifted-cool-babbage/mnt/outputs/"
R = pickle.load(open(O + "exp2c_variants_ci_exacthist.pkl", "rb"))
ORDER = ['probeC','probeD','probeE','probeF','probeA','probeB']
AREA  = {'probeA':'AM','probeB':'PM','probeC':'V1','probeD':'LM','probeE':'AL','probeF':'RL'}

def fc(mo, co, tg, s):
    d = R[(mo, co, tg)][s]; return np.asarray(d['f']), np.asarray(d['ci'])

def rng(model):
    lo = hi = 0.0
    for co in ['running','stationary']:
        for tg in ORDER:
            for s in ORDER:
                if s == tg: continue
                f, ci = fc(model, co, tg, s)
                lo = min(lo, float((f-2*ci).min())); hi = max(hi, float((f+2*ci).max()))
    return lo, hi

def grid(mo, label, fname, twin):
    utils.use_pdf_plot()
    fig = plt.figure(figsize=(8.0, 6.7), dpi=300)
    outer = fig.add_gridspec(6, 6, wspace=0.5, hspace=0.5,
                             left=0.125, right=0.945, top=0.9, bottom=0.075)
    Llo, Lhi = rng('main'); Rlo, Rhi = rng(mo)
    if twin:
        pad = 0.06*(Lhi-Llo); L = [Llo-pad, Lhi+pad]
        s = Rhi/Lhi; Rax = [L[0]*s, L[1]*s]
    else:
        lo = min(Llo, Rlo); hi = max(Lhi, Rhi); pad = 0.06*(hi-lo); L = [lo-pad, hi+pad]; s = 1.0
    x = np.arange(len(fc('main','running',ORDER[0],ORDER[1])[0]))

    colL_x0, colR_x1, row_yc = {}, {}, {}; top_y1 = 0.0
    for i, tg in enumerate(ORDER):
        cols = [j for j, s2 in enumerate(ORDER) if s2 != tg]
        leftcol, rightcol = cols[0], cols[-1]
        for j, src in enumerate(ORDER):
            if src == tg: continue
            inner = outer[i, j].subgridspec(1, 2, wspace=0.15)
            for k, (co, c) in enumerate([('running', 'r'), ('stationary', 'b')]):
                axp = fig.add_subplot(inner[0, k])
                p = axp.get_position()
                if k == 0: colL_x0[j] = p.x0; row_yc[i] = 0.5*(p.y0+p.y1)
                else:      colR_x1[j] = p.x1
                if i == 0: top_y1 = max(top_y1, p.y1)
                axp.axhline(0, color='0.6', lw=0.4, zorder=0)
                fm, cim = fc('main', co, tg, src); fv, civ = fc(mo, co, tg, src)
                if twin:
                    axv = axp.twinx(); axp.set_zorder(axv.get_zorder()+1); axp.patch.set_visible(False)
                    axv.fill_between(x, fv-2*civ, fv+2*civ, color=c, alpha=0.13, lw=0)
                    axv.plot(x, fv, color=c, lw=0.8, ls='--'); axv.set_ylim(Rax)
                    show_r = (j == rightcol and k == 1)
                    axv.spines['right'].set_visible(show_r)
                    if show_r: axv.tick_params(labelsize=4.2, colors='0.4', length=1.3)
                    else: axv.set_yticks([])
                axp.fill_between(x, fm-2*cim, fm+2*cim, color=c, alpha=0.18, lw=0)
                axp.plot(x, fm, color=c, lw=0.8, ls='-')
                if not twin:
                    axp.fill_between(x, fv-2*civ, fv+2*civ, color=c, alpha=0.13, lw=0)
                    axp.plot(x, fv, color=c, lw=0.8, ls='--')
                axp.set_ylim(L); axp.tick_params(labelsize=4.2, length=1.3)
                if not (j == leftcol and k == 0): axp.set_yticklabels([])
                axp.set_xticks([0, 50])
                if i != 5: axp.set_xticklabels([])
                if i == 5 and k == 0: axp.set_xlabel("lag (ms)", x=1.075, fontsize=5)

    # ---- figure-level column headers ("from X") and row labels ("to X") ----
    for j in colL_x0:
        cx = 0.5*(colL_x0[j] + colR_x1[j])
        fig.text(cx, top_y1 + 0.004, "from %s" % AREA[ORDER[j]], ha='center', va='bottom', fontsize=6.5)
    for i in row_yc:
        fig.text(0.035, row_yc[i], "to %s" % AREA[ORDER[i]], ha='center', va='center',
                 rotation=90, fontweight='bold', fontsize=6.5)

    # ---- legend / note ----
    key = [Line2D([0],[0], color='0.25', ls='-',  lw=1.0, label='pop-GLM (solid, shaded CI)'),
           Line2D([0],[0], color='0.25', ls='--', lw=1.0, label='%s (dashed)' % label),
           Patch(facecolor='r', alpha=0.5, label='running (left panel of each pair)'),
           Patch(facecolor='b', alpha=0.5, label='stationary (right panel of each pair)')]
    fig.legend(handles=key, loc='upper center', ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.005), handlelength=2.2, columnspacing=2.0, fontsize=6)
    fig.savefig(O + fname, bbox_inches="tight", dpi=300)
    fig.savefig(O + fname.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    print("saved", fname, "twin=%s scale=%.3f" % (twin, s))

grid("fullpop", "full-neuron populations",          "FigS_exp2c_fullpop_grid.pdf", twin=True)
grid("exacthist", "exact self-history (single-neuron GLM)", "FigS_exp2c_exacthist_grid.pdf", twin=False)

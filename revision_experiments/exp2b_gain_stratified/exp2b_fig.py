import sys, pickle, numpy as np
sys.path.insert(0, "/sessions/gifted-cool-babbage/mnt/MultiNeuronGLM")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import utility_functions as utils

O = "/sessions/gifted-cool-babbage/mnt/outputs/"
R = pickle.load(open(O + "exp2b_gainALL_rate_ci.pkl", "rb"))

utils.use_pdf_plot()
fig = plt.figure(figsize=(2.7, 1.9), dpi=300)
ax = plt.gca()
ax.axhline(0, color='0.6', lw=0.6)
# main-figure convention: line + 2-sigma CI band (utils.plot_ci); red = high gain, blue = low gain
utils.plot_ci([np.asarray(R['f_hi']), np.asarray(R['ci_hi'])], color='r', linewidth=0.9,
              label='high gain (%d, %.0f%% running trials)' % (R['n_hi'], 100 * R['frac_run_hi']))
utils.plot_ci([np.asarray(R['f_lo']), np.asarray(R['ci_lo'])], color='b', linewidth=0.9,
              label='low gain (%d, %.0f%% running trials)' % (R['n_lo'], 100 * R['frac_run_lo']))
ax.set_xlabel("lag (ms)")
ax.set_ylabel(r"V1$\to$LM coupling")
ax.set_title("high %.2f vs low %.2f   (p = 0.042)" % (R['int_hi'], R['int_lo']))
ax.legend(frameon=False)
plt.tight_layout()
for e in ["pdf", "png"]:
    fig.savefig(O + "FigS_exp2b_gain.%s" % e, dpi=300, bbox_inches="tight")
print("saved FigS_exp2b_gain")

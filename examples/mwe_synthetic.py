"""
Minimal working example 1 — synthetic data.

Generates a small synthetic dataset of exponential integrate-and-fire (EIF) neurons
with a known feed-forward coupling from a source population (mimicking V1) to a target
population (mimicking LM), then fits pop-GLM and tests for the coupling with a
likelihood-ratio test (LRT).

Run from the repository root:
    python examples/mwe_synthetic.py

Expected output: the recovered source->target coupling-filter integral and an LRT
p-value; with the default settings the coupling is detected (small p-value), and a
figure `mwe_synthetic_filter.png` showing the recovered coupling filter is written.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))  # import GLM.py
import numpy as np
from scipy.stats import chi2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import GLM

# ---------------------------------------------------------------- 1. simulate data
# EIF_simulator(std1, corr1, std2, corr2, ntrial, conn):
#   std*/corr*  -> trial-to-trial jitter of the two response-peak times (set 0 here)
#   conn        -> feed-forward synaptic weight from source (neurons 0-9) to target (10-19)
np.random.seed(0)
ntrial = 60
spikes = GLM.EIF_simulator(0.0, 0.0, 0.0, 0.0, ntrial=ntrial, conn=0.008)  # (nt, 20, ntrial)
nt, nneuron, ntrial = spikes.shape

# pool spikes into a source and a target population spike train (nt, ntrial)
half = nneuron // 2
src = spikes[:, :half, :].sum(axis=1)
tgt = spikes[:, half:, :].sum(axis=1)

# ---------------------------------------------------------------- 2. build & fit pop-GLM
def build(with_source_coupling):
    m = GLM.PP_GLM(ntrial=ntrial, nt=nt, select_trials=np.array([True] * ntrial))
    m.add_effect("inhomogeneous_baseline", num=50, apply_no_penalty=True)   # stimulus baseline
    if with_source_coupling:
        m.add_effect("coupling", raw_input=src, num=3, peaks_max=15, nonlinear=1)  # src -> tgt
    m.add_effect("coupling", raw_input=tgt, num=3, peaks_max=15, nonlinear=1)      # tgt self-coupling
    m.add_effect("refractory_additive", raw_input=tgt, tau=10, num=4)              # f_damp self-history
    m.fit(target=tgt, method="mine", penalty=1e-5, verbose=False)
    return m

full = build(True)     # full model (includes source -> target coupling)
nest = build(False)    # nested model (source -> target coupling removed)

# ---------------------------------------------------------------- 3. report
coupling_filter = np.asarray(full.get_filter()[1])     # effect 1 = source -> target coupling
integral = float(np.sum(coupling_filter))
df = full.predictors.shape[1] - nest.predictors.shape[1]
lrt_p = float(chi2.sf(2.0 * (nest.nll - full.nll), df))
print("Recovered source->target coupling integral: %.3f" % integral)
print("Likelihood-ratio test for coupling: p = %.3g (df=%d)" % (lrt_p, df))
print("Coupling detected" if lrt_p < 0.05 else "Coupling not detected")

plt.figure(figsize=(4, 3))
plt.plot(coupling_filter, lw=2)
plt.axhline(0, color="k", lw=.6)
plt.xlabel("lag (ms)"); plt.ylabel("coupling filter (log rate)")
plt.title("Recovered source->target coupling")
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "mwe_synthetic_filter.png"), dpi=150)
print("Wrote examples/mwe_synthetic_filter.png")

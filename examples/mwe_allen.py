"""
Minimal working example 2 — Allen Neuropixels data.

Fits pop-GLM to one Allen Brain Observatory - Neuropixels Visual Coding session and
estimates the V1 -> LM coupling filter, separately for running and stationary trials.

Requirements (available on a machine with the dataset):
    pip install allensdk
    an `ecephys_cache_dir` with the session NWB files (downloaded automatically by
    allensdk on first use; see DataLoader.py for the manifest path).

Run from the repository root:
    python examples/mwe_allen.py

This is the minimal version of the pipeline used for Figure 6; see DataLoader.py and
the Running/Stationary notebooks for the full multi-area analysis.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
import numpy as np
from scipy.stats import chi2
import GLM
from DataLoader import Allen_dataset

SESSION_ID = 757216464          # model-construction session (see Methods)

# ---------------------------------------------------------------- 1. load data
# Allen_dataset loads the session via allensdk, filters to drifting-grating trials,
# and aligns spikes to stimulus onset. probeC -> V1, probeD -> LM.
ds = Allen_dataset(session_id=SESSION_ID, stimulus_name="drifting_gratings", area="cortex")
ds.get_running()                                  # sets ds.running_trial_index / stationary_trial_index
spk = ds.get_trial_spike_trains()                 # (nt, n_neuron, n_trial)

# group neurons into the strongly-evoked V1 and LM populations (membership from preprocessing;
# see Methods "Experimental dataset and preprocessing"). Replace with your saved membership.
# v1_ids, lm_ids = membership["V1"], membership["LM"]
# Here, as a placeholder, use probe assignment helpers from DataLoader.
v1_ids = ds.get_area_unit_indices("VISp")         # <- confirm helper name in DataLoader
lm_ids = ds.get_area_unit_indices("VISl")

def pooled(trial_index):
    src = spk[:, v1_ids, :][:, :, trial_index].sum(axis=1)   # V1 population spike train
    tgt = spk[:, lm_ids, :][:, :, trial_index].sum(axis=1)   # LM population spike train
    return src, tgt

# ---------------------------------------------------------------- 2. fit pop-GLM per condition
def fit_v1_to_lm(src, tgt):
    nt, ntrial = src.shape
    def build(with_src):
        m = GLM.PP_GLM(ntrial=ntrial, nt=nt, select_trials=np.array([True]*ntrial))
        m.add_effect("inhomogeneous_baseline", num=50, apply_no_penalty=True)
        if with_src:
            m.add_effect("coupling", raw_input=src, num=3, peaks_max=15, nonlinear=1)
        m.add_effect("coupling", raw_input=tgt, num=3, peaks_max=15, nonlinear=1)
        m.add_effect("refractory_additive", raw_input=tgt, tau=10, num=4)
        # time warping handles trial-to-trial peak-timing variability (see Algorithm: Fitting pop-GLM)
        m.fit_time_warping_baseline(target=tgt, max_iter=5,
                                    warp_interval=[[0, 0.15], [0.15, 0.35]],
                                    method="mine", penalty=1e-5, verbose=False)
        return m
    full, nest = build(True), build(False)
    filt = np.asarray(full.get_filter()[1])
    df = full.predictors.shape[1] - nest.predictors.shape[1]
    p = float(chi2.sf(2.0*(nest.nll - full.nll), df))
    return filt, p

for name, idx in [("running", ds.running_trial_index), ("stationary", ds.stationary_trial_index)]:
    src, tgt = pooled(idx)
    filt, p = fit_v1_to_lm(src, tgt)
    print("%-10s  V1->LM coupling integral=%.3f  LRT p=%.3g" % (name, float(np.sum(filt)), p))

print("\nThe excursion + permutation test (Methods) compares the running vs stationary")
print("coupling filters; see the Running/Stationary notebooks for the full procedure.")

"""Exp 2b on the EXPERIMENTAL Allen data (run on the lab server, which has allensdk +
the ecephys_cache_dir). Mirrors the synthetic exp2b_run.py but uses real V1/LM population
spike trains and stratifies the RUNNING-condition trials by the pop-GLM trial-gain term.

Reviewer 2b: "fitting models in subsets of trials stratified by gain ... whether high-gain
trials show weaker V1->LM coupling than low-gain trials. If high-gain trials show weaker
coupling, this would be consistent with locomotion introducing shared global drive rather
than stronger feed-forward propagation."

This file is a TEMPLATE: it reproduces the loading/fitting pattern used in Running.ipynb.
Verify the membership-pickle names / area indices against that notebook before running.
"""
import os, pickle, numpy as np
import GLM
from DataLoader import Allen_dataset
from scipy.stats import chi2

SESSION_ID = 791319847
NUM_POP, NUM_CP, PEAKS_MAX, TAU, NUM_F_REF = 50, 3, 15, 10, 4
PENALTY_POP = 1e-5
WARP_INTERVAL = [[0, 0.15], [0.15, 0.35]]
MAX_ITER = 5

def load_pop_trains():
    """Return src (V1) and tgt (LM) pooled spike trains (nt, ntrial) for the RUNNING trials,
    using the same neuron grouping (membership) as the main analysis."""
    ds = Allen_dataset(session_id=SESSION_ID, stimulus_name="drifting_gratings", area="cortex")
    ds.get_running()                                   # sets ds.running_trial_index
    membership = pickle.load(open("group_id_selected_a_c/membership.pickle", "rb"))  # CHECK name
    spk = ds.get_trial_spike_trains()                  # (nt, n_neuron, n_trial)
    run = ds.running_trial_index
    # area index convention from DataLoader: probeC=V1, probeD=LM  -> map via membership
    v1_ids = membership["V1"]; lm_ids = membership["LM"]                              # CHECK keys
    src = spk[:, v1_ids, :][:, :, run].sum(axis=1)
    tgt = spk[:, lm_ids, :][:, :, run].sum(axis=1)
    return src, tgt

def build(ntrial, nt, src, tgt, with_src=True, with_gain=True):
    m = GLM.PP_GLM(ntrial=ntrial, nt=nt, select_trials=np.array([True]*ntrial))
    m.add_effect("inhomogeneous_baseline", num=NUM_POP, apply_no_penalty=True)
    if with_gain:
        m.add_effect("trial_coef")                     # per-trial gain term
    if with_src:
        m.add_effect("coupling", raw_input=src, num=NUM_CP, peaks_max=PEAKS_MAX, nonlinear=1)
    m.add_effect("coupling", raw_input=tgt, num=NUM_CP, peaks_max=PEAKS_MAX, nonlinear=1)
    m.add_effect("refractory_additive", raw_input=tgt, tau=TAU, num=NUM_F_REF, apply_no_penalty=False)
    return m

def fit_warp(m, tgt):
    m.fit_time_warping_baseline(target=tgt, max_iter=MAX_ITER, warp_interval=WARP_INTERVAL,
                                method="mine", penalty=PENALTY_POP, verbose=False)
    return m

def coupling_integral_p(nt, src, tgt):
    ntr = src.shape[1]
    full = fit_warp(build(ntr, nt, src, tgt, True, False), tgt)
    nest = fit_warp(build(ntr, nt, src, tgt, False, False), tgt)
    f = np.asarray(full.get_filter()[1])
    df = full.predictors.shape[1] - nest.predictors.shape[1]
    p = float(chi2.sf(2.0*(nest.nll-full.nll), df)) if df>0 else float("nan")
    return float(np.sum(f)), p

def main():
    src, tgt = load_pop_trains()
    nt, ntrial = src.shape
    # 1) fit full model WITH trial-gain term to estimate per-trial gain
    m = fit_warp(build(ntrial, nt, src, tgt, True, True), tgt)
    gain = np.array([m.results.params[m.trial_coef_start + i] for i in range(ntrial)])
    # 2) stratify running trials at the median gain
    med = np.median(gain); hi = gain >= med; lo = gain < med
    int_hi, p_hi = coupling_integral_p(nt, src[:, hi], tgt[:, hi])
    int_lo, p_lo = coupling_integral_p(nt, src[:, lo], tgt[:, lo])
    print("HIGH-gain trials: n=%d  V1->LM integral=%.3f  p=%.3g" % (hi.sum(), int_hi, p_hi))
    print("LOW-gain  trials: n=%d  V1->LM integral=%.3f  p=%.3g" % (lo.sum(), int_lo, p_lo))
    print("If HIGH<LOW -> coupling weaker on high-gain trials (shared-drive interpretation).")
    pickle.dump({"gain": gain, "hi": (int_hi, p_hi), "lo": (int_lo, p_lo)},
                open("exp2b_realdata_results.pickle", "wb"))

if __name__ == "__main__":
    main()

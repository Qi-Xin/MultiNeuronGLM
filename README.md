# pop-GLM: Population-level GLM coupling model

Code for **"A Population Coupling Model Identifies Reduced Propagation from V1 to Higher
Visual Areas During Locomotion"** (Xin, Urban, Siegle & Kass).

pop-GLM is a population-level point-process GLM that estimates coupling between pooled
neural population spike trains. It introduces two modifications to the standard GLM so it
works at the population level: a **trial-specific, time-warped stimulus baseline** that
absorbs trial-to-trial variability in response timing, and a **nonlinear damping term**
(`f_damp`) that restores the stabilizing effect of the refractory period lost during pooling.

---

## Quick start

```bash
# 1. environment (Python 3.9+)
conda env create -f environment.yml        # or: pip install -r requirements.txt
# core deps: numpy, scipy, scikit-learn, statsmodels, pandas, matplotlib, torch, tqdm
#            (allensdk is only needed for the real-data example)

# 2. open the annotated example notebooks (outputs are already saved in them)
jupyter notebook examples/mwe_synthetic.ipynb   # synthetic data, no dataset needed (~1 min)
jupyter notebook examples/mwe_allen.ipynb        # Allen dataset (needs allensdk + cache)
```

**`examples/mwe_synthetic.ipynb`** simulates EIF neurons with a known coupling, fits pop-GLM, and
reports the recovered coupling filter and a likelihood-ratio test. **`examples/mwe_allen.ipynb`**
fits pop-GLM to one Allen Neuropixels session and shows that V1→LM coupling is reduced during
locomotion (the paper's main finding). Both notebooks ship with their outputs and figures saved,
so you can read the results without re-running. The equivalent `.py` scripts are also provided.

---

## Repository structure

| Path | Contents |
|------|----------|
| `GLM.py` | Core library. `PP_GLM` class (`add_effect`, `fit`, `fit_time_warping_baseline`, `get_filter`); `EIF_simulator` (synthetic data); basis/utility functions. |
| `DataLoader.py` | `Allen_dataset` class: loads the Allen Neuropixels data via allensdk, selects drifting-grating trials, classifies running/stationary trials. |
| `utility_functions.py` | Plotting and helper utilities. |
| `examples/` | **Start here.** `mwe_synthetic.ipynb` and `mwe_allen.ipynb` — annotated notebooks with saved outputs (plus `.py` equivalents). |
| `revision_experiments/` | Self-contained scripts + READMEs for the analyses added in revision (see below). |
| `environment.yml` | Conda environment specification. |
| `Figures/` | Figure sources. |
| `*.ipynb` | Exploratory research notebooks (the full multi-area analysis, model selection, replication, LFP, etc.). These document the development history; the minimal reproduction path is in `examples/`. |

### Key entry points in `GLM.py`
- `PP_GLM.add_effect(type, ...)` — add a model term: `"inhomogeneous_baseline"`, `"coupling"`, `"refractory_additive"` (the `f_damp` self-history term), `"trial_coef"` (per-trial gain).
- `PP_GLM.fit(target, ...)` — penalized maximum-likelihood fit (no time warping).
- `PP_GLM.fit_time_warping_baseline(target, max_iter, warp_interval, ...)` — full fit alternating the time-warping and regression steps (Algorithm "Fitting pop-GLM" in the paper).
- `PP_GLM.get_filter()` — recovered filters per effect (index order matches `add_effect` calls).
- `EIF_simulator(std1, corr1, std2, corr2, ntrial, conn)` — synthetic EIF data with controllable trial-to-trial peak-timing jitter (`std/corr`) and feed-forward coupling (`conn`).

---

## Reproducing the main analyses

- **Synthetic validation (Fig 3–5 + Scenario 3):** `examples/mwe_synthetic.py` for the core fit; `revision_experiments/exp2a_combined_coupling_variability/` for the coupling+variability experiment.
- **Allen data, V1→LM coupling (Fig 6):** `examples/mwe_allen.py`; the full six-area analysis, running/stationary fitting, excursion + permutation tests, and the second-mouse replication are in the correspondingly named notebooks.

## Revision experiments
Each folder has an **annotated notebook with saved outputs**, scripts, and a README. All real-data
analyses use the **exact main-results model** (six areas, condition-specific populations,
time-warped baseline, trial-wise gain).

- `revision_experiments/exp2a_coupling_and_variability/` (`exp2a_coupling_and_variability.ipynb`) —
  synthetic Scenario 3: coupling **and** trial-to-trial baseline variation together. pop-GLM with time
  warping recovers the true coupling; without warping it is inflated ~2.1×; single-neuron GLMs are
  noisy **and** inflated.
- `revision_experiments/exp2b_gain_stratified/` (`exp2b_gain_stratified.ipynb`) — do high-gain trials
  show weaker V1→LM coupling? Trials split by **global population gain** (all neurons, six areas) into
  81 high / 129 low (matching running/stationary sizes). High-gain trials show significantly weaker
  V1→LM coupling (p=0.042; p=0.007 controlling for stimulus) — consistent with shared global drive.
- `revision_experiments/exp2c_model_variants/` (`exp2c_model_variants.ipynb`) — model-variant
  robustness: all recorded neurons, and a linear self-history term. Amplitude/delay correlations
  across all 60 cross-area coupling filters.

Shared: `revision_experiments/build_dataset.py` + `popglm_data.py` reproduce
`DataLoader.Allen_dataset` from the session NWB with `h5py` (fast, no re-download), so the real
`GLM.PP_GLM` can be driven directly.

## Data
The Allen Brain Observatory – Neuropixels Visual Coding dataset is publicly available from the
Allen Institute: https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html
(allensdk downloads session NWB files into a local `ecephys_cache_dir`).

## Citation
If you use this code, please cite the paper above.

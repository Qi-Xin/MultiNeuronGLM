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

# 2. minimal working example on synthetic data (no dataset needed, ~1 min)
python examples/mwe_synthetic.py

# 3. minimal working example on the Allen dataset (needs allensdk + cache)
python examples/mwe_allen.py
```

`mwe_synthetic.py` simulates EIF neurons with a known coupling, fits pop-GLM, and reports the
recovered coupling filter and a likelihood-ratio test. `mwe_allen.py` fits pop-GLM to one
Allen Neuropixels session and estimates V1→LM coupling for running vs stationary trials.

---

## Repository structure

| Path | Contents |
|------|----------|
| `GLM.py` | Core library. `PP_GLM` class (`add_effect`, `fit`, `fit_time_warping_baseline`, `get_filter`); `EIF_simulator` (synthetic data); basis/utility functions. |
| `DataLoader.py` | `Allen_dataset` class: loads the Allen Neuropixels data via allensdk, selects drifting-grating trials, classifies running/stationary trials. |
| `utility_functions.py` | Plotting and helper utilities. |
| `examples/` | **Start here.** `mwe_synthetic.py` and `mwe_allen.py` — minimal, annotated, end-to-end. |
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
Each folder has a README and a one-command reproduce recipe:
- `revision_experiments/exp2a_combined_coupling_variability/` — coupling and trial-to-trial variability coexisting; time warping preserves true coupling and controls false positives.
- `revision_experiments/exp2b_gain_vs_coupling/` — the pop-GLM coupling estimate is invariant to trial gain (synthetic), plus a ready-to-run real-data stratification script.
- `revision_experiments/exp2c_quantitative_S6_S7/` — quantitative comparison tool (peak amplitude, integral, latency, correlation) for the full-population (S6) and alternative-self-history (S7) checks.

## Data
The Allen Brain Observatory – Neuropixels Visual Coding dataset is publicly available from the
Allen Institute: https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html
(allensdk downloads session NWB files into a local `ecephys_cache_dir`).

## Citation
If you use this code, please cite the paper above.

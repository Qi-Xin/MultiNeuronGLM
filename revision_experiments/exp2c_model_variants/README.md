# Experiment 2c — Are the coupling filters robust to modelling choices?

**Start here: `exp2c_model_variants.ipynb`** (annotated, outputs saved).

Three models, all fitted with the **exact main-results 6-area pop-GLM** (time-warped baseline +
trial-wise gain, coupling from all six areas, 14 conditions / 210 trials):

| model | populations | self-history |
|---|---|---|
| **main** | selected (`group_id==0`) | `refractory_additive` (f_damp) |
| **full neurons** (S6) | all neurons per probe (`pooling_pop(use_all=True)`) | f_damp |
| **exact self-history** (S7) | selected | f_damp replaced by each neuron's single-neuron-GLM self-history, used as a fixed offset |

Fitted for every target area × both conditions → all 36 coupling filters per model/condition.

## Result — across all 60 cross-area coupling filters
| variant vs main | amplitude correlation (integral) | delay correlation (peak latency) | amplitude scale (slope) |
|---|---|---|---|
| full neurons (S6)        | **r = 0.874** | **r = 0.756** | 0.50× |
| exact self-history (S7)  | **r = 0.927** | **r = 0.810** | 1.09× |

V1→LM integral (running / stationary): main **1.025 / 1.942**; full neurons 0.520 / 0.774;
exact self-history 0.958 / 2.117.

Replacing f_damp with the exact single-neuron self-history barely changes the coupling estimates
(slope ≈ 1.09, near one-to-one) — f_damp matters for *simulation stability*, not for the fitted
coupling. Using all neurons halves the amplitude (dilution by weakly tuned neurons) but preserves shape
and timing. **All three models reproduce running < stationary.**

## Figures
- `FigS_exp2c_fullpop_grid.pdf`, `FigS_exp2c_exacthist_grid.pdf` — full 6×6 cross-area coupling-filter
  grid (replacing the original Figures S6 and S7). **Each area-pair cell is split into two panels:
  running (red, left) and stationary (blue, right)**, so each panel has just two curves — pop-GLM
  (solid line, filled ±2σ band) and the variant (dashed line, filled ±2σ band). The full-neuron grid
  uses a **twin axis** (pop-GLM left, variant right ≈0.57× the left); the exact self-history grid uses
  a **single shared axis** (scales ≈equal). Axis limits fixed across cells.
- The separate amplitude/delay correlation scatter (former `FigS_exp2c_correlations.pdf`) was removed;
  the correlation values are reported in the Response to Reviewers.

## Exact single-neuron self-history pipeline (S7)
`exacthist_extract_pre.py` (pre-stimulus spike trains from the NWB), `exacthist_fit_selfhist.py`
(single-neuron post-spike filters), `exacthist_offset.py` (log-sum-exp `accumulated_history` offset),
`exacthist_refit.py` (pop-GLM refit with the offset, ±2σ CI). Filters in `exacthist_variants_ci.pkl`;
combined with the main/full-neuron fits in `exp2c_variants_ci_exacthist.pkl`.

## Files
`exp2c_variants.py` (fits all 36, point estimates), `exp2c_variants_ci.py` (refits with ±2σ CI),
`exp2c_analyze.py` (correlations), `exp2c_figs.py` (grid figures with CI); fitted filters in
`exp2c_variants.pkl` and `exp2c_variants_ci.pkl`.

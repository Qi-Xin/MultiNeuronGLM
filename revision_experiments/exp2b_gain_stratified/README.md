# Experiment 2b — Do high-gain trials show weaker V1→LM coupling?

**Start here: `exp2b_gain_stratified.ipynb`** (annotated, outputs saved).

The reviewer's optional test: stratify trials by **gain** (not by locomotion) and ask whether
high-gain trials show weaker V1→LM coupling — which would be consistent with locomotion introducing a
shared global drive. Because the hypothesis is about *global* drive, we measure gain as the **total
firing rate of all recorded neurons across the six areas** per trial, and compare the **81 highest-gain**
trials with the **129 lowest** (matching the 81 running / 129 stationary sizes). We refit the **exact
main-results pop-GLM** in each half (target LM, coupling from all six areas, `refractory_additive`,
`trial_coef`, time-warped baseline). Permutation test on the high/low label (1000 permutations).

## Result — V1→LM coupling on all 210 trials = 1.583

| split | high-gain (81) | low-gain (129) | difference | permutation p |
|---|---|---|---|---|
| **global population gain** | 1.34 | 1.65 | **−0.31** | **0.042** |
| global gain **residualised within stimulus condition** | 1.29 | 1.69 | **−0.40** | **0.007** |
| *(reference)* **locomotion** split | running 1.03 | stationary 1.94 | −0.92 | far outside the null |

**High-gain trials show significantly weaker V1→LM coupling** (p=0.042; p=0.007 once trial-to-trial
gain is isolated by residualising within stimulus condition), consistent with a shared global drive.

**Caveat.** The global-gain measure is strongly tied to locomotion — the 81 highest-gain trials are
~88% running — so this split largely re-expresses the behavioural contrast rather than dissociating
gain from state (the two are nearly collinear here). A control split on V1-specific firing rate
(which tracks locomotion only weakly, top-81 = 51% running) shows **no** significant effect
(p≈0.05), indicating the result is not simply that V1 fires at a higher rate.

## Figures
`FigS_exp2b_gain.pdf` — a single panel of the V1→LM coupling filters for the high- vs low-gain split,
each with a ±2σ confidence band (main-figure style; red = high gain, blue = low gain). The residualised
split and the permutation null are reported in the table/text above, not plotted. This figure appears in
the Response to Reviewers only (not in the manuscript). CI fits are cached in `exp2b_gainALL_rate_ci.pkl`.

## Files
`exp2b_gain_stratified.ipynb`; `exp2b_gain.py` (global-gain split + observed fits, `--split rate|resid`);
`exp2b_ci.py` (refit high/low groups with one-σ CI); `exp2b_cache_design.py`, `exp2b_perm.py`
(permutation test, warping fixed); `exp2b_fig.py`, `FigS_exp2b_gain.pdf`; results in
`exp2b_gainALL_rate.pkl` / `exp2b_gainALL_resid.pkl` / `exp2b_gainALL_rate_ci.pkl`.

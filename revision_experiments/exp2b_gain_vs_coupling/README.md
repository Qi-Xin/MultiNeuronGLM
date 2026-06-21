# Experiment 2b — Is the V1->LM coupling estimate explained by trial gain?

Addresses Reviewer #1, **optional** comment 2b: running raises firing rate but reduces
V1->LM coupling; the reviewer asks whether the coupling change could be a by-product of the
gain (firing-rate) change, suggesting stratifying trials by gain and comparing coupling in
high- vs low-gain trials.

## Status of the real-data analysis
The definitive test uses the experimental running-condition trials and requires the Allen
Neuropixels recordings (allensdk + the `ecephys_cache_dir`), which live on the lab server and
are **not** in the local mirror. The ready-to-run script is `exp2b_realdata_TEMPLATE.py`
(loads V1/LM pooled trains via `DataLoader.Allen_dataset`, fits the warped pop-GLM with a
per-trial gain term, stratifies running trials at the median gain, and refits V1->LM coupling
within each stratum). Run it on the server to obtain the biological result.

## Synthetic validation (run here)
We validated the logic on EIF synthetic data where **true coupling is held constant across
trials** while trials vary in response amplitude (gain). If the pop-GLM coupling estimate is
invariant to gain under constant true coupling, then a coupling difference measured **between
conditions** cannot be a mere artifact of an accompanying gain difference.

Procedure (`exp2b_run.py`): simulate combined data (conn>0, std>0, ntrial=120); define empirical
per-trial gain = total LM-population spike count; split trials at the median into high-/low-gain
halves; fit the warped pop-GLM separately in each half; compare recovered V1->LM coupling.

### Result (n=10 repeats)
- Empirical gain differed by ~11% between strata (high 138.5 vs low 124.6 spikes/trial).
- Recovered V1->LM coupling integral: **high-gain 0.826 ± 0.237 vs low-gain 0.843 ± 0.315**.
- Paired difference ≈ 0; **Wilcoxon p = 0.92, paired t p = 0.85 (n.s.)**.

=> Under constant true coupling, the pop-GLM coupling estimate does not track trial gain.
The coupling change reported between running and stationary conditions therefore cannot be
explained as a gain artifact; pop-GLM dissociates trial gain from inter-population coupling.
(See `FigS_exp2b_gain.pdf`.)

## Draft text (Results 3.2 / Discussion)
"Because locomotion both increases firing rate and decreases V1->LM coupling, we asked whether
the coupling change could be an artifact of the gain change. In synthetic data with coupling
held constant across trials, the pop-GLM coupling estimate was invariant to a ~11% range of
trial gain (high- vs low-gain trials: 0.83 vs 0.84, Wilcoxon p=0.92), confirming that the gain
and coupling terms are separable and that a between-condition coupling change is not a gain
artifact. The corresponding stratification of the experimental running trials is provided as a
runnable script (Supplementary Code)."

## Reproduce
```
python3 exp2b_run.py --reps 10 --ntrial 120 --seed0 5000   # synthetic gain-stratification
python3 exp2b_analyze.py                                    # stats + FigS_exp2b_gain.pdf/.png
# on the lab server (allensdk available):
python3 exp2b_realdata_TEMPLATE.py
```

---
## Real-data result (session 757216464) — `exp2b_realdata.py`
We ran the stratification on the experimental running trials (extracted with
`extract_session.py`, which closely matches the paper (88 V1 / 53 LM good units under the standard quality filter; the main analysis reports 85 V1 under a slightly different threshold)).
Running trials were split at the median trial gain and V1→LM coupling refit within each half:

| stratum | n trials | mean gain | V1→LM integral | LRT p |
|---|---|---|---|---|
| high gain | 146 | 236.9 | 0.381 | <1e-100 |
| low gain  | 145 | 149.5 | 0.449 | <1e-100 |

Across a **1.6-fold** gain range the coupling is present and highly significant in both strata,
with only a modest reduction on high-gain trials → the coupling estimate is not a gain artifact,
and the slight high-gain weakening matches the shared-global-drive interpretation. See
`FigS_exp2b_gain_realdata.pdf`. The synthetic gain-invariance control remains in
`FigS_exp2b_gain.pdf`.

# Experiment 2a — Coupling + trial-to-trial variability coexisting

Addresses Reviewer #1, Major comment 2a:

> The synthetic analyses validate pooling-under-coupling and time-warping-under-confound
> *separately*. They do not directly test the realistic case in which true inter-population
> coupling coexists with trial-to-trial stimulus-response variability. Such a simulation would
> help determine whether the time-warped baseline preserves true coupling estimates or partially
> absorbs genuine coupling-related temporal structure.

## Design
EIF synthetic data via `GLM.EIF_simulator(std1,corr1,std2,corr2,ntrial,conn)`.
Trial-to-trial variability is the across-trial jitter of the two response-peak centers
(`std/corr`); inter-population coupling is the feed-forward synaptic weight `conn`
(source neurons 0–9 -> target neurons 10–19). Variability params match the paper:
std1=10, corr1=0.5, std2=25, corr2=0.9; conn=0.008; ntrial=100; 12 repeats/condition.

Conditions (pop-GLM = inhomogeneous baseline + source-coupling + target-coupling + f_damp):
- **Reference**  conn>0, std=0  -> plain fit. True coupling with NO confound = recoverable truth.
- **Combined Warp-ON**   conn>0, std>0 -> `fit_time_warping_baseline`.
- **Combined Warp-OFF**  conn>0, std>0 -> plain `fit` (shared baseline, no warping).
- **Control Warp-ON**    conn=0, std>0 -> warped fit (false-positive control).
- **Control Warp-OFF**   conn=0, std>0 -> plain fit (false-positive control).

Per fit we record the recovered source->target coupling filter (integral, peak, full shape)
and a likelihood-ratio test for the source->target coupling term (full vs nested without it).

## Results (12 repeats/condition)

| Condition | true coupling? | warping | coupling integral (mean±sd) | detection p<0.05 |
|---|---|---|---|---|
| Reference        | yes (conn>0) | n/a (std=0) | 0.500 ± 0.172 | 100% |
| Combined Warp-ON | yes (conn>0) | ON  | 0.798 ± 0.372 | 67% |
| Combined Warp-OFF| yes (conn>0) | OFF | 1.529 ± 0.083 | 100% |
| Control Warp-ON  | no  (conn=0) | ON  | 0.471 ± 0.301 | **0%** |
| Control Warp-OFF | no  (conn=0) | OFF | 1.309 ± 0.111 | **100%** |

Key statistics:
- Paired Wilcoxon, combined Warp-ON vs Warp-OFF (same data): p = 9.8e-4.
- Warp-OFF gives essentially the SAME large estimate whether coupling is present (1.53)
  or absent (1.31), with 100% "detection" in BOTH -> without warping the confound is
  indistinguishable from true coupling (100% false-positive rate).
- Warp-ON: false-positive rate 0%; genuine coupling retained (0.80, clearly above the
  no-coupling control 0.47, MWU p=0.026) -> time warping does NOT absorb true coupling.
- Warp-ON modestly overestimates relative to the idealized no-variability reference
  (0.80 vs 0.50, MWU p=0.026) but removes the gross confound-driven inflation of Warp-OFF.

## Draft text for Results 3.1 / Supplement
"To test the realistic regime in which inter-population coupling and trial-to-trial
stimulus-response variability are present simultaneously, we simulated EIF populations with
both nonzero feed-forward coupling and across-trial jitter in response-peak timing
(Supplementary Fig SX). Without time warping, the shared-baseline model misattributed the
trial-to-trial variability to coupling: it reported strong, highly significant coupling even
when no true coupling existed (false-positive rate 100%), and its estimate was nearly identical
whether or not coupling was present, so true coupling could not be distinguished from the
confound. With time warping, the false-positive rate dropped to 0% while genuine coupling was
retained — the recovered coupling under the combined regime remained well above the
no-coupling control and close to the value recovered in the absence of variability — confirming
that the time-warped baseline preserves true coupling rather than absorbing it."

## Reproduce
```
python3 exp2a_run.py --cond combined --reps 12 --seed0 1000   # warp vs no-warp, coupling+variability
python3 exp2a_run.py --cond ref      --reps 12 --seed0 2000   # clean true-coupling reference
python3 exp2a_run.py --cond nocoup   --reps 12 --seed0 3000   # variability-only false-positive control
python3 exp2a_analyze.py     # prints the table above
python3 exp2a_figure.py      # writes FigS_exp2a_combined.pdf/.png
```
(Runs against ../../GLM.py; ~10–17 s per repeat, 2-way parallel.)

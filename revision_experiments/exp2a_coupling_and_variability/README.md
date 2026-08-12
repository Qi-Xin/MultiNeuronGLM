# Experiment 2a — Coupling *and* trial-to-trial baseline variation both present

**Start here: `exp2a_coupling_and_variability.ipynb`** (annotated, outputs saved).

The paper's synthetic study (`STAR Synthetic dataset; Pillow et al.ipynb`) already covers the two
one-at-a-time cases: **Sensitivity** (coupling, no variation → filter should be nonzero) and
**Time warping** (no coupling, variation → filter should be zero). Reviewer 2a asks for the
combined case.

Hyperparameters exactly as in that notebook's Sensitivity section: EIF, `conn=0.0075`, `ntrial=100`,
baseline `num=20`, coupling `num=3, peaks_max=15, nonlinear=1`, `refractory_additive(tau=10,num=4)`,
`penalty=1e0`, warping `max_iter=5, warp_interval=[[0,0.15],[0.15,0.35]]`. Baseline variation uses the
Time-warping settings `std1=10, corr1=0.5, std2=25, corr2=0.9`.

## Result (6 simulations; single-neuron = 10 targets × 10 sources = 100 filters per simulation)

| model | coupling-filter integral |
|---|---|
| reference: coupling, **no** variation (Sensitivity) | 0.750 ± 0.171 |
| **pop-GLM (time warping)** | **0.737 ± 0.115** — nonzero, detected in 5/6 sims (LRT p<0.05) |
| **pop-GLM without time warping** | **1.535 ± 0.080** — inflated (2.08×) |
| single-neuron GLM (600 filters) | 1.533 ± **1.386** — noisy **and** inflated |

1. pop-GLM with time warping **recovers the true coupling** (matches the no-variation reference) — the
   time-warped baseline does not absorb genuine coupling.
2. Dropping the time-warping component **inflates** the filter ~2.1×: the unmodelled trial-to-trial
   variation is misattributed to coupling.
3. Single-neuron GLMs are **noisy and inflated** — their mean lies on the *no-warp* curve, not on
   pop-GLM. A single neuron carries too little information to estimate per-trial shifts, so a
   single-neuron GLM cannot implement time warping and inherits the same confound bias, while also
   being far more variable (sd 1.39 vs 0.08). Pooling is what makes the correction possible.

The figure (`FigS_exp2a_scenario3.pdf`, from `exp2a_ci_fig.py`) follows the paper's synthetic-figure
style (`utils.use_pdf_plot`): a **representative simulation** with **95% confidence intervals**. The
no-warp filter's CI lies **above** the time-warped one (inflated), and single-neuron filters are noisy
and inflated. The panels are drawn by `panel_pop(ax, D)` / `panel_single(ax, D)`, so they can be
dropped into a subplot of the main synthetic figure later. (`exp2a_s3.pkl` holds all 6 simulations for
the across-simulation summary table.)

## Files
`exp2a_s3.py` (simulate + fit all three models), `exp2a_s3_fig.py` (figure), `exp2a_s3.pkl` (results).

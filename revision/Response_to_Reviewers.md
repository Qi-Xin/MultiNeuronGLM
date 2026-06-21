# Response to Reviewers — PCOMPBIOL-D-26-00465

**A Population Coupling Model Identifies Reduced Propagation from V1 to Higher Visual Areas During Locomotion**

We thank the Academic Editor and Reviewer #1 for their careful and constructive assessment. We were glad that the Reviewer found the manuscript "scientifically interesting and largely convincing" and the two modifications "well motivated and supported by the results." We have addressed every point below. Reviewer and editor comments are in *italics*; our responses follow. Manuscript locations refer to the revised manuscript; a list of all changes is in `CHANGES.md`, and a tracked-changes version can be produced with `latexdiff` (see note at the end).

---

## Editor — Journal Requirements

**1. CRediT author contributions.**
We have completed the CRediT contributions for all four authors (Q. Xin, K. N. Urban, J. H. Siegle, R. E. Kass) in the submission form. *(Action completed in Editorial Manager by the corresponding author.)*

**2. Manuscript source file (.tex).**
We now provide the LaTeX source (`manuscript_revised.tex`) under the "LaTeX Source File" item type, with the compiled PDF as the "Manuscript."

**3. Author Summary (150–200 words).**
An Author Summary is included between the Abstract and the Introduction. It has been expanded to 179 words and rewritten for a broad audience.

**4. Ethics Statement.**
This study analyzes the publicly available Allen Brain Observatory – Neuropixels Visual Coding dataset; no new animal experiments were performed. We have added an **Ethics Statement** at the start of the Materials and Methods stating that all in vivo procedures underlying the dataset were performed by the Allen Institute under protocols approved by its IACUC. *Author action: please confirm/insert the exact Allen Institute IACUC protocol number if the journal requires it.*

**5. Figures as separate .tif/.eps.**
We will upload each main figure as a separate TIFF/EPS file processed through the PLOS NAAS tool. *(Figure export is in progress; the figure sources are in the Overleaf project.)*

**6. Supporting Information separated from the manuscript.**
The supplement has been reorganized (see Reviewer Minor 1c) into **Supplementary Results** and **Supplementary Figures and Tables**; we will upload these as separate "Supporting Information" files, each with a legend after the reference list.

**7. Figure 1B copyright / clip-art.**
*Author action required: please confirm whether the icons in Figure 1B were drawn by the authors. If not, we will replace them with open-source equivalents (Wikimedia Commons / OpenClipart) or obtain permission.*

**8. Financial Disclosure.**
Revised to full-sentence form with per-author initials and funder role: "Q.X., K.U., and R.E.K. were supported by National Institutes of Health grant R01 MH064537 (to R.E.K.). The funders had no role in study design, data collection and analysis, decision to publish, or preparation of the manuscript." *(Also to be entered in the submission form.)*

**9. Competing Interests.**
Updated to the standard wording: "The authors have declared that no competing interests exist." (Added to the manuscript and to be set in the submission form.)

**10. Data Availability.**
The Data and Code Availability section gives the full links: the code at https://github.com/Qi-Xin/MultiNeuronGLM (now with a structured README and two minimal working examples) and the dataset at the Allen Institute. *Author action: confirm the final GitHub release/commit or DOI to cite.*

---

## Reviewer #1 — Major issue 1 (Reproducibility and implementation)

**1a. *Minimal working examples, README, annotated notebooks.***
We agree this is essential. The repository now contains:
- a structured top-level `README.md` describing the repository layout, environment/dependencies, and how to run each component;
- **two minimal working examples** under `examples/`: (i) `mwe_synthetic.py` — generate EIF neurons and fit pop-GLM end-to-end on synthetic data; (ii) `mwe_allen.py` — fit pop-GLM to the Allen dataset (annotated, with the data-loading steps);
- a `revision_experiments/` directory containing self-contained, documented scripts for all new analyses (below), each with its own README and a one-command reproduce recipe.
The minimal reproduction path is now clearly separated from the exploratory notebooks.

**1b. *Step-by-step fitting/training algorithm.***
We have added a new Materials and Methods subsection **"Fitting and training procedure"** and a schematic algorithm box (**Algorithm: Fitting pop-GLM**), parallel to the excursion algorithm. It specifies: initialization of the baseline template from the trial-averaged PSTH; initialization of the warping (identity) and gains (zero); the penalized-maximum-likelihood (Newton–Raphson) regression step that jointly estimates all basis coefficients and trial gains (baseline unpenalized); the greedy per-trial warping-landmark update via line search with exponential-moving-average smoothing; how `f_damp` is updated within the regression step; the alternation between the warping and regression steps; and the stopping criterion (relative change in negative log-likelihood below tolerance, or maximum iterations).

**1c. *Promote `f_damp` into the main text; reference Fig S4.***
In Section 2.3/2.4 we now state explicitly that `f_damp` is represented as a weighted sum of basis functions of the recent population spike density Λ(t), parameterized with four raised-cosine basis functions, and we reference the basis figure (Supplementary Fig, "basis" panel C) directly when introducing this parameterization.

---

## Reviewer #1 — Minor issue 1 (Organization)

**1a. *Move the selected-vs-full-population comparison out of Methods.***
Done. The Methods now retain only the neuron-selection method and point to the new **Supplementary Results** subsection "Selected versus full recorded population," where the comparison is presented.

**1b. *Move Table S1 and the running/stationary model selection to Supplementary Results.***
Done. The "Model selection for running versus stationary classification" prose and Table S1 (speed model selection) now appear together in **Supplementary Results**.

**1c. *Restructure the supplement; order by main-text logic.***
The supplement is now split into **Supplementary Results** and **Supplementary Figures and Tables**, with an opening note that it follows the main-text logic (model design → synthetic validation → biological application). Subsections are named by function.

**1d. *Group the two damping-motivating analyses.***
The analytical result (lack of refractory effects in linear self-history filters) and the self-history model selection are now grouped under a single **"Model design: why population spike trains require a nonlinear damping term"** subsection, since they jointly motivate the damping correction.

**1e. *Move multiple-comparisons correction to a main-text "Statistical testing" section; keep the algorithm in the supplement.***
We added a main-text Materials and Methods subsection **"Statistical testing"** that briefly describes the likelihood-ratio tests (synthetic data), the excursion/permutation tests (experimental data), and the multiple-comparisons correction (Bonferroni plus the exact permutation step-down procedure). The step-down algorithm (Algorithm: Testing Procedure) remains in the supplement and is referenced from the main text.

---

## Reviewer #1 — Other suggestions

**2a. *Test coupling and trial-to-trial variability coexisting.***
We added this experiment (new **Scenario 3** in Results 3.1 and a new supplementary figure). We simulated EIF populations with **both** nonzero feed-forward coupling **and** across-trial jitter in the two response-peak times (12 repeats/condition), and fit pop-GLM with and without time warping.
- Without time warping, the model misattributed the variability to coupling: a **100% false-positive rate** when no true coupling existed, and a nearly identical (inflated) estimate whether or not coupling was present — i.e., true coupling could not be distinguished from the confound.
- With time warping, the **false-positive rate fell to 0%** while genuine coupling was retained: the recovered coupling stayed well above the no-coupling control and close to the no-variability reference.
This directly shows the time-warped baseline **preserves** true coupling rather than absorbing it. (Code: `revision_experiments/exp2a_combined_coupling_variability/`.)

**2b. *Test whether coupling changes are explained by trial gain.***
We performed this test on the experimental data (session 757216464). We stratified the running trials by their gain (total LM-population spike count per trial) and refit the V1→LM coupling within each half. Across a **1.6-fold** range of trial gain, the coupling remained present and highly significant in **both** strata (high-gain integral 0.38, low-gain integral 0.45, both p < 10^-100; new supplementary figure), with only a modest reduction on high-gain trials. The coupling change is therefore not an artifact of the gain change, and the slight high-gain weakening is consistent with locomotion introducing shared global drive rather than abolishing feed-forward propagation. As a control, in synthetic data with coupling held constant across trials the estimate was gain-invariant (paired Wilcoxon p = 0.92). (Code: `revision_experiments/exp2b_gain_vs_coupling/`, incl. `exp2b_realdata.py`.)

**2c. *Quantitative comparison for Figs S6 and S7 instead of visual inspection.***
We computed the requested metrics (filter integral, peak amplitude, peak latency, correlation) on the experimental data (session 757216464). **S6 (full vs selected population):** the V1→LM coupling filters from the selected strongly-evoked population (18 V1, 11 LM) and the full recorded population (all quality-passing units) are highly correlated (**r = 0.96**, matched latency); the full population shows a lower amplitude, as expected from dilution by weakly tuned neurons. **S7 (self-history treatment):** the coupling filters from the main f_damp model and a linear-self-history model are nearly identical (**r = 0.998**). Both confirm the V1–LM result is preserved. A new supplementary figure shows both overlays; the reusable tool is `exp2c_quantify.py`. (Code: `revision_experiments/exp2c_quantitative_S6_S7/`, incl. `exp2c_realdata.py`.)

**2d. *Discuss generalizability (heterogeneous populations, spontaneous activity, weakly evoked responses, tasks without repeated templates).***
We added a Discussion paragraph reframing the scope from limitation to generalizability, discussing each regime explicitly: heterogeneous populations (clustering into sub-populations or multiple template components), spontaneous activity and tasks without a repeated template (more flexible data-driven alignment or latent-variable models), and weakly evoked responses (pooling still helps but the time-warping benefit shrinks without a well-defined peak). We note that the core idea — coupling between pooled population spike trains with a damped self-history term — remains valid in these regimes, with the baseline/alignment components generalized.

---

## Reviewer #1 — Writing-related adjustments

**3.1. *Intro schematic equation.***
The schematic now first presents the generic single-neuron GLM as `log firing rate = stimulus effect + self-history effect with damping + coupling effects`, and then, after introducing the population modifications, the population form `log population firing rate = trial-specific baseline + self-history effect with damping + population coupling effects`, matching the text.

**3.2. *Sentence at p2, line 42.***
Rewritten as: "In sensory cortical areas, subsets of neurons can often be identified that exhibit similar stimulus-evoked spiking patterns, allowing them to be treated as functionally defined populations."

**3.3. *Introduce BIC properly.***
BIC is now defined at first use in the new "Statistical testing" subsection: BIC = −2 log L̂ + k log n, with L̂ the maximized likelihood, k the number of parameters, and n the number of observations; reported values are improvements relative to a baseline model.

---

## Note on the tracked-changes file
A clean compiled PDF is provided. The PLOS "Revised Manuscript with Track Changes" can be generated in one step from the two provided source files:
`latexdiff manuscript_original_backup.tex manuscript_revised.tex > manuscript_trackchanges.tex` (e.g., via Overleaf's built-in latexdiff). A complete itemized change log is in `CHANGES.md`.

## Items needing author confirmation before submission
- Allen Institute IACUC protocol number (Requirement 4).
- Figure 1B clip-art provenance (Requirement 7).
- Final GitHub release/commit or data DOI to cite (Requirement 10).
- Export main figures to TIFF/EPS via NAAS (Requirement 5).

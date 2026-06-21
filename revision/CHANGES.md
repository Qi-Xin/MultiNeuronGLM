# Change log — revised manuscript

Maps each change to the reviewer/editor point it addresses. Source files:
`manuscript_original_backup.tex` (as submitted) and `manuscript_revised.tex` (revised).
Generate a tracked-changes PDF with:
`latexdiff manuscript_original_backup.tex manuscript_revised.tex > manuscript_trackchanges.tex`
(e.g., on Overleaf).

## Main text — additions
- **Author Summary** rewritten and expanded to 179 words (Editor 3).
- **Section 2.1 (Overview):** schematic equations restructured — generic single-neuron GLM
  (`stimulus + self-history with damping + coupling`) then the population form
  (`trial-specific baseline + self-history with damping + population coupling`) (Writing 3.1).
- **Section 2.3:** `f_damp` made explicit as a weighted sum of four raised-cosine basis
  functions of the recent population spike density Λ(t), with a direct reference to the basis
  figure (Major 1c).
- **Introduction:** sentence rewritten to "In sensory cortical areas, subsets of neurons can
  often be identified that exhibit similar stimulus-evoked spiking patterns, allowing them to be
  treated as functionally defined populations." (Writing 3.2).
- **Results 3.1:** new **Scenario 3** paragraph (coupling + trial-to-trial variability coexisting)
  with new supplementary figure (Major/Other 2a).
- **Methods:** new **"Ethics statement"** subsection (Editor 4).
- **Methods:** new **"Fitting and training procedure"** subsection + **Algorithm: Fitting
  pop-GLM** box (Major 1b).
- **Methods:** new **"Statistical testing"** subsection — LRT/excursion/permutation tests,
  multiple-comparisons correction (moved from supplement), and the BIC definition (Minor 1e,
  Writing 3.3).
- **Discussion:** new generalizability paragraph (heterogeneous populations, spontaneous
  activity, weakly evoked responses, tasks without repeated templates) (Other 2d).
- **Discussion:** new paragraph on gain vs. coupling with new supplementary figure (Other 2b).
- **Funding** sentence rewritten with per-author initials + funder role (Editor 8); new
  **Competing interests** statement (Editor 9).

## Main text — moved out (to Supplementary Results)
- Selected-vs-full-population comparison moved from Methods to Supplementary Results;
  a pointer remains in Methods (Minor 1a).
- Running/stationary model-selection prose + Table S1 moved from Methods to Supplementary
  Results (Minor 1b).

## Supplement — restructured
- Split into **Supplementary Results** and **Supplementary Figures and Tables**, ordered by the
  main-text logic (model design → synthetic validation → biological application) (Minor 1c).
- "Lack of refractory effects" + "Model selection for self-history effects" grouped under a
  single "Model design" subsection (Minor 1d).
- Multiple-comparisons prose moved to the main text; **Algorithm: Testing Procedure** remains in
  the supplement, referenced from the main text (Minor 1e).
- S6 (full population) and S7 (alternative self-history) descriptions augmented with quantitative
  measures (integral, peak amplitude, latency, correlation) (Other 2c).
- Two new supplementary figures added: combined coupling+variability (2a) and gain invariance (2b).

## New cross-references / labels added
`alg: fitting`, `sec:fitting`, `sec:statistical testing`, `sec:supp full vs selected`,
`fig: synthetic combined`, `fig: gain invariance`, `sec:Results with experimental data`.

## Verification
Revised manuscript compiles cleanly (pdflatex + bibtex), 42 pages, 0 undefined references,
0 errors. A draft-figure PDF (`manuscript_revised_draftfigures.pdf`) is included; compile with
the real figures on Overleaf for the final PDF.

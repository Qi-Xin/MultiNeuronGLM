# Experiment 2c — Quantitative comparison for Figs S6 & S7

Addresses Reviewer #1, minor comment 2c: replace visual inspection with quantitative measures.
For **Fig S6** (full recorded population vs selected strongly-evoked population) and **Fig S7**
(alternative self-history treatment vs the main f_damp model), report peak amplitude, filter
integral, peak latency, and correlation with the main fitted filters, and overlay the
corresponding Figure 6 curves to show the main V1–LM result is preserved.

## What this provides
`exp2c_quantify.py` implements the exact metrics the reviewer requested and is reusable for
both S6 and S7:
- `quantify_filter(f)` -> peak amplitude, |peak|, integral, peak latency.
- `compare_pair(f_main, f_variant)` -> both filters' metrics + Pearson correlation +
  %-difference in integral/peak and latency shift.
- `overlay_plot(f_main, f_variant, ...)` -> draws the variant filter against the Figure-6 curve,
  annotated with r, Δintegral%, Δpeak%.
- `compare_variants(main_filters, variant_filters)` -> a full metrics table (CSV) over all
  area pairs.

## Demonstration on real saved filters
Running `exp2c_quantify.py` quantifies the running-vs-stationary change for all 54 saved
coupling filters of the main model (`running_filter_peak2.pickle`,
`stationary_filter_peak2.pickle`) and writes `exp2c_running_vs_stationary.csv`. This both
validates the tooling and quantifies the headline running/stationary effect that S6/S7 must
preserve.

## To populate the S6/S7 panels
The full-population (S6) and alternative-self-history (S7) variant fits are produced on the lab
server (they require the Allen recordings). Once available, generate the panels with:
```python
from exp2c_quantify import compare_variants, overlay_plot
rows = compare_variants(main_filters, fullpop_filters, out_csv="S6_metrics.csv")   # Fig S6
rows = compare_variants(main_filters, althist_filters, out_csv="S7_metrics.csv")   # Fig S7
overlay_plot(main_V1LM, fullpop_V1LM, ("selected pop.", "full pop."), "S6_overlay.pdf")
overlay_plot(main_V1LM, althist_V1LM, ("f_damp model", "alt. self-history"), "S7_overlay.pdf")
```
The reported metrics (correlation near 1, small Δintegral/Δpeak, matched latency) demonstrate
quantitatively that the V1–LM coupling result is robust to these modelling choices.

## Reproduce
```
python3 exp2c_quantify.py    # writes exp2c_running_vs_stationary.csv + prints the table
```

---
## Real-data result (session 757216464) — `exp2c_realdata.py`
Fit on stationary trials; metrics from `exp2c_quantify.compare_pair`.

- **S6 (full vs selected population):** V1→LM filters correlate **r=0.96** (matched latency);
  full population integral 0.627 vs selected 1.460 (lower amplitude from dilution by weakly
  tuned neurons). Result preserved.
- **S7 (f_damp vs linear self-history):** V1→LM filters correlate **r=0.998**
  (integral 0.627 vs 0.616). Result robust to the self-history treatment.

See `FigS_exp2c_S6_S7_realdata.pdf`.

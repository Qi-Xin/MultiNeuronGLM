"""Exp 2c (Reviewer minor comment 2c): replace visual inspection of Figs S6 & S7 with
quantitative comparison. The reviewer asks, for S6 (full-population vs selected strongly-evoked
population) and S7 (alternative self-history vs the main f_damp model), to report quantitative
measures -- peak amplitude, filter integral, peak latency, correlation with the main fitted
filters -- and to overlay the corresponding Figure 6 curves.

This module provides the metric functions + an overlay plotter, reusable for both S6 and S7.
`compare_variants()` takes the main (Figure 6) filter and a variant filter and returns the table.
A demonstration on the saved main-model running/stationary filters is run in __main__.
The variant fits (full-population / alt-self-history) are produced on the lab server; point
`compare_variants` at those arrays to populate the S6/S7 quantitative panels.
"""
import os, pickle, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

def quantify_filter(f, dt=1.0):
    f = np.asarray(f, dtype=float).ravel()
    i = int(np.argmax(np.abs(f)))
    return {"peak_amp": float(f[i]),               # signed amplitude at the dominant lag
            "peak_abs": float(np.abs(f[i])),
            "integral": float(np.sum(f) * dt),      # area under the filter
            "peak_latency": float(i * dt)}          # lag (ms) of the dominant peak

def compare_pair(f_main, f_variant, dt=1.0, names=("main", "variant")):
    f_main = np.asarray(f_main, float).ravel(); f_variant = np.asarray(f_variant, float).ravel()
    n = min(len(f_main), len(f_variant)); a, b = f_main[:n], f_variant[:n]
    qm, qv = quantify_filter(a, dt), quantify_filter(b, dt)
    corr = float(np.corrcoef(a, b)[0, 1])
    def pct(x, y): return float("nan") if x == 0 else 100.0 * (y - x) / abs(x)
    return {names[0]: qm, names[1]: qv,
            "correlation": corr,
            "pct_diff_integral": pct(qm["integral"], qv["integral"]),
            "pct_diff_peak": pct(qm["peak_amp"], qv["peak_amp"]),
            "diff_peak_latency_ms": qv["peak_latency"] - qm["peak_latency"]}

def overlay_plot(f_main, f_variant, labels, out, title="", dt=1.0):
    a = np.asarray(f_main, float).ravel(); b = np.asarray(f_variant, float).ravel()
    t = np.arange(len(a)) * dt
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.axhline(0, color="k", lw=.6)
    ax.plot(t, a, color="#222", lw=2.2, label="%s (Fig 6)" % labels[0])
    ax.plot(np.arange(len(b)) * dt, b, color="#d62728", lw=2, ls="--", label=labels[1])
    c = compare_pair(a, b, dt)
    ax.set_xlabel("lag (ms)"); ax.set_ylabel("coupling filter (log rate)")
    ax.set_title(title + ("\nr=%.3f  Δintegral=%.1f%%  Δpeak=%.1f%%"
                 % (c["correlation"], c["pct_diff_integral"], c["pct_diff_peak"])), fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight", dpi=200)
    return c

def compare_variants(main_filters, variant_filters, key_names=None, dt=1.0, out_csv=None):
    """main_filters/variant_filters: dict key->1d filter (or (filter,ci))-> uses [0].
    Returns list of metric dicts; optionally writes a CSV."""
    rows = []
    for k in main_filters:
        if k not in variant_filters: continue
        fm = np.asarray(main_filters[k]); fm = fm[0] if fm.ndim == 2 else fm
        fv = np.asarray(variant_filters[k]); fv = fv[0] if fv.ndim == 2 else fv
        c = compare_pair(fm, fv, dt)
        rows.append({"key": key_names.get(k, str(k)) if key_names else str(k),
                     "main_integral": c["main"]["integral"], "var_integral": c["variant"]["integral"],
                     "pct_diff_integral": c["pct_diff_integral"],
                     "main_peak": c["main"]["peak_amp"], "var_peak": c["variant"]["peak_amp"],
                     "pct_diff_peak": c["pct_diff_peak"],
                     "main_latency": c["main"]["peak_latency"], "var_latency": c["variant"]["peak_latency"],
                     "correlation": c["correlation"]})
    if out_csv:
        import csv
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    return rows

# ---------------- demonstration on saved main-model filters ----------------
if __name__ == "__main__":
    REPO = "/sessions/gifted-cool-babbage/mnt/MultiNeuronGLM"
    run = pickle.load(open(os.path.join(REPO, "running_filter_peak2.pickle"), "rb"))
    sta = pickle.load(open(os.path.join(REPO, "stationary_filter_peak2.pickle"), "rb"))
    # demo: quantify running-vs-stationary change for every saved coupling filter
    rows = compare_variants(sta, run, dt=1.0,
                            out_csv="/sessions/gifted-cool-babbage/mnt/outputs/exp2c_running_vs_stationary.csv")
    print("Quantified %d coupling filters (stationary=main vs running=variant)." % len(rows))
    # show the filters with the largest running-induced integral change
    rows_sorted = sorted(rows, key=lambda r: abs(r["pct_diff_integral"]), reverse=True)
    print("\n%-10s %10s %10s %9s %7s" % ("key", "sta_int", "run_int", "d_int%", "corr"))
    for r in rows_sorted[:8]:
        print("%-10s %10.3f %10.3f %8.1f%% %7.3f"
              % (r["key"], r["main_integral"], r["var_integral"], r["pct_diff_integral"], r["correlation"]))
    print("\nWrote exp2c_running_vs_stationary.csv (full table).")
    print("For S6/S7: call compare_variants(main_filters, variant_filters) with the full-population")
    print("and alt-self-history fits, and overlay_plot() to add the Fig-6 curve to each panel.")

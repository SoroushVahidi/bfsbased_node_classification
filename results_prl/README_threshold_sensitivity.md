# Threshold sensitivity — evidence package

## What exists in-repo (audit)

- **No** dedicated `exp02_threshold_sensitivity/` or precomputed threshold-sweep CSV was found under `node_classification_manuscript/`.
- Final diagnostics JSONs (`logs/*_selective_graph_correction_*.json`) store **one** selected threshold per run; they **omit** the full `selection_trace` (per-candidate curves) from disk.
- The implementation in `bfsbased-full-investigate-homophil.py` evaluates a **candidate threshold list** on **validation** only, then applies the winner to test; **per-threshold test curves are not exported** in saved JSON.

## What was generated here (cross-run empirical relationship)

These artifacts are **not** a controlled sweep of threshold at fixed data realization. They summarize **across the 30 SGC runs per dataset** (10 splits × 3 repeats) how **validation-selected threshold** co-varies with **test uncertain fraction** and **Δ(SGC−MLP)**.

Files:
- `threshold_sensitivity_crossruns.jsonl` — one row per SGC run with threshold, uncertain/changed fractions, accuracies, delta vs MLP (matched split/repeat).
- `threshold_uncertainty_crossrun_summary.csv` — per-dataset means/std and Pearson corr(threshold, uncertain_fraction).
- `threshold_sensitivity_mini_table_tertiles.csv` — per dataset, runs split into **low/mid/high threshold tertiles** (descriptive only).
- Figures: `fig_threshold_vs_uncertain_fraction_crossruns.{png,pdf}`, `fig_threshold_vs_delta_crossruns.{png,pdf}`, `fig_threshold_sensitivity_panels_crossruns.{png,pdf}`.

## Caveat for PRL

For a **true** threshold-sensitivity subsection (fixed split, sweep τ on validation, report test), you would need either:
- instrumenting `choose_uncertainty_threshold` to log **per-τ** val/test metrics to disk, or  
- a lightweight re-run script — **not** included here.

The cross-run plots still support **narrow** claims about **gating**: higher selected thresholds tend to associate with different uncertain fractions across datasets; they do **not** prove monotonicity of test accuracy in τ for a single split.

Source JSONL: `logs/comparison_runs_manuscript_final_validation.jsonl`.

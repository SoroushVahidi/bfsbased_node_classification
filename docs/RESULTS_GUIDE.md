# Results Guide

This document helps reviewers quickly map files to claims.

## Canonical main results

- Frozen per-split source: `reports/final_method_v3_results.csv`
- Main benchmark table: `tables/main_results_selective_correction.md`
- Machine-readable main table: `tables/main_results_selective_correction.csv`

## Canonical supporting analyses

- Method narrative: `reports/final_method_v3_analysis.md`
- Safety analysis: `reports/safety_analysis.md`
- Regime analysis: `reports/final_v3_regime_analysis.md`

## Canonical derived tables

- Setup: `tables/experimental_setup_selective_correction.md`
- Ablation: `tables/ablation_selective_correction.md`
- Sensitivity: `tables/sensitivity_selective_correction.md`

## Canonical figures

- `figures/graphical_abstract_selective_correction_v3.png`
- `figures/correction_rate_vs_homophily.png`
- `figures/safety_comparison.png`
- `figures/reliability_vs_accuracy.png`

## Archived / exploratory references

Anything under `archive/` is preserved for provenance and historical context.
It is explicitly non-canonical unless otherwise noted in a local README.

## Fresh ablation rerun (canonical-candidate)

- Full rerun output (6 datasets × 10 splits): `reports/final_v3_ablation_rerun_results.csv`
- Summary table: `tables/ablation_selective_correction_rerun.md`
- Machine-readable summary: `tables/ablation_selective_correction_rerun.csv`
- Comparison note vs historical mixed ablation table: `reports/final_v3_ablation_rerun_analysis.md`

These files are intentionally separate from frozen canonical artifacts and are the recommended evidence when citing a current-code ablation rerun.


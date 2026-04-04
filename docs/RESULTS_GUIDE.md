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

## Expert Systems lightweight strengthening bundle (non-canonical, manuscript-supporting)

These artifacts are a reduced-scope diagnostic package (tagged `20260404_reduced`) that
adds correction-behavior, bounded-intervention safety, and local gate-sensitivity evidence.
They are **not** replacements for frozen canonical files.

- Audit memo: `reports/expert_systems_canonical_audit_20260404_reduced.md`
- Lightweight ablation source: `reports/final_v3_expert_systems_lightweight_ablation_20260404_reduced.csv`
- Correction diagnostics source: `reports/final_v3_correction_diagnostics_20260404_reduced.csv`
- Gate sensitivity source: `reports/final_v3_expert_systems_gate_sensitivity_20260404_reduced.csv`
- Correction diagnostics table: `tables/final_v3_correction_diagnostics_20260404_reduced.md`
- Bounded intervention table: `tables/safety_bounded_intervention_expert_systems_lightweight_20260404_reduced.md`
- Gate sensitivity table: `tables/sensitivity_selective_correction_expert_systems_lightweight_20260404_reduced.md`

## Full-scope appendix support bundle (canonical-candidate)

This full-scope package extends the same diagnostics to the canonical six datasets and
all ten fixed splits. It is intended for appendix/supplementary support and remains
separate from frozen canonical files.

- Gate sensitivity narrative: `reports/gate_sensitivity_fullscope_20260404.md`
- Gate sensitivity table (CSV): `tables/gate_sensitivity_fullscope_20260404.csv`
- Gate sensitivity table (Markdown): `tables/gate_sensitivity_fullscope_20260404.md`
- Bounded-intervention narrative: `reports/bounded_intervention_fullscope_20260404.md`
- Bounded-intervention table (CSV): `tables/bounded_intervention_fullscope_20260404.csv`
- Bounded-intervention table (Markdown): `tables/bounded_intervention_fullscope_20260404.md`

For historical continuity, keep the reduced `20260404_reduced` bundle, but cite this
full-scope bundle when addressing the prior scope caveat.

## Correction explanation audit (non-canonical, manuscript-supporting)

Lightweight node-level interpretability bundle for FINAL_V3 correction decisions (changed nodes only).

- Node-level export: `reports/correction_explanation_nodes_20260404.csv`
- Audit narrative: `reports/correction_explanation_audit_20260404.md`
- Case studies: `reports/correction_case_studies_20260404.md`
- Summary table (CSV): `tables/correction_explanation_summary_20260404.csv`
- Summary table (Markdown): `tables/correction_explanation_summary_20260404.md`

## Harmful-correction failure audit (non-canonical, manuscript-supporting)

Lightweight post hoc analysis of harmful changed-node corrections using the explanation export.

- Audit narrative: `reports/harmful_correction_failure_audit_20260404.md`
- Case studies: `reports/harmful_correction_case_studies_20260404.md`
- Summary table (CSV): `tables/harmful_correction_failure_summary_20260404.csv`
- Summary table (Markdown): `tables/harmful_correction_failure_summary_20260404.md`


## Post hoc safeguard simulation for harmful corrections (non-canonical, manuscript-supporting)

Lightweight post hoc simulation on existing changed-node explanation exports to evaluate simple safeguards that may reduce harmful corrections while preserving beneficial ones.

- Simulation script: `scripts/posthoc_safeguard_simulation.py`
- Narrative report: `reports/posthoc_safeguard_simulation_20260404.md`
- Case studies: `reports/posthoc_safeguard_case_studies_20260404.md`
- Summary table (CSV): `tables/posthoc_safeguard_simulation_20260404.csv`
- Summary table (Markdown): `tables/posthoc_safeguard_simulation_20260404.md`

## FINAL_V3 regime-mapping analysis (non-canonical, manuscript-supporting)

Lightweight descriptive regime map that reuses existing full-scope outputs to characterize when FINAL_V3 helps more vs less.

- Analysis script: `scripts/final_v3_regime_mapping.py`
- Narrative report: `reports/final_v3_regime_mapping_20260404.md`
- Regime table (CSV): `tables/final_v3_regime_mapping_20260404.csv`
- Regime table (Markdown): `tables/final_v3_regime_mapping_20260404.md`

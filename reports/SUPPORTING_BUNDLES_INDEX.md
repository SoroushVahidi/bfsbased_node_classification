# Supporting Bundles Index

This index covers all non-canonical artifact bundles in `reports/` and `tables/`.
For canonical artifacts, see [`docs/RESULTS_GUIDE.md`](../docs/RESULTS_GUIDE.md).

Status key:
- **Canonical** — frozen paper evidence; do not overwrite.
- **Appendix-support** — full-scope, current-code runs; suitable for supplementary sections.
- **Manuscript-supporting** — diagnostic evidence for specific manuscript arguments; non-canonical.
- **Exploratory** — experimental or partial runs; informative but not claim-bearing.
- **Provenance** — historical or analytical notes; not intended for citation.

---

## 1. Core canonical artifacts (brief reference)

| Artifact | Path | Status |
|----------|------|--------|
| Per-split result log | `reports/final_method_v3_results.csv` | **Canonical / frozen** |
| Main results table | `tables/main_results_selective_correction.{md,csv}` | **Canonical** |
| Experimental setup table | `tables/experimental_setup_selective_correction.{md,csv}` | **Canonical** |
| Ablation table | `tables/ablation_selective_correction.{md,csv}` | **Canonical** |
| Sensitivity table | `tables/sensitivity_selective_correction.{md,csv}` | **Canonical** |
| Method narrative | `reports/final_method_v3_analysis.md` | **Canonical** |
| Safety analysis | `reports/safety_analysis.md` | **Canonical** |
| Regime analysis | `reports/final_v3_regime_analysis.md` | **Canonical** |

See [`docs/RESULTS_GUIDE.md`](../docs/RESULTS_GUIDE.md) for the full canonical artifact map.

---

## 2. Fresh ablation rerun

Full-scope 6-dataset × 10-split ablation rerun using current code. Recommended over the
historical mixed ablation table when citing a current-code ablation.

| Artifact | Path | Status |
|----------|------|--------|
| Rerun source CSV | `reports/final_v3_ablation_rerun_results.csv` | **Appendix-support** |
| Rerun summary table (MD) | `tables/ablation_selective_correction_rerun.md` | **Appendix-support** |
| Rerun summary table (CSV) | `tables/ablation_selective_correction_rerun.csv` | **Appendix-support** |
| Comparison note vs historical | `reports/final_v3_ablation_rerun_analysis.md` | **Appendix-support** |

---

## 3. Expert Systems lightweight strengthening bundle

Reduced-scope diagnostic package (three datasets, tagged `20260404_reduced`) and full-scope
extension (six datasets × 10 splits, tagged `20260404_fullscope`). Includes correction-behavior
diagnostics, bounded-intervention safety, and gate-sensitivity evidence.

### 3a. Reduced-scope (three datasets)

| Artifact | Path | Status |
|----------|------|--------|
| Audit memo | `reports/expert_systems_canonical_audit_20260404_reduced.md` | **Manuscript-supporting** |
| Ablation source CSV | `reports/final_v3_expert_systems_lightweight_ablation_20260404_reduced.csv` | **Manuscript-supporting** |
| Correction diagnostics CSV | `reports/final_v3_correction_diagnostics_20260404_reduced.csv` | **Manuscript-supporting** |
| Gate sensitivity CSV | `reports/final_v3_expert_systems_gate_sensitivity_20260404_reduced.csv` | **Manuscript-supporting** |
| Correction diagnostics table | `tables/final_v3_correction_diagnostics_20260404_reduced.md` | **Manuscript-supporting** |
| Bounded intervention table | `tables/safety_bounded_intervention_expert_systems_lightweight_20260404_reduced.md` | **Manuscript-supporting** |
| Gate sensitivity table | `tables/sensitivity_selective_correction_expert_systems_lightweight_20260404_reduced.md` | **Manuscript-supporting** |

### 3b. Full-scope (six datasets × 10 splits) — preferred for appendix

| Artifact | Path | Status |
|----------|------|--------|
| Audit memo | `reports/expert_systems_canonical_audit_20260404_fullscope.md` | **Appendix-support** |
| Ablation source CSV | `reports/final_v3_expert_systems_lightweight_ablation_20260404_fullscope.csv` | **Appendix-support** |
| Correction diagnostics CSV | `reports/final_v3_correction_diagnostics_20260404_fullscope.csv` | **Appendix-support** |
| Gate sensitivity CSV | `reports/final_v3_expert_systems_gate_sensitivity_20260404_fullscope.csv` | **Appendix-support** |
| Correction diagnostics table | `tables/final_v3_correction_diagnostics_20260404_fullscope.md` | **Appendix-support** |
| Ablation table | `tables/ablation_selective_correction_expert_systems_lightweight_20260404_fullscope.md` | **Appendix-support** |
| Bounded intervention table | `tables/safety_bounded_intervention_expert_systems_lightweight_20260404_fullscope.md` | **Appendix-support** |
| Gate sensitivity table | `tables/sensitivity_selective_correction_expert_systems_lightweight_20260404_fullscope.md` | **Appendix-support** |

---

## 4. Gate sensitivity (full-scope)

Full-scope gate-sensitivity sweep across all six canonical datasets and all 10 splits.

| Artifact | Path | Status |
|----------|------|--------|
| Narrative report | `reports/gate_sensitivity_fullscope_20260404.md` | **Appendix-support** |
| Gate sensitivity table (CSV) | `tables/gate_sensitivity_fullscope_20260404.csv` | **Appendix-support** |
| Gate sensitivity table (MD) | `tables/gate_sensitivity_fullscope_20260404.md` | **Appendix-support** |

---

## 5. Bounded-intervention safety (full-scope)

Full-scope bounded-intervention analysis (wins/ties/losses vs MLP) across all six datasets
and 10 splits.

| Artifact | Path | Status |
|----------|------|--------|
| Narrative report | `reports/bounded_intervention_fullscope_20260404.md` | **Appendix-support** |
| Bounded-intervention table (CSV) | `tables/bounded_intervention_fullscope_20260404.csv` | **Appendix-support** |
| Bounded-intervention table (MD) | `tables/bounded_intervention_fullscope_20260404.md` | **Appendix-support** |

See also the reduced-scope equivalent in §3a above, and the intermediate reports:
- `reports/final_v3_bounded_intervention_report_20260404_fullscope.md`
- `reports/final_v3_bounded_intervention_report_20260404_reduced.md`

---

## 6. Correction behavior (lightweight)

Per-dataset summary of how FINAL_V3 corrects nodes: routed fraction, changed fraction,
mean reliability on corrected nodes.

| Artifact | Path | Status |
|----------|------|--------|
| Correction behavior report (fullscope) | `reports/final_v3_correction_behavior_report_20260404_fullscope.md` | **Manuscript-supporting** |
| Correction behavior report (reduced) | `reports/final_v3_correction_behavior_report_20260404_reduced.md` | **Manuscript-supporting** |

---

## 7. Bucket safety analysis

Bucketed safety analysis by MLP test-margin tertile (low / medium / high confidence).
Supports the feature-first conservatism argument.

| Artifact | Path | Status |
|----------|------|--------|
| Full-scope bucket safety summary | `reports/final_v3_bucket_safety_summary.md` | **Manuscript-supporting** |
| Full-scope bucket safety conclusions | `reports/final_v3_bucket_safety_conclusions.md` | **Manuscript-supporting** |
| Structural bucket safety summary | `reports/final_v3_bucket_safety_struct_summary.md` | **Manuscript-supporting** |
| Bucket safety main table (MD) | `tables/final_v3_bucket_safety_main.md` | **Manuscript-supporting** |
| Bucket safety main table (CSV) | `tables/final_v3_bucket_safety_main.csv` | **Manuscript-supporting** |
| Structural bucket safety table (MD) | `tables/final_v3_bucket_safety_struct_main.md` | **Manuscript-supporting** |
| Structural bucket safety table (CSV) | `tables/final_v3_bucket_safety_struct_main.csv` | **Manuscript-supporting** |
| Early partial bucket safety (3 datasets) | `reports/margin_bucket_safety_summary.md` | **Provenance** |

---

## 8. Correction explanation audit

Node-level interpretability bundle for FINAL_V3 correction decisions (changed nodes only).

| Artifact | Path | Status |
|----------|------|--------|
| Node-level export | `reports/correction_explanation_nodes_20260404.csv` | **Manuscript-supporting** |
| Audit narrative | `reports/correction_explanation_audit_20260404.md` | **Manuscript-supporting** |
| Case studies | `reports/correction_case_studies_20260404.md` | **Manuscript-supporting** |
| Summary table (CSV) | `tables/correction_explanation_summary_20260404.csv` | **Manuscript-supporting** |
| Summary table (MD) | `tables/correction_explanation_summary_20260404.md` | **Manuscript-supporting** |

---

## 9. Harmful-correction failure audit

Post hoc analysis of harmful changed-node corrections using the explanation export.

| Artifact | Path | Status |
|----------|------|--------|
| Audit narrative | `reports/harmful_correction_failure_audit_20260404.md` | **Manuscript-supporting** |
| Case studies | `reports/harmful_correction_case_studies_20260404.md` | **Manuscript-supporting** |
| Summary table (CSV) | `tables/harmful_correction_failure_summary_20260404.csv` | **Manuscript-supporting** |
| Summary table (MD) | `tables/harmful_correction_failure_summary_20260404.md` | **Manuscript-supporting** |

---

## 10. Post hoc safeguard simulation

Post hoc simulation on existing changed-node explanation exports evaluating simple safeguards
that may reduce harmful corrections while preserving beneficial ones.

| Artifact | Path | Status |
|----------|------|--------|
| Simulation script | `scripts/posthoc_safeguard_simulation.py` | **Manuscript-supporting** |
| Narrative report | `reports/posthoc_safeguard_simulation_20260404.md` | **Manuscript-supporting** |
| Case studies | `reports/posthoc_safeguard_case_studies_20260404.md` | **Manuscript-supporting** |
| Summary table (CSV) | `tables/posthoc_safeguard_simulation_20260404.csv` | **Manuscript-supporting** |
| Summary table (MD) | `tables/posthoc_safeguard_simulation_20260404.md` | **Manuscript-supporting** |

---

## 11. Regime mapping

Descriptive regime map characterizing when FINAL_V3 corrects more or less, using full-scope outputs.

| Artifact | Path | Status |
|----------|------|--------|
| Analysis script | `scripts/final_v3_regime_mapping.py` | **Manuscript-supporting** |
| Narrative report | `reports/final_v3_regime_mapping_20260404.md` | **Manuscript-supporting** |
| Regime table (CSV) | `tables/final_v3_regime_mapping_20260404.csv` | **Manuscript-supporting** |
| Regime table (MD) | `tables/final_v3_regime_mapping_20260404.md` | **Manuscript-supporting** |
| Regime summary table (MD) | `tables/final_v3_regime_summary.md` | **Manuscript-supporting** |
| Regime summary table (CSV) | `tables/final_v3_regime_summary.csv` | **Manuscript-supporting** |

---

## 12. Targeted-benefit decomposition

Subgroup decomposition showing where FINAL_V3 gains arise (routed, changed, unchanged nodes).

| Artifact | Path | Status |
|----------|------|--------|
| Analysis script | `code/bfsbased_node_classification/run_final_v3_targeted_benefit_decomposition.py` | **Manuscript-supporting** |
| Narrative report | `reports/final_v3_targeted_benefit_decomposition_20260404.md` | **Manuscript-supporting** |
| Subgroup table (CSV) | `tables/final_v3_targeted_benefit_decomposition_20260404.csv` | **Manuscript-supporting** |
| Subgroup table (MD) | `tables/final_v3_targeted_benefit_decomposition_20260404.md` | **Manuscript-supporting** |

---

## 13. MS-HSGC exploratory experiment

Non-canonical lightweight and medium-tuning runs of the MS-HSGC variant (multi-scale
homophily-sensitive graph correction). Informative context only.

| Artifact | Path | Status |
|----------|------|--------|
| Light run summary | `reports/ms_hsgc_light_summary.md` | **Exploratory** |
| Medium run summary | `reports/ms_hsgc_medium_summary.md` | **Exploratory** |
| Light results CSV | `reports/ms_hsgc_results_light.csv` | **Exploratory** |
| Medium results CSV | `reports/ms_hsgc_results_medium.csv` | **Exploratory** |
| Full 10-split targeted results | `reports/ms_hsgc_results_targeted_10split.csv` | **Exploratory** |
| Light delta vs FINAL_V3 | `reports/ms_hsgc_vs_final_v3_light.csv` | **Exploratory** |
| Medium delta vs FINAL_V3 | `reports/ms_hsgc_vs_final_v3_medium.csv` | **Exploratory** |
| 10-split delta vs FINAL_V3 | `reports/ms_hsgc_vs_final_v3_targeted_10split.csv` | **Exploratory** |
| H1-max sensitivity CSV | `reports/ms_hsgc_h1max_sensitivity_targeted_10split.csv` | **Exploratory** |
| Light comparison table | `tables/ms_hsgc_light_comparison.md` | **Exploratory** |
| Medium comparison table | `tables/ms_hsgc_medium_comparison.md` | **Exploratory** |

---

## 14. CA-MRC exploratory experiment

Non-canonical light run of CA-MRC (Compatibility-Aware Multi-hop Residual Correction).

| Artifact | Path | Status |
|----------|------|--------|
| Light run summary | `reports/ca_mrc_light_summary.md` | **Exploratory** |
| Light results CSV | `reports/ca_mrc_results_light.csv` | **Exploratory** |
| Delta vs FINAL_V3 CSV | `reports/ca_mrc_vs_final_v3_light.csv` | **Exploratory** |
| Light comparison table | `tables/ca_mrc_light_comparison.md` | **Exploratory** |
| Compatibility matrices directory | `reports/ca_mrc_compat_matrices/` | **Exploratory** |

---

## 15. CA-MRD exploratory experiment

Non-canonical light run of CA-MRD (alternate compatibility-aware variant, separate run tag).

| Artifact | Path | Status |
|----------|------|--------|
| Light run summary | `reports/ca_mrd_light_summary_ca_mrd_light.md` | **Exploratory** |
| Light results CSV | `reports/ca_mrd_results_ca_mrd_light.csv` | **Exploratory** |
| Delta vs FINAL_V3 CSV | `reports/ca_mrd_vs_final_v3_ca_mrd_light.csv` | **Exploratory** |
| Light comparison table | `tables/ca_mrd_light_comparison_ca_mrd_light.md` | **Exploratory** |

---

## 16. Recon-behavior experiment

Non-canonical lightweight multi-dataset sweep comparing FINAL_V3 vs MS-HSGC in a
reconstruction-behavior regime.

| Artifact | Path | Status |
|----------|------|--------|
| Summary report | `reports/recon_behavior_summary_recon_sweep.md` | **Exploratory** |
| Results CSV | `reports/recon_behavior_results_recon_sweep.csv` | **Exploratory** |
| Binned analysis CSV | `reports/recon_behavior_binned_recon_sweep.csv` | **Exploratory** |
| Recon delta vs FINAL_V3 CSV | `reports/recon_ms_hsgc_vs_final_v3_recon_sweep.csv` | **Exploratory** |
| Comparison table | `tables/recon_behavior_comparison_recon_sweep.md` | **Exploratory** |

---

## 17. Provenance-only materials

These reports are not intended for citation. They are preserved for development provenance.

| Artifact | Path | Status | Note |
|----------|------|--------|------|
| Parallel-tracks planning analysis | `reports/parallel_tracks_analysis_2026-04-03.md` | **Provenance** | Exploratory branching analysis from 2026-04-03; not a claim-bearing result |
| TRIPLE_TRUST_SGC method notes | `reports/triple_trust_sgc.md` | **Provenance** | Experimental variant design notes; not evaluated at scale |
| DJGNN reports and tables | `reports/djgnn/`, `tables/djgnn/` | **Provenance** | DJGNN integration material; not canonical |

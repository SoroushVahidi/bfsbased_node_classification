# Canonical pipeline audit (Expert Systems lightweight pass)

## Canonical now
- `code/final_method_v3.py` is a stable import shim for reviewers.
- `code/bfsbased_node_classification/final_method_v3.py` contains FINAL_V3 logic.
- `code/bfsbased_node_classification/run_final_evaluation.py` executes MLP, SGC v1, V2, and FINAL_V3 over fixed splits and writes tagged CSV reruns.
- `scripts/run_all_selective_correction_results.sh` rebuilds tables/figures from frozen CSVs (no training).
- `reports/final_method_v3_results.csv` is frozen canonical split-level evidence.
- `tables/main_results_selective_correction.*` are canonical main-table products built from the frozen CSV.

## Evidence thin spots before this pass
- `tables/ablation_selective_correction.*` was partly historical: only FINAL_V3 recomputed, non-FINAL rows pulled from archived snapshot.
- `tables/sensitivity_selective_correction.*` was historical 3-split rho grid, not a current-code rerun.
- Correction behavior diagnostics (routed fraction, changed fraction, changed precision, mean reliability/margin on corrected nodes) were not packaged in a compact canonical-candidate table.
- Split-level bounded-intervention summary (wins/ties/losses vs MLP) was not consolidated into a dedicated manuscript-ready table.
- Runtime-bounded reduced plan used in this pass: citeseer, cora, pubmed, texas.
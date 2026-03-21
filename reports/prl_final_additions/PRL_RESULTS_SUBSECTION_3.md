# Results — subsection 3 (gate / threshold): evidence-backed guidance

## Answers (from repository artifacts)

| Question | Answer |
|----------|--------|
| **Full per-threshold sweeps on fixed splits?** | **No.** `results_prl/README_threshold_sensitivity.md` states diagnostics omit full per-candidate traces on disk. |
| **What evidence exists?** | Per **run**: validation-selected `threshold`, test `uncertain_fraction_test`, `changed_fraction_test`, `delta_vs_mlp` in `results_prl/threshold_sensitivity_crossruns.jsonl` (built from `logs/comparison_runs_manuscript_final_validation.jsonl`). |
| **τ associated with uncertain / corrected fraction?** | **Yes (cross-run):** Pearson **corr(τ, uncertain)** in `results_prl/threshold_uncertainty_crossrun_summary.csv`; extended correlations in `tables/prl_final_additions/prl_subsection3_crossrun_correlations.csv`. |
| **Gate controls coverage?** | **Partially:** higher τ tends to co-vary with uncertain coverage **across runs**; not a controlled single-split experiment. |
| **τ vs gain?** | **Mixed:** same correlation table — **do not** claim monotonic improvement with stricter τ (tertiles in `results_prl/threshold_sensitivity_mini_table_tertiles.csv` can show opposite trends). |
| **Qualitative node-level?** | **Yes, limited:** `logs/manuscript_case_studies_final_validation.md` (split **0** only). |

## Recommended title (safest)

1. **Uncertainty gate behavior across validation runs** (best)  
2. **Validation-selected threshold and correction coverage across runs**  
3. ~~Threshold sensitivity~~ — avoid as **standalone** title (implies controlled sweep).

## Suggested figure / table for this subsection

- **Optional figure:** one panel from `results_prl/fig_threshold_sensitivity_panels_crossruns.pdf` **or** text-only + cite correlation table.  
- **Table:** one row per dataset (Cora, CiteSeer, PubMed, Chameleon): mean τ, mean uncertain fraction, corr(τ, uncertain) — values in `threshold_uncertainty_crossrun_summary.csv`.

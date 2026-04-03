# Naming migration map

This document records every file renamed or moved during the venue-neutral
reorganization for backward traceability.

| Old path | New path | Reason for change |
|----------|----------|-------------------|
| `prl_final_additions/` | `archive/legacy_venue_specific/prl_final_additions/` | Venue-specific bundle moved to archive |
| `results_prl/` | `archive/legacy_venue_specific/results_prl/` | Venue-specific results moved to archive |
| `logs/prl_resubmission/` | `archive/legacy_venue_specific/logs_prl_resubmission/` | Venue-specific logs moved to archive |
| `logs/prl_final_additions/` | `archive/legacy_venue_specific/logs_prl_final_additions/` | Venue-specific logs moved to archive |
| `reports/prl_resubmission/` | `archive/legacy_venue_specific/reports_resubmission_protocols/` | Venue-specific reports moved to archive |
| `reports/prl_final_additions/` | `archive/legacy_venue_specific/reports_prl_final_additions/` | Venue-specific reports moved to archive |
| `tables/prl_final_additions/` | `archive/legacy_venue_specific/tables_prl_final_additions/` | Venue-specific tables moved to archive |
| `tables/prl_resubmission/` | `archive/legacy_venue_specific/tables_prl_resubmission/` | Venue-specific tables moved to archive |
| `figures/prl_final_additions/` | `archive/legacy_venue_specific/figures_prl_final_additions/` | Venue-specific figures moved to archive |
| `slurm/run_prl_resubmission_analysis.sbatch` | `archive/legacy_venue_specific/slurm/run_prl_resubmission_analysis.sbatch` | Venue-specific HPC script moved to archive |
| `slurm/run_prl_resubmission_core.sbatch` | `archive/legacy_venue_specific/slurm/run_prl_resubmission_core.sbatch` | Venue-specific HPC script moved to archive |
| `slurm/run_prl_resubmission_core_group1.sbatch` | `archive/legacy_venue_specific/slurm/run_prl_resubmission_core_group1.sbatch` | Venue-specific HPC script moved to archive |
| `slurm/run_prl_resubmission_core_group2.sbatch` | `archive/legacy_venue_specific/slurm/run_prl_resubmission_core_group2.sbatch` | Venue-specific HPC script moved to archive |
| `slurm/run_prl_resubmission_merge_analysis.sbatch` | `archive/legacy_venue_specific/slurm/run_prl_resubmission_merge_analysis.sbatch` | Venue-specific HPC script moved to archive |
| `slurm/run_prl_structural_group1.sbatch` | `archive/legacy_venue_specific/slurm/run_prl_structural_group1.sbatch` | Venue-specific HPC script moved to archive |
| `slurm/run_prl_structural_group2.sbatch` | `archive/legacy_venue_specific/slurm/run_prl_structural_group2.sbatch` | Venue-specific HPC script moved to archive |
| `slurm/run_prl_structural_merge_analysis.sbatch` | `archive/legacy_venue_specific/slurm/run_prl_structural_merge_analysis.sbatch` | Venue-specific HPC script moved to archive |
| `reports/prl_gap_analysis.md` | `archive/legacy_venue_specific/reports_misc/prl_gap_analysis.md` | Venue-specific report moved to archive |
| `reports/prl_insertable_text.md` | `archive/legacy_venue_specific/reports_misc/prl_insertable_text.md` | Venue-specific report moved to archive |
| `reports/prl_propagation_baseline_definition.md` | `archive/legacy_venue_specific/reports_misc/prl_propagation_baseline_definition.md` | Venue-specific report moved to archive |
| `reports/prl_related_work_patch.md` | `archive/legacy_venue_specific/reports_misc/prl_related_work_patch.md` | Venue-specific report moved to archive |
| `reports/prl_repo_audit.md` | `archive/legacy_venue_specific/reports_misc/prl_repo_audit.md` | Venue-specific report moved to archive |
| `reports/prl_statistical_summary.md` | `archive/legacy_venue_specific/reports_misc/prl_statistical_summary.md` | Venue-specific report moved to archive |
| `reports/prl_submission_readiness.md` | `archive/legacy_venue_specific/reports_misc/prl_submission_readiness.md` | Venue-specific report moved to archive |
| `logs/prl_statistical_summary.csv` | `archive/legacy_venue_specific/prl_statistical_summary.csv` | Venue-specific log moved to archive |
| `scripts/run_all_prl_results.sh` | `scripts/run_all_selective_correction_results.sh` | Renamed to venue-neutral name |
| `scripts/prl_final_additions/` | `scripts/build_artifacts/` | Renamed to venue-neutral name |
| `tables/main_results_prl.md` | `tables/main_results_selective_correction.md` | Renamed to venue-neutral name |
| `tables/main_results_prl.csv` | `tables/main_results_selective_correction.csv` | Renamed to venue-neutral name |
| `tables/experimental_setup_prl.md` | `tables/experimental_setup_selective_correction.md` | Renamed to venue-neutral name |
| `tables/experimental_setup_prl.csv` | `tables/experimental_setup_selective_correction.csv` | Renamed to venue-neutral name |
| `tables/ablation_prl.md` | `tables/ablation_selective_correction.md` | Renamed to venue-neutral name |
| `tables/ablation_prl.csv` | `tables/ablation_selective_correction.csv` | Renamed to venue-neutral name |
| `tables/sensitivity_prl.md` | `tables/sensitivity_selective_correction.md` | Renamed to venue-neutral name |
| `tables/sensitivity_prl.csv` | `tables/sensitivity_selective_correction.csv` | Renamed to venue-neutral name |
| `figures/prl_graphical_abstract_v3.png` | `figures/graphical_abstract_selective_correction_v3.png` | Renamed to venue-neutral name |
| `figures/prl_graphical_abstract_v3.pdf` | `figures/graphical_abstract_selective_correction_v3.pdf` | Renamed to venue-neutral name |
| `figures/prl_graphical_abstract_v3.svg` | `figures/graphical_abstract_selective_correction_v3.svg` | Renamed to venue-neutral name |
| `README_PRL_MANUSCRIPT.md` | `README_MANUSCRIPT_ARTIFACTS.md` | Renamed to venue-neutral name |
| `code/bfsbased_node_classification/prl_resubmission_runner.py` | `code/bfsbased_node_classification/resubmission_runner.py` | Renamed to venue-neutral name |
| `code/bfsbased_node_classification/analyze_prl_resubmission.py` | `code/bfsbased_node_classification/analyze_resubmission_results.py` | Renamed to venue-neutral name |
| `code/bfsbased_node_classification/merge_prl_resubmission_runs.py` | `code/bfsbased_node_classification/merge_resubmission_runs.py` | Renamed to venue-neutral name |

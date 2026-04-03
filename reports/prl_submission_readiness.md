# PRL submission readiness

This note summarizes the current repository state for a narrow Pattern
Recognition Letters submission built around `FINAL_V3`.

## Strongest points

- The canonical method/package is now clearly identified as `FINAL_V3`.
- The main benchmark is frozen and reproducible from:
  - `reports/final_method_v3_results.csv`
  - `tables/main_results_prl.*`
  - `scripts/run_all_prl_results.sh`
- The paper already has a coherent narrow story:
  - strong MLP baseline
  - selective graph correction
  - reduced harmful graph influence
  - conservative behavior on harder regimes
- The repo now includes:
  - `reports/prl_repo_audit.md`
  - `reports/prl_gap_analysis.md`
  - `reports/prl_propagation_baseline_definition.md`
  - `reports/prl_related_work_patch.md`
  - `reports/prl_statistical_summary.md`

## Biggest remaining scientific weaknesses

- The main frozen paper package is still narrower than the broader resubmission package.
- `tables/ablation_prl.*` and `tables/sensitivity_prl.*` remain historical snapshots unless replaced by current-code outputs.
- There is no tracked manuscript or bibliography file inside this repository, so related-work coverage can only be supported indirectly through patch notes.
- Some auxiliary `FINAL_V3` diagnostics are available only for splits `0–4`, though the core benchmark itself is fully standardized on 10 splits.

## Newly found vs newly computed

### Already present and reusable

- frozen 10-split `FINAL_V3` benchmark
- main benchmark table
- safety analysis
- setup table
- canonical figure refresh path
- older threshold cross-run data
- resubmission baseline and ablation package for the supplementary `UG-SGC` line

### Newly generated in this pass

- `reports/prl_repo_audit.md`
- `reports/prl_gap_analysis.md`
- `reports/prl_propagation_baseline_definition.md`
- `reports/prl_related_work_patch.md`
- `reports/prl_statistical_summary.md`
- `logs/prl_statistical_summary.csv`
- refreshed threshold cross-run figures under `results_prl/`

### In progress during this pass

- current-code `FINAL_V3` ablation and gate-usefulness package:
  - `logs/prl_ablation_results.csv`
  - `logs/prl_ablation_summary.csv`
  - `logs/prl_gate_analysis.csv`
  - `reports/prl_ablation_interpretation.md`
  - `reports/prl_gate_analysis.md`
  - script: `scripts/prl_final_additions/compute_final_v3_ablation_package.py`
  - execution path:
    `source /apps/easybuild/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh && conda activate feedback-weighted-maximization && python3 scripts/prl_final_additions/compute_final_v3_ablation_package.py`

## Recommended manuscript inserts

Main paper:
- benchmark table from `tables/main_results_prl.md`
- setup clarification from `tables/experimental_setup_prl.md`
- safety paragraph from `reports/safety_analysis.md`
- compact statistical wording from `reports/prl_statistical_summary.md`

Supplement / compressed discussion:
- propagation-baseline definition from `reports/prl_propagation_baseline_definition.md`
- related-work positioning from `reports/prl_related_work_patch.md`
- ablation/gate summary from the new `FINAL_V3` ablation package once complete

## Submission-readiness call

Current status:
- **close to submission-ready for a narrow PRL paper**

What would make it stronger:
- finish and review the new `FINAL_V3` ablation/gate package
- then use that compact evidence to support the central selective-correction claim

If the manuscript keeps the scope narrow and avoids broad heterophily claims,
the repository evidence is already strong enough for drafting. The only
remaining improvement that materially strengthens the paper is the current
focused `FINAL_V3` ablation/gate package.

# PRL resubmission run status

## Focused smoke test

- output tag: `prl_resubmission_smoke_cora_s0_v3`
- runs file:
  `logs/prl_resubmission/comparison_runs_prl_resubmission_smoke_cora_s0_v3.jsonl`
- analysis outputs:
  - `tables/prl_resubmission/baseline_comparison_prl_resubmission_smoke_cora_s0_v3.csv`
  - `tables/prl_resubmission/ablation_comparison_prl_resubmission_smoke_cora_s0_v3.csv`
  - `reports/prl_resubmission/protocol_prl_resubmission_smoke_cora_s0_v3.{json,md}`
  - `reports/prl_resubmission/summary_prl_resubmission_smoke_cora_s0_v3.md`
  - `reports/prl_resubmission/claims_and_risks_prl_resubmission_smoke_cora_s0_v3.md`

Purpose:
- validate the sanitized loader,
- validate the new baseline code path,
- validate SGC ablation records and analysis generation.

## Initial monolithic Wulver job

- sbatch script: `slurm/run_prl_resubmission_core.sbatch`
- submitted job ID: `889218`
- status: cancelled in favor of parallel grouped jobs to reduce wall-clock time

## Active grouped Wulver jobs

### Group 1

- sbatch script:
  `slurm/run_prl_resubmission_core_group1.sbatch`
- job ID:
  `889227`
- datasets:
  `cora`, `citeseer`, `pubmed`
- output tag:
  `prl_resubmission_core_group1_r1`
- runs file:
  `logs/prl_resubmission/comparison_runs_prl_resubmission_core_group1_r1.jsonl`

### Group 2

- sbatch script:
  `slurm/run_prl_resubmission_core_group2.sbatch`
- job ID:
  `889228`
- datasets:
  `chameleon`, `wisconsin`
- output tag:
  `prl_resubmission_core_group2_r1`
- runs file:
  `logs/prl_resubmission/comparison_runs_prl_resubmission_core_group2_r1.jsonl`

### Dependent merge + analysis

- sbatch script:
  `slurm/run_prl_resubmission_merge_analysis.sbatch`
- job ID:
  `889229`
- dependency:
  `afterok:889227:889228`
- merged output tag:
  `prl_resubmission_core_merged_r1`
- merged runs file:
  `logs/prl_resubmission/comparison_runs_prl_resubmission_core_merged_r1.jsonl`

Configured methods for both group jobs:
- `mlp_only`
- `prop_only`
- `gcn`
- `appnp`
- `selective_graph_correction`
- `gated_mlp_prop`
- `sgc_no_gate`
- `sgc_no_feature_similarity`
- `sgc_no_graph_neighbor`
- `sgc_no_compatibility`
- `sgc_mlp_plus_graph`
- `sgc_knn_enabled`

Notes:
- The first submission attempt failed because the cluster enforces account-specific QOS; the script now uses `#SBATCH --qos=standard`.
- The import path for `bfsbased-full-investigate-homophil.py` was sanitized before this run so the legacy top-level notebook experiment block is no longer executed during runner imports.

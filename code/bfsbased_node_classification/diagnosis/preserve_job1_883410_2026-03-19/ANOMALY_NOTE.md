# JOB 1 anomaly note (883410)

This bundle preserves outputs from JOB 1 that appear broken / incomplete for manuscript tables.

## Observed checks (from on-Wulver verification)
- Raw JSONL row count: `420`
- Per-dataset counts: `60` rows each for `cora, citeseer, chameleon, texas, actor, cornell, wisconsin`
- Per-method counts in JSONL:
  - `mlp_only`: `210` rows, **all** with `success=false`
  - `prop_only`: `210` rows, all with `success=true`
  - `selective_graph_correction`: **missing entirely** from the JSONL

## CSV outputs
- `manuscript_gain_over_mlp_final_validation_main.csv`: only header (1 line)
- `manuscript_correction_behavior_final_validation_main.csv`: only header (1 line)
- `manuscript_regime_analysis_final_validation_main.csv`: has dataset rows, but SGC columns are `N/A`

## Notable error string found in JSONL sample
Example `notes` observed in JSONL lines for `mlp_only`:
- `train_mlp_and_predict() got an unexpected keyword argument 'val_indices'`

## Provenance
- JOB 1 script: `run_selective_validation_main_r3.sbatch`
- Reproducibility snapshot directory: `run_metadata/`

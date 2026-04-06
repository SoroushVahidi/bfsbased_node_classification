# Wulver artifact retention plan (2026-04-04)

This plan applies the repository artifact-retention policy to the current Wulver sweep situation.

Reference status memo: `reports/wulver_experiments_status_20260404.md`.

## Scope covered

- baseline comparison outputs (`outputs/baseline_comparison/*`)
- label-budget outputs (`outputs/label_budget/*`)
- graph-noise robustness outputs (`outputs/graph_noise_robustness/*`)
- raw Slurm runtime logs (`slurm/*.out`, `slurm/*.err`)
- transient caches/temp data (PyG/torch caches and related runtime artifacts)

## What will be committed

- this retention plan (`reports/wulver_artifact_retention_plan_20260404.md`)
- draft manifest (`run_metadata/manifests/manifest_wulver_sweeps_20260404_draft.json`)
- optional Slurm runtime note summaries from `scripts/artifacts/summarize_slurm_logs.py`
- code/docs/scripts needed to regenerate and audit runs

## What will remain off Git

- raw per-run JSONL under `outputs/...`
- raw Slurm `.out` and `.err` logs
- transient caches (for example `data/pyg_cache/`)

## Off-Git archive plan

Create tar archives (or equivalent object-storage bundles) for each family when needed:

- `wulver_baseline_cmp_20260403_150529`
- `wulver_feature_posthoc_applied_intel_20260403_174551`
- `wulver_label_budget_applied_intel_20260403_180036`
- `wulver_graph_noise_robustness_applied_intel_20260403_181618`
- optional Slurm logs bundle for timeout/error diagnosis

For each archive, record in a manifest:

- archive filename
- sha256 checksum
- size in bytes
- storage location pointer (filesystem path/object URI)

## Deletion policy after extraction

After summary notes and manifest fields are complete:

- delete local `outputs/...` trees if disk pressure requires,
- delete local `slurm/*.out` and `slurm/*.err` after key notes are captured,
- keep or regenerate caches as needed (never version them).

## Known incomplete shards (from status memo)

- `bl_cmp`: `902966_2` timeout
- `bl_cmp_rem`: `902972_0`, `902972_2` timeout
- `bl_featph`: `903033_2` timeout
- `djgnn_arr`: `902987_2` running at snapshot time

Action: finish or backfill missing shards first for fully complete family-level archives.

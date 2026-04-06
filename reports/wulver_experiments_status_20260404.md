# Wulver experiment status (2026-04-04 snapshot)

This report records the current state of the Slurm runs launched for baseline comparison, feature-posthoc baselines, label-budget sweep, and graph-noise robustness.

## Important note about "all results"

Raw sweep outputs live under `outputs/` and are intentionally gitignored:

- `outputs/baseline_comparison/`
- `outputs/label_budget/`
- `outputs/graph_noise_robustness/`

So the repository stores **code + reproducible run details + status metadata**, not large per-run JSONL blobs.

## Jobs and states (`sacct`)

- `902966` (`bl_cmp`): 5 completed, 1 timeout (`902966_2`)
- `902972` (`bl_cmp_rem`): 1 completed, 2 timeouts (`902972_0`, `902972_2`)
- `903033` (`bl_featph`): 8 completed, 1 timeout (`903033_2`)
- `903042` (`lb_budget`): 20 completed, 0 timeout
- `903063` (`gn_robust`): 24 completed, 0 timeout
- `902987` (`djgnn_arr`): 8 completed, 1 running (`902987_2`)

Queue check at snapshot time (`squeue`) showed only:

- `902987_2 djgnn_arr R`

## Output tags and file coverage

- Baseline comparison: `wulver_baseline_cmp_20260403_150529`
  - per-run JSONL files present: 9 (`runs_<dataset>.jsonl`)
- Feature-posthoc: `wulver_feature_posthoc_applied_intel_20260403_174551`
  - per-run JSONL files present: 9
- Label budget: `wulver_label_budget_applied_intel_20260403_180036`
  - per-run JSONL files present: 20 (`runs_<dataset>_p{10,20,40,100}.jsonl`)
- Graph noise robustness: `wulver_graph_noise_robustness_applied_intel_20260403_181618`
  - per-run JSONL files present: 24

## Aggregation commands

Run from repo root after all desired shards are complete:

```bash
python3 scripts/baseline_comparison_wulver/aggregate_and_report.py \
  --glob "outputs/baseline_comparison/wulver_baseline_cmp_20260403_150529/per_run/runs_*.jsonl" \
  --output-dir "outputs/baseline_comparison/wulver_baseline_cmp_20260403_150529"
```

```bash
python3 scripts/baseline_comparison_wulver/aggregate_and_report.py \
  --glob "outputs/baseline_comparison/wulver_feature_posthoc_applied_intel_20260403_174551/per_run/runs_*.jsonl" \
  --output-dir "outputs/baseline_comparison/wulver_feature_posthoc_applied_intel_20260403_174551"
```

```bash
python3 scripts/baseline_comparison_wulver/aggregate_label_budget.py \
  --glob "outputs/label_budget/wulver_label_budget_applied_intel_20260403_180036/per_run/runs_*.jsonl" \
  --output-dir "outputs/label_budget/wulver_label_budget_applied_intel_20260403_180036"
```

```bash
python3 scripts/baseline_comparison_wulver/aggregate_graph_noise.py \
  --glob "outputs/graph_noise_robustness/wulver_graph_noise_robustness_applied_intel_20260403_181618/per_run/runs_*.jsonl" \
  --output-dir "outputs/graph_noise_robustness/wulver_graph_noise_robustness_applied_intel_20260403_181618"
```

## Backfill needed for full baseline completeness

Timeout shards to rerun/backfill:

- `bl_cmp`: `902966_2`
- `bl_cmp_rem`: `902972_0`, `902972_2`
- `bl_featph`: `903033_2`

Recommendation: resubmit only failed dataset shards with the same output tag where merge-safe, or with a new suffix tag and combine at aggregation time.

# Wulver baseline comparison (FINAL_V3 vs baselines)

Parent index: [`../README.md`](../README.md). Slurm file listing: [`../../slurm/README.md`](../../slurm/README.md).

## What runs

- **Internal:** `MLP_ONLY`, `BASELINE_SGC_V1`, `V2_MULTIBRANCH`, `FINAL_V3` (heuristic gate).
- **External:** `H2GCN`, `GCNII`, `GeomGCN`, `LINKX`, `ACMII_GCN_PLUSPLUS` via `standard_node_baselines.run_baseline`.
- **Datasets:** cora, citeseer, pubmed, chameleon, texas, wisconsin — **10 splits** each (0–9).
- **Outputs:** `outputs/baseline_comparison/<TAG>/per_run/runs_<dataset>.jsonl` (never touches frozen canonical CSVs).

## Launch

Compute nodes use **Anaconda `python3` without PyTorch** unless you activate a conda env. The sbatch script runs `conda activate` on `BASELINE_CONDA_ENV` (default `feedback-weighted-maximization`) and checks `import torch, torch_geometric`.

```bash
export REPO_ROOT=/path/to/bfsbased_node_classification_git   # optional
export BASELINE_OUTPUT_TAG=my_tag_20260403                  # optional
export BASELINE_CONDA_ENV=your_env_with_torch                # if default env lacks torch
./scripts/baseline_comparison_wulver/launch_wulver.sh
```

Slurm directives live in `slurm/baseline_comparison_array.sbatch` (partition `general`, qos `standard`, account `ikoutis` — adjust if your account differs).

### Why a job can fail in seconds

Slurm writes `slurm/baseline_comparison_*.out` / `.err` **before** the batch script starts. If the `slurm/` directory does not exist under the submission working directory, the job may fail immediately with no logs. **Always** run `launch_wulver.sh` (it runs `mkdir -p slurm`) or create `slurm/` yourself before `sbatch`. The repo includes `slurm/.gitkeep` so clones have an empty `slurm/` directory.

If `REPO_ROOT` / the submit directory is wrong, you get a fast failure after the script starts; the batch file now checks for `code/bfsbased_node_classification/baseline_comparison_suite.py` and prints a clear error.

### Remaining datasets (actor, cornell, squirrel)

The main array covers six datasets: cora, citeseer, pubmed, chameleon, texas, wisconsin. A follow-on array runs **actor**, **cornell**, and **squirrel** with the **same** `BASELINE_OUTPUT_TAG` so `per_run/runs_*.jsonl` merges for aggregation:

```bash
export BASELINE_OUTPUT_TAG=<same tag as main job>
./scripts/baseline_comparison_wulver/launch_remaining_datasets.sh
```

Or: `sbatch --export=ALL,REPO_ROOT,BASELINE_OUTPUT_TAG,BASELINE_CONDA_ENV slurm/baseline_comparison_remaining.sbatch`

## After jobs finish

```bash
python3 scripts/baseline_comparison_wulver/aggregate_and_report.py \
  --glob "outputs/baseline_comparison/<TAG>/per_run/runs_*.jsonl" \
  --output-dir "outputs/baseline_comparison/<TAG>"
```

This writes `summaries/mean_std_by_method_dataset.csv` and `BASELINE_COMPARISON_REPORT.md`.

## Graph noise / edge corruption robustness

Deterministic **equal-count rewire**: remove `k = round(f·|E|)` random undirected edges, add `k` random absent edges (see `code/bfsbased_node_classification/graph_corruption.py`). Features, labels, and splits unchanged.

```bash
./scripts/baseline_comparison_wulver/launch_graph_noise_wulver.sh
```

- **Driver:** `code/bfsbased_node_classification/graph_noise_robustness_sweep.py`
- **Slurm:** `slurm/graph_noise_robustness_array.sbatch` (24 tasks: 6 datasets × 4 noise levels, **72G**, **56h**)
- **Aggregate:**

```bash
python3 scripts/baseline_comparison_wulver/aggregate_graph_noise.py \
  --glob "outputs/graph_noise_robustness/<TAG>/per_run/runs_*.jsonl" \
  --output-dir "outputs/graph_noise_robustness/<TAG>"
```

## Label-budget sweep (scarce labels vs FINAL_V3)

Subsamples **only** the official training nodes (val/test unchanged). Fractions **0.1, 0.2, 0.4, 1.0**; seeds are deterministic per `(split_id, train_fraction)`.

```bash
./scripts/baseline_comparison_wulver/launch_label_budget_wulver.sh
```

- **Driver:** `code/bfsbased_node_classification/label_budget_sweep.py`
- **Slurm:** `slurm/label_budget_sweep_array.sbatch` (20 tasks: 5 datasets × 4 fractions, **72G** RAM, **56h**, CPU-only)
- **Outputs:** `outputs/label_budget/<TAG>/per_run/runs_<dataset>_p{10,20,40,100}.jsonl`
- **Aggregate:**

```bash
python3 scripts/baseline_comparison_wulver/aggregate_label_budget.py \
  --glob "outputs/label_budget/<TAG>/per_run/runs_*.jsonl" \
  --output-dir "outputs/label_budget/<TAG>"
```

## Feature-first / post-hoc baselines (Applied Intelligence competitors)

SGC (Wu), Correct & Smooth, CLP, GraphSAGE, APPNP — **MLP anchor** + these methods only (skips BASELINE_SGC_V1, V2, FINAL_V3 in-suite to save time vs the full comparison job).

```bash
./scripts/baseline_comparison_wulver/launch_feature_posthoc_wulver.sh
# Optional: export FEATURE_POSTHOC_OUTPUT_TAG=my_tag
```

Slurm: `slurm/baseline_comparison_feature_posthoc_array.sbatch` (9 datasets, **64G** RAM, 48h, `general` / `standard` / `ikoutis`). Aggregation uses the same `aggregate_and_report.py` as above.

## Driver (local / debug)

```bash
python3 code/bfsbased_node_classification/baseline_comparison_suite.py \
  --datasets texas \
  --splits 0 \
  --output-tag debug_cmp \
  --dry-run
```

```bash
python3 code/bfsbased_node_classification/baseline_comparison_suite.py \
  --datasets cora --splits 0 \
  --output-tag debug_featph \
  --external-baselines-only \
  --external-baselines sgc correct_and_smooth clp graphsage appnp \
  --dry-run
```

## Metrics (JSONL)

Per row: `test_acc`, `val_acc` (where available), `train_acc` (external baselines), `delta_vs_mlp`, `runtime_sec`, `peak_rss_kb_after` (Python `ru_maxrss`), FINAL_V3 correction fields (`n_helped`, `n_hurt`, `harmful_overwrite_rate`, …), `slurm_job_id`, `hostname`.

Cluster memory: also use `sacct -j <JOBID> --format=MaxRSS,Elapsed`.

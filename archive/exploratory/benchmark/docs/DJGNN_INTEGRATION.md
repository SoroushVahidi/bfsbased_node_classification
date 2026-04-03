# Diffusion-Jump GNN (DJ-GNN) baseline integration

## Upstream source

- Repository: [AhmedBegggaUA/Diffusion-Jump-GNNs](https://github.com/AhmedBegggaUA/Diffusion-Jump-GNNs)
- Integrated modules (vendored, lightly adapted): `code/baselines/djgnn/`
  - `models.py` — `DJ` architecture (removed fixed `torch.manual_seed(1234)` from `__init__` so split seeds from this repo apply)
  - `pump.py` — pump / orthogonality terms (unchanged logic)
  - `training.py` — `train` / `val` / `test` steps adapted from upstream `utils.py`

## What this repo adds

- **`code/bfsbased_node_classification/run_djgnn_benchmark.py`** — CLI entrypoint aligned with `run_final_evaluation.py`:
  - Loads graphs via the same legacy `load_dataset` path and **GEO-GCN-style** splits: `data/splits/{dataset}_split_0.6_0.2_{i}.npz` (actor → `film_split_...`).
  - Default datasets: Cora, CiteSeer, PubMed, Chameleon, Texas, Wisconsin, Actor, Cornell, Squirrel (nine-way list used alongside the baseline comparison suite).
  - Seeds per split: `(1337 + split_id) * 100` (same convention as `run_final_evaluation.py` / `baseline_comparison_suite.py`).
- **Hyperparameters**: Texas / Wisconsin / Cornell follow README examples in the upstream repo; other datasets use documented defaults in `run_djgnn_benchmark.py` (`_DATASET_HP`).
- **Outputs** (all filenames include `--output-tag`):
  - `logs/djgnn/runs_<tag>.jsonl` — one JSON object per split run
  - `reports/djgnn/per_split_<tag>.csv`
  - `reports/djgnn/summary_<tag>.csv` — mean/std test accuracy per dataset
  - `reports/djgnn/failures_<tag>.jsonl` — structured load/preflight/train failures (never silent skips)
  - `tables/djgnn/summary_<tag>.csv` — copy of summary for tables namespace
  - `run_metadata/djgnn/meta_<tag>.json` — run config, git revision, upstream URL

## Dataset / split mapping

| This repo key | Upstream `main.py` | PyG loader (via `bfsbased-full-investigate-homophil.py`) |
|---------------|-------------------|-----------------------------------------------------------|
| cora | cora | Planetoid |
| citeseer | citeseer | Planetoid |
| pubmed | pubmed | Planetoid |
| chameleon | chamaleon (typo upstream) | WikipediaNetwork chameleon |
| texas | texas | WebKB |
| wisconsin | wisconsin | WebKB |
| actor | actor (name `film` for splits) | Actor |
| cornell | cornell | WebKB |
| squirrel | squirrel | WikipediaNetwork |

## Dependencies

No new packages beyond the existing stack: **PyTorch**, **PyTorch Geometric**, **NumPy**. Upstream lists many optional packages (matplotlib, seaborn, OGB, …) for their notebooks and `main_scalable.py`; the integrated baseline only needs the core stack above.

## Local run

From repository root (paths resolve relative to root; the runner `chdir`s to `REPO_ROOT`):

```bash
# Full-style run (long); use a Conda env with torch + torch_geometric
conda activate feedback-weighted-maximization
python3 code/bfsbased_node_classification/run_djgnn_benchmark.py \
  --output-tag my_djgnn_$(date +%Y%m%d) \
  --split-dir data/splits \
  --device cuda:0   # or cpu
```

Smoke-style (few epochs, small graph):

```bash
python3 code/bfsbased_node_classification/run_djgnn_benchmark.py \
  --output-tag smoke \
  --datasets texas \
  --splits 0 \
  --epochs 5 \
  --device cpu
```

Useful flags:

- `--max-nodes N` — skip datasets with more than `N` nodes (dense **N×N** adjacency; `0` = no limit).
- `--max-dense-adj-bytes B` — skip if a single dense adj estimate exceeds `B` bytes.
- `--epochs`, `--hidden-channels`, `--n-layers`, `--lr`, `--dropout`, `--wd` — override hyperparameters for all datasets.

## Wulver / Slurm

Adjust partition, account, and optional GPU lines to match your site.

```bash
cd /path/to/bfsbased_node_classification_git
export DJGNN_OUTPUT_TAG=djgnn_wulver_$(date +%Y%m%d_%H%M%S)
export DJGNN_CONDA_ENV=feedback-weighted-maximization
# Optional: export DJGNN_DEVICE=cuda:0

sbatch slurm/djgnn_benchmark_smoke.sbatch
sbatch slurm/djgnn_benchmark_array.sbatch    # 9 array tasks, one dataset each
sbatch slurm/djgnn_benchmark_full.sbatch     # single long job, all datasets
```

`djgnn_benchmark_array.sbatch` sets `OUT_TAG="${DJGNN_OUTPUT_TAG:-...}__${DATASET}"` so each array task writes its own `runs_<tag>.jsonl` and CSVs (no clobbering). Merge artifacts offline if you need a single table (e.g. `cat logs/djgnn/runs_${BASE}__*.jsonl > combined.jsonl`).

## Limitations / deviations

1. **Dense memory**: The upstream `DJ` model operates on a dense adjacency matrix (**O(N²)** memory). Large graphs (especially **PubMed**) may require **large CPU RAM or a GPU** with sufficient memory. Oversized graphs are skipped with a row in `failures_<tag>.jsonl` when limits are hit.
2. **`main_scalable.py`**: Not wired in; only the standard `main.py` training loop is integrated.
3. **Login-node smoke**: A short run on **Texas** (small **N**) succeeded in validation; **Cora** was OOM-killed on a low-memory login node — use Slurm with adequate memory or GPU for full benchmarks.

## Canonical pipelines

This integration does **not** modify `FINAL_V3` / `run_final_evaluation.py` behavior except adding an optional `root=` argument to `_load_data` (default `"data/"` preserves previous call sites).

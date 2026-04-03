# Extended datasets: OGB + TabGraphs (hm-categories)

This repository’s primary benchmarks use GEO-GCN-style **10 random splits** (`data/splits/*_split_0.6_0.2_{0..9}.npz`). The three datasets below use each benchmark’s **official single split** instead. Only **`split_id=0`** is supported; splits `1..9` are rejected with a clear error.

| CLI / code name   | Source | Split |
|-------------------|--------|--------|
| `ogbn-arxiv`      | [OGB node prediction](https://snap-stanford.github.io/ogb-web/docs/nodeprop/) | Official `get_idx_split()` |
| `ogbn-products`   | OGB | Official `get_idx_split()` |
| `hm-categories`   | [TabGraphs](https://github.com/yandex-research/tabgraphs) (Zenodo) | Official `train_mask.csv` / `valid_mask.csv` / `test_mask.csv` |

Aliases: `ogbn_arxiv`, `ogbn_products`, `hm_categories` (underscores) are accepted in `load_dataset` and normalized consistently.

## Dependencies

```bash
pip install ogb pandas PyYAML scikit-learn
```

(Also listed in `requirements.txt`.)

## Data layout

| Dataset | On-disk location (under repo `data/`) |
|---------|--------------------------------------|
| OGB | `data/ogb/` (created by the `ogb` package; do not commit) |
| hm-categories | `data/tabgraphs/hm-categories/` (you unzip the Zenodo archive here) |

Override for hm-categories only:

```bash
export TABGRAPHS_HM_CATEGORIES_DIR=/path/to/unzipped/hm-categories
```

## Prepare split `.npz` files (required for training runners)

From the **repository root**:

### ogbn-arxiv

```bash
python3 scripts/prepare_extended_dataset_splits.py --dataset ogbn-arxiv
```

Downloads via OGB on first run; writes `data/splits/ogbn_arxiv_split_0.6_0.2_0.npz`.

### ogbn-products

```bash
python3 scripts/prepare_extended_dataset_splits.py --dataset ogbn-products
```

Writes `data/splits/ogbn_products_split_0.6_0.2_0.npz`.

**Caveats:** ~2.4M nodes and ~61M edges — **large RAM** and long runtimes. Many single-machine workflows need a cluster or sampling-based baselines; dense adjacency methods (e.g. integrated DJ-GNN) are generally **not** appropriate at full scale.

### hm-categories (TabGraphs)

1. Download the TabGraphs benchmark from the [Zenodo record](https://zenodo.org/records/14506079) linked in the [TabGraphs README](https://github.com/yandex-research/tabgraphs).
2. Unzip the **`hm-categories`** archive into `data/tabgraphs/hm-categories/` so you have files such as `features.csv`, `edgelist.csv`, `info.yaml`, `train_mask.csv`, `valid_mask.csv`, `test_mask.csv`.
3. Run:

```bash
python3 scripts/prepare_extended_dataset_splits.py --dataset hm-categories \
  --tabgraphs-dir data/tabgraphs/hm-categories
```

Writes `data/splits/hm_categories_split_0.6_0.2_0.npz`.

> TabGraphs notes the benchmark is superseded on GitHub by **GraphLand**; this integration targets the **published TabGraphs hm-categories** layout as requested.

## Integration points

- **`load_dataset`** in `bfsbased-full-investigate-homophil.py` dispatches to `extended_graph_datasets.py`.
- **Split files** use `split_paths.split_npz_prefix()` so names stay consistent with `run_final_evaluation._load_split` and `manuscript_runner._resolve_split_file`.
- **Do not overwrite** canonical CSVs in `reports/` / `tables/`; use a new `--output-tag` for any new runs.

## Smoke tests (after prepare + deps)

From repo root, with a conda env that has `torch`, `torch_geometric`, and the packages above:

```bash
# OGB-ArXiv: one split, tagged output
python3 code/bfsbased_node_classification/run_final_evaluation.py \
  --split-dir data/splits \
  --datasets ogbn-arxiv \
  --splits 0 \
  --output-tag smoke_ogbn_arxiv_$(date +%Y%m%d) \
  --gate heuristic
```

```bash
python3 code/bfsbased_node_classification/run_final_evaluation.py \
  --split-dir data/splits \
  --datasets ogbn-products \
  --splits 0 \
  --output-tag smoke_ogbn_products_$(date +%Y%m%d) \
  --gate heuristic
```

```bash
python3 code/bfsbased_node_classification/run_final_evaluation.py \
  --split-dir data/splits \
  --datasets hm-categories \
  --splits 0 \
  --output-tag smoke_hm_categories_$(date +%Y%m%d) \
  --gate heuristic
```

Lightweight loader-only check (downloads OGB data on first use):

```bash
cd /path/to/repo && PYTHONPATH=code/bfsbased_node_classification python3 -c "
from extended_graph_datasets import load_ogb_node_dataset
d = load_ogb_node_dataset('ogbn-arxiv', 'data/')[0]
print('nodes', d.num_nodes, 'edges', d.edge_index.size(1))
"
```

For hm-categories (after unzipping TabGraphs files):

```bash
PYTHONPATH=code/bfsbased_node_classification python3 -c "
from extended_graph_datasets import load_hm_categories_dataset
d = load_hm_categories_dataset('data/')[0]
print('nodes', d.num_nodes, 'edges', d.edge_index.size(1))
"
```

## Memory and runtime caveats

- **`run_final_evaluation.py` / `manuscript_runner.py`** dynamically import the large legacy notebook module (`bfsbased-full-investigate-homophil.py`) before calling `load_dataset`. On **low-RAM login nodes**, combining that import with **full-graph** tensors for **ogbn-arxiv** (~169k nodes, ~1.1M undirected edges after symmetrization) can hit **OOM** (process killed). Prefer a **Slurm / HPC** allocation with sufficient RAM, or validate loaders using `extended_graph_datasets.load_ogb_node_dataset` alone.
- **ogbn-products** is far larger; treat full-graph experiments as **cluster-scale** unless you implement sampling yourself (out of scope here).
- **hm-categories** size depends on the Zenodo drop; preprocessing uses **pandas + scikit-learn** and mirrors the TabGraphs GNN dataloader’s feature pipeline for **multiclass** tasks only.

## References

- OGB: [Node property prediction](https://snap-stanford.github.io/ogb-web/docs/nodeprop/), [paper](https://cs.stanford.edu/people/jure/pubs/ogb-neurips20.pdf).
- TabGraphs: [GitHub](https://github.com/yandex-research/tabgraphs), [paper HTML](https://arxiv.org/html/2409.14500v1).

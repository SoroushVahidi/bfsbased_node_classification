# Begga-style heterophily baselines (external comparison)

This repository’s **canonical** method is **FINAL_V3**. The models below are **external
baselines only** for comparison. They share the repo’s **60/20/20 split protocol**
(`data/splits/*.npz`), **JSONL/CSV logging** via `resubmission_runner.py`, and
**output-tag** safety (never overwrite frozen evidence without an explicit tag).

| Baseline | `run_baseline` / resubmission `method` | Upstream reference | Integration |
|----------|----------------------------------------|--------------------|-------------|
| **H2GCN** | `h2gcn` | [GemsLab/H2GCN](https://github.com/GemsLab/H2GCN) | **Verified / existing**: compact PyTorch-style 1-hop/2-hop separation + concat; no signac/TF stack. |
| **GCNII** | `gcnii` | [chennnM/GCNII](https://github.com/chennnM/GCNII) | **New**: inline PyTorch GCNII layers + sym-normalized adjacency (Chen et al.). |
| **Geom-GCN** | `geom_gcn` | [bingzhewei/geom-gcn](https://github.com/bingzhewei/geom-gcn) | **New**: **compact** bi-branch aggregation (row-norm “in” + reversed-edge “out”); not the full directional preprocessing/splits from the official release. |
| **LINKX** | `linkx` | [CUAI/Non-Homophily-Large-Scale](https://github.com/CUAI/Non-Homophily-Large-Scale) | **New**: feature MLP + structure MLP on row-normalized adjacency (Lim et al.). |
| **ACMII-GCN++** (family) | `acmii_gcn_plus_plus` | [SitaoLuan/ACM-GNN](https://github.com/SitaoLuan/ACM-GNN) | **New**: **single-layer** ACM channel mixing (LP/HP/ID) per Luan et al. The resubmission label **`acmii_gcn_plus_plus`** denotes this repo’s **ACM-style** compact baseline for tables that reference the ACM/ACM-II family; it is **not** a full reproduction of every multi-layer “++” variant in the upstream codebase. |

Archived inline reimplementations under `archive/exploratory/benchmark/runners/` (same five families) are **superseded** for the main pipeline by `standard_node_baselines.py` + `resubmission_runner.py`.

## How to run

**Resubmission JSONL (recommended; uses same protocol as other baselines):**

```bash
python3 code/bfsbased_node_classification/resubmission_runner.py \
  --datasets texas \
  --splits 0 \
  --repeats 1 \
  --methods h2gcn gcnii geom_gcn linkx acmii_gcn_plus_plus \
  --output-tag begga_smoke_$(date +%Y%m%d)
```

**Programmatic:**

```python
from standard_node_baselines import run_baseline, BEGGA_HETEROPHILY_BASELINE_METHODS
# run_baseline("gcnii", data, train_idx, val_idx, test_idx, seed=42, max_epochs=500, patience=100)
```

## Smoke test (one split, CPU)

```bash
python3 scripts/smoke_begga_baselines.py
```

## Dependencies

Requires **PyTorch** and **torch_geometric** (for `add_self_loops`, `remove_self_loops`, `degree`).
See root `requirements.txt`. CPU-only installs are supported.

## Deviations from upstream (summary)

- **H2GCN**: No TensorFlow/signac; 2-hop edges built without `torch_sparse` spspmm (see `standard_node_baselines.py` docstring).
- **Geom-GCN**: Minimal geometric branches on the **current** undirected `edge_index` instead of the official geom splits/preprocessing pipeline.
- **ACM**: Single ACM mixing layer as a **stand-in** for paper naming “ACMII-GCN++” in broad comparison tables.

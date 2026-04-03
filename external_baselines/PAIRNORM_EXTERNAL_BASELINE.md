# PairNorm external baseline integration note

## What PairNorm is

**PairNorm** (Zhao & Akoglu, ICLR 2020) is a normalization method for GNN
hidden representations designed to mitigate over-smoothing in deep message
passing models.

PairNorm is a normalization layer, not a standalone graph architecture. In this
repository it is integrated as:

- **DeepGCN + PairNorm** baseline (`gcn_pairnorm`)

## Source reference

- Official implementation: https://github.com/LingxiaoShawn/PairNorm

## Integration choices in this repository

- Added a compact `PairNorm` module and `DeepGCNPairNorm` model inside
  `code/bfsbased_node_classification/standard_node_baselines.py`.
- Added runner:
  `code/bfsbased_node_classification/pairnorm_baseline_runner.py`.
- Reused repository split protocol (`data/splits`) and logging/reporting style
  (`JSONL + summary CSV`).
- Wired `gcn_pairnorm` into `resubmission_runner.py` for comparison runs.

## Deviations from upstream

- Only minimal PairNorm logic was adapted (no full repo transplant).
- This integration focuses on **DeepGCN + PairNorm** as the practical baseline
  path for this codebase; GAT-based PairNorm variants are not included.
- Uses this repo's training loop and compact validation grid conventions.

## Not canonical FINAL_V3

PairNorm here is an **external comparison baseline** and is not part of the
canonical `FINAL_V3` package.

## Smoke test command

```bash
python3 code/bfsbased_node_classification/pairnorm_baseline_runner.py \
  --split-dir data/splits --datasets cora --splits 0 \
  --output-tag smoke_pairnorm_YYYYMMDD
```

# FSGNN external baseline integration note

## What FSGNN is

**FSGNN** (Feature Selection Graph Neural Network) is a node-classification
model that builds a bank of hop-wise propagated feature matrices and learns a
soft feature-selection weighting over those hops before final classification.

In this repository we use a compact variant following the public code
structure:

1. Build hop feature bank from normalized adjacency powers (with and without
   self-loops).
2. Apply per-hop linear projections.
3. Learn softmax attention weights over hop branches.
4. Concatenate, activate, dropout, then classify.

## Source reference

- Official/public implementation:
  https://github.com/sunilkmaurya/FSGNN

## Integration choices in this repository

- Implemented inline under
  `code/bfsbased_node_classification/standard_node_baselines.py`.
- Added standalone runner
  `code/bfsbased_node_classification/fsgnn_baseline_runner.py`.
- Reused repository split loading (`data/splits`, 60/20/20 protocol) and
  logging style (`JSONL + summary CSV`).
- Kept CPU-compatible training and conservative default output tagging.

## Deviations from upstream

- Adapted to this repository's runner/evaluation conventions instead of
  upstream scripts.
- Uses a compact validation-selected hyperparameter grid (including
  `feat_type` in `{all, homophily, heterophily}`) rather than full upstream
  script parameter sweeps.
- No full upstream repo vendoring; only minimal model/training logic was
  adapted.

## Not canonical FINAL_V3

FSGNN here is an **external comparison baseline** and is not part of the
canonical `FINAL_V3` method.

## Smoke test command

```bash
python3 code/bfsbased_node_classification/fsgnn_baseline_runner.py \
  --split-dir data/splits --datasets cora --splits 0 \
  --output-tag smoke_fsgnn_YYYYMMDD
```

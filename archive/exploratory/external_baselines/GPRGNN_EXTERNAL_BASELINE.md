# GPRGNN external baseline integration note

## What GPRGNN is

**GPRGNN** (Adaptive Universal Generalized PageRank Graph Neural Network) is a
node-classification model that combines:

1. a feature MLP (`X -> hidden -> logits`), and
2. a learnable generalized PageRank propagation layer over graph structure.

The propagation output is:

\[
\sum_{k=0}^{K} \gamma_k S^k Z
\]

where \(Z\) are pre-propagation logits from the MLP, \(S\) is normalized graph
adjacency, and \(\gamma_k\) are learned coefficients.

## Source reference

- Official repository used as implementation reference:
  https://github.com/jianhao2016/GPRGNN
- Paper: Chien et al., ICLR 2021, "Adaptive Universal Generalized PageRank
  Graph Neural Network."

## Integration choices in this repository

This repo uses a **compact inline implementation** in
`code/bfsbased_node_classification/standard_node_baselines.py` and does **not**
vendor the full upstream repository.

### Why this was chosen

- Keeps dependency footprint aligned with existing stack (PyTorch + PyG-style
  sparse propagation utilities already used here).
- Reuses this repository's canonical GEO-GCN split files in `data/splits/`.
- Keeps training/evaluation logging format identical to existing baseline
  runners (`JSONL + summary CSV`).

### Deviations from upstream

- The training loop is adapted to this repository's fixed split protocol
  (predefined train/val/test masks), rather than the upstream random split
  helper path.
- Hyperparameter search uses a compact validation grid tailored to existing
  runner conventions (`lr`, `weight_decay`, `alpha`, `dprate`, `Init`), rather
  than reproducing every setting from upstream scripts.
- Execution target is CPU by default for reproducibility with the existing
  pipeline.

## Not canonical FINAL_V3

This baseline is **external comparison only** and is not part of the canonical
`FINAL_V3` method package.

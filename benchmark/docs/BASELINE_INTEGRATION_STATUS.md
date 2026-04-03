# Baseline Integration Status

This document tracks the integration status of all 8 GNN baselines for the node classification benchmark.

## Summary Table

| Method | Status | Source | License | Notes |
|--------|--------|--------|---------|-------|
| GPRGNN | ✅ wrapped | [GitHub](https://github.com/jianhao2016/GPRGNN) | MIT | Inline PyG reimplementation |
| GCNII | ✅ wrapped | [GitHub](https://github.com/chennnM/GCNII) | MIT | Inline reimplementation, up to 64 layers |
| H2GCN | ✅ wrapped | [GitHub](https://github.com/GemsLab/H2GCN) | MIT | Inline reimplementation, ego+1hop+2hop |
| LINKX | ✅ wrapped | [GitHub](https://github.com/CUAI/Non-Homophily-Large-Scale) | MIT | Inline reimplementation, feature+structure branches |
| FSGNN | ✅ wrapped | [GitHub](https://github.com/sunilkmaurya/FSGNN) | MIT | Inline reimplementation, multi-scale smoothing |
| ACM-GNN | ✅ wrapped | [GitHub](https://github.com/SitaoLuan/ACM-GNN) | MIT | Inline reimplementation, LP+HP+ID mixing |
| OrderedGNN | 🔲 planned | [GitHub](https://github.com/LUMIA-Group/OrderedGNN) | MIT | Phase C: chunk-based ordered propagation |
| DJ-GNN | 🔲 planned | [GitHub](https://github.com/BUPT-GAMMA/DJ-GNN) | MIT | Phase C: decoupled feature+topology learning |

## Detailed Status

### GPRGNN (Adaptive Universal Generalized PageRank GNN)
- **Paper:** Chien et al., ICLR 2021
- **Status:** Wrapped (inline reimplementation)
- **What was done:** Implemented GPR propagation with learnable softmax-normalized coefficients initialized from Personalized PageRank (PPR). Uses sparse D^{-1/2}(A+I)D^{-1/2} normalization.
- **Architecture:** MLP (2-layer) + K-hop GPR propagation with gamma weights
- **Deviations from paper:** Coefficients initialized via PPR then optimized; softmax normalization applied to gamma weights for stability.

### GCNII (Simple and Deep Graph Convolutional Networks)
- **Paper:** Chen et al., ICML 2020
- **Status:** Wrapped (inline reimplementation)
- **What was done:** Implemented initial residual connections and identity mapping. beta = log(theta/l + 1) per layer. W^(l) = (1-beta)*I + beta*W_tilde.
- **Architecture:** Linear embedding → num_layers GCNII convolutions → Linear classifier
- **Config:** Default 64 layers (set via `num_layers` in `gcnii.yaml`; config file takes precedence over the code default of 16).

### H2GCN (Beyond Homophily in Graph Neural Networks)
- **Paper:** Zhu et al., NeurIPS 2020
- **Status:** Wrapped (inline reimplementation)
- **What was done:** Implemented ego embedding (r0) + 1-hop neighbor aggregation (r1) + 2-hop neighbor aggregation (r2). Concatenation [r0, r1, r2] → classifier.
- **Architecture:** Separate W0 (ego) and W1 (neighbors, shared), row-normalized adjacency without self-loops.
- **Deviations from paper:** Uses row normalization for neighbor aggregation. num_rounds fixed at 2.

### LINKX
- **Paper:** Lim et al., NeurIPS 2021
- **Status:** Wrapped (inline reimplementation)
- **What was done:** Separate feature MLP on x and structure MLP on A@x (1-hop aggregation). Additive combination → final linear layer.
- **Architecture:** MLP_f(x) + MLP_a(A@x) → ReLU → Linear → C
- **Note:** Uses 1-hop aggregated features as structure signal (simplified from original which uses full adjacency rows).

### FSGNN (Feature Selection GNN)
- **Paper:** Maurya et al.
- **Status:** Wrapped (inline reimplementation)
- **What was done:** Pre-computes K+1 smoothed feature matrices [x, A@x, ..., A^K@x] and concatenates them. Simple MLP classifier on concatenated features.
- **Architecture:** Precomputed smoothed features → Linear(K+1)*d → ReLU → Dropout → Linear → C
- **Note:** K=3 by default (4 feature scales total).

### ACM-GNN (Adaptive Channel Mixing GNN)
- **Paper:** Luan et al., NeurIPS 2022
- **Status:** Wrapped (inline reimplementation)
- **What was done:** Computes low-pass (LP: A_sym@H), high-pass (HP: H - A_sym@H), and identity (ID: H) channels. Per-node softmax attention selects the mixture.
- **Architecture:** Linear embed → ACMLayer(LP, HP, ID mixing with attention) → Dropout → Linear → C

### OrderedGNN
- **Paper:** Wei et al., ICLR 2023
- **Status:** Planned (Phase C)
- **Blocker:** Requires chunk-based ordered propagation mechanism with specialized memory ordering of message passing. Integration requires more careful architectural alignment with the official code.
- **Next steps:** Clone official repo to `external_baselines/orderedgnn/`, wrap in `run_orderedgnn.py`.

### DJ-GNN
- **Status:** Planned (Phase C)
- **Blocker:** Requires a two-stage decoupled architecture for feature learning and topology refinement with a specialized joint training objective.
- **Next steps:** Clone official repo to `external_baselines/djgnn/`, wrap in `run_djgnn.py`.

## Fairness and Split Protocol

All integrated baselines use:
- **Identical splits:** The same canonical `.npz` split files in `data/splits/` used by the main FINAL_V3 method.
- **Split format:** `{dataset}_split_0.6_0.2_{split_id}.npz` (train/val/test ratio ~60/20/20%)
- **Seeding:** Each run uses `torch.manual_seed(seed)` before model instantiation.
- **Early stopping:** Patience-based stopping on validation accuracy (patience=100 epochs by default).
- **Hyperparameter selection:** Default configs in `benchmark/configs/` are used; no dataset-specific tuning.
- **Test set:** Evaluated at the epoch with best validation accuracy; not used for hyperparameter selection.

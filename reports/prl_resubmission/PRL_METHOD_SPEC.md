# PRL method specification recovered from code

> **Important:** This document describes **selective graph correction as implemented in** `bfsbased-full-investigate-homophil.py` / **`manuscript_runner.py`** (margin-based uncertainty gate, `threshold_high`, multi-weight search). That corresponds to **SGC v1** in the FINAL_V3 benchmark table — **not** the submission method **FINAL_V3**, which adds the **reliability score R(v)**, **ρ**, and a reduced profile grid (see `reports/final_method_v3_analysis.md` and `code/bfsbased_node_classification/final_method_v3.py`).

Canonical implementation:
- `code/bfsbased_node_classification/bfsbased-full-investigate-homophil.py`
- Primary driver:
  `code/bfsbased_node_classification/manuscript_runner.py`

## Operational definition

1. **Base classifier**
- A feature-only MLP is trained first.
- Manuscript-default configuration is now standardized in code as
  `DEFAULT_MANUSCRIPT_MLP_KWARGS = {hidden=64, layers=2, dropout=0.5, lr=0.01, epochs=300}`.
- The main run path uses the MLP as the primary classifier before any graph correction.

2. **Uncertainty score**
- For each node, the code computes the MLP class probabilities.
- Confidence is summarized by the **margin**:
  `top1_prob - top2_prob`.
- Lower margin means more uncertainty.

3. **Selective gate**
- The gate is defined by a global validation-selected threshold `threshold_high`.
- A node is marked uncertain iff:
  `mlp_margin < threshold_high`.
- Confident nodes keep the MLP prediction unchanged.

4. **Correction evidence**
- For each node and class, the method builds:
  - `feature_similarity`
  - `feature_knn_vote` (optional; off by default)
  - `graph_neighbor_support`
  - `compatibility_support`
- Evidence is built once from the graph, train labels, MLP probabilities, and node features.

5. **Combined class score**
- The final class-wise score is:
  `b1 * log(p_mlp) + b2 * feature_similarity + b3 * feature_knn_vote + b4 * graph_neighbor_support + b5 * compatibility_support`
- This is implemented in `build_selective_correction_scores(...)`.

6. **Weight selection**
- The current canonical code evaluates a compact candidate set of weight vectors on the validation split.
- Default candidate set:
  - `(1.0, 0.6, 0.0, 0.5, 0.3)`
  - `(1.0, 0.9, 0.0, 0.3, 0.2)`
  - `(1.0, 0.4, 0.0, 0.9, 0.5)`
  - `(1.2, 0.5, 0.0, 0.4, 0.2)`

7. **Threshold selection**
- For each candidate weight vector, the code searches threshold candidates on the **validation** set only.
- Default threshold candidates are the union of:
  - fixed values `0.05, 0.10, 0.15, 0.20, 0.25, 0.30`
  - validation-margin quantiles `0.2..0.8`
- Selection objective is lexicographic:
  1. maximize validation accuracy
  2. among ties, minimize changed fraction
  3. among ties, minimize uncertain fraction

8. **Final prediction**
- For confident nodes: keep MLP prediction.
- For uncertain nodes: replace with the argmax of the selected weighted combined score.

## Important manuscript-facing clarifications

- This is a **feature-first** method: the MLP is always the starting point.
- Graph information is **not** globally propagated to all nodes.
- The gate and the score weights are selected on **validation only**.
- `feature-kNN` exists in code but is **disabled by default** in the current canonical reported configuration.
- The aggregated correction-behavior table is therefore **descriptive**, not by itself a causal ablation.

## New PRL-resubmission package

To strengthen the paper beyond descriptive diagnostics, the repo now adds:
- compact standard graph baselines: `GCN`, `APPNP`
- true SGC ablations:
  - `sgc_no_gate`
  - `sgc_no_feature_similarity`
  - `sgc_no_graph_neighbor`
  - `sgc_no_compatibility`
  - `sgc_mlp_plus_graph`
  - `sgc_knn_enabled`
- focused runner:
  `code/bfsbased_node_classification/prl_resubmission_runner.py`

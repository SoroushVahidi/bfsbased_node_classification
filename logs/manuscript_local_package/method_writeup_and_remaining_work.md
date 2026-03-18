# Method write-up support and remaining-work checklist

## Selective_graph_correction_predictclass (concise method description)

### Input / output
- **Input:** graph data (features, edges, labels/masks), trained or trainable MLP, and validation/test masks under canonical splits.
- **Output:** final node predictions and test accuracy, plus diagnostics (uncertain fraction, changed-from-MLP fraction, selected threshold/weights, node-level component scores).

### Uncertainty criterion
For each node, compute MLP class probabilities and margin `margin = p_top1 - p_top2`. A node is treated as **uncertain** if its margin is below a validation-selected threshold `threshold_high`.

### Correction score
For uncertain nodes only, compute class-wise correction scores:

S_c(v) = b1 * log(p_mlp_c(v) + eps)
       + b2 * feature_similarity_c(v)
       + b3 * feature_knn_c(v)
       + b4 * graph_neighbor_support_c(v)
       + b5 * compatibility_support_c(v)

In the current local evidence package, feature-kNN is disabled (`b3` contribution effectively zero).

### Threshold/weight selection
`threshold_high` and weight tuple `(b1..b5)` are chosen using validation nodes only. The chosen settings are then frozen and applied to test-time uncertain nodes.

### Fallback behavior
Confident nodes keep MLP predictions unchanged. In low-uncertainty/weak-graph regimes, selective correction often changes very few (or zero) nodes, producing near-MLP behavior by design.

## Why this is different from propagation-first methods
Propagation-first methods make graph diffusion the primary prediction mechanism. Here, the MLP remains the primary classifier, and graph evidence is used selectively and interpretably as a correction layer only on low-margin nodes. This changes both the optimization target (protect strong feature predictions first) and the failure mode (prefer MLP-like fallback when correction evidence is weak), which is central to the regime-aware narrative.

# Minimum remaining Wulver work

## A. essential before drafting
1. Final larger validation run with the same method set and diagnostics (no new method family), expanding dataset coverage and repeats.
2. Confirmatory repeat-based analysis of gain stability and sign tests under larger compute budget.
3. Final manuscript-grade statistics bundle (paired tests/CIs across all final datasets and splits).
4. Finalized figure-generation pass from the expanded run outputs (same figure schema as local package).

## B. nice to have
1. Additional control datasets beyond the local set to stress-test regime boundaries.
2. Extra ablation slices (e.g., stricter fallback thresholds) only if needed to answer reviewer concerns.
3. Rebuttal-ready supplementary diagnostics at larger scale (more case-study batches).

# Should we go to Wulver now?
Yes. The high-value local manuscript structuring work is now in place; what remains is predominantly compute-heavy confirmatory validation and expanded repeat coverage.

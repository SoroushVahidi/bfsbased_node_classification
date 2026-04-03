# PRL resubmission experiment summary

## Datasets

- `cora`

## Baseline package

- Existing baselines retained: `mlp_only`, `prop_only`.
- Newly added standard graph baselines: `gcn`, `appnp`.
- Alternative gate baseline: `gated_mlp_prop`.

## Ablations

- `sgc_no_gate` = SGC without gate
- `sgc_no_feature_similarity` = SGC w/o feature similarity
- `sgc_no_graph_neighbor` = SGC w/o graph-neighbor support
- `sgc_no_compatibility` = SGC w/o compatibility
- `sgc_mlp_plus_graph` = SGC MLP+graph only
- `sgc_knn_enabled` = SGC with feature-kNN enabled

## Main caveat

- This package strengthens baselines and ablations, but it does not convert the paper into a universal multi-regime claim.

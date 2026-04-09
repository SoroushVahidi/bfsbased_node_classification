# PRL resubmission experiment summary

## Datasets

- `chameleon`
- `citeseer`
- `cora`
- `pubmed`
- `wisconsin`

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
- `selective_graph_correction_structural` = UG-SGC-S structural
- `sgcs_margin_gate_only` = UG-SGC-S without structure-aware gate
- `sgcs_no_far` = UG-SGC-S near only
- `sgcs_far_only` = UG-SGC-S far only
- `sgcs_no_gate` = UG-SGC-S without gate

## Main caveat

- This package strengthens baselines and ablations, but it does not convert the paper into a universal multi-regime claim.

# Manuscript Fairness Report — manuscript_final_validation_job2
Generated: 2026-03-20T18:32:13

## Core Result Table
| Dataset | mlp_only | prop_only | selective_graph_correction | prop_mlp_select |
|---------|----------|-----------|---------------------------|-----------------|
| pubmed | 0.8723 ± 0.0037 | 0.8090 ± 0.0053 | 0.8841 ± 0.0036 | N/A |
| squirrel | 0.3218 ± 0.0171 | 0.3035 ± 0.0184 | 0.3220 ± 0.0174 | N/A |

## Gain Over MLP (selective_graph_correction - mlp_only)
| Dataset | Delta | SGC wins | MLP wins | Ties |
|---------|-------|----------|----------|------|
| pubmed | +0.0117 | 30 | 0 | 0 |
| squirrel | +0.0002 | 14 | 14 | 2 |

## Notes on Fairness
- All methods use **canonical GEO-GCN splits** (no random fallback).
- MLP is trained with early stopping on val set for all methods.
- SGC threshold and b-weights are tuned on val set; test set is never seen during tuning.
- prop_only params are tuned via `robust_random_search` on train k-fold splits (no val leakage).
- prop_mlp_select uses val accuracy to choose between MLP and prop_only (no test leakage).

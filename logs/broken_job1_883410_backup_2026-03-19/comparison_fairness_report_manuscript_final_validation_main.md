# Manuscript Fairness Report — manuscript_final_validation_main
Generated: 2026-03-18T20:25:00

## Core Result Table
| Dataset | mlp_only | prop_only | selective_graph_correction | prop_mlp_select |
|---------|----------|-----------|---------------------------|-----------------|
| actor | N/A | 0.3213 ± 0.0189 | N/A | N/A |
| chameleon | N/A | 0.3819 ± 0.0412 | N/A | N/A |
| citeseer | N/A | 0.6815 ± 0.0377 | N/A | N/A |
| cora | N/A | 0.8142 ± 0.0261 | N/A | N/A |
| cornell | N/A | 0.5009 ± 0.0887 | N/A | N/A |
| texas | N/A | 0.5703 ± 0.1055 | N/A | N/A |
| wisconsin | N/A | 0.5386 ± 0.0874 | N/A | N/A |

## Gain Over MLP (selective_graph_correction - mlp_only)
| Dataset | Delta | SGC wins | MLP wins | Ties |
|---------|-------|----------|----------|------|

## Notes on Fairness
- All methods use **canonical GEO-GCN splits** (no random fallback).
- MLP is trained with early stopping on val set for all methods.
- SGC threshold and b-weights are tuned on val set; test set is never seen during tuning.
- prop_only params are tuned via `robust_random_search` on train k-fold splits (no val leakage).
- prop_mlp_select uses val accuracy to choose between MLP and prop_only (no test leakage).

# Manuscript Fairness Report — manuscript_final_validation_main
Generated: 2026-03-19T21:58:03

## Core Result Table
| Dataset | mlp_only | prop_only | selective_graph_correction | prop_mlp_select |
|---------|----------|-----------|---------------------------|-----------------|
| actor | 0.3439 ± 0.0066 | 0.3213 ± 0.0189 | 0.3430 ± 0.0076 | N/A |
| chameleon | 0.4620 ± 0.0185 | 0.3819 ± 0.0412 | 0.4620 ± 0.0199 | N/A |
| citeseer | 0.6797 ± 0.0192 | 0.6815 ± 0.0377 | 0.7049 ± 0.0202 | N/A |
| cora | 0.7074 ± 0.0221 | 0.8142 ± 0.0261 | 0.7425 ± 0.0217 | N/A |
| cornell | 0.7189 ± 0.0590 | 0.5009 ± 0.0887 | 0.7189 ± 0.0607 | N/A |
| texas | 0.7874 ± 0.0635 | 0.5703 ± 0.1055 | 0.7883 ± 0.0688 | N/A |
| wisconsin | 0.8471 ± 0.0479 | 0.5386 ± 0.0874 | 0.8451 ± 0.0476 | N/A |

## Gain Over MLP (selective_graph_correction - mlp_only)
| Dataset | Delta | SGC wins | MLP wins | Ties |
|---------|-------|----------|----------|------|
| actor | -0.0009 | 10 | 16 | 4 |
| chameleon | -0.0000 | 13 | 13 | 4 |
| citeseer | +0.0251 | 30 | 0 | 0 |
| cora | +0.0351 | 30 | 0 | 0 |
| cornell | -0.0000 | 5 | 5 | 20 |
| texas | +0.0009 | 5 | 4 | 21 |
| wisconsin | -0.0020 | 1 | 3 | 26 |

## Notes on Fairness
- All methods use **canonical GEO-GCN splits** (no random fallback).
- MLP is trained with early stopping on val set for all methods.
- SGC threshold and b-weights are tuned on val set; test set is never seen during tuning.
- prop_only params are tuned via `robust_random_search` on train k-fold splits (no val leakage).
- prop_mlp_select uses val accuracy to choose between MLP and prop_only (no test leakage).

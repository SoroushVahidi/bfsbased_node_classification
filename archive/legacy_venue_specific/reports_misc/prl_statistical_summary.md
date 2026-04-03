# PRL statistical summary

This summary uses the frozen split-level rows in `reports/final_method_v3_results.csv`.
Each dataset contributes 10 paired split realizations for `MLP_ONLY`, `BASELINE_SGC_V1`, `V2_MULTIBRANCH`, and `FINAL_V3`.

| Dataset | Method | Mean Δ vs MLP (pp) | 95% bootstrap CI (pp) | Std Δ (pp) | Wins | Losses | Ties |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Chameleon | BASELINE_SGC_V1 | +0.15 | [+0.07, +0.26] | 0.17 | 5 | 0 | 5 |
| Chameleon | V2_MULTIBRANCH | +0.11 | [-0.04, +0.26] | 0.25 | 6 | 3 | 1 |
| Chameleon | FINAL_V3 | +0.02 | [-0.11, +0.15] | 0.21 | 2 | 1 | 7 |
| Citeseer | BASELINE_SGC_V1 | +2.67 | [+2.01, +3.35] | 1.09 | 10 | 0 | 0 |
| Citeseer | V2_MULTIBRANCH | +2.58 | [+2.02, +3.14] | 0.90 | 10 | 0 | 0 |
| Citeseer | FINAL_V3 | +2.59 | [+1.95, +3.27] | 1.05 | 10 | 0 | 0 |
| Cora | BASELINE_SGC_V1 | +3.06 | [+2.29, +3.84] | 1.28 | 10 | 0 | 0 |
| Cora | V2_MULTIBRANCH | +2.94 | [+2.29, +3.62] | 1.10 | 10 | 0 | 0 |
| Cora | FINAL_V3 | +2.90 | [+2.31, +3.52] | 0.99 | 10 | 0 | 0 |
| Pubmed | BASELINE_SGC_V1 | +1.28 | [+1.15, +1.44] | 0.24 | 10 | 0 | 0 |
| Pubmed | V2_MULTIBRANCH | +1.26 | [+1.14, +1.41] | 0.22 | 10 | 0 | 0 |
| Pubmed | FINAL_V3 | +1.29 | [+1.16, +1.44] | 0.23 | 10 | 0 | 0 |
| Texas | BASELINE_SGC_V1 | +0.00 | [-1.35, +1.35] | 2.09 | 3 | 3 | 4 |
| Texas | V2_MULTIBRANCH | +0.27 | [-0.54, +1.08] | 1.46 | 2 | 1 | 7 |
| Texas | FINAL_V3 | +0.54 | [-0.54, +1.62] | 1.62 | 3 | 1 | 6 |
| Wisconsin | BASELINE_SGC_V1 | -0.00 | [-0.59, +0.59] | 0.88 | 1 | 1 | 8 |
| Wisconsin | V2_MULTIBRANCH | +0.20 | [+0.00, +0.59] | 0.59 | 1 | 0 | 9 |
| Wisconsin | FINAL_V3 | +0.20 | [+0.00, +0.59] | 0.59 | 1 | 0 | 9 |

## Notes

- These are paired split-level comparisons against `MLP_ONLY` within the frozen 10-split benchmark.
- Bootstrap CIs summarize uncertainty in the mean paired delta over the available splits.
- This is manuscript-support evidence, not a claim of universal statistical superiority.

# PRL Publication Tables (historical snapshot)

> **For the manuscript, use** `tables/main_results_prl.md` **(regenerated from** `reports/final_method_v3_results.csv` **) so all PRL-facing numbers stay aligned.**
>
> The per-split source rows live in `final_tables_prl.csv` in this folder and feed `scripts/prl_final_additions/rebuild_final_v3_results_10split.py`.

## Table 1: Test Accuracy (mean ± std, 10 splits)

| Dataset | MLP | SGC (v1) | V2 Multibranch | **FINAL V3** |
|---|---|---|---|---|
| cora | 71.27 ± 2.69 | 74.33 ± 2.00 | 74.21 ± 2.08 | 74.16 ± 2.14 |
| citeseer | 67.97 ± 1.74 | 70.64 ± 1.83 | 70.55 ± 1.87 | 70.56 ± 1.83 |
| pubmed | 87.20 ± 0.35 | 88.48 ± 0.35 | 88.46 ± 0.36 | 88.49 ± 0.35 |
| chameleon | 46.32 ± 2.37 | 46.47 ± 2.37 | 46.43 ± 2.43 | 46.34 ± 2.41 |
| texas | 80.54 ± 6.71 | 80.54 ± 7.63 | 80.81 ± 7.59 | 81.08 ± 7.55 |
| wisconsin | 84.31 ± 4.21 | 84.31 ± 4.02 | 84.51 ± 4.06 | 84.51 ± 4.06 |
| **Mean** | 72.93 | **74.13** | **74.16** | **74.19** |

## Table 2: Improvement over MLP (Δ percentage points)

| Dataset | SGC (v1) | V2 Multibranch | **FINAL V3** |
|---|---|---|---|
| cora | +3.06 | +2.94 | +2.90 |
| citeseer | +2.67 | +2.58 | +2.59 |
| pubmed | +1.28 | +1.26 | +1.29 |
| chameleon | +0.15 | +0.11 | +0.02 |
| texas | +0.00 | +0.27 | +0.54 |
| wisconsin | -0.00 | +0.20 | +0.20 |
| **Mean** | **+1.19** | **+1.23** | **+1.26** |

## Table 3: Win / Tie / Loss vs MLP (per split)

| Dataset | SGC v1 W/T/L | V2 W/T/L | **V3 W/T/L** |
|---|---|---|---|
| cora | 10/0/0 | 10/0/0 | 10/0/0 |
| citeseer | 10/0/0 | 10/0/0 | 10/0/0 |
| pubmed | 10/0/0 | 10/0/0 | 10/0/0 |
| chameleon | 5/5/0 | 6/1/3 | 2/7/1 |
| texas | 3/4/3 | 2/7/1 | 3/6/1 |
| wisconsin | 1/8/1 | 1/9/0 | 1/9/0 |

## Table 4: Safety — Harmful Corrections

| Dataset | V3 total helped | V3 total hurt | V3 net | V3 precision |
|---|---|---|---|---|
| cora | 157 | 13 | +144 | 0.92 |
| citeseer | 199 | 39 | +160 | 0.82 |
| pubmed | 834 | 326 | +508 | 0.72 |
| chameleon | 10 | 9 | +1 | 0.52 |
| texas | 3 | 1 | +2 | 0.75 |
| wisconsin | 1 | 0 | +1 | 1.00 |

## Table 5: Safety — Splits where SGC v1 harms (Δ < 0)

| Dataset | Split | V1 Δ | V3 Δ |
|---|---|---|---|
| texas | 4 | -2.70 | +0.00 |
| texas | 5 | -2.70 | -2.70 |
| texas | 9 | -2.70 | +0.00 |
| wisconsin | 8 | -1.96 | +0.00 |

## Table 6: V3 Behavior Summary

| Dataset | Correction rate | Precision | Helped/split | Hurt/split |
|---|---|---|---|---|
| cora | 23.7% | 0.92 | 15.7 | 1.3 |
| citeseer | 20.0% | 0.82 | 19.9 | 3.9 |
| pubmed | 23.9% | 0.72 | 83.4 | 32.6 |
| chameleon | 5.8% | 0.52 | 1.0 | 0.9 |
| texas | 2.2% | 0.75 | 0.3 | 0.1 |
| wisconsin | 1.8% | 1.00 | 0.1 | 0.0 |

## Table 7: Ablation Study (mean Δ vs MLP)

| Ablation | chameleon | cora | texas | Mean |
|---|---|---|---|---|
| FULL_V3 | +0.00 | +2.49 | +1.08 | +1.19 |
| A1_NO_RELIABILITY | +0.22 | +2.58 | +0.54 | +1.11 |
| A2_PROFILE_A_ONLY | +0.04 | +1.41 | +1.08 | +0.84 |
| A3_PROFILE_B_ONLY | -0.09 | +2.49 | +1.08 | +1.16 |

### Ablation — Harmful corrections (total hurt)

| Ablation | chameleon | cora | texas |
|---|---|---|---|
| FULL_V3 | 6 | 8 | 0 |
| A1_NO_RELIABILITY | 7 | 11 | 1 |
| A2_PROFILE_A_ONLY | 5 | 4 | 0 |
| A3_PROFILE_B_ONLY | 6 | 8 | 0 |

## Table 8: Sensitivity to ρ (mean Δ vs MLP, 3 splits)

| ρ | chameleon | cora | texas | Mean | Total hurt |
|---|---|---|---|---|---|
| 0.0 | +0.15 | +2.21 | +0.90 | +1.09 | 11 |
| 0.2 | +0.15 | +2.21 | +0.90 | +1.09 | 11 |
| 0.3 | +0.15 | +2.08 | +0.90 | +1.04 | 10 |
| 0.4 | +0.15 | +1.48 | +0.00 | +0.54 | 5 |
| 0.5 | -0.07 | +0.47 | +0.00 | +0.13 | 1 |
| 0.6 | +0.00 | +0.00 | +0.00 | +0.00 | 0 |
| 0.7 | +0.00 | +0.00 | +0.00 | +0.00 | 0 |

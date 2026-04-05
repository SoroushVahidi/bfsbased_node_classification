# Baseline comparison (automated summary)

This report is generated from merged JSONL shards. Interpret with paper context.

## Mean test accuracy (higher is better)

| Method | chameleon | citeseer | cora | cornell | texas | wisconsin | Overall mean |
|---|---|---|---|---|---|---|---|
| ACMII_GCN_PLUSPLUS | 0.4811 | 0.7533 | 0.8666 | 0.7432 | 0.8514 | 0.8235 | 0.7532 |
| BASELINE_SGC_V1 | 0.4671 | 0.7063 | 0.7421 | 0.7081 | 0.8054 | 0.8431 | 0.7120 |
| FINAL_V3 | 0.4662 | 0.7053 | 0.7398 | 0.7081 | 0.8108 | 0.8451 | 0.7126 |
| GCNII | 0.5243 | 0.7632 | 0.8781 | 0.4649 | 0.6081 | 0.6980 | 0.6561 |
| GeomGCN | 0.6634 | 0.7416 | 0.8509 | 0.6946 | 0.8189 | 0.8039 | 0.7622 |
| H2GCN | 0.6553 | 0.7488 | 0.8694 | 0.7486 | 0.8270 | 0.8255 | 0.7791 |
| LINKX | 0.5853 | 0.7559 | 0.8451 | 0.7486 | 0.8162 | 0.8333 | 0.7641 |
| MLP_ONLY | 0.4664 | 0.6805 | 0.7056 | 0.7135 | 0.8054 | 0.8431 | 0.7024 |
| V2_MULTIBRANCH | 0.4654 | 0.7053 | 0.7398 | 0.7081 | 0.8081 | 0.8451 | 0.7120 |

## Mean runtime (seconds; lower is faster, approximate)

| Method | chameleon | citeseer | cora | cornell | texas | wisconsin |
|---|---|---|---|---|---|---|
| ACMII_GCN_PLUSPLUS | 435.94 | 537.80 | 617.68 | 162.27 | 197.42 | 128.72 |
| BASELINE_SGC_V1 | 1.04 | 0.68 | 0.26 | 0.24 | 0.17 | 0.11 |
| FINAL_V3 | 0.77 | 0.44 | 0.18 | 0.17 | 0.01 | 0.02 |
| GCNII | 5273.66 | 2735.96 | 3619.67 | 168.53 | 150.42 | 292.93 |
| GeomGCN | 1282.21 | 626.98 | 464.77 | 259.46 | 245.86 | 180.03 |
| H2GCN | 1696.78 | 892.94 | 444.27 | 339.75 | 332.84 | 301.04 |
| LINKX | 662.69 | 800.86 | 681.34 | 211.55 | 205.65 | 166.99 |
| MLP_ONLY | 185.45 | 218.61 | 128.85 | 118.34 | 106.82 | 86.98 |
| V2_MULTIBRANCH | 0.94 | 0.49 | 0.19 | 0.22 | 0.02 | 0.02 |

## Discussion prompts (fill in after review)

- **Where FINAL_V3 wins vs graph baselines:** compare rows on low-homophily datasets (Texas, Wisconsin, Chameleon).
- **Where it loses:** citation graphs (Cora/Citeseer/Pubmed) where deep GNNs / LINKX may dominate.
- **Safety:** use `n_hurt` and `harmful_overwrite_rate` from JSONL for FINAL_V3 vs baselines (baselines: no correction metrics).
- **Compute:** compare `runtime_sec` and Slurm MaxRSS (`sacct`) for fair cluster-side memory.

Source glob: `outputs/baseline_comparison/wulver_baseline_cmp_20260403_150529/per_run/runs_[chtw]*.jsonl`
Input files: 6
Total rows: 540

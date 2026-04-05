# Baseline comparison (automated summary)

This report is generated from merged JSONL shards. Interpret with paper context.

## Mean test accuracy (higher is better)

| Method | actor | chameleon | citeseer | cora | cornell | squirrel | texas | wisconsin | Overall mean |
|---|---|---|---|---|---|---|---|---|---|
| APPNP | 0.3415 | 0.5276 | 0.7671 | 0.8807 | 0.4757 | 0.3939 | 0.5892 | 0.5706 | 0.5683 |
| CLP | 0.3402 | 0.4711 | 0.7515 | 0.8254 | 0.7595 | 0.2371 | 0.7946 | 0.8569 | 0.6295 |
| CORRECT_AND_SMOOTH | 0.2559 | 0.4910 | 0.7732 | 0.8783 | 0.7216 | 0.3344 | 0.7892 | 0.8157 | 0.6324 |
| GRAPHSAGE | 0.3511 | 0.6336 | 0.7606 | 0.8698 | 0.7054 | 0.4554 | 0.8243 | 0.7412 | 0.6677 |
| MLP_ONLY | 0.3439 | 0.4664 | 0.6805 | 0.7056 | 0.7135 | 0.3213 | 0.8054 | 0.8431 | 0.6100 |
| SGC | 0.2934 | 0.6252 | 0.7651 | 0.8680 | 0.4865 | 0.4529 | 0.5676 | 0.5353 | 0.5743 |

## Mean runtime (seconds; lower is faster, approximate)

| Method | actor | chameleon | citeseer | cora | cornell | squirrel | texas | wisconsin |
|---|---|---|---|---|---|---|---|---|
| APPNP | 716.93 | 425.44 | 260.70 | 348.97 | 124.38 | 1191.00 | 155.70 | 149.37 |
| CLP | 219.82 | 205.42 | 190.76 | 300.19 | 110.94 | 230.54 | 158.32 | 125.53 |
| CORRECT_AND_SMOOTH | 222.37 | 207.68 | 191.13 | 296.44 | 111.39 | 248.24 | 154.88 | 126.56 |
| GRAPHSAGE | 370.90 | 507.63 | 358.61 | 453.91 | 270.80 | 819.04 | 657.86 | 305.22 |
| MLP_ONLY | 138.29 | 141.43 | 113.49 | 153.40 | 76.88 | 153.00 | 107.11 | 92.72 |
| SGC | 30.61 | 43.46 | 25.70 | 38.28 | 9.73 | 65.57 | 6.94 | 24.03 |

## Discussion prompts (fill in after review)

- **Where FINAL_V3 wins vs graph baselines:** compare rows on low-homophily datasets (Texas, Wisconsin, Chameleon).
- **Where it loses:** citation graphs (Cora/Citeseer/Pubmed) where deep GNNs / LINKX may dominate.
- **Safety:** use `n_hurt` and `harmful_overwrite_rate` from JSONL for FINAL_V3 vs baselines (baselines: no correction metrics).
- **Compute:** compare `runtime_sec` and Slurm MaxRSS (`sacct`) for fair cluster-side memory.

Source glob: `outputs/baseline_comparison/wulver_feature_posthoc_applied_intel_20260403_174551/per_run/runs_[acsctw]*.jsonl`
Input files: 8
Total rows: 480

# FINAL_V3 full ablation rerun (current merged code)

Datasets: cora, citeseer, pubmed, chameleon, texas, wisconsin. Splits: 0-9.

| Variant | Mean test acc | Std | Δ vs MLP (pp) | Δ vs FINAL_V3 (pp) |
| --- | ---: | ---: | ---: | ---: |
| FINAL_V3 | 0.7434 | 0.1399 | +1.27 | +0.00 |
| NO_RELIABILITY_GATE | 0.7428 | 0.1392 | +1.21 | -0.06 |
| BALANCED_PROFILE_ONLY | 0.7389 | 0.1399 | +0.82 | -0.45 |
| GRAPH_HEAVY_PROFILE_ONLY | 0.7433 | 0.1400 | +1.27 | -0.01 |
| NO_COMPATIBILITY_TERM | 0.7392 | 0.1394 | +0.85 | -0.42 |
| NO_PROTOTYPE_TERM | 0.7432 | 0.1400 | +1.26 | -0.02 |

## Per-dataset Δ vs MLP (pp)

| Variant | Cora | CiteSeer | PubMed | Chameleon | Texas | Wisconsin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FINAL_V3 | +3.00 | +2.53 | +1.30 | +0.09 | +0.54 | +0.20 |
| NO_RELIABILITY_GATE | +3.22 | +2.69 | +1.29 | +0.07 | +0.00 | -0.00 |
| BALANCED_PROFILE_ONLY | +1.77 | +1.48 | +0.88 | +0.07 | +0.54 | +0.20 |
| GRAPH_HEAVY_PROFILE_ONLY | +3.00 | +2.53 | +1.30 | +0.04 | +0.54 | +0.20 |
| NO_COMPATIBILITY_TERM | +2.11 | +1.68 | +1.03 | +0.09 | +0.00 | +0.20 |
| NO_PROTOTYPE_TERM | +2.96 | +2.51 | +1.29 | +0.04 | +0.54 | +0.20 |

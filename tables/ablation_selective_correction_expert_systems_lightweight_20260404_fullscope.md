# FINAL_V3 compact ablation rerun (lightweight)

10 splits; datasets: chameleon, citeseer, cora, pubmed, texas, wisconsin.

| Variant | Chameleon Δ(pp) | Citeseer Δ(pp) | Cora Δ(pp) | Pubmed Δ(pp) | Texas Δ(pp) | Wisconsin Δ(pp) | Mean Δ(pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ABL_BALANCED_PROFILE_ONLY | +0.07 | +1.48 | +1.77 | +0.88 | +0.54 | +0.20 | +0.82 |
| ABL_GRAPH_HEAVY_PROFILE_ONLY | +0.04 | +2.53 | +3.00 | +1.30 | +0.54 | +0.20 | +1.27 |
| ABL_NO_COMPATIBILITY | +0.09 | +1.68 | +2.11 | +1.03 | +0.00 | +0.20 | +0.85 |
| ABL_NO_PROTOTYPE | +0.04 | +2.51 | +2.96 | +1.29 | +0.54 | +0.20 | +1.26 |
| ABL_NO_RELIABILITY_GATE | +0.07 | +2.69 | +3.22 | +1.29 | +0.00 | -0.00 | +1.21 |
| FINAL_V3 | +0.09 | +2.53 | +3.00 | +1.30 | +0.54 | +0.20 | +1.27 |

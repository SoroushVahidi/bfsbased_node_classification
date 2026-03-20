# Manuscript evidence tables (manuscript_focus_r3)

## A. Core result table
| dataset | method | mean ± std test acc | mean runtime (s) | success |
|---|---|---:|---:|---:|
| chameleon | mlp_only | 0.4591 ± 0.0185 | 0.9641 | 30/30 |
| chameleon | prop_only | 0.3961 ± 0.0379 | 49.9656 | 30/30 |
| chameleon | selective_graph_correction | 0.4599 ± 0.0195 | 0.2829 | 30/30 |
| citeseer | mlp_only | 0.6858 ± 0.0204 | 1.6460 | 30/30 |
| citeseer | prop_only | 0.6903 ± 0.0279 | 89.1384 | 30/30 |
| citeseer | selective_graph_correction | 0.7102 ± 0.0223 | 0.2146 | 30/30 |
| cora | mlp_only | 0.7078 ± 0.0208 | 0.9413 | 30/30 |
| cora | prop_only | 0.8173 ± 0.0230 | 35.4103 | 30/30 |
| cora | selective_graph_correction | 0.7432 ± 0.0208 | 0.1085 | 30/30 |
| texas | mlp_only | 0.7793 ± 0.0554 | 0.2204 | 30/30 |
| texas | prop_only | 0.5685 ± 0.0991 | 1.3623 | 30/30 |
| texas | selective_graph_correction | 0.7775 ± 0.0566 | 0.0063 | 30/30 |
| wisconsin | mlp_only | 0.8549 ± 0.0474 | 0.2426 | 30/30 |
| wisconsin | prop_only | 0.5399 ± 0.0866 | 1.8083 | 30/30 |
| wisconsin | selective_graph_correction | 0.8516 ± 0.0466 | 0.0087 | 30/30 |

## B. Gain-over-MLP table
| dataset | selective - mlp | delta std | wins/losses/ties | sign-test p | selective - prop | avg uncertain frac | avg changed frac |
|---|---:|---:|---:|---:|---:|---:|---:|
| chameleon | +0.0007 | 0.0043 | 16/14/0 | 0.856 | +0.0638 | 0.0692 | 0.0130 |
| citeseer | +0.0244 | 0.0070 | 30/0/0 | 1.86e-09 | +0.0199 | 0.2138 | 0.0494 |
| cora | +0.0353 | 0.0082 | 30/0/0 | 1.86e-09 | -0.0741 | 0.2142 | 0.0461 |
| texas | -0.0018 | 0.0155 | 11/19/0 | 0.2 | +0.2090 | 0.0360 | 0.0144 |
| wisconsin | -0.0033 | 0.0073 | 0/30/0 | 1.86e-09 | +0.3118 | 0.0144 | 0.0039 |

## C. Correction behavior table
| dataset | acc confident | acc uncertain | avg threshold_high | most common weight tuple (count) | feature_knn used | zero-uncertain runs | zero-changed runs |
|---|---:|---:|---:|---|---:|---:|---:|
| chameleon | 0.4695 | 0.3279 | 0.1150 | (1.0, 0.4, 0.0, 0.9, 0.5) (12) | 0/30 | 0/30 | 0/30 |
| citeseer | 0.7623 | 0.5240 | 0.5590 | (1.0, 0.4, 0.0, 0.9, 0.5) (30) | 0/30 | 0/30 | 0/30 |
| cora | 0.7893 | 0.5755 | 0.5917 | (1.0, 0.4, 0.0, 0.9, 0.5) (30) | 0/30 | 0/30 | 0/30 |
| texas | 0.7988 | 0.1474 | 0.0533 | (1.0, 0.6, 0.0, 0.5, 0.3) (27) | 0/30 | 9/30 | 17/30 |
| wisconsin | 0.8614 | 0.1308 | 0.0600 | (1.0, 0.6, 0.0, 0.5, 0.3) (22) | 0/30 | 17/30 | 24/30 |

## D. Dataset-regime table
| dataset | nodes | edges | classes | avg degree | edge homophily | mlp acc | prop acc | selective acc | selective - mlp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chameleon | 2277 | 62792 | 5 | 27.577 | 0.231 | 0.4591 | 0.3961 | 0.4599 | +0.0007 |
| citeseer | 3327 | 9104 | 6 | 2.736 | 0.736 | 0.6858 | 0.6903 | 0.7102 | +0.0244 |
| cora | 2708 | 10556 | 7 | 3.898 | 0.810 | 0.7078 | 0.8173 | 0.7432 | +0.0353 |
| texas | 183 | 574 | 5 | 3.137 | 0.087 | 0.7793 | 0.5685 | 0.7775 | -0.0018 |
| wisconsin | 251 | 916 | 5 | 3.649 | 0.192 | 0.8549 | 0.5399 | 0.8516 | -0.0033 |

## Correlation diagnostics (descriptive only)
- corr_delta_vs_uncertain_frac: +0.9692
- corr_delta_vs_edge_homophily: +0.9797
- corr_delta_vs_mlp_acc: -0.0715
- corr_delta_vs_prop_acc: +0.8764

## Global split-level win/loss/tie vs MLP
- wins: 87
- losses: 63
- ties: 0
- avg changed-from-MLP fraction across datasets: 0.0254
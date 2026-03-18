# Case studies (manuscript_focus_r3)

## citeseer (promising)
- diagnostics file: `/workspace/logs/citeseer_selective_graph_correction_split0_rep0.json`
- changed nodes: 33 | corrected: 24 | harmed: 4
| node | true | mlp | final | outcome | mlp_prob | mlp_margin | dominant_term | dom_val |
|---:|---:|---:|---:|---|---:|---:|---|---:|
| 33 | 5 | 2 | 5 | corrected | 0.3248 | 0.1078 | mlp_term | -1.7117 |
| 319 | 4 | 2 | 4 | corrected | 0.3561 | 0.0134 | mlp_term | -1.0709 |
| 400 | 3 | 1 | 3 | corrected | 0.6935 | 0.4768 | mlp_term | -1.5292 |
| 726 | 5 | 4 | 5 | corrected | 0.4720 | 0.1517 | mlp_term | -1.1387 |
| 1016 | 1 | 2 | 1 | corrected | 0.3429 | 0.1176 | mlp_term | -1.4903 |
| 1274 | 5 | 2 | 5 | corrected | 0.4584 | 0.0324 | mlp_term | -0.8532 |
| 1689 | 3 | 0 | 3 | corrected | 0.3714 | 0.1138 | mlp_term | -2.1079 |
| 1859 | 2 | 1 | 2 | corrected | 0.7568 | 0.5137 | mlp_term | -1.4144 |
| 1866 | 0 | 5 | 0 | corrected | 0.3511 | 0.0757 | mlp_term | -1.4814 |
| 2117 | 1 | 3 | 1 | corrected | 0.3491 | 0.0424 | mlp_term | -1.9117 |
| 2125 | 2 | 3 | 2 | corrected | 0.5671 | 0.1683 | mlp_term | -0.9194 |
| 2140 | 5 | 0 | 5 | corrected | 0.2634 | 0.0060 | mlp_term | -1.3569 |

## chameleon (neutral)
- diagnostics file: `/workspace/logs/chameleon_selective_graph_correction_split0_rep0.json`
- changed nodes: 2 | corrected: 1 | harmed: 1
| node | true | mlp | final | outcome | mlp_prob | mlp_margin | dominant_term | dom_val |
|---:|---:|---:|---:|---|---:|---:|---|---:|
| 457 | 2 | 0 | 2 | corrected | 0.5000 | 0.0032 | mlp_term | -0.6996 |
| 2258 | 0 | 0 | 1 | harmed | 0.4857 | 0.0405 | mlp_term | -0.8093 |

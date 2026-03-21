# Case Studies — Selective Graph Correction

For each dataset, we show corrected test nodes from split 0.

## pubmed

- threshold_high=0.86  b=(1.00,0.40,0.00,0.90,0.50)
- 783/3944 test nodes uncertain | 137 corrected

### Sample corrected nodes (first 10)

| node_id | mlp_pred | final_pred | mlp_margin | was_uncertain | dom_component |
|---------|----------|-------------|--------------|----------------|---------------|
| 22 | 1 | 2 | 0.265 | True | mlp_term |
| 32 | 2 | 1 | 0.253 | True | mlp_term |
| 186 | 1 | 2 | 0.181 | True | graph_neighbor_term |
| 466 | 2 | 1 | 0.332 | True | mlp_term |
| 497 | 1 | 2 | 0.216 | True | mlp_term |
| 704 | 2 | 1 | 0.143 | True | mlp_term |
| 836 | 0 | 1 | 0.478 | True | mlp_term |
| 904 | 2 | 1 | 0.391 | True | mlp_term |
| 1037 | 2 | 1 | 0.058 | True | mlp_term |
| 1489 | 2 | 1 | 0.341 | True | mlp_term |

## squirrel

- threshold_high=0.05  b=(1.00,0.90,0.00,0.30,0.20)
- 22/1041 test nodes uncertain | 7 corrected

### Sample corrected nodes (first 10)

| node_id | mlp_pred | final_pred | mlp_margin | was_uncertain | dom_component |
|---------|----------|-------------|--------------|----------------|---------------|
| 1341 | 2 | 0 | 0.026 | True | mlp_term |
| 1659 | 4 | 2 | 0.012 | True | mlp_term |
| 2139 | 1 | 0 | 0.009 | True | mlp_term |
| 2166 | 4 | 1 | 0.018 | True | mlp_term |
| 2703 | 3 | 1 | 0.042 | True | mlp_term |
| 4422 | 3 | 4 | 0.024 | True | mlp_term |
| 4705 | 2 | 1 | 0.015 | True | mlp_term |


# Case Studies — Selective Graph Correction

For each dataset, we show corrected test nodes from split 0.

## cora

- threshold_high=0.60  b=(1.00,0.40,0.00,0.90,0.50)
- 101/497 test nodes uncertain | 17 corrected

### Sample corrected nodes (first 10)

| node_id | mlp_pred | final_pred | mlp_margin | was_uncertain | dom_component |
|---------|----------|-------------|--------------|----------------|---------------|
| 209 | 5 | 4 | 0.310 | True | mlp_term |
| 543 | 3 | 0 | 0.185 | True | mlp_term |
| 803 | 0 | 5 | 0.012 | True | mlp_term |
| 1076 | 5 | 6 | 0.024 | True | mlp_term |
| 1087 | 0 | 4 | 0.193 | True | mlp_term |
| 1129 | 0 | 4 | 0.271 | True | mlp_term |
| 1222 | 1 | 3 | 0.109 | True | mlp_term |
| 1340 | 0 | 3 | 0.176 | True | mlp_term |
| 1399 | 4 | 3 | 0.006 | True | mlp_term |
| 1696 | 5 | 0 | 0.075 | True | mlp_term |

## citeseer

- threshold_high=0.56  b=(1.00,0.40,0.00,0.90,0.50)
- 142/666 test nodes uncertain | 45 corrected

### Sample corrected nodes (first 10)

| node_id | mlp_pred | final_pred | mlp_margin | was_uncertain | dom_component |
|---------|----------|-------------|--------------|----------------|---------------|
| 33 | 4 | 1 | 0.093 | True | mlp_term |
| 125 | 5 | 2 | 0.276 | True | mlp_term |
| 143 | 5 | 2 | 0.249 | True | mlp_term |
| 171 | 5 | 4 | 0.015 | True | mlp_term |
| 226 | 2 | 4 | 0.041 | True | mlp_term |
| 252 | 5 | 4 | 0.418 | True | mlp_term |
| 293 | 4 | 3 | 0.189 | True | mlp_term |
| 347 | 1 | 2 | 0.010 | True | mlp_term |
| 355 | 5 | 1 | 0.294 | True | mlp_term |
| 356 | 2 | 5 | 0.189 | True | mlp_term |

## chameleon

- threshold_high=0.10  b=(1.00,0.40,0.00,0.90,0.50)
- 34/456 test nodes uncertain | 7 corrected

### Sample corrected nodes (first 10)

| node_id | mlp_pred | final_pred | mlp_margin | was_uncertain | dom_component |
|---------|----------|-------------|--------------|----------------|---------------|
| 243 | 3 | 4 | 0.026 | True | mlp_term |
| 361 | 3 | 0 | 0.039 | True | mlp_term |
| 414 | 1 | 3 | 0.081 | True | mlp_term |
| 838 | 0 | 4 | 0.022 | True | mlp_term |
| 1074 | 0 | 2 | 0.052 | True | mlp_term |
| 1273 | 0 | 2 | 0.017 | True | mlp_term |
| 1680 | 1 | 2 | 0.076 | True | mlp_term |

## texas

- threshold_high=0.05  b=(1.00,0.60,0.00,0.50,0.30)
- 2/37 test nodes uncertain | 2 corrected

### Sample corrected nodes (first 10)

| node_id | mlp_pred | final_pred | mlp_margin | was_uncertain | dom_component |
|---------|----------|-------------|--------------|----------------|---------------|
| 84 | 3 | 4 | 0.006 | True | mlp_term |
| 172 | 4 | 0 | 0.023 | True | mlp_term |

## actor

- threshold_high=0.05  b=(1.20,0.50,0.00,0.40,0.20)
- 127/1520 test nodes uncertain | 26 corrected

### Sample corrected nodes (first 10)

| node_id | mlp_pred | final_pred | mlp_margin | was_uncertain | dom_component |
|---------|----------|-------------|--------------|----------------|---------------|
| 46 | 1 | 0 | 0.035 | True | mlp_term |
| 334 | 2 | 1 | 0.031 | True | mlp_term |
| 733 | 1 | 0 | 0.035 | True | mlp_term |
| 812 | 2 | 4 | 0.022 | True | mlp_term |
| 1176 | 1 | 2 | 0.010 | True | mlp_term |
| 1194 | 1 | 2 | 0.001 | True | mlp_term |
| 1366 | 4 | 1 | 0.004 | True | mlp_term |
| 1439 | 0 | 3 | 0.003 | True | mlp_term |
| 1494 | 2 | 1 | 0.017 | True | mlp_term |
| 1935 | 1 | 2 | 0.015 | True | mlp_term |

## cornell

- threshold_high=0.05  b=(1.00,0.60,0.00,0.50,0.30)
- 1/37 test nodes uncertain | 0 corrected

_No changed nodes in this run._

## wisconsin

- threshold_high=0.05  b=(1.00,0.60,0.00,0.50,0.30)
- 0/51 test nodes uncertain | 0 corrected

_No changed nodes in this run._


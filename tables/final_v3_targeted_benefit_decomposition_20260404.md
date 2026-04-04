# FINAL_V3 targeted-benefit decomposition

## Dataset × subgroup weighted summary

| dataset   | subgroup   |   n_nodes |   n_test_total |   fraction_test_nodes |   mlp_acc_subgroup |   final_acc_subgroup |   delta_subgroup |   mean_reliability |   mean_mlp_margin |   changed_precision |
|:----------|:-----------|----------:|---------------:|----------------------:|-------------------:|---------------------:|-----------------:|-------------------:|------------------:|--------------------:|
| chameleon | changed    |        53 |           4560 |                0.0116 |             0.2264 |               0.3019 |           0.0755 |             0.4707 |            0.0490 |              0.4086 |
| chameleon | routed     |       440 |           4560 |                0.0965 |             0.4045 |               0.4136 |           0.0091 |             0.5658 |            0.0801 |              0.4086 |
| chameleon | unchanged  |      4507 |           4560 |                0.9884 |             0.4768 |               0.4768 |           0.0000 |             0.5062 |            0.6625 |            nan      |
| citeseer  | changed    |       306 |           6176 |                0.0495 |             0.1503 |               0.6765 |           0.5261 |             0.4188 |            0.2069 |              0.8072 |
| citeseer  | routed     |      1295 |           6176 |                0.2097 |             0.4301 |               0.5544 |           0.1243 |             0.5596 |            0.3252 |              0.8072 |
| citeseer  | unchanged  |      5870 |           6176 |                0.9505 |             0.7085 |               0.7085 |           0.0000 |             0.6269 |            0.8249 |            nan      |
| cora      | changed    |       196 |           4970 |                0.0394 |             0.0765 |               0.8367 |           0.7602 |             0.4343 |            0.1812 |              0.9148 |
| cora      | routed     |      1133 |           4970 |                0.2280 |             0.4360 |               0.5675 |           0.1315 |             0.5635 |            0.3579 |              0.9148 |
| cora      | unchanged  |      4774 |           4970 |                0.9606 |             0.7340 |               0.7340 |           0.0000 |             0.6550 |            0.8290 |            nan      |
| pubmed    | changed    |      1162 |          39440 |                0.0295 |             0.2642 |               0.7040 |           0.4398 |             0.4277 |            0.2368 |              0.7274 |
| pubmed    | routed     |      9443 |          39440 |                0.2394 |             0.6625 |               0.7166 |           0.0541 |             0.6427 |            0.5925 |              0.7274 |
| pubmed    | unchanged  |     38278 |          39440 |                0.9705 |             0.8906 |               0.8906 |           0.0000 |             0.7111 |            0.9118 |            nan      |
| texas     | changed    |         7 |            370 |                0.0189 |             0.1429 |               0.4286 |           0.2857 |             0.3860 |            0.0139 |              0.3000 |
| texas     | routed     |         8 |            370 |                0.0216 |             0.1250 |               0.3750 |           0.2500 |             0.4015 |            0.0169 |              0.3000 |
| texas     | unchanged  |       363 |            370 |                0.9811 |             0.8182 |               0.8182 |           0.0000 |             0.3830 |            0.7124 |            nan      |
| wisconsin | changed    |         2 |            510 |                0.0039 |             0.0000 |               0.5000 |           0.5000 |             0.4016 |            0.0162 |              0.1000 |
| wisconsin | routed     |         9 |            510 |                0.0176 |             0.3333 |               0.4444 |           0.1111 |             0.5555 |            0.0199 |              0.1000 |
| wisconsin | unchanged  |       508 |            510 |                0.9961 |             0.8465 |               0.8465 |           0.0000 |             0.4459 |            0.8008 |            nan      |

## Global subgroup summary (weighted across all splits and datasets)

| dataset   | subgroup   |   n_nodes |   n_test_total |   fraction_test_nodes |   mlp_acc_subgroup |   final_acc_subgroup |   delta_subgroup |   mean_reliability |   mean_mlp_margin |   changed_precision |
|:----------|:-----------|----------:|---------------:|----------------------:|-------------------:|---------------------:|-----------------:|-------------------:|------------------:|--------------------:|
| ALL       | changed    |      1726 |          56026 |                0.0308 |             0.2207 |               0.7005 |           0.4797 |             0.4281 |            0.1469 |              0.5430 |
| ALL       | routed     |     12328 |          56026 |                0.2200 |             0.6075 |               0.6746 |           0.0672 |             0.5589 |            0.2698 |              0.5430 |
| ALL       | unchanged  |     54300 |          56026 |                0.9692 |             0.8219 |               0.8219 |           0.0000 |             0.5547 |            0.7902 |            nan      |

## Compact dataset table

| dataset   |   routed_fraction |   changed_fraction |   delta_routed |   delta_changed |   delta_unchanged | interpretation_tag              |
|:----------|------------------:|-------------------:|---------------:|----------------:|------------------:|:--------------------------------|
| chameleon |            0.0965 |             0.0116 |         0.0091 |          0.0755 |            0.0000 | conservative/noisy              |
| citeseer  |            0.2097 |             0.0495 |         0.1243 |          0.5261 |            0.0000 | high-value selective correction |
| cora      |            0.2280 |             0.0394 |         0.1315 |          0.7602 |            0.0000 | high-value selective correction |
| pubmed    |            0.2394 |             0.0295 |         0.0541 |          0.4398 |            0.0000 | high-value selective correction |
| texas     |            0.0216 |             0.0189 |         0.2500 |          0.2857 |            0.0000 | high-value selective correction |
| wisconsin |            0.0176 |             0.0039 |         0.1111 |          0.5000 |            0.0000 | high-value selective correction |

# Manuscript correction behavior (FINAL_V3)

Computed from `reports/final_method_v3_results.csv` (canonical FINAL_V3 + MLP rows).

| dataset | mlp_mean_acc | final_v3_mean_acc | delta_vs_mlp | corrected_frac_mean | correction_precision_mean | helped_total | hurt_total | wins | ties | losses |
|---|---|---|---|---|---|---|---|---|---|---|
| chameleon | 0.463158 | 0.463377 | 0.000219 | 0.058333 | 0.466667 | 10 | 9 | 2 | 7 | 1 |
| citeseer | 0.679692 | 0.705602 | 0.025909 | 0.200475 | 0.824504 | 199 | 39 | 10 | 0 | 0 |
| cora | 0.712676 | 0.741650 | 0.028974 | 0.236821 | 0.921676 | 157 | 13 | 10 | 0 | 0 |
| pubmed | 0.872008 | 0.884888 | 0.012880 | 0.239097 | 0.719478 | 834 | 326 | 10 | 0 | 0 |
| texas | 0.805405 | 0.810811 | 0.005405 | 0.021622 | 0.700000 | 3 | 1 | 3 | 6 | 1 |
| wisconsin | 0.843137 | 0.845098 | 0.001961 | 0.017647 | 0.900000 | 1 | 0 | 1 | 9 | 0 |

# Representative split examples (FINAL_V3)

Representative split per dataset is chosen as the split whose `delta_vs_mlp` is closest to that dataset's FINAL_V3 mean delta.

| dataset | split_id | mlp_acc | final_v3_acc | delta_vs_mlp | uncertain_frac | corrected_frac | n_changed | n_helped | n_hurt | selected_tau | selected_rho |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cora | 0 | 0.690141 | 0.718310 | 0.028169 | 0.265594 | 0.251509 | 14 | 14 | 0 | 0.679310 | 0.300000 |
| citeseer | 1 | 0.684685 | 0.710210 | 0.025526 | 0.264264 | 0.229730 | 29 | 23 | 6 | 0.703610 | 0.300000 |
| pubmed | 0 | 0.867647 | 0.879817 | 0.012170 | 0.244929 | 0.237830 | 126 | 87 | 39 | 0.921490 | 0.300000 |
| chameleon | 0 | 0.440789 | 0.440789 | 0.000000 | 0.214912 | 0.214912 | 2 | 1 | 1 | 0.200000 | 0.300000 |

# FINAL_V3 harmful-correction failure summary (20260404)

Source changed-node export: `reports/correction_explanation_nodes_20260404.csv`
Reference explanation audit: `reports/correction_explanation_audit_20260404.md`
Reference explanation summary: `tables/correction_explanation_summary_20260404.csv`
Context bounded-intervention CSV: `tables/bounded_intervention_fullscope_20260404.csv`
Context gate-sensitivity CSV: `tables/gate_sensitivity_fullscope_20260404.csv`

## Failure profile
- Harmful changed nodes: **84 / 341** (24.6% of changed).
- Mean reliability (harmful): **0.418** vs beneficial **0.433**.
- Mean margin (harmful): **0.242** vs beneficial **0.218**.
- Mean score gap (harmful): **0.495** vs beneficial **0.616**.
- Prototype/graph agreement (harmful): **47.6%** vs beneficial **63.3%**.
- Highest harmful concentration dataset: **pubmed** (81.0% of harmful cases).

### Harmful dataset distribution

| Dataset | Harmful share |
| --- | ---: |
| pubmed | 81.0% |
| citeseer | 13.1% |
| cora | 6.0% |

### Harmful split distribution (top 8)

| Dataset/split | Harmful count |
| --- | ---: |
| pubmed / 0 | 38 |
| pubmed / 1 | 30 |
| citeseer / 1 | 6 |
| citeseer / 0 | 5 |
| cora / 0 | 3 |
| cora / 1 | 2 |

## Dominant-driver shares (harmful vs beneficial)

| Driver | Harmful | Beneficial |
| --- | ---: | ---: |
| neighbor_support | 100.0% | 100.0% |
| prototype | 0.0% | 0.0% |
| compatibility | 0.0% | 0.0% |
| mlp_anchor | 0.0% | 0.0% |
| feature_knn | 0.0% | 0.0% |
| structural_far | 0.0% | 0.0% |

## Safeguard what-if (post hoc; no retraining)

| Rule | Harmful prevented | Harmful prevented % | Beneficial lost | Beneficial lost % | Prevented/Lost |
| --- | ---: | ---: | ---: | ---: | ---: |
| gap_ge_0p05 | 7 | 8.3% | 13 | 5.5% | 0.54 |
| proto_graph_agree | 44 | 52.4% | 87 | 36.7% | 0.51 |
| rel_ge_0p43_and_gap_ge_0p03 | 61 | 72.6% | 141 | 59.5% | 0.43 |
| rel_ge_0p45 | 70 | 83.3% | 179 | 75.5% | 0.39 |

# FINAL_V3 regime/failure analysis (reuse-first)

This analysis reuses canonical FINAL_V3 logs and archived resubmission baseline runs. Results are descriptive (small sample).

## Correlations with FINAL_V3 gain over MLP

| Variable | Pearson r |
|---|---:|
| edge_homophily | 0.851 |
| avg_degree | -0.497 |
| mlp_acc | 0.069 |
| reliability_gated_coverage | 0.840 |
| harmful_overwrite_rate | -0.141 |

## Findings

- FINAL_V3 tends to gain most on Cora/Citeseer/Pubmed and remains near-neutral on Chameleon/Texas/Wisconsin.
- Compared with archived graph-wide GCN/APPNP (5 datasets), FINAL_V3 is usually lower in raw accuracy, consistent with selective correction as a safer feature-first method rather than a universal graph learner.
- Harmful overwrite rates are lowest where correction precision is highest (notably Cora/Citeseer); they are highest on Chameleon/Texas.
- Gate coverage is moderate; many high-confidence nodes stay untouched (where available in canonical logs).
- Texas lacks archived GCN/APPNP records in the merged resubmission log, so those comparisons are omitted there.

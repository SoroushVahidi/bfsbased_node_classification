# FINAL_V3 regime mapping (20260404)

## Source artifacts (auto-detected newest)
- Regime summary: `tables/final_v3_regime_summary.csv`
- Experimental setup: `tables/experimental_setup_selective_correction.csv`
- Bounded intervention: `tables/bounded_intervention_fullscope_20260404.csv`
- Gate sensitivity: `tables/gate_sensitivity_fullscope_20260404.csv`
- Correction diagnostics: `reports/final_v3_correction_diagnostics_20260404_fullscope.csv`
- Node-level explanation: `reports/correction_explanation_nodes_20260404.csv`

## Key descriptive findings
- FINAL_V3 gains are heterogeneous across datasets; they are not uniform.
- Larger gains align more with correction quality metrics (changed precision, reliability) than with high routing volume.
- Harmful-overwrite signals trend negatively with gains; changed-node harmful-rate signals are limited to datasets with node-level exports.
- Sensitivity-span indicators suggest that highly unstable gate-response regimes are less consistently favorable.

## Correlation snapshot (Spearman, dataset-level)
Top positive correlates:
| factor               |   n_used |   spearman_corr_with_improvement |
|:---------------------|---------:|---------------------------------:|
| changed_fraction_pct |        6 |                         0.885714 |
| gate_delta_span_pp   |        6 |                         0.885714 |
| gate_delta_std_pp    |        6 |                         0.885714 |
| edge_homophily       |        6 |                         0.714286 |

Top negative correlates:
| factor                           |   n_used |   spearman_corr_with_improvement |
|:---------------------------------|---------:|---------------------------------:|
| harmful_overwrite_rate           |        6 |                       -0.657143  |
| mean_reliability_corrected_nodes |        6 |                        0.0857143 |
| routed_fraction_pct              |        6 |                        0.657143  |
| changed_precision                |        6 |                        0.657143  |

## Regime-oriented interpretation
This is a descriptive regime map, not a causal claim. The pattern supports selective correction as a bounded
intervention that works best when corrections are high quality and safety indicators remain controlled.


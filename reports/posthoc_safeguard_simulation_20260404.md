# Post hoc safeguard simulation (20260404)

## Scope and method
This is a post hoc simulation on existing FINAL_V3 changed-node exports only; no method redesign was run.
- Node-level explanation export used: `reports/correction_explanation_nodes_20260404.csv`
- Harmful correction summary used: `tables/harmful_correction_failure_summary_20260404.csv`
- Bounded intervention table used: `tables/bounded_intervention_fullscope_20260404.csv`
- Gate sensitivity table used: `tables/gate_sensitivity_fullscope_20260404.csv`

## Overall safeguard tradeoff
| rule                        |   harmful_prevented |   beneficial_lost |   harmful_prevention_rate |   beneficial_retention_rate |   changed_precision_baseline |   changed_precision_after |   changed_precision_delta_pp |   n_blocked_total |
|:----------------------------|--------------------:|------------------:|--------------------------:|----------------------------:|-----------------------------:|--------------------------:|-----------------------------:|------------------:|
| R4_graph_dominant_low_rel   |                  70 |               179 |                  0.833333 |                    0.244726 |                     0.738318 |                  0.805556 |                      6.72378 |               249 |
| R5_require_rel_and_gap      |                  61 |               148 |                  0.72619  |                    0.375527 |                     0.738318 |                  0.794643 |                      5.63251 |               209 |
| R1_reliability_lt_0p40      |                  52 |               129 |                  0.619048 |                    0.455696 |                     0.738318 |                  0.771429 |                      3.31108 |               181 |
| R3_no_proto_graph_agreement |                  44 |                87 |                  0.52381  |                    0.632911 |                     0.738318 |                  0.789474 |                      5.11559 |               131 |
| R2_score_gap_lt_0p30        |                  30 |                54 |                  0.357143 |                    0.772152 |                     0.738318 |                  0.772152 |                      3.38341 |                84 |

Primary conservative rule (retention-aware): **R2_score_gap_lt_0p30**.
Secondary stronger-but-costlier option: **R4_graph_dominant_low_rel**.

## Interpretation
- Harmful changed-node corrections are the minority in the changed-node pool and can be partially targeted.
- Simple threshold safeguards show clear tradeoffs between harmful prevention and beneficial retention.
- A conservative score-gap rule preserves most beneficial corrections while still blocking a subset of harmful ones.
- A stricter reliability + graph-dominant rule blocks more harmful cases but at a substantial retention cost.


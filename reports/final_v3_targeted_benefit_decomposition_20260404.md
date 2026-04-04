# FINAL_V3 targeted-benefit decomposition

## Artifact reuse
- Newest full-scope ablation source: `reports/final_v3_expert_systems_lightweight_ablation_20260404_fullscope.csv`
- Newest bounded-intervention source: `tables/bounded_intervention_fullscope_20260404.csv`
- Newest node-level explanation source: `reports/correction_explanation_nodes_20260404.csv`
- Minimal rerun performed: FINAL_V3-only, same six datasets × ten splits, to obtain full test-node routed/changed masks and subgroup accuracies.

## Global subgroup summary

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

## Answers to manuscript questions

- Gains concentrated on routed/changed nodes: **yes** (global changed-node delta = 0.4797; routed-node delta = 0.0672).
- Unchanged nodes preserve MLP behavior: **yes** (global unchanged-node delta = 0.0000).
- Method value from selective intervention rather than broad rewriting: **yes**, because changed/routed fractions are bounded while delta concentrates in changed subgroup.
- Bounded expert-system correction story reinforced: **yes**, subgroup decomposition aligns with targeted correction behavior.

## Manuscript-ready text (4–6 sentences)

FINAL_V3 intervenes on a bounded subset of test nodes (routed fraction well below 1.0 across datasets), and the largest positive deltas are concentrated in the changed subgroup rather than spread uniformly. Unchanged nodes remain near MLP behavior by construction, with near-zero delta in most datasets. Routed-node gains are positive on average, indicating the gate is selecting higher-value intervention regions. Overall, these subgroup patterns support interpreting FINAL_V3 as a selective expert correction layer rather than a graph-wide replacement learner. 

# FINAL_V3 node-level correction explanation audit (20260404)

This is a manuscript-supporting, non-canonical interpretability audit over changed FINAL_V3 test nodes.

## What was reconstructed
- Additive evidence components from FINAL_V3 score: MLP anchor, prototype similarity, graph neighbor support, compatibility, and zero-weight residual terms (feature_kNN, structural_far).
- Per-node class-wise contribution deltas between final corrected class and original MLP class.
- Dominant correction reason defined as the component with largest contribution gap.

## Findings
- Beneficial corrections: most common dominant driver = **neighbor_support**.
- Harmful corrections: most common dominant driver = **neighbor_support**.
- Harmful corrections have lower reliability than beneficial: **True**.
- Harmful corrections have smaller score gaps than beneficial: **True**.
- Harmful corrections are more single-component concentrated: **False**.
- Beneficial prototype/graph agreement rate: **63.3%**.
- Harmful prototype/graph agreement rate: **47.6%**.

## Outputs
- Node-level export: `reports/correction_explanation_nodes_20260404.csv`
- Summary table (CSV): `tables/correction_explanation_summary_20260404.csv`
- Summary table (MD): `tables/correction_explanation_summary_20260404.md`

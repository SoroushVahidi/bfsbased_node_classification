# FINAL_V3 harmful-correction failure audit (20260404)

This is a lightweight, non-canonical post hoc failure audit on changed-node explanations.

## Key observations
- Harmful cases are a minority among changed nodes (84/341).
- Harmful cases tend to show slightly lower reliability and smaller final-vs-original score gaps than beneficial changes.
- Dominant driver in this bundle remains graph neighbor support for both helpful and harmful changes.
- Best simple safeguard by prevented/lost ratio: `gap_ge_0p05` (prevented 7 harmful while losing 13 beneficial).

## Outputs
- Summary CSV: `tables/harmful_correction_failure_summary_20260404.csv`
- Summary Markdown: `tables/harmful_correction_failure_summary_20260404.md`
- Case studies: `reports/harmful_correction_case_studies_20260404.md`

# Ablation-style variants (supplementary)

**FINAL_V3** row: mean test Δ vs MLP (percentage points) over **10 splits**, recomputed from `reports/final_method_v3_results.csv` for Chameleon, Cora, and Texas.

**Other rows:** frozen snapshot from `reports/archive/superseded_final_tables_prl/final_tables_prl.md` (Table 7) from the same historical validation campaign — **not** re-executed against the current rebuild merge. Use for qualitative discussion; a full 10-split ablation re-run would require new experiments.

| Variant | Chameleon Δ (pp) | Cora Δ (pp) | Texas Δ (pp) | Mean (3 ds) | Source |
| --- | ---: | ---: | ---: | ---: | --- |
| FINAL_V3 | +0.02 | +2.90 | +0.54 | +1.15 | recomputed from final_method_v3_results.csv (10 splits) |
| A1_NO_RELIABILITY | +0.22 | +2.58 | +0.54 | +1.11 | archived final_tables_prl.md Table 7 (historical) |
| A2_PROFILE_A_ONLY | +0.04 | +1.41 | +1.08 | +0.84 | archived final_tables_prl.md Table 7 (historical) |
| A3_PROFILE_B_ONLY | -0.09 | +2.49 | +1.08 | +1.16 | archived final_tables_prl.md Table 7 (historical) |

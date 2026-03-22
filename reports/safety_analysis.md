# Safety analysis (FINAL_V3 vs baselines)

Source of truth: `reports/final_method_v3_results.csv` — six datasets × **10** GEO-GCN splits × four methods (**60** split realizations per method).

## Split-level Δ vs MLP

A split is counted as **harmful** when test accuracy is strictly lower than the MLP baseline on that same split (`delta_vs_mlp < 0`).

| Method | Harmful splits (of 60) | Where |
|--------|------------------------|-------|
| **FINAL_V3** | 2 | Chameleon split 4 (−0.44 pp); Texas split 5 (−2.70 pp) |
| **SGC v1** (`BASELINE_SGC_V1`) | 4 | Texas splits 4, 5, 9 (−2.70 pp each); Wisconsin split 8 (−1.96 pp) |
| **V2_MULTIBRANCH** | 4 | Chameleon splits 2, 6, 8; Texas split 5 |

## Interpretation

- **Texas split 4:** SGC v1 is worse than MLP (−2.70 pp); **FINAL_V3 matches MLP** on that split — the reliability gate suppresses the harmful correction pattern seen in v1.
- **Texas split 5:** In the frozen log, **all** compared methods (MLP baseline accuracy is higher) tie at the same reduced accuracy: SGC v1, V2, and FINAL_V3 all show −2.70 pp vs MLP. This is a hard realization for the whole family of corrections, not a unique v1 pathology.
- **Wisconsin split 8:** SGC v1 regresses; FINAL_V3 does not.
- **Chameleon:** FINAL_V3 remains conservative; one split (4) shows a small negative Δ vs MLP. Ten-split means stay near the MLP (see `tables/main_results_prl.md`).

## Figure

Bar summary of harmful-split counts: `figures/safety_comparison.png` (regenerate with `bash scripts/run_all_prl_results.sh`).

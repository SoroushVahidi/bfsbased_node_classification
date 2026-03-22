# Safety analysis (FINAL_V3 vs baselines)

Source of truth: `reports/final_method_v3_results.csv` (six datasets × five GEO-GCN splits × four methods).

## Split-level Δ vs MLP

A split is counted as **harmful** when test accuracy is strictly lower than the MLP baseline on that same split (`delta_vs_mlp < 0`).

| Method | Harmful splits (of 30) | Where |
|--------|------------------------|-------|
| **FINAL_V3** | 1 | Chameleon, split 4 (−0.44 pp) |
| **SGC v1** (`BASELINE_SGC_V1`) | 1 | Texas, split 4 (−2.70 pp) |
| **V2_MULTIBRANCH** | 1 | Chameleon, split 2 (−0.22 pp) |

## Interpretation

- **Texas:** SGC v1 applies graph correction even when the neighborhood is unreliable; FINAL_V3 suppresses correction via the reliability gate and **does not** exhibit the Texas split-4 failure.
- **Chameleon:** All methods are conservative; FINAL_V3 still has **one** split with a small regression vs MLP. Mean accuracy over five splits remains aligned with MLP (see `tables/main_results_prl.md`).

## Figure

Bar summary of harmful-split counts: `figures/safety_comparison.png` (regenerate with `bash scripts/run_all_prl_results.sh`).

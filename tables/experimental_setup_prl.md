# Experimental setup (PRL main benchmark)

Dataset statistics and edge homophily are merged from `logs/manuscript_regime_analysis_final_validation_main.csv` and, where needed, `logs/regime_analysis_manuscript_final_validation_job2.csv` (PubMed homophily), using the same homophily column as `figures/correction_rate_vs_homophily.png`.

**Split protocol:** 60%/20%/20% train/val/test; 10 fixed splits (indices 0–9); GEO-GCN `.npz` in `data/splits/`

**Compared methods (main table):** MLP_ONLY, BASELINE_SGC_V1, V2_MULTIBRANCH, FINAL_V3

| Dataset | # nodes | # classes | Edge homophily | Split protocol | Compared methods |
| --- | ---: | ---: | --- | --- | --- |
| Chameleon | 2277 | 5 | 0.231 | 60%/20%/20% train/val/test; 10 fixed splits (indices 0–9); GEO-GCN `.npz` in `data/splits/` | MLP_ONLY; BASELINE_SGC_V1; V2_MULTIBRANCH; FINAL_V3 |
| Citeseer | 3327 | 6 | 0.736 | 60%/20%/20% train/val/test; 10 fixed splits (indices 0–9); GEO-GCN `.npz` in `data/splits/` | MLP_ONLY; BASELINE_SGC_V1; V2_MULTIBRANCH; FINAL_V3 |
| Cora | 2708 | 7 | 0.810 | 60%/20%/20% train/val/test; 10 fixed splits (indices 0–9); GEO-GCN `.npz` in `data/splits/` | MLP_ONLY; BASELINE_SGC_V1; V2_MULTIBRANCH; FINAL_V3 |
| Pubmed | 19717 | 3 | 0.802 | 60%/20%/20% train/val/test; 10 fixed splits (indices 0–9); GEO-GCN `.npz` in `data/splits/` | MLP_ONLY; BASELINE_SGC_V1; V2_MULTIBRANCH; FINAL_V3 |
| Texas | 183 | 5 | 0.087 | 60%/20%/20% train/val/test; 10 fixed splits (indices 0–9); GEO-GCN `.npz` in `data/splits/` | MLP_ONLY; BASELINE_SGC_V1; V2_MULTIBRANCH; FINAL_V3 |
| Wisconsin | 251 | 5 | 0.192 | 60%/20%/20% train/val/test; 10 fixed splits (indices 0–9); GEO-GCN `.npz` in `data/splits/` | MLP_ONLY; BASELINE_SGC_V1; V2_MULTIBRANCH; FINAL_V3 |

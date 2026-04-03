# Main results (test accuracy, mean ± std over splits)

Source: `reports/final_method_v3_results.csv` — **10 GEO-GCN splits** (indices 0–9) per dataset, same protocol as the FINAL_V3 evaluation logs.

Regenerate this table and the figure set:
`python3 scripts/build_artifacts/build_v3_figures_and_tables.py`
(reads existing CSVs only; no experiments).

| Dataset | MLP | SGC v1 | V2 multi | FINAL_V3 |
| --- | --- | --- | --- | --- |
| Chameleon | 0.4632 ± 0.0237 | **0.4647 ± 0.0237** | 0.4643 ± 0.0243 | 0.4634 ± 0.0241 |
| Citeseer | 0.6797 ± 0.0174 | **0.7064 ± 0.0183** | 0.7055 ± 0.0187 | 0.7056 ± 0.0183 |
| Cora | 0.7127 ± 0.0269 | **0.7433 ± 0.0200** | 0.7421 ± 0.0208 | 0.7416 ± 0.0214 |
| Pubmed | 0.8720 ± 0.0035 | 0.8848 ± 0.0035 | 0.8846 ± 0.0036 | **0.8849 ± 0.0035** |
| Texas | 0.8054 ± 0.0671 | 0.8054 ± 0.0763 | 0.8081 ± 0.0759 | **0.8108 ± 0.0755** |
| Wisconsin | 0.8431 ± 0.0421 | 0.8431 ± 0.0402 | **0.8451 ± 0.0406** | **0.8451 ± 0.0406** |

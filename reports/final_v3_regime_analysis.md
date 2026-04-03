# FINAL_V3 Regime & Failure Analysis

This report is descriptive (6 datasets), not definitive causal evidence.

## Core findings
- Largest gains over MLP: cora (+0.0300), citeseer (+0.0253).
- Main failure/neutral regime: chameleon (+0.0009).
- FINAL_V3 typically changes only a small subset, with correction precision reported in the table.

## Correlations (dataset-level means, n=6)
- gain vs edge_homophily: r=0.847
- gain vs avg_degree: r=-0.470
- gain vs degree_var: r=-0.489
- gain vs mlp_baseline_strength: r=0.034
- gain vs correction_coverage: r=0.802
- gain vs harmful_overwrite_rate: r=-0.537
- gain vs low_conf_bucket_gain: r=0.063

## Cora & CiteSeer per-split (FINAL_V3 heuristic)

### Cora
| split | MLP | FINAL_V3 | Δ | low-conf gain | changed frac | harm count |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.6922 | 0.7203 | 0.0282 | 0.0972 | 0.0463 | 3 |
| 1 | 0.6982 | 0.7304 | 0.0322 | 0.1280 | 0.0402 | 2 |
| 2 | 0.7425 | 0.7666 | 0.0241 | 0.1143 | 0.0282 | 1 |
| 3 | 0.7002 | 0.7545 | 0.0543 | 0.2061 | 0.0604 | 0 |
| 4 | 0.7746 | 0.7968 | 0.0221 | 0.0827 | 0.0322 | 2 |
| 5 | 0.7103 | 0.7364 | 0.0262 | 0.1000 | 0.0362 | 1 |
| 6 | 0.6901 | 0.7203 | 0.0302 | 0.1282 | 0.0362 | 0 |
| 7 | 0.6740 | 0.7022 | 0.0282 | 0.1094 | 0.0423 | 3 |
| 8 | 0.6942 | 0.7203 | 0.0262 | 0.0942 | 0.0382 | 2 |
| 9 | 0.7042 | 0.7324 | 0.0282 | 0.1167 | 0.0342 | 1 |

### CiteSeer
| split | MLP | FINAL_V3 | Δ | low-conf gain | changed frac | harm count |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.6667 | 0.6907 | 0.0240 | 0.2222 | 0.0450 | 5 |
| 1 | 0.6952 | 0.7117 | 0.0165 | 0.0615 | 0.0420 | 6 |
| 2 | 0.6667 | 0.6982 | 0.0315 | 0.1167 | 0.0541 | 5 |
| 3 | 0.6577 | 0.6847 | 0.0270 | 0.1023 | 0.0586 | 6 |
| 4 | 0.7005 | 0.7123 | 0.0118 | 0.0481 | 0.0448 | 5 |
| 5 | 0.6698 | 0.6887 | 0.0189 | 0.0714 | 0.0401 | 3 |
| 6 | 0.6787 | 0.6952 | 0.0165 | 0.0632 | 0.0511 | 8 |
| 7 | 0.6772 | 0.7087 | 0.0315 | 0.1082 | 0.0556 | 5 |
| 8 | 0.6712 | 0.7117 | 0.0405 | 0.1617 | 0.0556 | 1 |
| 9 | 0.7282 | 0.7628 | 0.0345 | 0.1554 | 0.0435 | 2 |

## Interpretation
- FINAL_V3 is best interpreted as a selective mixed-regime method: it preserves strong MLP predictions while correcting uncertain nodes when graph evidence is reliable.
- Gains concentrate in uncertain/low-confidence subsets (positive low-confidence bucket gain) rather than broad graph-wide relabeling.
- On regimes where graph signals are less aligned, harmful-overwrite rates increase and net gains shrink.

## Artifacts
- `logs/final_v3_regime_analysis_per_split.csv`
- `logs/final_v3_regime_analysis_per_dataset.csv`
- `tables/final_v3_regime_summary.csv`
- `tables/final_v3_regime_summary.md`
- `reports/figures/final_v3_gain_vs_homophily.png`
- `reports/figures/final_v3_gain_vs_mlp_strength.png`
- `reports/figures/harm_rate_vs_coverage.png`
- `reports/figures/lowconf_gain_vs_overall_gain.png`

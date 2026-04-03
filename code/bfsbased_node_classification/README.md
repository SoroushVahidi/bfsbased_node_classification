# Node-classification code directory

This directory contains the implementation code for the research paper.

## Canonical FINAL_V3 scripts

| File | Role |
|------|------|
| `final_method_v3.py` | Core FINAL_V3 implementation |
| `../final_method_v3.py` | Stable import path (one level up) |
| `run_final_evaluation.py` | 10-split benchmark driver |
| `gcn_baseline_runner.py` | Standalone GCN baseline (Table 1) |
| `analyze_final_v3_regimes.py` | Regime analysis helper |
| `build_final_v3_regime_analysis.py` | Regime analysis builder |
| `bfsbased-full-investigate-homophil.py` | Core module (all method variants) |
| `standard_node_baselines.py` | GCN/APPNP training helpers |
| `make_splits.py` | Split generation utility |

## Legacy / exploratory scripts (kept for provenance)

| File | Status | Notes |
|------|--------|-------|
| `manuscript_runner.py` | 📁 **Legacy** (UG-SGC) | Original uncertainty-gated runner; not canonical FINAL_V3 |
| `resubmission_runner.py` | 🔬 **Exploratory** (UG-SGC-S) | Structural extension + GCN/APPNP baselines |
| `experimental_archived/` | 🗄️ **Archived** | Older diagnostic scripts |

Additional legacy and exploratory scripts have been moved to
`archive/legacy_code/` and `archive/exploratory_code/`.

## Notes

- Do not mix FINAL_V3 result files with UG-SGC or UG-SGC-S artifacts.
- `bfsbased-full-investigate-homophil.py` is dynamically loaded by
  `manuscript_runner.py` — it is not directly importable due to the hyphenated
  filename and notebook-style top-level code.
- Always pass `--output-tag` when re-running experiments to avoid overwriting
  canonical evidence files.

For the full repository map see the root `README.md` and `REPO_STATUS.md`.

License: [MIT](../../LICENSE)

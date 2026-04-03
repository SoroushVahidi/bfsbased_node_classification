# Node-classification code

This directory includes both active canonical code and retained historical runners.

## Canonical

| File | Role |
|---|---|
| `final_method_v3.py` | Canonical `FINAL_V3` implementation. |
| `../final_method_v3.py` | Stable import alias for canonical method. |
| `run_final_evaluation.py` | Canonical 10-split evaluation driver. |

## Supporting canonical utilities

| File | Role |
|---|---|
| `gcn_baseline_runner.py` | Baseline runner used in canonical comparison workflow. |
| `analyze_final_v3_regimes.py` | Regime analysis helper for canonical reports. |
| `build_final_v3_regime_analysis.py` | Builds canonical regime summary markdown. |
| `standard_node_baselines.py` | Shared baseline helpers used by multiple runners. |
| `make_splits.py` | Split generation utility. |

## Archived / exploratory in-place scripts

These scripts remain in-place for reproducibility compatibility but are **not canonical**:

- `manuscript_runner.py` (legacy UG-SGC line)
- `resubmission_runner.py` (exploratory structural extension)
- `triple_trust_sgc.py`, `run_triple_trust_experiments.py` (exploratory variants)
- `experimental_archived/` (historical diagnostics)

For archival context, see `../../archive/README.md` and `../../docs/METHOD_STATUS.md`.

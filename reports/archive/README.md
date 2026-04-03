# Archived reports

## `final_method_v3_results_5split.csv`

Snapshot of the previous canonical per-split log (**5** GEO-GCN splits only). Superseded by `reports/final_method_v3_results.csv` (10 splits).

## `superseded_final_tables_prl/`

- **`final_tables_prl.csv`** — Frozen **10-split** per-split metrics (MLP accuracy plus SGC v1 / V2 / FINAL_V3 test accuracy and correction fields). This is the rebuild source for the canonical log:

  ```bash
  python3 scripts/prl_final_additions/rebuild_final_v3_results_10split.py
  ```

- **`final_tables_prl.md`** — Historical rendered tables (may drift; do not treat as authoritative).

For PRL-facing merged results, cite **`tables/main_results_prl.md`** (regenerated from the canonical CSV).

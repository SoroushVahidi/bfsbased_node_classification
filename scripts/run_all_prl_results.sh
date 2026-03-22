#!/usr/bin/env bash
# Regenerate PRL evidence tables and figures from frozen CSVs (no model training).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> FINAL_V3 tables + figures (cached CSVs only)"
python3 scripts/prl_final_additions/build_prl_v3_figures_and_tables.py

echo "==> Experimental setup / ablation / sensitivity tables (CSV + MD, no training)"
python3 scripts/prl_final_additions/build_prl_auxiliary_tables.py

echo "==> Syntax-check FINAL_V3 entry points (no PyTorch import)"
python3 -m py_compile code/final_method_v3.py code/bfsbased_node_classification/final_method_v3.py
echo "OK: py_compile code/final_method_v3.py"
echo "    (Full import needs PyTorch; see scripts/README_REPRODUCE.md)"

echo "Done. Outputs: tables/main_results_prl.*, figures/prl_graphical_abstract_v3.png, ..."
echo "Optional extended bundle (mirrors prl_final_additions, inventory):"
echo "  python3 scripts/prl_final_additions/build_all_prl_artifacts.py"

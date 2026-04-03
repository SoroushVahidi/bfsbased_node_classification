#!/usr/bin/env python3
"""Repository sanity check.

Verifies that:
  1. All canonical frozen files exist and are non-empty.
  2. All canonical entry-point scripts are syntactically valid (py_compile).
  3. Referenced tables and figures are present.

Usage:
  python3 scripts/check_canonical.py

Exit code 0 means all checks passed; non-zero means at least one check failed.
"""
from __future__ import annotations

import os
import py_compile
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Lists of files to check
# ---------------------------------------------------------------------------

CANONICAL_DATA_FILES = [
    # Frozen numerical evidence
    "reports/final_method_v3_results.csv",
    # Main benchmark table
    "tables/main_results_prl.md",
    "tables/main_results_prl.csv",
    # Supporting tables
    "tables/experimental_setup_prl.md",
    "tables/ablation_prl.md",
    "tables/sensitivity_prl.md",
    # Figures
    "figures/prl_graphical_abstract_v3.png",
    "figures/correction_rate_vs_homophily.png",
    "figures/safety_comparison.png",
    "figures/reliability_vs_accuracy.png",
    # Analysis docs
    "reports/final_method_v3_analysis.md",
    "reports/safety_analysis.md",
    # GCN baseline results
    "reports/gcn_results_gcn_table1.csv",
    "logs/gcn_runs_gcn_table1.jsonl",
]

CANONICAL_SCRIPTS = [
    # Stable import / canonical implementation
    "code/final_method_v3.py",
    "code/bfsbased_node_classification/final_method_v3.py",
    # Runners
    "code/bfsbased_node_classification/run_final_evaluation.py",
    "code/bfsbased_node_classification/manuscript_runner.py",
    "code/bfsbased_node_classification/gcn_baseline_runner.py",
    # Build / refresh
    "scripts/run_all_prl_results.sh",
]

CANONICAL_DOCS = [
    "README.md",
    "ANALYSIS_GUIDE.md",
    "REPO_STATUS.md",
    "README_PRL_MANUSCRIPT.md",
    "scripts/README_REPRODUCE.md",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check(path: str) -> tuple[bool, str]:
    """Return (ok, message)."""
    abs_path = os.path.join(ROOT, path)
    if not os.path.exists(abs_path):
        return False, f"MISSING : {path}"
    if os.path.getsize(abs_path) == 0:
        return False, f"EMPTY   : {path}"
    return True, f"OK      : {path}"


def syntax_check(path: str) -> tuple[bool, str]:
    """Return (ok, message) for Python syntax check."""
    abs_path = os.path.join(ROOT, path)
    if not os.path.exists(abs_path):
        return False, f"MISSING : {path}"
    if not path.endswith(".py"):
        return True, f"SKIP    : {path} (not .py)"
    try:
        py_compile.compile(abs_path, doraise=True)
        return True, f"OK      : {path}"
    except py_compile.PyCompileError as exc:
        return False, f"SYNTAX  : {path} — {exc}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    failures: list[str] = []

    print("=" * 60)
    print("Canonical data / evidence files")
    print("=" * 60)
    for f in CANONICAL_DATA_FILES:
        ok, msg = check(f)
        print(msg)
        if not ok:
            failures.append(msg)

    print()
    print("=" * 60)
    print("Canonical scripts (existence + Python syntax)")
    print("=" * 60)
    for f in CANONICAL_SCRIPTS:
        ok, msg = syntax_check(f)
        print(msg)
        if not ok:
            failures.append(msg)

    print()
    print("=" * 60)
    print("Canonical documentation")
    print("=" * 60)
    for f in CANONICAL_DOCS:
        ok, msg = check(f)
        print(msg)
        if not ok:
            failures.append(msg)

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s) failed:")
        for msg in failures:
            print(f"  {msg}")
        return 1
    else:
        print(f"All {len(CANONICAL_DATA_FILES) + len(CANONICAL_SCRIPTS) + len(CANONICAL_DOCS)} checks passed.")
        return 0


if __name__ == "__main__":
    sys.exit(main())

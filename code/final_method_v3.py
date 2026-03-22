#!/usr/bin/env python3
"""
Publication entry point for **FINAL_V3** (reliability-gated selective graph correction).

The implementation lives in ``code/bfsbased_node_classification/final_method_v3.py``.
This module exists so reviewers can import a stable path from the ``code/`` folder.

Full benchmark reproduction (MLP, SGC v1, V2 ablation, FINAL_V3)::

    python3 code/bfsbased_node_classification/run_final_evaluation.py \\
      --split-dir data/splits \\
      --datasets cora citeseer pubmed chameleon texas wisconsin \\
      --splits 0 1 2 3 4

Regenerate tables and figures from the frozen per-split CSV (no training)::

    bash scripts/run_all_prl_results.sh
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_IMPL = Path(__file__).resolve().parent / "bfsbased_node_classification" / "final_method_v3.py"
_spec = importlib.util.spec_from_file_location("_bfsbased_final_method_v3_impl", _IMPL)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load FINAL_V3 implementation from {_IMPL}")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

compute_graph_reliability = _mod.compute_graph_reliability
final_method_v3 = _mod.final_method_v3

__all__ = ["compute_graph_reliability", "final_method_v3"]

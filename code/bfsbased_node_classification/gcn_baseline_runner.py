#!/usr/bin/env python3
"""Standalone GCN baseline runner for Table 1 of the manuscript.

Runs a 2-layer GCN (Kipf & Welling, ICLR 2017) on the 6 benchmark datasets
(Cora, CiteSeer, PubMed, Chameleon, Texas, Wisconsin) using the 10 predefined
60/20/20 splits.  Hyperparameters (lr, dropout, weight decay, hidden size) are
selected on the validation set via a small grid search; the test set is only
touched once, after the best configuration has been chosen.

Architecture
------------
  Input → Ã·X·W₁ → ReLU → Dropout → Ã·H₁·W₂  (Ã = D⁻¹/²(A+I)D⁻¹/²)
  with hidden = 64 units (default grid entry).

Training defaults (tunable per dataset via the validation grid)
---------------------------------------------------------------
  optimizer : Adam
  lr        : 0.01
  dropout   : 0.5
  weight_decay : 5e-4
  max_epochs   : 500
  patience     : 100  (early stopping on validation loss)

Usage
-----
  # Full run (all 6 datasets, 10 splits each):
  python gcn_baseline_runner.py --split-dir data/splits

  # Quick smoke-test (one dataset, one split):
  python gcn_baseline_runner.py --split-dir data/splits \\
      --datasets cora --splits 0 --output-tag smoke

Output
------
  logs/gcn_runs_{output_tag}.jsonl   – one JSON record per (dataset, split)
  reports/gcn_results_{output_tag}.csv – dataset, mean_test_acc, std_test_acc
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# Allow both "run from repo root" and "run from the script's own directory".
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import manuscript_runner as mr  # noqa: E402
from standard_node_baselines import run_baseline  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
LOGS_DIR = os.path.join(REPO_ROOT, "logs")
REPORTS_DIR = os.path.join(REPO_ROOT, "reports")

GCN_DATASETS = ["cora", "citeseer", "pubmed", "chameleon", "texas", "wisconsin"]

# Training budget for GCN (per problem-statement spec).
GCN_MAX_EPOCHS = 500
GCN_PATIENCE = 100


def run_gcn_experiment(
    datasets: Optional[List[str]] = None,
    split_ids: Optional[List[int]] = None,
    split_dir: Optional[str] = None,
    output_tag: str = "gcn_table1",
) -> Dict[str, Any]:
    """Run GCN on *datasets* × *split_ids* and write results.

    Parameters
    ----------
    datasets:
        List of dataset keys to evaluate (default: all six Table-1 datasets).
    split_ids:
        Which of the 10 pre-defined splits to use (default: 0–9).
    split_dir:
        Directory containing the ``{dataset}_split_0.6_0.2_{id}.npz`` files.
        Defaults to ``<repo_root>/data/splits``.
    output_tag:
        Suffix for output file names.

    Returns
    -------
    Dict with paths to the generated JSONL and CSV files.
    """
    if datasets is None:
        datasets = GCN_DATASETS[:]
    if split_ids is None:
        split_ids = list(range(10))

    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    mod = mr._load_full_investigate_module()  # type: ignore[attr-defined]
    coverage = mr.check_split_coverage(datasets, split_ids, split_dir)  # type: ignore[attr-defined]
    runnable_datasets = [ds for ds in datasets if coverage[ds]]
    if not runnable_datasets:
        raise SystemExit(
            "No runnable datasets found. "
            "Check --split-dir points to a directory with the NPZ split files."
        )

    runs_path = os.path.join(LOGS_DIR, f"gcn_runs_{output_tag}.jsonl")
    all_records: List[Dict[str, Any]] = []
    t_wall = time.perf_counter()

    with open(runs_path, "w", encoding="utf-8") as runs_file:
        for dataset_key in runnable_datasets:
            print(f"\n=== Dataset: {dataset_key} ===")
            _dataset, data = mr._load_dataset_and_data(mod, dataset_key)  # type: ignore[attr-defined]
            device = data.x.device
            n_total = int(data.num_nodes)
            e_total = int(data.edge_index.size(1))
            num_classes = int(data.y.max().item()) + 1

            for split_id in coverage[dataset_key]:
                current_seed = (1337 + split_id) * 100
                torch.manual_seed(current_seed)
                np.random.seed(current_seed)
                print(f"  split={split_id}  seed={current_seed}", end="  ", flush=True)

                train_idx, val_idx, test_idx, split_path = mr._load_split_npz(  # type: ignore[attr-defined]
                    dataset_key, split_id, device, split_dir
                )
                split_load = np.load(split_path)

                rec: Dict[str, Any] = {
                    "method": "gcn",
                    "dataset": dataset_key,
                    "split_id": split_id,
                    "seed": current_seed,
                    "num_nodes": n_total,
                    "num_edges": e_total,
                    "num_classes": num_classes,
                    "train_count": int(split_load["train_mask"].sum()),
                    "val_count": int(split_load["val_mask"].sum()),
                    "test_count": int(split_load["test_mask"].sum()),
                    "split_file": split_path,
                    "success": True,
                    "notes": "",
                    "val_acc": None,
                    "test_acc": None,
                    "best_config": None,
                    "runtime_sec": None,
                }

                t0 = time.perf_counter()
                try:
                    result = run_baseline(
                        "gcn",
                        data,
                        train_idx,
                        val_idx,
                        test_idx,
                        seed=current_seed,
                        max_epochs=GCN_MAX_EPOCHS,
                        patience=GCN_PATIENCE,
                    )
                    runtime = time.perf_counter() - t0
                    rec.update(
                        {
                            "val_acc": result.val_acc,
                            "test_acc": result.test_acc,
                            "best_config": result.best_config,
                            "runtime_sec": runtime,
                        }
                    )
                    print(
                        f"val={result.val_acc:.4f}  test={result.test_acc:.4f}"
                        f"  cfg={result.best_config}  ({runtime:.1f}s)"
                    )
                except Exception as exc:
                    rec["success"] = False
                    rec["notes"] = str(exc)
                    print(f"FAILED: {exc}")

                all_records.append(rec)
                runs_file.write(json.dumps(rec) + "\n")
                runs_file.flush()

    # -------------------------------------------------------------------------
    # Aggregate and write summary CSV
    # -------------------------------------------------------------------------
    by_dataset: Dict[str, List[float]] = defaultdict(list)
    for rec in all_records:
        if rec["success"] and rec["test_acc"] is not None:
            by_dataset[rec["dataset"]].append(float(rec["test_acc"]))

    csv_path = os.path.join(REPORTS_DIR, f"gcn_results_{output_tag}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "mean_test_acc", "std_test_acc", "n_splits"])
        for ds in datasets:
            accs = by_dataset.get(ds, [])
            if accs:
                writer.writerow(
                    [ds, f"{np.mean(accs):.4f}", f"{np.std(accs):.4f}", len(accs)]
                )
            else:
                writer.writerow([ds, "N/A", "N/A", 0])

    wall_time = time.perf_counter() - t_wall

    # Print a compact Table-1-style summary
    print("\n=== GCN Results (Table 1) ===")
    print(f"{'Dataset':<12}  {'Mean Acc':>9}  {'Std':>7}  {'N splits':>8}")
    print("-" * 44)
    for ds in datasets:
        accs = by_dataset.get(ds, [])
        if accs:
            print(
                f"{ds:<12}  {np.mean(accs) * 100:>8.2f}%"
                f"  {np.std(accs) * 100:>6.2f}%  {len(accs):>8}"
            )
        else:
            print(f"{ds:<12}  {'N/A':>9}  {'N/A':>7}  {0:>8}")

    print(f"\nCompleted {len(all_records)} runs in {wall_time:.1f}s")
    print(f"JSONL : {runs_path}")
    print(f"CSV   : {csv_path}")

    return {
        "runs_path": runs_path,
        "csv_path": csv_path,
        "total_runs": len(all_records),
        "wall_time_sec": wall_time,
        "output_tag": output_tag,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone GCN baseline runner. "
            "Produces a summary CSV suitable for Table 1 of the manuscript."
        )
    )
    parser.add_argument(
        "--split-dir",
        default=None,
        help="Directory containing {dataset}_split_0.6_0.2_{id}.npz files "
        "(default: <repo_root>/data/splits).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=f"Datasets to evaluate (default: {GCN_DATASETS}).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        type=int,
        default=None,
        help="Split IDs to use (default: 0–9).",
    )
    parser.add_argument(
        "--output-tag",
        default="gcn_table1",
        help="Tag appended to output file names (default: gcn_table1).",
    )
    args = parser.parse_args()

    out = run_gcn_experiment(
        datasets=args.datasets,
        split_ids=args.splits,
        split_dir=args.split_dir,
        output_tag=args.output_tag,
    )
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()

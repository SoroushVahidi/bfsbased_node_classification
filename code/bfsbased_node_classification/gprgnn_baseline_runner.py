#!/usr/bin/env python3
"""Standalone GPRGNN external baseline runner.

This script evaluates GPRGNN (Adaptive Universal Generalized PageRank GNN;
Chien et al., ICLR 2021) on the repository's canonical GEO-GCN 60/20/20
split files.

Outputs:
  - logs/gprgnn_runs_{output_tag}.jsonl
  - reports/gprgnn_results_{output_tag}.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch

import manuscript_runner as mr
from standard_node_baselines import run_baseline

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOGS_DIR = os.path.join(REPO_ROOT, "logs")
REPORTS_DIR = os.path.join(REPO_ROOT, "reports")

DEFAULT_DATASETS = ["cora", "citeseer", "pubmed", "chameleon", "texas", "wisconsin"]


def run_gprgnn(
    datasets: Optional[List[str]] = None,
    split_ids: Optional[List[int]] = None,
    split_dir: Optional[str] = None,
    output_tag: str = "gprgnn_external_baseline",
) -> Dict[str, Any]:
    if datasets is None:
        datasets = DEFAULT_DATASETS[:]
    if split_ids is None:
        split_ids = list(range(10))

    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    mod = mr._load_full_investigate_module()  # type: ignore[attr-defined]
    coverage = mr.check_split_coverage(datasets, split_ids, split_dir)  # type: ignore[attr-defined]
    runnable_datasets = [ds for ds in datasets if coverage[ds]]
    if not runnable_datasets:
        raise SystemExit("No runnable datasets found for GPRGNN baseline.")

    runs_path = os.path.join(LOGS_DIR, f"gprgnn_runs_{output_tag}.jsonl")
    csv_path = os.path.join(REPORTS_DIR, f"gprgnn_results_{output_tag}.csv")
    all_records: List[Dict[str, Any]] = []
    t_wall = time.perf_counter()

    with open(runs_path, "w", encoding="utf-8") as runs_file:
        for dataset_key in runnable_datasets:
            print(f"\n=== Dataset: {dataset_key} ===")
            _dataset, data = mr._load_dataset_and_data(mod, dataset_key)  # type: ignore[attr-defined]
            device = data.x.device

            for split_id in coverage[dataset_key]:
                seed = (1337 + split_id) * 100
                torch.manual_seed(seed)
                np.random.seed(seed)
                train_idx, val_idx, test_idx, split_path = mr._load_split_npz(  # type: ignore[attr-defined]
                    dataset_key, split_id, device, split_dir
                )
                split_load = np.load(split_path)

                rec: Dict[str, Any] = {
                    "method": "gprgnn",
                    "method_family": "baseline_external",
                    "dataset": dataset_key,
                    "split_id": split_id,
                    "seed": seed,
                    "split_file": split_path,
                    "train_count": int(split_load["train_mask"].sum()),
                    "val_count": int(split_load["val_mask"].sum()),
                    "test_count": int(split_load["test_mask"].sum()),
                    "success": True,
                    "notes": "",
                    "val_acc": None,
                    "test_acc": None,
                    "runtime_sec": None,
                    "config_json": None,
                }

                t0 = time.perf_counter()
                try:
                    result = run_baseline(
                        "gprgnn",
                        data,
                        train_idx,
                        val_idx,
                        test_idx,
                        seed=seed,
                        max_epochs=500,
                        patience=100,
                    )
                    runtime = time.perf_counter() - t0
                    rec.update(
                        {
                            "val_acc": result.val_acc,
                            "test_acc": result.test_acc,
                            "runtime_sec": runtime,
                            "config_json": json.dumps(result.best_config, sort_keys=True),
                        }
                    )
                    print(
                        f"  split={split_id} seed={seed} "
                        f"val={result.val_acc:.4f} test={result.test_acc:.4f} "
                        f"({runtime:.1f}s)"
                    )
                except Exception as exc:
                    rec["success"] = False
                    rec["notes"] = str(exc)
                    print(f"  split={split_id} FAILED: {exc}")

                all_records.append(rec)
                runs_file.write(json.dumps(rec) + "\n")
                runs_file.flush()

    by_dataset: Dict[str, List[float]] = defaultdict(list)
    for rec in all_records:
        if rec["success"] and rec["test_acc"] is not None:
            by_dataset[rec["dataset"]].append(float(rec["test_acc"]))

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "mean_test_acc", "std_test_acc", "n_splits"])
        for ds in datasets:
            accs = by_dataset.get(ds, [])
            if accs:
                writer.writerow([ds, f"{np.mean(accs):.4f}", f"{np.std(accs):.4f}", len(accs)])
            else:
                writer.writerow([ds, "N/A", "N/A", 0])

    wall_time = time.perf_counter() - t_wall
    print(f"\nCompleted {len(all_records)} GPRGNN runs in {wall_time:.1f}s")
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
    parser = argparse.ArgumentParser(description="Run standalone GPRGNN external baseline.")
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--splits", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--output-tag", default="gprgnn_external_baseline")
    args = parser.parse_args()

    out = run_gprgnn(
        datasets=args.datasets,
        split_ids=args.splits,
        split_dir=args.split_dir,
        output_tag=args.output_tag,
    )
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()

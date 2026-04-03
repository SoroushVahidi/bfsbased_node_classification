#!/usr/bin/env python3
"""Run APPNP baseline on fixed 10 splits and export dataset-level mean/std CSV."""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np

import manuscript_runner as mr
from standard_node_baselines import run_baseline

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_DATASETS = ["cora", "citeseer", "pubmed", "chameleon", "texas", "wisconsin"]


def _fail(dataset: str, step: str, exc: Exception) -> None:
    print("ERROR: APPNP baseline execution failed.")
    print(f"1) Dataset/step: {dataset} / {step}")
    print(f"2) Exact error: {type(exc).__name__}: {exc}")
    print(
        "3) Needed from user: confirm split files and dataset artifacts exist "
        "and share exact traceback/environment details."
    )
    raise SystemExit(1)


def run_appnp(
    datasets: List[str],
    split_dir: str,
    output_csv: str,
) -> str:
    mod = mr._load_full_investigate_module()  # type: ignore[attr-defined]
    split_ids = list(range(10))
    coverage = mr.check_split_coverage(datasets, split_ids, split_dir)
    per_dataset_scores: Dict[str, List[float]] = defaultdict(list)

    for dataset_key in datasets:
        if coverage.get(dataset_key, []) != split_ids:
            missing = sorted(set(split_ids) - set(coverage.get(dataset_key, [])))
            print("ERROR: APPNP baseline execution failed.")
            print(f"1) Dataset/step: {dataset_key} / split_coverage")
            print(f"2) Exact error: missing required predefined splits: {missing}")
            print(
                "3) Needed from user: provide all 10 predefined 60/20/20 split "
                ".npz files for this dataset in --split-dir."
            )
            raise SystemExit(1)

        try:
            _dataset, data = mr._load_dataset_and_data(mod, dataset_key)  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - runtime guard
            _fail(dataset_key, "dataset_loading", exc)

        for split_id in split_ids:
            try:
                train_idx, val_idx, test_idx, _ = mr._load_split_npz(
                    dataset_key, split_id, data.x.device, split_dir
                )  # type: ignore[attr-defined]
                seed = 1337 + split_id
                result = run_baseline(
                    "appnp",
                    data,
                    train_idx,
                    val_idx,
                    test_idx,
                    seed=seed,
                    max_epochs=200,
                    patience=100,
                )
            except Exception as exc:  # pragma: no cover - runtime guard
                _fail(dataset_key, f"split_{split_id}", exc)
            per_dataset_scores[dataset_key].append(float(result.test_acc))

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "mean_accuracy", "std"])
        writer.writeheader()
        for ds in datasets:
            scores = np.asarray(per_dataset_scores[ds], dtype=np.float64)
            writer.writerow(
                {
                    "dataset": ds,
                    "mean_accuracy": f"{float(scores.mean()):.4f}",
                    "std": f"{float(scores.std()):.4f}",
                }
            )

    print("APPNP baseline completed successfully.")
    print(f"Datasets: {', '.join(datasets)}")
    print("Splits: 10 predefined (0-9), train/val/test = 60/20/20")
    print("Hyperparameter search: alpha in {0.1, 0.2}, K in {5, 10}; selected on validation set only")
    print(f"Output CSV: {output_csv}")
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run standalone APPNP baseline on 10 predefined splits."
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--split-dir", default=os.path.join(REPO_ROOT, "data", "splits"))
    parser.add_argument(
        "--output-csv",
        default=os.path.join(
            REPO_ROOT, "tables", "prl_resubmission", "appnp_results.csv"
        ),
    )
    args = parser.parse_args()
    run_appnp(args.datasets, args.split_dir, args.output_csv)


if __name__ == "__main__":
    main()

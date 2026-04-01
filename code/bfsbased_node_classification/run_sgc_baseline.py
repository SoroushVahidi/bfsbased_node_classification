#!/usr/bin/env python3
"""
Standalone SGC baseline runner (Wu et al., ICML 2019).

Evaluates SGC on the 6 target datasets (cora, citeseer, pubmed, chameleon,
texas, wisconsin) using the pre-defined 60/20/20 splits, reports mean test
accuracy ± std across all 10 splits, and writes a summary CSV.

Usage:
    python run_sgc_baseline.py --split-dir data/splits [--output results/sgc_wu2019_results.csv]
    python run_sgc_baseline.py --datasets cora --splits 0 1 2  # quick smoke test
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root or from this directory
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import manuscript_runner as mr
from standard_node_baselines import run_sgc_wu2019

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))

TARGET_DATASETS = ["cora", "citeseer", "pubmed", "chameleon", "texas", "wisconsin"]

DEFAULT_OUTPUT = os.path.join(REPO_ROOT, "reports", "sgc_wu2019_results.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_splits(datasets: List[str], split_ids: List[int], split_dir: Optional[str]) -> Dict[str, List[int]]:
    """Return mapping dataset -> list of available split ids."""
    coverage: Dict[str, List[int]] = {}
    for ds in datasets:
        available = []
        for sid in split_ids:
            path = mr._resolve_split_file(ds, sid, split_dir)  # type: ignore[attr-defined]
            if path is not None and os.path.isfile(path):
                available.append(sid)
        coverage[ds] = available
    return coverage


def _run_dataset(
    mod,
    dataset_key: str,
    split_ids: List[int],
    split_dir: Optional[str],
    k: int = 2,
) -> Tuple[List[float], List[Dict]]:
    """Run SGC over all splits for one dataset.  Returns (test_accs, split_infos)."""
    _, data = mr._load_dataset_and_data(mod, dataset_key)  # type: ignore[attr-defined]
    device = data.x.device

    test_accs: List[float] = []
    split_infos: List[Dict] = []

    for split_id in split_ids:
        seed = 1337 + split_id
        train_idx, val_idx, test_idx, split_path = mr._load_split_npz(  # type: ignore[attr-defined]
            dataset_key, split_id, device, split_dir
        )
        torch.manual_seed(seed)
        np.random.seed(seed)

        result = run_sgc_wu2019(
            data,
            train_idx,
            val_idx,
            test_idx,
            seed=seed,
            k=k,
        )
        test_accs.append(result.test_acc)
        split_infos.append({
            "split_id": split_id,
            "val_acc": result.val_acc,
            "test_acc": result.test_acc,
            "best_config": result.best_config,
        })

    return test_accs, split_infos


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_sgc_evaluation(
    datasets: Optional[List[str]] = None,
    split_ids: Optional[List[int]] = None,
    split_dir: Optional[str] = None,
    output_csv: str = DEFAULT_OUTPUT,
    k: int = 2,
) -> None:
    if datasets is None:
        datasets = TARGET_DATASETS[:]
    if split_ids is None:
        split_ids = list(range(10))

    print("=" * 70)
    print("SGC Baseline Evaluation  (Wu et al., ICML 2019 — K={})".format(k))
    print("=" * 70)
    print(f"Datasets   : {datasets}")
    print(f"Splits     : {split_ids}")
    print(f"Output CSV : {output_csv}")
    print()

    # Check split coverage first — stop if nothing is runnable
    coverage = _check_splits(datasets, split_ids, split_dir)
    runnable = {ds: sids for ds, sids in coverage.items() if sids}
    missing = {ds: split_ids for ds, sids in coverage.items() if not sids}

    if missing:
        print("ERROR: The following datasets have no split files available:")
        for ds, sids in missing.items():
            print(f"  {ds}: expected splits {sids}")
        print("Provide --split-dir pointing to the directory containing the .npz files.")
        sys.exit(1)

    print(f"Split coverage OK: {list(runnable.keys())}")
    print()

    mod = mr._load_full_investigate_module()  # type: ignore[attr-defined]

    summary_rows: List[Dict] = []
    all_details: List[Dict] = []
    warnings: List[str] = []

    t_total = time.perf_counter()

    for dataset_key in datasets:
        available_splits = runnable.get(dataset_key, [])
        if not available_splits:
            warnings.append(f"{dataset_key}: no splits available — skipped.")
            continue

        print(f"--- {dataset_key.upper()} ({len(available_splits)} splits) ---")
        t0 = time.perf_counter()
        try:
            test_accs, split_infos = _run_dataset(mod, dataset_key, available_splits, split_dir, k=k)
        except Exception as exc:
            msg = f"{dataset_key}: evaluation failed — {exc}"
            print(f"  ERROR: {msg}")
            warnings.append(msg)
            continue

        elapsed = time.perf_counter() - t0
        mean_acc = float(np.mean(test_accs))
        std_acc = float(np.std(test_accs, ddof=0))

        # Collect per-split best configs to report the most common one
        all_wds = [si["best_config"]["weight_decay"] for si in split_infos]
        all_lrs = [si["best_config"]["lr"] for si in split_infos]
        # Most frequent weight_decay
        best_wd = Counter(all_wds).most_common(1)[0][0]
        best_lr = Counter(all_lrs).most_common(1)[0][0]

        for si in split_infos:
            all_details.append({
                "dataset": dataset_key,
                "split_id": si["split_id"],
                "val_acc": si["val_acc"],
                "test_acc": si["test_acc"],
                "lr": si["best_config"]["lr"],
                "weight_decay": si["best_config"]["weight_decay"],
                "k": si["best_config"]["k"],
            })

        summary_rows.append({
            "dataset": dataset_key,
            "mean_accuracy": mean_acc,
            "std": std_acc,
            "n_splits": len(test_accs),
            "best_lr": best_lr,
            "best_weight_decay": best_wd,
        })

        for si in split_infos:
            print(
                f"  split {si['split_id']}: "
                f"val={si['val_acc']:.4f}  test={si['test_acc']:.4f}  "
                f"wd={si['best_config']['weight_decay']}"
            )
        print(f"  => mean={mean_acc:.4f}  std={std_acc:.4f}  ({elapsed:.1f}s)")
        print()

    wall = time.perf_counter() - t_total

    # Write output CSV (dataset, mean_accuracy, std)
    if summary_rows:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset", "mean_accuracy", "std"])
            writer.writeheader()
            for row in summary_rows:
                writer.writerow({"dataset": row["dataset"], "mean_accuracy": row["mean_accuracy"], "std": row["std"]})
        csv_saved = True
    else:
        csv_saved = False

    # -----------------------------------------------------------------------
    # Final report
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("FINAL SUMMARY — SGC (Wu et al., ICML 2019)")
    print("=" * 70)
    if summary_rows:
        print(f"\n{'Dataset':<14} {'Mean Acc':>10} {'±Std':>8}  {'Best LR':>8}  {'Best WD':>10}  {'#Splits':>7}")
        print("-" * 65)
        for row in summary_rows:
            print(
                f"{row['dataset']:<14} {row['mean_accuracy']*100:>9.2f}%"
                f" {row['std']*100:>7.2f}%"
                f"  {row['best_lr']:>8.4f}"
                f"  {row['best_weight_decay']:>10.5f}"
                f"  {row['n_splits']:>7}"
            )
        print()
        print(f"Results saved to: {output_csv}")
    else:
        print("No results to report.")

    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(f"  [WARN] {w}")

    print(f"\nTotal wall time: {wall:.1f}s")
    print("=" * 70)

    if not csv_saved:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SGC (Wu et al., 2019) baseline on node classification datasets."
    )
    parser.add_argument(
        "--split-dir",
        default=None,
        help="Directory containing .npz split files (default: data/splits relative to repo root).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to evaluate (default: cora citeseer pubmed chameleon texas wisconsin).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        type=int,
        default=list(range(10)),
        help="Split IDs to evaluate (default: 0..9).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=2,
        help="Number of propagation steps K (default: 2).",
    )
    args = parser.parse_args()

    # If split_dir not given, try the canonical location
    split_dir = args.split_dir
    if split_dir is None:
        candidate = os.path.join(REPO_ROOT, "data", "splits")
        if os.path.isdir(candidate):
            split_dir = candidate

    run_sgc_evaluation(
        datasets=args.datasets,
        split_ids=args.splits,
        split_dir=split_dir,
        output_csv=args.output,
        k=args.k,
    )


if __name__ == "__main__":
    main()

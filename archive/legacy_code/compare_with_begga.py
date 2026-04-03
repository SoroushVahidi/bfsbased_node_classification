import os
import time
import json
import subprocess
from typing import Dict, Any, List

import numpy as np

import torch

from bfs_runner import run_bfs_baseline


def get_dj_repo_root() -> str:
    """
    Resolve the path to the Diffusion-Jump-GNNs repository.
    Priority:
      1. Environment variable DIFFUSION_JUMP_REPO
      2. ../Diffusion-Jump-GNNs relative to this file
    """
    env_path = os.environ.get("DIFFUSION_JUMP_REPO")
    if env_path:
        return os.path.abspath(env_path)
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "Diffusion-Jump-GNNs"))


def run_dj_subprocess(dataset: str, extra_args: List[str] = None) -> Dict[str, Any]:
    """
    Run Begga's main.py for a given dataset as a subprocess.
    Capture stdout and parse per-split Test Accuracy lines into a list.
    """
    dj_root = get_dj_repo_root()
    if extra_args is None:
        extra_args = []

    cmd = [
        "python",
        "main.py",
        "--dataset",
        dataset,
        "--cuda",
        "cpu",
    ] + extra_args

    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", f"dj_{dataset}.log")

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=dj_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    elapsed = time.time() - t0

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(proc.stdout)

    # Parse per-split test accuracies from lines like: "Test Accuracy:  <value>"
    test_accs: List[float] = []
    for line in proc.stdout.splitlines():
        line_stripped = line.strip()
        if line_stripped.startswith("Test Accuracy:"):
            try:
                # Expect format: 'Test Accuracy:  0.875'
                parts = line_stripped.split(":")
                val_str = parts[-1].strip()
                test_accs.append(float(val_str))
            except Exception:
                continue

    return {
        "dataset": dataset,
        "cmd": cmd,
        "runtime_sec": elapsed,
        "returncode": proc.returncode,
        "log_path": log_path,
        "per_split_test_acc": test_accs,
    }


def build_split_npz_path(dataset: str, split_id: int) -> str:
    """
    Build the path to the GEO-GCN split file inside Begga's repository,
    e.g., DIFFUSION_JUMP_REPO/splits/texas_split_0.6_0.2_0.npz
    """
    dj_root = get_dj_repo_root()
    # Note: Begga uses dataset.name for filenames ("film" for Actor),
    # but for the standard heterophily benchmarks we can assume that
    # dataset CLI names match split filenames (texas, wisconsin, cornell, etc.).
    fname = f"{dataset}_split_0.6_0.2_{split_id}.npz"
    return os.path.join(dj_root, "splits", fname)


def compare_dataset(dataset: str, *, seed: int = 1337) -> List[Dict[str, Any]]:
    """
    Run DJ-GNN (Begga) and our BFS baselines on a single dataset,
    using the same GEO-GCN split files, and return structured per-split records.
    """
    records: List[Dict[str, Any]] = []

    # 1) Run DJ-GNN once (covers all 10 splits) and parse its per-split test accuracies.
    dj_result = run_dj_subprocess(dataset)
    dj_test_accs = dj_result.get("per_split_test_acc", [])

    # 2) For each split, run our BFS baselines using the same .npz file.
    num_splits = max(len(dj_test_accs), 10)  # fall back to 10 if parsing returned fewer

    for split_id in range(num_splits):
        split_npz = build_split_npz_path(dataset, split_id)
        if not os.path.exists(split_npz):
            # If the split file is missing, record a failed run and continue.
            records.append(
                {
                    "method": "bfs_plain",
                    "dataset": dataset,
                    "split_id": split_id,
                    "val_acc": None,
                    "test_acc": None,
                    "seed": seed,
                    "runtime_sec": 0.0,
                    "notes": f"missing split file: {split_npz}",
                }
            )
            continue

        # DJ record for this split (if available)
        if split_id < len(dj_test_accs):
            records.append(
                {
                    "method": "dj_gnn_paper_faithful",
                    "dataset": dataset,
                    "split_id": split_id,
                    "val_acc": None,
                    "test_acc": float(dj_test_accs[split_id]),
                    "seed": 1234,  # hardcoded in Begga's main.py
                    "runtime_sec": dj_result["runtime_sec"] / max(len(dj_test_accs), 1),
                    "notes": "Begga main.py; best test acc over epochs",
                }
            )

        # Our BFS baselines
        t0 = time.time()
        try:
            val_acc, test_plain, test_priority = run_bfs_baseline(
                dataset_key=dataset,
                split_npz_path=split_npz,
                use_priority=True,
                seed=seed + split_id,
            )
            elapsed = time.time() - t0

            records.append(
                {
                    "method": "bfs_plain",
                    "dataset": dataset,
                    "split_id": split_id,
                    "val_acc": val_acc,
                    "test_acc": test_plain,
                    "seed": seed + split_id,
                    "runtime_sec": elapsed,
                    "notes": "bfsbased-full-investigate-homophil.py: predictclass",
                }
            )
            records.append(
                {
                    "method": "bfs_priority_v1",
                    "dataset": dataset,
                    "split_id": split_id,
                    "val_acc": val_acc,
                    "test_acc": test_priority,
                    "seed": seed + split_id,
                    "runtime_sec": elapsed,
                    "notes": "bfsbased-full-investigate-homophil.py: priority_bfs_predictclass",
                }
            )
        except Exception as e:
            elapsed = time.time() - t0
            records.append(
                {
                    "method": "bfs_plain",
                    "dataset": dataset,
                    "split_id": split_id,
                    "val_acc": None,
                    "test_acc": None,
                    "seed": seed + split_id,
                    "runtime_sec": elapsed,
                    "notes": f"exception: {e}",
                }
            )

    return records


def aggregate_results(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Aggregate per-split records into per-dataset/method summaries.
    """
    summaries: List[Dict[str, Any]] = []
    by_key: Dict[tuple, List[float]] = {}

    for r in records:
        if r["test_acc"] is None:
            continue
        key = (r["dataset"], r["method"])
        by_key.setdefault(key, []).append(float(r["test_acc"]))

    for (dataset, method), vals in by_key.items():
        arr = np.array(vals, dtype=float)
        summaries.append(
            {
                "dataset": dataset,
                "method": method,
                "n_runs": int(arr.size),
                "mean_test_acc": float(arr.mean()),
                "std_test_acc": float(arr.std()),
            }
        )
    return summaries


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="List of dataset names (e.g. texas wisconsin cornell chameleon squirrel cora citeseer pubmed actor).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Base random seed for our BFS runs.",
    )
    args = parser.parse_args()

    all_records: List[Dict[str, Any]] = []
    for ds in args.datasets:
        ds = ds.lower()
        recs = compare_dataset(ds, seed=args.seed)
        all_records.extend(recs)

    os.makedirs("logs", exist_ok=True)
    records_path = os.path.join("logs", "compare_with_begga_records.jsonl")
    with open(records_path, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")

    summaries = aggregate_results(all_records)
    summary_path = os.path.join("logs", "compare_with_begga_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    print(f"Wrote per-run records to {records_path}")
    print(f"Wrote summary to {summary_path}")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()


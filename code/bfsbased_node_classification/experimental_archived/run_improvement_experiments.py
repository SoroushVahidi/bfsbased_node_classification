#!/usr/bin/env python3
"""
Experiment runner for improved SGC variants.

Usage:
  # Phase 1 screening (small subset)
  python3 code/bfsbased_node_classification/run_improvement_experiments.py \
    --phase screening --split-dir data/splits

  # Phase 2 focused evaluation (broader)
  python3 code/bfsbased_node_classification/run_improvement_experiments.py \
    --phase focused --split-dir data/splits \
    --variants V1_RELIABILITY V5_KNN_ENHANCED V6_COMBO

  # Single variant test
  python3 code/bfsbased_node_classification/run_improvement_experiments.py \
    --phase screening --split-dir data/splits --variants V1_RELIABILITY
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# Make sibling imports work when run from repo root
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from improved_sgc_variants import VARIANT_REGISTRY


def _load_full_investigate_module():
    path = os.path.join(HERE, "bfsbased-full-investigate-homophil.py")
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    start_dataset_block = "dataset = load_dataset(DATASET_KEY, root)"
    end_dataset_block = "# In[21]:"
    legacy_run_block = 'LOG_DIR = "logs"'
    if start_dataset_block in source and end_dataset_block in source:
        a = source.index(start_dataset_block)
        b = source.index(end_dataset_block, a)
        source = source[:a] + "\n" + source[b:]
    if legacy_run_block in source:
        source = source[: source.index(legacy_run_block)]
    mod = importlib.util.module_from_spec(
        importlib.util.spec_from_loader("bfs_full_investigate", loader=None)
    )
    setattr(mod, "__file__", path)
    setattr(mod, "DATASET_KEY", "texas")
    exec(compile(source, path, "exec"), mod.__dict__)
    return mod


def _load_dataset_and_data(mod, dataset_key: str):
    root = "data/"
    dataset = mod.load_dataset(dataset_key, root)
    data = dataset[0]
    data.edge_index = torch.cat([data.edge_index, data.edge_index.flip(0)], dim=1)
    data.edge_index = torch.unique(data.edge_index, dim=1)
    return dataset, data


def _resolve_split_file(dataset_key: str, split_id: int, split_dir: Optional[str]):
    split_prefix = dataset_key.lower()
    if split_prefix == "actor":
        split_prefix = "film"
    fname = f"{split_prefix}_split_0.6_0.2_{split_id}.npz"
    candidates = []
    if split_dir:
        candidates.append(os.path.join(split_dir, fname))
    candidates.append(os.path.join(HERE, "data", "splits", fname))
    candidates.append(os.path.join("data", "splits", fname))
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def _load_split(dataset_key, split_id, device, split_dir):
    path = _resolve_split_file(dataset_key, split_id, split_dir)
    if path is None:
        raise FileNotFoundError(f"Split not found: {dataset_key} split {split_id}")
    sp = np.load(path)
    to_t = lambda m: torch.as_tensor(np.where(m)[0], dtype=torch.long, device=device)
    return to_t(sp["train_mask"]), to_t(sp["val_mask"]), to_t(sp["test_mask"])


def _simple_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.size == 0:
        return 0.0
    return float((y_true == y_pred).mean())


def run_baseline_sgc(data, train_idx, val_idx, test_idx, mod, seed):
    """Run the existing baseline selective_graph_correction."""
    train_np = mod._to_numpy_idx(train_idx)
    val_np = mod._to_numpy_idx(val_idx)
    test_np = mod._to_numpy_idx(test_idx)

    mlp_probs, _ = mod.train_mlp_and_predict(
        data, train_idx,
        **mod.DEFAULT_MANUSCRIPT_MLP_KWARGS, log_file=None,
    )

    mlp_info = mod.compute_mlp_margin(mlp_probs)
    mlp_pred = mlp_info["mlp_pred_all"]
    y_true = data.y.detach().cpu().numpy().astype(np.int64)
    mlp_test_acc = _simple_accuracy(y_true[test_np], mlp_pred[test_np])

    _, test_acc, info = mod.selective_graph_correction_predictclass(
        data, train_np, val_np, test_np,
        mlp_probs=mlp_probs, seed=seed, log_file=None,
        write_node_diagnostics=False,
    )

    return mlp_probs, mlp_test_acc, test_acc, info


def run_experiment(
    datasets: List[str],
    splits: List[int],
    variants: List[str],
    split_dir: str,
    output_tag: str,
    run_baseline: bool = True,
) -> List[Dict[str, Any]]:
    """Run experiments and return records."""
    print("Loading BFS module...")
    mod = _load_full_investigate_module()

    records = []

    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds}")
        print(f"{'='*60}")

        try:
            dataset, data = _load_dataset_and_data(mod, ds)
        except Exception as e:
            print(f"  [ERROR] Failed to load {ds}: {e}")
            continue

        device = data.x.device
        for split_id in splits:
            try:
                train_idx, val_idx, test_idx = _load_split(ds, split_id, device, split_dir)
            except FileNotFoundError as e:
                print(f"  [SKIP] {e}")
                continue

            train_np = mod._to_numpy_idx(train_idx)
            val_np = mod._to_numpy_idx(val_idx)
            test_np = mod._to_numpy_idx(test_idx)

            seed = (1337 + split_id) * 100
            torch.manual_seed(seed)
            np.random.seed(seed)

            print(f"\n  split={split_id} seed={seed}")

            mlp_probs = None
            mlp_test_acc = None
            baseline_test_acc = None

            if run_baseline:
                t0 = time.perf_counter()
                mlp_probs, mlp_test_acc, baseline_test_acc, baseline_info = \
                    run_baseline_sgc(data, train_idx, val_idx, test_idx, mod, seed)
                baseline_time = time.perf_counter() - t0

                records.append({
                    "dataset": ds,
                    "split_id": split_id,
                    "seed": seed,
                    "variant": "BASELINE_SGC",
                    "test_acc_mlp": mlp_test_acc,
                    "test_acc": baseline_test_acc,
                    "delta_vs_mlp": baseline_test_acc - mlp_test_acc,
                    "val_acc": baseline_info.get("val_acc_selective"),
                    "runtime_sec": baseline_time,
                    "details": json.dumps({
                        "threshold": baseline_info.get("selected_threshold_high"),
                        "weights": baseline_info.get("selected_weights"),
                        "frac_uncertain": baseline_info.get("fraction_test_nodes_uncertain"),
                        "frac_changed": baseline_info.get("fraction_test_nodes_changed_from_mlp"),
                    }),
                })
                print(f"    BASELINE_SGC: MLP={mlp_test_acc:.4f}  SGC={baseline_test_acc:.4f}  "
                      f"delta={baseline_test_acc - mlp_test_acc:+.4f}  ({baseline_time:.1f}s)")

                records.append({
                    "dataset": ds,
                    "split_id": split_id,
                    "seed": seed,
                    "variant": "MLP_ONLY",
                    "test_acc_mlp": mlp_test_acc,
                    "test_acc": mlp_test_acc,
                    "delta_vs_mlp": 0.0,
                    "val_acc": None,
                    "runtime_sec": 0.0,
                    "details": "{}",
                })

            for vname in variants:
                if vname not in VARIANT_REGISTRY:
                    print(f"    [SKIP] Unknown variant: {vname}")
                    continue

                fn = VARIANT_REGISTRY[vname]
                t0 = time.perf_counter()
                try:
                    val_acc, test_acc, info = fn(
                        data, train_np, val_np, test_np,
                        mlp_probs=mlp_probs, seed=seed, mod=mod,
                    )
                    runtime = time.perf_counter() - t0

                    this_mlp_acc = info.get("test_acc_mlp", mlp_test_acc)
                    delta = test_acc - (this_mlp_acc or 0.0)

                    records.append({
                        "dataset": ds,
                        "split_id": split_id,
                        "seed": seed,
                        "variant": vname,
                        "test_acc_mlp": this_mlp_acc,
                        "test_acc": test_acc,
                        "delta_vs_mlp": delta,
                        "val_acc": val_acc,
                        "runtime_sec": runtime,
                        "details": json.dumps({k: v for k, v in info.items()
                                               if k not in ("variant",)}),
                    })
                    print(f"    {vname}: test={test_acc:.4f}  delta={delta:+.4f}  ({runtime:.1f}s)")

                except Exception as e:
                    runtime = time.perf_counter() - t0
                    print(f"    {vname}: [FAILED] {e}")
                    records.append({
                        "dataset": ds,
                        "split_id": split_id,
                        "seed": seed,
                        "variant": vname,
                        "test_acc_mlp": mlp_test_acc,
                        "test_acc": None,
                        "delta_vs_mlp": None,
                        "val_acc": None,
                        "runtime_sec": runtime,
                        "details": json.dumps({"error": str(e)}),
                    })

    return records


def write_csv(records: List[Dict], path: str):
    if not records:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = ["dataset", "split_id", "seed", "variant", "test_acc_mlp",
              "test_acc", "delta_vs_mlp", "val_acc", "runtime_sec", "details"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"\nResults written to: {path}")


def print_summary(records: List[Dict]):
    """Print a summary table grouped by variant and dataset."""
    from collections import defaultdict
    by_variant_ds = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r.get("test_acc") is not None:
            by_variant_ds[r["variant"]][r["dataset"]].append(r["test_acc"])

    datasets = sorted(set(r["dataset"] for r in records))
    variants = sorted(set(r["variant"] for r in records))

    print(f"\n{'='*80}")
    print("SUMMARY: Mean test accuracy by variant and dataset")
    print(f"{'='*80}")

    header = f"{'Variant':<20s}" + "".join(f"{ds:>14s}" for ds in datasets) + f"{'Mean':>14s}"
    print(header)
    print("-" * len(header))

    for v in variants:
        parts = [f"{v:<20s}"]
        accs_all = []
        for ds in datasets:
            accs = by_variant_ds[v][ds]
            if accs:
                m = np.mean(accs)
                parts.append(f"{m:>13.4f}")
                accs_all.append(m)
            else:
                parts.append(f"{'N/A':>13s}")
        if accs_all:
            parts.append(f"{np.mean(accs_all):>13.4f}")
        else:
            parts.append(f"{'N/A':>13s}")
        print(" ".join(parts[:1]) + "".join(parts[1:]))

    # Delta vs MLP summary
    print(f"\n{'='*80}")
    print("SUMMARY: Mean delta vs MLP")
    print(f"{'='*80}")

    by_variant_ds_delta = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r.get("delta_vs_mlp") is not None:
            by_variant_ds_delta[r["variant"]][r["dataset"]].append(r["delta_vs_mlp"])

    header = f"{'Variant':<20s}" + "".join(f"{ds:>14s}" for ds in datasets) + f"{'Mean':>14s}"
    print(header)
    print("-" * len(header))

    for v in variants:
        parts = [f"{v:<20s}"]
        deltas_all = []
        for ds in datasets:
            deltas = by_variant_ds_delta[v][ds]
            if deltas:
                m = np.mean(deltas)
                parts.append(f"{m:>+13.4f}")
                deltas_all.append(m)
            else:
                parts.append(f"{'N/A':>13s}")
        if deltas_all:
            parts.append(f"{np.mean(deltas_all):>+13.4f}")
        else:
            parts.append(f"{'N/A':>13s}")
        print(" ".join(parts[:1]) + "".join(parts[1:]))

    print(f"{'='*80}")


SCREENING_CONFIG = {
    "datasets": ["cora", "chameleon", "texas"],
    "splits": [0, 1, 2],
    "variants": list(VARIANT_REGISTRY.keys()),
}

FOCUSED_CONFIG = {
    "datasets": ["cora", "citeseer", "pubmed", "chameleon", "texas", "wisconsin"],
    "splits": [0, 1, 2, 3, 4],
}


def main():
    parser = argparse.ArgumentParser(description="Run improved SGC variant experiments")
    parser.add_argument("--phase", choices=["screening", "focused", "custom"],
                        default="screening")
    parser.add_argument("--split-dir", default="data/splits")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--splits", nargs="+", type=int, default=None)
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--output-tag", default=None)
    parser.add_argument("--no-baseline", action="store_true")
    args = parser.parse_args()

    if args.phase == "screening":
        datasets = args.datasets or SCREENING_CONFIG["datasets"]
        splits = args.splits or SCREENING_CONFIG["splits"]
        variants = args.variants or SCREENING_CONFIG["variants"]
        tag = args.output_tag or "screening"
    elif args.phase == "focused":
        datasets = args.datasets or FOCUSED_CONFIG["datasets"]
        splits = args.splits or FOCUSED_CONFIG["splits"]
        variants = args.variants or list(VARIANT_REGISTRY.keys())
        tag = args.output_tag or "focused"
    else:
        datasets = args.datasets or ["cora"]
        splits = args.splits or [0]
        variants = args.variants or list(VARIANT_REGISTRY.keys())
        tag = args.output_tag or "custom"

    print(f"Phase: {args.phase}")
    print(f"Datasets: {datasets}")
    print(f"Splits: {splits}")
    print(f"Variants: {variants}")
    print(f"Output tag: {tag}")

    records = run_experiment(
        datasets=datasets,
        splits=splits,
        variants=variants,
        split_dir=args.split_dir,
        output_tag=tag,
        run_baseline=not args.no_baseline,
    )

    csv_path = f"reports/method_improvement_exploration_results_{tag}.csv"
    os.makedirs("reports", exist_ok=True)
    write_csv(records, csv_path)
    print_summary(records)

    return records


if __name__ == "__main__":
    main()

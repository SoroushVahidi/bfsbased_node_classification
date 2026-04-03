#!/usr/bin/env python3
"""
Benchmark runner for Diffusion-Jump GNN (DJ-GNN), integrated from upstream:

  https://github.com/AhmedBegggaUA/Diffusion-Jump-GNNs

Uses this repository's PyG data loading (`run_final_evaluation._load_data`) and
GEO-GCN-style split `.npz` files under `--split-dir` (same as `run_final_evaluation.py`).

Outputs (deterministic names, all include `--output-tag`):
  - logs/djgnn/runs_<tag>.jsonl
  - reports/djgnn/per_split_<tag>.csv
  - reports/djgnn/summary_<tag>.csv
  - reports/djgnn/failures_<tag>.jsonl
  - run_metadata/djgnn/meta_<tag>.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_adj

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BASELINES = os.path.abspath(os.path.join(HERE, "..", "baselines"))
if BASELINES not in sys.path:
    sys.path.insert(0, BASELINES)

from djgnn.models import DJ  # noqa: E402
from djgnn.training import test as dj_test  # noqa: E402
from djgnn.training import train as dj_train  # noqa: E402
from djgnn.training import val as dj_val  # noqa: E402

from run_final_evaluation import _load_data, _load_module, _load_split  # noqa: E402

METHOD_NAME = "djgnn"

# Hyperparameters: entries marked (README) match upstream README examples; others are
# reasonable defaults aligned with main.py when upstream did not specify a dataset.
_DEFAULT_HP = {
    "hidden_channels": 64,
    "dropout": 0.5,
    "lr": 0.03,
    "wd": 5e-4,
    "epochs": 700,
    "n_layers": 10,
    "num_centers": 5,
}

_DATASET_HP: Dict[str, Dict[str, Any]] = {
    "texas": {
        "hidden_channels": 64,
        "dropout": 0.2,
        "lr": 0.03,
        "wd": 5e-4,
        "epochs": 700,
        "n_layers": 20,
        "num_centers": 5,
    },  # upstream README
    "wisconsin": {
        "hidden_channels": 64,
        "dropout": 0.5,
        "lr": 0.03,
        "wd": 5e-4,
        "epochs": 700,
        "n_layers": 5,
        "num_centers": 5,
    },  # upstream README
    "cornell": {
        "hidden_channels": 128,
        "dropout": 0.5,
        "lr": 0.03,
        "wd": 1e-3,
        "epochs": 700,
        "n_layers": 5,
        "num_centers": 5,
    },  # upstream README
    "cora": {
        "hidden_channels": 64,
        "dropout": 0.5,
        "lr": 0.01,
        "wd": 5e-4,
        "epochs": 500,
        "n_layers": 10,
        "num_centers": 5,
    },
    "citeseer": {
        "hidden_channels": 64,
        "dropout": 0.5,
        "lr": 0.01,
        "wd": 5e-4,
        "epochs": 500,
        "n_layers": 10,
        "num_centers": 5,
    },
    "pubmed": {
        "hidden_channels": 64,
        "dropout": 0.5,
        "lr": 0.01,
        "wd": 5e-4,
        "epochs": 400,
        "n_layers": 8,
        "num_centers": 5,
    },
    "chameleon": {
        "hidden_channels": 64,
        "dropout": 0.35,
        "lr": 0.03,
        "wd": 5e-4,
        "epochs": 700,
        "n_layers": 15,
        "num_centers": 5,
    },
    "squirrel": {
        "hidden_channels": 64,
        "dropout": 0.35,
        "lr": 0.03,
        "wd": 5e-4,
        "epochs": 700,
        "n_layers": 12,
        "num_centers": 5,
    },
    "actor": {
        "hidden_channels": 64,
        "dropout": 0.5,
        "lr": 0.03,
        "wd": 5e-4,
        "epochs": 700,
        "n_layers": 8,
        "num_centers": 5,
    },
}


def _git_rev(root: str) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return None


def _idx_to_mask(n: int, idx: torch.Tensor, device: torch.device) -> torch.Tensor:
    m = torch.zeros(n, dtype=torch.bool, device=device)
    m[idx.long()] = True
    return m


def _merge_hp(ds: str, overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    h = {**_DEFAULT_HP, **_DATASET_HP.get(ds, {})}
    if overrides:
        h.update({k: v for k, v in overrides.items() if v is not None})
    return h


def _estimate_dense_adj_bytes(n: int) -> int:
    return n * n * 4


def run_one_split(
    *,
    data,
    adj: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    num_features: int,
    num_classes: int,
    hp: Dict[str, Any],
    seed: int,
    device: torch.device,
) -> Tuple[float, float, float, float, int]:
    """Returns best_test_acc, last_train_acc, last_val_acc, runtime_sec, n_params."""
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    model = DJ(
        in_channels=num_features,
        hidden_channels=int(hp["hidden_channels"]),
        num_centers=int(hp["num_centers"]),
        adj_dim=adj.shape[0],
        n_jumps=int(hp["n_layers"]),
        out_channels=num_classes,
        drop_out=float(hp["dropout"]),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(hp["lr"]), weight_decay=float(hp["wd"])
    )
    best_test = 0.0
    last_train = last_val = 0.0
    t0 = time.perf_counter()
    for _epoch in range(int(hp["epochs"])):
        _, acc_train = dj_train(adj, data, model, train_mask, optimizer, criterion)
        acc_val = dj_val(adj, data, model, val_mask)
        acc_test = dj_test(adj, data, model, test_mask)
        if acc_test > best_test:
            best_test = acc_test
        last_train = float(acc_train)
        last_val = float(acc_val)
    elapsed = time.perf_counter() - t0
    return float(best_test), last_train, last_val, elapsed, n_params


def main() -> None:
    parser = argparse.ArgumentParser(description="DJ-GNN benchmark (integrated baseline)")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "cora",
            "citeseer",
            "pubmed",
            "chameleon",
            "texas",
            "wisconsin",
            "actor",
            "cornell",
            "squirrel",
        ],
        help="Dataset keys (same as run_final_evaluation / baseline suite).",
    )
    parser.add_argument("--split-dir", default="data/splits")
    parser.add_argument(
        "--splits", nargs="+", type=int, default=list(range(10)), help="Split indices"
    )
    parser.add_argument("--output-tag", required=True, help="Tag for all output filenames")
    parser.add_argument(
        "--data-root",
        default="data/",
        help="Root passed to legacy load_dataset (PyG cache under data/)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="torch device, e.g. cuda:0 or cpu (default: cuda:0 if available else cpu)",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=20000,
        help="Skip datasets with more nodes (dense N×N adjacency); 0 = no limit",
    )
    parser.add_argument(
        "--max-dense-adj-bytes",
        type=int,
        default=8_000_000_000,
        help="Skip if estimated dense adj float32 bytes exceed this (default ~8GB)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for all datasets")
    parser.add_argument("--hidden-channels", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--wd", type=float, default=None)
    args = parser.parse_args()

    tag = args.output_tag
    out_logs = os.path.join(REPO_ROOT, "logs", "djgnn")
    out_reports = os.path.join(REPO_ROOT, "reports", "djgnn")
    out_tables = os.path.join(REPO_ROOT, "tables", "djgnn")
    out_meta = os.path.join(REPO_ROOT, "run_metadata", "djgnn")
    for d in (out_logs, out_reports, out_tables, out_meta):
        os.makedirs(d, exist_ok=True)

    jsonl_path = os.path.join(out_logs, f"runs_{tag}.jsonl")
    fail_path = os.path.join(out_reports, f"failures_{tag}.jsonl")
    per_split_csv = os.path.join(out_reports, f"per_split_{tag}.csv")
    summary_csv = os.path.join(out_reports, f"summary_{tag}.csv")
    table_csv = os.path.join(out_tables, f"summary_{tag}.csv")
    meta_path = os.path.join(out_meta, f"meta_{tag}.json")

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    overrides = {
        "epochs": args.epochs,
        "hidden_channels": args.hidden_channels,
        "n_layers": args.n_layers,
        "lr": args.lr,
        "dropout": args.dropout,
        "wd": args.wd,
    }

    split_dir = args.split_dir
    if not os.path.isabs(split_dir):
        split_dir = os.path.join(REPO_ROOT, split_dir)
    data_root = args.data_root
    if not os.path.isabs(data_root):
        data_root = os.path.join(REPO_ROOT, data_root)

    os.chdir(REPO_ROOT)

    meta = {
        "method": METHOD_NAME,
        "output_tag": tag,
        "device": str(device),
        "datasets": list(args.datasets),
        "splits": list(args.splits),
        "split_dir": split_dir,
        "data_root": data_root,
        "git_rev": _git_rev(REPO_ROOT),
        "upstream_repo": "https://github.com/AhmedBegggaUA/Diffusion-Jump-GNNs",
        "max_nodes": args.max_nodes,
        "max_dense_adj_bytes": args.max_dense_adj_bytes,
    }
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2)

    mod = _load_module()
    rows: List[Dict[str, Any]] = []
    open(jsonl_path, "w", encoding="utf-8").close()
    open(fail_path, "w", encoding="utf-8").close()

    def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, default=str) + "\n")

    for ds in args.datasets:
        hp = _merge_hp(ds, overrides)
        print(f"\n{'='*60}\nDataset: {ds}\n{'='*60}")
        try:
            data = _load_data(mod, ds, data_root)
        except Exception as e:
            rec = {
                "dataset": ds,
                "split_id": None,
                "method": METHOD_NAME,
                "success": False,
                "stage": "load_dataset",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "output_tag": tag,
            }
            append_jsonl(fail_path, rec)
            print(f"  [FAIL load_dataset] {e}")
            continue

        n = int(data.num_nodes)
        est = _estimate_dense_adj_bytes(n)
        if args.max_nodes and n > args.max_nodes:
            rec = {
                "dataset": ds,
                "split_id": None,
                "method": METHOD_NAME,
                "success": False,
                "stage": "preflight_nodes",
                "error": f"n_nodes={n} exceeds --max-nodes={args.max_nodes} (dense adjacency)",
                "output_tag": tag,
            }
            append_jsonl(fail_path, rec)
            print(f"  [SKIP] {rec['error']}")
            continue
        if est > args.max_dense_adj_bytes:
            rec = {
                "dataset": ds,
                "split_id": None,
                "method": METHOD_NAME,
                "success": False,
                "stage": "preflight_memory",
                "error": (
                    f"estimated dense adj bytes {est} exceeds "
                    f"--max-dense-adj-bytes={args.max_dense_adj_bytes}"
                ),
                "output_tag": tag,
            }
            append_jsonl(fail_path, rec)
            print(f"  [SKIP] {rec['error']}")
            continue

        data = data.to(device)
        adj = to_dense_adj(data.edge_index, max_num_nodes=n)[0].to(device)
        data.edge_index = None

        for sid in args.splits:
            seed = (1337 + int(sid)) * 100
            rec: Dict[str, Any] = {
                "dataset": ds,
                "split_id": sid,
                "method": METHOD_NAME,
                "seed": seed,
                "output_tag": tag,
                "hyperparameters": hp,
                "git_rev": meta["git_rev"],
                "device": str(device),
                "n_nodes": n,
                "cuda_available": torch.cuda.is_available(),
            }
            try:
                train_idx, val_idx, test_idx = _load_split(ds, sid, device, split_dir)
            except FileNotFoundError as e:
                rec["success"] = False
                rec["stage"] = "load_split"
                rec["error"] = str(e)
                append_jsonl(fail_path, rec)
                print(f"  [SKIP split] {e}")
                continue

            train_mask = _idx_to_mask(n, train_idx, device)
            val_mask = _idx_to_mask(n, val_idx, device)
            test_mask = _idx_to_mask(n, test_idx, device)

            try:
                best_test, tr_acc, va_acc, rt, n_params = run_one_split(
                    data=data,
                    adj=adj,
                    train_mask=train_mask,
                    val_mask=val_mask,
                    test_mask=test_mask,
                    num_features=int(data.x.size(1)),
                    num_classes=int(data.y.max().item()) + 1,
                    hp=hp,
                    seed=seed,
                    device=device,
                )
                rec["success"] = True
                rec["test_acc"] = best_test
                rec["train_acc_last_epoch"] = tr_acc
                rec["val_acc_last_epoch"] = va_acc
                rec["runtime_sec"] = rt
                rec["n_parameters"] = n_params
                rec["error"] = None
            except Exception as e:
                rec["success"] = False
                rec["stage"] = "train_eval"
                rec["error"] = str(e)
                rec["traceback"] = traceback.format_exc()
                append_jsonl(fail_path, rec)
                print(f"  [FAIL] split={sid} {e}")
                rows.append(rec)
                append_jsonl(jsonl_path, rec)
                continue

            rows.append(rec)
            append_jsonl(jsonl_path, rec)
            print(
                f"  split={sid} seed={seed} test_acc={best_test:.4f} "
                f"runtime_sec={rt:.1f}"
            )

    # per-split CSV
    ok_rows = [r for r in rows if r.get("success")]
    fieldnames = [
        "dataset",
        "split_id",
        "method",
        "seed",
        "test_acc",
        "train_acc_last_epoch",
        "val_acc_last_epoch",
        "runtime_sec",
        "n_parameters",
        "device",
        "git_rev",
        "output_tag",
        "hyperparameters_json",
    ]
    with open(per_split_csv, "w", newline="", encoding="utf-8") as cf:
        w = csv.DictWriter(cf, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in ok_rows:
            flat = {k: r.get(k) for k in fieldnames if k != "hyperparameters_json"}
            flat["hyperparameters_json"] = json.dumps(r.get("hyperparameters", {}))
            w.writerow(flat)

    # summary by dataset
    from collections import defaultdict

    by_ds: Dict[str, List[float]] = defaultdict(list)
    for r in ok_rows:
        by_ds[r["dataset"]].append(float(r["test_acc"]))

    summary_rows = []
    for ds, accs in sorted(by_ds.items()):
        summary_rows.append(
            {
                "dataset": ds,
                "method": METHOD_NAME,
                "n_splits": len(accs),
                "mean_test_acc": float(np.mean(accs)),
                "std_test_acc": float(np.std(accs)),
                "output_tag": tag,
            }
        )

    with open(summary_csv, "w", newline="", encoding="utf-8") as sf:
        if summary_rows:
            w = csv.DictWriter(sf, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
    # duplicate under tables/ for visibility
    with open(table_csv, "w", newline="", encoding="utf-8") as tf:
        if summary_rows:
            w = csv.DictWriter(tf, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)

    print(f"\nWrote:\n  {jsonl_path}\n  {per_split_csv}\n  {summary_csv}\n  {fail_path}\n  {meta_path}")


if __name__ == "__main__":
    main()

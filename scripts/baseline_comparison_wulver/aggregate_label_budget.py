#!/usr/bin/env python3
"""Aggregate label-budget JSONL shards: mean/std test accuracy by method × dataset × train_fraction."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple


def _mean_std(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(v)


def load_rows(paths: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    return [r for r in rows if r.get("method") and not str(r["method"]).startswith("__")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help='e.g. outputs/label_budget/TAG/per_run/*.jsonl')
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        print("No files matched:", args.glob, file=sys.stderr)
        sys.exit(1)

    rows = load_rows(paths)
    os.makedirs(args.output_dir, exist_ok=True)
    summ = os.path.join(args.output_dir, "summaries")
    os.makedirs(summ, exist_ok=True)

    by_key: Dict[Tuple[str, str, float], List[float]] = defaultdict(list)
    by_delta: Dict[Tuple[str, str, float], List[float]] = defaultdict(list)
    for r in rows:
        if not r.get("success", True):
            continue
        m = r.get("method")
        d = r.get("dataset")
        tf = r.get("train_fraction")
        ta = r.get("test_acc")
        dv = r.get("delta_vs_mlp")
        if m is None or d is None or tf is None or ta is None:
            continue
        by_key[(m, d, float(tf))].append(float(ta))
        if dv is not None and m != "MLP_ONLY":
            by_delta[(m, d, float(tf))].append(float(dv))

    out_csv = os.path.join(summ, "mean_std_by_method_dataset_train_fraction.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "dataset", "train_fraction", "mean_test_acc", "std_test_acc", "n_splits"])
        for (m, d, tf) in sorted(by_key.keys(), key=lambda x: (x[1], x[2], x[0])):
            xs = by_key[(m, d, tf)]
            mu, sd = _mean_std(xs)
            w.writerow([m, d, tf, f"{mu:.6f}", f"{sd:.6f}", len(xs)])

    delta_csv = os.path.join(summ, "mean_delta_vs_mlp_by_method_dataset_train_fraction.csv")
    with open(delta_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "dataset", "train_fraction", "mean_delta_vs_mlp", "std_delta_vs_mlp", "n_splits"])
        for (m, d, tf) in sorted(by_delta.keys(), key=lambda x: (x[1], x[2], x[0])):
            xs = by_delta[(m, d, tf)]
            mu, sd = _mean_std(xs)
            w.writerow([m, d, tf, f"{mu:.6f}", f"{sd:.6f}", len(xs)])

    # FINAL_V3 gain vs MLP trend helper (same file as delta for FINAL_V3 only)
    v3_csv = os.path.join(summ, "final_v3_delta_vs_mlp_by_dataset_train_fraction.csv")
    with open(v3_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "train_fraction", "mean_delta_vs_mlp", "std_delta_vs_mlp", "n_splits"])
        for (m, d, tf) in sorted(by_delta.keys(), key=lambda x: (x[1], x[2], x[0])):
            if m != "FINAL_V3":
                continue
            xs = by_delta[(m, d, tf)]
            mu, sd = _mean_std(xs)
            w.writerow([d, tf, f"{mu:.6f}", f"{sd:.6f}", len(xs)])

    print(f"Wrote:\n  {out_csv}\n  {delta_csv}\n  {v3_csv}")


if __name__ == "__main__":
    main()

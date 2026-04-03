#!/usr/bin/env python3
"""Aggregate graph-noise JSONL: mean/std by method × dataset × noise_fraction."""
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
    ap.add_argument("--glob", required=True)
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

    by_acc: Dict[Tuple[str, str, float], List[float]] = defaultdict(list)
    by_delta: Dict[Tuple[str, str, float], List[float]] = defaultdict(list)
    by_frac_corr: Dict[Tuple[str, str, float], List[float]] = defaultdict(list)

    for r in rows:
        if not r.get("success", True):
            continue
        m = r.get("method")
        d = r.get("dataset")
        nf = r.get("noise_fraction")
        ta = r.get("test_acc")
        dv = r.get("delta_vs_mlp")
        fc = r.get("frac_corrected")
        if m is None or d is None or nf is None or ta is None:
            continue
        key = (m, d, float(nf))
        by_acc[key].append(float(ta))
        if dv is not None and m != "MLP_ONLY":
            by_delta[key].append(float(dv))
        if fc is not None and m == "FINAL_V3":
            by_frac_corr[key].append(float(fc))

    out1 = os.path.join(summ, "mean_std_by_method_dataset_noise.csv")
    with open(out1, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "dataset", "noise_fraction", "mean_test_acc", "std_test_acc", "n_splits"])
        for key in sorted(by_acc.keys(), key=lambda x: (x[1], x[2], x[0])):
            xs = by_acc[key]
            mu, sd = _mean_std(xs)
            w.writerow([key[0], key[1], key[2], f"{mu:.6f}", f"{sd:.6f}", len(xs)])

    out2 = os.path.join(summ, "mean_delta_vs_mlp_by_method_dataset_noise.csv")
    with open(out2, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "dataset", "noise_fraction", "mean_delta_vs_mlp", "std_delta_vs_mlp", "n_splits"])
        for key in sorted(by_delta.keys(), key=lambda x: (x[1], x[2], x[0])):
            xs = by_delta[key]
            mu, sd = _mean_std(xs)
            w.writerow([key[0], key[1], key[2], f"{mu:.6f}", f"{sd:.6f}", len(xs)])

    out3 = os.path.join(summ, "final_v3_robustness_summary.csv")
    with open(out3, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "dataset",
                "noise_fraction",
                "mean_test_acc",
                "std_test_acc",
                "mean_delta_vs_mlp",
                "std_delta_vs_mlp",
                "mean_frac_corrected",
                "n_splits",
            ]
        )
        datasets = sorted({k[1] for k in by_acc if k[0] == "FINAL_V3"})
        noises = sorted({k[2] for k in by_acc if k[0] == "FINAL_V3"})
        for d in datasets:
            for nf in noises:
                k_acc = ("FINAL_V3", d, nf)
                if k_acc not in by_acc:
                    continue
                xs = by_acc[k_acc]
                mu, sd = _mean_std(xs)
                dlt = by_delta.get(k_acc, [])
                dm, ds = _mean_std(dlt) if dlt else (float("nan"), float("nan"))
                fc_list = by_frac_corr.get(k_acc, [])
                fm, _ = _mean_std(fc_list) if fc_list else (float("nan"), float("nan"))
                w.writerow([d, nf, f"{mu:.6f}", f"{sd:.6f}", f"{dm:.6f}", f"{ds:.6f}", f"{fm:.6f}", len(xs)])

    print(f"Wrote:\n  {out1}\n  {out2}\n  {out3}")


if __name__ == "__main__":
    main()

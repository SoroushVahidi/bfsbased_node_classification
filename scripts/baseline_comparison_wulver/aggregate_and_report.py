#!/usr/bin/env python3
"""Merge per-dataset JSONL shards and write summary CSV + markdown discussion."""
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
    return [r for r in rows if r.get("method") and not r["method"].startswith("__")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="e.g. outputs/baseline_comparison/TAG/per_run/runs_*.jsonl")
    ap.add_argument("--output-dir", required=True, help="summaries/ + tables/ parent")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        print("No files matched:", args.glob, file=sys.stderr)
        sys.exit(1)

    rows = load_rows(paths)
    os.makedirs(args.output_dir, exist_ok=True)
    summ_dir = os.path.join(args.output_dir, "summaries")
    tbl_dir = os.path.join(args.output_dir, "tables")
    os.makedirs(summ_dir, exist_ok=True)
    os.makedirs(tbl_dir, exist_ok=True)

    by_md: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    by_runtime: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    by_hurt: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for r in rows:
        if not r.get("success", True):
            continue
        m = r.get("method")
        d = r.get("dataset")
        if m is None or d is None:
            continue
        ta = r.get("test_acc")
        if ta is not None:
            by_md[(m, d)].append(float(ta))
        rt = r.get("runtime_sec")
        if rt is not None:
            by_runtime[(m, d)].append(float(rt))
        nh = r.get("n_hurt")
        if nh is not None:
            by_hurt[(m, d)].append(float(nh))

    methods = sorted({m for (m, _) in by_md.keys()})
    datasets = sorted({d for (_, d) in by_md.keys()})

    summary_csv = os.path.join(summ_dir, "mean_std_by_method_dataset.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["method", "dataset", "n_splits", "mean_test_acc", "std_test_acc", "mean_runtime_sec", "mean_n_hurt"]
        )
        for m in methods:
            for d in datasets:
                accs = by_md.get((m, d), [])
                rts = by_runtime.get((m, d), [])
                hs = by_hurt.get((m, d), [])
                mu, sd = _mean_std(accs)
                w.writerow(
                    [
                        m,
                        d,
                        len(accs),
                        f"{mu:.6f}" if not math.isnan(mu) else "",
                        f"{sd:.6f}" if not math.isnan(sd) else "",
                        f"{sum(rts)/len(rts):.4f}" if rts else "",
                        f"{sum(hs)/len(hs):.2f}" if hs else "",
                    ]
                )

    report = os.path.join(args.output_dir, "BASELINE_COMPARISON_REPORT.md")
    lines = [
        "# Baseline comparison (automated summary)",
        "",
        "This report is generated from merged JSONL shards. Interpret with paper context.",
        "",
        "## Mean test accuracy (higher is better)",
        "",
        "| Method | " + " | ".join(datasets) + " | Overall mean |",
        "|" + "---|" * (len(datasets) + 2),
    ]
    for m in methods:
        parts = [m]
        vals = []
        for d in datasets:
            accs = by_md.get((m, d), [])
            mu, _ = _mean_std(accs)
            parts.append(f"{mu:.4f}" if accs else "")
            if accs:
                vals.append(mu)
        parts.append(f"{sum(vals)/len(vals):.4f}" if vals else "")
        lines.append("| " + " | ".join(parts) + " |")

    lines += [
        "",
        "## Mean runtime (seconds; lower is faster, approximate)",
        "",
        "| Method | " + " | ".join(datasets) + " |",
        "|" + "---|" * (len(datasets) + 1),
    ]
    for m in methods:
        parts = [m]
        for d in datasets:
            rts = by_runtime.get((m, d), [])
            parts.append(f"{sum(rts)/len(rts):.2f}" if rts else "")
        lines.append("| " + " | ".join(parts) + " |")

    lines += [
        "",
        "## Discussion prompts (fill in after review)",
        "",
        "- **Where FINAL_V3 wins vs graph baselines:** compare rows on low-homophily datasets (Texas, Wisconsin, Chameleon).",
        "- **Where it loses:** citation graphs (Cora/Citeseer/Pubmed) where deep GNNs / LINKX may dominate.",
        "- **Safety:** use `n_hurt` and `harmful_overwrite_rate` from JSONL for FINAL_V3 vs baselines (baselines: no correction metrics).",
        "- **Compute:** compare `runtime_sec` and Slurm MaxRSS (`sacct`) for fair cluster-side memory.",
        "",
        f"Source glob: `{args.glob}`",
        f"Input files: {len(paths)}",
        f"Total rows: {len(rows)}",
    ]

    with open(report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("Wrote:", summary_csv)
    print("Wrote:", report)


if __name__ == "__main__":
    main()

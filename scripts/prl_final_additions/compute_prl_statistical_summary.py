#!/usr/bin/env python3
"""
Build a compact statistical summary for the frozen FINAL_V3 benchmark.

Uses the split-level rows in `reports/final_method_v3_results.csv` and writes:
- logs/prl_statistical_summary.csv
- reports/prl_statistical_summary.md

For each dataset/method pair, this script reports:
- mean paired delta vs MLP
- std of paired deltas
- wins/losses/ties vs MLP
- a bootstrap 95% CI for the mean paired delta
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "reports" / "final_method_v3_results.csv"
CSV_OUT = ROOT / "logs" / "prl_statistical_summary.csv"
MD_OUT = ROOT / "reports" / "prl_statistical_summary.md"
BOOTSTRAP_SEED = 1337
BOOTSTRAP_ROUNDS = 5000


def load_rows():
    with RESULTS.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def paired_bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        return float(values[0]), float(values[0])
    samples = []
    for _ in range(BOOTSTRAP_ROUNDS):
        idx = rng.integers(0, n, size=n)
        samples.append(float(values[idx].mean()))
    lo, hi = np.percentile(np.asarray(samples), [2.5, 97.5])
    return float(lo), float(hi)


def main():
    rows = load_rows()
    by_dataset_method = defaultdict(list)
    for row in rows:
        by_dataset_method[(row["dataset"], row["method"])].append(row)

    methods = ["BASELINE_SGC_V1", "V2_MULTIBRANCH", "FINAL_V3"]
    out_rows = []
    md_lines = [
        "# PRL statistical summary",
        "",
        "This summary uses the frozen split-level rows in `reports/final_method_v3_results.csv`.",
        "Each dataset contributes 10 paired split realizations for `MLP_ONLY`, `BASELINE_SGC_V1`, `V2_MULTIBRANCH`, and `FINAL_V3`.",
        "",
        "| Dataset | Method | Mean Δ vs MLP (pp) | 95% bootstrap CI (pp) | Std Δ (pp) | Wins | Losses | Ties |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for dataset in sorted({row["dataset"] for row in rows}):
        mlp_rows = sorted(by_dataset_method[(dataset, "MLP_ONLY")], key=lambda r: int(r["split_id"]))
        mlp_acc = np.asarray([float(r["test_acc"]) for r in mlp_rows], dtype=float)
        for method in methods:
            method_rows = sorted(by_dataset_method[(dataset, method)], key=lambda r: int(r["split_id"]))
            acc = np.asarray([float(r["test_acc"]) for r in method_rows], dtype=float)
            deltas = (acc - mlp_acc) * 100.0
            wins = int((deltas > 1e-12).sum())
            losses = int((deltas < -1e-12).sum())
            ties = int((np.abs(deltas) <= 1e-12).sum())
            ci_low, ci_high = paired_bootstrap_ci(deltas)
            out_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "mean_delta_pp": f"{deltas.mean():+.2f}",
                    "ci95_low_pp": f"{ci_low:+.2f}",
                    "ci95_high_pp": f"{ci_high:+.2f}",
                    "std_delta_pp": f"{deltas.std():.2f}",
                    "wins_vs_mlp": wins,
                    "losses_vs_mlp": losses,
                    "ties_vs_mlp": ties,
                    "n_pairs": len(deltas),
                }
            )
            md_lines.append(
                f"| {dataset.capitalize()} | {method} | {deltas.mean():+.2f} | [{ci_low:+.2f}, {ci_high:+.2f}] | "
                f"{deltas.std():.2f} | {wins} | {losses} | {ties} |"
            )

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "method",
                "mean_delta_pp",
                "ci95_low_pp",
                "ci95_high_pp",
                "std_delta_pp",
                "wins_vs_mlp",
                "losses_vs_mlp",
                "ties_vs_mlp",
                "n_pairs",
            ],
        )
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)

    md_lines.extend(
        [
            "",
            "## Notes",
            "",
            "- These are paired split-level comparisons against `MLP_ONLY` within the frozen 10-split benchmark.",
            "- Bootstrap CIs summarize uncertainty in the mean paired delta over the available splits.",
            "- This is manuscript-support evidence, not a claim of universal statistical superiority.",
            "",
        ]
    )
    MD_OUT.write_text("\n".join(md_lines), encoding="utf-8")
    print("Wrote", CSV_OUT)
    print("Wrote", MD_OUT)


if __name__ == "__main__":
    main()

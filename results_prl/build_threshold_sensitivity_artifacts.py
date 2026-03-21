#!/usr/bin/env python3
"""
Build threshold-vs-behavior artifacts from merged final JSONL.

This does NOT run a controlled threshold sweep. It joins SGC runs to MLP runs
(same dataset, split_id, repeat_id) and exports threshold, uncertain fraction,
delta vs MLP for cross-run analysis.

Input:  logs/comparison_runs_manuscript_final_validation.jsonl
Output: results_prl/threshold_sensitivity_crossruns.jsonl
        results_prl/threshold_uncertainty_crossrun_summary.csv
        results_prl/threshold_sensitivity_mini_table_tertiles.csv
        results_prl/fig_*.png / .pdf
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
JSONL = ROOT / "logs" / "comparison_runs_manuscript_final_validation.jsonl"
OUT = Path(__file__).resolve().parent


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else float("nan")


def main() -> None:
    keyed_mlp: dict[tuple, float] = {}
    rows_sgc: list[dict] = []
    with JSONL.open() as f:
        for line in f:
            o = json.loads(line)
            k = (o.get("dataset"), o.get("split_id"), o.get("repeat_id"))
            if o.get("method") == "mlp_only" and o.get("success"):
                keyed_mlp[k] = float(o["test_acc"])
            if o.get("method") == "selective_graph_correction" and o.get("success"):
                rows_sgc.append(o)

    out_rows: list[dict] = []
    for o in rows_sgc:
        k = (o.get("dataset"), o.get("split_id"), o.get("repeat_id"))
        mlp_acc = keyed_mlp.get(k)
        tc = int(o.get("test_count") or 1)
        nu = o.get("sgc_n_uncertain")
        nc = o.get("sgc_n_changed")
        sgc_acc = float(o["test_acc"]) if o.get("test_acc") is not None else None
        thr = o.get("sgc_threshold")
        unc_frac = (float(nu) / tc) if nu is not None else None
        chg_frac = (float(nc) / tc) if nc is not None else None
        delta = (sgc_acc - mlp_acc) if (sgc_acc is not None and mlp_acc is not None) else None
        out_rows.append(
            {
                "dataset": o.get("dataset"),
                "split_id": o.get("split_id"),
                "repeat_id": o.get("repeat_id"),
                "threshold": thr,
                "uncertain_fraction_test": unc_frac,
                "changed_fraction_test": chg_frac,
                "sgc_test_acc": sgc_acc,
                "mlp_test_acc": mlp_acc,
                "delta_vs_mlp": delta,
            }
        )

    OUT.mkdir(parents=True, exist_ok=True)
    cross = OUT / "threshold_sensitivity_crossruns.jsonl"
    with cross.open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")

    by_ds: dict[str, list] = defaultdict(list)
    for r in out_rows:
        by_ds[str(r["dataset"])].append(r)

    summ = OUT / "threshold_uncertainty_crossrun_summary.csv"
    with summ.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "dataset",
                "n_runs",
                "mean_threshold",
                "std_threshold",
                "mean_uncertain_frac",
                "std_uncertain_frac",
                "mean_delta_vs_mlp",
                "std_delta_vs_mlp",
                "corr_thr_vs_uncertain",
            ]
        )
        for ds in sorted(by_ds):
            R = by_ds[ds]
            thrs = [float(r["threshold"]) for r in R if r.get("threshold") is not None]
            ufs = [float(r["uncertain_fraction_test"]) for r in R if r.get("uncertain_fraction_test") is not None]
            dlt = [float(r["delta_vs_mlp"]) for r in R if r.get("delta_vs_mlp") is not None]
            pairs = [
                (float(r["threshold"]), float(r["uncertain_fraction_test"]))
                for r in R
                if r.get("threshold") is not None and r.get("uncertain_fraction_test") is not None
            ]
            xs = [a for a, _ in pairs]
            ys = [b for _, b in pairs]
            w.writerow(
                [
                    ds,
                    len(R),
                    statistics.mean(thrs) if thrs else "",
                    statistics.pstdev(thrs) if len(thrs) > 1 else "",
                    statistics.mean(ufs) if ufs else "",
                    statistics.pstdev(ufs) if len(ufs) > 1 else "",
                    statistics.mean(dlt) if dlt else "",
                    statistics.pstdev(dlt) if len(dlt) > 1 else "",
                    f"{pearson(xs, ys):.3f}" if len(pairs) > 1 else "",
                ]
            )

    focus = ["cora", "citeseer", "pubmed", "chameleon", "wisconsin", "texas"]
    mini_rows: list[dict] = []
    for ds in focus:
        R = [r for r in out_rows if r["dataset"] == ds and r.get("threshold") is not None]
        R.sort(key=lambda x: float(x["threshold"]))
        n = len(R)
        if n < 3:
            continue
        cuts = [0, n // 3, 2 * n // 3, n]
        labels = ["low_tertile", "mid_tertile", "high_tertile"]
        for j in range(3):
            chunk = R[cuts[j] : cuts[j + 1]]
            mini_rows.append(
                {
                    "dataset": ds,
                    "bucket": labels[j],
                    "mean_threshold": statistics.mean([float(x["threshold"]) for x in chunk]),
                    "mean_uncertain_fraction": statistics.mean(
                        [float(x["uncertain_fraction_test"]) for x in chunk if x["uncertain_fraction_test"] is not None]
                    ),
                    "mean_delta_vs_mlp": statistics.mean(
                        [float(x["delta_vs_mlp"]) for x in chunk if x["delta_vs_mlp"] is not None]
                    ),
                    "n_runs": len(chunk),
                }
            )

    with (OUT / "threshold_sensitivity_mini_table_tertiles.csv").open("w", newline="") as f:
        if mini_rows:
            w = csv.DictWriter(f, fieldnames=list(mini_rows[0].keys()))
            w.writeheader()
            w.writerows(mini_rows)

    colors = plt.cm.tab10(np.linspace(0, 1, len(focus)))
    fig1, ax1 = plt.subplots(figsize=(5.5, 3.5))
    fig2, ax2 = plt.subplots(figsize=(5.5, 3.5))
    for i, ds in enumerate(focus):
        R = [r for r in out_rows if r["dataset"] == ds]
        th = [float(r["threshold"]) for r in R]
        uf = [float(r["uncertain_fraction_test"]) for r in R if r["uncertain_fraction_test"] is not None]
        th_u = [float(r["threshold"]) for r in R if r["uncertain_fraction_test"] is not None]
        dl = [float(r["delta_vs_mlp"]) for r in R if r["delta_vs_mlp"] is not None]
        th_d = [float(r["threshold"]) for r in R if r["delta_vs_mlp"] is not None]
        ax1.scatter(th_u, uf, s=12, alpha=0.65, label=ds, color=colors[i])
        ax2.scatter(th_d, dl, s=12, alpha=0.65, label=ds, color=colors[i])

    ax1.set_xlabel("Selected uncertainty threshold (val-tuned)")
    ax1.set_ylabel("Test uncertain-node fraction")
    ax1.legend(fontsize=7, ncol=2, loc="best")
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(OUT / "fig_threshold_vs_uncertain_fraction_crossruns.png", dpi=200)
    fig1.savefig(OUT / "fig_threshold_vs_uncertain_fraction_crossruns.pdf")

    ax2.set_xlabel("Selected uncertainty threshold (val-tuned)")
    ax2.set_ylabel(r"$\Delta$ test acc (SGC $-$ MLP)")
    ax2.axhline(0, color="k", lw=0.6, alpha=0.5)
    ax2.legend(fontsize=7, ncol=2, loc="best")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(OUT / "fig_threshold_vs_delta_crossruns.png", dpi=200)
    fig2.savefig(OUT / "fig_threshold_vs_delta_crossruns.pdf")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, metric, ylab in zip(
        axes,
        ["uncertain_fraction_test", "delta_vs_mlp"],
        ["Test uncertain fraction", r"$\Delta$ test acc (SGC $-$ MLP)"],
    ):
        for i, ds in enumerate(focus):
            R = [r for r in out_rows if r["dataset"] == ds]
            th = [float(r["threshold"]) for r in R if r.get(metric) is not None]
            yv = [float(r[metric]) for r in R if r.get(metric) is not None]
            ax.scatter(th, yv, s=10, alpha=0.65, label=ds, color=colors[i])
        ax.set_xlabel("Selected threshold (val)")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
        if metric == "delta_vs_mlp":
            ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    axes[0].legend(fontsize=6, ncol=3, loc="upper right")
    axes[1].legend(fontsize=6, ncol=3, loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "fig_threshold_sensitivity_panels_crossruns.png", dpi=200)
    fig.savefig(OUT / "fig_threshold_sensitivity_panels_crossruns.pdf")

    print("Wrote:", cross, summ, "figures in", OUT)


if __name__ == "__main__":
    main()

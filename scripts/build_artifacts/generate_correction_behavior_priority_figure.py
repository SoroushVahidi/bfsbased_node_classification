#!/usr/bin/env python3
"""
Grouped bars: validation-selected threshold_high and avg uncertain fraction
for priority datasets (descriptive; from manuscript_correction_behavior_final_validation.csv).
"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "logs" / "manuscript_correction_behavior_final_validation.csv"
OUT = ROOT / "prl_final_additions" / "figures"
MIRROR = ROOT / "figures" / "prl_final_additions"

PRIORITY = ["cora", "citeseer", "pubmed", "chameleon"]


def main() -> None:
    rows = {r["dataset"]: r for r in csv.DictReader(CSV_PATH.open())}
    labels = []
    thr = []
    unc = []
    for ds in PRIORITY:
        r = rows[ds]
        labels.append(ds)
        thr.append(float(r["avg_threshold_high"]))
        unc.append(float(r["avg_uncertain_frac"]))

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    ax.bar(x - w / 2, thr, width=w, label=r"mean $\tau$ (threshold_high)", color="#4477AA")
    ax.bar(x + w / 2, unc, width=w, label="mean uncertain fraction", color="#EE6677")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("value (test-aggregated means over 30 runs)")
    ax.set_title("Gate statistics: priority datasets (descriptive)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = OUT / f"fig_gate_stats_priority_four.{ext}"
        fig.savefig(p, dpi=200 if ext == "png" else None)
    MIRROR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT / "fig_gate_stats_priority_four.pdf", MIRROR / "fig_gate_stats_priority_four.pdf")
    shutil.copy2(OUT / "fig_gate_stats_priority_four.png", MIRROR / "fig_gate_stats_priority_four.png")
    print("Wrote", OUT / "fig_gate_stats_priority_four.pdf")


if __name__ == "__main__":
    main()

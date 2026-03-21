#!/usr/bin/env python3
"""Bar chart: delta (SGC - MLP) from prl_main_benchmark_merged.csv."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TBL = ROOT / "prl_final_additions" / "tables" / "prl_main_benchmark_merged.csv"
OUT = ROOT / "prl_final_additions" / "figures"


def main() -> None:
    rows = list(csv.DictReader(TBL.open()))
    ds = [r["dataset"] for r in rows]
    delta = [float(r["delta_sgc_minus_mlp"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    colors = ["#228833" if d > 0.005 else ("#CC6677" if d < -0.005 else "#666666") for d in delta]
    ax.barh(ds[::-1], delta[::-1], color=colors[::-1])
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel(r"$\Delta$ test accuracy (SGC $-$ MLP)")
    ax.set_title("Merged validation: selective correction vs MLP (9 datasets)")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / "fig_delta_vs_mlp_bar.pdf")
    fig.savefig(OUT / "fig_delta_vs_mlp_bar.png", dpi=200)
    print("Wrote", OUT / "fig_delta_vs_mlp_bar.pdf")


if __name__ == "__main__":
    main()

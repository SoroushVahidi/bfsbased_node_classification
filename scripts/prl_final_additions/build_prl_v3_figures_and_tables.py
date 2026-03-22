#!/usr/bin/env python3
"""
PRL figure + table bundle for FINAL_V3 (no experiments — reads existing CSVs only).

Outputs:
  figures/prl_graphical_abstract_v3.png
  figures/correction_rate_vs_homophily.png
  figures/safety_comparison.png
  figures/reliability_vs_accuracy.png
  tables/main_results_prl.csv
  tables/main_results_prl.md
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrow, FancyBboxPatch

mpl.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "figures"
TBL = ROOT / "tables"
RESULTS = ROOT / "reports" / "final_method_v3_results.csv"
REGIME_MAIN = ROOT / "logs" / "manuscript_regime_analysis_final_validation_main.csv"
REGIME_JOB2 = ROOT / "logs" / "regime_analysis_manuscript_final_validation_job2.csv"


def load_homophily() -> dict[str, float]:
    h: dict[str, float] = {}
    for path in (REGIME_MAIN, REGIME_JOB2):
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ds = row["dataset"].strip().lower()
                hv = row.get("homophily", "").strip()
                if hv and hv.upper() != "N/A":
                    try:
                        h[ds] = float(hv)
                    except ValueError:
                        pass
    return h


def load_v3_results() -> list[dict]:
    rows = []
    with RESULTS.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["split_id"] = int(row["split_id"])
            row["test_acc"] = float(row["test_acc"])
            row["delta_vs_mlp"] = float(row["delta_vs_mlp"])
            row["frac_corrected"] = float(row["frac_corrected"] or 0)
            row["frac_confident"] = float(row["frac_confident"] or 0)
            row["frac_unreliable"] = float(row["frac_unreliable"] or 0)
            row["correction_precision"] = float(row["correction_precision"] or 1.0)
            row["tau"] = float(row["tau"]) if row.get("tau") not in ("", None) else float("nan")
            row["rho"] = float(row["rho"]) if row.get("rho") not in ("", None) else float("nan")
            nh, hh = row.get("n_helped", ""), row.get("n_hurt", "")
            row["n_helped"] = int(nh) if nh not in ("", None) else 0
            row["n_hurt"] = int(hh) if hh not in ("", None) else 0
            rows.append(row)
    return rows


def plot_graphical_abstract_v3(out: Path) -> None:
    """Minimal two-panel abstract: uncontrolled correction vs reliability-gated FINAL_V3."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            "font.size": 9,
            "axes.linewidth": 0.9,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(7.2, 2.8), dpi=150, gridspec_kw={"wspace": 0.28}
    )
    for ax in (ax_l, ax_r):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    # --- Left: standard / uncontrolled ---
    ax_l.set_title("Standard graph correction", fontsize=10, color="0.1", pad=6)
    cx, cy = 0.52, 0.48
    ax_l.add_patch(Circle((cx, cy), 0.1, facecolor="0.85", edgecolor="0.2", linewidth=1.2))
    ax_l.text(cx, cy, "v", ha="center", va="center", fontsize=11, color="0.15", fontweight="bold")
    bad = [(-0.05, 0.55), (0.08, 0.78), (0.72, 0.72), (0.82, 0.42), (0.12, 0.22)]
    for i, (bx, by) in enumerate(bad):
        ax_l.add_patch(Circle((bx, by), 0.055, facecolor="0.75", edgecolor="0.35", linewidth=0.8))
        ax_l.add_patch(
            FancyArrow(
                bx,
                by,
                (cx - bx) * 0.55,
                (cy - by) * 0.55,
                width=0.012,
                head_width=0.06,
                head_length=0.05,
                length_includes_head=True,
                facecolor="0.45",
                edgecolor="0.25",
                linewidth=0.5,
            )
        )
    ax_l.text(0.5, 0.08, "Heterophily: noisy neighbor influence", ha="center", fontsize=8, color="0.35")

    # --- Right: FINAL_V3 ---
    ax_r.set_title("FINAL_V3 (reliability-gated)", fontsize=10, color="0.1", pad=6)
    ax_r.add_patch(
        FancyBboxPatch(
            (0.06, 0.62),
            0.34,
            0.28,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            facecolor="0.92",
            edgecolor="0.35",
            linewidth=1.0,
        )
    )
    ax_r.text(0.23, 0.82, "MLP", ha="center", va="center", fontsize=9, fontweight="bold", color="0.2")
    ax_r.text(0.23, 0.7, "base pred.", ha="center", va="center", fontsize=7.5, color="0.4")
    ax_r.add_patch(
        FancyBboxPatch(
            (0.48, 0.62),
            0.22,
            0.28,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            facecolor="0.88",
            edgecolor="0.35",
            linewidth=1.0,
        )
    )
    ax_r.annotate(
        "",
        xy=(0.48, 0.76),
        xytext=(0.4, 0.76),
        arrowprops=dict(arrowstyle="-|>", color="0.35", lw=0.9, shrinkA=2, shrinkB=2),
    )
    ax_r.text(0.59, 0.76, "Gate\nR(v)", ha="center", va="center", fontsize=8, fontweight="bold", color="0.25")
    # High R branch
    ax_r.add_patch(
        FancyBboxPatch(
            (0.74, 0.68),
            0.22,
            0.22,
            boxstyle="round,pad=0.015,rounding_size=0.015",
            facecolor="1.0",
            edgecolor="0.25",
            linewidth=1.0,
        )
    )
    ax_r.text(0.85, 0.79, "graph\ncorrect", ha="center", va="center", fontsize=7.5, color="0.2")
    ax_r.annotate(
        "",
        xy=(0.85, 0.68),
        xytext=(0.64, 0.72),
        arrowprops=dict(arrowstyle="-|>", color="0.3", lw=1.0, shrinkA=0, shrinkB=2),
    )
    ax_r.text(0.68, 0.66, "high R", fontsize=7, color="0.4")
    # Low R branch
    ax_r.add_patch(
        FancyBboxPatch(
            (0.74, 0.28),
            0.22,
            0.22,
            boxstyle="round,pad=0.015,rounding_size=0.015",
            facecolor="0.94",
            edgecolor="0.35",
            linewidth=1.0,
        )
    )
    ax_r.text(0.85, 0.39, "keep\nMLP", ha="center", va="center", fontsize=7.5, color="0.2")
    ax_r.annotate(
        "",
        xy=(0.85, 0.5),
        xytext=(0.64, 0.68),
        arrowprops=dict(arrowstyle="-|>", color="0.3", lw=1.0, shrinkA=0, shrinkB=2),
    )
    ax_r.text(0.68, 0.52, "low R", fontsize=7, color="0.4")
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def plot_correction_vs_homophily(rows: list[dict], homophily: dict[str, float], out: Path) -> None:
    methods = {"FINAL_V3": [], "BASELINE_SGC_V1": []}
    by_ds: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["method"] not in methods:
            continue
        by_ds[r["dataset"]][r["method"]].append(r["frac_corrected"])

    datasets = sorted(by_ds.keys(), key=lambda d: homophily.get(d, 0.0))
    H = [homophily.get(d, float("nan")) for d in datasets]
    mean_v3 = [float(np.mean(by_ds[d]["FINAL_V3"])) for d in datasets]
    std_v3 = [float(np.std(by_ds[d]["FINAL_V3"])) for d in datasets]
    mean_v1 = [float(np.mean(by_ds[d]["BASELINE_SGC_V1"])) for d in datasets]
    std_v1 = [float(np.std(by_ds[d]["BASELINE_SGC_V1"])) for d in datasets]

    fig, ax = plt.subplots(figsize=(4.6, 3.2), dpi=150)
    ax.errorbar(
        H,
        mean_v3,
        yerr=std_v3,
        fmt="o-",
        color="0.15",
        ecolor="0.45",
        capsize=3,
        markersize=6,
        linewidth=1.2,
        label="FINAL_V3",
    )
    ax.errorbar(
        H,
        mean_v1,
        yerr=std_v1,
        fmt="s--",
        color="0.55",
        ecolor="0.65",
        capsize=3,
        markersize=5,
        linewidth=1.0,
        label="SGC v1",
    )
    short = {
        "cora": "Cora",
        "citeseer": "Cite",
        "pubmed": "Pubm",
        "chameleon": "Cham",
        "texas": "Tex",
        "wisconsin": "Wisc",
    }
    for i, d in enumerate(datasets):
        ax.annotate(
            short.get(d, d[:4]),
            (H[i], mean_v3[i]),
            textcoords="offset points",
            xytext=(4, 3),
            fontsize=7,
            color="0.35",
        )
    ax.set_xlabel("Edge homophily H")
    ax.set_ylabel("Fraction of nodes corrected")
    ax.set_title("Selective correction vs. homophily")
    ax.legend(frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(bottom=-0.02)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, color="0.75")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_safety(rows: list[dict], out: Path) -> None:
    """Count splits with delta_vs_mlp < 0 per dataset for v1 vs FINAL_V3."""
    by_ds = defaultdict(lambda: {"BASELINE_SGC_V1": 0, "FINAL_V3": 0})
    for r in rows:
        if r["method"] not in ("BASELINE_SGC_V1", "FINAL_V3"):
            continue
        ds = r["dataset"]
        if r["delta_vs_mlp"] < -1e-9:
            by_ds[ds][r["method"]] += 1

    datasets = sorted(by_ds.keys())
    x = np.arange(len(datasets))
    w = 0.35
    c1 = [by_ds[d]["BASELINE_SGC_V1"] for d in datasets]
    c2 = [by_ds[d]["FINAL_V3"] for d in datasets]

    fig, ax = plt.subplots(figsize=(5.2, 3.0), dpi=150)
    ax.bar(x - w / 2, c1, width=w, label="SGC v1", color="0.55", edgecolor="0.2", linewidth=0.6)
    ax.bar(x + w / 2, c2, width=w, label="FINAL_V3", color="0.25", edgecolor="0.2", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in datasets], rotation=25, ha="right")
    ax.set_ylabel("Harmful splits (Δ test < 0)")
    ax.set_title("Safety: splits worse than MLP")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(1, max(c1 + c2) + 0.5))
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_reliability_vs_accuracy(rows: list[dict], out: Path) -> None:
    """Per-split: unreliable fraction vs correction precision (FINAL_V3)."""
    xs, ys, sizes = [], [], []
    for r in rows:
        if r["method"] != "FINAL_V3":
            continue
        if r["frac_corrected"] < 1e-6 and r["n_helped"] + r["n_hurt"] == 0:
            continue
        xs.append(r["frac_unreliable"])
        ys.append(r["correction_precision"])
        sizes.append(20 + 200 * r["frac_corrected"])

    fig, ax = plt.subplots(figsize=(4.5, 3.2), dpi=150)
    ax.scatter(xs, ys, s=sizes, c="0.35", edgecolors="0.15", linewidths=0.5, alpha=0.85)
    if len(xs) >= 3:
        z = np.polyfit(xs, ys, 1)
        xp = np.linspace(min(xs), max(xs), 50)
        ax.plot(xp, np.poly1d(z)(xp), "--", color="0.5", linewidth=1.0, label="Linear fit")
        ax.legend(frameon=False, loc="best")
    ax.set_xlabel("Fraction of nodes unreliable (low R)")
    ax.set_ylabel("Correction precision")
    ax.set_title("Reliability signal vs. correction quality")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="both", linestyle=":", linewidth=0.6, color="0.8")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_main_tables(rows: list[dict]) -> tuple[list[list[str]], list[str]]:
    methods = ["MLP_ONLY", "BASELINE_SGC_V1", "V2_MULTIBRANCH", "FINAL_V3"]
    labels = {"MLP_ONLY": "MLP", "BASELINE_SGC_V1": "SGC v1", "V2_MULTIBRANCH": "V2 multi", "FINAL_V3": "FINAL_V3"}
    by_ds: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["method"] in methods:
            by_ds[r["dataset"]][r["method"]].append(r["test_acc"])

    datasets = sorted(by_ds.keys())
    header = ["Dataset"] + [labels[m] for m in methods]
    table: list[list[str]] = []
    for d in datasets:
        row = [d.capitalize()]
        cells = []
        for m in methods:
            vals = by_ds[d][m]
            mu = float(np.mean(vals))
            sd = float(np.std(vals))
            cells.append((mu, sd))
        best_v = max(c[0] for c in cells)
        for i, m in enumerate(methods):
            mu, sd = cells[i]
            s = f"{mu:.4f} ± {sd:.4f}"
            if abs(mu - best_v) < 1e-5:
                s = f"**{s}**"
            row.append(s)
        table.append(row)
    return table, header


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([c.replace("**", "") for c in r])


def write_md(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Main results (test accuracy, mean ± std over splits)",
        "",
        "Source: `reports/final_method_v3_results.csv` (same splits as FINAL_V3 evaluation).",
        "",
        "Regenerate this table and the PRL v3 figure set:",
        "`python3 scripts/prl_final_additions/build_prl_v3_figures_and_tables.py`",
        "(reads existing CSVs only; no experiments).",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for r in rows:
        line = "| " + " | ".join(r) + " |"
        lines.append(line)
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not RESULTS.is_file():
        raise SystemExit(f"Missing {RESULTS}")
    homophily = load_homophily()
    rows = load_v3_results()

    FIG.mkdir(parents=True, exist_ok=True)
    TBL.mkdir(parents=True, exist_ok=True)

    plot_graphical_abstract_v3(FIG / "prl_graphical_abstract_v3.png")
    plot_correction_vs_homophily(rows, homophily, FIG / "correction_rate_vs_homophily.png")
    plot_safety(rows, FIG / "safety_comparison.png")
    plot_reliability_vs_accuracy(rows, FIG / "reliability_vs_accuracy.png")

    table, header = build_main_tables(rows)
    write_csv(TBL / "main_results_prl.csv", header, table)
    write_md(TBL / "main_results_prl.md", header, table)
    print("Wrote figures to", FIG)
    print("Wrote tables to", TBL)


if __name__ == "__main__":
    main()

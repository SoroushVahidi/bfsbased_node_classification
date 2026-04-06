#!/usr/bin/env python3
"""
Figure + table bundle for FINAL_V3 (no experiments — reads existing CSVs only).

Outputs:
  figures/graphical_abstract_selective_correction_v3.png
  figures/graphical_abstract_selective_correction_v3.pdf
  figures/graphical_abstract_selective_correction_v3.svg
  figures/correction_rate_vs_homophily.png
  figures/safety_comparison.png
  figures/reliability_vs_accuracy.png
  tables/main_results_selective_correction.csv
  tables/main_results_selective_correction.md
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

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
    def _maybe_float(value: str | None, default_nan: bool = False) -> float:
        if value in ("", None):
            return float("nan") if default_nan else 0.0
        return float(value)

    rows = []
    with RESULTS.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["split_id"] = int(row["split_id"])
            row["test_acc"] = float(row["test_acc"])
            row["delta_vs_mlp"] = float(row["delta_vs_mlp"])
            row["frac_corrected"] = _maybe_float(row.get("frac_corrected"))
            row["frac_confident"] = _maybe_float(row.get("frac_confident"), default_nan=True)
            row["frac_unreliable"] = _maybe_float(row.get("frac_unreliable"), default_nan=True)
            row["correction_precision"] = _maybe_float(row.get("correction_precision"), default_nan=True)
            row["tau"] = _maybe_float(row.get("tau"), default_nan=True)
            row["rho"] = _maybe_float(row.get("rho"), default_nan=True)
            nh, hh = row.get("n_helped", ""), row.get("n_hurt", "")
            row["n_helped"] = int(nh) if nh not in ("", None) else 0
            row["n_hurt"] = int(hh) if hh not in ("", None) else 0
            rows.append(row)
    return rows


def plot_graphical_abstract_v3(out: Path) -> None:
    """Polished canonical graphical abstract for FINAL_V3."""
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
    c_text = "#1a1a1a"
    c_muted = "#6b6b6b"
    c_edge = "#4a4a4a"
    c_conf = "#5c7a99"
    c_unc = "#c45c26"
    c_accent = "#2d5a87"
    c_panel = "#f5f5f5"
    c_ok = "#4a7c59"

    def circle_layout(n: int, r: float = 0.36, cx: float = 0.5, cy: float = 0.5):
        pts = []
        for i in range(n):
            t = 2 * np.pi * i / n - np.pi / 2
            pts.append((cx + r * np.cos(t), cy + r * np.sin(t)))
        return pts

    def draw_input_panel(ax):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        pos = circle_layout(6)
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3), (2, 5)]
        for i, j in edges:
            ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], color=c_edge, lw=1.2, alpha=0.85)
        for x, y in pos:
            ax.add_patch(Circle((x, y), 0.065, facecolor=c_conf, edgecolor=c_edge, linewidth=1.0, zorder=3))
        ax.add_patch(Rectangle((0.78, 0.08), 0.18, 0.22, facecolor="white", edgecolor=c_muted, lw=0.8))
        for r in range(3):
            for c in range(3):
                ax.add_patch(
                    Rectangle(
                        (0.8 + c * 0.045, 0.12 + r * 0.055),
                        0.035,
                        0.04,
                        facecolor=c_muted,
                        alpha=0.35 + 0.2 * ((r + c) % 2),
                        edgecolor="none",
                    )
                )

    def draw_mlp_panel(ax):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(
            FancyBboxPatch(
                (0.12, 0.2),
                0.76,
                0.55,
                boxstyle="round,pad=0.02,rounding_size=0.03",
                facecolor=c_panel,
                edgecolor=c_accent,
                linewidth=1.2,
            )
        )
        ax.text(0.5, 0.72, "Strong MLP", ha="center", va="center", fontsize=10, weight="bold", color=c_text)
        ax.text(0.5, 0.58, "class scores", ha="center", va="center", fontsize=8.5, color=c_muted)
        bars = [0.85, 0.45, 0.25]
        bx, bw, gap = 0.22, 0.18, 0.12
        for k, h in enumerate(bars):
            x0 = bx + k * (bw + gap)
            ax.add_patch(Rectangle((x0, 0.22), bw, h * 0.35, facecolor=c_conf, edgecolor="none", alpha=0.9))
        ax.text(0.5, 0.12, "confidence", ha="center", va="center", fontsize=7.5, color=c_muted)

    def draw_gate_panel(ax):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.plot([0.1, 0.9], [0.52, 0.52], color=c_accent, lw=1.5, linestyle="--", alpha=0.9)
        ax.text(0.88, 0.56, "τ", fontsize=11, color=c_accent, ha="right")
        ax.text(0.5, 0.88, "Uncertainty gate", ha="center", fontsize=10, weight="bold", color=c_text)
        ax.text(0.5, 0.78, "low confidence only", ha="center", fontsize=7.8, color=c_muted)
        ax.add_patch(Circle((0.28, 0.32), 0.07, facecolor=c_conf, edgecolor=c_edge))
        ax.text(0.28, 0.32, "✓", ha="center", va="center", fontsize=11, color="white")
        ax.add_patch(Circle((0.72, 0.32), 0.07, facecolor=c_unc, edgecolor=c_edge))
        ax.text(0.72, 0.32, "?", ha="center", va="center", fontsize=12, weight="bold", color="white")
        ax.text(0.28, 0.14, "confident", ha="center", fontsize=7.5, color=c_muted)
        ax.text(0.72, 0.14, "uncertain", ha="center", fontsize=7.5, color=c_muted)

    def draw_correction_panel(ax):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.text(0.5, 0.9, "Selective graph correction", ha="center", fontsize=10, weight="bold", color=c_text)
        ax.text(0.5, 0.78, "neighbor support · similarity · compatibility", ha="center", fontsize=7.2, color=c_muted)
        cx, cy = 0.5, 0.38
        neigh = [(0.22, 0.42), (0.38, 0.62), (0.62, 0.62), (0.78, 0.42)]
        for x, y in neigh:
            ax.add_patch(Circle((x, y), 0.055, facecolor=c_conf, edgecolor=c_edge, lw=0.8))
            ax.add_patch(
                FancyArrowPatch(
                    (x + 0.04 * (cx - x), y + 0.04 * (cy - y)),
                    (cx - 0.05 * (cx - x), cy - 0.05 * (cy - y)),
                    arrowstyle="-|>",
                    mutation_scale=10,
                    color=c_edge,
                    lw=0.9,
                    alpha=0.85,
                )
            )
        ax.add_patch(Circle((cx, cy), 0.08, facecolor=c_unc, edgecolor=c_edge, lw=1.2))
        ax.text(cx, cy, "u", ha="center", va="center", fontsize=10, weight="bold", color="white")
        ax.text(0.5, 0.1, "only uncertain nodes", ha="center", fontsize=7.5, color=c_muted)

    def draw_outcome_panel(ax):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.text(0.5, 0.88, "Outcome", ha="center", fontsize=10, weight="bold", color=c_text)
        ax.text(0.5, 0.62, "Gains: Cora · CiteSeer · PubMed", ha="center", fontsize=8.5, color=c_ok, weight="medium")
        ax.text(0.5, 0.42, "Near-neutral on several benchmarks", ha="center", fontsize=8, color=c_muted)
        ax.text(0.5, 0.22, "Graph used selectively — not global propagation", ha="center", fontsize=7.8, color=c_text, style="italic")

    fig = plt.figure(figsize=(14.2, 4.2), dpi=150)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(
        2,
        5,
        height_ratios=[1, 0.22],
        hspace=0.35,
        wspace=0.28,
        left=0.04,
        right=0.96,
        top=0.88,
        bottom=0.12,
    )
    titles = [
        "Node features + graph",
        "Strong MLP baseline",
        "Uncertainty gate",
        "Selective graph correction",
        "Empirical outcome",
    ]
    panel_drawers = [draw_input_panel, draw_mlp_panel, draw_gate_panel, draw_correction_panel, draw_outcome_panel]
    for idx, (title, drawer) in enumerate(zip(titles, panel_drawers)):
        ax = fig.add_subplot(gs[0, idx])
        drawer(ax)
        ax.set_title(title, fontsize=9.5, pad=8, color=c_text, weight="medium")

    axf = fig.add_subplot(gs[1, :])
    axf.axis("off")
    axf.text(
        0.5,
        0.65,
        "Graph information is applied only where the MLP is uncertain.",
        ha="center",
        va="center",
        fontsize=11,
        color=c_text,
        weight="medium",
    )
    axf.text(
        0.5,
        0.2,
        "Selective correction improves some datasets while avoiding unnecessary global graph use.",
        ha="center",
        va="center",
        fontsize=8.5,
        color=c_muted,
    )
    fig.suptitle(
        "FINAL_V3: Reliability-Gated Selective Graph Correction for Node Classification",
        fontsize=11.5,
        weight="bold",
        color=c_text,
        y=0.98,
    )
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.08)
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

    fig, ax = plt.subplots(figsize=(7.4, 5.2), dpi=180)
    ax.errorbar(
        H,
        mean_v3,
        yerr=std_v3,
        fmt="o-",
        color="#1f2937",
        ecolor="#6b7280",
        capsize=4,
        markersize=7.5,
        linewidth=1.6,
        elinewidth=1.1,
        label="FINAL_V3",
        zorder=3,
    )
    ax.errorbar(
        H,
        mean_v1,
        yerr=std_v1,
        fmt="s--",
        color="#6b7280",
        ecolor="#9ca3af",
        capsize=4,
        markersize=6.5,
        linewidth=1.3,
        elinewidth=1.0,
        label="SGC v1",
        zorder=2,
    )
    names = {
        "cora": "Cora",
        "citeseer": "CiteSeer",
        "pubmed": "PubMed",
        "chameleon": "Chameleon",
        "texas": "Texas",
        "wisconsin": "Wisconsin",
    }
    # Hand-tuned offsets for readability (points) with leader lines.
    offsets = {
        "cora": (16, 10),
        "citeseer": (12, 18),
        "pubmed": (18, -2),
        "chameleon": (-30, -8),
        "texas": (-22, 14),
        "wisconsin": (-40, 6),
    }
    for i, d in enumerate(datasets):
        ax.annotate(
            names.get(d, d),
            (H[i], mean_v3[i]),
            textcoords="offset points",
            xytext=offsets.get(d, (8, 8)),
            fontsize=10,
            color="#374151",
            ha="left",
            va="center",
            arrowprops={
                "arrowstyle": "-",
                "lw": 0.8,
                "color": "#9ca3af",
                "shrinkA": 2,
                "shrinkB": 4,
            },
            zorder=4,
        )
    ax.set_xlabel("Edge homophily (H)", fontsize=12, labelpad=8)
    ax.set_ylabel("Fraction of nodes corrected", fontsize=12, labelpad=8)
    ax.set_title("Selective correction rate vs. homophily", fontsize=13, pad=12)
    ax.tick_params(axis="both", labelsize=10.5)
    ax.legend(frameon=False, loc="upper left", fontsize=10.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    y_max = max(max(mean_v3[i] + std_v3[i] for i in range(len(datasets))), max(mean_v1[i] + std_v1[i] for i in range(len(datasets))))
    ax.set_ylim(-0.015, min(1.0, y_max + 0.08))
    x_pad = 0.03
    ax.set_xlim(min(H) - x_pad, max(H) + x_pad)
    ax.grid(axis="y", linestyle=":", linewidth=0.8, color="#d1d5db")
    fig.subplots_adjust(left=0.12, right=0.97, top=0.90, bottom=0.15)
    fig.savefig(out, dpi=400, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def plot_safety(rows: list[dict], out: Path) -> None:
    """Count splits with delta_vs_mlp < 0 per dataset for v1 vs FINAL_V3."""
    by_ds = defaultdict(lambda: {"BASELINE_SGC_V1": 0, "FINAL_V3": 0})
    all_datasets = sorted({r["dataset"] for r in rows if r["method"] in ("BASELINE_SGC_V1", "FINAL_V3")})
    for ds in all_datasets:
        _ = by_ds[ds]
    for r in rows:
        if r["method"] not in ("BASELINE_SGC_V1", "FINAL_V3"):
            continue
        ds = r["dataset"]
        if r["delta_vs_mlp"] < -1e-9:
            by_ds[ds][r["method"]] += 1

    datasets = sorted(by_ds.keys())
    x = np.arange(len(datasets))
    w = 0.38
    c1 = [by_ds[d]["BASELINE_SGC_V1"] for d in datasets]
    c2 = [by_ds[d]["FINAL_V3"] for d in datasets]

    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=180)
    bars1 = ax.bar(
        x - w / 2,
        c1,
        width=w,
        label="SGC v1",
        color="#9ca3af",
        edgecolor="#374151",
        linewidth=1.0,
        zorder=3,
    )
    bars2 = ax.bar(
        x + w / 2,
        c2,
        width=w,
        label="FINAL_V3",
        color="#374151",
        edgecolor="#111827",
        linewidth=1.0,
        zorder=3,
    )
    ax.set_xticks(x)
    display_name = {
        "chameleon": "Chameleon",
        "citeseer": "CiteSeer",
        "cora": "Cora",
        "pubmed": "PubMed",
        "texas": "Texas",
        "wisconsin": "Wisconsin",
    }
    ax.set_xticklabels([display_name.get(d, d.capitalize()) for d in datasets], rotation=20, ha="right", fontsize=10.5)
    ax.set_ylabel("Harmful splits (Δ test < 0)", fontsize=12)
    ax.set_title("Safety comparison vs MLP across canonical datasets", fontsize=13, pad=10)
    ax.tick_params(axis="y", labelsize=10.5)
    ax.legend(frameon=False, fontsize=10.5, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ymax = max(1, max(c1 + c2))
    ax.set_ylim(0, ymax + 0.9)
    ax.set_yticks(np.arange(0, ymax + 1, 1))
    ax.grid(axis="y", linestyle=":", linewidth=0.8, color="#d1d5db", zorder=0)

    # Explicitly annotate every bar, including zeros.
    for bars in (bars1, bars2):
        for b in bars:
            h = b.get_height()
            xmid = b.get_x() + b.get_width() / 2
            ytxt = h + 0.05 if h > 0 else 0.08
            ax.text(xmid, ytxt, f"{int(h)}", ha="center", va="bottom", fontsize=9.5, color="#111827")

    fig.subplots_adjust(left=0.11, right=0.98, top=0.88, bottom=0.22)
    fig.savefig(out, dpi=400, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def plot_reliability_vs_accuracy(rows: list[dict], out: Path) -> None:
    """Per-split scatter: unreliable fraction vs correction precision (FINAL_V3 only).

    **Data source (canonical):** ``reports/final_method_v3_results.csv`` only — rows with
    ``method == FINAL_V3``. No mixing with ``logs/final_v3_regime_analysis_per_split.csv``:
    that file can differ from the frozen log for the same (dataset, split) (see
    cross-checks in the repo).

    **Completeness:** the frozen CSV has `frac_unreliable` populated for **30 of 60** splits;
    the remaining 30 rows have an empty `frac_unreliable` field, so they cannot be plotted
    on the x-axis without extrapolating. The figure annotates this; for a full 60-split
    panel, the canonical log would need to be regenerated with complete branch fields.

    Definitions (match ``code/bfsbased_node_classification/final_method_v3.py`` logging):
    - **x** — ``frac_unreliable`` = fraction of **test** nodes that are uncertain *and*
      unreliable (``margin < tau`` and ``R(v) < rho``), i.e. keep-MLP branch
      ``uncertain_unreliable_keep_mlp`` on the test mask.
    - **y** — ``correction_precision`` = ``n_helped / max(n_helped + n_hurt, 1)`` among
      nodes whose prediction **changed** vs MLP on test; if no change, implementation logs
      precision ``1.0`` (vacuous).
    - **Point size** — proportional to ``frac_corrected`` (fraction of test nodes in the
      correction mask), with a fixed scale ``s = 45 + 220 * frac_corrected`` (clamped) so
      comparable across splits.

    **Edge cases:** splits with ``n_helped + n_hurt == 0`` are plotted with open markers
    (precision may be vacuously 1.0 in the CSV). They are **not** dropped (the previous
    version omitted 9 such splits).

    **Trend:** no OLS line — the cloud is heterogeneous across datasets; Pearson *r* is
    shown as a small annotation only (exploratory, not a causal claim).
    """
    palette = {
        "cora": "#1f77b4",
        "citeseer": "#ff7f0e",
        "pubmed": "#2ca02c",
        "chameleon": "#d62728",
        "texas": "#9467bd",
        "wisconsin": "#8c564b",
    }
    display_name = {
        "cora": "Cora",
        "citeseer": "CiteSeer",
        "pubmed": "PubMed",
        "chameleon": "Chameleon",
        "texas": "Texas",
        "wisconsin": "Wisconsin",
    }

    n_final = sum(1 for r in rows if r["method"] == "FINAL_V3")
    pts: list[tuple[float, float, float, str, bool]] = []
    for r in rows:
        if r["method"] != "FINAL_V3":
            continue
        if not np.isfinite(r["frac_unreliable"]) or not np.isfinite(r["correction_precision"]):
            continue
        fu = float(r["frac_unreliable"])
        cp = float(r["correction_precision"])
        fc = float(r["frac_corrected"])
        nh = int(r["n_helped"])
        nj = int(r["n_hurt"])
        vacuous = nh + nj == 0
        pts.append((fu, cp, fc, r["dataset"], vacuous))

    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=180)
    for ds in sorted({p[3] for p in pts}):
        sub = [p for p in pts if p[3] == ds]
        xs = [p[0] for p in sub]
        ys = [p[1] for p in sub]
        sizes = [min(240, max(45, 45 + 220 * p[2])) for p in sub]
        vac = [p[4] for p in sub]
        # Filled markers for splits with help/hurt; open for vacuous precision
        for x, y, s, v in zip(xs, ys, sizes, vac):
            if v:
                ax.scatter(
                    [x],
                    [y],
                    s=s,
                    facecolors="none",
                    edgecolors=palette.get(ds, "#333333"),
                    linewidths=1.6,
                    alpha=0.95,
                    zorder=3,
                )
            else:
                ax.scatter(
                    [x],
                    [y],
                    s=s,
                    c=palette.get(ds, "#333333"),
                    edgecolors="white",
                    linewidths=0.6,
                    alpha=0.88,
                    zorder=3,
                )
        ax.scatter([], [], c=palette.get(ds, "#333333"), s=55, edgecolors="white", linewidths=0.6, label=display_name.get(ds, ds))

    if any(p[4] for p in pts):
        ax.scatter(
            [],
            [],
            s=55,
            facecolors="none",
            edgecolors="#333333",
            linewidths=1.6,
            label="vacuous precision (n_helped+n_hurt=0)",
        )

    if len(pts) >= 2:
        xa = np.array([p[0] for p in pts], dtype=float)
        ya = np.array([p[1] for p in pts], dtype=float)
        if np.std(xa) > 1e-12 and np.std(ya) > 1e-12:
            r_pear = float(np.corrcoef(xa, ya)[0, 1])
            ax.text(
                0.02,
                0.98,
                f"Pearson r = {r_pear:+.3f} (n={len(pts)} splits with x,y in log)",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9.5,
                color="#374151",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#f9fafb", edgecolor="#e5e7eb"),
            )
    ax.text(
        0.02,
        0.06,
        f"Note: frozen CSV has `frac_unreliable` for {len(pts)}/{n_final} splits;\n"
        "remaining splits omit this field (not plotted).",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#4b5563",
        linespacing=1.25,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fffbeb", edgecolor="#fcd34d"),
    )

    ax.set_xlabel("Fraction of test nodes uncertain and unreliable (low R)", fontsize=11.5, labelpad=8)
    ax.set_ylabel("Correction precision (n_helped / (n_helped + n_hurt))", fontsize=11.5, labelpad=8)
    ax.set_title("Reliability signal vs. correction precision (FINAL_V3, per split)", fontsize=12.5, pad=10)
    ax.tick_params(axis="both", labelsize=10.5)
    ax.set_xlim(-0.02, max(0.55, max(p[0] for p in pts) + 0.05) if pts else 1.0)
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=9, frameon=False, loc="lower right", ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="both", linestyle=":", linewidth=0.75, color="#d1d5db", alpha=0.9, zorder=0)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.92, bottom=0.20)
    fig.savefig(out, dpi=400, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.1)
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
        "Source: `reports/final_method_v3_results.csv` — **10 GEO-GCN splits** (indices 0–9) per dataset, same protocol as the FINAL_V3 evaluation logs.",
        "",
        "Regenerate this table and the figure set:",
        "`python3 scripts/build_artifacts/build_v3_figures_and_tables.py`",
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

    plot_graphical_abstract_v3(FIG / "graphical_abstract_selective_correction_v3.png")
    plot_correction_vs_homophily(rows, homophily, FIG / "correction_rate_vs_homophily.png")
    plot_safety(rows, FIG / "safety_comparison.png")
    plot_reliability_vs_accuracy(rows, FIG / "reliability_vs_accuracy.png")

    table, header = build_main_tables(rows)
    write_csv(TBL / "main_results_selective_correction.csv", header, table)
    write_md(TBL / "main_results_selective_correction.md", header, table)
    print("Wrote figures to", FIG)
    print("Wrote tables to", TBL)


if __name__ == "__main__":
    main()

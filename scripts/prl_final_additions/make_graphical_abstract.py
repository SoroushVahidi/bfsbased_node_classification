#!/usr/bin/env python3
"""
Submission graphical abstract — PRL manuscript (Elsevier-style, landscape).

Uncertainty-Gated Selective Graph Correction for Node Classification

Outputs PDF (vector), PNG (raster), SVG (editable) under figures/prl_final_additions/.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

mpl.use("Agg")
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
        "font.size": 9,
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
    }
)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "figures" / "prl_final_additions"
STEM = "graphical_abstract_prl"

# Minimal palette — conservative, print-safe
C_TEXT = "#1a1a1a"
C_MUTED = "#6b6b6b"
C_EDGE = "#4a4a4a"
C_CONF = "#5c7a99"
C_UNC = "#c45c26"
C_ACCENT = "#2d5a87"
C_PANEL = "#f5f5f5"
C_OK = "#4a7c59"


def circle_layout(n: int, r: float = 0.38, cx: float = 0.5, cy: float = 0.5):
    pts = []
    for i in range(n):
        t = 2 * math.pi * i / n - math.pi / 2
        pts.append((cx + r * math.cos(t), cy + r * math.sin(t)))
    return pts


def draw_small_graph(ax, uncertain_idx: set[int], highlight_edges: bool = False) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    n = 6
    pos = circle_layout(n, r=0.36)
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3), (2, 5)]
    for i, j in edges:
        ax.plot(
            [pos[i][0], pos[j][0]],
            [pos[i][1], pos[j][1]],
            color=C_EDGE,
            lw=1.2,
            alpha=0.85,
            zorder=1,
        )
    for i, (x, y) in enumerate(pos):
        if i in uncertain_idx:
            fc, ec = C_UNC, C_UNC
            lw = 1.4
        else:
            fc = C_CONF
            ec = C_EDGE
            lw = 1.0
        circ = Circle(
            (x, y),
            0.065,
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            zorder=3,
        )
        ax.add_patch(circ)
    # feature glyph (small matrix)
    ax.add_patch(
        Rectangle((0.78, 0.08), 0.18, 0.22, facecolor="white", edgecolor=C_MUTED, lw=0.8)
    )
    for r in range(3):
        for c in range(3):
            ax.add_patch(
                Rectangle(
                    (0.8 + c * 0.045, 0.12 + r * 0.055),
                    0.035,
                    0.04,
                    facecolor=C_MUTED,
                    alpha=0.35 + 0.2 * ((r + c) % 2),
                )
            )


def panel_mlp(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.12, 0.2),
            0.76,
            0.55,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            facecolor=C_PANEL,
            edgecolor=C_ACCENT,
            linewidth=1.2,
        )
    )
    ax.text(0.5, 0.72, "Strong MLP", ha="center", va="center", fontsize=10, weight="bold", color=C_TEXT)
    ax.text(0.5, 0.58, "class scores", ha="center", va="center", fontsize=8.5, color=C_MUTED)
    # pseudo confidence bars
    bars = [0.85, 0.45, 0.25]
    bx, bw, gap = 0.22, 0.18, 0.12
    for k, h in enumerate(bars):
        x0 = bx + k * (bw + gap)
        ax.add_patch(
            Rectangle(
                (x0, 0.22),
                bw,
                h * 0.35,
                facecolor=C_CONF,
                edgecolor="none",
                alpha=0.9,
            )
        )
    ax.text(0.5, 0.12, "confidence", ha="center", va="center", fontsize=7.5, color=C_MUTED)


def panel_gate(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # threshold line
    ax.plot([0.1, 0.9], [0.52, 0.52], color=C_ACCENT, lw=1.5, linestyle="--", alpha=0.9)
    ax.text(0.88, 0.56, "τ", fontsize=11, color=C_ACCENT, ha="right")
    ax.text(0.5, 0.88, "Uncertainty gate", ha="center", fontsize=10, weight="bold", color=C_TEXT)
    ax.text(0.5, 0.78, "low confidence only", ha="center", fontsize=7.8, color=C_MUTED)
    # icons: check vs question
    ax.add_patch(Circle((0.28, 0.32), 0.07, facecolor=C_CONF, edgecolor=C_EDGE))
    ax.text(0.28, 0.32, "✓", ha="center", va="center", fontsize=11, color="white")
    ax.add_patch(Circle((0.72, 0.32), 0.07, facecolor=C_UNC, edgecolor=C_EDGE))
    ax.text(0.72, 0.32, "?", ha="center", va="center", fontsize=12, weight="bold", color="white")
    ax.text(0.28, 0.14, "confident", ha="center", fontsize=7.5, color=C_MUTED)
    ax.text(0.72, 0.14, "uncertain", ha="center", fontsize=7.5, color=C_MUTED)


def panel_correction(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.5,
        0.9,
        "Selective graph correction",
        ha="center",
        fontsize=10,
        weight="bold",
        color=C_TEXT,
    )
    ax.text(
        0.5,
        0.78,
        "neighbor support · similarity · compatibility",
        ha="center",
        fontsize=7.2,
        color=C_MUTED,
    )
    # tiny subgraph: center uncertain, neighbors feed in
    cx, cy = 0.5, 0.38
    neigh = [(0.22, 0.42), (0.38, 0.62), (0.62, 0.62), (0.78, 0.42)]
    for x, y in neigh:
        ax.add_patch(Circle((x, y), 0.055, facecolor=C_CONF, edgecolor=C_EDGE, lw=0.8))
        arr = FancyArrowPatch(
            (x + 0.04 * (cx - x), y + 0.04 * (cy - y)),
            (cx - 0.05 * (cx - x), cy - 0.05 * (cy - y)),
            arrowstyle="-|>",
            mutation_scale=10,
            color=C_EDGE,
            lw=0.9,
            alpha=0.85,
        )
        ax.add_patch(arr)
    ax.add_patch(Circle((cx, cy), 0.08, facecolor=C_UNC, edgecolor=C_EDGE, lw=1.2))
    ax.text(cx, cy, "u", ha="center", va="center", fontsize=10, weight="bold", color="white")
    ax.text(0.5, 0.1, "only uncertain nodes", ha="center", fontsize=7.5, color=C_MUTED)


def panel_outcome(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.88, "Outcome", ha="center", fontsize=10, weight="bold", color=C_TEXT)
    ax.text(
        0.5,
        0.62,
        "Gains: Cora · CiteSeer · PubMed",
        ha="center",
        fontsize=8.5,
        color=C_OK,
        weight="medium",
    )
    ax.text(
        0.5,
        0.42,
        "Near-neutral on several benchmarks",
        ha="center",
        fontsize=8,
        color=C_MUTED,
    )
    ax.text(
        0.5,
        0.22,
        "Graph used selectively — not global propagation",
        ha="center",
        fontsize=7.8,
        color=C_TEXT,
        style="italic",
    )


def main() -> None:
    # Landscape 5-panel + footer band
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

    ax1 = fig.add_subplot(gs[0, 0])
    draw_small_graph(ax1, uncertain_idx=set())
    ax1.set_title(titles[0], fontsize=9.5, pad=8, color=C_TEXT, weight="medium")

    ax2 = fig.add_subplot(gs[0, 1])
    panel_mlp(ax2)
    ax2.set_title(titles[1], fontsize=9.5, pad=8, color=C_TEXT, weight="medium")

    ax3 = fig.add_subplot(gs[0, 2])
    panel_gate(ax3)
    ax3.set_title(titles[2], fontsize=9.5, pad=8, color=C_TEXT, weight="medium")

    ax4 = fig.add_subplot(gs[0, 3])
    panel_correction(ax4)
    ax4.set_title(titles[3], fontsize=9.5, pad=8, color=C_TEXT, weight="medium")

    ax5 = fig.add_subplot(gs[0, 4])
    panel_outcome(ax5)
    ax5.set_title(titles[4], fontsize=9.5, pad=8, color=C_TEXT, weight="medium")

    # Footer spanning all columns
    axf = fig.add_subplot(gs[1, :])
    axf.axis("off")
    axf.text(
        0.5,
        0.65,
        "Graph information is applied only where the MLP is uncertain.",
        ha="center",
        va="center",
        fontsize=11,
        color=C_TEXT,
        weight="medium",
    )
    axf.text(
        0.5,
        0.2,
        "Selective correction improves some datasets while avoiding unnecessary global graph use.",
        ha="center",
        va="center",
        fontsize=8.5,
        color=C_MUTED,
    )

    fig.suptitle(
        "Uncertainty-Gated Selective Graph Correction for Node Classification",
        fontsize=11.5,
        weight="bold",
        color=C_TEXT,
        y=0.98,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / f"{STEM}.pdf"
    png = OUT_DIR / f"{STEM}.png"
    svg = OUT_DIR / f"{STEM}.svg"

    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    print("Wrote", pdf)
    print("Wrote", png)
    print("Wrote", svg)


if __name__ == "__main__":
    main()

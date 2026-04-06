#!/usr/bin/env python3
"""Build manuscript-support artifacts for canonical FINAL_V3 evidence.

This script ONLY reads canonical evidence inputs and writes new manuscript-support
tables/figures under `tables/` and `figures/` without overwriting frozen canonical
artifacts.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"

CANONICAL_INPUTS = [
    ROOT / "reports" / "final_method_v3_results.csv",
    ROOT / "logs" / "final_v3_regime_analysis_per_dataset.csv",
    ROOT / "logs" / "final_v3_regime_analysis_per_split.csv",
    ROOT / "logs" / "gcn_runs_gcn_table1.jsonl",
    ROOT / "logs" / "manuscript_regime_analysis_final_validation_main.csv",
    ROOT / "logs" / "regime_analysis_manuscript_final_validation_job2.csv",
    ROOT / "reports" / "final_method_v3_analysis.md",
    ROOT / "reports" / "safety_analysis.md",
    ROOT / "reports" / "final_v3_regime_analysis.md",
]


def _require_inputs() -> None:
    missing = [str(p) for p in CANONICAL_INPUTS if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing canonical inputs:\n- " + "\n- ".join(missing))


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _read_final_results() -> list[dict[str, str]]:
    path = ROOT / "reports" / "final_method_v3_results.csv"
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _group_by_dataset(rows: list[dict[str, str]], method: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        if r.get("method") == method:
            grouped[r["dataset"]].append(r)
    for ds in grouped:
        grouped[ds] = sorted(grouped[ds], key=lambda x: int(x["split_id"]))
    return dict(grouped)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _md_table(fieldnames: list[str], rows: list[dict[str, object]]) -> str:
    out = []
    out.append("| " + " | ".join(fieldnames) + " |")
    out.append("|" + "|".join(["---"] * len(fieldnames)) + "|")
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in fieldnames) + " |")
    return "\n".join(out) + "\n"


def _write_md(path: Path, title: str, note: str, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = [f"# {title}", "", note.strip(), "", _md_table(fieldnames, rows)]
    path.write_text("\n".join(body), encoding="utf-8")


def _f(x: float | None, nd: int = 6) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return f"{x:.{nd}f}"


def build_correction_behavior(rows: list[dict[str, str]]) -> None:
    final_by_ds = _group_by_dataset(rows, "FINAL_V3")
    mlp_by_ds = _group_by_dataset(rows, "MLP_ONLY")
    out_rows: list[dict[str, object]] = []
    for ds in sorted(final_by_ds):
        frows = final_by_ds[ds]
        mrows = mlp_by_ds.get(ds, [])
        mlp_acc = [_to_float(r["test_acc"]) for r in mrows]
        final_acc = [_to_float(r["test_acc"]) for r in frows]
        delta = [_to_float(r["delta_vs_mlp"]) for r in frows]
        corrected = [_to_float(r["frac_corrected"]) for r in frows]
        precision = [_to_float(r["correction_precision"]) for r in frows]
        helped = [int(float(r["n_helped"])) for r in frows if _to_float(r["n_helped"]) is not None]
        hurt = [int(float(r["n_hurt"])) for r in frows if _to_float(r["n_hurt"]) is not None]
        wins = sum(1 for d in delta if d is not None and d > 0)
        ties = sum(1 for d in delta if d is not None and abs(d) < 1e-12)
        losses = sum(1 for d in delta if d is not None and d < 0)
        out_rows.append(
            {
                "dataset": ds,
                "mlp_mean_acc": _f(mean([x for x in mlp_acc if x is not None])),
                "final_v3_mean_acc": _f(mean([x for x in final_acc if x is not None])),
                "delta_vs_mlp": _f(mean([x for x in delta if x is not None])),
                "corrected_frac_mean": _f(mean([x for x in corrected if x is not None])),
                "correction_precision_mean": _f(mean([x for x in precision if x is not None])),
                "helped_total": sum(helped),
                "hurt_total": sum(hurt),
                "wins": wins,
                "ties": ties,
                "losses": losses,
            }
        )

    cols = [
        "dataset",
        "mlp_mean_acc",
        "final_v3_mean_acc",
        "delta_vs_mlp",
        "corrected_frac_mean",
        "correction_precision_mean",
        "helped_total",
        "hurt_total",
        "wins",
        "ties",
        "losses",
    ]
    _write_csv(TABLES / "manuscript_correction_behavior_final_v3.csv", cols, out_rows)
    _write_md(
        TABLES / "manuscript_correction_behavior_final_v3.md",
        "Manuscript correction behavior (FINAL_V3)",
        "Computed from `reports/final_method_v3_results.csv` (canonical FINAL_V3 + MLP rows).",
        cols,
        out_rows,
    )


def build_threshold_summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    final_by_ds = _group_by_dataset(rows, "FINAL_V3")
    out_rows: list[dict[str, object]] = []
    for ds in sorted(final_by_ds):
        frows = final_by_ds[ds]
        tau = [x for x in (_to_float(r["tau"]) for r in frows) if x is not None]
        rho = [x for x in (_to_float(r["rho"]) for r in frows) if x is not None]
        uncertain = [
            x
            for x in ((1.0 - _to_float(r["frac_confident"])) if _to_float(r["frac_confident"]) is not None else None for r in frows)
            if x is not None
        ]
        corrected = [x for x in (_to_float(r["frac_corrected"]) for r in frows) if x is not None]
        delta = [x for x in (_to_float(r["delta_vs_mlp"]) for r in frows) if x is not None]
        out_rows.append(
            {
                "dataset": ds,
                "mean_tau": _f(mean(tau)),
                "std_tau": _f(pstdev(tau) if len(tau) > 1 else 0.0),
                "mean_rho": _f(mean(rho)),
                "std_rho": _f(pstdev(rho) if len(rho) > 1 else 0.0),
                "mean_uncertain_frac": _f(mean(uncertain)),
                "mean_corrected_frac": _f(mean(corrected)),
                "mean_delta_vs_mlp": _f(mean(delta)),
            }
        )

    cols = [
        "dataset",
        "mean_tau",
        "std_tau",
        "mean_rho",
        "std_rho",
        "mean_uncertain_frac",
        "mean_corrected_frac",
        "mean_delta_vs_mlp",
    ]
    _write_csv(TABLES / "manuscript_threshold_summary_final_v3.csv", cols, out_rows)
    _write_md(
        TABLES / "manuscript_threshold_summary_final_v3.md",
        "Manuscript threshold/coverage summary (FINAL_V3)",
        "Uncertain fraction is computed as `1 - frac_confident`; all values come from canonical FINAL_V3 rows.",
        cols,
        out_rows,
    )
    return out_rows


def build_split_examples(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    targets = ["cora", "citeseer", "pubmed", "chameleon"]
    final_by_ds = _group_by_dataset(rows, "FINAL_V3")
    mlp_by_ds_split = {
        (r["dataset"], int(r["split_id"])): r for r in rows if r.get("method") == "MLP_ONLY"
    }
    out_rows: list[dict[str, object]] = []
    for ds in targets:
        if ds not in final_by_ds:
            continue
        frows = final_by_ds[ds]
        deltas = [_to_float(r["delta_vs_mlp"]) for r in frows]
        dmean = mean([d for d in deltas if d is not None])
        candidates = [r for r in frows if _to_float(r["tau"]) is not None and _to_float(r["rho"]) is not None]
        pick_from = candidates if candidates else frows
        best = min(
            pick_from,
            key=lambda r: abs((_to_float(r["delta_vs_mlp"]) or 0.0) - dmean),
        )
        split_id = int(best["split_id"])
        mlp_row = mlp_by_ds_split[(ds, split_id)]
        uncertain_frac = None
        frac_conf = _to_float(best["frac_confident"])
        if frac_conf is not None:
            uncertain_frac = 1.0 - frac_conf
        n_helped = int(float(best["n_helped"])) if _to_float(best["n_helped"]) is not None else 0
        n_hurt = int(float(best["n_hurt"])) if _to_float(best["n_hurt"]) is not None else 0
        out_rows.append(
            {
                "dataset": ds,
                "split_id": split_id,
                "mlp_acc": _f(_to_float(mlp_row["test_acc"])),
                "final_v3_acc": _f(_to_float(best["test_acc"])),
                "delta_vs_mlp": _f(_to_float(best["delta_vs_mlp"])),
                "uncertain_frac": _f(uncertain_frac),
                "corrected_frac": _f(_to_float(best["frac_corrected"])),
                "n_changed": n_helped + n_hurt,
                "n_helped": n_helped,
                "n_hurt": n_hurt,
                "selected_tau": _f(_to_float(best["tau"])),
                "selected_rho": _f(_to_float(best["rho"])),
            }
        )

    cols = [
        "dataset",
        "split_id",
        "mlp_acc",
        "final_v3_acc",
        "delta_vs_mlp",
        "uncertain_frac",
        "corrected_frac",
        "n_changed",
        "n_helped",
        "n_hurt",
        "selected_tau",
        "selected_rho",
    ]
    _write_csv(TABLES / "manuscript_split_examples_final_v3.csv", cols, out_rows)
    _write_md(
        TABLES / "manuscript_split_examples_final_v3.md",
        "Representative split examples (FINAL_V3)",
        "Representative split per dataset is chosen as the split whose `delta_vs_mlp` is closest to that dataset's FINAL_V3 mean delta.",
        cols,
        out_rows,
    )
    return out_rows


def build_threshold_behavior_figure(rows: list[dict[str, str]]) -> None:
    frows = [r for r in rows if r.get("method") == "FINAL_V3"]
    points: list[dict[str, object]] = []
    for r in frows:
        tau = _to_float(r["tau"])
        frac_conf = _to_float(r["frac_confident"])
        delta = _to_float(r["delta_vs_mlp"])
        if tau is None or frac_conf is None or delta is None:
            continue
        points.append(
            {
                "dataset": r["dataset"],
                "split_id": int(r["split_id"]),
                "tau": tau,
                "uncertain_frac": 1.0 - frac_conf,
                "delta_vs_mlp": delta,
            }
        )

    cols = ["dataset", "split_id", "tau", "uncertain_frac", "delta_vs_mlp"]
    _write_csv(TABLES / "manuscript_threshold_behavior_points_final_v3.csv", cols, points)

    palette = {
        "cora": "#1f77b4",
        "citeseer": "#ff7f0e",
        "pubmed": "#2ca02c",
        "chameleon": "#d62728",
        "texas": "#9467bd",
        "wisconsin": "#8c564b",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=170, constrained_layout=True)
    for ds in sorted({p["dataset"] for p in points}):
        ds_points = [p for p in points if p["dataset"] == ds]
        x = [p["tau"] for p in ds_points]
        y1 = [p["uncertain_frac"] for p in ds_points]
        y2 = [p["delta_vs_mlp"] for p in ds_points]
        c = palette.get(ds, "#333333")
        axes[0].scatter(x, y1, s=24, alpha=0.85, color=c, label=ds)
        axes[1].scatter(x, y2, s=24, alpha=0.85, color=c, label=ds)

    axes[0].set_title("Selected tau vs uncertain fraction")
    axes[0].set_xlabel("selected tau")
    axes[0].set_ylabel("uncertain fraction")
    axes[0].grid(alpha=0.25)
    axes[1].set_title("Selected tau vs delta vs MLP")
    axes[1].set_xlabel("selected tau")
    axes[1].set_ylabel("delta_vs_mlp")
    axes[1].axhline(0.0, color="#666666", lw=1.0, alpha=0.7)
    axes[1].grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.05), frameon=False)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "manuscript_threshold_behavior_final_v3.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_bounded_intervention_figure(rows: list[dict[str, str]]) -> None:
    frows = [r for r in rows if r.get("method") == "FINAL_V3"]
    split_points: list[dict[str, object]] = []
    for r in frows:
        corrected = _to_float(r["frac_corrected"])
        precision = _to_float(r["correction_precision"])
        n_helped = _to_float(r["n_helped"])
        n_hurt = _to_float(r["n_hurt"])
        if corrected is None or precision is None:
            continue
        harm_rate = None
        if n_helped is not None and n_hurt is not None and (n_helped + n_hurt) > 0:
            harm_rate = n_hurt / (n_helped + n_hurt)
        split_points.append(
            {
                "dataset": r["dataset"],
                "split_id": int(r["split_id"]),
                "corrected_frac": corrected,
                "correction_precision": precision,
                "harm_rate": harm_rate if harm_rate is not None else "",
            }
        )
    _write_csv(
        TABLES / "manuscript_bounded_intervention_points_final_v3.csv",
        ["dataset", "split_id", "corrected_frac", "correction_precision", "harm_rate"],
        split_points,
    )

    palette = {
        "cora": "#1f77b4",
        "citeseer": "#ff7f0e",
        "pubmed": "#2ca02c",
        "chameleon": "#d62728",
        "texas": "#9467bd",
        "wisconsin": "#8c564b",
    }
    fig, ax = plt.subplots(figsize=(6.6, 4.6), dpi=170, constrained_layout=True)
    for ds in sorted({p["dataset"] for p in split_points}):
        ds_pts = [p for p in split_points if p["dataset"] == ds]
        x = [float(p["corrected_frac"]) for p in ds_pts]
        y = [float(p["correction_precision"]) for p in ds_pts]
        c = palette.get(ds, "#444444")
        ax.scatter(x, y, s=18, alpha=0.22, color=c)
        mx = mean(x)
        my = mean(y)
        ax.scatter([mx], [my], s=95, marker="D", color=c, edgecolor="white", linewidth=0.8, label=ds)

    ax.set_title("Bounded intervention profile (FINAL_V3)")
    ax.set_xlabel("corrected fraction")
    ax.set_ylabel("correction precision")
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(title="dataset", ncol=2, frameon=False)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "manuscript_bounded_intervention_final_v3.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_runtime_table(rows: list[dict[str, str]]) -> None:
    by_method_ds: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        method = r.get("method", "")
        if method not in {"FINAL_V3", "BASELINE_SGC_V1", "V2_MULTIBRANCH"}:
            continue
        rt = _to_float(r.get("runtime_sec"))
        if rt is not None and rt > 0:
            by_method_ds[(r["dataset"], method)].append(rt)

    datasets = sorted({r["dataset"] for r in rows if r.get("method") == "FINAL_V3"})
    out_rows: list[dict[str, object]] = []
    for ds in datasets:
        final_vals = by_method_ds.get((ds, "FINAL_V3"), [])
        v1_vals = by_method_ds.get((ds, "BASELINE_SGC_V1"), [])
        v2_vals = by_method_ds.get((ds, "V2_MULTIBRANCH"), [])
        out_rows.append(
            {
                "dataset": ds,
                "mean_runtime_sec_final_v3": _f(mean(final_vals) if final_vals else None),
                "std_runtime_sec_final_v3": _f(pstdev(final_vals) if len(final_vals) > 1 else (0.0 if final_vals else None)),
                "mean_runtime_sec_v1": _f(mean(v1_vals) if v1_vals else None),
                "mean_runtime_sec_v2": _f(mean(v2_vals) if v2_vals else None),
            }
        )

    cols = [
        "dataset",
        "mean_runtime_sec_final_v3",
        "std_runtime_sec_final_v3",
        "mean_runtime_sec_v1",
        "mean_runtime_sec_v2",
    ]
    _write_csv(TABLES / "manuscript_runtime_final_v3.csv", cols, out_rows)
    note = (
        "Runtime fields are taken from canonical `reports/final_method_v3_results.csv` only. "
        "If v1/v2 runtime fields are blank in canonical rows, those cells are left empty."
    )
    _write_md(TABLES / "manuscript_runtime_final_v3.md", "Runtime/efficiency summary (FINAL_V3)", note, cols, out_rows)


def build_journal_graphical_abstract() -> None:
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13.6, 4.8), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    c_box = "#f7f9fc"
    c_edge = "#4a5a70"
    c_text = "#1b2430"
    c_unc = "#d16a1f"
    c_rel = "#2c7fb8"
    c_ok = "#2f7d4f"
    c_neu = "#5e6773"

    steps = [
        (0.03, 0.2, 0.14, 0.6, "1) MLP\nprediction"),
        (0.20, 0.2, 0.14, 0.6, "2) Uncertainty\ngate"),
        (0.37, 0.2, 0.14, 0.6, "3) Reliability\ngate"),
        (0.54, 0.2, 0.14, 0.6, "4) Selective\nscore S(v,c)"),
        (0.71, 0.2, 0.14, 0.6, "5) Correct only\nuncertain + reliable"),
    ]
    for x, y, w, h, label in steps:
        ax.add_patch(
            FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02", facecolor=c_box, edgecolor=c_edge, linewidth=1.2)
        )
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11, color=c_text)

    for i in range(len(steps) - 1):
        x1 = steps[i][0] + steps[i][2]
        x2 = steps[i + 1][0]
        y = 0.5
        ax.add_patch(
            FancyArrowPatch((x1 + 0.01, y), (x2 - 0.01, y), arrowstyle="-|>", mutation_scale=16, linewidth=1.4, color=c_edge)
        )

    ax.text(0.27, 0.14, "uncertain if confidence < tau", ha="center", va="center", fontsize=9.5, color=c_unc)
    ax.text(0.44, 0.14, "reliable if R(v) >= rho", ha="center", va="center", fontsize=9.5, color=c_rel)
    ax.text(
        0.90,
        0.64,
        "6) Helps in graph-helpful regimes",
        ha="center",
        va="center",
        fontsize=11,
        color=c_ok,
        weight="bold",
    )
    ax.text(
        0.90,
        0.43,
        "Conservative elsewhere",
        ha="center",
        va="center",
        fontsize=10.5,
        color=c_neu,
    )
    ax.text(
        0.50,
        0.92,
        "FINAL_V3: reliability-gated selective graph correction",
        ha="center",
        va="center",
        fontsize=13,
        color=c_text,
        weight="bold",
    )

    out = FIGURES / "graphical_abstract_final_v3_journal.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def audit_existing_targets() -> dict[str, bool]:
    checks = {
        "correction behavior table": (TABLES / "manuscript_correction_behavior_final_v3.md").exists(),
        "threshold/coverage summary table": (TABLES / "manuscript_threshold_summary_final_v3.md").exists(),
        "threshold behavior figure": (FIGURES / "manuscript_threshold_behavior_final_v3.png").exists(),
        "illustrative split examples table": (TABLES / "manuscript_split_examples_final_v3.md").exists(),
        "runtime/efficiency table": (TABLES / "manuscript_runtime_final_v3.md").exists(),
        "bounded-intervention figure": (FIGURES / "manuscript_bounded_intervention_final_v3.png").exists(),
        "graphical abstract (journal)": (FIGURES / "graphical_abstract_final_v3_journal.png").exists(),
    }
    return checks


def main() -> None:
    _require_inputs()
    pre = audit_existing_targets()
    rows = _read_final_results()
    build_correction_behavior(rows)
    build_threshold_summary(rows)
    build_split_examples(rows)
    build_threshold_behavior_figure(rows)
    build_bounded_intervention_figure(rows)
    build_runtime_table(rows)
    build_journal_graphical_abstract()
    post = audit_existing_targets()

    print("Artifact audit (before -> after):")
    for key in pre:
        print(f"- {key}: {pre[key]} -> {post[key]}")

    print("\nGenerated artifacts:")
    generated = [
        TABLES / "manuscript_correction_behavior_final_v3.csv",
        TABLES / "manuscript_correction_behavior_final_v3.md",
        TABLES / "manuscript_threshold_summary_final_v3.csv",
        TABLES / "manuscript_threshold_summary_final_v3.md",
        TABLES / "manuscript_split_examples_final_v3.csv",
        TABLES / "manuscript_split_examples_final_v3.md",
        TABLES / "manuscript_threshold_behavior_points_final_v3.csv",
        FIGURES / "manuscript_threshold_behavior_final_v3.png",
        TABLES / "manuscript_bounded_intervention_points_final_v3.csv",
        FIGURES / "manuscript_bounded_intervention_final_v3.png",
        TABLES / "manuscript_runtime_final_v3.csv",
        TABLES / "manuscript_runtime_final_v3.md",
        FIGURES / "graphical_abstract_final_v3_journal.png",
    ]
    for p in generated:
        print(f"- {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

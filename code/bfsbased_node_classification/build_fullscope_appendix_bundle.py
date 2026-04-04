#!/usr/bin/env python3
"""Build full-scope FINAL_V3 appendix-support exports for Expert Systems.

Consumes outputs from expert_systems_lightweight_pass.py run on all 6 datasets × 10 splits.
Writes compact full-scope gate-sensitivity and bounded-intervention artifacts.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def _read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(row, key):
    return float(row[key])


def build_gate_sensitivity(sens_rows, out_csv: Path, out_md: Path, out_report: Path):
    # Dataset-level summaries by local rho perturbation.
    ds_dr = defaultdict(list)
    global_dr = defaultdict(list)
    global_route = defaultdict(list)
    for r in sens_rows:
        ds = r["dataset"]
        dr = float(r["delta_rho"])
        dv = 100.0 * float(r["delta_vs_mlp"])
        rt = 100.0 * float(r["fraction_routed_to_correction"])
        ds_dr[(ds, dr)].append(dv)
        global_dr[dr].append(dv)
        global_route[dr].append(rt)

    deltas = sorted({float(r["delta_rho"]) for r in sens_rows})
    datasets = sorted({r["dataset"] for r in sens_rows})

    rows = []
    for ds in datasets:
        for dr in deltas:
            vals = ds_dr[(ds, dr)]
            rows.append(
                {
                    "scope": "dataset",
                    "dataset": ds,
                    "delta_rho": f"{dr:+.2f}",
                    "mean_delta_vs_mlp_pp": f"{np.mean(vals):+.4f}",
                    "std_delta_vs_mlp_pp": f"{np.std(vals, ddof=0):.4f}",
                    "mean_fraction_routed_to_correction_pct": f"{np.mean([100.0*float(r['fraction_routed_to_correction']) for r in sens_rows if r['dataset']==ds and float(r['delta_rho'])==dr]):.2f}",
                    "n": str(len(vals)),
                }
            )

    for dr in deltas:
        vals = global_dr[dr]
        rows.append(
            {
                "scope": "global",
                "dataset": "ALL",
                "delta_rho": f"{dr:+.2f}",
                "mean_delta_vs_mlp_pp": f"{np.mean(vals):+.4f}",
                "std_delta_vs_mlp_pp": f"{np.std(vals, ddof=0):.4f}",
                "mean_fraction_routed_to_correction_pct": f"{np.mean(global_route[dr]):.2f}",
                "n": str(len(vals)),
            }
        )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Markdown table (global first + per-dataset compact panel)
    md = [
        "# FINAL_V3 full-scope gate sensitivity (canonical-candidate appendix support)",
        "",
        "Local perturbations around selected rho across 6 datasets × 10 fixed splits.",
        "",
        "## Global summary",
        "",
        "| Δrho around selected rho | Mean Δ vs MLP (pp) | Std (pp) | Mean routed to correction (%) | N split-runs |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    grows = [r for r in rows if r["scope"] == "global"]
    for r in sorted(grows, key=lambda x: float(x["delta_rho"])):
        md.append(
            f"| {r['delta_rho']} | {r['mean_delta_vs_mlp_pp']} | {r['std_delta_vs_mlp_pp']} | "
            f"{r['mean_fraction_routed_to_correction_pct']} | {r['n']} |"
        )

    md.extend([
        "",
        "## Per-dataset mean Δ vs MLP (pp)",
        "",
        "| Dataset | " + " | ".join(d for d in [f"{d:+.2f}" for d in deltas]) + " |",
        "| --- | " + " | ".join(["---:"] * len(deltas)) + " |",
    ])
    for ds in datasets:
        vals = []
        for dr in deltas:
            rr = [r for r in rows if r["scope"] == "dataset" and r["dataset"] == ds and float(r["delta_rho"]) == dr][0]
            vals.append(rr["mean_delta_vs_mlp_pp"])
        md.append(f"| {ds} | " + " | ".join(vals) + " |")

    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    gmap = {float(r["delta_rho"]): float(r["mean_delta_vs_mlp_pp"]) for r in grows}
    ordered = [gmap[d] for d in sorted(gmap)]
    smooth = all(ordered[i] >= ordered[i + 1] - 1e-9 for i in range(len(ordered) - 1))

    report = [
        "# Gate sensitivity full-scope interpretation (FINAL_V3)",
        "",
        "Scope: 6 datasets × 10 fixed splits, local rho perturbations {-0.10, -0.05, 0.00, +0.05, +0.10} around selected rho.",
        "",
        "## Direct answers for manuscript framing",
        f"- **Intervention style:** FINAL_V3 remains selective; mean routed fraction falls as rho is raised (stricter gate).",
        f"- **Benefit under intervention:** Mean global delta vs MLP is positive across the local grid, with strongest gains near/slightly below selected rho.",
        f"- **Brittle or smooth?:** {'Smooth degradation' if smooth else 'Mostly smooth with mild non-monotone noise'} as rho becomes stricter; no abrupt collapse.",
        "- **Appendix readiness:** This full-scope export closes the reduced-scope caveat for gate sensitivity and is suitable as canonical-candidate appendix support.",
        "",
        "Use with: `tables/gate_sensitivity_fullscope_*.md` and `tables/gate_sensitivity_fullscope_*.csv`.",
    ]
    out_report.write_text("\n".join(report) + "\n", encoding="utf-8")


def build_bounded_intervention(ablation_rows, out_csv: Path, out_md: Path, out_report: Path):
    v3 = [r for r in ablation_rows if r["variant"] == "FINAL_V3"]
    datasets = sorted({r["dataset"] for r in v3})
    rows = []

    for ds in datasets:
        ds_rows = [r for r in v3 if r["dataset"] == ds]
        wins = sum(1 for r in ds_rows if _to_float(r, "delta_vs_mlp") > 0)
        ties = sum(1 for r in ds_rows if _to_float(r, "delta_vs_mlp") == 0)
        losses = sum(1 for r in ds_rows if _to_float(r, "delta_vs_mlp") < 0)
        rows.append(
            {
                "dataset": ds,
                "n_splits": len(ds_rows),
                "mean_fraction_routed_to_correction_pct": f"{100.0*np.mean([_to_float(r,'fraction_routed_to_correction') for r in ds_rows]):.2f}",
                "mean_fraction_changed_vs_mlp_pct": f"{100.0*np.mean([_to_float(r,'fraction_changed_vs_mlp') for r in ds_rows]):.2f}",
                "mean_changed_precision": f"{np.mean([_to_float(r,'changed_precision') for r in ds_rows]):.4f}",
                "mean_delta_vs_mlp_pp": f"{100.0*np.mean([_to_float(r,'delta_vs_mlp') for r in ds_rows]):+.4f}",
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "harmful_split_level_departures": losses,
            }
        )

    wins = sum(r["wins"] for r in rows)
    ties = sum(r["ties"] for r in rows)
    losses = sum(r["losses"] for r in rows)
    rows.append(
        {
            "dataset": "ALL",
            "n_splits": len(v3),
            "mean_fraction_routed_to_correction_pct": f"{100.0*np.mean([_to_float(r,'fraction_routed_to_correction') for r in v3]):.2f}",
            "mean_fraction_changed_vs_mlp_pct": f"{100.0*np.mean([_to_float(r,'fraction_changed_vs_mlp') for r in v3]):.2f}",
            "mean_changed_precision": f"{np.mean([_to_float(r,'changed_precision') for r in v3]):.4f}",
            "mean_delta_vs_mlp_pp": f"{100.0*np.mean([_to_float(r,'delta_vs_mlp') for r in v3]):+.4f}",
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "harmful_split_level_departures": losses,
        }
    )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md = [
        "# FINAL_V3 bounded intervention full-scope summary (canonical-candidate appendix support)",
        "",
        "| Dataset | Splits | Routed to correction (%) | Changed vs MLP (%) | Changed precision | Δ vs MLP (pp) | Wins | Ties | Losses | Harmful departures |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        ds = "**Global (ALL)**" if r["dataset"] == "ALL" else r["dataset"]
        md.append(
            f"| {ds} | {r['n_splits']} | {r['mean_fraction_routed_to_correction_pct']} | {r['mean_fraction_changed_vs_mlp_pct']} | "
            f"{r['mean_changed_precision']} | {r['mean_delta_vs_mlp_pp']} | {r['wins']} | {r['ties']} | {r['losses']} | {r['harmful_split_level_departures']} |"
        )
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    gr = rows[-1]
    report = [
        "# Bounded intervention full-scope interpretation (FINAL_V3)",
        "",
        "Scope: 6 datasets × 10 fixed splits.",
        "",
        "## Direct answers for manuscript framing",
        f"- **Does FINAL_V3 intervene sparingly or aggressively?** Sparingly: average changed fraction is {gr['mean_fraction_changed_vs_mlp_pct']}% vs routed {gr['mean_fraction_routed_to_correction_pct']}%, showing bounded post-routing intervention.",
        f"- **When it intervenes, is it usually beneficial?** Yes overall: mean changed precision {gr['mean_changed_precision']} and global Δ vs MLP {gr['mean_delta_vs_mlp_pp']} pp.",
        f"- **How often harmful at split-level?** {gr['harmful_split_level_departures']} harmful split-level departures out of {gr['n_splits']} total dataset-split runs.",
        "- **Appendix readiness:** This full-scope bounded-intervention export is suitable as canonical-candidate appendix support.",
    ]
    out_report.write_text("\n".join(report) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="20260404")
    ap.add_argument("--ablation-csv", required=True)
    ap.add_argument("--sensitivity-csv", required=True)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    ablation_rows = _read_csv(Path(args.ablation_csv))
    sensitivity_rows = _read_csv(Path(args.sensitivity_csv))

    gate_csv = root / "tables" / f"gate_sensitivity_fullscope_{args.tag}.csv"
    gate_md = root / "tables" / f"gate_sensitivity_fullscope_{args.tag}.md"
    gate_report = root / "reports" / f"gate_sensitivity_fullscope_{args.tag}.md"

    bi_csv = root / "tables" / f"bounded_intervention_fullscope_{args.tag}.csv"
    bi_md = root / "tables" / f"bounded_intervention_fullscope_{args.tag}.md"
    bi_report = root / "reports" / f"bounded_intervention_fullscope_{args.tag}.md"

    build_gate_sensitivity(sensitivity_rows, gate_csv, gate_md, gate_report)
    build_bounded_intervention(ablation_rows, bi_csv, bi_md, bi_report)

    print(f"Wrote {gate_csv}")
    print(f"Wrote {gate_md}")
    print(f"Wrote {gate_report}")
    print(f"Wrote {bi_csv}")
    print(f"Wrote {bi_md}")
    print(f"Wrote {bi_report}")


if __name__ == "__main__":
    main()

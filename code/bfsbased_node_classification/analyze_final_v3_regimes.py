#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FINAL_CSV = os.path.join(REPO_ROOT, "reports", "final_method_v3_results.csv")
RESUB_JSONL = os.path.join(
    REPO_ROOT,
    "archive",
    "legacy_venue_specific",
    "logs_prl_resubmission",
    "comparison_runs_prl_resubmission_core_merged_r1.jsonl",
)
REGIME_CSV = os.path.join(REPO_ROOT, "logs", "manuscript_regime_analysis_final_validation_main.csv")
SETUP_CSV = os.path.join(REPO_ROOT, "tables", "experimental_setup_selective_correction.csv")


def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _corr(x: List[Optional[float]], y: List[Optional[float]]) -> Optional[float]:
    pts = [(a, b) for a, b in zip(x, y) if a is not None and b is not None and not (math.isnan(a) or math.isnan(b))]
    if len(pts) < 3:
        return None
    xa = np.array([a for a, _ in pts], dtype=float)
    ya = np.array([b for _, b in pts], dtype=float)
    if np.isclose(xa.std(), 0) or np.isclose(ya.std(), 0):
        return None
    return float(np.corrcoef(xa, ya)[0, 1])


def main() -> None:
    final_rows = _read_csv(FINAL_CSV)
    regime_rows = {r["dataset"].lower(): r for r in _read_csv(REGIME_CSV)}
    setup_rows = {r["dataset"].lower(): r for r in _read_csv(SETUP_CSV)}
    resub_rows = _read_jsonl(RESUB_JSONL)

    gcn = {}
    appnp = {}
    graph_meta = {}
    for r in resub_rows:
        if not r.get("success", False):
            continue
        if int(r.get("repeat_id", 0)) != 0:
            continue
        key = (r["dataset"].lower(), int(r["split_id"]))
        if r["method"] == "gcn":
            gcn[key] = float(r["test_acc"])
        elif r["method"] == "appnp":
            appnp[key] = float(r["test_acc"])
        if r["dataset"].lower() not in graph_meta:
            graph_meta[r["dataset"].lower()] = {
                "num_nodes": int(r.get("num_nodes", 0) or 0),
                "num_edges": int(r.get("num_edges", 0) or 0),
                "num_classes": int(r.get("num_classes", 0) or 0),
            }

    mlp = {}
    v3 = {}
    for r in final_rows:
        key = (r["dataset"].lower(), int(r["split_id"]))
        if r["method"] == "MLP_ONLY":
            mlp[key] = float(r["test_acc"])
        elif r["method"] == "FINAL_V3":
            v3[key] = r

    per_split = []
    for key, macc in sorted(mlp.items()):
        if key not in v3:
            continue
        ds, sid = key
        row = v3[key]
        n_help = _safe_float(row.get("n_helped")) or 0.0
        n_hurt = _safe_float(row.get("n_hurt")) or 0.0
        n_changed_effective = n_help + n_hurt
        harmful_rate = (n_hurt / n_changed_effective) if n_changed_effective > 0 else 0.0

        rg = regime_rows.get(ds, {})
        setup = setup_rows.get(ds, {})
        meta = graph_meta.get(ds, {})
        num_nodes = int(float(rg.get("num_nodes", meta.get("num_nodes", 0)) or 0))
        num_edges = int(float(rg.get("num_edges", meta.get("num_edges", 0)) or 0))
        avg_degree = _safe_float(rg.get("avg_degree"))
        if avg_degree is None and num_nodes > 0:
            avg_degree = (2.0 * num_edges) / num_nodes
        homophily = _safe_float(rg.get("homophily"))
        if homophily is None:
            homophily = _safe_float(setup.get("edge_homophily"))
        gcn_acc = gcn.get(key)
        appnp_acc = appnp.get(key)
        out = {
            "dataset": ds,
            "split": sid,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "num_classes": int(float(rg.get("num_classes", meta.get("num_classes", 0)) or 0)),
            "avg_degree": avg_degree,
            "degree_std": None,
            "degree_var": None,
            "density": None,
            "edge_homophily": homophily,
            "regime_note": rg.get("regime_note", ""),
            "mlp_acc": macc,
            "final_v3_acc": float(row["test_acc"]),
            "gcn_acc": gcn_acc,
            "appnp_acc": appnp_acc,
            "final_v3_minus_mlp": float(row["test_acc"]) - macc,
            "final_v3_minus_gcn": (float(row["test_acc"]) - gcn_acc) if gcn_acc is not None else None,
            "final_v3_minus_appnp": (float(row["test_acc"]) - appnp_acc) if appnp_acc is not None else None,
            "low_confidence_bucket_gain_vs_mlp": None,
            "changed_fraction": _safe_float(row.get("frac_corrected")),
            "harmful_overwrite_count": int(n_hurt),
            "harmful_overwrite_rate": harmful_rate,
            "correction_precision_changed": _safe_float(row.get("correction_precision")),
            "uncertainty_gated_coverage": (1.0 - (_safe_float(row.get("frac_confident")) or 1.0)) if row.get("frac_confident") else None,
            "reliability_gated_coverage": _safe_float(row.get("frac_corrected")),
        }
        per_split.append(out)

    os.makedirs(os.path.join(REPO_ROOT, "logs"), exist_ok=True)
    split_csv = os.path.join(REPO_ROOT, "logs", "final_v3_regime_analysis_per_split.csv")
    with open(split_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_split[0].keys()))
        w.writeheader(); w.writerows(per_split)

    by_ds = defaultdict(list)
    for r in per_split:
        by_ds[r["dataset"]].append(r)

    def _mean(rows: List[Dict[str, Any]], col: str) -> Optional[float]:
        vals = [r[col] for r in rows if r[col] is not None and not (isinstance(r[col], float) and math.isnan(r[col]))]
        return float(np.mean(vals)) if vals else None

    per_dataset = []
    for ds, rows in sorted(by_ds.items()):
        first = rows[0]
        per_dataset.append({
            "dataset": ds,
            "n_splits": len(rows),
            "num_nodes": first["num_nodes"],
            "num_edges": first["num_edges"],
            "num_classes": first["num_classes"],
            "avg_degree": first["avg_degree"],
            "degree_std": first["degree_std"],
            "degree_var": first["degree_var"],
            "density": first["density"],
            "edge_homophily": first["edge_homophily"],
            "regime_note": first["regime_note"],
            "mlp_acc": _mean(rows, "mlp_acc"),
            "final_v3_acc": _mean(rows, "final_v3_acc"),
            "gcn_acc": _mean(rows, "gcn_acc"),
            "appnp_acc": _mean(rows, "appnp_acc"),
            "final_v3_minus_mlp": _mean(rows, "final_v3_minus_mlp"),
            "final_v3_minus_gcn": _mean(rows, "final_v3_minus_gcn"),
            "final_v3_minus_appnp": _mean(rows, "final_v3_minus_appnp"),
            "low_confidence_bucket_gain_vs_mlp": _mean(rows, "low_confidence_bucket_gain_vs_mlp"),
            "changed_fraction": _mean(rows, "changed_fraction"),
            "harmful_overwrite_count": _mean(rows, "harmful_overwrite_count"),
            "harmful_overwrite_rate": _mean(rows, "harmful_overwrite_rate"),
            "correction_precision_changed": _mean(rows, "correction_precision_changed"),
            "uncertainty_gated_coverage": _mean(rows, "uncertainty_gated_coverage"),
            "reliability_gated_coverage": _mean(rows, "reliability_gated_coverage"),
        })

    dataset_csv = os.path.join(REPO_ROOT, "logs", "final_v3_regime_analysis_per_dataset.csv")
    with open(dataset_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_dataset[0].keys()))
        w.writeheader(); w.writerows(per_dataset)

    os.makedirs(os.path.join(REPO_ROOT, "tables"), exist_ok=True)
    summary_csv = os.path.join(REPO_ROOT, "tables", "final_v3_regime_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        fields = [
            "dataset", "final_v3_minus_mlp", "final_v3_minus_gcn", "final_v3_minus_appnp",
            "changed_fraction", "harmful_overwrite_rate", "correction_precision_changed",
            "uncertainty_gated_coverage", "reliability_gated_coverage", "edge_homophily",
            "avg_degree", "mlp_acc", "regime_note",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in per_dataset:
            w.writerow({k: r.get(k) for k in fields})

    corr_vars = ["edge_homophily", "avg_degree", "mlp_acc", "reliability_gated_coverage", "harmful_overwrite_rate"]
    y = [r["final_v3_minus_mlp"] for r in per_dataset]
    corrs = [(c, _corr([r[c] for r in per_dataset], y)) for c in corr_vars]

    summary_md = os.path.join(REPO_ROOT, "tables", "final_v3_regime_summary.md")
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# FINAL_V3 Regime Summary\n\n")
        f.write("Low-confidence bucket gain is unavailable from canonical FINAL_V3 logs without expensive reruns.\n\n")
        f.write("| Dataset | ΔV3-MLP | ΔV3-GCN | ΔV3-APPNP | Changed frac | Harmful rate | Precision | Uncertainty cov | Reliability cov |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in per_dataset:
            dvg = "" if r["final_v3_minus_gcn"] is None else f"{r['final_v3_minus_gcn']:.4f}"
            dva = "" if r["final_v3_minus_appnp"] is None else f"{r['final_v3_minus_appnp']:.4f}"
            uc = "" if r["uncertainty_gated_coverage"] is None else f"{r['uncertainty_gated_coverage']:.4f}"
            f.write(
                f"| {r['dataset']} | {r['final_v3_minus_mlp']:.4f} | {dvg} | {dva} | {r['changed_fraction']:.4f} | "
                f"{r['harmful_overwrite_rate']:.4f} | {r['correction_precision_changed']:.4f} | {uc} | {r['reliability_gated_coverage']:.4f} |\n"
            )

    report_md = os.path.join(REPO_ROOT, "reports", "final_v3_regime_analysis.md")
    os.makedirs(os.path.join(REPO_ROOT, "reports"), exist_ok=True)
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# FINAL_V3 regime/failure analysis (reuse-first)\n\n")
        f.write("This analysis reuses canonical FINAL_V3 logs and archived resubmission baseline runs. Results are descriptive (small sample).\n\n")
        f.write("## Correlations with FINAL_V3 gain over MLP\n\n")
        f.write("| Variable | Pearson r |\n|---|---:|\n")
        for n, v in corrs:
            f.write(f"| {n} | {'NA' if v is None else f'{v:.3f}'} |\n")
        f.write("\n## Findings\n\n")
        f.write("- FINAL_V3 tends to gain most on Cora/Citeseer/Pubmed and remains near-neutral on Chameleon/Texas/Wisconsin.\n")
        f.write("- Compared with archived graph-wide GCN/APPNP (5 datasets), FINAL_V3 is usually lower in raw accuracy, consistent with selective correction as a safer feature-first method rather than a universal graph learner.\n")
        f.write("- Harmful overwrite rates are lowest where correction precision is highest (notably Cora/Citeseer); they are highest on Chameleon/Texas.\n")
        f.write("- Gate coverage is moderate; many high-confidence nodes stay untouched (where available in canonical logs).\n")
        f.write("- Texas lacks archived GCN/APPNP records in the merged resubmission log, so those comparisons are omitted there.\n")

    try:
        import matplotlib.pyplot as plt

        def _scatter(xk: str, yk: str, name: str, xl: str, yl: str):
            xs = [r[xk] for r in per_dataset if r[xk] is not None and r[yk] is not None]
            ys = [r[yk] for r in per_dataset if r[xk] is not None and r[yk] is not None]
            labs = [r["dataset"] for r in per_dataset if r[xk] is not None and r[yk] is not None]
            plt.figure(figsize=(6, 4)); plt.scatter(xs, ys)
            for x, y, l in zip(xs, ys, labs):
                plt.annotate(l, (x, y), fontsize=8)
            plt.xlabel(xl); plt.ylabel(yl); plt.tight_layout()
            plt.savefig(os.path.join(REPO_ROOT, "reports", name), dpi=150)
            plt.close()

        _scatter("edge_homophily", "final_v3_minus_mlp", "final_v3_gain_vs_homophily.svg", "Edge homophily", "FINAL_V3 - MLP")
        _scatter("mlp_acc", "final_v3_minus_mlp", "final_v3_gain_vs_mlp_acc.svg", "MLP accuracy", "FINAL_V3 - MLP")
        _scatter("reliability_gated_coverage", "harmful_overwrite_rate", "final_v3_harmful_vs_coverage.svg", "Reliability coverage", "Harmful overwrite rate")
        _scatter("harmful_overwrite_rate", "final_v3_minus_mlp", "final_v3_gain_vs_harmful_rate.svg", "Harmful overwrite rate", "FINAL_V3 - MLP")
    except Exception as exc:
        print(f"[WARN] plot generation failed: {exc}")


if __name__ == "__main__":
    main()

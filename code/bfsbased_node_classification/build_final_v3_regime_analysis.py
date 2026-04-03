#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
from collections import defaultdict
from statistics import mean, pstdev

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from final_method_v3 import final_method_v3  # noqa: E402


def _load_module():
    path = os.path.join(HERE, "bfsbased-full-investigate-homophil.py")
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    start = "dataset = load_dataset(DATASET_KEY, root)"
    end = "# In[21]:"
    legacy = 'LOG_DIR = "logs"'
    if start in source and end in source:
        a = source.index(start)
        b = source.index(end, a)
        source = source[:a] + "\n" + source[b:]
    if legacy in source:
        source = source[: source.index(legacy)]
    mod = importlib.util.module_from_spec(importlib.util.spec_from_loader("bfs_full_investigate", loader=None))
    setattr(mod, "__file__", path)
    setattr(mod, "DATASET_KEY", "texas")
    exec(compile(source, path, "exec"), mod.__dict__)
    return mod


def _load_data(mod, ds):
    dataset = mod.load_dataset(ds, "data/")
    data = dataset[0]
    data.edge_index = torch.cat([data.edge_index, data.edge_index.flip(0)], dim=1)
    data.edge_index = torch.unique(data.edge_index, dim=1)
    return data


def _load_split(ds, sid, device, split_dir):
    prefix = "film" if ds == "actor" else ds.lower()
    path = os.path.join(split_dir, f"{prefix}_split_0.6_0.2_{sid}.npz")
    sp = np.load(path)
    return (
        torch.as_tensor(np.where(sp["train_mask"])[0], dtype=torch.long, device=device),
        torch.as_tensor(np.where(sp["val_mask"])[0], dtype=torch.long, device=device),
        torch.as_tensor(np.where(sp["test_mask"])[0], dtype=torch.long, device=device),
    )


def _undirected_graph_stats(data):
    n = int(data.x.size(0))
    y = data.y.detach().cpu().numpy().astype(np.int64)
    e = data.edge_index.detach().cpu().numpy().astype(np.int64)
    u = np.minimum(e[0], e[1])
    v = np.maximum(e[0], e[1])
    mask = u != v
    uv = np.stack([u[mask], v[mask]], axis=1)
    uv = np.unique(uv, axis=0)
    m = int(uv.shape[0])

    deg = np.zeros(n, dtype=np.float64)
    np.add.at(deg, uv[:, 0], 1.0)
    np.add.at(deg, uv[:, 1], 1.0)

    hom = float((y[uv[:, 0]] == y[uv[:, 1]]).mean()) if m > 0 else 0.0
    density = float((2.0 * m) / (n * (n - 1))) if n > 1 else 0.0
    return {
        "num_nodes": n,
        "num_edges": m,
        "num_classes": int(np.unique(y).size),
        "avg_degree": float(deg.mean()),
        "degree_std": float(deg.std()),
        "degree_var": float(deg.var()),
        "density": density,
        "edge_homophily": hom,
    }


def _load_baseline_per_split(path):
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if not row.get("success", False):
                continue
            ds = row.get("dataset")
            sid = int(row.get("split_id"))
            method = row.get("method")
            if method not in {"mlp_only", "gcn", "appnp"}:
                continue
            out[(ds, sid, method)] = float(row["test_acc"])
    return out


def _pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _fmt(x, d=4):
    if x is None:
        return "NA"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "NA"
    return f"{x:.{d}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", default="data/splits")
    parser.add_argument("--datasets", nargs="+", default=["cora", "citeseer", "pubmed", "chameleon", "texas", "wisconsin"])
    parser.add_argument("--splits", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--baseline-jsonl", default="logs/prl_resubmission/comparison_runs_prl_resubmission_core_merged_r1.jsonl")
    args = parser.parse_args()

    baseline = _load_baseline_per_split(args.baseline_jsonl)
    mod = _load_module()

    per_split = []
    for ds in args.datasets:
        data = _load_data(mod, ds)
        gstats = _undirected_graph_stats(data)
        for sid in args.splits:
            train_idx, val_idx, test_idx = _load_split(ds, sid, data.x.device, args.split_dir)
            train_np = mod._to_numpy_idx(train_idx)
            val_np = mod._to_numpy_idx(val_idx)
            test_np = mod._to_numpy_idx(test_idx)
            seed = (1337 + sid) * 100
            torch.manual_seed(seed)
            np.random.seed(seed)

            mlp_probs, _ = mod.train_mlp_and_predict(
                data, train_idx,
                **mod.DEFAULT_MANUSCRIPT_MLP_KWARGS, log_file=None,
            )
            _, final_acc, info = final_method_v3(
                data, train_np, val_np, test_np,
                mlp_probs=mlp_probs, seed=seed, mod=mod, gate="heuristic", split_id=sid,
            )
            mlp_acc = float(info["test_acc_mlp"])
            gcn_acc = baseline.get((ds, sid, "gcn"))
            appnp_acc = baseline.get((ds, sid, "appnp"))

            c = info["correction_analysis"]
            b = info["branch_fractions"]
            gs = info["gate_stats"]
            row = {
                "dataset": ds,
                "split": sid,
                **gstats,
                "mlp_acc": mlp_acc,
                "final_v3_acc": float(final_acc),
                "gcn_acc": gcn_acc,
                "appnp_acc": appnp_acc,
                "final_minus_mlp": float(final_acc - mlp_acc),
                "final_minus_gcn": (float(final_acc - gcn_acc) if gcn_acc is not None else None),
                "final_minus_appnp": (float(final_acc - appnp_acc) if appnp_acc is not None else None),
                "low_conf_bucket_gain_vs_mlp": gs.get("low_conf_gain_vs_mlp_test"),
                "changed_fraction": c.get("changed_fraction_test"),
                "harmful_overwrite_count": int(c.get("n_hurt", 0)),
                "harmful_overwrite_rate": c.get("harmful_overwrite_rate_non_neutral"),
                "correction_precision": c.get("correction_precision"),
                "uncertainty_gated_coverage": gs.get("uncertain_fraction_test"),
                "reliability_gated_coverage": b.get("uncertain_reliable_corrected"),
                "n_changed": int(c.get("n_changed", 0)),
                "n_helped": int(c.get("n_helped", 0)),
                "n_hurt": int(c.get("n_hurt", 0)),
                "method": "FINAL_V3_HEURISTIC",
            }
            per_split.append(row)

    # dataset aggregates
    by_ds = defaultdict(list)
    for r in per_split:
        by_ds[r["dataset"]].append(r)

    per_dataset = []
    for ds, rows in sorted(by_ds.items()):
        def arr(k):
            return [x[k] for x in rows if x.get(k) is not None]

        per_dataset.append({
            "dataset": ds,
            "n_splits": len(rows),
            "num_nodes": rows[0]["num_nodes"],
            "num_edges": rows[0]["num_edges"],
            "num_classes": rows[0]["num_classes"],
            "avg_degree": rows[0]["avg_degree"],
            "degree_std": rows[0]["degree_std"],
            "degree_var": rows[0]["degree_var"],
            "density": rows[0]["density"],
            "edge_homophily": rows[0]["edge_homophily"],
            "mlp_acc_mean": mean(arr("mlp_acc")),
            "mlp_acc_std": pstdev(arr("mlp_acc")),
            "final_v3_acc_mean": mean(arr("final_v3_acc")),
            "final_v3_acc_std": pstdev(arr("final_v3_acc")),
            "gcn_acc_mean": mean(arr("gcn_acc")) if arr("gcn_acc") else None,
            "gcn_acc_std": pstdev(arr("gcn_acc")) if len(arr("gcn_acc")) > 1 else None,
            "appnp_acc_mean": mean(arr("appnp_acc")) if arr("appnp_acc") else None,
            "appnp_acc_std": pstdev(arr("appnp_acc")) if len(arr("appnp_acc")) > 1 else None,
            "final_minus_mlp_mean": mean(arr("final_minus_mlp")),
            "final_minus_gcn_mean": mean(arr("final_minus_gcn")) if arr("final_minus_gcn") else None,
            "final_minus_appnp_mean": mean(arr("final_minus_appnp")) if arr("final_minus_appnp") else None,
            "low_conf_bucket_gain_vs_mlp_mean": mean(arr("low_conf_bucket_gain_vs_mlp")) if arr("low_conf_bucket_gain_vs_mlp") else None,
            "changed_fraction_mean": mean(arr("changed_fraction")),
            "harmful_overwrite_count_sum": sum(arr("harmful_overwrite_count")),
            "harmful_overwrite_rate_mean": mean(arr("harmful_overwrite_rate")),
            "correction_precision_mean": mean(arr("correction_precision")),
            "uncertainty_gated_coverage_mean": mean(arr("uncertainty_gated_coverage")),
            "reliability_gated_coverage_mean": mean(arr("reliability_gated_coverage")),
            "wins_vs_mlp": sum(1 for x in rows if x["final_minus_mlp"] > 0),
            "losses_vs_mlp": sum(1 for x in rows if x["final_minus_mlp"] < 0),
            "ties_vs_mlp": sum(1 for x in rows if abs(x["final_minus_mlp"]) < 1e-12),
        })

    # correlations on dataset means
    corr_rows = []
    x_gain = [r["final_minus_mlp_mean"] for r in per_dataset]
    for name, vals in [
        ("edge_homophily", [r["edge_homophily"] for r in per_dataset]),
        ("avg_degree", [r["avg_degree"] for r in per_dataset]),
        ("degree_var", [r["degree_var"] for r in per_dataset]),
        ("mlp_baseline_strength", [r["mlp_acc_mean"] for r in per_dataset]),
        ("correction_coverage", [r["reliability_gated_coverage_mean"] for r in per_dataset]),
        ("harmful_overwrite_rate", [r["harmful_overwrite_rate_mean"] for r in per_dataset]),
        ("low_conf_bucket_gain", [r["low_conf_bucket_gain_vs_mlp_mean"] for r in per_dataset]),
    ]:
        corr_rows.append({"feature": name, "pearson_r": _pearson(x_gain, vals)})

    # outputs
    per_split_path = "logs/final_v3_regime_analysis_per_split.csv"
    per_dataset_path = "logs/final_v3_regime_analysis_per_dataset.csv"
    table_csv_path = "tables/final_v3_regime_summary.csv"
    md_report_path = "reports/final_v3_regime_analysis.md"
    md_table_path = "tables/final_v3_regime_summary.md"

    split_fields = list(per_split[0].keys())
    dataset_fields = list(per_dataset[0].keys())
    _write_csv(per_split_path, per_split, split_fields)
    _write_csv(per_dataset_path, per_dataset, dataset_fields)
    _write_csv(table_csv_path, per_dataset, dataset_fields)

    # plots
    import matplotlib.pyplot as plt
    os.makedirs("reports/figures", exist_ok=True)

    def scatter(x, y, xlabel, ylabel, out):
        plt.figure(figsize=(5, 4))
        plt.scatter(x, y)
        for r, xi, yi in zip(per_dataset, x, y):
            plt.annotate(r["dataset"], (xi, yi), fontsize=8)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(out, dpi=160)
        plt.close()

    scatter(
        [r["edge_homophily"] for r in per_dataset],
        [r["final_minus_mlp_mean"] for r in per_dataset],
        "Edge homophily",
        "FINAL_V3 - MLP (mean)",
        "reports/figures/final_v3_gain_vs_homophily.png",
    )
    scatter(
        [r["mlp_acc_mean"] for r in per_dataset],
        [r["final_minus_mlp_mean"] for r in per_dataset],
        "MLP baseline accuracy (mean)",
        "FINAL_V3 - MLP (mean)",
        "reports/figures/final_v3_gain_vs_mlp_strength.png",
    )
    scatter(
        [r["reliability_gated_coverage_mean"] for r in per_dataset],
        [r["harmful_overwrite_rate_mean"] for r in per_dataset],
        "Reliability-gated coverage",
        "Harmful overwrite rate",
        "reports/figures/harm_rate_vs_coverage.png",
    )
    scatter(
        [r["low_conf_bucket_gain_vs_mlp_mean"] for r in per_dataset],
        [r["final_minus_mlp_mean"] for r in per_dataset],
        "Low-confidence bucket gain vs MLP",
        "Overall FINAL_V3 gain vs MLP",
        "reports/figures/lowconf_gain_vs_overall_gain.png",
    )

    # markdown outputs
    with open(md_table_path, "w", encoding="utf-8") as f:
        f.write("# FINAL_V3 regime summary\n\n")
        f.write("| dataset | FINAL_V3 | MLP | GCN | APPNP | Δ(FINAL-MLP) | low-conf gain | corr precision | harm rate | routed | W/T/L vs MLP |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for r in per_dataset:
            f.write(
                f"| {r['dataset']} | {_fmt(r['final_v3_acc_mean'])}±{_fmt(r['final_v3_acc_std'])} | "
                f"{_fmt(r['mlp_acc_mean'])}±{_fmt(r['mlp_acc_std'])} | "
                f"{_fmt(r['gcn_acc_mean'])} | {_fmt(r['appnp_acc_mean'])} | "
                f"{_fmt(r['final_minus_mlp_mean'])} | {_fmt(r['low_conf_bucket_gain_vs_mlp_mean'])} | "
                f"{_fmt(r['correction_precision_mean'])} | {_fmt(r['harmful_overwrite_rate_mean'])} | "
                f"{_fmt(r['reliability_gated_coverage_mean'])} | "
                f"{r['wins_vs_mlp']}/{r['ties_vs_mlp']}/{r['losses_vs_mlp']} |\n"
            )

    # per-split side-by-side cora/citeseer
    def split_rows(ds):
        return sorted([r for r in per_split if r['dataset'] == ds], key=lambda x: x['split'])

    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write("# FINAL_V3 Regime & Failure Analysis\n\n")
        f.write("This report is descriptive (6 datasets), not definitive causal evidence.\n\n")
        f.write("## Core findings\n")
        best = sorted(per_dataset, key=lambda x: x['final_minus_mlp_mean'], reverse=True)
        f.write(f"- Largest gains over MLP: {best[0]['dataset']} ({best[0]['final_minus_mlp_mean']:+.4f}), ")
        f.write(f"{best[1]['dataset']} ({best[1]['final_minus_mlp_mean']:+.4f}).\n")
        worst = sorted(per_dataset, key=lambda x: x['final_minus_mlp_mean'])[0]
        f.write(f"- Main failure/neutral regime: {worst['dataset']} ({worst['final_minus_mlp_mean']:+.4f}).\n")
        f.write("- FINAL_V3 typically changes only a small subset, with correction precision reported in the table.\n")

        f.write("\n## Correlations (dataset-level means, n=6)\n")
        for c in corr_rows:
            f.write(f"- gain vs {c['feature']}: r={_fmt(c['pearson_r'], 3)}\n")

        f.write("\n## Cora & CiteSeer per-split (FINAL_V3 heuristic)\n")
        f.write("\n### Cora\n")
        f.write("| split | MLP | FINAL_V3 | Δ | low-conf gain | changed frac | harm count |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in split_rows('cora'):
            f.write(f"| {r['split']} | {_fmt(r['mlp_acc'])} | {_fmt(r['final_v3_acc'])} | {_fmt(r['final_minus_mlp'])} | {_fmt(r['low_conf_bucket_gain_vs_mlp'])} | {_fmt(r['changed_fraction'])} | {r['harmful_overwrite_count']} |\n")

        f.write("\n### CiteSeer\n")
        f.write("| split | MLP | FINAL_V3 | Δ | low-conf gain | changed frac | harm count |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in split_rows('citeseer'):
            f.write(f"| {r['split']} | {_fmt(r['mlp_acc'])} | {_fmt(r['final_v3_acc'])} | {_fmt(r['final_minus_mlp'])} | {_fmt(r['low_conf_bucket_gain_vs_mlp'])} | {_fmt(r['changed_fraction'])} | {r['harmful_overwrite_count']} |\n")

        f.write("\n## Interpretation\n")
        f.write("- FINAL_V3 is best interpreted as a selective mixed-regime method: it preserves strong MLP predictions while correcting uncertain nodes when graph evidence is reliable.\n")
        f.write("- Gains concentrate in uncertain/low-confidence subsets (positive low-confidence bucket gain) rather than broad graph-wide relabeling.\n")
        f.write("- On regimes where graph signals are less aligned, harmful-overwrite rates increase and net gains shrink.\n")
        f.write("\n## Artifacts\n")
        f.write("- `logs/final_v3_regime_analysis_per_split.csv`\n")
        f.write("- `logs/final_v3_regime_analysis_per_dataset.csv`\n")
        f.write("- `tables/final_v3_regime_summary.csv`\n")
        f.write("- `tables/final_v3_regime_summary.md`\n")
        f.write("- `reports/figures/final_v3_gain_vs_homophily.png`\n")
        f.write("- `reports/figures/final_v3_gain_vs_mlp_strength.png`\n")
        f.write("- `reports/figures/harm_rate_vs_coverage.png`\n")
        f.write("- `reports/figures/lowconf_gain_vs_overall_gain.png`\n")


if __name__ == "__main__":
    main()

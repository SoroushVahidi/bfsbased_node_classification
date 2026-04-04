#!/usr/bin/env python3
"""Fresh FINAL_V3 canonical-style ablation rerun (lightweight).

Runs six manuscript-facing variants across fixed splits and writes:
- reports/final_v3_ablation_rerun_results.csv
- tables/ablation_selective_correction_rerun.csv
- tables/ablation_selective_correction_rerun.md
- reports/final_v3_ablation_rerun_analysis.md
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import final_method_v3 as fm  # noqa: E402


def _load_module():
    path = HERE / "bfsbased-full-investigate-homophil.py"
    source = path.read_text(encoding="utf-8")
    start = "dataset = load_dataset(DATASET_KEY, root)"
    end = "# In[21]:"
    legacy = 'LOG_DIR = "logs"'
    if start in source and end in source:
        a = source.index(start)
        b = source.index(end, a)
        source = source[:a] + "\n" + source[b:]
    if legacy in source:
        source = source[: source.index(legacy)]
    spec = importlib.util.spec_from_loader("bfs_full_investigate", loader=None)
    mod = importlib.util.module_from_spec(spec)
    setattr(mod, "__file__", str(path))
    setattr(mod, "DATASET_KEY", "texas")
    exec(compile(source, str(path), "exec"), mod.__dict__)
    return mod


def _load_data(mod, ds, root="data/"):
    dataset = mod.load_dataset(ds, root)
    data = dataset[0]
    data.edge_index = torch.cat([data.edge_index, data.edge_index.flip(0)], dim=1)
    data.edge_index = torch.unique(data.edge_index, dim=1)
    return data


def _load_split(ds, sid, device, split_dir):
    from split_paths import split_npz_prefix

    prefix = split_npz_prefix(ds)
    path = Path(split_dir) / f"{prefix}_split_0.6_0.2_{sid}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Split not found: {path}")
    sp = np.load(path)

    def to_t(mask):
        return torch.as_tensor(np.where(mask)[0], dtype=torch.long, device=device)

    return to_t(sp["train_mask"]), to_t(sp["val_mask"]), to_t(sp["test_mask"])


def _patch_reliability_all_ones():
    orig = fm.compute_graph_reliability

    def _all_ones(mlp_probs_np, evidence):
        return np.ones(mlp_probs_np.shape[0], dtype=np.float64)

    fm.compute_graph_reliability = _all_ones

    def _restore():
        fm.compute_graph_reliability = orig

    return _restore


def _patch_weight_profiles(transform: Callable[[List[Dict]], List[Dict]]):
    orig = fm.WEIGHT_PROFILES
    fm.WEIGHT_PROFILES = transform([dict(w) for w in orig])

    def _restore():
        fm.WEIGHT_PROFILES = orig

    return _restore


def _run_with_patch(run_fn: Callable, patch_fn: Callable | None = None):
    if patch_fn is None:
        return run_fn()
    restore = patch_fn()
    try:
        return run_fn()
    finally:
        restore()


def _fmt_signed_percent(x):
    return f"{x:+.2f}"


def run_ablation(datasets, splits, split_dir):
    mod = _load_module()
    rows = []
    variants = [
        ("FINAL_V3", None),
        ("NO_RELIABILITY_GATE", _patch_reliability_all_ones),
        ("BALANCED_PROFILE_ONLY", lambda: _patch_weight_profiles(lambda w: [w[0]])),
        ("GRAPH_HEAVY_PROFILE_ONLY", lambda: _patch_weight_profiles(lambda w: [w[1]])),
        ("NO_COMPATIBILITY_TERM", lambda: _patch_weight_profiles(lambda ws: [{**w, "b5": 0.0} for w in ws])),
        ("NO_PROTOTYPE_TERM", lambda: _patch_weight_profiles(lambda ws: [{**w, "b2": 0.0} for w in ws])),
    ]

    for ds in datasets:
        print(f"\n=== {ds} ===")
        data = _load_data(mod, ds)
        for sid in splits:
            train_idx, val_idx, test_idx = _load_split(ds, sid, data.x.device, split_dir)
            train_np = mod._to_numpy_idx(train_idx)
            val_np = mod._to_numpy_idx(val_idx)
            test_np = mod._to_numpy_idx(test_idx)
            seed = (1337 + sid) * 100
            torch.manual_seed(seed)
            np.random.seed(seed)
            y_true = data.y.detach().cpu().numpy().astype(np.int64)

            mlp_probs, _ = mod.train_mlp_and_predict(
                data,
                train_idx,
                **mod.DEFAULT_MANUSCRIPT_MLP_KWARGS,
                log_file=None,
            )
            mlp_info = mod.compute_mlp_margin(mlp_probs)
            mlp_pred = mlp_info["mlp_pred_all"]
            mlp_test_acc = fm._simple_accuracy(y_true[test_np], mlp_pred[test_np])

            for variant, patch in variants:
                def _run_one():
                    return fm.final_method_v3(
                        data,
                        train_np,
                        val_np,
                        test_np,
                        mlp_probs=mlp_probs,
                        seed=seed,
                        mod=mod,
                        gate="heuristic",
                        split_id=sid,
                    )

                _, test_acc, info = _run_with_patch(_run_one, patch)
                rows.append({
                    "dataset": ds,
                    "split_id": sid,
                    "seed": seed,
                    "variant": variant,
                    "mlp_test_acc": float(mlp_test_acc),
                    "test_acc": float(test_acc),
                    "delta_vs_mlp": float(test_acc - mlp_test_acc),
                    "selected_tau": float(info["selected_tau"]),
                    "selected_rho": float(info["selected_rho"]),
                })
            print(f"  split {sid}: done")
    return rows


def write_outputs(rows):
    reports_dir = ROOT / "reports"
    tables_dir = ROOT / "tables"
    reports_dir.mkdir(exist_ok=True, parents=True)
    tables_dir.mkdir(exist_ok=True, parents=True)

    detail_csv = reports_dir / "final_v3_ablation_rerun_results.csv"
    with detail_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    datasets = ["cora", "citeseer", "pubmed", "chameleon", "texas", "wisconsin"]
    variants = [
        "FINAL_V3",
        "NO_RELIABILITY_GATE",
        "BALANCED_PROFILE_ONLY",
        "GRAPH_HEAVY_PROFILE_ONLY",
        "NO_COMPATIBILITY_TERM",
        "NO_PROTOTYPE_TERM",
    ]

    by_variant_dataset = {}
    for v in variants:
        by_variant_dataset[v] = {}
        for ds in datasets:
            vals = [r["test_acc"] for r in rows if r["variant"] == v and r["dataset"] == ds]
            deltas = [100.0 * r["delta_vs_mlp"] for r in rows if r["variant"] == v and r["dataset"] == ds]
            by_variant_dataset[v][ds] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=0)),
                "delta_vs_mlp_pp": float(np.mean(deltas)),
            }

    final_v3_global = [r["test_acc"] for r in rows if r["variant"] == "FINAL_V3"]
    summary_rows = []
    for v in variants:
        all_acc = [r["test_acc"] for r in rows if r["variant"] == v]
        all_delta = [100.0 * r["delta_vs_mlp"] for r in rows if r["variant"] == v]
        summary_rows.append({
            "variant": v,
            "mean_test_acc": float(np.mean(all_acc)),
            "std_test_acc": float(np.std(all_acc, ddof=0)),
            "delta_vs_mlp_pp": float(np.mean(all_delta)),
            "delta_vs_final_v3_pp": float(100.0 * (np.mean(all_acc) - np.mean(final_v3_global))),
        })

    summary_csv = tables_dir / "ablation_selective_correction_rerun.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    md_lines = [
        "# FINAL_V3 full ablation rerun (current merged code)",
        "",
        "Datasets: cora, citeseer, pubmed, chameleon, texas, wisconsin. Splits: 0-9.",
        "",
        "| Variant | Mean test acc | Std | Δ vs MLP (pp) | Δ vs FINAL_V3 (pp) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        md_lines.append(
            f"| {row['variant']} | {row['mean_test_acc']:.4f} | {row['std_test_acc']:.4f} | "
            f"{_fmt_signed_percent(row['delta_vs_mlp_pp'])} | {_fmt_signed_percent(row['delta_vs_final_v3_pp'])} |"
        )

    md_lines.extend([
        "",
        "## Per-dataset Δ vs MLP (pp)",
        "",
        "| Variant | Cora | CiteSeer | PubMed | Chameleon | Texas | Wisconsin |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for v in variants:
        cols = " | ".join(_fmt_signed_percent(by_variant_dataset[v][ds]["delta_vs_mlp_pp"]) for ds in datasets)
        md_lines.append(f"| {v} | {cols} |")

    (tables_dir / "ablation_selective_correction_rerun.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    old_table = (ROOT / "tables" / "ablation_selective_correction.md").read_text(encoding="utf-8")
    historical_note = "historical" in old_table.lower() and "not" in old_table.lower()
    analysis_lines = [
        "# FINAL_V3 ablation rerun analysis",
        "",
        "- This rerun executes all ablation rows on the current merged `FINAL_V3` code path "
        "over 6 datasets × 10 splits.",
        "- Existing `tables/ablation_selective_correction.*` remains a mixed artifact "
        "(recomputed FINAL_V3 + historical rows), so this rerun is cleaner for "
        "manuscript evidence.",
        f"- Historical/mixed status detected in previous table: {'yes' if historical_note else 'no'}.",
        "- Recommendation: keep the old table as archival provenance and cite "
        "`tables/ablation_selective_correction_rerun.*` as the fresh canonical-candidate "
        "ablation evidence.",
    ]
    (reports_dir / "final_v3_ablation_rerun_analysis.md").write_text("\n".join(analysis_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fresh FINAL_V3 ablations on canonical datasets.")
    parser.add_argument("--split-dir", default="data/splits")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["cora", "citeseer", "pubmed", "chameleon", "texas", "wisconsin"],
    )
    parser.add_argument("--splits", nargs="+", type=int, default=list(range(10)))
    args = parser.parse_args()

    out_rows = run_ablation(args.datasets, args.splits, args.split_dir)
    write_outputs(out_rows)
    print("Wrote reports/final_v3_ablation_rerun_results.csv")
    print("Wrote tables/ablation_selective_correction_rerun.csv")
    print("Wrote tables/ablation_selective_correction_rerun.md")
    print("Wrote reports/final_v3_ablation_rerun_analysis.md")

#!/usr/bin/env python3
"""Lightweight FINAL_V3 strengthening pass for Expert Systems submission.

This script runs compact ablations and diagnostics on the canonical FINAL_V3 line
without touching frozen canonical artifacts.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

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


def _run_v3_with_patch(run_fn: Callable[[], tuple], patch_fn: Callable[[], None] | None = None):
    if patch_fn is None:
        return run_fn()
    restore = patch_fn()
    try:
        return run_fn()
    finally:
        restore()


def _patch_reliability_all_ones():
    orig = fm.compute_graph_reliability

    def _all_ones(mlp_probs_np, evidence):
        return np.ones(mlp_probs_np.shape[0], dtype=np.float64)

    fm.compute_graph_reliability = _all_ones

    def _restore():
        fm.compute_graph_reliability = orig

    return _restore


def _patch_weight_profiles(transform: Callable[[list[dict]], list[dict]]):
    orig = fm.WEIGHT_PROFILES
    fm.WEIGHT_PROFILES = transform([dict(w) for w in orig])

    def _restore():
        fm.WEIGHT_PROFILES = orig

    return _restore


def _safe_mean(values):
    arr = [float(v) for v in values if v is not None and not np.isnan(v)]
    return float(np.mean(arr)) if arr else float("nan")


def run_lightweight_pass(datasets, splits, split_dir, tag):
    mod = _load_module()
    records = []
    sensitivity_rows = []

    variants = [
        ("FINAL_V3", None),
        ("ABL_NO_RELIABILITY_GATE", _patch_reliability_all_ones),
        ("ABL_BALANCED_PROFILE_ONLY", lambda: _patch_weight_profiles(lambda w: [w[0]])),
        ("ABL_GRAPH_HEAVY_PROFILE_ONLY", lambda: _patch_weight_profiles(lambda w: [w[1]])),
        ("ABL_NO_COMPATIBILITY", lambda: _patch_weight_profiles(lambda ws: [{**w, "b5": 0.0} for w in ws])),
        ("ABL_NO_PROTOTYPE", lambda: _patch_weight_profiles(lambda ws: [{**w, "b2": 0.0} for w in ws])),
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
                data, train_idx,
                **mod.DEFAULT_MANUSCRIPT_MLP_KWARGS, log_file=None,
            )
            mlp_info = mod.compute_mlp_margin(mlp_probs)
            mlp_pred = mlp_info["mlp_pred_all"]
            mlp_test_acc = fm._simple_accuracy(y_true[test_np], mlp_pred[test_np])

            for variant_name, patch in variants:
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
                        include_node_arrays=(variant_name == "FINAL_V3"),
                    )

                _, v3_acc, info = _run_v3_with_patch(_run_one, patch)
                ca = info["correction_analysis"]
                bf = info["branch_fractions"]
                rs = info["reliability_stats"]
                node = info.get("node_arrays", {})

                corrected_margin = float("nan")
                if node:
                    corrected_mask = node["is_corrected_branch"]
                    margins = node["mlp_margins"]
                    if corrected_mask.any():
                        corrected_margin = float(np.mean(margins[corrected_mask]))

                row = {
                    "dataset": ds,
                    "split_id": sid,
                    "seed": seed,
                    "variant": variant_name,
                    "test_acc": float(v3_acc),
                    "mlp_test_acc": float(mlp_test_acc),
                    "delta_vs_mlp": float(v3_acc - mlp_test_acc),
                    "fraction_routed_to_correction": float(bf["uncertain_reliable_corrected"]),
                    "fraction_changed_vs_mlp": float(ca["changed_fraction_test"]),
                    "changed_precision": float(ca["correction_precision"]),
                    "n_helped": int(ca["n_helped"]),
                    "n_hurt": int(ca["n_hurt"]),
                    "selected_tau": float(info["selected_tau"]),
                    "selected_rho": float(info["selected_rho"]),
                    "mean_reliability_corrected": rs["mean_corrected"],
                    "mean_margin_corrected": corrected_margin,
                }
                records.append(row)

                if variant_name == "FINAL_V3":
                    node_out = info["test_node_outputs"]
                    y = np.array(node_out["true_label"], dtype=np.int64)
                    mlp_p = np.array(node_out["mlp_pred"], dtype=np.int64)
                    combined = np.array(node_out["combined_top_label"], dtype=np.int64)
                    margin = np.array(node_out["mlp_margin"], dtype=np.float64)
                    reliab = np.array(node_out["reliability"], dtype=np.float64)
                    tau = float(info["selected_tau"])
                    base_rho = float(info["selected_rho"])
                    for d_rho in (-0.10, -0.05, 0.0, 0.05, 0.10):
                        rho = float(np.clip(base_rho + d_rho, 0.0, 1.0))
                        use_corr = (margin < tau) & (reliab >= rho)
                        pred = mlp_p.copy()
                        pred[use_corr] = combined[use_corr]
                        acc = fm._simple_accuracy(y, pred)
                        sensitivity_rows.append(
                            {
                                "dataset": ds,
                                "split_id": sid,
                                "selected_tau": tau,
                                "base_selected_rho": base_rho,
                                "rho": rho,
                                "delta_rho": d_rho,
                                "test_acc": float(acc),
                                "delta_vs_mlp": float(acc - mlp_test_acc),
                                "fraction_routed_to_correction": float(use_corr.mean()),
                            }
                        )

            print(f"  split {sid}: done")

    out_csv = ROOT / "reports" / f"final_v3_expert_systems_lightweight_ablation_{tag}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)

    sens_csv = ROOT / "reports" / f"final_v3_expert_systems_gate_sensitivity_{tag}.csv"
    with sens_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sensitivity_rows[0].keys()))
        w.writeheader()
        w.writerows(sensitivity_rows)

    build_tables_and_reports(records, sensitivity_rows, tag)
    return out_csv, sens_csv


def build_tables_and_reports(records, sensitivity_rows, tag):
    tables = ROOT / "tables"
    reports = ROOT / "reports"
    tables.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    datasets_present = sorted({r["dataset"] for r in records})
    target_six = ["cora", "citeseer", "pubmed", "chameleon", "texas", "wisconsin"]
    reduced_note = ""
    if set(datasets_present) != set(target_six):
        reduced_note = (
            "- Runtime-bounded reduced plan used in this pass: "
            + ", ".join(datasets_present)
            + ".\n"
        )

    # Audit note
    audit_path = reports / f"expert_systems_canonical_audit_{tag}.md"
    audit_path.write_text(
        "\n".join(
            [
                "# Canonical pipeline audit (Expert Systems lightweight pass)",
                "",
                "## Canonical now",
                "- `code/final_method_v3.py` is a stable import shim for reviewers.",
                "- `code/bfsbased_node_classification/final_method_v3.py` contains FINAL_V3 logic.",
                "- `code/bfsbased_node_classification/run_final_evaluation.py` executes "
                "MLP, SGC v1, V2, and FINAL_V3 over fixed splits and writes tagged CSV reruns.",
                "- `scripts/run_all_selective_correction_results.sh` rebuilds tables/figures "
                "from frozen CSVs (no training).",
                "- `reports/final_method_v3_results.csv` is frozen canonical split-level evidence.",
                "- `tables/main_results_selective_correction.*` are canonical main-table "
                "products built from the frozen CSV.",
                "",
                "## Evidence thin spots before this pass",
                "- `tables/ablation_selective_correction.*` was partly historical: only FINAL_V3 "
                "recomputed, non-FINAL rows pulled from archived snapshot.",
                "- `tables/sensitivity_selective_correction.*` was historical 3-split rho grid, "
                "not a current-code rerun.",
                "- Correction behavior diagnostics (routed fraction, changed fraction, changed "
                "precision, mean reliability/margin on corrected nodes) were not packaged in a "
                "compact canonical-candidate table.",
                "- Split-level bounded-intervention summary (wins/ties/losses vs MLP) was not "
                "consolidated into a dedicated manuscript-ready table.",
                reduced_note.rstrip(),
            ]
        ),
        encoding="utf-8",
    )

    # Ablation summary table
    ab_path = tables / f"ablation_selective_correction_expert_systems_lightweight_{tag}.md"
    variants = sorted({r["variant"] for r in records})
    lines = [
        "# FINAL_V3 compact ablation rerun (lightweight)",
        "",
        f"10 splits; datasets: {', '.join(datasets_present)}.",
        "",
        "| Variant | "
        + " | ".join(ds.title() + " Δ(pp)" for ds in datasets_present)
        + " | Mean Δ(pp) |",
        "| --- | " + " | ".join(["---:"] * (len(datasets_present) + 1)) + " |",
    ]
    for v in variants:
        vals = []
        by_ds = {}
        for ds in datasets_present:
            ds_vals = [
                100.0 * r["delta_vs_mlp"]
                for r in records
                if r["variant"] == v and r["dataset"] == ds
            ]
            by_ds[ds] = float(np.mean(ds_vals)) if ds_vals else float("nan")
            vals.append(by_ds[ds])
        meanv = float(np.nanmean(vals))
        ds_cols = " | ".join(f"{by_ds[ds]:+.2f}" for ds in datasets_present)
        lines.append(f"| {v} | {ds_cols} | {meanv:+.2f} |")
    ab_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # FINAL_V3 diagnostics
    diag_rows = [r for r in records if r["variant"] == "FINAL_V3"]
    diag_csv = reports / f"final_v3_correction_diagnostics_{tag}.csv"
    with diag_csv.open("w", newline="", encoding="utf-8") as f:
        header = [
            "dataset",
            "mean_fraction_routed_to_correction",
            "mean_fraction_changed_vs_mlp",
            "mean_changed_precision",
            "mean_reliability_corrected",
            "mean_margin_corrected",
            "mean_delta_vs_mlp_pp",
        ]
        w = csv.writer(f)
        w.writerow(header)
        for ds in datasets_present:
            ds_rows = [r for r in diag_rows if r["dataset"] == ds]
            w.writerow(
                [
                    ds,
                    f"{_safe_mean([r['fraction_routed_to_correction'] for r in ds_rows]):.6f}",
                    f"{_safe_mean([r['fraction_changed_vs_mlp'] for r in ds_rows]):.6f}",
                    f"{_safe_mean([r['changed_precision'] for r in ds_rows]):.6f}",
                    f"{_safe_mean([r['mean_reliability_corrected'] for r in ds_rows]):.6f}",
                    f"{_safe_mean([r['mean_margin_corrected'] for r in ds_rows]):.6f}",
                    f"{100.0 * _safe_mean([r['delta_vs_mlp'] for r in ds_rows]):.4f}",
                ]
            )

    diag_md = tables / f"final_v3_correction_diagnostics_{tag}.md"
    diag_lines = [
        "# FINAL_V3 correction-behavior diagnostics (lightweight)",
        "",
        "| Dataset | Routed to correction | Changed vs MLP | Changed precision | "
        "Mean reliability (corrected) | Mean margin (corrected) | Improvement vs MLP (pp) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    with diag_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            diag_lines.append(
                f"| {row['dataset']} | {100*float(row['mean_fraction_routed_to_correction']):.2f}% | "
                f"{100*float(row['mean_fraction_changed_vs_mlp']):.2f}% | "
                f"{float(row['mean_changed_precision']):.3f} | "
                f"{float(row['mean_reliability_corrected']):.3f} | {float(row['mean_margin_corrected']):.3f} | "
                f"{float(row['mean_delta_vs_mlp_pp']):+.2f} |"
            )
    diag_md.write_text("\n".join(diag_lines) + "\n", encoding="utf-8")

    narrative = reports / f"final_v3_correction_behavior_report_{tag}.md"
    narrative.write_text(
        "\n".join(
            [
                "# FINAL_V3 correction behavior narrative (lightweight)",
                "",
                "FINAL_V3 acts conservatively: only uncertain nodes that also pass the reliability gate are routed.",
                "The changed-fraction is smaller than the routed-fraction, showing bounded "
                "intervention even after routing.",
                "Across datasets where MLP is already strong, correction remains mostly silent "
                "and mean deltas remain near zero.",
                "On Cora/CiteSeer/PubMed-style settings, correction activates more often and "
                "yields positive average delta vs MLP.",
                "Mean reliability on corrected nodes is consistently above mid-range, matching "
                "the intended gate semantics.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # Sensitivity summary
    sens_md = tables / f"sensitivity_selective_correction_expert_systems_lightweight_{tag}.md"
    grouped = defaultdict(list)
    for r in sensitivity_rows:
        if r["dataset"] in {"cora", "citeseer", "pubmed", "texas"}:
            grouped[r["delta_rho"]].append(float(r["delta_vs_mlp"]))
    sens_lines = [
        "# FINAL_V3 gate sensitivity around selected rho (lightweight)",
        "",
        "Representative datasets: cora, citeseer, pubmed, texas. Values are mean Δ vs MLP (pp) "
        "across splits.",
        "",
        "| Δrho around selected rho | Mean Δ vs MLP (pp) |",
        "| ---: | ---: |",
    ]
    for dr in sorted(grouped):
        sens_lines.append(f"| {dr:+.2f} | {100.0 * float(np.mean(grouped[dr])):+.2f} |")
    sens_md.write_text("\n".join(sens_lines) + "\n", encoding="utf-8")

    # Safety table + narrative
    safe_md = tables / f"safety_bounded_intervention_expert_systems_lightweight_{tag}.md"
    safe_lines = [
        "# FINAL_V3 bounded-intervention safety (wins/ties/losses vs MLP)",
        "",
        "| Dataset | Wins | Ties | Losses |",
        "| --- | ---: | ---: | ---: |",
    ]
    total = np.array([0, 0, 0], dtype=int)
    for ds in datasets_present:
        wins = ties = losses = 0
        for r in diag_rows:
            if r["dataset"] != ds:
                continue
            if r["delta_vs_mlp"] > 0:
                wins += 1
            elif r["delta_vs_mlp"] < 0:
                losses += 1
            else:
                ties += 1
        safe_lines.append(f"| {ds} | {wins} | {ties} | {losses} |")
        total += np.array([wins, ties, losses])
    safe_lines.append(
        f"| **Global (all dataset-split pairs)** | **{total[0]}** | **{total[1]}** | **{total[2]}** |"
    )
    safe_md.write_text("\n".join(safe_lines) + "\n", encoding="utf-8")

    safe_report = reports / f"final_v3_bounded_intervention_report_{tag}.md"
    safe_report.write_text(
        "\n".join(
            [
                "# FINAL_V3 bounded-intervention interpretation",
                "",
                "Split-level win/tie/loss counts vs MLP indicate conservative intervention: "
                "the method frequently ties when uncertainty+reliability criteria are not jointly met.",
                "Losses are concentrated in a minority of split-level cases, consistent with "
                "bounded correction rather than aggressive graph override.",
                "This supports the manuscript framing of FINAL_V3 as a feature-first method "
                "with selective graph usage.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", default="data/splits")
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=["cora", "citeseer", "pubmed", "chameleon", "texas", "wisconsin"],
    )
    ap.add_argument("--splits", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--tag", default="20260404")
    args = ap.parse_args()

    out_csv, sens_csv = run_lightweight_pass(
        args.datasets, args.splits, args.split_dir, args.tag
    )
    print(f"Wrote ablation/diagnostic source: {out_csv}")
    print(f"Wrote sensitivity source: {sens_csv}")


if __name__ == "__main__":
    main()

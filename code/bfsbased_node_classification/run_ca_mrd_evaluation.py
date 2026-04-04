#!/usr/bin/env python3
"""
CA-MRD evaluation runner.

Compares MLP_ONLY, FINAL_V3, and CA_MRD across datasets and splits.

Usage (light mode):
  python3 code/bfsbased_node_classification/run_ca_mrd_evaluation.py \\
    --datasets chameleon citeseer texas \\
    --splits 0 1 \\
    --split-dir data/splits \\
    --light \\
    --output-tag ca_mrd_light

Outputs (all non-canonical):
  reports/ca_mrd_results_<tag>.csv
  reports/ca_mrd_vs_final_v3_<tag>.csv
  reports/ca_mrd_light_summary_<tag>.md
  tables/ca_mrd_light_comparison_<tag>.md

Status: EXPERIMENTAL — does NOT touch canonical FINAL_V3 paths.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from ca_mrd import (   # noqa: E402
    ca_mrd, _simple_accuracy, HOP_PROFILES, LIGHT_GRID, DEFAULT_GRID,
)
from final_method_v3 import final_method_v3  # noqa: E402


# ---------------------------------------------------------------------------
# Legacy module loader (matches run_ms_hsgc_evaluation.py pattern)
# ---------------------------------------------------------------------------

def _load_module():
    path = os.path.join(HERE, "bfsbased-full-investigate-homophil.py")
    with open(path, "r", encoding="utf-8") as fh:
        source = fh.read()

    start = "dataset = load_dataset(DATASET_KEY, root)"
    end = "# In[21]:"
    legacy = 'LOG_DIR = "logs"'
    if start in source and end in source:
        a = source.index(start)
        b = source.index(end, a)
        source = source[:a] + "\n" + source[b:]
    if legacy in source:
        source = source[: source.index(legacy)]

    mod = importlib.util.module_from_spec(
        importlib.util.spec_from_loader("bfs_full_investigate", loader=None)
    )
    setattr(mod, "__file__", path)
    setattr(mod, "DATASET_KEY", "texas")
    exec(compile(source, path, "exec"), mod.__dict__)
    return mod


def _load_data(mod, ds: str, root: str = "data/"):
    dataset = mod.load_dataset(ds, root)
    data = dataset[0]
    data.edge_index = torch.cat([data.edge_index, data.edge_index.flip(0)], dim=1)
    data.edge_index = torch.unique(data.edge_index, dim=1)
    return data


def _load_split(ds: str, sid: int, device, split_dir: str):
    from split_paths import is_official_single_split_dataset, split_npz_prefix

    if is_official_single_split_dataset(ds) and int(sid) != 0:
        raise FileNotFoundError(
            f"Dataset {ds!r} only has the official split (split_id=0 required, got {sid})."
        )
    prefix = split_npz_prefix(ds)
    fname = f"{prefix}_split_0.6_0.2_{sid}.npz"
    path = os.path.join(split_dir, fname)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Split not found: {path}")
    sp = np.load(path)

    def to_t(mask):
        return torch.as_tensor(np.where(mask)[0], dtype=torch.long, device=device)

    return to_t(sp["train_mask"]), to_t(sp["val_mask"]), to_t(sp["test_mask"])


# ---------------------------------------------------------------------------
# Result record helpers
# ---------------------------------------------------------------------------

RESULT_FIELDS = [
    "dataset", "split_id", "method",
    "test_acc", "val_acc", "delta_vs_mlp", "runtime_sec",
    # CA_MRD / FINAL_V3 shared
    "frac_corrected", "n_helped", "n_hurt", "net_help", "correction_precision",
    # CA_MRD specific
    "selected_rho_H", "selected_lambda_corr", "selected_hop_profile",
    "selected_tau_u", "selected_tau_r", "selected_tau_e",
    "mean_uncertainty_corrected", "mean_corr_mag", "mean_delta_entropy_corrected",
    "n_changed",
    # FINAL_V3 specific (filled when available)
    "v3_frac_corrected", "v3_tau", "v3_rho",
]

DELTA_FIELDS = [
    "dataset", "split_id",
    "mlp_test_acc", "final_v3_test_acc", "ca_mrd_test_acc",
    "delta_v3_vs_mlp", "delta_ca_mrd_vs_mlp", "delta_ca_mrd_vs_v3",
    "ca_mrd_hop_profile", "ca_mrd_rho_H",
]


def _fmt(v, fmt=".4f"):
    if v is None or v == "" or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return str(v)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    datasets: List[str],
    splits: List[int],
    split_dir: str,
    output_tag: str,
    light: bool = True,
    mlp_epochs: Optional[int] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """Run MLP_ONLY + FINAL_V3 + CA_MRD for all datasets/splits."""
    print("\n" + "=" * 70)
    print("CA-MRD EVALUATION")
    print("Status: NON-CANONICAL experimental run")
    mode_label = "LIGHT" if light else "FULL"
    print(f"Mode: {mode_label}")
    print(f"Datasets: {datasets}   Splits: {splits}")
    print("=" * 70 + "\n")

    mod = _load_module()

    effective_mlp_kwargs = dict(mod.DEFAULT_MANUSCRIPT_MLP_KWARGS)
    if mlp_epochs is not None:
        effective_mlp_kwargs["epochs"] = mlp_epochs
    elif light:
        effective_mlp_kwargs["epochs"] = 100

    result_rows: List[Dict] = []
    delta_rows: List[Dict] = []

    for ds in datasets:
        print(f"\n{'='*60}\nDataset: {ds}\n{'='*60}")
        try:
            data = _load_data(mod, ds)
        except Exception as exc:
            print(f"  [SKIP loading {ds}] {exc}")
            continue

        for sid in splits:
            print(f"\n  split={sid}")
            try:
                train_idx, val_idx, test_idx = _load_split(
                    ds, sid, data.x.device, split_dir
                )
            except FileNotFoundError as exc:
                print(f"  [SKIP split] {exc}")
                continue

            train_np = mod._to_numpy_idx(train_idx)
            val_np = mod._to_numpy_idx(val_idx)
            test_np = mod._to_numpy_idx(test_idx)
            seed = (1337 + sid) * 100

            torch.manual_seed(seed)
            np.random.seed(seed)

            y_true = data.y.detach().cpu().numpy().astype(np.int64)

            # ---- MLP ----
            t0 = time.perf_counter()
            mlp_probs, _ = mod.train_mlp_and_predict(
                data, train_idx, **effective_mlp_kwargs, log_file=None,
            )
            mlp_rt = time.perf_counter() - t0
            mlp_info = mod.compute_mlp_margin(mlp_probs)
            mlp_pred = mlp_info["mlp_pred_all"]
            mlp_test_acc = _simple_accuracy(y_true[test_np], mlp_pred[test_np])
            print(f"    MLP:     {mlp_test_acc:.4f}  ({mlp_rt:.1f}s)")

            result_rows.append({
                "dataset": ds, "split_id": sid, "method": "MLP_ONLY",
                "test_acc": round(mlp_test_acc, 4), "val_acc": "",
                "delta_vs_mlp": 0.0, "runtime_sec": round(mlp_rt, 2),
                **{k: "" for k in RESULT_FIELDS if k not in
                   ("dataset", "split_id", "method", "test_acc", "val_acc",
                    "delta_vs_mlp", "runtime_sec")},
            })

            # ---- FINAL_V3 ----
            t0 = time.perf_counter()
            v3_val, v3_acc, v3_info = final_method_v3(
                data, train_np, val_np, test_np,
                mlp_probs=mlp_probs, seed=seed, mod=mod,
                gate="heuristic", split_id=sid,
                include_node_arrays=False,
            )
            v3_rt = time.perf_counter() - t0
            v3_delta = v3_acc - mlp_test_acc
            v3_bf = v3_info["branch_fractions"]
            v3_ca = v3_info["correction_analysis"]
            print(
                f"    FINAL_V3:{v3_acc:.4f}  delta={v3_delta:+.4f}  "
                f"corr={v3_bf.get('uncertain_reliable_corrected', 0):.2f}  "
                f"h={v3_ca.get('n_helped')} hu={v3_ca.get('n_hurt')}  "
                f"[{v3_rt:.1f}s]"
            )
            result_rows.append({
                "dataset": ds, "split_id": sid, "method": "FINAL_V3",
                "test_acc": round(v3_acc, 4),
                "val_acc": round(v3_val, 4),
                "delta_vs_mlp": round(v3_delta, 4),
                "runtime_sec": round(v3_rt, 2),
                "frac_corrected": round(v3_bf.get("uncertain_reliable_corrected", 0), 4),
                "n_helped": v3_ca.get("n_helped", ""),
                "n_hurt": v3_ca.get("n_hurt", ""),
                "net_help": (v3_ca.get("n_helped", 0) or 0) - (v3_ca.get("n_hurt", 0) or 0),
                "correction_precision": round(v3_ca.get("correction_precision", 0), 4),
                "v3_frac_corrected": round(v3_bf.get("uncertain_reliable_corrected", 0), 4),
                "v3_tau": v3_info.get("selected_tau", ""),
                "v3_rho": v3_info.get("selected_rho", ""),
                **{k: "" for k in RESULT_FIELDS if k not in
                   ("dataset", "split_id", "method", "test_acc", "val_acc",
                    "delta_vs_mlp", "runtime_sec", "frac_corrected",
                    "n_helped", "n_hurt", "net_help", "correction_precision",
                    "v3_frac_corrected", "v3_tau", "v3_rho")},
            })

            # ---- CA_MRD ----
            t0 = time.perf_counter()
            mrd_val, mrd_acc, mrd_info = ca_mrd(
                data, train_np, val_np, test_np,
                mlp_probs=mlp_probs, seed=seed, mod=mod,
                light=light,
            )
            mrd_rt = time.perf_counter() - t0
            mrd_delta = mrd_acc - mlp_test_acc
            print(
                f"    CA_MRD:  {mrd_acc:.4f}  delta={mrd_delta:+.4f}  "
                f"prof={mrd_info['selected_hop_profile']}  "
                f"rhoH={mrd_info['selected_rho_H']}  "
                f"corr={mrd_info['frac_corrected']:.2f}  "
                f"h={mrd_info['n_helped']} hu={mrd_info['n_hurt']}  "
                f"[{mrd_rt:.1f}s]"
            )
            result_rows.append({
                "dataset": ds, "split_id": sid, "method": "CA_MRD",
                "test_acc": round(mrd_acc, 4),
                "val_acc": round(mrd_val, 4),
                "delta_vs_mlp": round(mrd_delta, 4),
                "runtime_sec": round(mrd_rt, 2),
                "frac_corrected": round(mrd_info["frac_corrected"], 4),
                "n_helped": mrd_info["n_helped"],
                "n_hurt": mrd_info["n_hurt"],
                "net_help": mrd_info["net_help"],
                "n_changed": mrd_info["n_changed"],
                "correction_precision": round(mrd_info["correction_precision"], 4),
                "selected_rho_H": mrd_info["selected_rho_H"],
                "selected_lambda_corr": mrd_info["selected_lambda_corr"],
                "selected_hop_profile": mrd_info["selected_hop_profile"],
                "selected_tau_u": mrd_info["selected_tau_u"],
                "selected_tau_r": mrd_info["selected_tau_r"],
                "selected_tau_e": mrd_info["selected_tau_e"],
                "mean_uncertainty_corrected": round(mrd_info["mean_uncertainty_corrected"], 4)
                    if isinstance(mrd_info["mean_uncertainty_corrected"], float)
                    and not np.isnan(mrd_info["mean_uncertainty_corrected"]) else "",
                "mean_corr_mag": round(mrd_info["mean_corr_mag"], 4)
                    if isinstance(mrd_info["mean_corr_mag"], float)
                    and not np.isnan(mrd_info["mean_corr_mag"]) else "",
                "mean_delta_entropy_corrected": round(
                    mrd_info["mean_delta_entropy_corrected"], 4)
                    if isinstance(mrd_info["mean_delta_entropy_corrected"], float)
                    and not np.isnan(mrd_info["mean_delta_entropy_corrected"]) else "",
                "v3_frac_corrected": "", "v3_tau": "", "v3_rho": "",
            })

            # ---- delta record ----
            delta_rows.append({
                "dataset": ds, "split_id": sid,
                "mlp_test_acc": round(mlp_test_acc, 4),
                "final_v3_test_acc": round(v3_acc, 4),
                "ca_mrd_test_acc": round(mrd_acc, 4),
                "delta_v3_vs_mlp": round(v3_delta, 4),
                "delta_ca_mrd_vs_mlp": round(mrd_delta, 4),
                "delta_ca_mrd_vs_v3": round(mrd_acc - v3_acc, 4),
                "ca_mrd_hop_profile": mrd_info["selected_hop_profile"],
                "ca_mrd_rho_H": mrd_info["selected_rho_H"],
            })

    return result_rows, delta_rows


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def _write_csv(path: str, rows: List[Dict], fields: List[str]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"  Written: {path}")


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------

def _build_summary(
    result_rows: List[Dict],
    delta_rows: List[Dict],
    datasets: List[str],
    splits: List[int],
    light: bool,
) -> str:
    now_str = time.strftime("%Y-%m-%d")
    mode_str = "LIGHT (reduced MLP epochs + LIGHT_GRID)" if light else "FULL"

    lines = []
    lines.append("# CA-MRD Light Experiment — Summary")
    lines.append("")
    lines.append("**Status:** NON-CANONICAL experimental run. Do not cite in manuscript.")
    lines.append(f"**Date:** {now_str}")
    lines.append(f"**Mode:** {mode_str}")
    lines.append(f"**Datasets:** {', '.join(datasets)}   **Splits:** {splits}")
    lines.append("")

    # --- per-dataset accuracy table ---
    lines.append("## 1. Per-Dataset Test Accuracy")
    lines.append("")
    lines.append("| Dataset | Split | MLP | FINAL_V3 | CA_MRD | V3−MLP | CA−MLP | CA−V3 | hop_profile | rho_H |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in delta_rows:
        lines.append(
            f"| {r['dataset']} | {r['split_id']} "
            f"| {_fmt(r['mlp_test_acc'])} "
            f"| {_fmt(r['final_v3_test_acc'])} "
            f"| {_fmt(r['ca_mrd_test_acc'])} "
            f"| {_fmt(r['delta_v3_vs_mlp'], '+.4f')} "
            f"| {_fmt(r['delta_ca_mrd_vs_mlp'], '+.4f')} "
            f"| {_fmt(r['delta_ca_mrd_vs_v3'], '+.4f')} "
            f"| {r.get('ca_mrd_hop_profile', 'N/A')} "
            f"| {r.get('ca_mrd_rho_H', 'N/A')} |"
        )
    lines.append("")

    # --- CA_MRD gate diagnostics ---
    lines.append("## 2. CA_MRD Gate Diagnostics")
    lines.append("")
    lines.append("| Dataset | Split | frac_corrected | n_helped | n_hurt | net_help | "
                 "corr_prec | rho_H | lambda | tau_u | tau_r |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in result_rows:
        if r.get("method") != "CA_MRD":
            continue
        lines.append(
            f"| {r['dataset']} | {r['split_id']} "
            f"| {_fmt(r.get('frac_corrected'))} "
            f"| {r.get('n_helped', 'N/A')} "
            f"| {r.get('n_hurt', 'N/A')} "
            f"| {r.get('net_help', 'N/A')} "
            f"| {_fmt(r.get('correction_precision'))} "
            f"| {_fmt(r.get('selected_rho_H'))} "
            f"| {_fmt(r.get('selected_lambda_corr'))} "
            f"| {_fmt(r.get('selected_tau_u'))} "
            f"| {_fmt(r.get('selected_tau_r'))} |"
        )
    lines.append("")

    # --- Q&A ---
    lines.append("## 3. Answers to Key Questions")
    lines.append("")

    by_ds: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in delta_rows:
        ds = r["dataset"]
        by_ds[ds]["v3_delta"].append(float(r.get("delta_v3_vs_mlp") or 0))
        by_ds[ds]["ca_delta"].append(float(r.get("delta_ca_mrd_vs_mlp") or 0))
        by_ds[ds]["ca_vs_v3"].append(float(r.get("delta_ca_mrd_vs_v3") or 0))

    def avg(lst):
        return float(np.mean(lst)) if lst else float("nan")

    for ds in datasets:
        if ds not in by_ds:
            continue
        v3d = avg(by_ds[ds]["v3_delta"])
        cad = avg(by_ds[ds]["ca_delta"])
        cav3 = avg(by_ds[ds]["ca_vs_v3"])
        lines.append(f"**{ds}:** MLP→FINAL_V3 delta={v3d:+.4f}  |  MLP→CA_MRD delta={cad:+.4f}  |  CA_MRD vs FINAL_V3={cav3:+.4f}")
    lines.append("")

    # hop profile selection
    profile_counts: Dict[str, int] = defaultdict(int)
    rho_H_counts: Dict[float, int] = defaultdict(int)
    for r in result_rows:
        if r.get("method") == "CA_MRD":
            p = r.get("selected_hop_profile", "")
            if p:
                profile_counts[str(p)] += 1
            rh = r.get("selected_rho_H", "")
            if rh != "":
                try:
                    rho_H_counts[float(rh)] += 1
                except (TypeError, ValueError):
                    pass

    lines.append("**Q4: Does H_reg help versus H = I (rho_H = 0.0)?**")
    if rho_H_counts:
        most_common_rho = max(rho_H_counts, key=lambda k: rho_H_counts[k])
        lines.append(
            f"Most-selected rho_H = {most_common_rho}  "
            f"(counts: " + ", ".join(f"{k}→{v}" for k, v in sorted(rho_H_counts.items())) + ")."
        )
        if most_common_rho == 0.0:
            lines.append("The identity (rho_H=0) is preferred — compatibility modeling "
                         "appears not helpful in this run. H_reg ≈ I.")
        else:
            lines.append(f"rho_H > 0 is preferred, suggesting compatibility "
                         "modulation improves validation accuracy.")
    else:
        lines.append("No rho_H data available.")
    lines.append("")

    lines.append("**Q5: Do multi-hop profiles outperform 1-hop only?**")
    if profile_counts:
        best_prof = max(profile_counts, key=lambda k: profile_counts[k])
        lines.append(
            f"Most-selected profile: {best_prof}  "
            "(counts: " + ", ".join(f"{k}→{v}" for k, v in sorted(profile_counts.items())) + ")."
        )
        if best_prof == "profile_A":
            lines.append("1-hop (profile_A) dominates — multi-hop does not appear "
                         "beneficial in this run.")
        else:
            lines.append(f"Multi-hop profile ({best_prof}) is preferred over 1-hop only.")
    else:
        lines.append("No profile data available.")
    lines.append("")

    # Overall assessment
    avg_cav3 = float(np.mean([avg(by_ds[ds]["ca_vs_v3"]) for ds in datasets if ds in by_ds]))
    ca_helped_any = any(
        (avg(by_ds[ds]["ca_delta"]) > 0.001) for ds in datasets if ds in by_ds
    )
    lines.append("**Q7: Is CA-MRD more promising than MS_HSGC?**")
    if ca_helped_any:
        lines.append("CA_MRD achieves positive delta vs MLP on at least one dataset, "
                     "suggesting it engages more than MS_HSGC in light runs. "
                     "Whether this generalises requires a full 10-split run.")
    else:
        lines.append("CA_MRD does not clearly outperform MLP in this light run. "
                     "Both CA_MRD and MS_HSGC appear similarly dormant — "
                     "investigate threshold tuning before concluding.")
    lines.append("")

    lines.append("**Overall verdict:**")
    if avg_cav3 >= -0.001:
        lines.append(
            f"CA_MRD mean delta vs FINAL_V3 = {avg_cav3:+.4f}. "
            "CA_MRD is competitive with FINAL_V3 on average in this light run. "
            "Promising enough to warrant a 10-split follow-up."
        )
    else:
        lines.append(
            f"CA_MRD mean delta vs FINAL_V3 = {avg_cav3:+.4f}. "
            "CA_MRD lags FINAL_V3 in this light run. "
            "Consider loosening the gate or extending the hop grid before a full benchmark."
        )
    lines.append("")

    lines.append("---")
    lines.append("*Non-canonical diagnostic. All canonical FINAL_V3 outputs are untouched.*")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Comparison markdown table
# ---------------------------------------------------------------------------

def _build_comparison_table(delta_rows: List[Dict]) -> str:
    lines = []
    lines.append("# CA-MRD Light Comparison Table")
    lines.append("")
    lines.append("**Status:** Non-canonical.")
    lines.append("")
    lines.append("| Dataset | Split | MLP | FINAL_V3 | CA_MRD | CA−V3 | profile | rho_H |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in delta_rows:
        lines.append(
            f"| {r['dataset']} | {r['split_id']} "
            f"| {_fmt(r['mlp_test_acc'])} "
            f"| {_fmt(r['final_v3_test_acc'])} "
            f"| {_fmt(r['ca_mrd_test_acc'])} "
            f"| {_fmt(r['delta_ca_mrd_vs_v3'], '+.4f')} "
            f"| {r.get('ca_mrd_hop_profile', 'N/A')} "
            f"| {r.get('ca_mrd_rho_H', 'N/A')} |"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CA-MRD evaluation runner")
    parser.add_argument("--split-dir", default="data/splits")
    parser.add_argument(
        "--datasets", nargs="+",
        default=["chameleon", "citeseer", "texas"],
    )
    parser.add_argument("--splits", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--output-tag", default="ca_mrd_light")
    parser.add_argument(
        "--light", action="store_true", default=True,
        help="Use light MLP settings and LIGHT_GRID. Non-canonical.",
    )
    parser.add_argument("--mlp-epochs", type=int, default=None)
    args = parser.parse_args()

    light = args.light
    result_rows, delta_rows = run_evaluation(
        datasets=args.datasets,
        splits=args.splits,
        split_dir=args.split_dir,
        output_tag=args.output_tag,
        light=light,
        mlp_epochs=args.mlp_epochs,
    )

    tag = f"_{args.output_tag}" if args.output_tag else ""
    results_path = f"reports/ca_mrd_results{tag}.csv"
    delta_path = f"reports/ca_mrd_vs_final_v3{tag}.csv"
    summary_path = f"reports/ca_mrd_light_summary{tag}.md"
    table_path = f"tables/ca_mrd_light_comparison{tag}.md"

    print("\n--- Writing outputs ---")
    _write_csv(results_path, result_rows, RESULT_FIELDS)
    _write_csv(delta_path, delta_rows, DELTA_FIELDS)

    summary_md = _build_summary(result_rows, delta_rows, args.datasets, args.splits, light)
    os.makedirs("reports", exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write(summary_md)
    print(f"  Written: {summary_path}")

    os.makedirs("tables", exist_ok=True)
    comparison_md = _build_comparison_table(delta_rows)
    with open(table_path, "w", encoding="utf-8") as fh:
        fh.write(comparison_md)
    print(f"  Written: {table_path}")

    print("\n--- Command used ---")
    print(
        f"python3 code/bfsbased_node_classification/run_ca_mrd_evaluation.py "
        f"{'--light ' if light else ''}"
        f"--datasets {' '.join(args.datasets)} "
        f"--splits {' '.join(str(s) for s in args.splits)} "
        f"--split-dir {args.split_dir} "
        f"--output-tag {args.output_tag}"
    )


if __name__ == "__main__":
    main()

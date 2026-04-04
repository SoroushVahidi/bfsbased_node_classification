#!/usr/bin/env python3
"""
CA-MRC evaluation runner.

Compares MLP_ONLY, FINAL_V3, and CA-MRC (+ optional ablations) across datasets and splits.

Usage (light):
  python3 code/bfsbased_node_classification/run_ca_mrc_evaluation.py \\
    --datasets chameleon citeseer texas \\
    --splits 0 1 \\
    --split-dir data/splits \\
    --light \\
    --output-tag ca_mrc_light

Outputs (light):
  reports/ca_mrc_results_light.csv
  reports/ca_mrc_vs_final_v3_light.csv

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
from typing import Any, Dict, List, Optional

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from final_method_v3 import final_method_v3, _simple_accuracy  # noqa: E402
from ca_mrc import ca_mrc, LIGHT_GRID  # noqa: E402


# ---------------------------------------------------------------------------
# Ablation modes to run by default in light mode
# ---------------------------------------------------------------------------

ABLATION_MODES = [
    "full",          # CA-MRC (compatibility-aware, gated)
    "no_compat",     # plain multi-hop residual, same gate
    "entropy_only",  # gate on entropy reduction only
    "mag_only",      # gate on magnitude only
]


# ---------------------------------------------------------------------------
# Module and data loading (mirrors run_ms_hsgc_evaluation.py)
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
            f"Dataset {ds!r} uses the official split only (split_id must be 0, got {sid})."
        )
    prefix = split_npz_prefix(ds)
    fname = f"{prefix}_split_0.6_0.2_{sid}.npz"
    path = os.path.join(split_dir, fname)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Split not found: {path}")
    sp_data = np.load(path)

    def to_t(mask):
        return torch.as_tensor(np.where(mask)[0], dtype=torch.long, device=device)

    return to_t(sp_data["train_mask"]), to_t(sp_data["val_mask"]), to_t(sp_data["test_mask"])


# ---------------------------------------------------------------------------
# CSV field definitions
# ---------------------------------------------------------------------------

RESULT_FIELDS = [
    "dataset", "split_id", "method", "ablation",
    "test_acc", "val_acc", "delta_vs_mlp",
    "frac_corrected",
    "mean_delta_entropy", "mean_corr_mag",
    "helped_count", "hurt_count", "net_help",
    "frac_corrected_low_margin", "frac_corrected_med_margin", "frac_corrected_high_margin",
    "alpha1", "alpha2", "lambda_corr", "tau_H", "tau_R",
    "runtime_sec",
]

DELTA_FIELDS = [
    "dataset", "split_id", "mlp_test_acc", "final_v3_test_acc",
    "ca_mrc_test_acc", "delta_ca_mrc_vs_final_v3",
]


# ---------------------------------------------------------------------------
# MLP kwargs
# ---------------------------------------------------------------------------

LIGHT_MLP_KWARGS = {
    "hidden": 64, "layers": 2, "dropout": 0.5, "lr": 0.01, "epochs": 100,
}


# ---------------------------------------------------------------------------
# Helper: build a result row
# ---------------------------------------------------------------------------

def _base_row(ds, sid, method, ablation, test_acc, val_acc, delta):
    return {f: "" for f in RESULT_FIELDS} | {
        "dataset": ds, "split_id": sid,
        "method": method, "ablation": ablation,
        "test_acc": test_acc, "val_acc": val_acc, "delta_vs_mlp": delta,
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    datasets: List[str],
    splits: List[int],
    split_dir: str,
    light: bool,
    ablations: List[str],
    save_compat: bool,
    output_tag: str,
) -> tuple:
    mlp_kwargs = LIGHT_MLP_KWARGS if light else None  # None → DEFAULT_MANUSCRIPT_MLP_KWARGS

    if light:
        print("\n*** LIGHT MODE ACTIVE — non-canonical, lightweight run ***")
        print(f"    MLP epochs: {LIGHT_MLP_KWARGS['epochs']}  (canonical: 300)  "
              f"Grid configs: {len(LIGHT_GRID['alpha2']) * len(LIGHT_GRID['lambda_corr']) * len(LIGHT_GRID['tau_H']) * len(LIGHT_GRID['tau_R'])} "
              f"× {len(LIGHT_GRID['alpha1'])} alpha1 × {len(LIGHT_GRID['alpha2'])} alpha2\n")

    mod = _load_module()
    records: List[Dict] = []
    delta_records: List[Dict] = []

    for ds in datasets:
        print(f"\n{'='*62}\nDataset: {ds}\n{'='*62}")
        try:
            data = _load_data(mod, ds)
        except Exception as exc:
            print(f"  [ERROR loading {ds}] {exc}")
            continue

        for sid in splits:
            try:
                train_idx, val_idx, test_idx = _load_split(ds, sid, data.x.device, split_dir)
            except FileNotFoundError as exc:
                print(f"  [SKIP] {exc}")
                continue

            train_np = mod._to_numpy_idx(train_idx)
            val_np   = mod._to_numpy_idx(val_idx)
            test_np  = mod._to_numpy_idx(test_idx)
            seed = (1337 + sid) * 100
            torch.manual_seed(seed)
            np.random.seed(seed)

            y_true = data.y.detach().cpu().numpy().astype(np.int64)
            print(f"  split={sid} seed={seed}")

            # ---- MLP ----
            eff_kw = mlp_kwargs if mlp_kwargs else mod.DEFAULT_MANUSCRIPT_MLP_KWARGS
            mlp_probs, _ = mod.train_mlp_and_predict(
                data, train_idx, **eff_kw, log_file=None,
            )
            mlp_info = mod.compute_mlp_margin(mlp_probs)
            mlp_pred = mlp_info["mlp_pred_all"]
            mlp_test_acc = _simple_accuracy(y_true[test_np], mlp_pred[test_np])

            row = _base_row(ds, sid, "MLP_ONLY", "", mlp_test_acc, None, 0.0)
            records.append(row)
            print(f"    MLP:          {mlp_test_acc:.4f}")

            # ---- FINAL_V3 ----
            t0 = time.perf_counter()
            v3_val, v3_test_acc, v3_info = final_method_v3(
                data, train_np, val_np, test_np,
                mlp_probs=mlp_probs, seed=seed, mod=mod,
                gate="heuristic", split_id=sid,
            )
            v3_time = time.perf_counter() - t0
            v3_delta = v3_test_acc - mlp_test_acc
            ca_v3 = v3_info["correction_analysis"]

            row = _base_row(ds, sid, "FINAL_V3", "", v3_test_acc, v3_val, v3_delta)
            row.update({
                "helped_count": ca_v3.get("n_helped", ""),
                "hurt_count":   ca_v3.get("n_hurt", ""),
                "runtime_sec":  v3_time,
            })
            records.append(row)
            print(f"    FINAL_V3:     {v3_test_acc:.4f}  (delta={v3_delta:+.4f})")

            # ---- CA-MRC ablations ----
            ca_mrc_full_acc = None  # used for delta CSV
            compat_dir = os.path.join("reports", "ca_mrc_compat_matrices", ds) if save_compat else None

            for abl in ablations:
                compat_path = None
                if save_compat and compat_dir and abl == "full":
                    os.makedirs(compat_dir, exist_ok=True)
                    compat_path = os.path.join(compat_dir, f"split_{sid}_C.csv")

                t0 = time.perf_counter()
                ca_val, ca_test_acc, ca_info = ca_mrc(
                    data, train_np, val_np, test_np,
                    mlp_probs=mlp_probs, seed=seed, mod=mod,
                    mlp_kwargs=eff_kw,
                    grid=LIGHT_GRID,
                    ablation=abl,
                    save_compat_matrix_path=compat_path,
                )
                ca_time = time.perf_counter() - t0
                ca_delta = ca_test_acc - mlp_test_acc

                row = _base_row(ds, sid, "CA_MRC", abl, ca_test_acc, ca_val, ca_delta)
                row.update({
                    "frac_corrected": ca_info["frac_corrected"],
                    "mean_delta_entropy": ca_info["mean_delta_entropy"],
                    "mean_corr_mag": ca_info["mean_corr_mag"],
                    "helped_count": ca_info["helped_count"],
                    "hurt_count": ca_info["hurt_count"],
                    "net_help": ca_info["net_help"],
                    "frac_corrected_low_margin": ca_info["frac_corrected_low_margin"],
                    "frac_corrected_med_margin": ca_info["frac_corrected_med_margin"],
                    "frac_corrected_high_margin": ca_info["frac_corrected_high_margin"],
                    "alpha1": ca_info["alpha1"],
                    "alpha2": ca_info["alpha2"],
                    "lambda_corr": ca_info["lambda_corr"],
                    "tau_H": ca_info["tau_H"],
                    "tau_R": ca_info["tau_R"],
                    "runtime_sec": ca_time,
                })
                records.append(row)
                print(
                    f"    CA_MRC[{abl:<14s}] {ca_test_acc:.4f}  (delta={ca_delta:+.4f})"
                    f"  [frac_corr={ca_info['frac_corrected']:.3f}"
                    f" helped={ca_info['helped_count']} hurt={ca_info['hurt_count']}"
                    f" net={ca_info['net_help']:+d}]"
                )

                if abl == "full":
                    ca_mrc_full_acc = ca_test_acc

            # ---- Delta record (CA_MRC full vs FINAL_V3) ----
            if ca_mrc_full_acc is not None:
                delta_records.append({
                    "dataset": ds,
                    "split_id": sid,
                    "mlp_test_acc": mlp_test_acc,
                    "final_v3_test_acc": v3_test_acc,
                    "ca_mrc_test_acc": ca_mrc_full_acc,
                    "delta_ca_mrc_vs_final_v3": ca_mrc_full_acc - v3_test_acc,
                })

    return records, delta_records


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def write_results_csv(records: List[Dict], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in RESULT_FIELDS})
    print(f"\nResults written: {path}")


def write_delta_csv(delta_records: List[Dict], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=DELTA_FIELDS)
        w.writeheader()
        for r in delta_records:
            w.writerow({k: r.get(k, "") for k in DELTA_FIELDS})
    print(f"Delta CSV written: {path}")


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_summary(records: List[Dict], delta_records: List[Dict]):
    datasets = sorted(set(r["dataset"] for r in records))

    by_method_ds: Dict = defaultdict(lambda: defaultdict(list))
    by_method_ds_delta: Dict = defaultdict(lambda: defaultdict(list))

    for r in records:
        key = r["method"] if r["method"] != "CA_MRC" else f"CA_MRC[{r['ablation']}]"
        ta = r.get("test_acc")
        if ta not in (None, ""):
            by_method_ds[key][r["dataset"]].append(float(ta))
        dv = r.get("delta_vs_mlp")
        if dv not in (None, ""):
            by_method_ds_delta[key][r["dataset"]].append(float(dv))

    methods_order = (
        ["MLP_ONLY", "FINAL_V3"]
        + [f"CA_MRC[{a}]" for a in ABLATION_MODES]
    )

    def _hdr():
        return f"{'Method':<26s}" + "".join(f"{ds:>14s}" for ds in datasets) + f"{'Mean':>10s}"

    print(f"\n{'='*90}")
    print("MEAN TEST ACCURACY")
    print(f"{'='*90}")
    hdr = _hdr()
    print(hdr)
    print("-" * len(hdr))
    for m in methods_order:
        if m not in by_method_ds:
            continue
        parts = [f"{m:<26s}"]
        vals = []
        for ds in datasets:
            accs = by_method_ds[m][ds]
            if accs:
                mu = float(np.mean(accs))
                parts.append(f"{mu:>13.4f}")
                vals.append(mu)
            else:
                parts.append(f"{'N/A':>13s}")
        parts.append(f"{float(np.mean(vals)):>9.4f}" if vals else "")
        print("".join(parts))

    print(f"\n{'='*90}")
    print("MEAN DELTA vs MLP")
    print(f"{'='*90}")
    print(hdr)
    print("-" * len(hdr))
    for m in methods_order:
        if m not in by_method_ds_delta:
            continue
        parts = [f"{m:<26s}"]
        vals = []
        for ds in datasets:
            deltas = by_method_ds_delta[m][ds]
            if deltas:
                mu = float(np.mean(deltas))
                parts.append(f"{mu:>+13.4f}")
                vals.append(mu)
            else:
                parts.append(f"{'N/A':>13s}")
        parts.append(f"{float(np.mean(vals)):>+9.4f}" if vals else "")
        print("".join(parts))

    if delta_records:
        deltas_all = [float(r["delta_ca_mrc_vs_final_v3"]) for r in delta_records
                      if r.get("delta_ca_mrc_vs_final_v3") not in (None, "")]
        if deltas_all:
            print(f"\n{'='*90}")
            print("CA_MRC[full] vs FINAL_V3 DELTA")
            print(f"{'='*90}")
            by_ds: Dict = defaultdict(list)
            for r in delta_records:
                d = r.get("delta_ca_mrc_vs_final_v3")
                if d not in (None, ""):
                    by_ds[r["dataset"]].append(float(d))
            for ds in datasets:
                if by_ds[ds]:
                    mu = float(np.mean(by_ds[ds]))
                    print(f"  {ds:<14s} mean delta = {mu:+.4f}")
            print(f"  Overall mean delta : {float(np.mean(deltas_all)):+.4f}")
    print(f"{'='*90}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CA-MRC evaluation runner")
    parser.add_argument("--split-dir", default="data/splits")
    parser.add_argument(
        "--datasets", nargs="+",
        default=["cora", "citeseer", "pubmed", "chameleon", "texas", "wisconsin"],
    )
    parser.add_argument("--splits", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--output-tag", default="ca_mrc")
    parser.add_argument(
        "--light", action="store_true",
        help="Lightweight mode: reduced MLP epochs + light grid. Non-canonical.",
    )
    parser.add_argument(
        "--ablations", nargs="+", default=ABLATION_MODES,
        help="Ablation modes to run. Choices: full no_compat entropy_only mag_only always_apply",
    )
    parser.add_argument(
        "--save-compat", action="store_true",
        help="Save estimated compatibility matrices to reports/ca_mrc_compat_matrices/",
    )
    args = parser.parse_args()

    records, delta_records = run_evaluation(
        args.datasets, args.splits, args.split_dir,
        light=args.light,
        ablations=args.ablations,
        save_compat=args.save_compat,
        output_tag=args.output_tag,
    )

    tag = args.output_tag
    if args.light:
        suffix = "_light"
    elif tag != "ca_mrc":
        suffix = f"_{tag}"
    else:
        suffix = ""
    results_path = f"reports/ca_mrc_results{suffix}.csv"
    delta_path = f"reports/ca_mrc_vs_final_v3{suffix}.csv"

    write_results_csv(records, results_path)
    write_delta_csv(delta_records, delta_path)
    print_summary(records, delta_records)


if __name__ == "__main__":
    main()

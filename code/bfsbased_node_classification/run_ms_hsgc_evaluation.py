#!/usr/bin/env python3
"""
MS_HSGC evaluation runner.

Compares MLP_ONLY, FINAL_V3, and MS_HSGC across datasets and splits.

Usage:
  python3 code/bfsbased_node_classification/run_ms_hsgc_evaluation.py \\
    --split-dir data/splits \\
    --datasets cora citeseer pubmed chameleon texas wisconsin \\
    --splits 0 1 2 3 4 5 6 7 8 9 \\
    --output-tag ms_hsgc

Outputs:
  reports/ms_hsgc_results.csv
  reports/ms_hsgc_vs_final_v3.csv

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
from typing import Dict, List

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from final_method_v3 import final_method_v3, _simple_accuracy  # noqa: E402
from ms_hsgc import ms_hsgc  # noqa: E402


# ---------------------------------------------------------------------------
# Module and data loading (mirrors run_final_evaluation.py)
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
    sp = np.load(path)

    def to_t(mask):
        return torch.as_tensor(np.where(mask)[0], dtype=torch.long, device=device)

    return to_t(sp["train_mask"]), to_t(sp["val_mask"]), to_t(sp["test_mask"])


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

MS_HSGC_RESULT_FIELDS = [
    "dataset", "split_id", "method", "test_acc", "val_acc", "delta_vs_mlp",
    "frac_confident", "frac_corrected_1hop", "frac_corrected_2hop",
    "frac_mlp_only_uncertain",
    "mean_H1", "mean_H2", "mean_DeltaH",
    "n_helped", "n_hurt", "correction_precision",
    "selected_tau", "selected_rho1", "selected_rho2",
    "selected_h1_max", "selected_delta_min",
    "selected_profile_1hop", "selected_profile_2hop",
    "runtime_sec",
]

DELTA_FIELDS = [
    "dataset", "split_id", "mlp_test_acc", "final_v3_test_acc", "ms_hsgc_test_acc",
    "delta_ms_hsgc_vs_final_v3",
]


def run_evaluation(
    datasets: List[str],
    splits: List[int],
    split_dir: str,
    output_tag: str,
) -> tuple:
    mod = _load_module()
    records: List[Dict] = []
    delta_records: List[Dict] = []

    for ds in datasets:
        print(f"\n{'='*60}\nDataset: {ds}\n{'='*60}")
        try:
            data = _load_data(mod, ds)
        except Exception as e:
            print(f"  [ERROR loading {ds}] {e}")
            continue

        for sid in splits:
            try:
                train_idx, val_idx, test_idx = _load_split(
                    ds, sid, data.x.device, split_dir
                )
            except FileNotFoundError as e:
                print(f"  [SKIP] {e}")
                continue

            train_np = mod._to_numpy_idx(train_idx)
            val_np = mod._to_numpy_idx(val_idx)
            test_np = mod._to_numpy_idx(test_idx)
            seed = (1337 + sid) * 100
            torch.manual_seed(seed)
            np.random.seed(seed)

            y_true = data.y.detach().cpu().numpy().astype(np.int64)
            print(f"  split={sid} seed={seed}")

            # ---- MLP baseline ----
            mlp_probs, _ = mod.train_mlp_and_predict(
                data, train_idx,
                **mod.DEFAULT_MANUSCRIPT_MLP_KWARGS, log_file=None,
            )
            mlp_info = mod.compute_mlp_margin(mlp_probs)
            mlp_pred = mlp_info["mlp_pred_all"]
            mlp_test_acc = _simple_accuracy(y_true[test_np], mlp_pred[test_np])

            mlp_row = _base_row(ds, sid, "MLP_ONLY", mlp_test_acc, None, 0.0)
            records.append(mlp_row)
            print(f"    MLP:        {mlp_test_acc:.4f}")

            # ---- FINAL_V3 ----
            t0 = time.perf_counter()
            v3_val, v3_test_acc, v3_info = final_method_v3(
                data, train_np, val_np, test_np,
                mlp_probs=mlp_probs, seed=seed, mod=mod,
                gate="heuristic", split_id=sid,
            )
            v3_time = time.perf_counter() - t0
            v3_delta = v3_test_acc - mlp_test_acc
            ca = v3_info["correction_analysis"]
            bf = v3_info["branch_fractions"]

            v3_row = _base_row(ds, sid, "FINAL_V3", v3_test_acc, v3_val, v3_delta)
            v3_row.update({
                "frac_confident": bf.get("confident_keep_mlp"),
                "n_helped": ca.get("n_helped"),
                "n_hurt": ca.get("n_hurt"),
                "correction_precision": ca.get("correction_precision"),
                "runtime_sec": v3_time,
            })
            records.append(v3_row)
            print(f"    FINAL_V3:   {v3_test_acc:.4f}  (delta={v3_delta:+.4f})")

            # ---- MS_HSGC ----
            t0 = time.perf_counter()
            ms_val, ms_test_acc, ms_info = ms_hsgc(
                data, train_np, val_np, test_np,
                mlp_probs=mlp_probs, seed=seed, mod=mod,
            )
            ms_time = time.perf_counter() - t0
            ms_delta = ms_test_acc - mlp_test_acc

            ms_row = _base_row(ds, sid, "MS_HSGC", ms_test_acc, ms_val, ms_delta)
            ms_row.update({
                "frac_confident": ms_info["frac_confident"],
                "frac_corrected_1hop": ms_info["frac_corrected_1hop"],
                "frac_corrected_2hop": ms_info["frac_corrected_2hop"],
                "frac_mlp_only_uncertain": ms_info["frac_mlp_only_uncertain"],
                "mean_H1": ms_info["mean_H1"],
                "mean_H2": ms_info["mean_H2"],
                "mean_DeltaH": ms_info["mean_DeltaH"],
                "n_helped": ms_info["n_helped"],
                "n_hurt": ms_info["n_hurt"],
                "correction_precision": ms_info["correction_precision"],
                "selected_tau": ms_info["selected_tau"],
                "selected_rho1": ms_info["selected_rho1"],
                "selected_rho2": ms_info["selected_rho2"],
                "selected_h1_max": ms_info["selected_h1_max"],
                "selected_delta_min": ms_info["selected_delta_min"],
                "selected_profile_1hop": ms_info["selected_profile_1hop"],
                "selected_profile_2hop": ms_info["selected_profile_2hop"],
                "runtime_sec": ms_time,
            })
            records.append(ms_row)
            print(
                f"    MS_HSGC:    {ms_test_acc:.4f}  (delta={ms_delta:+.4f})  "
                f"[helped={ms_info['n_helped']} hurt={ms_info['n_hurt']} "
                f"prec={ms_info['correction_precision']:.2f} "
                f"1h={ms_info['frac_corrected_1hop']:.2f} "
                f"2h={ms_info['frac_corrected_2hop']:.2f}]"
            )

            # ---- Delta record ----
            delta_records.append({
                "dataset": ds,
                "split_id": sid,
                "mlp_test_acc": mlp_test_acc,
                "final_v3_test_acc": v3_test_acc,
                "ms_hsgc_test_acc": ms_test_acc,
                "delta_ms_hsgc_vs_final_v3": ms_test_acc - v3_test_acc,
            })

    return records, delta_records


def _base_row(ds, sid, method, test_acc, val_acc, delta):
    return {
        "dataset": ds, "split_id": sid, "method": method,
        "test_acc": test_acc, "val_acc": val_acc, "delta_vs_mlp": delta,
        "frac_confident": "", "frac_corrected_1hop": "", "frac_corrected_2hop": "",
        "frac_mlp_only_uncertain": "", "mean_H1": "", "mean_H2": "", "mean_DeltaH": "",
        "n_helped": "", "n_hurt": "", "correction_precision": "",
        "selected_tau": "", "selected_rho1": "", "selected_rho2": "",
        "selected_h1_max": "", "selected_delta_min": "",
        "selected_profile_1hop": "", "selected_profile_2hop": "",
        "runtime_sec": "",
    }


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def write_results_csv(records: List[Dict], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MS_HSGC_RESULT_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in MS_HSGC_RESULT_FIELDS})
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
    methods = ["MLP_ONLY", "FINAL_V3", "MS_HSGC"]
    datasets = sorted(set(r["dataset"] for r in records))

    by_md: Dict = defaultdict(lambda: defaultdict(list))
    by_md_delta: Dict = defaultdict(lambda: defaultdict(list))

    for r in records:
        ta = r.get("test_acc")
        if ta not in (None, ""):
            by_md[r["method"]][r["dataset"]].append(float(ta))
        dv = r.get("delta_vs_mlp")
        if dv not in (None, ""):
            by_md_delta[r["method"]][r["dataset"]].append(float(dv))

    def _hdr():
        return f"{'Method':<20s}" + "".join(f"{ds:>14s}" for ds in datasets) + f"{'Mean':>10s}"

    print(f"\n{'='*90}")
    print("MEAN TEST ACCURACY")
    print(f"{'='*90}")
    hdr = _hdr()
    print(hdr)
    print("-" * len(hdr))
    for m in methods:
        parts = [f"{m:<20s}"]
        vals = []
        for ds in datasets:
            accs = by_md[m][ds]
            if accs:
                mu = float(np.mean(accs))
                parts.append(f"{mu:>13.4f}")
                vals.append(mu)
            else:
                parts.append(f"{'N/A':>13s}")
        parts.append(f"{float(np.mean(vals)):>9.4f}" if vals else f"{'N/A':>9s}")
        print("".join(parts))

    print(f"\n{'='*90}")
    print("MEAN DELTA vs MLP")
    print(f"{'='*90}")
    hdr = _hdr()
    print(hdr)
    print("-" * len(hdr))
    for m in methods:
        parts = [f"{m:<20s}"]
        vals = []
        for ds in datasets:
            deltas = by_md_delta[m][ds]
            if deltas:
                mu = float(np.mean(deltas))
                parts.append(f"{mu:>+13.4f}")
                vals.append(mu)
            else:
                parts.append(f"{'N/A':>13s}")
        parts.append(f"{float(np.mean(vals)):>+9.4f}" if vals else f"{'N/A':>9s}")
        print("".join(parts))

    if delta_records:
        deltas_all = [r["delta_ms_hsgc_vs_final_v3"] for r in delta_records
                      if r.get("delta_ms_hsgc_vs_final_v3") not in (None, "")]
        if deltas_all:
            print(f"\n{'='*90}")
            print("MS_HSGC vs FINAL_V3 DELTA")
            print(f"{'='*90}")
            by_ds: Dict = defaultdict(list)
            for r in delta_records:
                d = r.get("delta_ms_hsgc_vs_final_v3")
                if d not in (None, ""):
                    by_ds[r["dataset"]].append(float(d))
            for ds in datasets:
                if by_ds[ds]:
                    mu = float(np.mean(by_ds[ds]))
                    print(f"  {ds:<12s} mean delta = {mu:+.4f}")
            print(f"  Overall mean delta: {float(np.mean(deltas_all)):+.4f}")

    print(f"{'='*90}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MS_HSGC evaluation runner")
    parser.add_argument("--split-dir", default="data/splits")
    parser.add_argument(
        "--datasets", nargs="+",
        default=["cora", "citeseer", "pubmed", "chameleon", "texas", "wisconsin"],
    )
    parser.add_argument("--splits", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--output-tag", default="ms_hsgc")
    args = parser.parse_args()

    records, delta_records = run_evaluation(
        args.datasets, args.splits, args.split_dir, args.output_tag
    )

    tag = args.output_tag
    suffix = f"_{tag}" if tag != "ms_hsgc" else ""
    results_path = f"reports/ms_hsgc_results{suffix}.csv"
    delta_path = f"reports/ms_hsgc_vs_final_v3{suffix}.csv"

    write_results_csv(records, results_path)
    write_delta_csv(delta_records, delta_path)
    print_summary(records, delta_records)


if __name__ == "__main__":
    main()

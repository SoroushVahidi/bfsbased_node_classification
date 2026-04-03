#!/usr/bin/env python3
"""
Final evaluation runner: compares MLP, baseline SGC (v1), V2_MULTIBRANCH, and FINAL_V3.

Usage:
  python3 code/bfsbased_node_classification/run_final_evaluation.py \
    --split-dir data/splits

Default: **10 splits** (indices 0–9), matching `reports/final_method_v3_results.csv`.

If `--output-tag` is left at the default `final_v3`, the runner overwrites the
canonical path `reports/final_method_v3_results.csv`. Supplying a different tag
creates `reports/final_method_v3_results_<tag>.csv` instead.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
import time
from collections import defaultdict
from typing import List, Optional

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from final_method_v3 import final_method_v3, _simple_accuracy  # noqa: E402
from improved_sgc_variants import v2_multibranch_correction  # noqa: E402


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
    mod = importlib.util.module_from_spec(
        importlib.util.spec_from_loader("bfs_full_investigate", loader=None)
    )
    setattr(mod, "__file__", path)
    setattr(mod, "DATASET_KEY", "texas")
    exec(compile(source, path, "exec"), mod.__dict__)
    return mod


def _load_data(mod, ds, root="data/"):
    dataset = mod.load_dataset(ds, root)
    data = dataset[0]
    data.edge_index = torch.cat([data.edge_index, data.edge_index.flip(0)], dim=1)
    data.edge_index = torch.unique(data.edge_index, dim=1)
    return data


def _load_split(ds, sid, device, split_dir):
    prefix = "film" if ds == "actor" else ds.lower()
    fname = f"{prefix}_split_0.6_0.2_{sid}.npz"
    path = os.path.join(split_dir, fname)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Split not found: {path}")
    sp = np.load(path)

    def to_t(mask):
        return torch.as_tensor(np.where(mask)[0], dtype=torch.long, device=device)

    return to_t(sp["train_mask"]), to_t(sp["val_mask"]), to_t(sp["test_mask"])


def run_evaluation(
    datasets,
    splits,
    split_dir,
    output_tag,
    *,
    gate_mode: str,
):
    mod = _load_module()
    records = []

    for ds in datasets:
        print(f"\n{'='*60}\nDataset: {ds}\n{'='*60}")
        try:
            data = _load_data(mod, ds)
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue

        for sid in splits:
            try:
                train_idx, val_idx, test_idx = _load_split(ds, sid, data.x.device, split_dir)
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

            # --- MLP baseline ---
            mlp_probs, _ = mod.train_mlp_and_predict(
                data, train_idx,
                **mod.DEFAULT_MANUSCRIPT_MLP_KWARGS, log_file=None,
            )
            mlp_info = mod.compute_mlp_margin(mlp_probs)
            mlp_pred = mlp_info["mlp_pred_all"]
            mlp_test_acc = _simple_accuracy(y_true[test_np], mlp_pred[test_np])

            records.append({
                "dataset": ds, "split_id": sid, "seed": seed,
                "method": "MLP_ONLY",
                "test_acc": mlp_test_acc, "delta_vs_mlp": 0.0,
                "n_helped": 0, "n_hurt": 0, "correction_precision": 1.0,
                "frac_confident": 1.0, "frac_unreliable": 0.0, "frac_corrected": 0.0,
                "tau": None, "rho": None, "runtime_sec": 0.0,
            })
            print(f"    MLP:         {mlp_test_acc:.4f}")

            # --- Baseline SGC (v1) ---
            t0 = time.perf_counter()
            _, v1_acc, v1_info = mod.selective_graph_correction_predictclass(
                data, train_np, val_np, test_np,
                mlp_probs=mlp_probs, seed=seed, log_file=None,
                write_node_diagnostics=False,
            )
            v1_time = time.perf_counter() - t0

            v1_thr = v1_info.get("selected_threshold_high", 0.2)
            v1_delta = v1_acc - mlp_test_acc
            records.append({
                "dataset": ds, "split_id": sid, "seed": seed,
                "method": "BASELINE_SGC_V1",
                "test_acc": v1_acc, "delta_vs_mlp": v1_delta,
                "n_helped": None, "n_hurt": None, "correction_precision": None,
                "frac_confident": None,
                "frac_unreliable": None,
                # For the legacy v1 baseline we only log uncertain-node coverage;
                # the column name is kept for CSV compatibility with older tables.
                "frac_corrected": float(v1_info.get("fraction_test_nodes_uncertain", 0)),
                "tau": v1_thr, "rho": None,
                "runtime_sec": v1_time,
            })
            print(f"    SGC_V1:      {v1_acc:.4f}  (delta={v1_delta:+.4f})")

            # --- V2_MULTIBRANCH ---
            t0 = time.perf_counter()
            v2_val, v2_acc, v2_info = v2_multibranch_correction(
                data, train_np, val_np, test_np,
                mlp_probs=mlp_probs, seed=seed, mod=mod,
            )
            v2_time = time.perf_counter() - t0
            v2_delta = v2_acc - mlp_test_acc
            records.append({
                "dataset": ds, "split_id": sid, "seed": seed,
                "method": "V2_MULTIBRANCH",
                "test_acc": v2_acc, "delta_vs_mlp": v2_delta,
                "n_helped": None, "n_hurt": None, "correction_precision": None,
                "frac_confident": v2_info.get("fraction_test_no_correct"),
                "frac_unreliable": v2_info.get("fraction_test_feat_branch"),
                "frac_corrected": v2_info.get("fraction_test_full_branch"),
                "tau": v2_info.get("selected_threshold"),
                "rho": v2_info.get("selected_reliability_threshold"),
                "runtime_sec": v2_time,
            })
            print(f"    V2_MULTI:    {v2_acc:.4f}  (delta={v2_delta:+.4f})")

            gates_to_run = [gate_mode] if gate_mode != "both" else ["heuristic", "adaptive"]
            for gate_name in gates_to_run:
                t0 = time.perf_counter()
                _, v3_acc, v3_info = final_method_v3(
                    data, train_np, val_np, test_np,
                    mlp_probs=mlp_probs, seed=seed, mod=mod,
                    gate=gate_name, split_id=sid,
                )
                v3_time = time.perf_counter() - t0
                v3_delta = v3_acc - mlp_test_acc
                ca = v3_info["correction_analysis"]
                bf = v3_info["branch_fractions"]
                method_name = "FINAL_V3" if gate_mode != "both" else f"FINAL_V3_{gate_name.upper()}"
                records.append({
                    "dataset": ds, "split_id": sid, "seed": seed,
                    "method": method_name,
                    "test_acc": v3_acc, "delta_vs_mlp": v3_delta,
                    "n_helped": ca["n_helped"], "n_hurt": ca["n_hurt"],
                    "correction_precision": ca["correction_precision"],
                    "frac_confident": bf["confident_keep_mlp"],
                    "frac_unreliable": bf["uncertain_unreliable_keep_mlp"],
                    "frac_corrected": bf["uncertain_reliable_corrected"],
                    "tau": v3_info.get("selected_tau"),
                    "rho": v3_info["selected_rho"],
                    "gate_mode": v3_info.get("gate_mode"),
                    "gate_uncertain_frac": v3_info.get("gate_stats", {}).get("uncertain_fraction_test"),
                    "gate_bce_loss": v3_info.get("gate_stats", {}).get("gate_bce_loss"),
                    "runtime_sec": v3_time,
                })
                print(f"    {method_name:<18s} {v3_acc:.4f}  (delta={v3_delta:+.4f})  "
                      f"[helped={ca['n_helped']} hurt={ca['n_hurt']} prec={ca['correction_precision']:.2f}]")

    return records


def write_csv(records, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = [
        "dataset", "split_id", "seed", "method", "test_acc", "delta_vs_mlp",
        "n_helped", "n_hurt", "correction_precision",
        "frac_confident", "frac_unreliable", "frac_corrected",
        "tau", "rho", "gate_mode", "gate_uncertain_frac", "gate_bce_loss", "runtime_sec",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"\nResults: {path}")


def print_summary(records):
    methods = sorted(set(r["method"] for r in records), key=lambda x: (
        ["MLP_ONLY", "BASELINE_SGC_V1", "V2_MULTIBRANCH"].index(x)
        if x in ["MLP_ONLY", "BASELINE_SGC_V1", "V2_MULTIBRANCH"] else 99, x
    ))
    datasets = sorted(set(r["dataset"] for r in records))

    by_md = defaultdict(lambda: defaultdict(list))
    by_md_delta = defaultdict(lambda: defaultdict(list))
    by_md_hurt = defaultdict(lambda: defaultdict(list))

    for r in records:
        if r["test_acc"] is not None:
            by_md[r["method"]][r["dataset"]].append(r["test_acc"])
            by_md_delta[r["method"]][r["dataset"]].append(r["delta_vs_mlp"])
        if r.get("n_hurt") is not None:
            by_md_hurt[r["method"]][r["dataset"]].append(r["n_hurt"])

    print(f"\n{'='*90}")
    print("MEAN TEST ACCURACY")
    print(f"{'='*90}")

    # Win/Tie/Loss vs MLP by split + routed fraction
    print("\n" + "=" * 90)
    print("WIN/TIE/LOSS vs MLP")
    print("=" * 90)
    gate_methods = [m for m in methods if m.startswith("FINAL_V3")]
    for m in gate_methods:
        print(f"  {m}:")
        for ds in datasets:
            mlp = [r for r in records if r["dataset"] == ds and r["method"] == "MLP_ONLY"]
            cur = [r for r in records if r["dataset"] == ds and r["method"] == m]
            by_split_mlp = {r["split_id"]: r["test_acc"] for r in mlp}
            by_split_cur = {r["split_id"]: r["test_acc"] for r in cur}
            wins = ties = losses = 0
            for sid, cur_acc in by_split_cur.items():
                mlp_acc = by_split_mlp.get(sid)
                if mlp_acc is None:
                    continue
                if cur_acc > mlp_acc:
                    wins += 1
                elif cur_acc < mlp_acc:
                    losses += 1
                else:
                    ties += 1
            print(f"    {ds:<10s} W/T/L = {wins}/{ties}/{losses}")
        routed = [r.get("gate_uncertain_frac") for r in records if r["method"] == m and r.get("gate_uncertain_frac") is not None]
        if routed:
            print(f"    mean routed fraction: {100.0 * float(np.mean(routed)):.2f}%")
    hdr = f"{'Method':<20s}" + "".join(f"{ds:>14s}" for ds in datasets) + f"{'Mean':>10s}"
    print(hdr)
    print("-" * len(hdr))
    for m in methods:
        parts = [f"{m:<20s}"]
        vals = []
        for ds in datasets:
            accs = by_md[m][ds]
            if accs:
                mu = np.mean(accs)
                parts.append(f"{mu:>13.4f}")
                vals.append(mu)
            else:
                parts.append(f"{'N/A':>13s}")
        parts.append(f"{np.mean(vals):>9.4f}" if vals else f"{'N/A':>9s}")
        print("".join(parts))

    print(f"\n{'='*90}")
    print("MEAN DELTA vs MLP")
    print(f"{'='*90}")
    hdr = f"{'Method':<20s}" + "".join(f"{ds:>14s}" for ds in datasets) + f"{'Mean':>10s}"
    print(hdr)
    print("-" * len(hdr))
    for m in methods:
        parts = [f"{m:<20s}"]
        vals = []
        for ds in datasets:
            deltas = by_md_delta[m][ds]
            if deltas:
                mu = np.mean(deltas)
                parts.append(f"{mu:>+13.4f}")
                vals.append(mu)
            else:
                parts.append(f"{'N/A':>13s}")
        parts.append(f"{np.mean(vals):>+9.4f}" if vals else f"{'N/A':>9s}")
        print("".join(parts))

    print(f"\n{'='*90}")
    print("TOTAL HARMFUL CORRECTIONS (n_hurt sum across splits)")
    print(f"{'='*90}")
    for m in gate_methods:
        for ds in datasets:
            hurts = by_md_hurt[m][ds]
            if hurts:
                print(f"  {m} on {ds}: {sum(hurts)} hurt across {len(hurts)} splits "
                      f"(mean {np.mean(hurts):.1f}/split)")

    # Variance analysis
    print(f"\n{'='*90}")
    print("STD DEV OF TEST ACCURACY ACROSS SPLITS")
    print(f"{'='*90}")
    hdr = f"{'Method':<20s}" + "".join(f"{ds:>14s}" for ds in datasets)
    print(hdr)
    print("-" * len(hdr))
    for m in methods:
        parts = [f"{m:<20s}"]
        for ds in datasets:
            accs = by_md[m][ds]
            if len(accs) > 1:
                parts.append(f"{np.std(accs):>13.4f}")
            else:
                parts.append(f"{'N/A':>13s}")
        print("".join(parts))
    print(f"{'='*90}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", default="data/splits")
    parser.add_argument("--datasets", nargs="+",
                        default=["cora", "citeseer", "pubmed", "chameleon", "texas", "wisconsin"])
    parser.add_argument(
        "--splits", nargs="+", type=int,
        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    )
    parser.add_argument("--output-tag", default="final_v3")
    parser.add_argument("--gate", choices=["heuristic", "adaptive", "both"], default="heuristic")
    args = parser.parse_args()

    records = run_evaluation(
        args.datasets,
        args.splits,
        args.split_dir,
        args.output_tag,
        gate_mode=args.gate,
    )

    csv_path = "reports/final_method_v3_results.csv"
    if args.output_tag != "final_v3":
        csv_path = f"reports/final_method_v3_results_{args.output_tag}.csv"
    write_csv(records, csv_path)
    print_summary(records)


if __name__ == "__main__":
    main()

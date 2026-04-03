#!/usr/bin/env python3
"""Targeted evaluation of a local-homophily-gated FINAL_V3 variant."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from final_method_v3 import final_method_v3

REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


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


def _load_data(mod, ds: str):
    dataset = mod.load_dataset(ds, "data/")
    data = dataset[0]
    data.edge_index = torch.cat([data.edge_index, data.edge_index.flip(0)], dim=1)
    data.edge_index = torch.unique(data.edge_index, dim=1)
    return data


def _load_split(ds: str, sid: int, device, split_dir: str):
    prefix = "film" if ds == "actor" else ds.lower()
    path = os.path.join(split_dir, f"{prefix}_split_0.6_0.2_{sid}.npz")
    sp = np.load(path)
    to_t = lambda m: torch.as_tensor(np.where(m)[0], dtype=torch.long, device=device)
    return to_t(sp["train_mask"]), to_t(sp["val_mask"]), to_t(sp["test_mask"])


def _acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    return float((y_true == y_pred).mean())


def _bucket_masks(margins: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    q33, q67 = np.percentile(margins, [100.0 / 3.0, 200.0 / 3.0])
    low = margins < q33
    med = (margins >= q33) & (margins < q67)
    high = margins >= q67
    return low, med, high


def _changed_quality(before: np.ndarray, after: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    changed = after != before
    n_changed = int(changed.sum())
    if n_changed == 0:
        return {
            "changed_fraction": 0.0,
            "harmful_overwrite_count": 0,
            "harmful_overwrite_rate": 0.0,
            "changed_precision": 1.0,
            "n_helped": 0,
            "n_hurt": 0,
        }
    before_ok = before[changed] == y[changed]
    after_ok = after[changed] == y[changed]
    n_helped = int((~before_ok & after_ok).sum())
    n_hurt = int((before_ok & ~after_ok).sum())
    denom = max(n_helped + n_hurt, 1)
    return {
        "changed_fraction": float(n_changed / len(y)),
        "harmful_overwrite_count": n_hurt,
        "harmful_overwrite_rate": float(n_hurt / n_changed),
        "changed_precision": float(n_helped / denom),
        "n_helped": n_helped,
        "n_hurt": n_hurt,
    }


def run_experiment(datasets: List[str], splits: List[int], split_dir: str, tag: str):
    mod = _load_module()

    per_split_rows: List[Dict[str, Any]] = []
    per_bucket_rows: List[Dict[str, Any]] = []
    per_node_rows: List[Dict[str, Any]] = []

    for ds in datasets:
        print(f"\n[dataset] {ds}")
        data = _load_data(mod, ds)
        y_all = data.y.detach().cpu().numpy().astype(np.int64)

        for sid in splits:
            print(f"  split={sid}")
            train_idx, val_idx, test_idx = _load_split(ds, sid, data.x.device, split_dir)
            train_np = mod._to_numpy_idx(train_idx)
            val_np = mod._to_numpy_idx(val_idx)
            test_np = mod._to_numpy_idx(test_idx)
            seed = (1337 + sid) * 100
            np.random.seed(seed)
            torch.manual_seed(seed)

            mlp_probs, _ = mod.train_mlp_and_predict(data, train_idx, **mod.DEFAULT_MANUSCRIPT_MLP_KWARGS, log_file=None)
            mlp_info = mod.compute_mlp_margin(mlp_probs)
            mlp_pred = mlp_info["mlp_pred_all"][test_np]
            mlp_margin = mlp_info["mlp_margin_all"][test_np]
            y_test = y_all[test_np]

            _, canonical_acc, canonical_info = final_method_v3(
                data,
                train_np,
                val_np,
                test_np,
                mlp_probs=mlp_probs,
                seed=seed,
                mod=mod,
                include_node_arrays=True,
            )
            _, local_acc, local_info = final_method_v3(
                data,
                train_np,
                val_np,
                test_np,
                mlp_probs=mlp_probs,
                seed=seed,
                mod=mod,
                include_node_arrays=True,
                use_local_agreement_gate=True,
                local_agreement_search_modes=[
                    "none",
                    "linear",
                    "hard_threshold",
                    "shifted_linear",
                    "exact_local_linear",
                    "exact_local_hard_threshold",
                    "exact_local_shifted_linear",
                ],
                local_agreement_eta_candidates=[0.3, 0.4, 0.5, 0.6, 0.7],
            )

            can_pred = canonical_info["node_arrays"]["final_predictions"]
            loc_pred = local_info["node_arrays"]["final_predictions"]
            mlp_acc = _acc(y_test, mlp_pred)
            assert abs(mlp_acc - canonical_info["test_acc_mlp"]) < 1e-12

            low_mask, med_mask, high_mask = _bucket_masks(mlp_margin)
            bucket_map = {"low": low_mask, "medium": med_mask, "high": high_mask}
            for bname, bmask in bucket_map.items():
                if int(bmask.sum()) == 0:
                    continue
                mlp_bacc = _acc(y_test[bmask], mlp_pred[bmask])
                can_bacc = _acc(y_test[bmask], can_pred[bmask])
                loc_bacc = _acc(y_test[bmask], loc_pred[bmask])
                per_bucket_rows.extend([
                    {
                        "dataset": ds,
                        "split": sid,
                        "bucket": bname,
                        "method": "MLP",
                        "bucket_size": int(bmask.sum()),
                        "bucket_acc": mlp_bacc,
                        "delta_vs_mlp": 0.0,
                    },
                    {
                        "dataset": ds,
                        "split": sid,
                        "bucket": bname,
                        "method": "FINAL_V3_CANONICAL",
                        "bucket_size": int(bmask.sum()),
                        "bucket_acc": can_bacc,
                        "delta_vs_mlp": can_bacc - mlp_bacc,
                    },
                    {
                        "dataset": ds,
                        "split": sid,
                        "bucket": bname,
                        "method": "FINAL_V3_LOCAL_AGREEMENT",
                        "bucket_size": int(bmask.sum()),
                        "bucket_acc": loc_bacc,
                        "delta_vs_mlp": loc_bacc - mlp_bacc,
                    },
                ])

            local_quality = _changed_quality(mlp_pred, loc_pred, y_test)
            canonical_quality = _changed_quality(mlp_pred, can_pred, y_test)
            high_changed_local = float((loc_pred[high_mask] != mlp_pred[high_mask]).mean()) if high_mask.any() else 0.0

            per_split_rows.append({
                "dataset": ds,
                "split": sid,
                "mlp_acc": mlp_acc,
                "final_v3_canonical_acc": canonical_acc,
                "final_v3_local_acc": local_acc,
                "local_minus_mlp": local_acc - mlp_acc,
                "local_minus_canonical": local_acc - canonical_acc,
                "canonical_minus_mlp": canonical_acc - mlp_acc,
                "low_bucket_acc_local": _acc(y_test[low_mask], loc_pred[low_mask]),
                "low_bucket_delta_vs_mlp_local": _acc(y_test[low_mask], loc_pred[low_mask]) - _acc(y_test[low_mask], mlp_pred[low_mask]),
                "high_bucket_delta_vs_mlp_local": _acc(y_test[high_mask], loc_pred[high_mask]) - _acc(y_test[high_mask], mlp_pred[high_mask]),
                "changed_fraction_local": local_quality["changed_fraction"],
                "harmful_overwrite_count_local": local_quality["harmful_overwrite_count"],
                "harmful_overwrite_rate_local": local_quality["harmful_overwrite_rate"],
                "changed_precision_local": local_quality["changed_precision"],
                "changed_fraction_canonical": canonical_quality["changed_fraction"],
                "harmful_overwrite_count_canonical": canonical_quality["harmful_overwrite_count"],
                "harmful_overwrite_rate_canonical": canonical_quality["harmful_overwrite_rate"],
                "changed_precision_canonical": canonical_quality["changed_precision"],
                "high_conf_untouched_rate_local": 1.0 - high_changed_local,
                "selected_local_mode": local_info["local_agreement"]["selected_mode"],
                "selected_local_score_family": local_info["local_agreement"].get("selected_score_family"),
                "selected_local_eta": local_info["local_agreement"].get("selected_eta"),
                "mean_local_agreement_test": local_info["local_agreement"]["mean_score_test"],
                "mean_local_multiplier_test": local_info["local_agreement"]["mean_multiplier_test"],
            })

            na_local = local_info["node_arrays"]
            na_can = canonical_info["node_arrays"]
            for i, nid in enumerate(na_local["test_indices"]):
                per_node_rows.append({
                    "dataset": ds,
                    "split": sid,
                    "node_id": int(nid),
                    "true_label": int(na_local["true_labels"][i]),
                    "mlp_pred": int(na_local["mlp_predictions"][i]),
                    "canonical_pred": int(na_can["final_predictions"][i]),
                    "local_pred": int(na_local["final_predictions"][i]),
                    "mlp_margin": float(na_local["mlp_margins"][i]),
                    "h_v_soft": float(na_local["local_agreement_scores"][i]),
                    "h_v_selected": float(na_local["selected_local_scores"][i]),
                    "local_multiplier": float(na_local["local_graph_multiplier"][i]),
                    "local_gate_active": bool(na_local["local_graph_multiplier"][i] > 0),
                    "is_uncertain": bool(na_local["is_uncertain"][i]),
                    "passes_reliability_gate": bool(na_local["passes_reliability_gate"][i]),
                    "is_corrected_branch": bool(na_local["is_corrected_branch"][i]),
                    "is_changed_local": bool(na_local["is_changed"][i]),
                    "is_changed_helpful_local": bool(na_local["is_changed_helpful"][i]),
                    "is_changed_harmful_local": bool(na_local["is_changed_harmful"][i]),
                })

    # write outputs
    os.makedirs(os.path.join(REPO_ROOT, "logs"), exist_ok=True)
    os.makedirs(os.path.join(REPO_ROOT, "tables"), exist_ok=True)
    os.makedirs(os.path.join(REPO_ROOT, "reports"), exist_ok=True)

    per_node_csv = os.path.join(REPO_ROOT, "logs", "final_v3_local_agreement_per_node.csv")
    per_split_csv = os.path.join(REPO_ROOT, "logs", "final_v3_local_agreement_per_split.csv")
    per_bucket_csv = os.path.join(REPO_ROOT, "logs", "final_v3_local_agreement_per_bucket.csv")

    def _write_csv(path: str, rows: List[Dict[str, Any]]):
        if not rows:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    _write_csv(per_node_csv, per_node_rows)
    _write_csv(per_split_csv, per_split_rows)
    _write_csv(per_bucket_csv, per_bucket_rows)

    # dataset-level table
    by_ds: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in per_split_rows:
        by_ds[row["dataset"]].append(row)

    main_rows = []
    for ds, rows in sorted(by_ds.items()):
        def m(col: str) -> float:
            return float(np.mean([r[col] for r in rows]))

        main_rows.append({
            "dataset": ds,
            "n_splits": len(rows),
            "mlp_acc": m("mlp_acc"),
            "canonical_acc": m("final_v3_canonical_acc"),
            "local_acc": m("final_v3_local_acc"),
            "local_minus_mlp": m("local_minus_mlp"),
            "local_minus_canonical": m("local_minus_canonical"),
            "low_bucket_delta_vs_mlp_local": m("low_bucket_delta_vs_mlp_local"),
            "high_bucket_delta_vs_mlp_local": m("high_bucket_delta_vs_mlp_local"),
            "changed_fraction_local": m("changed_fraction_local"),
            "harmful_overwrite_rate_local": m("harmful_overwrite_rate_local"),
            "changed_precision_local": m("changed_precision_local"),
            "high_conf_untouched_rate_local": m("high_conf_untouched_rate_local"),
        })

    main_csv = os.path.join(REPO_ROOT, "tables", "final_v3_local_agreement_main.csv")
    _write_csv(main_csv, main_rows)

    main_md = os.path.join(REPO_ROOT, "tables", "final_v3_local_agreement_main.md")
    with open(main_md, "w", encoding="utf-8") as f:
        f.write("# FINAL_V3 local agreement main table\n\n")
        f.write("| Dataset | MLP | Canonical V3 | Local-gated V3 | Local-MLP | Local-Canonical | Low-bucket Δ | High-bucket Δ | Harmful rate | Precision | High-conf untouched |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in main_rows:
            f.write(
                f"| {r['dataset']} | {r['mlp_acc']:.4f} | {r['canonical_acc']:.4f} | {r['local_acc']:.4f} | "
                f"{r['local_minus_mlp']:+.4f} | {r['local_minus_canonical']:+.4f} | "
                f"{r['low_bucket_delta_vs_mlp_local']:+.4f} | {r['high_bucket_delta_vs_mlp_local']:+.4f} | "
                f"{r['harmful_overwrite_rate_local']:.4f} | {r['changed_precision_local']:.4f} | "
                f"{r['high_conf_untouched_rate_local']:.4f} |\n"
            )

    summary_md = os.path.join(REPO_ROOT, "reports", "final_v3_local_agreement_summary.md")
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# FINAL_V3 + local agreement gating summary\n\n")
        f.write(f"Tag: `{tag}`. Datasets: {datasets}. Splits: {splits}.\n\n")
        f.write("## Dataset-level outcomes\n\n")
        for r in main_rows:
            f.write(
                f"- **{r['dataset']}**: local-minus-canonical={r['local_minus_canonical']:+.4f}, "
                f"low-bucket Δ vs MLP={r['low_bucket_delta_vs_mlp_local']:+.4f}, "
                f"high-conf untouched={r['high_conf_untouched_rate_local']:.3f}.\n"
            )
        f.write("\n## Notes\n\n")
        f.write("- Local gate attenuates graph-only terms; MLP term remains unchanged.\n")
        f.write("- No true test labels are used in gating; train labels + unlabeled soft MLP probs only.\n")
        f.write("- Search includes none/linear/hard-threshold/shifted-linear with eta grid {0.3..0.7}.\n")

    print("\nWrote:")
    for p in [per_node_csv, per_split_csv, per_bucket_csv, main_csv, main_md, summary_md]:
        print(f"  - {p}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", default="data/splits")
    ap.add_argument("--datasets", nargs="*", default=["chameleon", "texas", "citeseer", "cora"])
    ap.add_argument("--splits", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--tag", default="targeted")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args.datasets, args.splits, args.split_dir, args.tag)

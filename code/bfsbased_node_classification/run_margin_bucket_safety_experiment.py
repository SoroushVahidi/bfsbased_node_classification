#!/usr/bin/env python3
"""Canonical FINAL_V3 bucketed safety experiment.

Runs MLP, FINAL_V3, GCN, APPNP on fixed GEO-GCN splits, exports:
- per-node safety records (direct from FINAL_V3 node outputs)
- per-bucket metrics
- per-split metrics
- manuscript-ready summary/table files

Also supports a targeted structural extension comparison via:
--compare-structural-lowconf-term
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from final_method_v3 import final_method_v3
from standard_node_baselines import run_baseline


def _simple_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    return float((y_true == y_pred).mean())


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


def _load_data(mod, dataset_key: str):
    dataset = mod.load_dataset(dataset_key, "data/")
    data = dataset[0]
    data.edge_index = torch.cat([data.edge_index, data.edge_index.flip(0)], dim=1)
    data.edge_index = torch.unique(data.edge_index, dim=1)
    return data


def _load_split(dataset_key: str, split_id: int, device, split_dir: str):
    prefix = "film" if dataset_key == "actor" else dataset_key.lower()
    fname = f"{prefix}_split_0.6_0.2_{split_id}.npz"
    path = os.path.join(split_dir, fname)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    sp = np.load(path)
    to_t = lambda m: torch.as_tensor(np.where(m)[0], dtype=torch.long, device=device)
    return to_t(sp["train_mask"]), to_t(sp["val_mask"]), to_t(sp["test_mask"])


def _bucket_labels(margins_test: np.ndarray) -> np.ndarray:
    q1, q2 = np.quantile(margins_test, [1.0 / 3.0, 2.0 / 3.0])
    out = np.full(margins_test.shape[0], "medium", dtype=object)
    out[margins_test <= q1] = "low"
    out[margins_test > q2] = "high"
    return out


def _run_graph_baseline(
    method: str,
    data,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    *,
    seed: int,
    max_epochs: int,
    patience: int,
):
    return run_baseline(
        method,
        data,
        train_idx,
        val_idx,
        test_idx,
        seed=seed,
        max_epochs=max_epochs,
        patience=patience,
    )


def run_bucket_safety(
    *,
    datasets: List[str],
    splits: List[int],
    split_dir: str,
    baseline_max_epochs: int,
    baseline_patience: int,
    enable_lowconf_structural_term: bool,
    b6_candidates: List[float],
    out_per_node: str,
    out_per_bucket: str,
    out_per_split: str,
    out_report: str,
    out_table_csv: str,
    out_table_md: str,
) -> Dict[str, Any]:
    def _makedirs_for_output(path: str) -> None:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)

    _makedirs_for_output(out_per_node)
    _makedirs_for_output(out_report)
    _makedirs_for_output(out_table_csv)
    mod = _load_module()
    per_node_rows: List[Dict[str, Any]] = []
    per_bucket_rows: List[Dict[str, Any]] = []
    per_split_rows: List[Dict[str, Any]] = []

    for ds in datasets:
        print(f"[dataset] {ds}", flush=True)
        data = _load_data(mod, ds)
        y_true_all = data.y.detach().cpu().numpy().astype(np.int64)

        for sid in splits:
            print(f"  [split] {sid}", flush=True)
            seed = (1337 + sid) * 100
            torch.manual_seed(seed)
            np.random.seed(seed)

            train_idx, val_idx, test_idx = _load_split(ds, sid, data.x.device, split_dir)
            train_np = mod._to_numpy_idx(train_idx)
            val_np = mod._to_numpy_idx(val_idx)
            test_np = mod._to_numpy_idx(test_idx)

            mlp_cfg = mod.DEFAULT_MANUSCRIPT_MLP_KWARGS
            mlp_probs, _ = mod.train_mlp_and_predict(
                data,
                train_idx,
                hidden=mlp_cfg["hidden"],
                layers=mlp_cfg["layers"],
                dropout=mlp_cfg["dropout"],
                lr=mlp_cfg["lr"],
                epochs=mlp_cfg["epochs"],
                log_file=None,
            )
            mlp_info = mod.compute_mlp_margin(mlp_probs)
            mlp_pred_all = mlp_info["mlp_pred_all"]
            mlp_margin_all = mlp_info["mlp_margin_all"]
            mlp_acc = _simple_accuracy(y_true_all[test_np], mlp_pred_all[test_np])

            _, v3_acc, v3_info = final_method_v3(
                data,
                train_np,
                val_np,
                test_np,
                mlp_probs=mlp_probs,
                seed=seed,
                mod=mod,
                enable_lowconf_structural_term=enable_lowconf_structural_term,
                lowconf_structural_b6_candidates=b6_candidates,
            )
            node_out = v3_info["test_node_outputs"]
            v3_pred_test = np.asarray(node_out["final_pred"], dtype=np.int64)
            mlp_pred_test = np.asarray(node_out["mlp_pred"], dtype=np.int64)
            margins_test = np.asarray(node_out["mlp_margin"], dtype=np.float64)
            buckets = _bucket_labels(margins_test)

            gcn_res = _run_graph_baseline(
                "gcn", data, train_idx, val_idx, test_idx,
                seed=seed, max_epochs=baseline_max_epochs, patience=baseline_patience,
            )
            appnp_res = _run_graph_baseline(
                "appnp", data, train_idx, val_idx, test_idx,
                seed=seed, max_epochs=baseline_max_epochs, patience=baseline_patience,
            )
            gcn_pred_all = gcn_res.probs.argmax(dim=1).detach().cpu().numpy().astype(np.int64)
            appnp_pred_all = appnp_res.probs.argmax(dim=1).detach().cpu().numpy().astype(np.int64)
            gcn_pred_test = gcn_pred_all[test_np]
            appnp_pred_test = appnp_pred_all[test_np]
            gcn_acc = _simple_accuracy(y_true_all[test_np], gcn_pred_test)
            appnp_acc = _simple_accuracy(y_true_all[test_np], appnp_pred_test)

            y_test = y_true_all[test_np]
            beneficial = ((mlp_pred_test != y_test) & (v3_pred_test == y_test)).astype(np.int64)
            harmful = ((mlp_pred_test == y_test) & (v3_pred_test != y_test)).astype(np.int64)
            changed = (v3_pred_test != mlp_pred_test).astype(np.int64)

            for i, node_id in enumerate(test_np.tolist()):
                per_node_rows.append(
                    {
                        "dataset": ds,
                        "split_id": sid,
                        "node_id": int(node_id),
                        "bucket": str(buckets[i]),
                        "true_label": int(y_test[i]),
                        "mlp_pred": int(mlp_pred_test[i]),
                        "final_v3_pred": int(v3_pred_test[i]),
                        "gcn_pred": int(gcn_pred_test[i]),
                        "appnp_pred": int(appnp_pred_test[i]),
                        "final_v3_changed_from_mlp": int(changed[i]),
                        "final_v3_beneficial_change": int(beneficial[i]),
                        "final_v3_harmful_change": int(harmful[i]),
                        "mlp_margin": float(node_out["mlp_margin"][i]),
                        "passed_uncertainty_gate": int(node_out["passed_uncertainty_gate"][i]),
                        "passed_reliability_gate": int(node_out["passed_reliability_gate"][i]),
                        "reliability": float(node_out["reliability"][i]),
                        "selected_tau": float(node_out["selected_tau"]),
                        "selected_rho": float(node_out["selected_rho"]),
                        "selected_b6": float(v3_info.get("selected_b6", 0.0)),
                        "selected_weights_json": json.dumps(node_out["selected_weights"], sort_keys=True),
                        "combined_top_score": float(node_out["combined_top_score"][i]),
                        "combined_top_label": int(node_out["combined_top_label"][i]),
                    }
                )

            for bucket in ["low", "medium", "high"]:
                mask = buckets == bucket
                if not np.any(mask):
                    continue
                n = int(mask.sum())
                mlp_acc_b = _simple_accuracy(y_test[mask], mlp_pred_test[mask])
                v3_acc_b = _simple_accuracy(y_test[mask], v3_pred_test[mask])
                gcn_acc_b = _simple_accuracy(y_test[mask], gcn_pred_test[mask])
                appnp_acc_b = _simple_accuracy(y_test[mask], appnp_pred_test[mask])
                changed_n = int(changed[mask].sum())
                bene_n = int(beneficial[mask].sum())
                harm_n = int(harmful[mask].sum())
                precision = float(bene_n / max(bene_n + harm_n, 1))
                harm_rate = float(harm_n / max(changed_n, 1)) if changed_n > 0 else 0.0
                per_bucket_rows.append(
                    {
                        "dataset": ds,
                        "split_id": sid,
                        "bucket": bucket,
                        "n_test_nodes": n,
                        "mlp_acc": mlp_acc_b,
                        "final_v3_acc": v3_acc_b,
                        "gcn_acc": gcn_acc_b,
                        "appnp_acc": appnp_acc_b,
                        "final_v3_delta_vs_mlp": v3_acc_b - mlp_acc_b,
                        "gcn_delta_vs_mlp": gcn_acc_b - mlp_acc_b,
                        "appnp_delta_vs_mlp": appnp_acc_b - mlp_acc_b,
                        "final_v3_changed_fraction": float(changed_n / n),
                        "final_v3_beneficial_change_count": bene_n,
                        "final_v3_harmful_change_count": harm_n,
                        "final_v3_harmful_overwrite_rate": harm_rate,
                        "final_v3_correction_precision": precision,
                    }
                )

            per_split_rows.append(
                {
                    "dataset": ds,
                    "split_id": sid,
                    "mlp_acc": mlp_acc,
                    "final_v3_acc": v3_acc,
                    "gcn_acc": gcn_acc,
                    "appnp_acc": appnp_acc,
                    "final_v3_delta_vs_mlp": v3_acc - mlp_acc,
                    "gcn_delta_vs_mlp": gcn_acc - mlp_acc,
                    "appnp_delta_vs_mlp": appnp_acc - mlp_acc,
                    "final_v3_changed_count": int(changed.sum()),
                    "final_v3_beneficial_change_count": int(beneficial.sum()),
                    "final_v3_harmful_change_count": int(harmful.sum()),
                    "final_v3_correction_precision": float(beneficial.sum() / max(beneficial.sum() + harmful.sum(), 1)),
                    "final_v3_harmful_overwrite_rate": float(harmful.sum() / max(changed.sum(), 1)) if int(changed.sum()) > 0 else 0.0,
                    "selected_tau": float(v3_info["selected_tau"]),
                    "selected_rho": float(v3_info["selected_rho"]),
                    "selected_b6": float(v3_info.get("selected_b6", 0.0)),
                    "enable_lowconf_structural_term": bool(enable_lowconf_structural_term),
                }
            )

    # write raw outputs
    pd.DataFrame(per_node_rows).to_csv(out_per_node, index=False)
    pd.DataFrame(per_bucket_rows).to_csv(out_per_bucket, index=False)
    pd.DataFrame(per_split_rows).to_csv(out_per_split, index=False)

    bucket_df = pd.DataFrame(per_bucket_rows)
    split_df = pd.DataFrame(per_split_rows)

    def _weighted_mean(frame: pd.DataFrame, value_col: str, weight_col: str) -> float:
        weights = frame[weight_col].to_numpy(dtype=float)
        values = frame[value_col].to_numpy(dtype=float)
        total_weight = weights.sum()
        if total_weight <= 0:
            return float(np.nan)
        return float(np.average(values, weights=weights))

    def _summarize_bucket(frame: pd.DataFrame) -> pd.Series:
        n_test_nodes = int(frame["n_test_nodes"].sum())
        beneficial_sum = int(frame["final_v3_beneficial_change_count"].sum())
        harmful_sum = int(frame["final_v3_harmful_change_count"].sum())
        changed_sum = float((frame["final_v3_changed_fraction"] * frame["n_test_nodes"]).sum())
        total_corrections = beneficial_sum + harmful_sum

        return pd.Series(
            {
                "n_test_nodes": n_test_nodes,
                "mlp_acc": _weighted_mean(frame, "mlp_acc", "n_test_nodes"),
                "final_v3_acc": _weighted_mean(frame, "final_v3_acc", "n_test_nodes"),
                "gcn_acc": _weighted_mean(frame, "gcn_acc", "n_test_nodes"),
                "appnp_acc": _weighted_mean(frame, "appnp_acc", "n_test_nodes"),
                "final_v3_delta_vs_mlp": _weighted_mean(frame, "final_v3_delta_vs_mlp", "n_test_nodes"),
                "final_v3_changed_fraction": _weighted_mean(frame, "final_v3_changed_fraction", "n_test_nodes"),
                "final_v3_beneficial_change_count": beneficial_sum,
                "final_v3_harmful_change_count": harmful_sum,
                "final_v3_harmful_overwrite_rate": (
                    float(harmful_sum / changed_sum) if changed_sum > 0 else float(np.nan)
                ),
                "final_v3_correction_precision": (
                    float(beneficial_sum / total_corrections) if total_corrections > 0 else float(np.nan)
                ),
            }
        )

    bucket_main = (
        bucket_df.groupby(["dataset", "bucket"], as_index=False)
        .apply(_summarize_bucket, include_groups=False)
        .reset_index()
    )
    bucket_main.to_csv(out_table_csv, index=False)
    with open(out_table_md, "w", encoding="utf-8") as f:
        f.write("# FINAL_V3 bucket safety main table\n\n")
        f.write(bucket_main.to_markdown(index=False))
        f.write("\n")

    split_main = (
        split_df.groupby("dataset", as_index=False)
        .agg(
            mlp_acc=("mlp_acc", "mean"),
            final_v3_acc=("final_v3_acc", "mean"),
            gcn_acc=("gcn_acc", "mean"),
            appnp_acc=("appnp_acc", "mean"),
            final_v3_delta_vs_mlp=("final_v3_delta_vs_mlp", "mean"),
            final_v3_harmful_change_count=("final_v3_harmful_change_count", "sum"),
            final_v3_beneficial_change_count=("final_v3_beneficial_change_count", "sum"),
        )
    )

    with open(out_report, "w", encoding="utf-8") as f:
        f.write("# FINAL_V3 canonical bucket safety summary\n\n")
        f.write("Datasets: " + ", ".join(datasets) + "\n\n")
        f.write("Splits: " + ", ".join(map(str, splits)) + "\n\n")
        f.write("## Bucket-level means\n\n")
        f.write(bucket_main.to_markdown(index=False))
        f.write("\n\n## Split-level dataset means\n\n")
        f.write(split_main.to_markdown(index=False))
        f.write("\n")

    return {
        "per_node": out_per_node,
        "per_bucket": out_per_bucket,
        "per_split": out_per_split,
        "report": out_report,
        "table_csv": out_table_csv,
        "table_md": out_table_md,
        "n_per_node": len(per_node_rows),
        "n_per_bucket": len(per_bucket_rows),
        "n_per_split": len(per_split_rows),
    }


def _parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical FINAL_V3 bucket safety experiment.")
    parser.add_argument("--split-dir", default=os.path.join(REPO_ROOT, "data", "splits"))
    parser.add_argument("--datasets", nargs="+", default=["cora", "citeseer", "pubmed", "chameleon", "texas", "wisconsin"])
    parser.add_argument("--splits", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--baseline-max-epochs", type=int, default=160)
    parser.add_argument("--baseline-patience", type=int, default=60)
    parser.add_argument("--compare-structural-lowconf-term", action="store_true")
    parser.add_argument("--b6-candidates", default="0.0,0.2,0.4")

    parser.add_argument("--out-per-node", default=os.path.join(REPO_ROOT, "logs", "final_v3_bucket_safety_per_node.csv"))
    parser.add_argument("--out-per-bucket", default=os.path.join(REPO_ROOT, "logs", "final_v3_bucket_safety_per_bucket.csv"))
    parser.add_argument("--out-per-split", default=os.path.join(REPO_ROOT, "logs", "final_v3_bucket_safety_per_split.csv"))
    parser.add_argument("--out-report", default=os.path.join(REPO_ROOT, "reports", "final_v3_bucket_safety_summary.md"))
    parser.add_argument("--out-table-csv", default=os.path.join(REPO_ROOT, "tables", "final_v3_bucket_safety_main.csv"))
    parser.add_argument("--out-table-md", default=os.path.join(REPO_ROOT, "tables", "final_v3_bucket_safety_main.md"))
    args = parser.parse_args()

    out = run_bucket_safety(
        datasets=args.datasets,
        splits=args.splits,
        split_dir=args.split_dir,
        baseline_max_epochs=args.baseline_max_epochs,
        baseline_patience=args.baseline_patience,
        enable_lowconf_structural_term=bool(args.compare_structural_lowconf_term),
        b6_candidates=_parse_float_list(args.b6_candidates),
        out_per_node=args.out_per_node,
        out_per_bucket=args.out_per_bucket,
        out_per_split=args.out_per_split,
        out_report=args.out_report,
        out_table_csv=args.out_table_csv,
        out_table_md=args.out_table_md,
    )
    print(out)


if __name__ == "__main__":
    main()

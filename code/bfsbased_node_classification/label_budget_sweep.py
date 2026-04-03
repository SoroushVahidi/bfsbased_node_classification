#!/usr/bin/env python3
"""
Label-budget sweep: subsample training labels from the official train split only;
validation and test masks are unchanged.

Writes JSONL compatible with baseline-style aggregation plus train_fraction / n_train_*.
Does not touch canonical reports/final_method_v3_results.csv.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import socket
import sys
import time
import traceback
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from final_method_v3 import final_method_v3, _simple_accuracy  # noqa: E402
from run_final_evaluation import _load_data, _load_module, _load_split  # noqa: E402
from standard_node_baselines import run_baseline  # noqa: E402

_EXTERNAL_LABEL = {
    "clp": "CLP",
    "correct_and_smooth": "CORRECT_AND_SMOOTH",
    "sgc": "SGC",
    "graphsage": "GRAPHSAGE",
    "appnp": "APPNP",
}


def _acc_np(pred: np.ndarray, y: np.ndarray, idx: np.ndarray) -> float:
    if idx.size == 0:
        return float("nan")
    return float((pred[idx] == y[idx]).mean())


def _subsample_train_idx(
    train_idx: torch.Tensor,
    train_fraction: float,
    subsample_seed: int,
) -> Tuple[torch.Tensor, int, int]:
    """Subset of original training nodes only; deterministic given seed."""
    n_full = int(train_idx.numel())
    if float(train_fraction) >= 0.999:
        return train_idx.clone(), n_full, n_full
    k = max(1, int(round(float(train_fraction) * n_full)))
    k = min(k, n_full)
    rng = np.random.default_rng(int(subsample_seed) % (2**32 - 1))
    positions = rng.permutation(n_full)
    chosen = positions[:k]
    tr_np = train_idx.cpu().numpy()[chosen]
    tr_sorted = np.sort(tr_np.astype(np.int64))
    out = torch.as_tensor(tr_sorted, dtype=torch.long, device=train_idx.device)
    return out, int(k), n_full


def _record_base(
    *,
    dataset: str,
    split_id: int,
    seed: int,
    method: str,
    output_tag: str,
    train_fraction: float,
    n_train_labeled: int,
    n_train_full: int,
    subsample_seed: int,
    mlp_test_acc: float,
) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "split_id": split_id,
        "seed": seed,
        "method": method,
        "output_tag": output_tag,
        "train_fraction": float(train_fraction),
        "n_train_labeled": int(n_train_labeled),
        "n_train_full": int(n_train_full),
        "subsample_seed": int(subsample_seed),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "cuda_available": bool(torch.cuda.is_available()),
        "delta_vs_mlp": None,
        "test_acc": None,
        "val_acc": None,
        "train_acc": None,
        "mlp_test_acc": float(mlp_test_acc),
        "n_helped": None,
        "n_hurt": None,
        "frac_corrected": None,
        "changed_fraction_test": None,
        "correction_precision": None,
        "tau": None,
        "rho": None,
        "runtime_sec": None,
        "peak_rss_kb_after": None,
        "success": True,
        "error": None,
        "notes": None,
    }


def _rss_kb() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def run_label_budget_shard(
    dataset: str,
    train_fraction: float,
    splits: List[int],
    split_dir: str,
    output_tag: str,
    output_jsonl: str,
    *,
    baseline_max_epochs: int = 500,
    baseline_patience: int = 100,
    gate: str = "heuristic",
) -> str:
    mod = _load_module()
    os.makedirs(os.path.dirname(os.path.abspath(output_jsonl)) or ".", exist_ok=True)

    baseline_keys: List[str] = ["clp", "correct_and_smooth", "sgc", "graphsage", "appnp"]

    with open(output_jsonl, "w", encoding="utf-8") as outf:
        try:
            data = _load_data(mod, dataset)
        except Exception as e:
            rec = {
                "dataset": dataset,
                "split_id": None,
                "method": "__DATASET_LOAD__",
                "success": False,
                "error": str(e),
                "output_tag": output_tag,
                "train_fraction": float(train_fraction),
            }
            outf.write(json.dumps(rec, default=str) + "\n")
            return output_jsonl

        y_true = data.y.detach().cpu().numpy().astype(np.int64)

        for sid in splits:
            try:
                train_idx_full, val_idx, test_idx = _load_split(
                    dataset, sid, data.x.device, split_dir
                )
            except FileNotFoundError as e:
                print(f"  [SKIP split] {e}")
                continue

            seed = (1337 + sid) * 100
            subsample_seed = (1337 + sid) * 1_000_007 + int(round(float(train_fraction) * 10_000))
            train_idx, n_lab, n_full = _subsample_train_idx(
                train_idx_full, train_fraction, subsample_seed
            )

            train_np = mod._to_numpy_idx(train_idx)
            val_np = mod._to_numpy_idx(val_idx)
            test_np = mod._to_numpy_idx(test_idx)

            torch.manual_seed(seed)
            np.random.seed(seed)

            # --- MLP (reduced train) ---
            t0 = time.perf_counter()
            mlp_probs, _ = mod.train_mlp_and_predict(
                data,
                train_idx,
                **mod.DEFAULT_MANUSCRIPT_MLP_KWARGS,
                log_file=None,
            )
            mlp_time = time.perf_counter() - t0
            mlp_info = mod.compute_mlp_margin(mlp_probs)
            mlp_pred = mlp_info["mlp_pred_all"]
            mlp_test_acc = _simple_accuracy(y_true[test_np], mlp_pred[test_np])
            mlp_val_acc = _acc_np(mlp_pred, y_true, val_np)
            mlp_train_acc = _acc_np(mlp_pred, y_true, train_np)

            r = _record_base(
                dataset=dataset,
                split_id=sid,
                seed=seed,
                method="MLP_ONLY",
                output_tag=output_tag,
                train_fraction=train_fraction,
                n_train_labeled=n_lab,
                n_train_full=n_full,
                subsample_seed=subsample_seed,
                mlp_test_acc=mlp_test_acc,
            )
            r["test_acc"] = mlp_test_acc
            r["val_acc"] = mlp_val_acc
            r["train_acc"] = mlp_train_acc
            r["delta_vs_mlp"] = 0.0
            r["runtime_sec"] = float(mlp_time)
            r["peak_rss_kb_after"] = _rss_kb()
            r["notes"] = "label_budget_mlp_reduced_train"
            outf.write(json.dumps(r, default=str) + "\n")
            outf.flush()
            print(
                f"  split={sid} frac={train_fraction} n_train={n_lab}/{n_full} "
                f"MLP test={mlp_test_acc:.4f}"
            )

            # --- FINAL_V3 ---
            t1 = time.perf_counter()
            try:
                _, v3_acc, v3_info = final_method_v3(
                    data,
                    train_np,
                    val_np,
                    test_np,
                    mlp_probs=mlp_probs,
                    seed=seed,
                    mod=mod,
                    gate=gate,
                    split_id=sid,
                )
                v3_time = time.perf_counter() - t1
                ca = v3_info["correction_analysis"]
                bf = v3_info["branch_fractions"]
                r2 = _record_base(
                    dataset=dataset,
                    split_id=sid,
                    seed=seed,
                    method="FINAL_V3",
                    output_tag=output_tag,
                    train_fraction=train_fraction,
                    n_train_labeled=n_lab,
                    n_train_full=n_full,
                    subsample_seed=subsample_seed,
                    mlp_test_acc=mlp_test_acc,
                )
                r2["test_acc"] = float(v3_acc)
                r2["val_acc"] = v3_info.get("val_acc_v3")
                r2["delta_vs_mlp"] = float(v3_acc - mlp_test_acc)
                r2["n_helped"] = ca["n_helped"]
                r2["n_hurt"] = ca["n_hurt"]
                r2["correction_precision"] = ca["correction_precision"]
                r2["frac_corrected"] = bf.get("uncertain_reliable_corrected")
                r2["changed_fraction_test"] = ca.get("changed_fraction_test")
                r2["tau"] = v3_info.get("selected_tau")
                r2["rho"] = v3_info.get("selected_rho")
                r2["runtime_sec"] = float(v3_time)
                r2["peak_rss_kb_after"] = _rss_kb()
                r2["notes"] = "label_budget_final_v3"
                outf.write(json.dumps(r2, default=str) + "\n")
                outf.flush()
                print(f"    FINAL_V3: {v3_acc:.4f} d={v3_acc - mlp_test_acc:+.4f}")
            except Exception as e:
                r2 = _record_base(
                    dataset=dataset,
                    split_id=sid,
                    seed=seed,
                    method="FINAL_V3",
                    output_tag=output_tag,
                    train_fraction=train_fraction,
                    n_train_labeled=n_lab,
                    n_train_full=n_full,
                    subsample_seed=subsample_seed,
                    mlp_test_acc=mlp_test_acc,
                )
                r2["success"] = False
                r2["error"] = f"{e}\n{traceback.format_exc()}"
                outf.write(json.dumps(r2, default=str) + "\n")
                outf.flush()
                print(f"    FINAL_V3 FAILED: {e}")

            # --- External baselines (same reduced train) ---
            for bkey in baseline_keys:
                label = _EXTERNAL_LABEL.get(bkey, bkey.upper())
                t2 = time.perf_counter()
                try:
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    br = run_baseline(
                        bkey,
                        data,
                        train_idx,
                        val_idx,
                        test_idx,
                        seed=seed,
                        max_epochs=baseline_max_epochs,
                        patience=baseline_patience,
                    )
                    bt = time.perf_counter() - t2
                    pred = br.probs.argmax(dim=1).cpu().numpy()
                    r3 = _record_base(
                        dataset=dataset,
                        split_id=sid,
                        seed=seed,
                        method=label,
                        output_tag=output_tag,
                        train_fraction=train_fraction,
                        n_train_labeled=n_lab,
                        n_train_full=n_full,
                        subsample_seed=subsample_seed,
                        mlp_test_acc=mlp_test_acc,
                    )
                    r3["test_acc"] = float(br.test_acc)
                    r3["val_acc"] = float(br.val_acc)
                    r3["train_acc"] = float((pred[train_np] == y_true[train_np]).mean())
                    r3["delta_vs_mlp"] = float(br.test_acc - mlp_test_acc)
                    r3["runtime_sec"] = float(bt)
                    r3["peak_rss_kb_after"] = _rss_kb()
                    r3["notes"] = f"external_baseline_key={bkey}"
                    outf.write(json.dumps(r3, default=str) + "\n")
                    outf.flush()
                    print(f"    {label}: test={br.test_acc:.4f}")
                except Exception as e:
                    r3 = _record_base(
                        dataset=dataset,
                        split_id=sid,
                        seed=seed,
                        method=label,
                        output_tag=output_tag,
                        train_fraction=train_fraction,
                        n_train_labeled=n_lab,
                        n_train_full=n_full,
                        subsample_seed=subsample_seed,
                        mlp_test_acc=mlp_test_acc,
                    )
                    r3["success"] = False
                    r3["error"] = f"{e}\n{traceback.format_exc()}"
                    outf.write(json.dumps(r3, default=str) + "\n")
                    outf.flush()
                    print(f"    {label} FAILED: {e}")

    return output_jsonl


def main() -> None:
    p = argparse.ArgumentParser(description="Label-budget sweep (one dataset × one train fraction per invocation).")
    p.add_argument("--dataset", required=True, help="Single dataset key, e.g. cora")
    p.add_argument("--train-fraction", type=float, required=True, help="e.g. 0.1, 0.2, 0.4, 1.0")
    p.add_argument("--splits", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--split-dir", default="data/splits")
    p.add_argument("--output-tag", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--baseline-max-epochs", type=int, default=500)
    p.add_argument("--baseline-patience", type=int, default=100)
    p.add_argument("--gate", choices=["heuristic", "adaptive"], default="heuristic")
    args = p.parse_args()

    path = run_label_budget_shard(
        args.dataset,
        args.train_fraction,
        args.splits,
        args.split_dir,
        args.output_tag,
        args.output_jsonl,
        baseline_max_epochs=args.baseline_max_epochs,
        baseline_patience=args.baseline_patience,
        gate=args.gate,
    )
    print(f"\nWrote: {path}")


if __name__ == "__main__":
    main()

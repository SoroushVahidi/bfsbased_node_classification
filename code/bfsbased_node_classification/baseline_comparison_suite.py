#!/usr/bin/env python3
"""
Wulver-scale baseline comparison: internal methods + Begga external baselines.

Writes JSONL (one object per line) under outputs/baseline_comparison/<tag>/per_run/
Never overwrites canonical reports/final_method_v3_results.csv unless you opt in elsewhere.

Use ``--external-baselines-only`` to run only ``--external-baselines`` after the
manuscript MLP anchor (skips BASELINE_SGC_V1, V2_MULTIBRANCH, FINAL_V3) for
feature-first / post-hoc baseline sweeps.
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
from typing import Any, Dict, List, Optional

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from final_method_v3 import final_method_v3, _simple_accuracy  # noqa: E402
from improved_sgc_variants import v2_multibranch_correction  # noqa: E402
from run_final_evaluation import _load_data, _load_module, _load_split  # noqa: E402
from standard_node_baselines import (  # noqa: E402
    BEGGA_HETEROPHILY_BASELINE_METHODS,
    PAPER_GRAPH_BASELINE_METHODS,
    run_baseline,
)

_EXTERNAL_DISPLAY = {
    "h2gcn": "H2GCN",
    "gcnii": "GCNII",
    "geom_gcn": "GeomGCN",
    "linkx": "LINKX",
    "acmii_gcn_plus_plus": "ACMII_GCN_PLUSPLUS",
    "appnp": "APPNP",
    "clp": "CLP",
    "correct_and_smooth": "CORRECT_AND_SMOOTH",
    "cs": "CORRECT_AND_SMOOTH",
    "graphsage": "GRAPHSAGE",
    "sgc": "SGC",
    "sgc_wu2019": "SGC",
}


def _rss_kb() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _acc_np(pred: np.ndarray, y: np.ndarray, idx: np.ndarray) -> float:
    if idx.size == 0:
        return float("nan")
    return float((pred[idx] == y[idx]).mean())


def _mlp_pred_from_probs(mlp_probs: torch.Tensor) -> np.ndarray:
    return mlp_probs.detach().cpu().numpy().argmax(axis=1).astype(np.int64)


def _record_base(
    *,
    dataset: str,
    split_id: int,
    seed: int,
    method: str,
    output_tag: str,
    mlp_test_acc: float,
) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "split_id": split_id,
        "seed": seed,
        "method": method,
        "output_tag": output_tag,
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
        "n_changed": None,
        "correction_precision": None,
        "frac_confident": None,
        "frac_unreliable": None,
        "frac_corrected": None,
        "changed_fraction_test": None,
        "harmful_overwrite_rate": None,
        "tau": None,
        "rho": None,
        "gate_mode": None,
        "runtime_sec": None,
        "mlp_runtime_sec": None,
        "peak_rss_kb_after": None,
        "n_parameters": None,
        "success": True,
        "error": None,
        "notes": None,
    }


def run_suite(
    datasets: List[str],
    splits: List[int],
    split_dir: str,
    output_tag: str,
    out_dir: str,
    *,
    gate_mode: str = "heuristic",
    external_baselines: Optional[List[str]] = None,
    baseline_max_epochs: int = 500,
    baseline_patience: int = 100,
    output_jsonl: Optional[str] = None,
    external_baselines_only: bool = False,
) -> str:
    if external_baselines is None:
        external_baselines = list(BEGGA_HETEROPHILY_BASELINE_METHODS)
    if external_baselines_only and not external_baselines:
        raise ValueError("external_baselines_only requires a non-empty external_baselines list")

    mod = _load_module()
    jsonl_path = output_jsonl or os.path.join(out_dir, "runs.jsonl")
    os.makedirs(os.path.dirname(os.path.abspath(jsonl_path)) or ".", exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    mode = "a" if os.path.isfile(jsonl_path) else "w"
    with open(jsonl_path, mode, encoding="utf-8") as outf:

        for ds in datasets:
            print(f"\n{'='*60}\nDataset: {ds}\n{'='*60}")
            try:
                data = _load_data(mod, ds)
            except Exception as e:
                rec = {
                    "dataset": ds,
                    "split_id": None,
                    "method": "__DATASET_LOAD__",
                    "success": False,
                    "error": str(e),
                    "output_tag": output_tag,
                }
                outf.write(json.dumps(rec, default=str) + "\n")
                outf.flush()
                print(f"  [SKIP dataset] {e}")
                continue

            y_true = data.y.detach().cpu().numpy().astype(np.int64)

            for sid in splits:
                try:
                    train_idx, val_idx, test_idx = _load_split(ds, sid, data.x.device, split_dir)
                except FileNotFoundError as e:
                    print(f"  [SKIP split] {e}")
                    continue

                train_np = mod._to_numpy_idx(train_idx)
                val_np = mod._to_numpy_idx(val_idx)
                test_np = mod._to_numpy_idx(test_idx)
                seed = (1337 + sid) * 100
                torch.manual_seed(seed)
                np.random.seed(seed)

                # --- MLP ---
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
                    dataset=ds,
                    split_id=sid,
                    seed=seed,
                    method="MLP_ONLY",
                    output_tag=output_tag,
                    mlp_test_acc=mlp_test_acc,
                )
                r["test_acc"] = mlp_test_acc
                r["val_acc"] = mlp_val_acc
                r["train_acc"] = mlp_train_acc
                r["delta_vs_mlp"] = 0.0
                r["runtime_sec"] = float(mlp_time)
                r["mlp_runtime_sec"] = float(mlp_time)
                r["peak_rss_kb_after"] = _rss_kb()
                outf.write(json.dumps(r, default=str) + "\n")
                outf.flush()
                print(f"  split={sid} MLP test={mlp_test_acc:.4f} val={mlp_val_acc:.4f}")

                if not external_baselines_only:
                    # --- BASELINE_SGC_V1 ---
                    t0 = time.perf_counter()
                    try:
                        _, v1_acc, v1_info = mod.selective_graph_correction_predictclass(
                            data,
                            train_np,
                            val_np,
                            test_np,
                            mlp_probs=mlp_probs,
                            seed=seed,
                            log_file=None,
                            write_node_diagnostics=False,
                        )
                        v1_time = time.perf_counter() - t0
                        r = _record_base(
                            dataset=ds,
                            split_id=sid,
                            seed=seed,
                            method="BASELINE_SGC_V1",
                            output_tag=output_tag,
                            mlp_test_acc=mlp_test_acc,
                        )
                        r["test_acc"] = float(v1_acc)
                        r["val_acc"] = None
                        r["train_acc"] = None
                        r["delta_vs_mlp"] = float(v1_acc - mlp_test_acc)
                        r["frac_corrected"] = float(v1_info.get("fraction_test_nodes_uncertain", 0))
                        r["tau"] = v1_info.get("selected_threshold_high", 0.2)
                        r["runtime_sec"] = float(v1_time)
                        r["peak_rss_kb_after"] = _rss_kb()
                        outf.write(json.dumps(r, default=str) + "\n")
                        outf.flush()
                        print(f"    SGC_V1: {v1_acc:.4f}")
                    except Exception as e:
                        r = _record_base(
                            dataset=ds,
                            split_id=sid,
                            seed=seed,
                            method="BASELINE_SGC_V1",
                            output_tag=output_tag,
                            mlp_test_acc=mlp_test_acc,
                        )
                        r["success"] = False
                        r["error"] = f"{e}\n{traceback.format_exc()}"
                        outf.write(json.dumps(r, default=str) + "\n")
                        outf.flush()
                        print(f"    SGC_V1 FAILED: {e}")

                    # --- V2_MULTIBRANCH ---
                    t0 = time.perf_counter()
                    try:
                        _v2_val, v2_acc, v2_info = v2_multibranch_correction(
                            data,
                            train_np,
                            val_np,
                            test_np,
                            mlp_probs=mlp_probs,
                            seed=seed,
                            mod=mod,
                        )
                        v2_time = time.perf_counter() - t0
                        r = _record_base(
                            dataset=ds,
                            split_id=sid,
                            seed=seed,
                            method="V2_MULTIBRANCH",
                            output_tag=output_tag,
                            mlp_test_acc=mlp_test_acc,
                        )
                        r["test_acc"] = float(v2_acc)
                        r["delta_vs_mlp"] = float(v2_acc - mlp_test_acc)
                        r["frac_confident"] = v2_info.get("fraction_test_no_correct")
                        r["frac_unreliable"] = v2_info.get("fraction_test_feat_branch")
                        r["frac_corrected"] = v2_info.get("fraction_test_full_branch")
                        r["tau"] = v2_info.get("selected_threshold")
                        r["rho"] = v2_info.get("selected_reliability_threshold")
                        r["runtime_sec"] = float(v2_time)
                        r["peak_rss_kb_after"] = _rss_kb()
                        outf.write(json.dumps(r, default=str) + "\n")
                        outf.flush()
                        print(f"    V2_MULTI: {v2_acc:.4f}")
                    except Exception as e:
                        r = _record_base(
                            dataset=ds,
                            split_id=sid,
                            seed=seed,
                            method="V2_MULTIBRANCH",
                            output_tag=output_tag,
                            mlp_test_acc=mlp_test_acc,
                        )
                        r["success"] = False
                        r["error"] = f"{e}\n{traceback.format_exc()}"
                        outf.write(json.dumps(r, default=str) + "\n")
                        outf.flush()

                    # --- FINAL_V3 ---
                    gates_to_run = [gate_mode] if gate_mode != "both" else ["heuristic", "adaptive"]
                    for gate_name in gates_to_run:
                        t0 = time.perf_counter()
                        try:
                            _, v3_acc, v3_info = final_method_v3(
                                data,
                                train_np,
                                val_np,
                                test_np,
                                mlp_probs=mlp_probs,
                                seed=seed,
                                mod=mod,
                                gate=gate_name,
                                split_id=sid,
                            )
                            v3_time = time.perf_counter() - t0
                            ca = v3_info["correction_analysis"]
                            bf = v3_info["branch_fractions"]
                            method_name = "FINAL_V3" if gate_mode != "both" else f"FINAL_V3_{gate_name.upper()}"
                            r = _record_base(
                                dataset=ds,
                                split_id=sid,
                                seed=seed,
                                method=method_name,
                                output_tag=output_tag,
                                mlp_test_acc=mlp_test_acc,
                            )
                            r["test_acc"] = float(v3_acc)
                            r["val_acc"] = v3_info.get("val_acc_v3")
                            r["delta_vs_mlp"] = float(v3_acc - mlp_test_acc)
                            r["n_helped"] = ca["n_helped"]
                            r["n_hurt"] = ca["n_hurt"]
                            r["n_changed"] = ca.get("n_changed")
                            r["correction_precision"] = ca["correction_precision"]
                            r["frac_confident"] = bf["confident_keep_mlp"]
                            r["frac_unreliable"] = bf["uncertain_unreliable_keep_mlp"]
                            r["frac_corrected"] = bf["uncertain_reliable_corrected"]
                            r["changed_fraction_test"] = ca.get("changed_fraction_test")
                            r["harmful_overwrite_rate"] = ca.get("harmful_overwrite_rate_non_neutral")
                            r["tau"] = v3_info.get("selected_tau")
                            r["rho"] = v3_info.get("selected_rho")
                            r["gate_mode"] = v3_info.get("gate_mode")
                            r["runtime_sec"] = float(v3_time)
                            r["mlp_runtime_sec"] = float(v3_info.get("runtime_sec", {}).get("mlp", mlp_time))
                            r["peak_rss_kb_after"] = _rss_kb()
                            r["v3_runtime_breakdown"] = v3_info.get("runtime_sec")
                            outf.write(json.dumps(r, default=str) + "\n")
                            outf.flush()
                            print(
                                f"    {method_name}: {v3_acc:.4f} "
                                f"[helped={ca['n_helped']} hurt={ca['n_hurt']}]"
                            )
                        except Exception as e:
                            r = _record_base(
                                dataset=ds,
                                split_id=sid,
                                seed=seed,
                                method="FINAL_V3",
                                output_tag=output_tag,
                                mlp_test_acc=mlp_test_acc,
                            )
                            r["success"] = False
                            r["error"] = f"{e}\n{traceback.format_exc()}"
                            outf.write(json.dumps(r, default=str) + "\n")
                            outf.flush()

                # --- External baselines (same MLP seed; independent training) ---
                for bname in external_baselines:
                    display = _EXTERNAL_DISPLAY.get(bname, bname)
                    t0 = time.perf_counter()
                    try:
                        torch.manual_seed(seed)
                        np.random.seed(seed)
                        br = run_baseline(
                            bname,
                            data,
                            train_idx,
                            val_idx,
                            test_idx,
                            seed=seed,
                            max_epochs=baseline_max_epochs,
                            patience=baseline_patience,
                        )
                        bt = time.perf_counter() - t0
                        r = _record_base(
                            dataset=ds,
                            split_id=sid,
                            seed=seed,
                            method=display,
                            output_tag=output_tag,
                            mlp_test_acc=mlp_test_acc,
                        )
                        r["test_acc"] = float(br.test_acc)
                        r["val_acc"] = float(br.val_acc)
                        pred = br.probs.argmax(dim=1).cpu().numpy()
                        r["train_acc"] = float((pred[train_np] == y_true[train_np]).mean())
                        r["delta_vs_mlp"] = float(br.test_acc - mlp_test_acc)
                        r["runtime_sec"] = float(bt)
                        r["peak_rss_kb_after"] = _rss_kb()
                        r["notes"] = f"external_baseline_key={bname}"
                        outf.write(json.dumps(r, default=str) + "\n")
                        outf.flush()
                        print(f"    {display}: test={br.test_acc:.4f} val={br.val_acc:.4f}")
                    except Exception as e:
                        r = _record_base(
                            dataset=ds,
                            split_id=sid,
                            seed=seed,
                            method=display,
                            output_tag=output_tag,
                            mlp_test_acc=mlp_test_acc,
                        )
                        r["success"] = False
                        r["error"] = f"{e}\n{traceback.format_exc()}"
                        outf.write(json.dumps(r, default=str) + "\n")
                        outf.flush()
                        print(f"    {bname} FAILED: {e}")

    return jsonl_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline comparison suite (Wulver).")
    parser.add_argument("--split-dir", default="data/splits")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["cora", "citeseer", "pubmed", "chameleon", "texas", "wisconsin"],
    )
    parser.add_argument("--splits", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--output-tag", required=True)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Per-run directory (default: outputs/baseline_comparison/<tag>/per_run)",
    )
    parser.add_argument("--gate", choices=["heuristic", "adaptive", "both"], default="heuristic")
    parser.add_argument("--baseline-max-epochs", type=int, default=500)
    parser.add_argument("--baseline-patience", type=int, default=100)
    parser.add_argument(
        "--external-baselines",
        nargs="*",
        default=None,
        help=(
            "Keys passed to run_baseline. Default: Begga heterophily set. "
            "Also supported: " + ", ".join(PAPER_GRAPH_BASELINE_METHODS) + ", appnp"
        ),
    )
    parser.add_argument(
        "--external-baselines-only",
        action="store_true",
        help=(
            "After MLP_ONLY, skip BASELINE_SGC_V1 / V2_MULTIBRANCH / FINAL_V3; "
            "only run --external-baselines (requires explicit baseline list)."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output-jsonl",
        default=None,
        help="Full path to JSONL (default: <output-dir>/runs.jsonl). Use for array jobs.",
    )
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(HERE, "..", ".."))
    out_dir = args.output_dir
    if out_dir is None:
        out_dir = os.path.join(repo_root, "outputs", "baseline_comparison", args.output_tag, "per_run")

    if args.dry_run:
        print("DRY RUN — would write to:", out_dir)
        print("  datasets:", args.datasets)
        print("  splits:", args.splits)
        print("  tag:", args.output_tag)
        print("  external:", args.external_baselines or list(BEGGA_HETEROPHILY_BASELINE_METHODS))
        print("  external_baselines_only:", args.external_baselines_only)
        return

    os.makedirs(out_dir, exist_ok=True)
    path = run_suite(
        args.datasets,
        args.splits,
        args.split_dir,
        args.output_tag,
        out_dir,
        gate_mode=args.gate,
        external_baselines=list(args.external_baselines)
        if args.external_baselines
        else None,
        baseline_max_epochs=args.baseline_max_epochs,
        baseline_patience=args.baseline_patience,
        output_jsonl=args.output_jsonl,
        external_baselines_only=bool(args.external_baselines_only),
    )
    print(f"\nWrote: {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Focused resubmission experiment runner.

Adds:
  - compact standard graph baselines (GCN, APPNP)
  - external baseline (H2GCN; compact in-repo adaptation)
  - external baseline (DeepGCN + PairNorm; lightweight in-repo implementation)
  - external baseline (FSGNN; lightweight in-repo implementation)
  - external baseline (GPRGNN; lightweight in-repo implementation)
  - true ablations for selective graph correction (SGC)
  - method/protocol records that are easier to cite in a rewritten manuscript

This runner intentionally writes to a separate output tag so it does not
overwrite the existing manuscript_final_validation package.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

import manuscript_runner as mr
from standard_node_baselines import run_baseline
from triple_trust_sgc import triple_trust_sgc_predictclass

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOGS_DIR = os.path.join(REPO_ROOT, "logs", "resubmission_runs")

CORE_DATASETS = ["cora", "citeseer", "pubmed", "chameleon", "texas", "wisconsin"]

DEFAULT_METHODS = [
    "mlp_only",
    "prop_only",
    "gcn",
    "gcn_pairnorm",
    "appnp",
    "h2gcn",
    "fsgnn",
    "gprgnn",
    "sgc_wu2019",
    "selective_graph_correction",
    "selective_graph_correction_structural",
    "gated_mlp_prop",
    "sgc_no_gate",
    "sgc_no_feature_similarity",
    "sgc_no_graph_neighbor",
    "sgc_no_compatibility",
    "sgc_mlp_plus_graph",
    "sgc_knn_enabled",
    "sgcs_margin_gate_only",
    "sgcs_no_far",
    "sgcs_far_only",
    "sgcs_no_gate",
    "triple_trust_sgc",
]


def _default_sgc_weight_candidates() -> List[Dict[str, float]]:
    return [
        {"b1": 1.0, "b2": 0.6, "b3": 0.0, "b4": 0.5, "b5": 0.3},
        {"b1": 1.0, "b2": 0.9, "b3": 0.0, "b4": 0.3, "b5": 0.2},
        {"b1": 1.0, "b2": 0.4, "b3": 0.0, "b4": 0.9, "b5": 0.5},
        {"b1": 1.2, "b2": 0.5, "b3": 0.0, "b4": 0.4, "b5": 0.2},
    ]


def _variant_kwargs(method: str) -> Dict[str, Any]:
    base = _default_sgc_weight_candidates()
    if method == "selective_graph_correction":
        return {}
    if method == "sgc_no_gate":
        return {"threshold_candidates": [1.1]}
    if method == "sgc_no_feature_similarity":
        return {
            "weight_candidates": [{**cfg, "b2": 0.0} for cfg in base],
        }
    if method == "sgc_no_graph_neighbor":
        return {
            "weight_candidates": [{**cfg, "b4": 0.0} for cfg in base],
        }
    if method == "sgc_no_compatibility":
        return {
            "weight_candidates": [{**cfg, "b5": 0.0} for cfg in base],
        }
    if method == "sgc_mlp_plus_graph":
        return {
            "weight_candidates": [
                {"b1": 1.0, "b2": 0.0, "b3": 0.0, "b4": 0.9, "b5": 0.0},
                {"b1": 1.0, "b2": 0.0, "b3": 0.0, "b4": 0.5, "b5": 0.0},
                {"b1": 1.2, "b2": 0.0, "b3": 0.0, "b4": 0.4, "b5": 0.0},
            ],
        }
    if method == "sgc_knn_enabled":
        return {
            "enable_feature_knn": True,
            "feature_knn_k": 5,
            "weight_candidates": [
                {"b1": 1.0, "b2": 0.6, "b3": 0.2, "b4": 0.5, "b5": 0.3},
                {"b1": 1.0, "b2": 0.4, "b3": 0.4, "b4": 0.9, "b5": 0.5},
                {"b1": 1.0, "b2": 0.9, "b3": 0.2, "b4": 0.3, "b5": 0.2},
            ],
        }
    raise ValueError(f"Unknown SGC variant: {method}")


def _structural_variant_kwargs(method: str) -> Dict[str, Any]:
    base = [
        {"b1": 1.0, "b2": 0.4, "b3": 0.0, "b4": 0.2, "b5": 0.3, "b6": 0.9},
        {"b1": 1.0, "b2": 0.6, "b3": 0.0, "b4": 0.3, "b5": 0.2, "b6": 0.8},
        {"b1": 1.1, "b2": 0.4, "b3": 0.0, "b4": 0.1, "b5": 0.2, "b6": 1.0},
    ]
    if method == "selective_graph_correction_structural":
        return {"use_structure_quality_gate": True, "weight_candidates": base}
    if method == "sgcs_margin_gate_only":
        return {"use_structure_quality_gate": False, "weight_candidates": base}
    if method == "sgcs_no_far":
        return {
            "use_structure_quality_gate": True,
            "weight_candidates": [{**cfg, "b6": 0.0, "b4": max(float(cfg["b4"]), 0.6)} for cfg in base],
        }
    if method == "sgcs_far_only":
        return {
            "use_structure_quality_gate": True,
            "weight_candidates": [{**cfg, "b4": 0.0, "b6": max(float(cfg["b6"]), 0.8)} for cfg in base],
        }
    if method == "sgcs_no_gate":
        return {
            "use_structure_quality_gate": False,
            "threshold_candidates": [1.1],
            "weight_candidates": base,
        }
    raise ValueError(f"Unknown structural SGC variant: {method}")


def _build_record(**kwargs: Any) -> Dict[str, Any]:
    rec = mr._build_record(  # type: ignore[attr-defined]
        dataset_key=kwargs["dataset_key"],
        num_nodes=kwargs["num_nodes"],
        num_edges=kwargs["num_edges"],
        num_classes=kwargs["num_classes"],
        train_count=kwargs["train_count"],
        val_count=kwargs["val_count"],
        test_count=kwargs["test_count"],
        split_path=kwargs["split_path"],
        split_id=kwargs["split_id"],
        repeat_id=kwargs["repeat_id"],
        seed=kwargs["seed"],
        method=kwargs["method"],
    )
    rec["method_family"] = kwargs.get("method_family", "primary")
    rec["method_variant"] = kwargs.get("method_variant", kwargs["method"])
    rec["config_json"] = kwargs.get("config_json")
    rec["val_metric_source"] = kwargs.get("val_metric_source", "val_acc")
    return rec


def run_resubmission(
    datasets: Optional[List[str]] = None,
    split_ids: Optional[List[int]] = None,
    repeats: int = 1,
    methods: Optional[List[str]] = None,
    split_dir: Optional[str] = None,
    output_tag: str = "resubmission_core_r1",
    rs_num_samples: int = 30,
    rs_n_splits: int = 3,
) -> Dict[str, Any]:
    if datasets is None:
        datasets = CORE_DATASETS[:]
    if split_ids is None:
        split_ids = list(range(10))
    if methods is None:
        methods = DEFAULT_METHODS[:]

    os.makedirs(LOGS_DIR, exist_ok=True)
    mod = mr._load_full_investigate_module()  # type: ignore[attr-defined]
    setattr(mod, "LOG_DIR", LOGS_DIR)
    coverage = mr.check_split_coverage(datasets, split_ids, split_dir)  # type: ignore[attr-defined]
    runnable_datasets = [ds for ds in datasets if coverage[ds]]
    if not runnable_datasets:
        raise SystemExit("No runnable datasets found for resubmission package.")

    dataset_meta: Dict[str, Dict[str, Any]] = {}
    all_records: List[Dict[str, Any]] = []
    runs_path = os.path.join(LOGS_DIR, f"comparison_runs_{output_tag}.jsonl")
    runs_file = open(runs_path, "w", encoding="utf-8")
    t_wall = time.perf_counter()
    total_runs = 0

    for dataset_key in runnable_datasets:
        print(f"\n=== Dataset: {dataset_key} ===")
        dataset, data = mr._load_dataset_and_data(mod, dataset_key)  # type: ignore[attr-defined]
        device = data.x.device
        n_total = int(data.num_nodes)
        e_total = int(data.edge_index.size(1))
        num_classes = int(data.y.max().item()) + 1
        dataset_meta[dataset_key] = {
            "num_nodes": n_total,
            "num_edges": e_total,
            "num_classes": num_classes,
            "avg_degree": float(e_total) / float(max(n_total, 1)),
            "homophily": mr._edge_homophily(data),  # type: ignore[attr-defined]
        }

        for split_id in coverage[dataset_key]:
            for repeat in range(repeats):
                run_seed = 1337 + split_id
                current_seed = run_seed * 100 + repeat
                torch.manual_seed(current_seed)
                np.random.seed(current_seed)
                print(f"  split={split_id} repeat={repeat+1}/{repeats} seed={current_seed}")
                train_idx, val_idx, test_idx, split_path = mr._load_split_npz(  # type: ignore[attr-defined]
                    dataset_key, split_id, device, split_dir
                )
                split_load = np.load(split_path)
                rec_base = dict(
                    dataset_key=dataset_key,
                    num_nodes=n_total,
                    num_edges=e_total,
                    num_classes=num_classes,
                    train_count=int(split_load["train_mask"].sum()),
                    val_count=int(split_load["val_mask"].sum()),
                    test_count=int(split_load["test_mask"].sum()),
                    split_path=split_path,
                    split_id=split_id,
                    repeat_id=repeat,
                    seed=current_seed,
                )

                train_np = mod._to_numpy_idx(train_idx)
                val_np = mod._to_numpy_idx(val_idx)
                test_np = mod._to_numpy_idx(test_idx)

                mlp_probs = None
                acc_val_mlp = None
                acc_test_mlp = None
                prop_params = None
                prop_dict = None
                tuning_time_prop = 0.0

                # MLP
                if any(
                    m in methods
                    for m in [
                        "mlp_only",
                        "selective_graph_correction",
                        "selective_graph_correction_structural",
                        "triple_trust_sgc",
                        "gated_mlp_prop",
                        "sgc_no_gate",
                        "sgc_no_feature_similarity",
                        "sgc_no_graph_neighbor",
                        "sgc_no_compatibility",
                        "sgc_mlp_plus_graph",
                        "sgc_knn_enabled",
                        "sgcs_margin_gate_only",
                        "sgcs_no_far",
                        "sgcs_far_only",
                        "sgcs_no_gate",
                    ]
                ):
                    rec = _build_record(**rec_base, method="mlp_only", method_family="baseline")
                    t0 = time.perf_counter()
                    try:
                        mlp_cfg = getattr(mod, "DEFAULT_MANUSCRIPT_MLP_KWARGS")
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
                        mlp_time = time.perf_counter() - t0
                        preds = mlp_probs.argmax(dim=1)
                        acc_val_mlp = float((preds[val_idx] == data.y[val_idx]).float().mean().item())
                        acc_test_mlp = float((preds[test_idx] == data.y[test_idx]).float().mean().item())
                        rec.update(
                            {
                                "val_acc": acc_val_mlp,
                                "test_acc": acc_test_mlp,
                                "method_runtime_sec": mlp_time,
                                "total_runtime_sec": mlp_time,
                                "parameter_source": "DEFAULT_MANUSCRIPT_MLP_KWARGS",
                                "config_json": json.dumps(mlp_cfg, sort_keys=True),
                            }
                        )
                    except Exception as e:
                        rec["success"] = False
                        rec["notes"] = str(e)
                    if "mlp_only" in methods:
                        all_records.append(rec)
                        runs_file.write(json.dumps(rec) + "\n")
                        runs_file.flush()
                        total_runs += 1

                # propagation-only
                if any(m in methods for m in ["prop_only", "gated_mlp_prop"]):
                    t0 = time.perf_counter()
                    try:
                        params_prop, _, _ = mod.robust_random_search(
                            mod.predictclass,
                            data,
                            train_np,
                            num_samples=rs_num_samples,
                            n_splits=rs_n_splits,
                            repeats=1,
                            seed=current_seed,
                            log_file=None,
                            mlp_probs=None,
                        )
                        tuning_time_prop = time.perf_counter() - t0
                        prop_params = params_prop
                        prop_dict = {f"a{i+1}": p for i, p in enumerate(params_prop)}
                    except Exception as e:
                        print(f"    [WARN] prop tuning failed: {e}")

                    if "prop_only" in methods:
                        rec = _build_record(**rec_base, method="prop_only", method_family="baseline")
                        if prop_dict is None:
                            rec["success"] = False
                            rec["notes"] = "prop tuning failed"
                        else:
                            t1 = time.perf_counter()
                            try:
                                _, acc_val_prop = mod.predictclass(
                                    data,
                                    train_np,
                                    val_np,
                                    **prop_dict,
                                    seed=current_seed,
                                    mlp_probs=None,
                                    log_file=None,
                                )
                                _, acc_test_prop = mod.predictclass(
                                    data,
                                    train_np,
                                    test_np,
                                    **prop_dict,
                                    seed=current_seed,
                                    mlp_probs=None,
                                    log_file=None,
                                )
                                method_time = time.perf_counter() - t1
                                rec.update(
                                    {
                                        "val_acc": float(acc_val_prop),
                                        "test_acc": float(acc_test_prop),
                                        "tuning_runtime_sec": tuning_time_prop,
                                        "method_runtime_sec": method_time,
                                        "total_runtime_sec": tuning_time_prop + method_time,
                                        "parameter_source": "robust_random_search",
                                        "a_values": prop_params,
                                    }
                                )
                            except Exception as e:
                                rec["success"] = False
                                rec["notes"] = str(e)
                        all_records.append(rec)
                        runs_file.write(json.dumps(rec) + "\n")
                        runs_file.flush()
                        total_runs += 1

                # standard baselines
                for method in [
                    m
                    for m in methods
                    if m in {"gcn", "gcn_pairnorm", "appnp", "h2gcn", "fsgnn", "gprgnn", "sgc_wu2019"}
                ]:
                    rec = _build_record(**rec_base, method=method, method_family="baseline")
                    t0 = time.perf_counter()
                    try:
                        baseline_kwargs: Dict[str, Any] = (
                            {"max_epochs": 500, "patience": 100} if method == "gcn" else {}
                        )
                        result = run_baseline(
                            method,
                            data,
                            train_idx,
                            val_idx,
                            test_idx,
                            seed=current_seed,
                            **baseline_kwargs,
                        )
                        method_time = time.perf_counter() - t0
                        rec.update(
                            {
                                "val_acc": result.val_acc,
                                "test_acc": result.test_acc,
                                "method_runtime_sec": method_time,
                                "total_runtime_sec": method_time,
                                "parameter_source": "small_val_grid_search",
                                "config_json": json.dumps(result.best_config, sort_keys=True),
                            }
                        )
                    except Exception as e:
                        rec["success"] = False
                        rec["notes"] = str(e)
                    all_records.append(rec)
                    runs_file.write(json.dumps(rec) + "\n")
                    runs_file.flush()
                    total_runs += 1

                # SGC family and ablations
                sgc_methods = [
                    m
                    for m in methods
                    if m
                    in {
                        "selective_graph_correction",
                        "sgc_no_gate",
                        "sgc_no_feature_similarity",
                        "sgc_no_graph_neighbor",
                        "sgc_no_compatibility",
                        "sgc_mlp_plus_graph",
                        "sgc_knn_enabled",
                        "selective_graph_correction_structural",
                        "sgcs_margin_gate_only",
                        "sgcs_no_far",
                        "sgcs_far_only",
                        "sgcs_no_gate",
                        "triple_trust_sgc",
                    }
                ]
                for method in sgc_methods:
                    rec = _build_record(
                        **rec_base,
                        method=method,
                        method_family="ablation" if method != "selective_graph_correction" else "primary",
                        method_variant=method,
                    )
                    t0 = time.perf_counter()
                    try:
                        if method == "triple_trust_sgc":
                            variant_kwargs = {
                                "alpha_class": 0.5,
                                "gamma_source": 2.0,
                                "lambda_deg": 0.2,
                                "lambda_proto": 0.3,
                                "rho_target": 0.15,
                                "kappa_update": 0.65,
                                "refresh_fraction": 0.25,
                                "compatibility_update_eta": 0.2,
                                "enable_target_labelability_gate": True,
                                "q_weights": [0.35, 0.30, 0.25, 0.10],
                                "b1": 1.0,
                                "b2": 0.6,
                                "b4": 0.5,
                                "b5": 0.3,
                                "max_refresh_rounds": 3,
                            }
                            _, acc_sgc, info = triple_trust_sgc_predictclass(
                                data,
                                train_np,
                                val_np,
                                test_np,
                                mlp_probs=mlp_probs,
                                seed=current_seed,
                                mod=mod,
                                **variant_kwargs,
                            )
                        elif method.startswith("sgcs") or method == "selective_graph_correction_structural":
                            variant_kwargs = _structural_variant_kwargs(method)
                            _, acc_sgc, info = mod.selective_graph_correction_structural_predictclass(
                                data,
                                train_np,
                                val_np,
                                test_np,
                                mlp_probs=mlp_probs,
                                seed=current_seed,
                                log_file=None,
                                log_prefix=f"{method}_{dataset_key}_s{split_id}_r{repeat}",
                                diagnostics_run_id=f"{method}_{dataset_key}_s{split_id}_r{repeat}",
                                dataset_key_for_logs=f"{dataset_key}_{method}",
                                write_node_diagnostics=True,
                                **variant_kwargs,
                            )
                        else:
                            variant_kwargs = _variant_kwargs(method)
                            _, acc_sgc, info = mod.selective_graph_correction_predictclass(
                                data,
                                train_np,
                                val_np,
                                test_np,
                                mlp_probs=mlp_probs,
                                seed=current_seed,
                                log_file=None,
                                log_prefix=f"{method}_{dataset_key}_s{split_id}_r{repeat}",
                                diagnostics_run_id=f"{method}_{dataset_key}_s{split_id}_r{repeat}",
                                dataset_key_for_logs=f"{dataset_key}_{method}",
                                write_node_diagnostics=True,
                                **variant_kwargs,
                            )
                        method_time = time.perf_counter() - t0
                        weights = info.get("selected_weights", {}) or {}
                        frac_u = float(info.get("fraction_test_nodes_uncertain", 0.0) or 0.0)
                        frac_changed = float(info.get("fraction_test_nodes_changed_from_mlp", 0.0) or 0.0)
                        n_test = int(len(test_np))
                        rec.update(
                            {
                                "val_acc": float(info.get("val_acc_selective", 0.0)),
                                "test_acc": float(acc_sgc),
                                "method_runtime_sec": method_time,
                                "total_runtime_sec": method_time,
                                "parameter_source": info.get("mode", "feature_first_selective_weighted_correction_v1"),
                                "sgc_threshold": float(info.get("selected_threshold_high", 0.0)),
                                "sgc_b_weights": [
                                    float(weights.get("b1", 0.0)),
                                    float(weights.get("b2", 0.0)),
                                    float(weights.get("b3", 0.0)),
                                    float(weights.get("b4", 0.0)),
                                    float(weights.get("b5", 0.0)),
                                    float(weights.get("b6", 0.0)),
                                ],
                                "sgc_n_uncertain": int(round(frac_u * n_test)),
                                "sgc_n_changed": int(round(frac_changed * n_test)),
                                "sgc_diagnostics_path": info.get("diagnostics_path"),
                                "config_json": json.dumps(variant_kwargs, sort_keys=True, default=str),
                            }
                        )
                        if method == "triple_trust_sgc":
                            rec.update(
                                {
                                    "tt_mean_q_corrected": float(info.get("mean_q_corrected", 0.0)),
                                    "tt_mean_source_trust_pseudo": float(info.get("mean_source_trust_pseudo", 0.0)),
                                    "tt_average_target_labelability": float(
                                        info.get("average_target_labelability", 0.0)
                                    ),
                                    "tt_pseudo_update_eligible_count": int(info.get("pseudo_update_eligible_count", 0)),
                                    "tt_class_trust_vector": info.get("class_trust_vector"),
                                    "tt_per_class_trusted_mass": info.get("per_class_trusted_mass"),
                                    "tt_trust_neighbor_concentration_mean": float(
                                        info.get("trust_neighbor_concentration_mean", 0.0)
                                    ),
                                    "tt_compatibility_update_stats": info.get("compatibility_update_stats"),
                                    "tt_n_helped": int(info.get("correction_analysis", {}).get("n_helped", 0)),
                                    "tt_n_hurt": int(info.get("correction_analysis", {}).get("n_hurt", 0)),
                                }
                            )
                    except Exception as e:
                        rec["success"] = False
                        rec["notes"] = str(e)
                    all_records.append(rec)
                    runs_file.write(json.dumps(rec) + "\n")
                    runs_file.flush()
                    total_runs += 1

                # learned gate baseline
                if "gated_mlp_prop" in methods:
                    rec = _build_record(**rec_base, method="gated_mlp_prop", method_family="ablation")
                    t0 = time.perf_counter()
                    try:
                        _, acc_gate, info = mod.gated_mlp_prop_predictclass(
                            data,
                            train_np,
                            val_np,
                            test_np,
                            prop_params=prop_dict,
                            mlp_probs=mlp_probs,
                            seed=current_seed,
                            log_file=None,
                            diagnostics_run_id=f"gated_mlp_prop_{dataset_key}_s{split_id}_r{repeat}",
                            dataset_key_for_logs=f"{dataset_key}_gated_mlp_prop",
                        )
                        method_time = time.perf_counter() - t0
                        rec.update(
                            {
                                "val_acc": float(
                                    info.get("val_acc_gated", 0.0)
                                    or info.get("val_acc_final", 0.0)
                                    or 0.0
                                ),
                                "test_acc": float(acc_gate),
                                "method_runtime_sec": method_time,
                                "total_runtime_sec": method_time,
                                "parameter_source": info.get("mode", "gated_mlp_prop"),
                                "sgc_diagnostics_path": info.get("diagnostics_path"),
                                "config_json": json.dumps(
                                    {
                                        "prop_params_present": prop_dict is not None,
                                        "gate_mode": (
                                            info.get("gate_state", {}).get("mode")
                                            if isinstance(info.get("gate_state"), dict)
                                            else None
                                        ),
                                    },
                                    sort_keys=True,
                                ),
                            }
                        )
                    except Exception as e:
                        rec["success"] = False
                        rec["notes"] = str(e)
                    all_records.append(rec)
                    runs_file.write(json.dumps(rec) + "\n")
                    runs_file.flush()
                    total_runs += 1

    runs_file.close()
    wall_time = time.perf_counter() - t_wall
    print(f"\nCompleted {total_runs} runs in {wall_time:.1f}s")
    return {
        "runs_path": runs_path,
        "dataset_meta": dataset_meta,
        "total_runs": total_runs,
        "wall_time_sec": wall_time,
        "output_tag": output_tag,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Focused resubmission experiment runner.")
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--splits", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--output-tag", default="resubmission_core_r1")
    parser.add_argument("--rs-num-samples", type=int, default=30)
    parser.add_argument("--rs-n-splits", type=int, default=3)
    args = parser.parse_args()

    out = run_resubmission(
        datasets=args.datasets,
        split_ids=args.splits,
        repeats=args.repeats,
        methods=args.methods,
        split_dir=args.split_dir,
        output_tag=args.output_tag,
        rs_num_samples=args.rs_num_samples,
        rs_n_splits=args.rs_n_splits,
    )
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()

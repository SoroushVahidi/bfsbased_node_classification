#!/usr/bin/env python3
"""
Manuscript phase experiment runner.

Compares:
  - mlp_only
  - prop_only
  - selective_graph_correction
  - prop_mlp_select  (optional)

Datasets priority:
  Tier 1: cora, citeseer
  Tier 2: chameleon, texas, actor, cornell
  Tier 3: wisconsin
  (squirrel / pubmed only if runtime allows)

Splits: 0..9  |  Repeats: 1 (state that repeats=3 should run on Wulver)

Usage:
    python manuscript_runner.py [--split-dir PATH] [--datasets DATASET ...]
                                [--splits 0..9] [--repeats 1]
                                [--methods mlp_only prop_only selective_graph_correction]
                                [--output-tag TAG]
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


# -----------------------------------------------------------------------
# Module helpers (mirrors comparison_runner.py patterns)
# -----------------------------------------------------------------------

def _get_repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _load_full_investigate_module():
    """Dynamically load bfsbased-full-investigate-homophil.py."""
    here = _get_repo_root()
    path = os.path.join(here, "bfsbased-full-investigate-homophil.py")
    spec = importlib.util.spec_from_file_location("bfs_full_investigate", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    setattr(mod, "DATASET_KEY", "texas")  # satisfy top-level dataset-load code
    spec.loader.exec_module(mod)          # type: ignore[arg-type]
    return mod


def _load_dataset_and_data(mod, dataset_key: str):
    root = "data/"
    dataset = mod.load_dataset(dataset_key, root)
    data = dataset[0]
    data.edge_index = torch.cat([data.edge_index, data.edge_index.flip(0)], dim=1)
    data.edge_index = torch.unique(data.edge_index, dim=1)
    return dataset, data


def _resolve_split_file(dataset_key: str, split_id: int, split_dir: Optional[str]) -> Optional[str]:
    # GEO-GCN uses `film_split_*` for the Actor dataset.
    split_prefix = dataset_key.lower()
    if split_prefix == "actor":
        split_prefix = "film"
    fname = f"{split_prefix}_split_0.6_0.2_{split_id}.npz"
    candidates = []
    if split_dir:
        candidates.append(os.path.join(split_dir, fname))
    env_dir = os.environ.get("GEO_GCN_SPLIT_DIR")
    if env_dir:
        candidates.append(os.path.join(env_dir, fname))
    repo_root = _get_repo_root()
    candidates.append(os.path.join(repo_root, fname))
    data_root = os.path.join(repo_root, "data")
    if os.path.isdir(data_root):
        for root_d, _dirs, files in os.walk(data_root):
            if fname in files:
                candidates.append(os.path.join(root_d, fname))
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return None


def _load_split_npz(
    dataset_key: str, split_id: int, device: torch.device, split_dir: Optional[str]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    path = _resolve_split_file(dataset_key, split_id, split_dir)
    if path is None:
        raise FileNotFoundError(
            f"Split file not found for dataset={dataset_key}, split_id={split_id}"
        )
    split = np.load(path)
    to_t = lambda mask: torch.as_tensor(np.where(mask)[0], dtype=torch.long, device=device)
    return to_t(split["train_mask"]), to_t(split["val_mask"]), to_t(split["test_mask"]), path


def _edge_homophily(data) -> float:
    """Simple edge homophily H = P(y_u == y_v over edges)."""
    try:
        y = data.y.detach().cpu().numpy()
        ei = data.edge_index.detach().cpu().numpy()
        src, dst = ei[0], ei[1]
        mask = (y[src] >= 0) & (y[dst] >= 0)
        if mask.sum() == 0:
            return float("nan")
        return float((y[src[mask]] == y[dst[mask]]).mean())
    except Exception:
        return float("nan")


# -----------------------------------------------------------------------
# Split coverage check
# -----------------------------------------------------------------------

ALL_CANDIDATE_DATASETS = [
    "cora", "citeseer",
    "chameleon", "texas", "actor", "cornell",
    "wisconsin",
    "squirrel", "pubmed",
]

PRIORITY_ORDER = [
    "cora", "citeseer",
    "chameleon", "texas", "actor", "cornell",
    "wisconsin",
    "squirrel", "pubmed",
]


def check_split_coverage(
    datasets: List[str],
    split_ids: List[int],
    split_dir: Optional[str],
) -> Dict[str, List[int]]:
    """
    Returns a dict: dataset -> list of available split_ids.
    Prints a coverage report.
    """
    print("\n=== Split Coverage Check ===")
    coverage: Dict[str, List[int]] = {}
    for ds in datasets:
        avail = []
        for sid in split_ids:
            if _resolve_split_file(ds, sid, split_dir) is not None:
                avail.append(sid)
        coverage[ds] = avail
        n_avail = len(avail)
        n_total = len(split_ids)
        status = "FULL" if n_avail == n_total else (f"PARTIAL ({n_avail}/{n_total})" if n_avail > 0 else "MISSING")
        print(f"  {ds:12s}: {status}")
    print()
    return coverage


# -----------------------------------------------------------------------
# Per-run record builder
# -----------------------------------------------------------------------

def _build_record(
    dataset_key: str,
    num_nodes: int,
    num_edges: int,
    num_classes: int,
    train_count: int,
    val_count: int,
    test_count: int,
    split_path: str,
    split_id: int,
    repeat_id: int,
    seed: int,
    method: str,
) -> Dict[str, Any]:
    return {
        "method": method,
        "dataset": dataset_key,
        "split_id": split_id,
        "repeat_id": repeat_id,
        "seed": seed,
        "success": True,
        "notes": "",
        "split_file": split_path,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_classes": num_classes,
        "train_count": train_count,
        "val_count": val_count,
        "test_count": test_count,
        "val_acc": None,
        "test_acc": None,
        "tuning_runtime_sec": 0.0,
        "method_runtime_sec": 0.0,
        "total_runtime_sec": 0.0,
        "parameter_source": None,
        "a_values": None,
        "sgc_threshold": None,
        "sgc_b_weights": None,
        "sgc_n_uncertain": None,
        "sgc_n_changed": None,
        "sgc_diagnostics_path": None,
    }


# -----------------------------------------------------------------------
# Main experiment runner
# -----------------------------------------------------------------------

def run_manuscript_experiment(
    datasets: Optional[List[str]] = None,
    split_ids: Optional[List[int]] = None,
    repeats: int = 1,
    methods: Optional[List[str]] = None,
    split_dir: Optional[str] = None,
    output_tag: str = "manuscript_phase",
    rs_num_samples: int = 30,
    rs_n_splits: int = 3,
    sgc_tune_n_samples: int = 25,
):
    """
    Run the manuscript-phase experiment.

    For repeats=1 (default) the run is feasible on a laptop/CI.
    For the final manuscript-quality benchmark, use repeats=3 on Wulver.
    """
    if datasets is None:
        datasets = PRIORITY_ORDER[:]
    if split_ids is None:
        split_ids = list(range(10))
    if methods is None:
        methods = ["mlp_only", "prop_only", "selective_graph_correction", "prop_mlp_select"]

    os.makedirs("logs", exist_ok=True)

    # ------------------------------------------------------------------
    # Step 0: Check split coverage and filter datasets
    # ------------------------------------------------------------------
    coverage = check_split_coverage(datasets, split_ids, split_dir)
    runnable_datasets = [ds for ds in datasets if len(coverage[ds]) > 0]
    excluded = [ds for ds in datasets if len(coverage[ds]) == 0]

    if excluded:
        print(f"Excluding datasets with NO canonical splits: {excluded}")
    if not runnable_datasets:
        print("No datasets have any canonical splits available. Aborting.")
        sys.exit(1)

    print(f"Datasets to run: {runnable_datasets}")
    print(f"Methods: {methods}")
    print(f"Split ids per dataset: varies (only available ids run)")
    print(f"Repeats: {repeats}")
    if repeats < 3:
        print(
            "NOTE: repeats=1 produces single-seed estimates. "
            "Run repeats=3 on Wulver for manuscript-quality stability estimates."
        )

    # ------------------------------------------------------------------
    # Load the BFS module once
    # ------------------------------------------------------------------
    print("\nLoading bfsbased-full-investigate-homophil module...")
    mod = _load_full_investigate_module()

    # ------------------------------------------------------------------
    # Output files
    # ------------------------------------------------------------------
    ts = time.strftime("%Y%m%d_%H%M%S")
    runs_path = os.path.join("logs", f"comparison_runs_{output_tag}.jsonl")
    summary_path = os.path.join("logs", f"comparison_summary_{output_tag}.json")
    fairness_path = os.path.join("logs", f"comparison_fairness_report_{output_tag}.md")
    gain_path = os.path.join("logs", f"gain_over_mlp_{output_tag}.csv")
    correction_path = os.path.join("logs", f"correction_behavior_{output_tag}.csv")
    regime_path = os.path.join("logs", f"regime_analysis_{output_tag}.csv")

    runs_file = open(runs_path, "w", encoding="utf-8")
    all_records: List[Dict[str, Any]] = []
    dataset_meta: Dict[str, Dict[str, Any]] = {}

    total_runs_done = 0
    t_wall_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Experiment loop
    # ------------------------------------------------------------------
    for dataset_key in runnable_datasets:
        avail_splits = coverage[dataset_key]
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_key}  ({len(avail_splits)} splits)")
        print(f"{'='*60}")

        try:
            dataset, data = _load_dataset_and_data(mod, dataset_key)
        except Exception as e:
            print(f"  [ERROR] Failed to load dataset {dataset_key}: {e}")
            continue

        device = data.x.device
        N_total = int(data.num_nodes)
        E_total = int(data.edge_index.size(1))
        num_classes = int(data.y.max().item()) + 1
        homophily = _edge_homophily(data)

        dataset_meta[dataset_key] = {
            "num_nodes": N_total,
            "num_edges": E_total,
            "num_classes": num_classes,
            "avg_degree": float(E_total) / float(max(N_total, 1)),
            "homophily": homophily,
        }

        for split_id in avail_splits:
            for repeat in range(repeats):
                run_seed = 1337 + split_id
                current_seed = run_seed * 100 + repeat
                torch.manual_seed(current_seed)
                np.random.seed(current_seed)

                print(f"  split={split_id}  repeat={repeat+1}/{repeats}  seed={current_seed}")

                try:
                    train_idx, val_idx, test_idx, split_path = _load_split_npz(
                        dataset_key, split_id, device, split_dir
                    )
                except FileNotFoundError as e:
                    print(f"    [SKIP] {e}")
                    continue

                train_np = mod._to_numpy_idx(train_idx)
                val_np = mod._to_numpy_idx(val_idx)
                test_np = mod._to_numpy_idx(test_idx)

                split_load = np.load(split_path)
                rec_kwargs = dict(
                    dataset_key=dataset_key,
                    num_nodes=N_total,
                    num_edges=E_total,
                    num_classes=num_classes,
                    train_count=int(split_load["train_mask"].sum()),
                    val_count=int(split_load["val_mask"].sum()),
                    test_count=int(split_load["test_mask"].sum()),
                    split_path=split_path,
                    split_id=split_id,
                    repeat_id=repeat,
                    seed=current_seed,
                )

                # --------------------------------------------------------
                # 1) MLP only
                # --------------------------------------------------------
                mlp_probs = None
                preds_mlp_all = None
                acc_val_mlp = None
                acc_test_mlp = None

                if "mlp_only" in methods or "selective_graph_correction" in methods or "prop_mlp_select" in methods:
                    rec = _build_record(**rec_kwargs, method="mlp_only")
                    t0 = time.perf_counter()
                    try:
                        mlp_probs, _ = mod.train_mlp_and_predict(
                            data, train_idx,
                            hidden=64, layers=2, dropout=0.5, lr=0.01, epochs=300,
                            log_file=None, val_indices=val_idx, patience=30,
                        )
                        mlp_time = time.perf_counter() - t0
                        _, preds_mlp_all = mlp_probs.max(dim=1)
                        acc_val_mlp = float((preds_mlp_all[val_idx] == data.y[val_idx]).float().mean())
                        acc_test_mlp = float((preds_mlp_all[test_idx] == data.y[test_idx]).float().mean())
                        rec.update({
                            "val_acc": acc_val_mlp,
                            "test_acc": acc_test_mlp,
                            "tuning_runtime_sec": 0.0,
                            "method_runtime_sec": mlp_time,
                            "total_runtime_sec": mlp_time,
                            "parameter_source": "mlp_early_stopping_val",
                        })
                    except Exception as e:
                        rec["success"] = False
                        rec["notes"] = str(e)
                        print(f"    [WARN] mlp_only failed: {e}")

                    if "mlp_only" in methods:
                        all_records.append(rec)
                        runs_file.write(json.dumps(rec) + "\n")
                        runs_file.flush()
                    total_runs_done += 1

                # --------------------------------------------------------
                # 2) Prop only
                # --------------------------------------------------------
                params_prop = None
                dict_params_prop = None
                tuning_time_prop = 0.0

                if "prop_only" in methods or "prop_mlp_select" in methods:
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
                        dict_params_prop = {f"a{i+1}": p for i, p in enumerate(params_prop)}
                    except Exception as e:
                        tuning_time_prop = time.perf_counter() - t0
                        print(f"    [WARN] prop tuning failed: {e}")

                    if "prop_only" in methods:
                        rec = _build_record(**rec_kwargs, method="prop_only")
                        if dict_params_prop is not None:
                            t1 = time.perf_counter()
                            try:
                                _, acc_val_prop = mod.predictclass(
                                    data, train_np, val_np,
                                    **dict_params_prop, seed=current_seed,
                                    mlp_probs=None, log_file=None,
                                )
                                _, acc_test_prop = mod.predictclass(
                                    data, train_np, test_np,
                                    **dict_params_prop, seed=current_seed,
                                    mlp_probs=None, log_file=None,
                                )
                                method_time = time.perf_counter() - t1
                                rec.update({
                                    "val_acc": float(acc_val_prop),
                                    "test_acc": float(acc_test_prop),
                                    "tuning_runtime_sec": tuning_time_prop,
                                    "method_runtime_sec": method_time,
                                    "total_runtime_sec": tuning_time_prop + method_time,
                                    "parameter_source": "robust_random_search",
                                    "a_values": params_prop,
                                })
                            except Exception as e:
                                rec["success"] = False
                                rec["notes"] = str(e)
                                print(f"    [WARN] prop_only failed: {e}")
                        else:
                            rec["success"] = False
                            rec["notes"] = "prop tuning failed"
                        all_records.append(rec)
                        runs_file.write(json.dumps(rec) + "\n")
                        runs_file.flush()
                    total_runs_done += 1

                # --------------------------------------------------------
                # 3) Selective graph correction
                # --------------------------------------------------------
                if "selective_graph_correction" in methods and mlp_probs is not None:
                    rec = _build_record(**rec_kwargs, method="selective_graph_correction")
                    t0 = time.perf_counter()
                    try:
                        # New SGC signature (feature-first selective weighted correction).
                        _, acc_sgc, info = mod.selective_graph_correction_predictclass(
                            data,
                            train_np,
                            val_np,
                            test_np,
                            mlp_probs=mlp_probs,
                            seed=current_seed,
                            log_file=None,
                            log_prefix=f"sgc_{dataset_key}_s{split_id}_r{repeat}",
                            diagnostics_run_id=f"{dataset_key}_s{split_id}_r{repeat}",
                            dataset_key_for_logs=dataset_key,
                            write_node_diagnostics=True,
                        )
                        method_time = time.perf_counter() - t0
                        # Pull the tuned/selected configuration + summary stats from SGC.
                        val_acc_sgc = float(info.get("val_acc_selective", 0.0))
                        n_test = int(len(test_np))
                        frac_u = float(info.get("fraction_test_nodes_uncertain", 0.0) or 0.0)
                        frac_changed = float(info.get("fraction_test_nodes_changed_from_mlp", 0.0) or 0.0)
                        sgc_n_uncertain = int(round(frac_u * n_test))
                        sgc_n_changed = int(round(frac_changed * n_test))

                        weights = info.get("selected_weights", {}) or {}
                        sgc_b_weights = [
                            float(weights.get("b1", 0.0)),
                            float(weights.get("b2", 0.0)),
                            float(weights.get("b3", 0.0)),
                            float(weights.get("b4", 0.0)),
                            float(weights.get("b5", 0.0)),
                        ]

                        rec.update({
                            "val_acc": val_acc_sgc,
                            "test_acc": float(acc_sgc),
                            "tuning_runtime_sec": 0.0,
                            "method_runtime_sec": method_time,
                            "total_runtime_sec": method_time,
                            "parameter_source": info.get("mode", "feature_first_selective_weighted_correction_v1"),
                            "sgc_threshold": float(info.get("selected_threshold_high", 0.0)),
                            "sgc_b_weights": sgc_b_weights,
                            "sgc_n_uncertain": sgc_n_uncertain,
                            "sgc_n_changed": sgc_n_changed,
                            "sgc_diagnostics_path": info.get("diagnostics_path"),
                        })
                    except Exception as e:
                        rec["success"] = False
                        rec["notes"] = str(e)
                        print(f"    [WARN] selective_graph_correction failed: {e}")
                    all_records.append(rec)
                    runs_file.write(json.dumps(rec) + "\n")
                    runs_file.flush()
                    total_runs_done += 1

                # --------------------------------------------------------
                # 4) Prop + MLP select (derived; only if both available)
                # --------------------------------------------------------
                if (
                    "prop_mlp_select" in methods
                    and mlp_probs is not None
                    and dict_params_prop is not None
                    and acc_val_mlp is not None
                ):
                    rec = _build_record(**rec_kwargs, method="prop_mlp_select")
                    try:
                        _, acc_val_prop_sel = mod.predictclass(
                            data, train_np, val_np,
                            **dict_params_prop, seed=current_seed,
                            mlp_probs=None, log_file=None,
                        )
                        _, acc_test_prop_sel = mod.predictclass(
                            data, train_np, test_np,
                            **dict_params_prop, seed=current_seed,
                            mlp_probs=None, log_file=None,
                        )
                        use_mlp = acc_val_mlp > float(acc_val_prop_sel)
                        final_test = float(acc_test_mlp if use_mlp else acc_test_prop_sel)
                        rec.update({
                            "val_acc": float(max(acc_val_mlp, acc_val_prop_sel)),
                            "test_acc": final_test,
                            "tuning_runtime_sec": tuning_time_prop,
                            "method_runtime_sec": 0.0,
                            "total_runtime_sec": tuning_time_prop,
                            "parameter_source": "val_selection_between_mlp_and_prop",
                            "a_values": params_prop,
                        })
                    except Exception as e:
                        rec["success"] = False
                        rec["notes"] = str(e)
                    all_records.append(rec)
                    runs_file.write(json.dumps(rec) + "\n")
                    runs_file.flush()
                    total_runs_done += 1

    runs_file.close()
    wall_time = time.perf_counter() - t_wall_start

    print(f"\nTotal runs completed: {total_runs_done}  Wall time: {wall_time:.1f}s")
    print(f"Raw records written to: {runs_path}")

    # ------------------------------------------------------------------
    # Generate analysis tables
    # ------------------------------------------------------------------
    _generate_summary(all_records, dataset_meta, summary_path, fairness_path,
                      gain_path, correction_path, regime_path, output_tag)

    return {
        "runs_path": runs_path,
        "summary_path": summary_path,
        "fairness_path": fairness_path,
        "gain_path": gain_path,
        "correction_path": correction_path,
        "regime_path": regime_path,
        "total_runs": total_runs_done,
        "wall_time_sec": wall_time,
    }


# -----------------------------------------------------------------------
# Analysis table generation
# -----------------------------------------------------------------------

def _generate_summary(
    records: List[Dict[str, Any]],
    dataset_meta: Dict[str, Dict[str, Any]],
    summary_path: str,
    fairness_path: str,
    gain_path: str,
    correction_path: str,
    regime_path: str,
    output_tag: str,
):
    """Generate all 5 manuscript-oriented analysis tables."""

    if not records:
        print("[WARN] No records to summarize.")
        return

    # Group records
    by_ds_method: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    by_ds_method_runtime: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    by_ds_method_rec: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))

    for rec in records:
        if not rec.get("success", False):
            continue
        ds = rec["dataset"]
        method = rec["method"]
        acc = rec.get("test_acc")
        if acc is not None:
            by_ds_method[ds][method].append(float(acc))
            by_ds_method_runtime[ds][method].append(float(rec.get("total_runtime_sec", 0.0)))
            by_ds_method_rec[ds][method].append(rec)

    methods_all = ["mlp_only", "prop_only", "selective_graph_correction", "prop_mlp_select"]

    # ------------------------------------------------------------------
    # 1. Core result table (JSON summary)
    # ------------------------------------------------------------------
    summary: Dict[str, Any] = {
        "output_tag": output_tag,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "datasets": {},
    }

    for ds in sorted(by_ds_method.keys()):
        ds_entry = {}
        for method in methods_all:
            accs = by_ds_method[ds].get(method, [])
            if accs:
                ds_entry[method] = {
                    "mean": float(np.mean(accs)),
                    "std": float(np.std(accs)),
                    "n": len(accs),
                    "runs": accs,
                    "mean_runtime_sec": float(np.mean(by_ds_method_runtime[ds].get(method, [0.0]))),
                }
            else:
                ds_entry[method] = None
        summary["datasets"][ds] = ds_entry

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to: {summary_path}")

    # ------------------------------------------------------------------
    # 2. Gain-over-MLP table (CSV)
    # ------------------------------------------------------------------
    gain_rows = []
    gain_header = [
        "dataset", "sgc_mean", "mlp_mean", "delta",
        "n_splits_sgc_wins", "n_splits_mlp_wins", "n_splits_ties",
        "avg_uncertain_fraction", "avg_changed_fraction",
    ]

    for ds in sorted(by_ds_method.keys()):
        mlp_accs = by_ds_method[ds].get("mlp_only", [])
        sgc_accs = by_ds_method[ds].get("selective_graph_correction", [])
        sgc_recs = by_ds_method_rec[ds].get("selective_graph_correction", [])

        if not mlp_accs or not sgc_accs:
            continue

        sgc_mean = float(np.mean(sgc_accs))
        mlp_mean = float(np.mean(mlp_accs))
        delta = sgc_mean - mlp_mean

        # Per-split win/loss/tie (align by split_id × repeat_id)
        mlp_map = {(r["split_id"], r["repeat_id"]): r.get("test_acc") for r in by_ds_method_rec[ds].get("mlp_only", [])}
        sgc_map = {(r["split_id"], r["repeat_id"]): r.get("test_acc") for r in sgc_recs}
        common_keys = set(mlp_map) & set(sgc_map)
        wins = sum(1 for k in common_keys if sgc_map[k] > mlp_map[k] + 1e-6)
        losses = sum(1 for k in common_keys if sgc_map[k] < mlp_map[k] - 1e-6)
        ties = len(common_keys) - wins - losses

        # Uncertain / changed fractions
        uncertain_fracs = []
        changed_fracs = []
        for r in sgc_recs:
            n_t = r.get("test_count") or r.get("num_nodes", 1)
            n_u = r.get("sgc_n_uncertain")
            n_c = r.get("sgc_n_changed")
            if n_u is not None and n_t > 0:
                uncertain_fracs.append(n_u / n_t)
            if n_c is not None and n_u and n_u > 0:
                changed_fracs.append(n_c / n_u)

        gain_rows.append({
            "dataset": ds,
            "sgc_mean": f"{sgc_mean:.4f}",
            "mlp_mean": f"{mlp_mean:.4f}",
            "delta": f"{delta:+.4f}",
            "n_splits_sgc_wins": wins,
            "n_splits_mlp_wins": losses,
            "n_splits_ties": ties,
            "avg_uncertain_fraction": f"{np.mean(uncertain_fracs):.3f}" if uncertain_fracs else "N/A",
            "avg_changed_fraction": f"{np.mean(changed_fracs):.3f}" if changed_fracs else "N/A",
        })

    _write_csv(gain_path, gain_header, gain_rows)
    print(f"Gain-over-MLP table written to: {gain_path}")

    # ------------------------------------------------------------------
    # 3. Correction-behavior table (CSV)
    # ------------------------------------------------------------------
    cb_header = [
        "dataset", "n_runs",
        "avg_threshold_high",
        "avg_b1", "avg_b2", "avg_b3", "avg_b4",
        "use_feature_knn",
        "avg_uncertain_frac",
        "avg_changed_of_uncertain_frac",
        "zero_uncertain_runs",
        "zero_changed_runs",
        "acc_on_confident_mean",
        "acc_on_uncertain_mean",
    ]
    cb_rows = []

    for ds in sorted(by_ds_method_rec.keys()):
        sgc_recs = by_ds_method_rec[ds].get("selective_graph_correction", [])
        if not sgc_recs:
            continue

        thresholds = [r["sgc_threshold"] for r in sgc_recs if r.get("sgc_threshold") is not None]
        b_weights = [r["sgc_b_weights"] for r in sgc_recs if r.get("sgc_b_weights") is not None]
        uncertain_fracs = []
        changed_fracs = []
        zero_unc = 0
        zero_chg = 0
        acc_conf_list = []
        acc_unc_list = []

        for r in sgc_recs:
            n_t = r.get("test_count", 1) or 1
            n_u = r.get("sgc_n_uncertain") or 0
            n_c = r.get("sgc_n_changed") or 0
            uncertain_fracs.append(n_u / n_t)
            changed_fracs.append(n_c / max(n_u, 1))
            if n_u == 0:
                zero_unc += 1
            if n_c == 0:
                zero_chg += 1

        cb_rows.append({
            "dataset": ds,
            "n_runs": len(sgc_recs),
            "avg_threshold_high": f"{np.mean(thresholds):.3f}" if thresholds else "N/A",
            "avg_b1": f"{np.mean([w[0] for w in b_weights]):.3f}" if b_weights else "N/A",
            "avg_b2": f"{np.mean([w[1] for w in b_weights]):.3f}" if b_weights else "N/A",
            "avg_b3": f"{np.mean([w[2] for w in b_weights]):.3f}" if b_weights else "N/A",
            "avg_b4": f"{np.mean([w[3] for w in b_weights]):.3f}" if b_weights else "N/A",
            "use_feature_knn": "no",
            "avg_uncertain_frac": f"{np.mean(uncertain_fracs):.3f}" if uncertain_fracs else "N/A",
            "avg_changed_of_uncertain_frac": f"{np.mean(changed_fracs):.3f}" if changed_fracs else "N/A",
            "zero_uncertain_runs": zero_unc,
            "zero_changed_runs": zero_chg,
            "acc_on_confident_mean": "see_diagnostics",
            "acc_on_uncertain_mean": "see_diagnostics",
        })

    _write_csv(correction_path, cb_header, cb_rows)
    print(f"Correction-behavior table written to: {correction_path}")

    # ------------------------------------------------------------------
    # 4. Dataset-regime table (CSV)
    # ------------------------------------------------------------------
    regime_header = [
        "dataset", "num_nodes", "num_edges", "num_classes",
        "avg_degree", "homophily",
        "mlp_mean", "mlp_std",
        "prop_mean", "prop_std",
        "sgc_mean", "sgc_std",
        "sgc_delta_vs_mlp",
        "regime_note",
    ]
    regime_rows = []

    for ds in sorted(by_ds_method.keys()):
        meta = dataset_meta.get(ds, {})
        mlp_accs = by_ds_method[ds].get("mlp_only", [])
        prop_accs = by_ds_method[ds].get("prop_only", [])
        sgc_accs = by_ds_method[ds].get("selective_graph_correction", [])

        mlp_m = float(np.mean(mlp_accs)) if mlp_accs else float("nan")
        mlp_s = float(np.std(mlp_accs)) if mlp_accs else float("nan")
        prop_m = float(np.mean(prop_accs)) if prop_accs else float("nan")
        prop_s = float(np.std(prop_accs)) if prop_accs else float("nan")
        sgc_m = float(np.mean(sgc_accs)) if sgc_accs else float("nan")
        sgc_s = float(np.std(sgc_accs)) if sgc_accs else float("nan")
        delta = sgc_m - mlp_m if not (np.isnan(sgc_m) or np.isnan(mlp_m)) else float("nan")

        h = meta.get("homophily", float("nan"))
        if not np.isnan(h):
            note = "homophilic" if h >= 0.5 else "heterophilic"
        else:
            note = "unknown"

        regime_rows.append({
            "dataset": ds,
            "num_nodes": meta.get("num_nodes", "N/A"),
            "num_edges": meta.get("num_edges", "N/A"),
            "num_classes": meta.get("num_classes", "N/A"),
            "avg_degree": f"{meta.get('avg_degree', float('nan')):.2f}",
            "homophily": f"{h:.3f}" if not np.isnan(h) else "N/A",
            "mlp_mean": f"{mlp_m:.4f}" if not np.isnan(mlp_m) else "N/A",
            "mlp_std": f"{mlp_s:.4f}" if not np.isnan(mlp_s) else "N/A",
            "prop_mean": f"{prop_m:.4f}" if not np.isnan(prop_m) else "N/A",
            "prop_std": f"{prop_s:.4f}" if not np.isnan(prop_s) else "N/A",
            "sgc_mean": f"{sgc_m:.4f}" if not np.isnan(sgc_m) else "N/A",
            "sgc_std": f"{sgc_s:.4f}" if not np.isnan(sgc_s) else "N/A",
            "sgc_delta_vs_mlp": f"{delta:+.4f}" if not np.isnan(delta) else "N/A",
            "regime_note": note,
        })

    _write_csv(regime_path, regime_header, regime_rows)
    print(f"Regime-analysis table written to: {regime_path}")

    # ------------------------------------------------------------------
    # 5. Fairness report (Markdown)
    # ------------------------------------------------------------------
    lines = [
        f"# Manuscript Fairness Report — {output_tag}\n",
        f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%S')}\n\n",
        "## Core Result Table\n",
        "| Dataset | mlp_only | prop_only | selective_graph_correction | prop_mlp_select |\n",
        "|---------|----------|-----------|---------------------------|-----------------|\n",
    ]
    for ds in sorted(by_ds_method.keys()):
        row = [ds]
        for method in methods_all:
            accs = by_ds_method[ds].get(method, [])
            if accs:
                row.append(f"{np.mean(accs):.4f} ± {np.std(accs):.4f}")
            else:
                row.append("N/A")
        lines.append("| " + " | ".join(row) + " |\n")

    lines.append("\n## Gain Over MLP (selective_graph_correction - mlp_only)\n")
    lines.append("| Dataset | Delta | SGC wins | MLP wins | Ties |\n")
    lines.append("|---------|-------|----------|----------|------|\n")
    for r in gain_rows:
        lines.append(
            f"| {r['dataset']} | {r['delta']} | {r['n_splits_sgc_wins']} | {r['n_splits_mlp_wins']} | {r['n_splits_ties']} |\n"
        )

    lines.append("\n## Notes on Fairness\n")
    lines.append(
        "- All methods use **canonical GEO-GCN splits** (no random fallback).\n"
        "- MLP is trained with early stopping on val set for all methods.\n"
        "- SGC threshold and b-weights are tuned on val set; test set is never seen during tuning.\n"
        "- prop_only params are tuned via `robust_random_search` on train k-fold splits (no val leakage).\n"
        "- prop_mlp_select uses val accuracy to choose between MLP and prop_only (no test leakage).\n"
    )

    with open(fairness_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Fairness report written to: {fairness_path}")

    # ------------------------------------------------------------------
    # Print core result table to stdout
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"CORE RESULT TABLE ({output_tag})")
    print("=" * 70)
    col_w = 28
    print(f"{'Dataset':<14} {'mlp_only':>{col_w}} {'prop_only':>{col_w}} {'sgc':>{col_w}}")
    print("-" * 80)
    for ds in sorted(by_ds_method.keys()):
        row_parts = [f"{ds:<14}"]
        for method in ["mlp_only", "prop_only", "selective_graph_correction"]:
            accs = by_ds_method[ds].get(method, [])
            if accs:
                row_parts.append(f"{np.mean(accs):.4f} ± {np.std(accs):.4f}".rjust(col_w))
            else:
                row_parts.append("N/A".rjust(col_w))
        print("".join(row_parts))
    print("=" * 70)


def _write_csv(path: str, header: List[str], rows: List[Dict[str, Any]]):
    """Write a simple CSV file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(h, "")) for h in header) + "\n")


# -----------------------------------------------------------------------
# Case study helper (called separately or in post-processing)
# -----------------------------------------------------------------------

def generate_case_studies(
    runs_jsonl_path: str,
    output_path: str,
    target_datasets: Optional[List[str]] = None,
):
    """
    Read per-node diagnostics from SGC runs and produce a markdown case-study file.

    Expects diagnostics JSON files in logs/ (written by selective_graph_correction_predictclass).
    Falls back to run-level summaries if node-level diagnostics are unavailable.
    """
    if not os.path.isfile(runs_jsonl_path):
        print(f"[WARN] Runs file not found: {runs_jsonl_path}")
        return

    records = []
    with open(runs_jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    sgc_records = [r for r in records if r.get("method") == "selective_graph_correction"]
    if target_datasets:
        sgc_records = [r for r in sgc_records if r["dataset"] in target_datasets]

    lines = [
        "# Case Studies — Selective Graph Correction\n\n",
        "For each dataset, we show corrected test nodes from split 0.\n\n",
    ]

    ds_covered = set()
    for rec in sgc_records:
        ds = rec["dataset"]
        if ds in ds_covered:
            continue
        # Load diagnostics file produced by selective_graph_correction_predictclass.
        # Prefer the explicit path stored in the run record.
        diag_path = rec.get("sgc_diagnostics_path")
        if not diag_path or not os.path.isfile(diag_path):
            # Fallback: attempt a best-effort guess based on naming conventions.
            run_id = f"{ds}_s{rec['split_id']}_r{rec['repeat_id']}"
            candidates = [
                os.path.join("logs", f"{ds}_selective_graph_correction_{run_id}.json"),
                os.path.join("logs", f"sgc_{run_id}_diagnostics.json"),
            ]
            diag_path = next((c for c in candidates if os.path.isfile(c)), None)
            if not diag_path:
                continue
        with open(diag_path, encoding="utf-8") as df:
            diag = json.load(df)
        node_diags = diag.get("test_node_diagnostics") or diag.get("node_diagnostics", []) or []
        if not node_diags:
            continue

        changed = [nd for nd in node_diags if nd.get("changed_from_mlp", False)]

        lines.append(f"## {ds}\n\n")
        lines.append(
            f"- threshold_high={float(diag.get('selected_threshold_high', 0.0)):.2f}  "
            f"b=({float((diag.get('selected_weights', {}) or {}).get('b1', 0.0)):.2f},"
            f"{float((diag.get('selected_weights', {}) or {}).get('b2', 0.0)):.2f},"
            f"{float((diag.get('selected_weights', {}) or {}).get('b3', 0.0)):.2f},"
            f"{float((diag.get('selected_weights', {}) or {}).get('b4', 0.0)):.2f},"
            f"{float((diag.get('selected_weights', {}) or {}).get('b5', 0.0)):.2f})\n"
        )
        n_test = len(node_diags)
        n_uncertain = sum(1 for nd in node_diags if nd.get("was_uncertain", False))
        n_changed = sum(1 for nd in node_diags if nd.get("changed_from_mlp", False))
        lines.append(f"- {n_uncertain}/{n_test} test nodes uncertain | {n_changed} corrected\n\n")

        if changed:
            lines.append("### Sample corrected nodes (first 10)\n\n")
            lines.append(
                "| node_id | mlp_pred | final_pred | mlp_margin | was_uncertain | dom_component |\n"
            )
            lines.append("|---------|----------|-------------|--------------|----------------|---------------|\n")
            for nd in changed[:10]:
                comps = nd.get("winning_class_components", {}) or {}
                dom = max(comps, key=lambda k: abs(comps[k])) if comps else "?"
                lines.append(
                    f"| {nd.get('node_id')} | {nd.get('mlp_pred')} | {nd.get('final_pred')} | "
                    f"{float(nd.get('mlp_margin', 0.0)):.3f} | {bool(nd.get('was_uncertain', False))} | {dom} |\n"
                )
            lines.append("\n")
        else:
            lines.append("_No changed nodes in this run._\n\n")

        ds_covered.add(ds)

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Case studies written to: {output_path}")


# -----------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Manuscript phase experiment: mlp_only vs prop_only vs selective_graph_correction"
    )
    parser.add_argument(
        "--split-dir", default=None,
        help="Directory containing GEO-GCN split .npz files."
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Datasets to run. Default: all priority-ordered candidates."
    )
    parser.add_argument(
        "--splits", nargs="+", type=int, default=list(range(10)),
        help="Split IDs to use (default: 0..9)."
    )
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="Repeats per split (default: 1). Use 3 on Wulver for final manuscript run."
    )
    parser.add_argument(
        "--methods", nargs="+",
        default=["mlp_only", "prop_only", "selective_graph_correction", "prop_mlp_select"],
        help="Methods to run."
    )
    parser.add_argument(
        "--output-tag", default="manuscript_phase",
        help="Tag for output file names."
    )
    parser.add_argument(
        "--rs-num-samples", type=int, default=30,
        help="Number of random search samples for prop_only tuning."
    )
    parser.add_argument(
        "--sgc-tune-samples", type=int, default=25,
        help="Number of weight-search samples for SGC tuning."
    )
    parser.add_argument(
        "--case-studies-only", action="store_true",
        help="Skip experiment; only generate case-study markdown from existing logs."
    )
    args = parser.parse_args()

    if args.case_studies_only:
        cs_path = os.path.join("logs", f"case_studies_{args.output_tag}.md")
        generate_case_studies(
            os.path.join("logs", f"comparison_runs_{args.output_tag}.jsonl"),
            cs_path,
            target_datasets=args.datasets,
        )
        return

    artifacts = run_manuscript_experiment(
        datasets=args.datasets,
        split_ids=args.splits,
        repeats=args.repeats,
        methods=args.methods,
        split_dir=args.split_dir,
        output_tag=args.output_tag,
        rs_num_samples=args.rs_num_samples,
        sgc_tune_n_samples=args.sgc_tune_samples,
    )

    # Generate case studies
    cs_path = os.path.join("logs", f"case_studies_{args.output_tag}.md")
    generate_case_studies(artifacts["runs_path"], cs_path)

    print("\n=== Output Artifacts ===")
    for key, val in artifacts.items():
        if isinstance(val, str):
            print(f"  {key}: {val}")
    print(f"  case_studies: {cs_path}")

    print("\n=== Manuscript Readiness Assessment ===")
    _print_manuscript_readiness(artifacts)


def _print_manuscript_readiness(artifacts: Dict[str, Any]):
    summary_path = artifacts.get("summary_path")
    if not summary_path or not os.path.isfile(summary_path):
        print("No summary available.")
        return

    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    datasets = summary.get("datasets", {})
    gains = []
    for ds, ds_data in datasets.items():
        mlp = ds_data.get("mlp_only")
        sgc = ds_data.get("selective_graph_correction")
        if mlp and sgc:
            delta = sgc["mean"] - mlp["mean"]
            gains.append((ds, delta, sgc["mean"], mlp["mean"]))

    if gains:
        positive = [(ds, d, s, m) for ds, d, s, m in gains if d > 0.005]
        neutral = [(ds, d, s, m) for ds, d, s, m in gains if -0.005 <= d <= 0.005]
        negative = [(ds, d, s, m) for ds, d, s, m in gains if d < -0.005]

        print(f"\nSGC vs MLP:")
        print(f"  Positive (delta > +0.5%): {[ds for ds,*_ in positive]}")
        print(f"  Neutral  (|delta| <= 0.5%): {[ds for ds,*_ in neutral]}")
        print(f"  Negative (delta < -0.5%): {[ds for ds,*_ in negative]}")

        if len(positive) > len(negative) + len(neutral):
            verdict = "positive-method paper"
        elif len(positive) > 0 and len(negative) > 0:
            verdict = "regime-analysis paper"
        else:
            verdict = "diagnostic/negative paper"
        print(f"\n  Evidence most consistent with: **{verdict}**")
    else:
        print("  No SGC vs MLP comparison data yet.")

    print(
        "\nNote: Run with repeats=3 on Wulver for manuscript-quality stability estimates.\n"
        "Note: Case studies require per-node diagnostics JSON files in logs/."
    )


if __name__ == "__main__":
    main()

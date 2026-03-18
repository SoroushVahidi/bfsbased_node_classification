import os
import time
import json
import importlib.util
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch


def _load_full_investigate_module():
    """
    Dynamically load bfsbased-full-investigate-homophil.py as a module.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "bfsbased-full-investigate-homophil.py")
    spec = importlib.util.spec_from_file_location("bfs_full_investigate", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    # Provide a default DATASET_KEY to satisfy top-level code;
    # we will call load_dataset(...) explicitly per dataset later.
    setattr(mod, "DATASET_KEY", "texas")
    spec.loader.exec_module(mod)  # type: ignore[arg-type]
    return mod


def _get_repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _get_dj_repo_root() -> Optional[str]:
    """
    Resolve Diffusion-Jump-GNNs repo root if present.
    """
    env = os.environ.get("DIFFUSION_JUMP_REPO")
    if env:
        return os.path.abspath(env)
    # fallback to ../Diffusion-Jump-GNNs
    here = _get_repo_root()
    candidate = os.path.abspath(os.path.join(here, "..", "Diffusion-Jump-GNNs"))
    return candidate if os.path.isdir(candidate) else None


def resolve_split_file(
    dataset_key: str,
    split_id: int,
    *,
    split_dir: Optional[str] = None,
) -> Optional[str]:
    """
    Resolve the path to a GEO-GCN split .npz file with the following priority:
      a) explicit split_dir (CLI arg)
      b) GEO_GCN_SPLIT_DIR env var
      c) DIFFUSION_JUMP_REPO/splits
      d) our repo root
      e) any subdirectory under our shared data root
    """
    fname = f"{dataset_key.lower()}_split_0.6_0.2_{split_id}.npz"

    # (a) explicit split_dir
    if split_dir:
        cand = os.path.join(split_dir, fname)
        if os.path.isfile(cand):
            return cand

    # (b) GEO_GCN_SPLIT_DIR env var
    env_dir = os.environ.get("GEO_GCN_SPLIT_DIR")
    if env_dir:
        cand = os.path.join(env_dir, fname)
        if os.path.isfile(cand):
            return cand

    # (c) DIFFUSION_JUMP_REPO/splits
    dj_root = _get_dj_repo_root()
    if dj_root:
        cand = os.path.join(dj_root, "splits", fname)
        if os.path.isfile(cand):
            return cand

    # (d) our repo root
    repo_root = _get_repo_root()
    cand_root = os.path.join(repo_root, fname)
    if os.path.isfile(cand_root):
        return cand_root

    # (e) search under shared data directory
    data_root = os.path.join(repo_root, "data")
    if os.path.isdir(data_root):
        for root, _dirs, files in os.walk(data_root):
            if fname in files:
                return os.path.join(root, fname)

    return None


def validate_required_splits(
    datasets: List[str],
    split_ids: List[int],
    *,
    split_dir: Optional[str] = None,
) -> Dict[Tuple[str, int], Optional[str]]:
    """
    Resolve and validate split files for the requested datasets/split_ids.
    Writes a split check report and returns a mapping (dataset, split_id) -> path or None.
    """
    mapping: Dict[Tuple[str, int], Optional[str]] = {}
    for ds in datasets:
        for sid in split_ids:
            p = resolve_split_file(ds, sid, split_dir=split_dir)
            mapping[(ds, sid)] = p

    os.makedirs("logs", exist_ok=True)
    report_json = os.path.join("logs", "comparison_split_check_stage2.json")
    report_md = os.path.join("logs", "comparison_split_check_stage2.md")

    # JSON report
    js_payload = []
    for (ds, sid), path in mapping.items():
        js_payload.append(
            {
                "dataset": ds,
                "split_id": sid,
                "found": path is not None,
                "path": path,
            }
        )
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(js_payload, f, indent=2)

    # Markdown report
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# Split Check – Stage 2\n\n")
        f.write("| Dataset | Split ID | Found | Path |\n")
        f.write("|---------|----------|-------|------|\n")
        for (ds, sid), path in mapping.items():
            f.write(f"| {ds} | {sid} | {'yes' if path else 'no'} | {path or ''} |\n")

    return mapping


def _load_dataset_and_data(mod, dataset_key: str):
    """
    Load dataset and apply the same preprocessing as in the main script.
    """
    root = "data/"
    dataset = mod.load_dataset(dataset_key, root)
    data = dataset[0]
    # Undirected + unique edges as in main script
    data.edge_index = torch.cat([data.edge_index, data.edge_index.flip(0)], dim=1)
    data.edge_index = torch.unique(data.edge_index, dim=1)
    return dataset, data


def _load_split_npz(dataset_key: str, split_id: int, device: torch.device, split_dir: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    """
    Load GEO-GCN split .npz using the resolution helper.
    """
    split_path = resolve_split_file(dataset_key, split_id, split_dir=split_dir)
    if split_path is None:
        raise FileNotFoundError(f"Split file not found for dataset={dataset_key}, split_id={split_id}")
    split = np.load(split_path)
    train_idx = torch.as_tensor(np.where(split["train_mask"])[0], dtype=torch.long, device=device)
    val_idx = torch.as_tensor(np.where(split["val_mask"])[0], dtype=torch.long, device=device)
    test_idx = torch.as_tensor(np.where(split["test_mask"])[0], dtype=torch.long, device=device)
    return train_idx, val_idx, test_idx, split_path


def _record_common_info(mod, dataset_key: str, dataset, data, split_path: str, split_id: int, repeat_id: int, method: str, seed: int) -> Dict[str, Any]:
    """
    Build the common metadata part of a per-run record.
    """
    N = int(data.num_nodes)
    E = int(data.edge_index.size(1))
    num_classes = int(data.y.max().item()) + 1
    train_mask = np.load(split_path)["train_mask"]
    val_mask = np.load(split_path)["val_mask"]
    test_mask = np.load(split_path)["test_mask"]

    record: Dict[str, Any] = {
        "method": method,
        "dataset": dataset_key,
        "split_id": split_id,
        "repeat_id": repeat_id,
        "seed": seed,
        "success": True,
        "notes": "",
        "dataset_root": "data/",
        "split_file": split_path,
        "num_nodes": N,
        "num_edges": E,
        "num_classes": num_classes,
        "train_count": int(train_mask.sum()),
        "val_count": int(val_mask.sum()),
        "test_count": int(test_mask.sum()),
        "graph_symmetrized": True,
        "self_loops_added": False,
        "features_normalized_for_propagation": True,
        "graph_treated_as_undirected": True,
        # runtime fields to be filled later
        "tuning_runtime_sec": 0.0,
        "method_runtime_sec": 0.0,
        "total_runtime_sec": 0.0,
        # parameter provenance
        "parameter_source": None,
        "parameter_tuning_target_method": None,
        "tuned_separately_for_this_method": None,
        "is_distinct_method_variant": True,
        "parent_method": None,
    }
    return record


def run_stage2_internal(split_dir: Optional[str] = None):
    """
    Stage 2: internal comparison on 2 datasets (texas, chameleon),
    splits 0..9, 1 repeat, with selected methods.
    """
    mod = _load_full_investigate_module()

    datasets = ["texas", "chameleon"]
    split_ids = list(range(10))
    repeats = 1

    methods_to_run = [
        "mlp_only",
        "prop_only",
        "mlp_refined",
        "prop_mlp_select",
        # Optional new method (left off by default for backward-compatible runs):
        # "gated_mlp_prop",
        "priority_bfs",
        "ensemble_multi_source_bfs",
    ]

    # Preflight: validate splits
    split_map = validate_required_splits(datasets, split_ids, split_dir=split_dir)
    missing = [(ds, sid) for (ds, sid), p in split_map.items() if p is None]
    if missing:
        print("Split validation failed. Missing the following splits:")
        for ds, sid in missing:
            print(f"  - dataset={ds}, split_id={sid}")
        print("See logs/comparison_split_check_stage2.{json,md} for details.")
        return

    records: List[Dict[str, Any]] = []

    for dataset_key in datasets:
        dataset, data = _load_dataset_and_data(mod, dataset_key)
        device = data.x.device
        N_total = int(data.num_nodes)

        for split_id in split_ids:
            for repeat in range(repeats):
                run_seed = 1337 + split_id
                current_seed = run_seed * 100 + repeat
                torch.manual_seed(current_seed)
                np.random.seed(current_seed)

                split_path = split_map[(dataset_key, split_id)]
                train_idx, val_idx, test_idx, _ = _load_split_npz(dataset_key, split_id, device=device, split_dir=split_dir)

                train_np = mod._to_numpy_idx(train_idx)
                val_np = mod._to_numpy_idx(val_idx)
                test_np = mod._to_numpy_idx(test_idx)

                # 1) MLP only
                if "mlp_only" in methods_to_run:
                    rec = _record_common_info(mod, dataset_key, dataset, data, split_path, split_id, repeat, "mlp_only", current_seed)
                    t0 = time.perf_counter()
                    mlp_probs, _ = mod.train_mlp_and_predict(
                        data,
                        train_idx,
                        hidden=64,
                        layers=2,
                        dropout=0.5,
                        lr=0.01,
                        epochs=200,
                        log_file=None,
                    )
                    tuning_time = time.perf_counter() - t0
                    max_probs, preds_mlp_all = mlp_probs.max(dim=1)
                    t1 = time.perf_counter()
                    acc_val_mlp = (preds_mlp_all[val_idx] == data.y[val_idx]).float().mean().item()
                    acc_test_mlp = (preds_mlp_all[test_idx] == data.y[test_idx]).float().mean().item()
                    method_time = time.perf_counter() - t1

                    rec.update(
                        {
                            "val_acc": float(acc_val_mlp),
                            "test_acc": float(acc_test_mlp),
                            "tuning_runtime_sec": tuning_time,
                            "method_runtime_sec": method_time,
                            "total_runtime_sec": tuning_time + method_time,
                            "parameter_source": "mlp_only_self_tuned",
                            "parameter_tuning_target_method": "mlp_only",
                            "tuned_separately_for_this_method": True,
                            "is_distinct_method_variant": True,
                            "parent_method": None,
                            # MLP hyperparams
                            "mlp_hidden": 64,
                            "mlp_layers": 2,
                            "mlp_dropout": 0.5,
                            "mlp_lr": 0.01,
                            "mlp_weight_decay": 5e-4,
                            "mlp_epochs": 200,
                        }
                    )
                    records.append(rec)
                else:
                    mlp_probs = None
                    max_probs = None

                # 2) Prop only (predictclass) + tuned params
                # Tune once per split/repeat for predictclass; reused elsewhere.
                t0 = time.perf_counter()
                params_prop, _, _ = mod.robust_random_search(
                    mod.predictclass,
                    data,
                    train_np,
                    num_samples=mod.RS_NUM_SAMPLES,
                    n_splits=mod.RS_SPLITS,
                    repeats=mod.RS_REPEATS,
                    seed=current_seed,
                    log_file=None,
                    mlp_probs=None,
                )
                tuning_time_prop = time.perf_counter() - t0
                dict_params_prop = {f"a{i+1}": p for i, p in enumerate(params_prop)}

                # 2a) Prop only
                if "prop_only" in methods_to_run:
                    rec = _record_common_info(mod, dataset_key, dataset, data, split_path, split_id, repeat, "prop_only", current_seed)
                    t1 = time.perf_counter()
                    _, acc_val_prop = mod.predictclass(
                        data,
                        train_np,
                        val_np,
                        **dict_params_prop,
                        seed=current_seed,
                        mlp_probs=None,
                        log_file=None,
                    )
                    _, acc_test_prop = mod.predictclass(
                        data,
                        train_np,
                        test_np,
                        **dict_params_prop,
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
                            "parameter_source": "tuned_on_predictclass_then_reused",
                            "parameter_tuning_target_method": "predictclass",
                            "tuned_separately_for_this_method": True,
                            "is_distinct_method_variant": True,
                            "parent_method": None,
                            "a_values": params_prop,
                        }
                    )
                    records.append(rec)

                # 3) MLP + refined (full val + test evaluation)
                if "mlp_refined" in methods_to_run and mlp_probs is not None:
                    rec = _record_common_info(mod, dataset_key, dataset, data, split_path, split_id, repeat, "mlp_refined", current_seed)
                    # Tuning for hybrid
                    t0 = time.perf_counter()
                    params_hyb, _, _ = mod.robust_random_search(
                        mod.predictclass,
                        data,
                        train_np,
                        num_samples=mod.RS_NUM_SAMPLES,
                        n_splits=mod.RS_SPLITS,
                        repeats=mod.RS_REPEATS,
                        seed=current_seed,
                        log_file=None,
                        mlp_probs=mlp_probs,
                    )
                    tuning_time_hyb = time.perf_counter() - t0
                    dict_params_hyb = {f"a{i+1}": p for i, p in enumerate(params_hyb)}

                    # Build pseudo-labels as in main script (val step)
                    CONFIDENCE_THRESHOLD = 0.90
                    max_probs_local, preds_mlp_all_local = mlp_probs.max(dim=1)
                    is_unlabeled = torch.ones(N_total, dtype=torch.bool, device=device)
                    is_unlabeled[train_idx] = False
                    is_high_conf = (max_probs_local >= CONFIDENCE_THRESHOLD) & is_unlabeled

                    mask_val_support = is_high_conf.clone()
                    mask_val_support[val_idx] = False
                    pseudo_idx_val = torch.where(mask_val_support)[0].cpu().numpy()

                    aug_train_val = np.concatenate([train_np, pseudo_idx_val])
                    orig_y = data.y.clone()
                    data.y[pseudo_idx_val] = preds_mlp_all_local[pseudo_idx_val]

                    t1 = time.perf_counter()
                    try:
                        _, acc_val_refined = mod.predictclass(
                            data,
                            aug_train_val,
                            val_np,
                            **dict_params_hyb,
                            seed=current_seed,
                            mlp_probs=mlp_probs,
                            log_file=None,
                        )
                    finally:
                        data.y = orig_y.clone()
                    method_time = time.perf_counter() - t1

                    # Test step: pseudo-labels excluding test nodes
                    mask_test_support = is_high_conf.clone()
                    mask_test_support[test_idx] = False
                    pseudo_idx_test = torch.where(mask_test_support)[0].cpu().numpy()

                    aug_train_test = np.concatenate([train_np, pseudo_idx_test])
                    data.y[pseudo_idx_test] = preds_mlp_all_local[pseudo_idx_test]
                    t2 = time.perf_counter()
                    try:
                        _, acc_test_refined = mod.predictclass(
                            data,
                            aug_train_test,
                            test_np,
                            **dict_params_hyb,
                            seed=current_seed,
                            mlp_probs=mlp_probs,
                            log_file=None,
                        )
                    finally:
                        data.y = orig_y.clone()
                    test_time = time.perf_counter() - t2

                    total_method_time = method_time + test_time
                    # Match main script behavior:
                    # choose refined only when its validation performance beats MLP.
                    use_refined = float(acc_val_refined) > float(acc_val_mlp)
                    final_val_acc = float(acc_val_refined if use_refined else acc_val_mlp)
                    final_test_acc = float(acc_test_refined if use_refined else acc_test_mlp)
                    rec.update(
                        {
                            "val_acc": final_val_acc,
                            "test_acc": final_test_acc,
                            "val_acc_refined_raw": float(acc_val_refined),
                            "test_acc_refined_raw": float(acc_test_refined),
                            "val_acc_mlp_raw": float(acc_val_mlp),
                            "test_acc_mlp_raw": float(acc_test_mlp),
                            "used_refined_variant": bool(use_refined),
                            "tuning_runtime_sec": tuning_time_hyb,
                            "method_runtime_sec": total_method_time,
                            "total_runtime_sec": tuning_time_hyb + total_method_time,
                            "parameter_source": "tuned_on_predictclass_then_reused",
                            "parameter_tuning_target_method": "predictclass_with_mlp",
                            "tuned_separately_for_this_method": True,
                            "is_distinct_method_variant": True,
                            "parent_method": "mlp_only",
                            "a_values": params_hyb,
                        }
                    )
                    records.append(rec)

                # 4) Prop + MLP (select) – derived from existing val/test numbers
                if "prop_mlp_select" in methods_to_run and mlp_probs is not None:
                    # We recompute quick val/test accuracies for MLP and prop-only:
                    _, acc_val_prop = mod.predictclass(
                        data,
                        train_np,
                        val_np,
                        **dict_params_prop,
                        seed=current_seed,
                        mlp_probs=None,
                        log_file=None,
                    )
                    max_probs_tmp, preds_mlp_all_tmp = mlp_probs.max(dim=1)
                    acc_val_mlp = (preds_mlp_all_tmp[val_idx] == data.y[val_idx]).float().mean().item()
                    acc_test_mlp = (preds_mlp_all_tmp[test_idx] == data.y[test_idx]).float().mean().item()
                    _, acc_test_prop = mod.predictclass(
                        data,
                        train_np,
                        test_np,
                        **dict_params_prop,
                        seed=current_seed,
                        mlp_probs=None,
                        log_file=None,
                    )

                    use_mlp = acc_val_mlp > float(acc_val_prop)
                    final_test = float(acc_test_mlp if use_mlp else acc_test_prop)

                    rec = _record_common_info(mod, dataset_key, dataset, data, split_path, split_id, repeat, "prop_mlp_select", current_seed)
                    rec.update(
                        {
                            "val_acc": float(max(acc_val_mlp, acc_val_prop)),
                            "test_acc": final_test,
                            "tuning_runtime_sec": tuning_time_prop,  # reusing prop tuning cost
                            "method_runtime_sec": 0.0,
                            "total_runtime_sec": tuning_time_prop,
                            "parameter_source": "tuned_on_predictclass_then_reused",
                            "parameter_tuning_target_method": "predictclass",
                            "tuned_separately_for_this_method": False,
                            "is_distinct_method_variant": True,
                            "parent_method": "prop_only",
                            "a_values": params_prop,
                        }
                    )
                    records.append(rec)

                # 5) Learned node-wise gate between MLP and propagation
                if "gated_mlp_prop" in methods_to_run and mlp_probs is not None and hasattr(mod, "gated_mlp_prop_predictclass"):
                    rec = _record_common_info(mod, dataset_key, dataset, data, split_path, split_id, repeat, "gated_mlp_prop", current_seed)
                    t1 = time.perf_counter()
                    _, acc_test_gated, gate_info = mod.gated_mlp_prop_predictclass(
                        data,
                        train_np,
                        val_np,
                        test_np,
                        prop_params=dict_params_prop,
                        mlp_probs=mlp_probs,
                        seed=current_seed,
                        log_prefix=f"{dataset_key}:STAGE2-GATED-MLP-PROP",
                        log_file=None,
                        diagnostics_run_id=f"split{split_id}_rep{repeat}",
                        write_node_diagnostics=True,
                        dataset_key_for_logs=dataset_key,
                    )
                    method_time = time.perf_counter() - t1
                    rec.update(
                        {
                            "val_acc": float(gate_info.get("val_acc_gated")) if gate_info.get("val_acc_gated") is not None else None,
                            "test_acc": float(acc_test_gated),
                            "tuning_runtime_sec": tuning_time_prop,  # gate reuses propagation tuning
                            "method_runtime_sec": method_time,
                            "total_runtime_sec": tuning_time_prop + method_time,
                            "parameter_source": "gate_trained_on_validation_nodes",
                            "parameter_tuning_target_method": "gated_mlp_prop",
                            "tuned_separately_for_this_method": True,
                            "is_distinct_method_variant": True,
                            "parent_method": "prop_mlp_select",
                            "a_values": params_prop,
                            "gate_mode": gate_info.get("gate_mode"),
                            "gate_train_size": gate_info.get("gate_train_size"),
                            "gate_accuracy_on_val_supervised_nodes": gate_info.get("gate_accuracy_on_val_supervised_nodes"),
                            "fraction_test_nodes_assigned_to_mlp": gate_info.get("fraction_test_nodes_assigned_to_mlp"),
                            "fraction_test_nodes_assigned_to_propagation": gate_info.get("fraction_test_nodes_assigned_to_propagation"),
                            "gate_feature_names": gate_info.get("gate_feature_names"),
                            "gate_diagnostics_path": gate_info.get("diagnostics_path"),
                        }
                    )
                    records.append(rec)

                # 6) Priority-BFS
                if "priority_bfs" in methods_to_run:
                    rec = _record_common_info(mod, dataset_key, dataset, data, split_path, split_id, repeat, "priority_bfs", current_seed)
                    t1 = time.perf_counter()
                    _, acc_test_priority = mod.priority_bfs_predictclass(
                        data,
                        train_np,
                        test_np,
                        **dict_params_prop,
                        seed=current_seed,
                        log_prefix=f"{dataset_key}:STAGE2-PRIORITY-BFS",
                        log_file=None,
                        margin_threshold=0.05,
                        refinement_margin_threshold=0.02,
                        max_deferrals_per_node=5,
                        record_diagnostics=False,
                    )
                    method_time = time.perf_counter() - t1
                    rec.update(
                        {
                            "val_acc": None,
                            "test_acc": float(acc_test_priority),
                            "tuning_runtime_sec": tuning_time_prop,
                            "method_runtime_sec": method_time,
                            "total_runtime_sec": tuning_time_prop + method_time,
                            "parameter_source": "tuned_on_predictclass_then_reused",
                            "parameter_tuning_target_method": "predictclass",
                            "tuned_separately_for_this_method": False,
                            "is_distinct_method_variant": True,
                            "parent_method": "prop_only",
                            "a_values": params_prop,
                        }
                    )
                    records.append(rec)

                # 7) Ensemble multi-source BFS (distinct because of multiple restarts)
                if "ensemble_multi_source_bfs" in methods_to_run:
                    rec = _record_common_info(mod, dataset_key, dataset, data, split_path, split_id, repeat, "ensemble_multi_source_bfs", current_seed)
                    t1 = time.perf_counter()
                    _, acc_test_ens = mod.ensemble_multi_source_bfs_predictclass(
                        data,
                        train_np,
                        test_np,
                        **dict_params_prop,
                        num_restarts=5,
                        base_seed=current_seed,
                        seed_mode="all_train",
                        seeds_per_class=None,
                        margin_threshold=0.05,
                        refinement_margin_threshold=0.02,
                        max_deferrals_per_node=5,
                        log_prefix=f"{dataset_key}:STAGE2-ENS-MULTI-SOURCE",
                        log_file=None,
                    )
                    method_time = time.perf_counter() - t1
                    rec.update(
                        {
                            "val_acc": None,
                            "test_acc": float(acc_test_ens),
                            "tuning_runtime_sec": tuning_time_prop,
                            "method_runtime_sec": method_time,
                            "total_runtime_sec": tuning_time_prop + method_time,
                            "parameter_source": "tuned_on_predictclass_then_reused",
                            "parameter_tuning_target_method": "predictclass",
                            "tuned_separately_for_this_method": False,
                            "is_distinct_method_variant": True,
                            "parent_method": "priority_bfs",
                            "a_values": params_prop,
                            "num_restarts": 5,
                            "vote_aggregation": "majority_vote_smallest_label_tie_break",
                        }
                    )
                    records.append(rec)

    # Write Stage 2 outputs
    os.makedirs("logs", exist_ok=True)
    runs_path = os.path.join("logs", "comparison_runs_stage2.jsonl")
    with open(runs_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # Simple summary: per-dataset/method mean/std of test_acc (ignoring None)
    summary: List[Dict[str, Any]] = []
    by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in records:
        key = (r["dataset"], r["method"])
        by_key.setdefault(key, []).append(r)
    for (ds, method), items in by_key.items():
        test_vals = [float(rr["test_acc"]) for rr in items if rr.get("test_acc") is not None]
        arr = np.array(test_vals, dtype=float) if test_vals else np.array([], dtype=float)
        runtime_vals = [float(rr["total_runtime_sec"]) for rr in items if rr.get("total_runtime_sec") is not None]
        success_count = int(sum(1 for rr in items if bool(rr.get("success", False))))
        total_count = int(len(items))
        parameter_sources = sorted({str(rr["parameter_source"]) for rr in items if rr.get("parameter_source") is not None})
        tuning_targets = sorted(
            {str(rr["parameter_tuning_target_method"]) for rr in items if rr.get("parameter_tuning_target_method") is not None}
        )
        summary.append(
            {
                "dataset": ds,
                "method": method,
                "n_runs": int(arr.size),
                "n_runs_with_test_acc": int(arr.size),
                "mean_test_acc": float(arr.mean()) if arr.size > 0 else None,
                "std_test_acc": float(arr.std()) if arr.size > 0 else None,
                "mean_total_runtime_sec": float(np.mean(runtime_vals)) if runtime_vals else None,
                "success_count": success_count,
                "total_count": total_count,
                "parameter_source_values": parameter_sources,
                "parameter_tuning_target_method_values": tuning_targets,
            }
        )

    summary_path = os.path.join("logs", "comparison_summary_stage2.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Fairness / configuration report (Stage 2)
    fairness_path = os.path.join("logs", "comparison_fairness_report_stage2.md")
    with open(fairness_path, "w", encoding="utf-8") as f:
        f.write("# Fairness / Configuration Report – Stage 2 Internal Pilot (texas, chameleon)\n\n")
        f.write("## Datasets\n\n")
        f.write("- texas\n- chameleon\n\n")
        f.write("## Splits and repeats\n\n")
        f.write("- Splits: GEO-GCN splits (split_id = 0..9)\n")
        f.write("- Repeats per split: 1\n\n")
        f.write("## Methods included\n\n")
        for m in methods_to_run:
            if m == "multi_source_priority_bfs":
                f.write(f"- {m} (excluded from Stage 2 summary; not distinct from priority-BFS yet)\n")
            else:
                f.write(f"- {m}\n")
        f.write("\n## Parameter sharing\n\n")
        f.write("- `params_prop` tuned once per dataset/split/repeat via `predictclass` (robust_random_search) and reused by:\n")
        f.write("  - prop_only\n  - priority_bfs\n  - ensemble_multi_source_bfs\n")
        if "gated_mlp_prop" in methods_to_run:
            f.write("  - gated_mlp_prop (plus a validation-trained node-wise gate)\n")
        f.write("- `params_hyb` tuned separately for `mlp_refined`.\n")
        f.write("- MLP hyperparameters are specific to `mlp_only`.\n\n")
        f.write("## Runtime accounting policy\n\n")
        f.write("- `tuning_runtime_sec` measures the time spent in hyperparameter search or training (for MLP).\n")
        f.write("- `method_runtime_sec` measures the time for the final evaluation calls (val/test) using tuned parameters.\n")
        f.write("- `total_runtime_sec = tuning_runtime_sec + method_runtime_sec`.\n")
        f.write("- For methods that reuse parameters (e.g., priority_bfs, ensemble_multi_source_bfs), the original tuning cost is reported again so that\n")
        f.write("  comparisons can be made either including or excluding amortized tuning cost.\n\n")
        f.write("## Possible unfairness / caveats\n\n")
        f.write("- Parameter reuse: `params_prop` tuned on `predictclass` and reused by `prop_only`, `priority_bfs`, and `ensemble_multi_source_bfs`.\n")
        f.write("- Experimental methods: `priority_bfs` and `ensemble_multi_source_bfs` are more recent, complex propagation variants.\n")
        f.write("- Hybrid method: `mlp_refined` uses MLP pseudo-labels to augment the train set before propagation.\n")
        f.write("- Runtime: ensemble multi-source BFS incurs extra runtime by design due to multiple restarts.\n")
        f.write("- Order sensitivity: propagation methods remain order-sensitive due to priority-queue traversal and sequential commits.\n\n")
        if "gated_mlp_prop" in methods_to_run:
            f.write("- Gate data sparsity: `gated_mlp_prop` trains only on validation nodes where MLP/prop disagree in correctness, which can be small on some splits.\n\n")
        f.write("## Methods excluded or merged\n\n")
        f.write("- `multi_source_priority_bfs_predictclass` currently uses the same seed set (`seed_mode=\"all_train\"`) as `priority_bfs`.\n")
        f.write("  It is therefore **not reported as a separate row**; it is treated as a configuration wrapper around priority-BFS.\n")

    print(f"Wrote Stage 2 per-run records to {runs_path}")
    print(f"Wrote Stage 2 summary to {summary_path}")
    print(f"Wrote Stage 2 fairness report to {fairness_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run Stage 2 internal comparison harness.")
    parser.add_argument(
        "--split-dir",
        default=None,
        help="Optional directory containing GEO-GCN split .npz files.",
    )
    args = parser.parse_args()

    run_stage2_internal(split_dir=args.split_dir)


if __name__ == "__main__":
    main()


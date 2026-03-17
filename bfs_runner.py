import os
import time
import json
import importlib.util
from typing import Tuple, Optional

import numpy as np
import torch


def _load_full_investigate_module():
    """
    Dynamically load bfsbased-full-investigate-homophil.py as a module.
    This avoids renaming the original script while letting us reuse its helpers.
    """
    path = os.path.join(os.path.dirname(__file__), "bfsbased-full-investigate-homophil.py")
    spec = importlib.util.spec_from_file_location("bfs_full_investigate", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[arg-type]
    return mod


def _load_dataset_and_data(dataset_key: str):
    """
    Use the existing load_dataset(DATASET_KEY, root) from bfsbased-full-investigate-homophil.py
    to construct a (dataset, data) pair identical to the main script.
    """
    mod = _load_full_investigate_module()
    root = "data/"
    dataset = mod.load_dataset(dataset_key, root)
    data = dataset[0]

    # Apply the same undirected + unique-edge preprocessing as in the main script.
    data.edge_index = torch.cat([data.edge_index, data.edge_index.flip(0)], dim=1)
    data.edge_index = torch.unique(data.edge_index, dim=1)
    return mod, dataset, data


def _load_split_from_npz(npz_path: str, device: torch.device):
    """
    Load train/val/test indices from a GEO-GCN style split .npz file.
    """
    split = np.load(npz_path)
    train_idx = torch.as_tensor(np.where(split["train_mask"])[0], dtype=torch.long, device=device)
    val_idx = torch.as_tensor(np.where(split["val_mask"])[0], dtype=torch.long, device=device)
    test_idx = torch.as_tensor(np.where(split["test_mask"])[0], dtype=torch.long, device=device)
    return train_idx, val_idx, test_idx


def run_bfs_baseline(
    dataset_key: str,
    split_npz_path: str,
    *,
    use_priority: bool = False,
    seed: int = 1337,
) -> Tuple[Optional[float], float, float]:
    """
    Run our BFS-based classifier on a single dataset/split.

    Returns:
        (val_acc, test_acc_plain, test_acc_priority)
        - If use_priority is False, test_acc_priority == test_acc_plain.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    mod, dataset, data = _load_dataset_and_data(dataset_key)
    device = data.x.device
    num_nodes = int(data.num_nodes)

    train_idx, val_idx, test_idx = _load_split_from_npz(split_npz_path, device=device)
    train_np = mod._to_numpy_idx(train_idx)
    val_np = mod._to_numpy_idx(val_idx)
    test_np = mod._to_numpy_idx(test_idx)

    # Hyperparameter search for propagation-only using the existing robust_random_search.
    rs_params, _, _ = mod.robust_random_search(
        mod.predictclass,
        data,
        train_np,
        num_samples=mod.RS_NUM_SAMPLES if hasattr(mod, "RS_NUM_SAMPLES") else 50,
        n_splits=mod.RS_SPLITS if hasattr(mod, "RS_SPLITS") else 3,
        repeats=mod.RS_REPEATS if hasattr(mod, "RS_REPEATS") else 1,
        seed=seed,
        log_file=None,
        mlp_probs=None,
    )
    if rs_params is None or len(rs_params) != 8:
        raise RuntimeError("robust_random_search returned invalid parameters for BFS baseline.")

    params_dict = {f"a{i+1}": p for i, p in enumerate(rs_params)}

    # Plain BFS propagation: val and test.
    _, val_acc_plain = mod.predictclass(
        data,
        train_np,
        val_np,
        **params_dict,
        seed=seed,
        mlp_probs=None,
        log_file=None,
    )
    _, test_acc_plain = mod.predictclass(
        data,
        train_np,
        test_np,
        **params_dict,
        seed=seed,
        mlp_probs=None,
        log_file=None,
    )

    if not use_priority:
        return float(val_acc_plain), float(test_acc_plain), float(test_acc_plain)

    # Priority-BFS variant, using the same tuned parameters.
    if not hasattr(mod, "priority_bfs_predictclass"):
        # If the new function is not present, fall back to plain BFS.
        return float(val_acc_plain), float(test_acc_plain), float(test_acc_plain)

    _, test_acc_priority = mod.priority_bfs_predictclass(
        data,
        train_np,
        test_np,
        **params_dict,
        seed=seed,
        log_prefix=f"{dataset_key}:PRIORITY-BFS",
        log_file=None,
        record_diagnostics=False,
    )

    return float(val_acc_plain), float(test_acc_plain), float(test_acc_priority)


def main():
    """
    Simple manual entry point for testing this runner directly.
    Not used by the comparison harness, but handy for sanity checks.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset key, e.g., texas, wisconsin, cornell, ...")
    parser.add_argument("--split_npz", required=True, help="Path to GEO-GCN split .npz file.")
    parser.add_argument("--use_priority", action="store_true", help="Also run the priority-BFS variant.")
    args = parser.parse_args()

    t0 = time.time()
    val_acc, test_plain, test_prior = run_bfs_baseline(
        args.dataset,
        args.split_npz,
        use_priority=args.use_priority,
    )
    elapsed = time.time() - t0
    record = {
        "method": "bfs_priority_v1" if args.use_priority else "bfs_plain",
        "dataset": args.dataset,
        "split_npz": args.split_npz,
        "val_acc": val_acc,
        "test_acc_plain": test_plain,
        "test_acc_priority": test_prior,
        "runtime_sec": elapsed,
    }
    os.makedirs("logs", exist_ok=True)
    out_path = os.path.join("logs", f"bfs_runner_{args.dataset}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()


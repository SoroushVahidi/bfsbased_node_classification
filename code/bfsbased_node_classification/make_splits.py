#!/usr/bin/env python3
import argparse
import os
from typing import Dict, Tuple

import numpy as np


def _stratified_split(
    y: np.ndarray,
    *,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    y = np.asarray(y, dtype=np.int64)
    n = int(y.shape[0])

    train_idx = []
    val_idx = []
    test_idx = []

    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_c = int(idx.size)
        n_train = int(np.floor(train_frac * n_c))
        n_val = int(np.floor(val_frac * n_c))
        # remainder goes to test
        tr = idx[:n_train]
        va = idx[n_train : n_train + n_val]
        te = idx[n_train + n_val :]
        train_idx.append(tr)
        val_idx.append(va)
        test_idx.append(te)

    train_idx = np.concatenate(train_idx) if train_idx else np.array([], dtype=np.int64)
    val_idx = np.concatenate(val_idx) if val_idx else np.array([], dtype=np.int64)
    test_idx = np.concatenate(test_idx) if test_idx else np.array([], dtype=np.int64)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    # Safety: ensure disjoint + cover all nodes
    all_idx = np.concatenate([train_idx, val_idx, test_idx])
    if all_idx.size != n or np.unique(all_idx).size != n:
        raise RuntimeError("Split masks are not a partition of nodes.")

    return train_idx, val_idx, test_idx


def _idx_to_mask(idx: np.ndarray, n: int) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    m[np.asarray(idx, dtype=np.int64)] = True
    return m


def make_split_npz(
    *,
    dataset_key: str,
    y: np.ndarray,
    split_id: int,
    out_dir: str,
    train_frac: float,
    val_frac: float,
    seed_base: int,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    n = int(y.shape[0])
    seed = int(seed_base) + int(split_id)
    tr, va, te = _stratified_split(y, train_frac=train_frac, val_frac=val_frac, seed=seed)
    payload: Dict[str, np.ndarray] = {
        "train_mask": _idx_to_mask(tr, n),
        "val_mask": _idx_to_mask(va, n),
        "test_mask": _idx_to_mask(te, n),
    }
    out_path = os.path.join(out_dir, f"{dataset_key.lower()}_split_0.6_0.2_{split_id}.npz")
    np.savez_compressed(out_path, **payload)
    return out_path


def main():
    p = argparse.ArgumentParser(description="Generate minimal split .npz files (train/val/test masks).")
    p.add_argument("--dataset", required=True, help="Dataset key (e.g., actor, cornell).")
    p.add_argument("--out-dir", default=os.path.join("data", "splits"), help="Output directory for .npz splits.")
    p.add_argument("--num-splits", type=int, default=10, help="How many split IDs to generate (default 10: 0..9).")
    p.add_argument("--train-frac", type=float, default=0.6)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed-base", type=int, default=1337)
    args = p.parse_args()

    try:
        import torch
        from torch_geometric.datasets import Actor, WebKB
    except Exception as e:
        raise SystemExit(
            "Missing torch/torch_geometric. Create the conda env first, then rerun.\n"
            f"Import error: {e}"
        )

    ds = args.dataset.lower()
    root = "data"
    if ds == "actor":
        dataset = Actor(root=os.path.join(root, "Film"))
    elif ds == "cornell":
        dataset = WebKB(root=root, name="Cornell")
    else:
        raise SystemExit(f"Unsupported dataset={args.dataset!r}. Only actor/cornell for now.")

    data = dataset[0]
    y = data.y.detach().cpu().numpy()

    made = []
    for sid in range(int(args.num_splits)):
        out = make_split_npz(
            dataset_key=ds,
            y=y,
            split_id=sid,
            out_dir=args.out_dir,
            train_frac=float(args.train_frac),
            val_frac=float(args.val_frac),
            seed_base=int(args.seed_base),
        )
        made.append(out)

    print("Wrote split files:")
    for m in made:
        print(f"  - {m}")


if __name__ == "__main__":
    main()


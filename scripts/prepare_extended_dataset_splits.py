#!/usr/bin/env python3
"""
Materialize split .npz files for ogbn-arxiv, ogbn-products, and TabGraphs hm-categories.

- OGB: uses official train/valid/test indices from ``get_idx_split()`` (split_id fixed to 0).
- hm-categories: reads official train_mask.csv / valid_mask.csv / test_mask.csv from the
  unzipped TabGraphs folder (row order aligned with features.csv).

Run from repository root:

  python3 scripts/prepare_extended_dataset_splits.py --dataset ogbn-arxiv
  python3 scripts/prepare_extended_dataset_splits.py --dataset ogbn-products
  python3 scripts/prepare_extended_dataset_splits.py --dataset hm-categories \\
      --tabgraphs-dir data/tabgraphs/hm-categories

Output: data/splits/{prefix}_split_0.6_0.2_0.npz
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_CODE = os.path.join(_REPO_ROOT, "code", "bfsbased_node_classification")
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)


def _write_npz(out_dir: str, prefix: str, train: np.ndarray, val: np.ndarray, test: np.ndarray) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{prefix}_split_0.6_0.2_0.npz")
    np.savez_compressed(path, train_mask=train, val_mask=val, test_mask=test)
    return path


def prepare_ogb(name: str, *, data_root: str, out_dir: str) -> str:
    try:
        from ogb.nodeproppred import NodePropPredDataset
    except ImportError as e:
        raise SystemExit("Install ogb: pip install ogb") from e

    from split_paths import split_npz_prefix

    ogb_root = os.path.join(data_root, "ogb")
    ds = NodePropPredDataset(name=name, root=ogb_root)
    split_idx = ds.get_idx_split()
    graph, _label = ds[0]
    n = int(graph["num_nodes"])
    train_m = np.zeros(n, dtype=bool)
    val_m = np.zeros(n, dtype=bool)
    test_m = np.zeros(n, dtype=bool)

    def as_idx(key: str) -> np.ndarray:
        x = split_idx[key]
        if hasattr(x, "numpy"):
            x = x.numpy()
        return np.asarray(x, dtype=np.int64).reshape(-1)

    train_m[as_idx("train")] = True
    val_m[as_idx("valid")] = True
    test_m[as_idx("test")] = True

    prefix = split_npz_prefix(name)
    path = _write_npz(out_dir, prefix, train_m, val_m, test_m)
    if not bool(np.all(train_m | val_m | test_m)):
        print(
            f"WARNING: {name} official split does not cover all {n} nodes; "
            "some pipeline steps assume GEO-style full coverage.",
            file=sys.stderr,
        )
    return path


def prepare_hm_categories(tabgraphs_dir: str, *, out_dir: str) -> str:
    try:
        import pandas as pd
    except ImportError as e:
        raise SystemExit("Install pandas: pip install pandas") from e

    from split_paths import split_npz_prefix

    folder = os.path.abspath(tabgraphs_dir)
    feats = pd.read_csv(os.path.join(folder, "features.csv"), index_col=0)
    n = len(feats)

    def load_mask(fname: str) -> np.ndarray:
        raw = pd.read_csv(os.path.join(folder, fname), index_col=0)
        ser = raw.iloc[:, 0] if raw.shape[1] else raw.squeeze()
        ser = ser.reindex(feats.index)
        if ser.isna().any():
            raise ValueError(f"{fname} is missing rows for some feature indices")
        m = ser.values.astype(bool).reshape(-1)
        if m.size != n:
            raise ValueError(f"{fname} length {m.size} != num nodes {n}")
        return m

    train_m = load_mask("train_mask.csv")
    val_m = load_mask("valid_mask.csv")
    test_m = load_mask("test_mask.csv")
    if (train_m & val_m).any() or (train_m & test_m).any() or (val_m & test_m).any():
        raise ValueError("Overlapping masks in TabGraphs split files.")
    if not (train_m | val_m | test_m).all():
        raise ValueError("TabGraphs masks do not cover all nodes; refusing to write npz.")

    prefix = split_npz_prefix("hm-categories")
    return _write_npz(out_dir, prefix, train_m, val_m, test_m)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        required=True,
        choices=("ogbn-arxiv", "ogbn-products", "hm-categories"),
    )
    p.add_argument("--data-root", default=os.path.join(_REPO_ROOT, "data"))
    p.add_argument("--out-dir", default=None, help="Default: {data-root}/splits")
    p.add_argument(
        "--tabgraphs-dir",
        default=None,
        help="Unzipped hm-categories folder (default: {data-root}/tabgraphs/hm-categories)",
    )
    args = p.parse_args()
    out_dir = args.out_dir or os.path.join(args.data_root, "splits")

    if args.dataset in ("ogbn-arxiv", "ogbn-products"):
        path = prepare_ogb(args.dataset, data_root=args.data_root, out_dir=out_dir)
    else:
        tg = args.tabgraphs_dir or os.path.join(args.data_root, "tabgraphs", "hm-categories")
        path = prepare_hm_categories(tg, out_dir=out_dir)

    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

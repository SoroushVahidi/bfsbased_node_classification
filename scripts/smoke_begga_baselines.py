#!/usr/bin/env python3
"""CPU smoke test: all five Begga heterophily baselines.

If ``data/splits/<dataset>_split_*.npz`` files exist, uses ``texas`` split 0 via
``manuscript_runner``. Otherwise builds a small synthetic ``torch_geometric``
graph so the smoke test still validates imports and forward/training loops.
"""
from __future__ import annotations

import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_CODE = os.path.join(_REPO, "code", "bfsbased_node_classification")
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

import torch  # noqa: E402
from torch_geometric.data import Data  # noqa: E402

from standard_node_baselines import (  # noqa: E402
    BEGGA_HETEROPHILY_BASELINE_METHODS,
    run_baseline,
)


def _synthetic_bundle():
    torch.manual_seed(0)
    n, f, c = 120, 16, 4
    edge_index = torch.randint(0, n, (2, 500), dtype=torch.long)
    x = torch.randn(n, f)
    y = torch.randint(0, c, (n,), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y, num_nodes=n)
    train_idx = torch.arange(0, 60, dtype=torch.long)
    val_idx = torch.arange(60, 85, dtype=torch.long)
    test_idx = torch.arange(85, 120, dtype=torch.long)
    return data, train_idx, val_idx, test_idx


def main() -> None:
    split_dir = os.path.join(_REPO, "data", "splits")
    use_real = os.path.isdir(split_dir) and any(
        fn.endswith(".npz") for fn in os.listdir(split_dir)
    )
    if use_real:
        import manuscript_runner as mr

        mod = mr._load_full_investigate_module()
        dataset_key = "texas"
        split_id = 0
        _, data = mr._load_dataset_and_data(mod, dataset_key)
        device = data.x.device
        train_idx, val_idx, test_idx, _ = mr._load_split_npz(
            dataset_key, split_id, device, split_dir
        )
        label = f"dataset={dataset_key} split={split_id}"
    else:
        data, train_idx, val_idx, test_idx = _synthetic_bundle()
        label = "synthetic graph (no data/splits/*.npz in checkout)"

    seed = 42
    max_epochs = 80
    patience = 40
    print(f"Begga baselines smoke ({label})")
    for method in BEGGA_HETEROPHILY_BASELINE_METHODS:
        torch.manual_seed(seed)
        r = run_baseline(
            method,
            data,
            train_idx,
            val_idx,
            test_idx,
            seed=seed,
            max_epochs=max_epochs,
            patience=patience,
        )
        print(f"  {method:22s}  val={r.val_acc:.4f}  test={r.test_acc:.4f}")


if __name__ == "__main__":
    main()

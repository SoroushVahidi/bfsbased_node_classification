"""
Shared data loading and split adapter for all benchmark baselines.
Provides clean interface to load PyG datasets and canonical splits.
"""
import os
import sys
import numpy as np

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_BENCH_DIR)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'code', 'bfsbased_node_classification'))

DATASET_INFO = {
    "cora":      ("Planetoid",      "Cora"),
    "citeseer":  ("Planetoid",      "CiteSeer"),
    "pubmed":    ("Planetoid",      "PubMed"),
    "chameleon": ("WikipediaNetwork","chameleon"),
    "squirrel":  ("WikipediaNetwork","squirrel"),
    "texas":     ("WebKB",          "Texas"),
    "cornell":   ("WebKB",          "Cornell"),
    "wisconsin": ("WebKB",          "Wisconsin"),
    "actor":     ("Actor",          None),
}

_SPLIT_PREFIX_OVERRIDE = {"actor": "film"}


def _get_default_data_root():
    return os.path.join(_REPO_ROOT, "data", "pyg_cache")


def _get_default_split_dir():
    return os.path.join(_REPO_ROOT, "data", "splits")


def load_dataset(dataset_key: str, data_root=None):
    """Load a PyG dataset object for the given key."""
    try:
        import torch
        from torch_geometric import datasets as pyg_datasets
    except ImportError as e:
        raise ImportError(f"PyTorch Geometric is required: {e}")

    if dataset_key not in DATASET_INFO:
        raise ValueError(f"Unknown dataset '{dataset_key}'. Known: {list(DATASET_INFO.keys())}")

    loader_cls_name, name_arg = DATASET_INFO[dataset_key]

    if data_root is None:
        data_root = _get_default_data_root()
    os.makedirs(data_root, exist_ok=True)

    ds_root = os.path.join(data_root, dataset_key)

    if loader_cls_name == "Planetoid":
        dataset = pyg_datasets.Planetoid(root=ds_root, name=name_arg)
    elif loader_cls_name == "WikipediaNetwork":
        dataset = pyg_datasets.WikipediaNetwork(
            root=ds_root, name=name_arg, geom_gcn_preprocess=True
        )
    elif loader_cls_name == "WebKB":
        dataset = pyg_datasets.WebKB(root=ds_root, name=name_arg)
    elif loader_cls_name == "Actor":
        dataset = pyg_datasets.Actor(root=ds_root)
    else:
        raise ValueError(f"Unknown loader class '{loader_cls_name}'")

    return dataset


def load_split(dataset_key: str, split_id: int, data_root=None, split_dir=None):
    """
    Load a canonical split npz for the given dataset and split_id.

    Returns dict with keys:
      train_idx, val_idx, test_idx  (LongTensor)
      train_mask, val_mask, test_mask  (BoolTensor)
    """
    import torch

    if split_dir is None:
        split_dir = _get_default_split_dir()

    prefix = _SPLIT_PREFIX_OVERRIDE.get(dataset_key, dataset_key)
    split_file = os.path.join(split_dir, f"{prefix}_split_0.6_0.2_{split_id}.npz")

    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")

    data = np.load(split_file)
    train_mask = torch.tensor(data["train_mask"], dtype=torch.bool)
    val_mask   = torch.tensor(data["val_mask"],   dtype=torch.bool)
    test_mask  = torch.tensor(data["test_mask"],  dtype=torch.bool)

    train_idx = train_mask.nonzero(as_tuple=False).squeeze(1)
    val_idx   = val_mask.nonzero(as_tuple=False).squeeze(1)
    test_idx  = test_mask.nonzero(as_tuple=False).squeeze(1)

    return dict(
        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
        train_idx=train_idx,   val_idx=val_idx,   test_idx=test_idx,
        split_file=split_file,
    )


def build_data_bundle(dataset_key: str, split_id: int, data_root=None, split_dir=None):
    """
    Build a complete data bundle for the given dataset and split.

    Returns dict with:
      x, y, edge_index, train_mask, val_mask, test_mask,
      train_idx, val_idx, test_idx,
      n_nodes, n_features, n_classes, dataset_key, split_id, split_file
    """
    import torch
    from torch_geometric.utils import to_undirected

    dataset = load_dataset(dataset_key, data_root=data_root)
    split   = load_split(dataset_key, split_id, data_root=data_root, split_dir=split_dir)

    # For multi-graph datasets (WebKB, WikipediaNetwork), use graph 0
    pyg_data = dataset[0]

    x = pyg_data.x.float()
    y = pyg_data.y.long()

    # If y is 2-D (one-hot or multi-label), take argmax
    if y.dim() > 1:
        y = y.argmax(dim=1)

    # Symmetrize and deduplicate edge_index
    edge_index = to_undirected(pyg_data.edge_index, num_nodes=x.size(0))

    n_nodes    = x.size(0)
    n_features = x.size(1)
    n_classes  = int(y.max().item()) + 1

    bundle = dict(
        x=x,
        y=y,
        edge_index=edge_index,
        train_mask=split["train_mask"],
        val_mask=split["val_mask"],
        test_mask=split["test_mask"],
        train_idx=split["train_idx"],
        val_idx=split["val_idx"],
        test_idx=split["test_idx"],
        n_nodes=n_nodes,
        n_features=n_features,
        n_classes=n_classes,
        dataset_key=dataset_key,
        split_id=split_id,
        split_file=split["split_file"],
    )
    return bundle

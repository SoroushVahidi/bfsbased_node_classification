"""
Loaders for OGB node-property datasets and TabGraphs-style CSV benchmarks.

Sources:
- OGB: https://snap-stanford.github.io/ogb-web/docs/nodeprop/  (pip package `ogb`)
- TabGraphs: https://github.com/yandex-research/tabgraphs  (Zenodo archives; hm-categories)
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data


class SingleGraphDataset:
    """Minimal PyG-style dataset wrapping one `Data` object (matches `dataset[0]` usage)."""

    def __init__(self, data: Data, num_classes: int):
        self._data = data
        self.num_classes = int(num_classes)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> Data:
        if idx != 0:
            raise IndexError("Single-graph dataset only supports index 0")
        return self._data


def _normalize_ogb_name(dataset_key: str) -> str:
    k = dataset_key.lower().replace("_", "-")
    if k not in {"ogbn-arxiv", "ogbn-products"}:
        raise ValueError(f"Not an OGB node dataset key: {dataset_key!r}")
    return k


def load_ogb_node_dataset(dataset_key: str, root: str) -> SingleGraphDataset:
    """
    Load ogbn-arxiv or ogbn-products via official OGB Python loaders.

    `root` should be the repo data root (e.g. 'data/'); downloads go under
    ``{root}/ogb/`` by default.
    """
    try:
        from ogb.nodeproppred import NodePropPredDataset
    except ImportError as e:
        raise ImportError(
            "The `ogb` package is required for OGB datasets. Install with:\n"
            "  pip install ogb\n"
        ) from e

    name = _normalize_ogb_name(dataset_key)
    ogb_root = os.path.join(os.path.expanduser(root.rstrip("/")), "ogb")
    os.makedirs(ogb_root, exist_ok=True)

    dataset = NodePropPredDataset(name=name, root=ogb_root)
    graph: Dict[str, Any]
    label: np.ndarray
    graph, label = dataset[0]

    edge_index = torch.from_numpy(graph["edge_index"]).long()
    x = torch.from_numpy(graph["node_feat"]).float()
    y = torch.from_numpy(label).view(-1).long()
    num_nodes = int(graph.get("num_nodes", x.size(0)))
    if y.numel() != num_nodes:
        raise RuntimeError(f"OGB label length {y.numel()} != num_nodes {num_nodes}")

    data = Data(x=x, edge_index=edge_index, y=y)
    data.num_nodes = num_nodes

    n_cls = int(dataset.num_classes)
    return SingleGraphDataset(data, n_cls)


def _read_tabgraphs_folder(
    folder: str,
) -> Tuple[Data, int, Dict[str, Any]]:
    """
    Load TabGraphs CSV layout (features.csv, edgelist.csv, masks, info.yaml).
    Preprocessing mirrors ``source/gnns/datasets.py`` (without DGL): imputation,
    numeric scaling, one-hot categoricals.
    """
    try:
        import yaml
        import pandas as pd
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except ImportError as e:
        raise ImportError(
            "hm-categories requires PyYAML, pandas, and scikit-learn. Install with:\n"
            "  pip install pyyaml pandas scikit-learn\n"
        ) from e

    info_path = os.path.join(folder, "info.yaml")
    if not os.path.isfile(info_path):
        raise FileNotFoundError(
            f"TabGraphs dataset folder missing info.yaml: {folder}\n"
            "Download the benchmark zip from Zenodo (see docs/DATASETS_EXTENDED.md), "
            "unzip, and point --tabgraphs-dir at the hm-categories folder."
        )

    with open(info_path, "r", encoding="utf-8") as f:
        info = yaml.safe_load(f)

    features_df = pd.read_csv(os.path.join(folder, "features.csv"), index_col=0)
    id2row = {nid: i for i, nid in enumerate(features_df.index)}
    n = len(features_df)

    num_names: List[str] = list(info.get("num_feature_names") or [])
    bin_names: List[str] = list(info.get("bin_feature_names") or [])
    cat_names: List[str] = list(info.get("cat_feature_names") or [])
    target_name = info["target_name"]
    task = info.get("task", "multiclass_classification")

    if task != "multiclass_classification":
        raise ValueError(
            f"hm-categories loader currently supports only multiclass_classification "
            f"(got task={task!r}). Extend extended_graph_datasets.py if needed."
        )

    X_blocks: List[np.ndarray] = []

    if num_names:
        X_num = features_df[num_names].values.astype(np.float32)
        if info.get("has_nans_in_num_features"):
            X_num = SimpleImputer(strategy="median").fit_transform(X_num).astype(np.float32)
        X_num = StandardScaler().fit_transform(X_num).astype(np.float32)
        X_blocks.append(X_num)
    if bin_names:
        X_blocks.append(features_df[bin_names].values.astype(np.float32))
    if cat_names:
        X_cat = features_df[cat_names].values
        enc = OneHotEncoder(sparse_output=False, dtype=np.float32, handle_unknown="ignore")
        X_blocks.append(enc.fit_transform(X_cat))

    if not X_blocks:
        raise ValueError("No feature columns found in info.yaml for hm-categories.")

    x_np = np.concatenate(X_blocks, axis=1)
    y_np = features_df[target_name].values.astype(np.int64)

    edges_df = pd.read_csv(os.path.join(folder, "edgelist.csv"))
    src = edges_df.iloc[:, 0].to_numpy()
    dst = edges_df.iloc[:, 1].to_numpy()

    def row_pos(node_id) -> int:
        if node_id in id2row:
            return int(id2row[node_id])
        key_int = int(node_id)
        if key_int in id2row:
            return int(id2row[key_int])
        ks = str(node_id)
        if ks in id2row:
            return int(id2row[ks])
        raise KeyError(f"Edge endpoint {node_id!r} not in features.csv index")

    ei_list: List[Tuple[int, int]] = []
    for u, v in zip(src, dst):
        ei_list.append((row_pos(u), row_pos(v)))

    edge_index = np.array(ei_list, dtype=np.int64).T
    x = torch.from_numpy(x_np).float()
    y = torch.from_numpy(y_np).long()
    ei = torch.from_numpy(edge_index).long()

    data = Data(x=x, edge_index=ei, y=y)
    data.num_nodes = n
    num_classes = int(info["num_classes"])
    return data, num_classes, info


def load_hm_categories_dataset(root: str) -> SingleGraphDataset:
    """
    Load TabGraphs ``hm-categories`` from ``{root}/tabgraphs/hm-categories/``
    (or override with env TABGRAPHS_HM_CATEGORIES_DIR).
    """
    env_dir = os.environ.get("TABGRAPHS_HM_CATEGORIES_DIR", "").strip()
    if env_dir:
        folder = os.path.abspath(env_dir)
    else:
        folder = os.path.join(os.path.expanduser(root.rstrip("/")), "tabgraphs", "hm-categories")
    if not os.path.isdir(folder):
        alt = os.path.join(os.path.expanduser(root.rstrip("/")), "tabgraphs", "hm_categories")
        if os.path.isdir(alt):
            folder = alt
        else:
            raise FileNotFoundError(
                f"hm-categories data not found at:\n  {folder}\n"
                f"Set TABGRAPHS_HM_CATEGORIES_DIR to the unzipped dataset directory, "
                f"or place files under data/tabgraphs/hm-categories/. "
                f"See docs/DATASETS_EXTENDED.md."
            )

    data, num_classes, _info = _read_tabgraphs_folder(folder)
    return SingleGraphDataset(data, num_classes)

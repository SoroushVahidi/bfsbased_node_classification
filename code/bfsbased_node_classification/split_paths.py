"""
Canonical naming for GEO-style split .npz files under data/splits/.

Filenames use pattern: {prefix}_split_0.6_0.2_{split_id}.npz
"""
from __future__ import annotations

import re

_EXTENDED_SINGLE_SPLIT = frozenset({"ogbn-arxiv", "ogbn-products", "hm-categories"})


def canonical_dataset_key(dataset_key: str) -> str:
    """Normalize user/CLI aliases (hyphens vs underscores) for membership tests."""
    return (dataset_key or "").strip().lower().replace("_", "-")


def split_npz_prefix(dataset_key: str) -> str:
    """
    Prefix used inside split .npz filenames (matches historical repo behavior).

    - actor -> film (GEO-GCN convention)
    - ogbn-arxiv -> ogbn_arxiv (hyphens -> underscores)
    """
    k = (dataset_key or "").strip().lower()
    if k == "actor":
        return "film"
    normalized = re.sub(r"[^a-z0-9]+", "_", k)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or k


def is_official_single_split_dataset(dataset_key: str) -> bool:
    """Benchmark provides one official split; only split_id=0 is supported."""
    return canonical_dataset_key(dataset_key) in _EXTENDED_SINGLE_SPLIT

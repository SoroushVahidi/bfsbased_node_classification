#!/usr/bin/env python3
"""
Deterministic undirected edge corruption for robustness experiments.

Protocol (documented for reviewers):
  1. Collapse ``edge_index`` to unique undirected edges ``(u, v)`` with ``u < v``,
     ignoring self-loops.
  2. Let ``E`` be that set, ``n_e = |E|``.
  3. For ``noise_fraction > 0``, set ``k = min(n_e, max(0, round(noise_fraction * n_e)))``.
  4. Sample ``k`` edges uniformly **without** replacement from ``E`` and remove them.
  5. Sample ``k`` new undirected edges uniformly from all pairs ``(u, v)``, ``u != v``,
     not already present, using rejection sampling (deterministic RNG).
  6. Rebuild a **bidirectional** ``edge_index`` (each undirected edge ``(u,v)`` becomes
     ``(u,v)`` and ``(v,u)``), then ``torch.unique(..., dim=1)``.

Node features, labels, and masks are **not** modified by this module — only ``edge_index``.

At ``noise_fraction == 0``, the returned tensor is a clone of the input (no RNG use).
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch


def corrupt_graph_rewire_equal_count(
    edge_index: torch.Tensor,
    num_nodes: int,
    noise_fraction: float,
    seed: int,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    Remove k random undirected edges and add k random new undirected edges.

    Returns
    -------
    new_edge_index : Tensor [2, E']
        On the same device as ``edge_index``.
    stats : dict
        Counts for logging (undirected edge counts before/after, k, etc.).
    """
    device = edge_index.device
    dtype = edge_index.dtype

    ei = edge_index.detach().cpu().numpy()
    pairs_set = set()
    for j in range(ei.shape[1]):
        u, v = int(ei[0, j]), int(ei[1, j])
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        pairs_set.add((a, b))

    edges_list = list(pairs_set)
    n_e = len(edges_list)
    stats: Dict[str, int] = {
        "n_edges_unique_undir_before": n_e,
        "edges_removed_k": 0,
        "edges_added_k": 0,
        "n_edges_unique_undir_after": n_e,
    }

    if float(noise_fraction) <= 1e-15:
        out = edge_index.clone().to(device=device)
        stats["n_edges_unique_undir_after"] = n_e
        return out, stats

    if n_e == 0:
        out = edge_index.clone().to(device=device)
        stats["n_edges_unique_undir_after"] = 0
        return out, stats

    k = int(round(float(noise_fraction) * n_e))
    k = max(0, min(n_e, k))
    if k == 0:
        out = edge_index.clone().to(device=device)
        return out, stats

    rng = np.random.default_rng(int(seed) % (2**32 - 1))
    remove_ix = rng.choice(n_e, size=k, replace=False)
    removed = {edges_list[i] for i in remove_ix}
    new_set = {e for e in edges_list if e not in removed}
    stats["edges_removed_k"] = k

    added = 0
    max_attempts = max(50000, k * 500)
    attempts = 0
    while added < k and attempts < max_attempts:
        attempts += 1
        u = int(rng.integers(0, num_nodes))
        v = int(rng.integers(0, num_nodes))
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in new_set:
            continue
        new_set.add((a, b))
        added += 1

    stats["edges_added_k"] = added
    stats["n_edges_unique_undir_after"] = len(new_set)

    rows: list = []
    cols: list = []
    for a, b in new_set:
        rows.extend([a, b])
        cols.extend([b, a])
    out = torch.tensor([rows, cols], dtype=dtype, device=device)
    out = torch.unique(out, dim=1)
    return out, stats


CORRUPTION_PROTOCOL_ID = (
    "undirected_equal_count_rewire_v1: remove k=round(f*|E|) random undirected edges, "
    "add k new random absent undirected edges (rejection sampling), rebuild bidirectional edge_index"
)

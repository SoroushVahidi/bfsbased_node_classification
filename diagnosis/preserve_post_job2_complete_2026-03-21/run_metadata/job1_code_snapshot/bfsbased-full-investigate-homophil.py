#!/usr/bin/env python
# coding: utf-8

# In[17]:


import os
import random
import itertools
import heapq
from collections import Counter, defaultdict
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import networkx as nx

# ✅ avoid seaborn (it imports pandas)
import matplotlib.pyplot as plt

from sortedcontainers import SortedSet

from torch_geometric.datasets import WebKB, WikipediaNetwork, Actor, FacebookPagePage, SNAPDataset
from torch_geometric.utils import to_networkx


# 

# In[1]:


from torch_geometric.datasets import Planetoid

from torch_geometric.datasets import LINKXDataset
# =====================================================
# Select dataset (UNCOMMENT ONE LINE ONLY)
# =====================================================

#DATASET_KEY = "cornell"
#DATASET_KEY = "texas"
#DATASET_KEY = "chameleon"
# DATASET_KEY = "squirrel"
#DATASET_KEY = "wisconsin"
# DATASET_KEY = "cora"
# DATASET_KEY = "citeseer"
# DATASET_KEY = "pubmed"
# DATASET_KEY = "actor"
# DATASET_KEY = "penn94"

root = "data/"

# =====================================================
# Dataset loader
# =====================================================

from torch_geometric.datasets import WebKB, WikipediaNetwork, Planetoid, Actor
# from torch_geometric.datasets import LINKXDataset

def load_dataset(dataset_key: str, root: str):
    k = dataset_key.lower()

    if k in {"cornell", "texas", "wisconsin"}:
        return WebKB(root=root, name=k.capitalize())

    if k in {"chameleon", "squirrel"}:
        return WikipediaNetwork(root=root, name=k)

    if k in {"cora", "citeseer", "pubmed"}:
        return Planetoid(root=f"{root}/Planetoid", name=k.capitalize())

    if k == "actor":
        return Actor(root=f"{root}/Film")

    # if k == "penn94":
    #     return LINKXDataset(root=root, name="penn94")

    raise ValueError(f"Unknown DATASET_KEY={dataset_key!r}")

dataset = load_dataset(DATASET_KEY, root)
data = dataset[0]

# =====================================================
# Make graph undirected + remove duplicate edges
# =====================================================

data.edge_index = torch.cat(
    [data.edge_index, data.edge_index.flip(0)],
    dim=1
)
data.edge_index = torch.unique(data.edge_index, dim=1)

print(f"[dataset] loaded {DATASET_KEY} | nodes={data.num_nodes} | edges={data.edge_index.size(1)}")


# ✅ Convert to NetworkX Graph to Find Components
G = to_networkx(data, to_undirected=True)
components = list(nx.connected_components(G))
num_components = len(components)

print(f"\n✅ Loaded  dataset with {data.num_nodes} nodes, {data.edge_index.shape[1]} edges")
print(f"   - Features: {data.x.shape[1]}")
print(f"   - Number of Classes: {len(set(data.y.numpy()))}")
print(f"   - Number of Components: {num_components}")



# In[21]:


import os
import time
import heapq
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict, Counter

# =====================================================
# BLOCK 1: Helper Classes (Timer, Logging)
# =====================================================

class Timer:
    def __init__(self): self.t0 = time.time()
    def reset(self): self.t0 = time.time()
    def elapsed(self): return time.time() - self.t0

def write_log(msg, log_file, tag=None, also_print=False, flush=True):
    """
    Writes to log file. Defaults to NO console output.
    """
    timestamp = time.strftime("%H:%M:%S")
    prefix = f"[{timestamp}]"
    if tag: prefix += f"[{tag}]"
    formatted_msg = f"{prefix} {msg}"
    
    if also_print:
        print(formatted_msg)
        
    if log_file:
        try:
            with open(log_file, "a") as f:
                f.write(formatted_msg + "\n")
                if flush:
                    f.flush()
                    os.fsync(f.fileno())
        except Exception as e:
            print(f"[LOG ERROR] Could not write to {log_file}: {e}")

def log_kv(log_file, tag, **kwargs):
    kvs = " | ".join(f"{k}={v}" for k, v in kwargs.items())
    write_log(kvs, log_file, tag=tag)

def _simple_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.size == 0: return 0.0
    return float((y_true == y_pred).mean())

# ============================================================
# BLOCK 2: Robust Random Search & Homophily Utils
# ============================================================

def edge_homophily_fast(data, edge_index=None, ignore_unlabeled=True, shrinkage=5.0, y=None):
    """
    Computes edge homophily: h = P(y_u == y_v) among edges with labeled endpoints.
    Allows passing custom 'y' to prevent leakage.
    """
    if y is None:
        y = data.y
    if y is None: return float("nan"), 0, 0

    y = y.detach().cpu().numpy().astype(np.int64, copy=False)
    ei = data.edge_index if edge_index is None else edge_index
    ei = ei.detach().cpu().numpy()
    src, dst = ei[0], ei[1]

    if ignore_unlabeled:
        # Assuming -1 denotes unlabeled/masked
        mask = (y[src] >= 0) & (y[dst] >= 0)
        src, dst = src[mask], dst[mask]

    m_used = int(src.size)
    if m_used == 0: return float("nan"), 0, 0

    ys, yd = y[src], y[dst]
    h_raw = float((ys == yd).mean())

    classes = np.unique(np.concatenate([ys, yd], axis=0))
    C_used = max(int(classes.size), 1)

    baseline = 1.0 / float(C_used)
    h_hat = (m_used * h_raw + shrinkage * baseline) / (m_used + shrinkage)

    return float(h_hat), C_used, m_used

def _kfold_plan(indices, n_splits, shuffle, rng):
    indices = np.asarray(indices, dtype=np.int64)
    n = int(indices.size)
    order = indices.copy()
    if shuffle: rng.shuffle(order)
    
    fold_sizes = np.full(n_splits, n // n_splits, dtype=np.int64)
    fold_sizes[: (n % n_splits)] += 1
    bounds = np.zeros(n_splits + 1, dtype=np.int64)
    bounds[1:] = np.cumsum(fold_sizes)
    return order, bounds

def _kfold_train_val_from_plan(order, bounds, k):
    s, e = int(bounds[k]), int(bounds[k + 1])
    if s == 0: tr = order[e:]
    elif e == order.size: tr = order[:s]
    else: tr = np.concatenate([order[:s], order[e:]], axis=0)
    va = order[s:e]
    return tr, va

def _trimmed_mean(x, trim=0.1):
    x = np.asarray(x, dtype=float)
    if x.size == 0: return float("nan")
    x.sort()
    k = int(np.floor(trim * x.size))
    if 2 * k >= x.size: return float(np.mean(x))
    return float(np.mean(x[k:-k]))

def robust_random_search(
    model, data, base_indices,
    *,
    num_samples=50, n_splits=3, repeats=1, maxi=1.0, seed=1337,
    log_file=None, 
    **model_kwargs
):
    """
    Finds hyperparameters a1..a8 based on Graph Homophily.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(int(seed))
    base_indices = np.asarray(base_indices, dtype=np.int64)
    
    # 1. Analyze Homophily (Strictly on Training Set)
    # Mask all non-training nodes to -1 to prevent leakage
    y_train_only = data.y.clone()
    train_mask = torch.zeros(y_train_only.size(0), dtype=torch.bool, device=y_train_only.device)
    train_mask[base_indices] = True
    y_train_only[~train_mask] = -1
    
    h_val, C_used, _ = edge_homophily_fast(data, data.edge_index, y=y_train_only)
    baseline = 1.0 / max(C_used, 1)
    
    # Determine bounds based on Homophily
    is_hetero = h_val < (baseline + 0.05)
    
    if is_hetero:
        # Heterophilic Strategy
        a2_bounds = (-0.5, 0.1)  
        a8_bounds = (0.5, 1.5)
        a3_bounds = (0.0, 0.5) 
        fallback_params = [0.1, -0.1, 0.1, 0.2, 0.1, 0.05, 0.5, 1.0] 
        mode_str = "HETERO"
    else:
        # Homophilic Strategy (Optimized for Cora/Citeseer)
        # a2 (Neighbors) should be dominant [1.0, 8.0]
        # a3 (Features) crucial for isolated nodes (Citeseer) [0.0, 3.0]
        a2_bounds = (1.0, 8.0)
        a8_bounds = (0.0, 1.0)
        a3_bounds = (0.0, 3.0)
        fallback_params = [0.1, 5.0, 1.0, 0.1, 0.1, 0.05, 0.5, 0.5]
        mode_str = "HOMO"

    if log_file:
        write_log(f"Graph H={h_val:.3f} (Base={baseline:.3f}) -> Mode={mode_str}", log_file, tag="TUNING")

    best_params = fallback_params 
    best_score = -float("inf")
    
    # K-Fold Plan
    n_splits = min(int(n_splits), len(base_indices))
    if n_splits < 2: n_splits = 2

    for s in range(num_samples):
        # Sample parameters respecting the domain
        params = [
            rng.uniform(0.0, 0.5),           # a1: Prior
            rng.uniform(*a2_bounds),         # a2: Neighbor
            rng.uniform(*a3_bounds),         # a3: Similarity
            rng.uniform(0.0, 0.5),           # a4: Degree
            rng.uniform(0.0, 0.3),           # a5: Count
            rng.uniform(0.0, 0.1),           # a6: Tie
            rng.uniform(0.4, 0.9),           # a7: Centroid
            rng.uniform(*a8_bounds),         # a8: Compat
        ]

        scores = []
        for r in range(repeats):
            rng_r = np.random.default_rng(seed + s*100 + r)
            order, bounds = _kfold_plan(base_indices, n_splits, True, rng_r)
            for k in range(n_splits):
                tr, va = _kfold_train_val_from_plan(order, bounds, k)
                try:
                    _, sc = model(data, tr, va, *params, **model_kwargs)
                    if np.isfinite(sc): scores.append(sc)
                except Exception:
                    pass 

        if not scores: continue
        agg = np.mean(scores)
        if agg > best_score:
            best_score = agg
            best_params = params
            
    if best_score == -float("inf") and log_file:
        write_log("All tuning trials failed. Using fallback.", log_file, tag="TUNING_WARN")
    
    return best_params, best_score, {}

# =====================================================
# BLOCK 3: PredictClass (Updated with Weighted Centroids)
# =====================================================

def _update_D_weighted(D_matrix, c_insts, all_lbls, dev, conf_vec, x_z):
    with torch.no_grad():
        conf_t = torch.from_numpy(conf_vec).to(dev).float()
        for c_idx, lbl in enumerate(all_lbls):
            if not c_insts[lbl]:
                continue
            idx_t = torch.tensor(list(c_insts[lbl]), device=dev)
            w = conf_t[idx_t]
            w_sum = w.sum()
            if w_sum < 1e-9:
                continue
            cent = (x_z[idx_t] * w[:, None]).sum(0, keepdim=True) / w_sum
            D_matrix[:, c_idx] = ((F.cosine_similarity(x_z, cent) + 1) / 2).cpu().numpy()


def predictclass(
    data, train_indices, test_indices,
    a1, a2, a3, a4, a5, a6, a7, a8,
    seed: int = 1337,
    log_prefix: str = "predictclass",
    log_every: int | None = None,
    log_file: str | None = None,
    mlp_probs: Optional[torch.Tensor] = None, 
    min_labeled_neighbors_for_full_weight: int = 2, 
    return_details: bool = False,
    **kwargs
):
    t_all = Timer()
    device = data.x.device
    N = int(data.num_nodes)
    
    train_indices = np.asarray(train_indices, dtype=np.int64)
    test_indices  = np.asarray(test_indices,  dtype=np.int64)
    y_true = data.y.detach().cpu().numpy().astype(np.int64)
    
    # Robust Class Count Calculation
    # Determine C from MLP probs if available (dense), otherwise from labels
    C_data = int(y_true.max()) + 1
    if mlp_probs is not None:
        C_mlp = mlp_probs.shape[1]
        C = max(C_data, C_mlp)
    else:
        C = C_data
        
    lbl2i = {i: i for i in range(C)}
    lbl_arr = np.arange(C, dtype=np.int64)
    
    mlp_probs_np = mlp_probs.detach().cpu().numpy() if mlp_probs is not None else None

    # Graph
    edge_index = torch.cat([data.edge_index, data.edge_index.flip(0)], dim=1)
    edge_index = torch.unique(edge_index, dim=1)
    ei = edge_index.detach().cpu().numpy()
    src, dst = ei[0], ei[1]
    
    neighbors = [[] for _ in range(N)]
    for u, v in zip(src, dst): neighbors[u].append(v)
    deg = np.array([len(n) for n in neighbors], dtype=np.int32)

    # Init State
    y_pred = np.full(N, -1, dtype=np.int64)
    y_pred[train_indices] = y_true[train_indices]
    
    train_mask_t = torch.zeros(N, device=device, dtype=torch.bool)
    train_mask_t[train_indices] = True
    
    # Feature Stats
    with torch.no_grad():
        x_tr = data.x[train_mask_t]
        if x_tr.size(0) > 1:
            mu, sigma = x_tr.mean(0, keepdim=True), x_tr.std(0, keepdim=True).clamp_min(1e-6)
            xz = (data.x - mu) / sigma
        else:
            xz = data.x
        
    class_insts = defaultdict(set)
    for i in train_indices: class_insts[y_pred[i]].add(i)

    # Matrices
    counts = np.bincount([lbl2i[y_pred[i]] for i in train_indices], minlength=C)
    prior = (counts + 1.0) / (len(train_indices) + C)
    
    compat_cts = np.ones((C, C)) * 2.0 
    tr_src, tr_dst = src[np.isin(src, train_indices)], dst[np.isin(src, train_indices)]
    mask = np.isin(tr_dst, train_indices)
    if mask.any():
        u, v = tr_src[mask], tr_dst[mask]
        np.add.at(compat_cts, ([lbl2i[y_pred[i]] for i in v], [lbl2i[y_pred[i]] for i in u]), 1)
    compat = compat_cts / compat_cts.sum(axis=0, keepdims=True)

    # --- SOFT INITIALIZATION & CONFIDENCE TRACKING ---
    nbr_dist = np.zeros((N, C), dtype=np.float64)
    nbr_conf_sum = np.zeros(N, dtype=np.float64)
    
    node_state_vec = np.zeros((N, C), dtype=np.float64)
    node_state_conf = np.zeros(N, dtype=np.float64)
    
    for u in range(N):
        if y_pred[u] != -1:
            # Training node: Hard label, Certainty = 1.0
            lbl_idx = lbl2i[y_pred[u]]
            node_state_vec[u, lbl_idx] = 1.0
            node_state_conf[u] = 1.0
        elif mlp_probs_np is not None:
            # Unlabeled node with MLP: Soft label, Certainty = max probability
            node_state_vec[u] = mlp_probs_np[u]
            node_state_conf[u] = float(mlp_probs_np[u].max())
        else:
            # Unlabeled without MLP: No contribution yet
            pass
            
    # Scatter to neighbors
    for u in range(N):
        if node_state_conf[u] > 0:
            for v in neighbors[u]:
                nbr_dist[v] += node_state_vec[u]
                nbr_conf_sum[v] += node_state_conf[u]

    # Sim Matrix D
    D = np.zeros((N, C))
    _update_D_weighted(D, class_insts, lbl_arr, device, node_state_conf, xz)

    # Optional node-level details for downstream gating/analysis.
    if return_details:
        prop_full_scores = np.zeros((N, C), dtype=np.float64)
        prop_pred_all = np.full(N, -1, dtype=np.int64)
        prop_top1 = np.zeros(N, dtype=np.float64)
        prop_top2 = np.zeros(N, dtype=np.float64)
        prop_margin = np.zeros(N, dtype=np.float64)
        support_labeled_neighbors = np.zeros(N, dtype=np.int32)
        neighbor_class_entropy = np.zeros(N, dtype=np.float64)
        similarity_gap = np.zeros(N, dtype=np.float64)
        compatibility_gap = np.zeros(N, dtype=np.float64)
        deferred_count = np.zeros(N, dtype=np.int32)  # not used in vanilla predictclass (v1)
        reconsidered_in_refinement = np.zeros(N, dtype=np.int32)  # not used in vanilla predictclass (v1)

        for i in train_indices:
            ii = int(i)
            cls = int(y_pred[ii])
            prop_pred_all[ii] = cls
            prop_full_scores[ii, cls] = 1.0
            prop_top1[ii] = 1.0
            prop_top2[ii] = 0.0
            prop_margin[ii] = 1.0

    # Logging: Feature Similarity Stats
    if log_file:
        avg_max_sim = D.max(axis=1).mean()
        write_log(f"Avg Max Feature Similarity: {avg_max_sim:.4f}", log_file, tag=f"{log_prefix}:FEATS")

    # Priority Queue
    pq = []
    def calc_prio(u):
        w = 0.0
        if deg[u] > 0:
            # Rank by average neighbor certainty
            w = nbr_conf_sum[u] / float(deg[u])
        
        # Add Feature Similarity to Priority
        # User Req: "the more the feature vector... near the nearest centroid... more sure"
        s = D[u].max() 
        
        # Add MLP confidence if available
        if mlp_probs_np is not None: s += mlp_probs_np[u].max()
        
        return -(w*a4 + s), u

    for u in range(N):
        if y_pred[u] == -1: heapq.heappush(pq, calc_prio(u))

    labeled_new = 0
    debug_stats = defaultdict(list)
    
    while pq:
        _, u = heapq.heappop(pq)
        if y_pred[u] != -1: continue
        
        # 1. Prior
        s_prior = a1 * prior
        
        # 2. Neighbors (Soft Distribution)
        if deg[u] > 0:
            dist_vec = nbr_dist[u] / float(deg[u])
            r = min(nbr_conf_sum[u] / float(max(min_labeled_neighbors_for_full_weight, 1)), 1.0)
            s_neigh = (a2 * r) * dist_vec
            support_count_u = int(sum(1 for v in neighbors[u] if y_pred[v] != -1))
            dist_mass = dist_vec.sum()
            if dist_mass > 1e-12:
                prob_vec = dist_vec / dist_mass
                nz = prob_vec > 1e-12
                entropy_u = float(-(prob_vec[nz] * np.log(prob_vec[nz])).sum())
            else:
                entropy_u = 0.0
        else:
            s_neigh = np.zeros(C)
            r = 0.0
            support_count_u = 0
            entropy_u = 0.0
            
        # Compatibility (Vectorized Soft)
        total_mass = nbr_dist[u].sum()
        if total_mass > 1e-6:
            norm_dist = nbr_dist[u] / total_mass
            compat_raw = norm_dist @ compat
            s_compat = (a8 * r) * compat_raw
        else:
            compat_raw = np.full(C, 1.0 / C)
            s_compat = np.full(C, (a8 * r) / C)
            
        # 3. Similarity (Feature Vector to Weighted Centroid)
        # User Req: "predict class also should have one more parameter" -> a3 handles this
        s_sim = a3 * D[u]

        final_scores = s_prior + s_neigh + s_sim + s_compat
        
        best = np.argmax(final_scores)
        best_score = float(final_scores[best])
        tmp_scores = final_scores.copy()
        tmp_scores[best] = -1e100
        second_score = float(tmp_scores.max())
        lbl = lbl_arr[best]
        y_pred[u] = lbl
        class_insts[lbl].add(u)
        labeled_new += 1

        if return_details:
            sim_raw = D[u]
            sim_best = int(np.argmax(sim_raw))
            sim_tmp = sim_raw.copy()
            sim_tmp[sim_best] = -1e100
            sim_second = float(sim_tmp.max())
            compat_best = int(np.argmax(compat_raw))
            compat_tmp = compat_raw.copy()
            compat_tmp[compat_best] = -1e100
            compat_second = float(compat_tmp.max())

            prop_pred_all[u] = int(lbl)
            prop_full_scores[u] = final_scores
            prop_top1[u] = best_score
            prop_top2[u] = second_score
            prop_margin[u] = best_score - second_score
            support_labeled_neighbors[u] = support_count_u
            neighbor_class_entropy[u] = entropy_u
            similarity_gap[u] = float(sim_raw[sim_best] - sim_second)
            compatibility_gap[u] = float(compat_raw[compat_best] - compat_second)
        
        # --- DYNAMIC UPDATE ---
        # Update node u's state to Hard Label with Confidence a7 (Propagation Confidence)
        old_vec = node_state_vec[u]
        old_conf = node_state_conf[u]
        
        new_vec = np.zeros(C, dtype=np.float64)
        new_vec[lbl2i[lbl]] = 1.0
        # User Req: "The more we are sure... more is the weight"
        # We use a7 as the confidence/weight for propagated nodes
        new_conf = float(a7) 
        
        # Update global state vectors for centroid calculation later
        node_state_vec[u] = new_vec
        node_state_conf[u] = new_conf
        
        delta_vec = new_vec - old_vec
        delta_conf = new_conf - old_conf
        
        # Propagate changes to neighbors
        for v in neighbors[u]:
            nbr_dist[v] += delta_vec
            nbr_conf_sum[v] += delta_conf
            
        if log_file and labeled_new <= 5:
            debug_stats['prior'].append(abs(s_prior[best]))
            debug_stats['neigh'].append(abs(s_neigh[best]))
            debug_stats['sim'].append(abs(s_sim[best]))
            debug_stats['compat'].append(abs(s_compat[best]))

        if labeled_new % max(N//5, 1) == 0:
            # Recompute weighted centroids using updated node_state_conf
            _update_D_weighted(D, class_insts, lbl_arr, device, node_state_conf, xz)
            pq = [] 
            for x in range(N):
                if y_pred[x] == -1: heapq.heappush(pq, calc_prio(x))
    
    if log_file:
        p = np.mean(debug_stats['prior']) if debug_stats['prior'] else 0
        n = np.mean(debug_stats['neigh']) if debug_stats['neigh'] else 0
        s = np.mean(debug_stats['sim']) if debug_stats['sim'] else 0
        c = np.mean(debug_stats['compat']) if debug_stats['compat'] else 0
        write_log(f"Avg Term Magnitudes (First 5): Prior={p:.3f}, Neigh={n:.3f}, Sim={s:.3f}, Compat={c:.3f}", log_file, tag=f"{log_prefix}:DEBUG")

    test_acc = _simple_accuracy(y_true[test_indices], y_pred[test_indices])
    preds_test_t = torch.tensor(y_pred[test_indices], device=device)
    if return_details:
        details = {
            "prop_pred_all": prop_pred_all,
            "prop_full_scores": prop_full_scores,
            "prop_top1_score": prop_top1,
            "prop_top2_score": prop_top2,
            "prop_margin": prop_margin,
            "degree": deg.copy(),
            "support_labeled_neighbors": support_labeled_neighbors,
            "neighbor_class_entropy": neighbor_class_entropy,
            "similarity_gap": similarity_gap,
            "compatibility_gap": compatibility_gap,
            "deferred_count": deferred_count,
            "reconsidered_in_refinement": reconsidered_in_refinement,
        }
        return preds_test_t, test_acc, details
    return preds_test_t, test_acc


def priority_bfs_predictclass(
    data,
    train_indices,
    test_indices,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7,
    a8,
    *,
    seed: int = 1337,
    log_prefix: str = "priority_bfs",
    log_file: str | None = None,
    enable_prior: bool = True,
    enable_neighbor: bool = True,
    enable_compat: bool = True,
    enable_similarity: bool = True,
    enable_path_reliability: bool = False,
    enable_deferral: bool = True,
    margin_threshold: float = 0.05,
    refinement_margin_threshold: float = 0.02,
    max_deferrals_per_node: int = 5,
    record_diagnostics: bool = True,
    diagnostics_only_test_nodes: bool = True,
    diagnostics_run_id: str | None = None,
    min_labeled_neighbors_for_full_weight: int = 2,
    refresh_fraction: float = 0.2,
):
    """
    Priority-frontier BFS-style propagation with interpretable scoring, deferral, and refinement.

    Component semantics are kept consistent with the existing `predictclass`:
      - a1: prior term,
      - a2: neighbor support,
      - a3: feature similarity support,
      - a8: compatibility support.

    The score formula is:
        final_scores = a1 * prior_component
                     + a2 * neighbor_component
                     + a3 * similarity_component
                     + a8 * compatibility_component
    with inactive components forced to zero.

    Path reliability is present only as a placeholder (currently zero).
    """
    rng = np.random.default_rng(seed)
    device = data.x.device
    N = int(data.num_nodes)

    train_indices = np.asarray(train_indices, dtype=np.int64)
    test_indices = np.asarray(test_indices, dtype=np.int64)
    y_true = data.y.detach().cpu().numpy().astype(np.int64)

    C = int(y_true.max()) + 1
    lbl2i = {i: i for i in range(C)}
    lbl_arr = np.arange(C, dtype=np.int64)

    # Graph
    edge_index = torch.cat([data.edge_index, data.edge_index.flip(0)], dim=1)
    edge_index = torch.unique(edge_index, dim=1)
    ei = edge_index.detach().cpu().numpy()
    src, dst = ei[0], ei[1]

    neighbors = [[] for _ in range(N)]
    for u, v in zip(src, dst):
        neighbors[u].append(v)
    deg = np.array([len(nbrs) for nbrs in neighbors], dtype=np.int32)

    # Initial labels
    y_pred = np.full(N, -1, dtype=np.int64)
    y_pred[train_indices] = y_true[train_indices]

    train_mask_t = torch.zeros(N, device=device, dtype=torch.bool)
    train_mask_t[train_indices] = True

    # Feature standardization
    with torch.no_grad():
        x_tr = data.x[train_mask_t]
        if x_tr.size(0) > 1:
            mu, sigma = x_tr.mean(0, keepdim=True), x_tr.std(0, keepdim=True).clamp_min(1e-6)
            xz = (data.x - mu) / sigma
        else:
            xz = data.x

    class_insts = defaultdict(set)
    for i in train_indices:
        class_insts[y_pred[i]].add(int(i))

    # Priors
    counts = np.bincount([lbl2i[y_pred[i]] for i in train_indices], minlength=C)
    prior = (counts + 1.0) / (len(train_indices) + C)

    # Compatibility from train-train edges
    compat_cts = np.ones((C, C)) * 2.0
    tr_src = src[np.isin(src, train_indices)]
    tr_dst = dst[np.isin(src, train_indices)]
    mask = np.isin(tr_dst, train_indices)
    if mask.any():
        u_tr, v_tr = tr_src[mask], tr_dst[mask]
        np.add.at(
            compat_cts,
            ([lbl2i[y_pred[i]] for i in v_tr], [lbl2i[y_pred[i]] for i in u_tr]),
            1,
        )
    compat = compat_cts / compat_cts.sum(axis=0, keepdims=True)

    # Soft neighbor evidence via node state vectors
    nbr_dist = np.zeros((N, C), dtype=np.float64)
    nbr_conf_sum = np.zeros(N, dtype=np.float64)

    node_state_vec = np.zeros((N, C), dtype=np.float64)
    node_state_conf = np.zeros(N, dtype=np.float64)

    for u in range(N):
        if y_pred[u] != -1:
            lbl_idx = lbl2i[y_pred[u]]
            node_state_vec[u, lbl_idx] = 1.0
            node_state_conf[u] = 1.0

    for u in range(N):
        if node_state_conf[u] > 0:
            for v in neighbors[u]:
                nbr_dist[v] += node_state_vec[u]
                nbr_conf_sum[v] += node_state_conf[u]

    # Feature similarity matrix
    D = np.zeros((N, C), dtype=np.float64)
    _update_D_weighted(D, class_insts, lbl_arr, device, node_state_conf, xz)

    # Diagnostics
    record_mask = np.zeros(N, dtype=bool)
    if diagnostics_only_test_nodes:
        record_mask[test_indices] = True
    else:
        record_mask[:] = True
    diagnostics = {}
    commit_margin = np.zeros(N, dtype=np.float64)
    first_pass_label = np.full(N, -1, dtype=np.int64)
    active_components = [
        name
        for name, enabled in [
            ("prior", enable_prior),
            ("neighbor", enable_neighbor),
            ("compat", enable_compat),
            ("similarity", enable_similarity),
        ]
        if enabled
    ]

    refresh_every = max(int(N * float(refresh_fraction)), 1)

    # Initial frontier priority (before margins are known)
    def initial_priority(u: int) -> float:
        if deg[u] == 0:
            return 0.0
        w = 0.0
        if nbr_conf_sum[u] > 0:
            w = nbr_conf_sum[u] / float(deg[u])
        s = D[u].max()
        return -(a4 * w + s)

    pq = []
    deferrals = np.zeros(N, dtype=np.int32)

    for u in range(N):
        if y_pred[u] == -1:
            heapq.heappush(pq, (initial_priority(u), u))

    labeled_new = 0
    deferred_any = np.zeros(N, dtype=bool)
    forced_commits = 0

    while pq:
        prio, u = heapq.heappop(pq)
        if y_pred[u] != -1:
            continue

        # Components
        if enable_prior:
            s_prior = a1 * prior
        else:
            s_prior = np.zeros(C, dtype=np.float64)

        if enable_neighbor and deg[u] > 0:
            dist_vec = nbr_dist[u] / float(deg[u])
            r = min(
                nbr_conf_sum[u] / float(max(min_labeled_neighbors_for_full_weight, 1)),
                1.0,
            )
            s_neigh = (a2 * r) * dist_vec
        else:
            s_neigh = np.zeros(C, dtype=np.float64)

        total_mass = nbr_dist[u].sum()
        if enable_compat:
            if total_mass > 1e-6:
                norm_dist = nbr_dist[u] / total_mass
                s_compat = (a8 * 1.0) * (norm_dist @ compat)
            else:
                s_compat = np.full(C, (a8 * 1.0) / C, dtype=np.float64)
        else:
            s_compat = np.zeros(C, dtype=np.float64)

        if enable_similarity:
            s_sim = a3 * D[u]
        else:
            s_sim = np.zeros(C, dtype=np.float64)

        # Path reliability placeholder (currently zero)
        if enable_path_reliability:
            s_path = np.zeros(C, dtype=np.float64)
        else:
            s_path = np.zeros(C, dtype=np.float64)

        final_scores = s_prior + s_neigh + s_compat + s_sim + s_path

        best_idx = int(np.argmax(final_scores))
        best_score = float(final_scores[best_idx])
        tmp = final_scores.copy()
        tmp[best_idx] = -1e100
        second_score = float(tmp.max())
        margin = best_score - second_score

        # Deferral for low-confidence nodes
        if enable_deferral and margin < margin_threshold:
            if deferrals[u] < max_deferrals_per_node:
                deferrals[u] += 1
                deferred_any[u] = True
                heapq.heappush(pq, (-margin, u))
                continue
            else:
                forced_commits += 1  # fall through to forced commit with current scores

        chosen_lbl = int(lbl_arr[best_idx])
        y_pred[u] = chosen_lbl
        class_insts[chosen_lbl].add(int(u))
        labeled_new += 1
        commit_margin[u] = margin
        first_pass_label[u] = chosen_lbl

        # Update neighbor soft counts
        old_vec = node_state_vec[u]
        old_conf = node_state_conf[u]

        new_vec = np.zeros(C, dtype=np.float64)
        new_vec[best_idx] = 1.0
        new_conf = float(a7)

        node_state_vec[u] = new_vec
        node_state_conf[u] = new_conf

        delta_vec = new_vec - old_vec
        delta_conf = new_conf - old_conf

        for v in neighbors[u]:
            nbr_dist[v] += delta_vec
            nbr_conf_sum[v] += delta_conf

        # Diagnostics at commit time
        if record_diagnostics and record_mask[u]:
            diagnostics[int(u)] = {
                "node_id": int(u),
                "pred_label": int(chosen_lbl),
                "top_score": float(best_score),
                "second_score": float(second_score),
                "margin": float(margin),
                "frontier_priority_at_commit": float(prio),
                "first_pass_label": int(chosen_lbl),
                "first_pass_margin": float(margin),
                "final_label_after_refinement": int(chosen_lbl),  # may be updated later
                "changed_in_refinement": False,
                "active_components": active_components,
                "components": {
                    "prior": float(s_prior[best_idx]) if enable_prior else 0.0,
                    "neighbor": float(s_neigh[best_idx]) if enable_neighbor else 0.0,
                    "compat": float(s_compat[best_idx]) if enable_compat else 0.0,
                    "similarity": float(s_sim[best_idx]) if enable_similarity else 0.0,
                    "path_reliability": 0.0,
                },
            }

        # Periodic refresh of similarity and frontier
        if labeled_new % refresh_every == 0:
            _update_D_weighted(D, class_insts, lbl_arr, device, node_state_conf, xz)
            pq = []
            for x in range(N):
                if y_pred[x] == -1:
                    heapq.heappush(pq, (initial_priority(x), x))

    # Refinement: relabel only low-margin, non-train nodes in a single static pass
    low_conf_mask = (commit_margin < refinement_margin_threshold)
    for idx in train_indices:
        low_conf_mask[int(idx)] = False

    reconsidered_refinement = int(low_conf_mask.sum())
    changed_refinement = 0

    # Neighbor support based on first-pass labels
    nbr_dist_ref = np.zeros((N, C), dtype=np.float64)
    for u in range(N):
        if y_pred[u] != -1:
            vec = np.zeros(C, dtype=np.float64)
            vec[lbl2i[y_pred[u]]] = 1.0
            for v in neighbors[u]:
                nbr_dist_ref[v] += vec

    for u in range(N):
        if not low_conf_mask[u]:
            continue

        if enable_prior:
            s_prior = a1 * prior
        else:
            s_prior = np.zeros(C, dtype=np.float64)

        if enable_neighbor and deg[u] > 0:
            dist_vec = nbr_dist_ref[u] / float(deg[u])
            s_neigh = a2 * dist_vec
        else:
            s_neigh = np.zeros(C, dtype=np.float64)

        total_mass = nbr_dist_ref[u].sum()
        if enable_compat:
            if total_mass > 1e-6:
                norm_dist = nbr_dist_ref[u] / total_mass
                s_compat = (a8 * 1.0) * (norm_dist @ compat)
            else:
                s_compat = np.full(C, (a8 * 1.0) / C, dtype=np.float64)
        else:
            s_compat = np.zeros(C, dtype=np.float64)

        if enable_similarity:
            s_sim = a3 * D[u]
        else:
            s_sim = np.zeros(C, dtype=np.float64)

        final_scores = s_prior + s_neigh + s_compat + s_sim
        best_idx = int(np.argmax(final_scores))
        new_lbl = int(lbl_arr[best_idx])

        if new_lbl != y_pred[u]:
            changed_refinement += 1
            y_pred[u] = new_lbl

        if record_diagnostics and record_mask[u]:
            entry = diagnostics.get(int(u))
            if entry is None:
                entry = {
                    "node_id": int(u),
                    "pred_label": int(new_lbl),
                    "top_score": float(final_scores[best_idx]),
                    "second_score": float(
                        np.max(np.where(
                            np.arange(C) != best_idx,
                            final_scores,
                            -1e100
                        ))
                    ),
                    "margin": float(commit_margin[u]),
                    "frontier_priority_at_commit": None,
                    "first_pass_label": int(first_pass_label[u]),
                    "first_pass_margin": float(commit_margin[u]),
                    "final_label_after_refinement": int(new_lbl),
                    "changed_in_refinement": new_lbl != first_pass_label[u],
                    "active_components": active_components,
                    "components": {},
                }
            else:
                entry["final_label_after_refinement"] = int(new_lbl)
                entry["changed_in_refinement"] = bool(new_lbl != first_pass_label[u])
            diagnostics[int(u)] = entry

    # Diagnostics JSON
    if record_diagnostics and diagnostics:
        try:
            run_id = diagnostics_run_id or f"{log_prefix}_{int(time.time())}"
            if "LOG_DIR" in globals():
                diag_path = os.path.join(
                    LOG_DIR, f"{DATASET_KEY.lower()}_priority_bfs_{run_id}.json"
                )
            else:
                diag_path = f"priority_bfs_{run_id}.json"
            with open(diag_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "dataset": DATASET_KEY,
                        "seed": seed,
                        "num_nodes": N,
                        "num_classes": C,
                        "summary": {
                            "committed_nodes": int((y_pred != -1).sum()),
                            "deferred_at_least_once": int(deferred_any.sum()),
                            "forced_commits_after_max_deferrals": int(forced_commits),
                            "refinement_reconsidered": int(reconsidered_refinement),
                            "refinement_label_changes": int(changed_refinement),
                            "active_components": active_components,
                        },
                        "diagnostics": diagnostics,
                    },
                    f,
                )
            if log_file:
                write_log(
                    f"Priority-BFS diagnostics written to {diag_path}",
                    log_file,
                    tag=f"{log_prefix}:DIAG",
                )
        except Exception as e:
            if log_file:
                write_log(
                    f"Failed to write diagnostics JSON: {e}",
                    log_file,
                    tag=f"{log_prefix}:DIAG_ERR",
                )

    test_acc = _simple_accuracy(y_true[test_indices], y_pred[test_indices])
    return torch.tensor(y_pred[test_indices], device=device), test_acc


def multi_source_priority_bfs_predictclass(
    data,
    train_indices,
    test_indices,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7,
    a8,
    *,
    seed: int = 1337,
    log_prefix: str = "multi_source_bfs",
    log_file: str | None = None,
    seed_mode: str = "all_train",
    seeds_per_class: int | None = None,
    enable_deferral: bool = True,
    margin_threshold: float = 0.05,
    refinement_margin_threshold: float = 0.02,
    max_deferrals_per_node: int = 5,
    record_diagnostics: bool = True,
    diagnostics_only_test_nodes: bool = True,
):
    """
    Seed-configurable wrapper around priority_bfs_predictclass.

    v1:
      - seed_mode == "all_train" uses all train_indices as seeds.
      - Other modes are reserved for future extensions.
    """
    rng = np.random.default_rng(seed)

    train_indices = np.asarray(train_indices, dtype=np.int64)

    # Seed selection hook (v1: all_train only).
    if seed_mode == "all_train":
        seed_indices = train_indices
    else:
        # Future: topk_per_class, random_subset_per_class, etc.
        seed_indices = train_indices

    return priority_bfs_predictclass(
        data,
        seed_indices,
        test_indices,
        a1,
        a2,
        a3,
        a4,
        a5,
        a6,
        a7,
        a8,
        seed=seed,
        log_prefix=log_prefix,
        log_file=log_file,
        enable_prior=True,
        enable_neighbor=True,
        enable_compat=True,
        enable_similarity=True,
        enable_path_reliability=False,
        enable_deferral=enable_deferral,
        margin_threshold=margin_threshold,
        refinement_margin_threshold=refinement_margin_threshold,
        max_deferrals_per_node=max_deferrals_per_node,
        record_diagnostics=record_diagnostics,
        diagnostics_only_test_nodes=diagnostics_only_test_nodes,
    )


def ensemble_multi_source_bfs_predictclass(
    data,
    train_indices,
    test_indices,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7,
    a8,
    *,
    num_restarts: int = 5,
    base_seed: int = 1337,
    seed_mode: str = "all_train",
    seeds_per_class: int | None = None,
    margin_threshold: float = 0.05,
    refinement_margin_threshold: float = 0.02,
    max_deferrals_per_node: int = 5,
    log_prefix: str = "ensemble_multi_source_bfs",
    log_file: str | None = None,
):
    """
    Ensemble wrapper over multi_source_priority_bfs_predictclass.

    v1:
      - Runs multi_source_priority_bfs_predictclass num_restarts times.
      - Aggregates labels on test_indices via majority vote.
    """
    device = data.x.device
    train_indices = np.asarray(train_indices, dtype=np.int64)
    test_indices = np.asarray(test_indices, dtype=np.int64)

    all_preds = []
    y_true = data.y.detach().cpu().numpy().astype(np.int64)

    for r in range(num_restarts):
        seed = base_seed + r
        preds_test, _ = multi_source_priority_bfs_predictclass(
            data,
            train_indices,
            test_indices,
            a1,
            a2,
            a3,
            a4,
            a5,
            a6,
            a7,
            a8,
            seed=seed,
            log_prefix=f"{log_prefix}_run{r}",
            log_file=log_file,
            seed_mode=seed_mode,
            seeds_per_class=seeds_per_class,
            enable_deferral=True,
            margin_threshold=margin_threshold,
            refinement_margin_threshold=refinement_margin_threshold,
            max_deferrals_per_node=max_deferrals_per_node,
            record_diagnostics=False,
        )
        all_preds.append(preds_test.detach().cpu().numpy().astype(np.int64))

    if not all_preds:
        return torch.tensor(y_true[test_indices], device=device), 0.0

    preds_stack = np.stack(all_preds, axis=0)  # [R, T]
    T = preds_stack.shape[1]
    final_preds = np.zeros(T, dtype=np.int64)

    for j in range(T):
        labels, counts = np.unique(preds_stack[:, j], return_counts=True)
        best_lbl = labels[np.argmax(counts)]
        final_preds[j] = best_lbl

    test_acc = _simple_accuracy(y_true[test_indices], final_preds)

    if log_file:
        write_log(
            f"Ensemble multi-source BFS: num_restarts={num_restarts} | test_acc={test_acc:.4f}",
            log_file,
            tag=f"{log_prefix}:SUMMARY",
        )

    return torch.tensor(final_preds, device=device), float(test_acc)

# =====================================================
# BLOCK 4: MLP Model
# =====================================================

def _to_long_tensor(idx, device):
    return torch.as_tensor(idx, dtype=torch.long, device=device)

def _to_numpy_idx(idx):
    if isinstance(idx, torch.Tensor): return idx.cpu().numpy()
    return idx

class SimpleMLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super(SimpleMLP, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(nn.Linear(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.convs.append(nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index=None):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x)
        return x

def train_mlp_and_predict(data, train_indices, hidden=64, layers=2, dropout=0.5, lr=0.01, epochs=200, log_file=None):
    device = data.x.device
    input_dim = data.x.shape[1]
    num_classes = int(data.y.max().item()) + 1
    
    model = SimpleMLP(input_dim, hidden, num_classes, layers, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()
    
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    train_mask[train_indices] = True
    
    model.train()
    if log_file:
        write_log(f"    [MLP] Training for {epochs} epochs...", log_file)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x)
        loss = loss_fn(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        
        if log_file and (epoch + 1) % 50 == 0:
            write_log(f"      Epoch {epoch+1:03d} | Loss: {loss.item():.4f}", log_file)
        
    model.eval()
    with torch.no_grad():
        logits = model(data.x)
        probs = F.softmax(logits, dim=1)
        
    return probs, model


def _top1_top2_margin(scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (pred, top1, top2, margin) row-wise for score matrix [N, C].
    """
    scores = np.asarray(scores, dtype=np.float64)
    pred = np.argmax(scores, axis=1).astype(np.int64)
    top1 = scores[np.arange(scores.shape[0]), pred]
    tmp = scores.copy()
    tmp[np.arange(scores.shape[0]), pred] = -1e100
    top2 = tmp.max(axis=1)
    margin = top1 - top2
    return pred, top1, top2, margin


def _safe_sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def compute_mlp_margin(mlp_probs: torch.Tensor) -> Dict[str, np.ndarray]:
    """
    Compute MLP predictions, confidence, and margin from probabilities.
    """
    probs_np = mlp_probs.detach().cpu().numpy().astype(np.float64)
    pred, top1, top2, margin = _top1_top2_margin(probs_np)
    return {
        "mlp_probs_np": probs_np,
        "mlp_pred_all": pred,
        "mlp_top1_all": top1,
        "mlp_top2_all": top2,
        "mlp_margin_all": margin,
    }


def _build_selective_correction_evidence(
    data,
    train_indices: np.ndarray,
    *,
    mlp_probs_np: np.ndarray,
    min_labeled_neighbors_for_full_weight: int = 2,
    enable_feature_knn: bool = False,
    feature_knn_k: int = 5,
) -> Dict[str, np.ndarray]:
    """
    Build class-wise evidence matrices for feature-first selective graph correction.
    """
    device = data.x.device
    y_true = data.y.detach().cpu().numpy().astype(np.int64)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    N = int(data.num_nodes)
    C = int(max(int(y_true.max()) + 1, mlp_probs_np.shape[1]))

    # Graph (same preprocessing style as predictclass)
    edge_index = torch.cat([data.edge_index, data.edge_index.flip(0)], dim=1)
    edge_index = torch.unique(edge_index, dim=1)
    ei = edge_index.detach().cpu().numpy()
    src, dst = ei[0], ei[1]

    neighbors = [[] for _ in range(N)]
    for u, v in zip(src, dst):
        neighbors[int(u)].append(int(v))
    deg = np.array([len(nbrs) for nbrs in neighbors], dtype=np.int32)

    train_mask_np = np.zeros(N, dtype=bool)
    train_mask_np[train_indices] = True

    # Compatibility from train-train edges
    compat_cts = np.ones((C, C), dtype=np.float64) * 2.0
    tr_src = src[train_mask_np[src]]
    tr_dst = dst[train_mask_np[src]]
    mask_tt = train_mask_np[tr_dst]
    if mask_tt.any():
        u_tt, v_tt = tr_src[mask_tt], tr_dst[mask_tt]
        np.add.at(compat_cts, (y_true[v_tt], y_true[u_tt]), 1.0)
    compat = compat_cts / compat_cts.sum(axis=0, keepdims=True)

    # Feature standardization (train stats)
    train_mask_t = torch.zeros(N, device=device, dtype=torch.bool)
    train_mask_t[train_indices] = True
    with torch.no_grad():
        x_tr = data.x[train_mask_t]
        if x_tr.size(0) > 1:
            mu = x_tr.mean(0, keepdim=True)
            sigma = x_tr.std(0, keepdim=True).clamp_min(1e-6)
            xz = (data.x - mu) / sigma
        else:
            xz = data.x

    # Soft node states: train nodes are hard labels, others use MLP probs.
    node_state_vec = np.zeros((N, C), dtype=np.float64)
    node_state_conf = np.zeros(N, dtype=np.float64)
    class_insts = defaultdict(set)
    for i in train_indices:
        ii = int(i)
        yi = int(y_true[ii])
        node_state_vec[ii, yi] = 1.0
        node_state_conf[ii] = 1.0
        class_insts[yi].add(ii)
    unlabeled_idx = np.where(~train_mask_np)[0]
    node_state_vec[unlabeled_idx] = mlp_probs_np[unlabeled_idx]
    node_state_conf[unlabeled_idx] = mlp_probs_np[unlabeled_idx].max(axis=1)

    # Neighbor evidence aggregation
    nbr_dist = np.zeros((N, C), dtype=np.float64)
    nbr_conf_sum = np.zeros(N, dtype=np.float64)
    support_train_neighbors = np.zeros(N, dtype=np.int32)

    for u in range(N):
        if node_state_conf[u] <= 0:
            continue
        vec_u = node_state_vec[u]
        conf_u = node_state_conf[u]
        for v in neighbors[u]:
            nbr_dist[v] += vec_u
            nbr_conf_sum[v] += conf_u
            if train_mask_np[u]:
                support_train_neighbors[v] += 1

    # Feature-prototype similarity matrix D
    D = np.zeros((N, C), dtype=np.float64)
    lbl_arr = np.arange(C, dtype=np.int64)
    _update_D_weighted(D, class_insts, lbl_arr, device, node_state_conf, xz)

    graph_neighbor_support = np.zeros((N, C), dtype=np.float64)
    compatibility_support = np.zeros((N, C), dtype=np.float64)
    neighbor_class_entropy = np.zeros(N, dtype=np.float64)

    for u in range(N):
        if deg[u] > 0:
            dist_vec = nbr_dist[u] / float(deg[u])
            dist_mass = dist_vec.sum()
            if dist_mass > 1e-12:
                neigh_prob = dist_vec / dist_mass
            else:
                neigh_prob = np.zeros(C, dtype=np.float64)
            graph_neighbor_support[u] = neigh_prob

            total_mass = nbr_dist[u].sum()
            if total_mass > 1e-12:
                norm_dist = nbr_dist[u] / total_mass
                compatibility_support[u] = norm_dist @ compat
            else:
                compatibility_support[u] = np.full(C, 1.0 / C, dtype=np.float64)

            nz = neigh_prob > 1e-12
            neighbor_class_entropy[u] = float(-(neigh_prob[nz] * np.log(neigh_prob[nz])).sum()) if nz.any() else 0.0
        else:
            graph_neighbor_support[u] = np.zeros(C, dtype=np.float64)
            compatibility_support[u] = np.full(C, 1.0 / C, dtype=np.float64)
            neighbor_class_entropy[u] = 0.0

    # Optional feature-space kNN vote (off by default for v1).
    feature_knn_vote = np.zeros((N, C), dtype=np.float64)
    if enable_feature_knn and feature_knn_k > 0 and train_indices.size > 0:
        x_np = xz.detach().cpu().numpy().astype(np.float64)
        x_train = x_np[train_indices]
        y_train = y_true[train_indices]
        k = min(int(feature_knn_k), int(train_indices.size))
        if k > 0:
            x_norm = np.linalg.norm(x_np, axis=1, keepdims=True) + 1e-12
            x_train_norm = np.linalg.norm(x_train, axis=1, keepdims=True) + 1e-12
            sim = (x_np @ x_train.T) / (x_norm * x_train_norm.T)
            top_idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
            top_sim = np.take_along_axis(sim, top_idx, axis=1)
            top_lbl = y_train[top_idx]
            top_w = np.clip(top_sim, 0.0, None) + 1e-6
            for c in range(C):
                mask_c = (top_lbl == c)
                feature_knn_vote[:, c] = (top_w * mask_c).sum(axis=1)
            row_sum = feature_knn_vote.sum(axis=1, keepdims=True)
            nz = row_sum[:, 0] > 1e-12
            feature_knn_vote[nz] /= row_sum[nz]

    return {
        "feature_similarity": D,
        "feature_knn_vote": feature_knn_vote,
        "graph_neighbor_support": graph_neighbor_support,
        "compatibility_support": compatibility_support,
        "node_degree": deg.astype(np.float64),
        "support_train_neighbors": support_train_neighbors.astype(np.float64),
        "neighbor_class_entropy": neighbor_class_entropy,
    }


def build_selective_correction_scores(
    mlp_probs_np: np.ndarray,
    evidence: Dict[str, np.ndarray],
    *,
    b1: float,
    b2: float,
    b3: float,
    b4: float,
    b5: float,
    eps: float = 1e-9,
) -> Dict[str, np.ndarray]:
    """
    Build interpretable class-wise weighted score components.
    """
    mlp_term = float(b1) * np.log(np.clip(mlp_probs_np, eps, 1.0))
    feature_similarity_term = float(b2) * evidence["feature_similarity"]
    feature_knn_term = float(b3) * evidence["feature_knn_vote"]
    graph_neighbor_term = float(b4) * evidence["graph_neighbor_support"]
    compatibility_term = float(b5) * evidence["compatibility_support"]
    combined_scores = (
        mlp_term
        + feature_similarity_term
        + feature_knn_term
        + graph_neighbor_term
        + compatibility_term
    )
    return {
        "combined_scores": combined_scores,
        "mlp_term": mlp_term,
        "feature_similarity_term": feature_similarity_term,
        "feature_knn_term": feature_knn_term,
        "graph_neighbor_term": graph_neighbor_term,
        "compatibility_term": compatibility_term,
    }


def choose_uncertainty_threshold(
    y_true: np.ndarray,
    val_indices: np.ndarray,
    mlp_pred_all: np.ndarray,
    mlp_margin_all: np.ndarray,
    combined_scores: np.ndarray,
    *,
    threshold_candidates: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Select global margin threshold on validation only.
    """
    val_indices = np.asarray(val_indices, dtype=np.int64)
    val_margins = mlp_margin_all[val_indices]

    if threshold_candidates is None:
        q = np.quantile(val_margins, [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        base = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30], dtype=np.float64)
        threshold_candidates = sorted(set(np.round(np.concatenate([q, base]), 6).tolist()))
    threshold_candidates = [float(t) for t in threshold_candidates if np.isfinite(t)]
    if not threshold_candidates:
        threshold_candidates = [0.2]

    best = None
    for t in threshold_candidates:
        uncertain_val = val_margins < t
        final_val = mlp_pred_all[val_indices].copy()
        if uncertain_val.any():
            final_val[uncertain_val] = np.argmax(combined_scores[val_indices][uncertain_val], axis=1).astype(np.int64)
        val_acc = _simple_accuracy(y_true[val_indices], final_val)
        changed_frac = float((final_val != mlp_pred_all[val_indices]).mean())
        uncertain_frac = float(uncertain_val.mean())
        key = (val_acc, -changed_frac, -uncertain_frac)
        if best is None or key > best["key"]:
            best = {
                "threshold_high": float(t),
                "val_acc": float(val_acc),
                "changed_fraction_val": changed_frac,
                "uncertain_fraction_val": uncertain_frac,
                "key": key,
            }

    assert best is not None
    best["threshold_candidates_evaluated"] = threshold_candidates
    return best


def summarize_selective_correction_behavior(
    y_true: np.ndarray,
    node_indices: np.ndarray,
    mlp_pred_all: np.ndarray,
    final_pred_all: np.ndarray,
    uncertain_mask_all: np.ndarray,
) -> Dict[str, Optional[float]]:
    """
    Summarize behavior on a node subset (validation/test).
    """
    idx = np.asarray(node_indices, dtype=np.int64)
    u = uncertain_mask_all[idx]
    c = ~u
    out: Dict[str, Optional[float]] = {
        "fraction_uncertain": float(u.mean()) if idx.size > 0 else 0.0,
        "fraction_changed_from_mlp": float((final_pred_all[idx] != mlp_pred_all[idx]).mean()) if idx.size > 0 else 0.0,
        "acc_all": float(_simple_accuracy(y_true[idx], final_pred_all[idx])) if idx.size > 0 else None,
        "acc_uncertain": None,
        "acc_confident": None,
    }
    if u.any():
        out["acc_uncertain"] = float(_simple_accuracy(y_true[idx][u], final_pred_all[idx][u]))
    if c.any():
        out["acc_confident"] = float(_simple_accuracy(y_true[idx][c], final_pred_all[idx][c]))
    return out


def selective_graph_correction_predictclass(
    data,
    train_indices,
    val_indices,
    test_indices,
    *,
    mlp_probs: Optional[torch.Tensor] = None,
    seed: int = 1337,
    log_prefix: str = "selective_graph_correction",
    log_file: str | None = None,
    diagnostics_run_id: str | None = None,
    dataset_key_for_logs: Optional[str] = None,
    mlp_runtime_sec: Optional[float] = None,
    threshold_candidates: Optional[List[float]] = None,
    weight_candidates: Optional[List[Dict[str, float]]] = None,
    enable_feature_knn: bool = False,
    feature_knn_k: int = 5,
    write_node_diagnostics: bool = True,
) -> Tuple[torch.Tensor, float, Dict[str, Any]]:
    """
    Feature-first selective correction:
      - keep MLP for confident nodes,
      - apply weighted graph/feature correction only for uncertain nodes.
    """
    t_all0 = time.perf_counter()
    device = data.x.device
    y_true = data.y.detach().cpu().numpy().astype(np.int64)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    val_indices = np.asarray(val_indices, dtype=np.int64)
    test_indices = np.asarray(test_indices, dtype=np.int64)

    # 1) MLP (primary classifier)
    mlp_train_runtime_sec = 0.0
    if mlp_probs is None:
        t_mlp0 = time.perf_counter()
        mlp_probs, _ = train_mlp_and_predict(
            data,
            train_indices,
            hidden=64,
            layers=2,
            dropout=0.5,
            lr=0.01,
            epochs=200,
            log_file=log_file,
        )
        mlp_train_runtime_sec = time.perf_counter() - t_mlp0

    mlp_info = compute_mlp_margin(mlp_probs)
    mlp_probs_np = mlp_info["mlp_probs_np"]
    mlp_pred_all = mlp_info["mlp_pred_all"]
    mlp_top1_all = mlp_info["mlp_top1_all"]
    mlp_margin_all = mlp_info["mlp_margin_all"]

    # 2) Build correction evidence once
    t_e0 = time.perf_counter()
    evidence = _build_selective_correction_evidence(
        data,
        train_indices,
        mlp_probs_np=mlp_probs_np,
        enable_feature_knn=enable_feature_knn,
        feature_knn_k=feature_knn_k,
    )
    evidence_runtime_sec = time.perf_counter() - t_e0

    # 3) Validation-only selection of weights + uncertainty threshold
    if weight_candidates is None:
        weight_candidates = [
            {"b1": 1.0, "b2": 0.6, "b3": 0.0, "b4": 0.5, "b5": 0.3},
            {"b1": 1.0, "b2": 0.9, "b3": 0.0, "b4": 0.3, "b5": 0.2},
            {"b1": 1.0, "b2": 0.4, "b3": 0.0, "b4": 0.9, "b5": 0.5},
            {"b1": 1.2, "b2": 0.5, "b3": 0.0, "b4": 0.4, "b5": 0.2},
        ]

    t_sel0 = time.perf_counter()
    best_cfg: Optional[Dict[str, Any]] = None
    best_components: Optional[Dict[str, np.ndarray]] = None

    for w_cfg in weight_candidates:
        comps = build_selective_correction_scores(
            mlp_probs_np,
            evidence,
            b1=w_cfg["b1"],
            b2=w_cfg["b2"],
            b3=w_cfg["b3"],
            b4=w_cfg["b4"],
            b5=w_cfg["b5"],
        )
        th = choose_uncertainty_threshold(
            y_true,
            val_indices,
            mlp_pred_all,
            mlp_margin_all,
            comps["combined_scores"],
            threshold_candidates=threshold_candidates,
        )
        key = (th["val_acc"], -th["changed_fraction_val"], -th["uncertain_fraction_val"])
        if best_cfg is None or key > best_cfg["key"]:
            best_cfg = {
                "weights": dict(w_cfg),
                "threshold": th,
                "key": key,
            }
            best_components = comps

    assert best_cfg is not None and best_components is not None
    selection_runtime_sec = time.perf_counter() - t_sel0

    selected_threshold_high = float(best_cfg["threshold"]["threshold_high"])
    selected_weights = best_cfg["weights"]

    # 4) Inference with selected config
    t_inf0 = time.perf_counter()
    uncertain_mask_all = mlp_margin_all < selected_threshold_high
    final_pred_all = mlp_pred_all.copy()
    if uncertain_mask_all.any():
        final_pred_all[uncertain_mask_all] = np.argmax(
            best_components["combined_scores"][uncertain_mask_all], axis=1
        ).astype(np.int64)
    inference_runtime_sec = time.perf_counter() - t_inf0

    test_pred = final_pred_all[test_indices]
    test_acc = _simple_accuracy(y_true[test_indices], test_pred)
    val_acc = _simple_accuracy(y_true[val_indices], final_pred_all[val_indices])
    test_acc_mlp = _simple_accuracy(y_true[test_indices], mlp_pred_all[test_indices])

    val_summary = summarize_selective_correction_behavior(
        y_true, val_indices, mlp_pred_all, final_pred_all, uncertain_mask_all
    )
    test_summary = summarize_selective_correction_behavior(
        y_true, test_indices, mlp_pred_all, final_pred_all, uncertain_mask_all
    )

    selective_overhead_runtime_sec = evidence_runtime_sec + selection_runtime_sec + inference_runtime_sec
    total_runtime_sec = mlp_train_runtime_sec + selective_overhead_runtime_sec
    if mlp_runtime_sec is not None:
        avg_runtime_overhead_over_mlp_sec = float(selective_overhead_runtime_sec)
    else:
        avg_runtime_overhead_over_mlp_sec = float(selective_overhead_runtime_sec)

    diagnostics_path = None
    if write_node_diagnostics:
        run_id = diagnostics_run_id or f"{log_prefix}_{int(time.time())}"
        dataset_slug = (dataset_key_for_logs or globals().get("DATASET_KEY", "dataset")).lower()
        out_dir = LOG_DIR if "LOG_DIR" in globals() else "logs"
        os.makedirs(out_dir, exist_ok=True)
        diagnostics_path = os.path.join(out_dir, f"{dataset_slug}_selective_graph_correction_{run_id}.json")

        node_rows = []
        for node_id in test_indices.tolist():
            nid = int(node_id)
            final_lbl = int(final_pred_all[nid])
            node_rows.append(
                {
                    "node_id": nid,
                    "mlp_pred": int(mlp_pred_all[nid]),
                    "mlp_max_prob": float(mlp_top1_all[nid]),
                    "mlp_margin": float(mlp_margin_all[nid]),
                    "was_uncertain": bool(uncertain_mask_all[nid]),
                    "final_pred": final_lbl,
                    "changed_from_mlp": bool(final_pred_all[nid] != mlp_pred_all[nid]),
                    "selected_threshold_high": float(selected_threshold_high),
                    "selected_method_mode": "feature_first_selective_weighted_correction_v1",
                    "winning_class_components": {
                        "mlp_term": float(best_components["mlp_term"][nid, final_lbl]),
                        "feature_similarity_term": float(best_components["feature_similarity_term"][nid, final_lbl]),
                        "feature_knn_term": float(best_components["feature_knn_term"][nid, final_lbl]),
                        "graph_neighbor_term": float(best_components["graph_neighbor_term"][nid, final_lbl]),
                        "compatibility_term": float(best_components["compatibility_term"][nid, final_lbl]),
                    },
                }
            )

        payload = {
            "dataset": dataset_key_for_logs or globals().get("DATASET_KEY", "dataset"),
            "seed": int(seed),
            "method": "selective_graph_correction",
            "selected_threshold_high": float(selected_threshold_high),
            "selected_weights": {k: float(v) for k, v in selected_weights.items()},
            "mode": "feature_first_selective_weighted_correction_v1",
            "feature_knn_used": bool(enable_feature_knn),
            "feature_knn_k": int(feature_knn_k),
            "runtime_sec": {
                "mlp_training_runtime_sec": float(mlp_train_runtime_sec),
                "evidence_runtime_sec": float(evidence_runtime_sec),
                "selection_runtime_sec": float(selection_runtime_sec),
                "inference_runtime_sec": float(inference_runtime_sec),
                "selective_overhead_runtime_sec": float(selective_overhead_runtime_sec),
                "total_runtime_sec": float(total_runtime_sec),
                "avg_runtime_overhead_over_mlp_sec": float(avg_runtime_overhead_over_mlp_sec),
            },
            "summary": {
                "val_acc_mlp": float(_simple_accuracy(y_true[val_indices], mlp_pred_all[val_indices])),
                "val_acc_selective": float(val_acc),
                "test_acc_mlp": float(test_acc_mlp),
                "test_acc_selective": float(test_acc),
                "val_fraction_uncertain": val_summary["fraction_uncertain"],
                "test_fraction_uncertain": test_summary["fraction_uncertain"],
                "val_fraction_changed_from_mlp": val_summary["fraction_changed_from_mlp"],
                "test_fraction_changed_from_mlp": test_summary["fraction_changed_from_mlp"],
                "acc_val_confident": val_summary["acc_confident"],
                "acc_val_uncertain": val_summary["acc_uncertain"],
                "acc_test_confident": test_summary["acc_confident"],
                "acc_test_uncertain": test_summary["acc_uncertain"],
            },
            "test_node_diagnostics": node_rows,
        }
        with open(diagnostics_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    if log_file:
        write_log(
            (
                f"SelectiveCorrection: test_acc={test_acc:.4f} vs MLP={test_acc_mlp:.4f} | "
                f"thr={selected_threshold_high:.4f} | uncertain_frac_test={test_summary['fraction_uncertain']:.3f} | "
                f"changed_frac_test={test_summary['fraction_changed_from_mlp']:.3f}"
            ),
            log_file,
            tag=f"{log_prefix}:SUMMARY",
        )

    info = {
        "selected_threshold_high": float(selected_threshold_high),
        "selected_weights": {k: float(v) for k, v in selected_weights.items()},
        "mode": "feature_first_selective_weighted_correction_v1",
        "feature_knn_used": bool(enable_feature_knn),
        "feature_knn_k": int(feature_knn_k),
        "val_acc_selective": float(val_acc),
        "test_acc_mlp": float(test_acc_mlp),
        "test_acc_selective": float(test_acc),
        "fraction_test_nodes_uncertain": float(test_summary["fraction_uncertain"] or 0.0),
        "fraction_test_nodes_changed_from_mlp": float(test_summary["fraction_changed_from_mlp"] or 0.0),
        "acc_test_confident_nodes": test_summary["acc_confident"],
        "acc_test_uncertain_nodes": test_summary["acc_uncertain"],
        "runtime_breakdown_sec": {
            "mlp_training_runtime_sec": float(mlp_train_runtime_sec),
            "evidence_runtime_sec": float(evidence_runtime_sec),
            "selection_runtime_sec": float(selection_runtime_sec),
            "inference_runtime_sec": float(inference_runtime_sec),
            "selective_overhead_runtime_sec": float(selective_overhead_runtime_sec),
            "total_runtime_sec": float(total_runtime_sec),
            "avg_runtime_overhead_over_mlp_sec": float(avg_runtime_overhead_over_mlp_sec),
        },
        "diagnostics_path": diagnostics_path,
        "selection_trace": {
            "best_val_acc": float(best_cfg["threshold"]["val_acc"]),
            "best_val_uncertain_fraction": float(best_cfg["threshold"]["uncertain_fraction_val"]),
            "best_val_changed_fraction": float(best_cfg["threshold"]["changed_fraction_val"]),
            "threshold_candidates_evaluated": best_cfg["threshold"]["threshold_candidates_evaluated"],
            "weight_candidates_evaluated": [{k: float(v) for k, v in cfg.items()} for cfg in weight_candidates],
        },
        "wall_clock_runtime_sec": float(time.perf_counter() - t_all0),
    }
    return torch.tensor(test_pred, device=device), float(test_acc), info


def _build_gate_feature_matrix(
    node_indices: np.ndarray,
    *,
    mlp_probs_np: np.ndarray,
    prop_details: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray]]:
    """
    Construct compact node-wise gate features from existing model outputs.
    """
    idx = np.asarray(node_indices, dtype=np.int64)

    # MLP-side
    mlp_pred = np.argmax(mlp_probs_np, axis=1).astype(np.int64)
    mlp_top1 = mlp_probs_np.max(axis=1)
    _, _, mlp_top2, mlp_margin = _top1_top2_margin(mlp_probs_np)

    # Propagation-side
    prop_pred = prop_details["prop_pred_all"].astype(np.int64)
    prop_margin = prop_details["prop_margin"]
    prop_top1_raw = prop_details["prop_top1_score"]
    # Score-scale differs from MLP probabilities; use a monotonic confidence proxy.
    prop_conf_proxy = _safe_sigmoid(prop_margin)

    agree = (mlp_pred == prop_pred).astype(np.float64)
    conf_gap = mlp_top1 - prop_conf_proxy

    feature_map = {
        "mlp_max_prob": mlp_top1,
        "mlp_margin": mlp_margin,
        "mlp_pred_class": mlp_pred.astype(np.float64),
        "prop_max_score": prop_top1_raw,
        "prop_margin": prop_margin,
        "prop_pred_class": prop_pred.astype(np.float64),
        "agree_mlp_prop": agree,
        "confidence_gap_mlp_minus_prop": conf_gap,
        "node_degree": prop_details["degree"].astype(np.float64),
        "support_labeled_neighbors": prop_details["support_labeled_neighbors"].astype(np.float64),
        "neighbor_class_entropy": prop_details["neighbor_class_entropy"],
        "similarity_gap": prop_details["similarity_gap"],
        "compatibility_gap": prop_details["compatibility_gap"],
        # v1 placeholders for vanilla propagation:
        "deferred_count": prop_details["deferred_count"].astype(np.float64),
        "reconsidered_in_refinement": prop_details["reconsidered_in_refinement"].astype(np.float64),
    }

    feature_names = [
        "mlp_max_prob",
        "mlp_margin",
        "mlp_pred_class",
        "prop_max_score",
        "prop_margin",
        "prop_pred_class",
        "agree_mlp_prop",
        "confidence_gap_mlp_minus_prop",
        "node_degree",
        "support_labeled_neighbors",
        "neighbor_class_entropy",
        "similarity_gap",
        "compatibility_gap",
        "deferred_count",
        "reconsidered_in_refinement",
    ]

    X = np.column_stack([feature_map[name][idx] for name in feature_names]).astype(np.float32)
    return X, feature_names, feature_map


def _train_logistic_gate(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    *,
    seed: int = 1337,
    epochs: int = 300,
    lr: float = 0.05,
    weight_decay: float = 1e-4,
) -> Dict[str, Any]:
    """
    Lightweight logistic regression in torch for interpretability and no extra deps.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

    if X.shape[0] == 0:
        return {"mode": "fallback", "reason": "no_supervised_examples", "feature_names": feature_names}
    uniq = np.unique(y)
    if uniq.size < 2:
        return {
            "mode": "fallback",
            "reason": "single_class_supervision",
            "single_class_value": float(uniq[0]),
            "feature_names": feature_names,
        }

    torch.manual_seed(int(seed))
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)

    mean = X_t.mean(dim=0, keepdim=True)
    std = X_t.std(dim=0, keepdim=True).clamp_min(1e-6)
    Xn = (X_t - mean) / std

    model = nn.Linear(Xn.size(1), 1)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(int(epochs)):
        opt.zero_grad()
        logits = model(Xn)
        loss = loss_fn(logits, y_t)
        loss.backward()
        opt.step()

    with torch.no_grad():
        logits = model(Xn)
        probs = torch.sigmoid(logits)
        hard = (probs >= 0.5).float()
        train_acc = float((hard == y_t).float().mean().item())

    w = model.weight.detach().cpu().numpy().reshape(-1)
    b = float(model.bias.detach().cpu().item())

    return {
        "mode": "learned",
        "feature_names": feature_names,
        "mean": mean.detach().cpu().numpy().reshape(-1),
        "std": std.detach().cpu().numpy().reshape(-1),
        "weights": w,
        "bias": b,
        "train_acc": train_acc,
        "train_size": int(X.shape[0]),
        "class_balance_mlp_target": float(y.mean()),
    }


def _gate_predict_proba(gate_state: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)

    mode = gate_state.get("mode")
    if mode != "learned":
        # fallback: fixed decision based on validation winner
        fallback_use_mlp = bool(gate_state.get("fallback_use_mlp", True))
        return np.full(X.shape[0], 1.0 if fallback_use_mlp else 0.0, dtype=np.float64)

    mean = np.asarray(gate_state["mean"], dtype=np.float64)
    std = np.asarray(gate_state["std"], dtype=np.float64)
    w = np.asarray(gate_state["weights"], dtype=np.float64)
    b = float(gate_state["bias"])
    Xn = (X.astype(np.float64) - mean[None, :]) / std[None, :]
    logits = Xn @ w + b
    return _safe_sigmoid(logits)


def gated_mlp_prop_predictclass(
    data,
    train_indices,
    val_indices,
    test_indices,
    *,
    prop_params: Optional[Dict[str, float]] = None,
    mlp_probs: Optional[torch.Tensor] = None,
    seed: int = 1337,
    log_prefix: str = "gated_mlp_prop",
    log_file: str | None = None,
    diagnostics_run_id: str | None = None,
    gate_train_epochs: int = 300,
    gate_lr: float = 0.05,
    gate_weight_decay: float = 1e-4,
    write_node_diagnostics: bool = True,
    dataset_key_for_logs: Optional[str] = None,
):
    """
    Hard node-wise selector between MLP and propagation predictions.

    Safety: gate supervision is built only from validation-node labels.
    """
    device = data.x.device
    y_true = data.y.detach().cpu().numpy().astype(np.int64)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    val_indices = np.asarray(val_indices, dtype=np.int64)
    test_indices = np.asarray(test_indices, dtype=np.int64)

    # 1) Base models
    if mlp_probs is None:
        mlp_probs, _ = train_mlp_and_predict(
            data, train_indices, hidden=64, layers=2, dropout=0.5, lr=0.01, epochs=200, log_file=log_file
        )
    mlp_probs_np = mlp_probs.detach().cpu().numpy()
    mlp_pred_all = np.argmax(mlp_probs_np, axis=1).astype(np.int64)
    _, mlp_top1_all, _, mlp_margin_all = _top1_top2_margin(mlp_probs_np)

    if prop_params is None:
        params_prop, _, _ = robust_random_search(
            predictclass,
            data,
            train_indices,
            num_samples=RS_NUM_SAMPLES,
            n_splits=RS_SPLITS,
            repeats=RS_REPEATS,
            seed=seed,
            log_file=log_file,
            mlp_probs=None,
        )
        prop_params = {f"a{i+1}": p for i, p in enumerate(params_prop)}

    eval_idx = np.unique(np.concatenate([val_indices, test_indices], axis=0))
    _preds_eval, _acc_eval, prop_details = predictclass(
        data,
        train_indices,
        eval_idx,
        **prop_params,
        seed=seed,
        log_prefix=f"{log_prefix}:prop",
        log_file=None,
        mlp_probs=None,
        return_details=True,
    )
    prop_pred_all = np.asarray(prop_details["prop_pred_all"], dtype=np.int64)
    prop_conf_proxy_all = _safe_sigmoid(np.asarray(prop_details["prop_margin"], dtype=np.float64))

    # 2) Build gate features
    X_val, feature_names, feature_map = _build_gate_feature_matrix(
        val_indices, mlp_probs_np=mlp_probs_np, prop_details=prop_details
    )
    X_test, _, _ = _build_gate_feature_matrix(
        test_indices, mlp_probs_np=mlp_probs_np, prop_details=prop_details
    )

    # 3) Gate supervision from validation-only labels
    mlp_correct_val = (mlp_pred_all[val_indices] == y_true[val_indices])
    prop_correct_val = (prop_pred_all[val_indices] == y_true[val_indices])
    supervised_mask = np.logical_xor(mlp_correct_val, prop_correct_val)
    y_gate = mlp_correct_val[supervised_mask].astype(np.float32)  # 1=prefer MLP, 0=prefer Prop
    X_gate = X_val[supervised_mask]

    gate_state = _train_logistic_gate(
        X_gate,
        y_gate,
        feature_names,
        seed=seed,
        epochs=gate_train_epochs,
        lr=gate_lr,
        weight_decay=gate_weight_decay,
    )

    # Fallback when supervision is insufficient: validation winner.
    acc_val_mlp = _simple_accuracy(y_true[val_indices], mlp_pred_all[val_indices])
    acc_val_prop = _simple_accuracy(y_true[val_indices], prop_pred_all[val_indices])
    if gate_state.get("mode") != "learned":
        gate_state["fallback_use_mlp"] = bool(acc_val_mlp >= acc_val_prop)

    # 4) Gate inference
    val_gate_prob = _gate_predict_proba(gate_state, X_val)
    test_gate_prob = _gate_predict_proba(gate_state, X_test)
    choose_mlp_val = val_gate_prob >= 0.5
    choose_mlp_test = test_gate_prob >= 0.5

    final_val_pred = np.where(choose_mlp_val, mlp_pred_all[val_indices], prop_pred_all[val_indices]).astype(np.int64)
    final_test_pred = np.where(choose_mlp_test, mlp_pred_all[test_indices], prop_pred_all[test_indices]).astype(np.int64)

    val_final_acc = _simple_accuracy(y_true[val_indices], final_val_pred)
    test_final_acc = _simple_accuracy(y_true[test_indices], final_test_pred)
    acc_test_mlp = _simple_accuracy(y_true[test_indices], mlp_pred_all[test_indices])
    acc_test_prop = _simple_accuracy(y_true[test_indices], prop_pred_all[test_indices])

    gate_val_acc_supervised = None
    if supervised_mask.any():
        gate_val_hard = (val_gate_prob[supervised_mask] >= 0.5).astype(np.int64)
        gate_target = y_gate.astype(np.int64)
        gate_val_acc_supervised = float((gate_val_hard == gate_target).mean())

    if log_file:
        write_log(
            (
                f"Gate mode={gate_state.get('mode')} | val_supervised={int(supervised_mask.sum())} | "
                f"val_gate_acc={gate_val_acc_supervised if gate_val_acc_supervised is not None else 'n/a'} | "
                f"test_acc(gated)={test_final_acc:.4f} | test_acc(mlp)={acc_test_mlp:.4f} | "
                f"test_acc(prop)={acc_test_prop:.4f} | frac_mlp={choose_mlp_test.mean():.3f}"
            ),
            log_file,
            tag=f"{log_prefix}:SUMMARY",
        )
        if gate_state.get("mode") == "learned":
            coefs_msg = ", ".join(
                f"{n}={float(w):+.3f}" for n, w in zip(feature_names, gate_state["weights"])
            )
            write_log(f"Gate coefficients (standardized): {coefs_msg}", log_file, tag=f"{log_prefix}:COEF")

    diagnostics_path = None
    if write_node_diagnostics:
        run_id = diagnostics_run_id or f"{log_prefix}_{int(time.time())}"
        dataset_slug = (dataset_key_for_logs or globals().get("DATASET_KEY", "dataset")).lower()
        out_dir = LOG_DIR if "LOG_DIR" in globals() else "logs"
        os.makedirs(out_dir, exist_ok=True)
        diagnostics_path = os.path.join(out_dir, f"{dataset_slug}_gated_mlp_prop_{run_id}.json")

        test_nodes_payload = []
        for j, node_id in enumerate(test_indices.tolist()):
            feature_values = {name: float(feature_map[name][node_id]) for name in feature_names}
            test_nodes_payload.append(
                {
                    "node_id": int(node_id),
                    "mlp_pred": int(mlp_pred_all[node_id]),
                    "mlp_confidence": float(mlp_top1_all[node_id]),
                    "mlp_margin": float(mlp_margin_all[node_id]),
                    "prop_pred": int(prop_pred_all[node_id]),
                    "prop_confidence": float(prop_conf_proxy_all[node_id]),
                    "prop_margin": float(prop_details["prop_margin"][node_id]),
                    "gate_alpha_mlp": float(test_gate_prob[j]),
                    "gate_decision_choose_mlp": bool(choose_mlp_test[j]),
                    "selected_method": "mlp" if bool(choose_mlp_test[j]) else "prop",
                    "mlp_prop_agree": bool(mlp_pred_all[node_id] == prop_pred_all[node_id]),
                    "final_pred": int(final_test_pred[j]),
                    "gate_features": feature_values,
                }
            )

        payload = {
            "dataset": dataset_key_for_logs or globals().get("DATASET_KEY", "dataset"),
            "seed": int(seed),
            "method": "gated_mlp_prop",
            "summary": {
                "val_supervised_nodes_for_gate": int(supervised_mask.sum()),
                "gate_accuracy_on_val_supervised_nodes": gate_val_acc_supervised,
                "fraction_test_nodes_assigned_to_mlp": float(choose_mlp_test.mean()),
                "fraction_test_nodes_assigned_to_propagation": float(1.0 - choose_mlp_test.mean()),
                "val_acc_mlp": float(acc_val_mlp),
                "val_acc_prop": float(acc_val_prop),
                "val_acc_gated": float(val_final_acc),
                "test_acc_mlp": float(acc_test_mlp),
                "test_acc_prop": float(acc_test_prop),
                "test_acc_gated": float(test_final_acc),
            },
            "gate_model": {
                "mode": gate_state.get("mode"),
                "feature_names": feature_names,
                "weights": [float(x) for x in np.asarray(gate_state.get("weights", []), dtype=np.float64).tolist()],
                "bias": float(gate_state.get("bias", 0.0)),
                "train_acc_on_supervised_subset": gate_state.get("train_acc"),
                "train_size": gate_state.get("train_size"),
                "fallback_use_mlp": gate_state.get("fallback_use_mlp"),
                "fallback_reason": gate_state.get("reason"),
            },
            "test_node_diagnostics": test_nodes_payload,
        }
        with open(diagnostics_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    info = {
        "gate_mode": gate_state.get("mode"),
        "gate_feature_names": feature_names,
        "gate_train_size": int(supervised_mask.sum()),
        "gate_accuracy_on_val_supervised_nodes": gate_val_acc_supervised,
        "val_acc_gated": float(val_final_acc),
        "val_acc_mlp": float(acc_val_mlp),
        "val_acc_prop": float(acc_val_prop),
        "test_acc_mlp": float(acc_test_mlp),
        "test_acc_prop": float(acc_test_prop),
        "fraction_test_nodes_assigned_to_mlp": float(choose_mlp_test.mean()),
        "fraction_test_nodes_assigned_to_propagation": float(1.0 - choose_mlp_test.mean()),
        "diagnostics_path": diagnostics_path,
        "gate_state": gate_state,
    }
    return torch.tensor(final_test_pred, device=device), float(test_final_acc), info

# =====================================================
# BLOCK 5: Main Execution Loop
# =====================================================

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
timestamp_str = time.strftime('%Y%m%d_%H%M%S')
LOG_FILE = os.path.join(LOG_DIR, f"{DATASET_KEY.lower()}_full-investigate-v2_{timestamp_str}.txt")

write_log(f"[START] Full Investigation for DATASET_KEY={DATASET_KEY}", LOG_FILE, tag="RUN")

num_iterations = 10 
EXPERIMENT_REPEATS = 5 
CONFIDENCE_THRESHOLD = 0.90 # Increased confidence threshold for safer pseudo-labeling
RS_NUM_SAMPLES = 50 
RS_SPLITS = 3
RS_REPEATS = 1

# Optional priority-frontier BFS experiment (disabled by default).
ENABLE_PRIORITY_BFS_EXPERIMENT = False
PRIORITY_BFS_MARGIN_THRESHOLD = 0.05
PRIORITY_BFS_REFINEMENT_MARGIN_THRESHOLD = 0.02

# Optional learned node-wise gate between MLP and propagation.
ENABLE_GATED_MLP_PROP_EXPERIMENT = False

# Optional feature-first selective graph correction (new side-by-side method).
ENABLE_SELECTIVE_GRAPH_CORRECTION_EXPERIMENT = False

# Optional multi-source and ensemble experiments (disabled by default).
ENABLE_MULTI_SOURCE_BFS_EXPERIMENT = False
ENABLE_ENSEMBLE_MULTI_SOURCE_BFS = False

results = {
    'mlp_only': [],
    'prop_only': [],
    'mlp_refined': [],
    'prop_mlp_selected': [],
    'selective_graph_correction': [],
    'selective_graph_correction_uncertain_fraction': [],
    'selective_graph_correction_changed_fraction': [],
    'gated_mlp_prop': [],
    'gated_mlp_fraction': [],
    'priority_bfs': [],
    'multi_source_bfs': [],
    'ensemble_multi_source_bfs': [],
    'refined_chosen_count': 0,
    'prop_mlp_switched_to_mlp': 0, 
    'total_runs': 0
}

num_classes = int(data.y.max().item()) + 1
device = data.x.device
N_total = int(data.num_nodes)

for run in range(num_iterations):
    run_seed = 1337 + run
    write_log(f"{'='*20} SPLIT {run} {'='*20}", LOG_FILE)

    split_path = f"{DATASET_KEY.lower()}_split_0.6_0.2_{run}.npz"
    if os.path.exists(split_path):
        split = np.load(split_path)
        train_idx = _to_long_tensor(np.where(split["train_mask"])[0], device=device)
        val_idx   = _to_long_tensor(np.where(split["val_mask"])[0], device=device)
        test_idx  = _to_long_tensor(np.where(split["test_mask"])[0], device=device)
    else:
        write_log(f"Split {split_path} not found. Generating random split.", LOG_FILE, tag="WARN")
        perm = torch.randperm(N_total, device=device)
        n_train = int(N_total * 0.6)
        n_val = int(N_total * 0.2)
        train_idx = perm[:n_train]
        val_idx = perm[n_train:n_train+n_val]
        test_idx = perm[n_train+n_val:]

    train_np = _to_numpy_idx(train_idx)
    val_np   = _to_numpy_idx(val_idx)
    test_np  = _to_numpy_idx(test_idx)

    log_kv(LOG_FILE, tag="SPLIT_INFO", n_train=len(train_np), n_val=len(val_np))

    for repeat in range(EXPERIMENT_REPEATS):
        results['total_runs'] += 1
        write_log(f"-- Repeat {repeat+1}/{EXPERIMENT_REPEATS} --", LOG_FILE)
        
        current_seed = run_seed * 100 + repeat 
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)
        
        # 1. MLP ONLY
        t_mlp0 = time.perf_counter()
        mlp_probs, _ = train_mlp_and_predict(data, train_idx, hidden=64, layers=2, dropout=0.5, epochs=200, log_file=LOG_FILE)
        mlp_runtime_sec = time.perf_counter() - t_mlp0
        max_probs, preds_mlp_all = mlp_probs.max(dim=1)
        
        acc_val_mlp = (preds_mlp_all[val_idx] == data.y[val_idx]).float().mean().item()
        acc_test_mlp = (preds_mlp_all[test_idx] == data.y[test_idx]).float().mean().item()
        
        results['mlp_only'].append(acc_test_mlp)
        write_log(f"  [1] MLP Only: Val={acc_val_mlp:.4f}, Test={acc_test_mlp:.4f}", LOG_FILE)

        # 2. PROP ONLY
        params_prop, _, _ = robust_random_search(
            predictclass, data, train_np,
            num_samples=RS_NUM_SAMPLES, n_splits=RS_SPLITS, repeats=RS_REPEATS,
            seed=current_seed, log_file=LOG_FILE, mlp_probs=None 
        )
        dict_params_prop = {f"a{i+1}": p for i, p in enumerate(params_prop)}
        
        _, acc_val_prop = predictclass(data, train_np, val_np, **dict_params_prop, seed=current_seed, mlp_probs=None, log_file=LOG_FILE)
        _, acc_test_prop = predictclass(data, train_np, test_np, **dict_params_prop, seed=current_seed, mlp_probs=None, log_file=LOG_FILE)
        
        results['prop_only'].append(acc_test_prop)
        write_log(f"  [2] Prop Only: Val={acc_val_prop:.4f}, Test={acc_test_prop:.4f}", LOG_FILE)

        # 3. MLP + REFINED (Hybrid)
        params_hyb, _, _ = robust_random_search(
            predictclass, data, train_np,
            num_samples=RS_NUM_SAMPLES, n_splits=RS_SPLITS, repeats=RS_REPEATS,
            seed=current_seed, log_file=None, mlp_probs=mlp_probs 
        )
        dict_params_hyb = {f"a{i+1}": p for i, p in enumerate(params_hyb)}

        is_unlabeled = torch.ones(N_total, dtype=torch.bool, device=device)
        is_unlabeled[train_idx] = False
        is_high_conf = (max_probs >= CONFIDENCE_THRESHOLD) & is_unlabeled
        
        # Val Step
        mask_val_support = is_high_conf.clone()
        mask_val_support[val_idx] = False
        pseudo_idx_val = torch.where(mask_val_support)[0].cpu().numpy()
        
        aug_train_val = np.concatenate([train_np, pseudo_idx_val])
        orig_y = data.y.clone()
        data.y[pseudo_idx_val] = preds_mlp_all[pseudo_idx_val]
        
        try:
            write_log("Running Val-Refined:", LOG_FILE)
            _, acc_val_refined = predictclass(
                data, aug_train_val, val_np, **dict_params_hyb, seed=current_seed, mlp_probs=mlp_probs, log_file=LOG_FILE
            )
        finally:
            data.y = orig_y.clone()

        use_refined = float(acc_val_refined) > acc_val_mlp
        if use_refined: results['refined_chosen_count'] += 1
        
        # Test Step
        mask_test_support = is_high_conf.clone()
        mask_test_support[test_idx] = False
        pseudo_idx_test = torch.where(mask_test_support)[0].cpu().numpy()
        
        aug_train_test = np.concatenate([train_np, pseudo_idx_test])
        data.y[pseudo_idx_test] = preds_mlp_all[pseudo_idx_test]
        try:
            write_log("Running Test-Refined:", LOG_FILE)
            _, acc_test_refined = predictclass(
                data, aug_train_test, test_np, **dict_params_hyb, seed=current_seed, mlp_probs=mlp_probs, log_file=LOG_FILE
            )
        finally:
            data.y = orig_y.clone()
            
        final_acc_s3 = float(acc_test_refined) if use_refined else acc_test_mlp
        results['mlp_refined'].append(final_acc_s3)
        
        decision_str = "REFINED" if use_refined else "MLP"
        write_log(f"  [3] MLP+Refined: Val(Ref)={acc_val_refined:.4f} vs Val(MLP)={acc_val_mlp:.4f} -> Chosen: {decision_str}. Test={final_acc_s3:.4f}", LOG_FILE)

        # 4. PROP + MLP (Selection)
        use_mlp_s4 = acc_val_mlp > float(acc_val_prop)
        
        if use_mlp_s4: 
            results['prop_mlp_switched_to_mlp'] += 1
            final_acc_s4 = acc_test_mlp
        else:
            final_acc_s4 = float(acc_test_prop)
            
        results['prop_mlp_selected'].append(final_acc_s4)
        decision_s4 = "MLP" if use_mlp_s4 else "PROP"
        write_log(f"  [4] Prop+MLP(Select): Val(Prop)={acc_val_prop:.4f} vs Val(MLP)={acc_val_mlp:.4f} -> Chosen: {decision_s4}. Test={final_acc_s4:.4f}", LOG_FILE)

        # 5. Optional: feature-first selective graph correction (MLP primary, graph correction on uncertain nodes)
        if ENABLE_SELECTIVE_GRAPH_CORRECTION_EXPERIMENT:
            _, acc_test_sel, sel_info = selective_graph_correction_predictclass(
                data,
                train_np,
                val_np,
                test_np,
                mlp_probs=mlp_probs,
                seed=current_seed,
                log_prefix=f"{DATASET_KEY}:TEST-SELECTIVE-CORR run={run} rep={repeat}",
                log_file=LOG_FILE,
                diagnostics_run_id=f"run{run}_rep{repeat}",
                dataset_key_for_logs=DATASET_KEY,
                mlp_runtime_sec=mlp_runtime_sec,
                enable_feature_knn=False,
                write_node_diagnostics=True,
            )
            results['selective_graph_correction'].append(float(acc_test_sel))
            results['selective_graph_correction_uncertain_fraction'].append(float(sel_info["fraction_test_nodes_uncertain"]))
            results['selective_graph_correction_changed_fraction'].append(float(sel_info["fraction_test_nodes_changed_from_mlp"]))
            write_log(
                (
                    f"  [5] SelectiveGraphCorrection: Test={acc_test_sel:.4f} | "
                    f"UncertainFrac={sel_info['fraction_test_nodes_uncertain']:.3f} | "
                    f"ChangedFrac={sel_info['fraction_test_nodes_changed_from_mlp']:.3f} | "
                    f"Thr={sel_info['selected_threshold_high']:.3f}"
                ),
                LOG_FILE,
            )

        # 6. Optional: learned node-wise gate (MLP vs propagation)
        if ENABLE_GATED_MLP_PROP_EXPERIMENT:
            _, acc_test_gated, gated_info = gated_mlp_prop_predictclass(
                data,
                train_np,
                val_np,
                test_np,
                prop_params=dict_params_prop,
                mlp_probs=mlp_probs,
                seed=current_seed,
                log_prefix=f"{DATASET_KEY}:TEST-GATED-MLP-PROP run={run} rep={repeat}",
                log_file=LOG_FILE,
                diagnostics_run_id=f"run{run}_rep{repeat}",
                write_node_diagnostics=True,
                dataset_key_for_logs=DATASET_KEY,
            )
            results['gated_mlp_prop'].append(float(acc_test_gated))
            results['gated_mlp_fraction'].append(float(gated_info["fraction_test_nodes_assigned_to_mlp"]))
            write_log(
                (
                    f"  [6] Gated MLP+Prop: Test={acc_test_gated:.4f} | "
                    f"GateValAcc={gated_info['gate_accuracy_on_val_supervised_nodes']} | "
                    f"FracMLP={gated_info['fraction_test_nodes_assigned_to_mlp']:.3f}"
                ),
                LOG_FILE,
            )

        # 7. Optional: Priority-BFS propagation (side-by-side, non-disruptive)
        if ENABLE_PRIORITY_BFS_EXPERIMENT:
            _, acc_test_priority = priority_bfs_predictclass(
                data,
                train_np,
                test_np,
                **dict_params_prop,
                seed=current_seed,
                log_prefix=f"{DATASET_KEY}:TEST-PRIORITY run={run} rep={repeat}",
                log_file=LOG_FILE,
                margin_threshold=PRIORITY_BFS_MARGIN_THRESHOLD,
                refinement_margin_threshold=PRIORITY_BFS_REFINEMENT_MARGIN_THRESHOLD,
                diagnostics_run_id=f"run{run}_rep{repeat}",
            )
            results['priority_bfs'].append(float(acc_test_priority))
            write_log(f"  [7] Priority-BFS: Test={acc_test_priority:.4f}", LOG_FILE)

        # 8. Optional: Multi-source priority BFS (single run, all_train seeds)
        if ENABLE_MULTI_SOURCE_BFS_EXPERIMENT:
            _, acc_test_ms = multi_source_priority_bfs_predictclass(
                data,
                train_np,
                test_np,
                **dict_params_prop,
                seed=current_seed,
                log_prefix=f"{DATASET_KEY}:TEST-MULTI-SOURCE run={run} rep={repeat}",
                log_file=LOG_FILE,
                seed_mode="all_train",
                seeds_per_class=None,
                enable_deferral=True,
                margin_threshold=PRIORITY_BFS_MARGIN_THRESHOLD,
                refinement_margin_threshold=PRIORITY_BFS_REFINEMENT_MARGIN_THRESHOLD,
                max_deferrals_per_node=5,
                record_diagnostics=True,
                diagnostics_only_test_nodes=True,
            )
            results['multi_source_bfs'].append(float(acc_test_ms))
            write_log(f"  [8] Multi-Source BFS: Test={acc_test_ms:.4f}", LOG_FILE)

        # 9. Optional: Ensemble multi-source BFS
        if ENABLE_ENSEMBLE_MULTI_SOURCE_BFS:
            _, acc_test_ens = ensemble_multi_source_bfs_predictclass(
                data,
                train_np,
                test_np,
                **dict_params_prop,
                num_restarts=5,
                base_seed=current_seed,
                seed_mode="all_train",
                seeds_per_class=None,
                margin_threshold=PRIORITY_BFS_MARGIN_THRESHOLD,
                refinement_margin_threshold=PRIORITY_BFS_REFINEMENT_MARGIN_THRESHOLD,
                max_deferrals_per_node=5,
                log_prefix=f"{DATASET_KEY}:TEST-ENS-MULTI-SOURCE run={run} rep={repeat}",
                log_file=LOG_FILE,
            )
            results['ensemble_multi_source_bfs'].append(float(acc_test_ens))
            write_log(f"  [9] Ensemble Multi-Source BFS: Test={acc_test_ens:.4f}", LOG_FILE)


msg = "\n" + "="*50 + f"\n FINAL SUMMARY ({DATASET_KEY})\n" + "="*50 + "\n"

def get_stats(arr):
    a = np.array(arr)
    return f"{np.mean(a):.4f} ± {np.std(a):.4f}"

msg += f"[1] MLP Only:          {get_stats(results['mlp_only'])}\n"
msg += f"[2] Prop Only:         {get_stats(results['prop_only'])}\n"
msg += f"[3] MLP + Refined:     {get_stats(results['mlp_refined'])}\n"
msg += f"    -> Refinement Chosen: {results['refined_chosen_count']}/{results['total_runs']} ({(results['refined_chosen_count']/results['total_runs'])*100:.1f}%)\n"
msg += f"[4] Prop + MLP(Select):{get_stats(results['prop_mlp_selected'])}\n"
msg += f"    -> Switched to MLP:   {results['prop_mlp_switched_to_mlp']}/{results['total_runs']} ({(results['prop_mlp_switched_to_mlp']/results['total_runs'])*100:.1f}%)\n"
if results['selective_graph_correction']:
    msg += f"[5] Selective Graph Correction: {get_stats(results['selective_graph_correction'])}\n"
    msg += f"    -> Mean uncertain fraction: {np.mean(np.array(results['selective_graph_correction_uncertain_fraction'])):.3f}\n"
    msg += f"    -> Mean changed fraction:   {np.mean(np.array(results['selective_graph_correction_changed_fraction'])):.3f}\n"
if results['gated_mlp_prop']:
    msg += f"[6] Gated MLP+Prop:    {get_stats(results['gated_mlp_prop'])}\n"
    msg += f"    -> Mean MLP assignment fraction: {np.mean(np.array(results['gated_mlp_fraction'])):.3f}\n"
if results['priority_bfs']:
    msg += f"[7] Priority-BFS:       {get_stats(results['priority_bfs'])}\n"
if results['multi_source_bfs']:
    msg += f"[8] Multi-Source BFS:   {get_stats(results['multi_source_bfs'])}\n"
if results['ensemble_multi_source_bfs']:
    msg += f"[9] Ensemble Multi-Source BFS: {get_stats(results['ensemble_multi_source_bfs'])}\n"

write_log(msg, LOG_FILE, tag="SUMMARY")
print(f"Logs written to {LOG_FILE}")


# In[19]:





# In[20]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[21]:





# In[22]:





# In[23]:





# In[24]:





# In[ ]:





# In[25]:





# In[26]:





# In[27]:





# In[28]:





# In[29]:





# In[30]:





# In[31]:





# In[32]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





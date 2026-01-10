#!/usr/bin/env python
# coding: utf-8

# In[68]:


import os
import random
import itertools
import heapq
from collections import Counter, defaultdict

import numpy as np
import torch
import networkx as nx

# ✅ avoid seaborn (it imports pandas)
import matplotlib.pyplot as plt

from sortedcontainers import SortedSet

from torch_geometric.datasets import WebKB, WikipediaNetwork, Actor, FacebookPagePage, SNAPDataset
from torch_geometric.utils import to_networkx


# 

# In[69]:


from torch_geometric.datasets import Planetoid

from torch_geometric.datasets import LINKXDataset
# =====================================================
# Select dataset (UNCOMMENT ONE LINE ONLY)
# =====================================================

#DATASET_KEY = "cornell"
#DATASET_KEY = "texas"
DATASET_KEY = "chameleon"
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



# In[71]:


import os
import time
import heapq
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict, Counter
from typing import Optional

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
        except Exception as e:
            # Fallback to print if file write fails, just to notify user
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
    log_file=None, # Added log_file argument
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
        fallback_params = [0.1, -0.1, 0.1, 0.2, 0.1, 0.05, 0.5, 1.0] 
        mode_str = "HETERO"
    else:
        # Homophilic Strategy
        a2_bounds = (0.0, 1.0)
        a8_bounds = (0.0, 0.6)
        fallback_params = [0.1, 0.5, 0.1, 0.2, 0.1, 0.05, 0.6, 0.2]
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
            rng.uniform(*a2_bounds),         # a2: Neighbor (Dynamic range)
            rng.uniform(0.0, 0.5),           # a3: Similarity
            rng.uniform(0.0, 0.5),           # a4: Degree
            rng.uniform(0.0, 0.3),           # a5: Count
            rng.uniform(0.0, 0.1),           # a6: Tie
            rng.uniform(0.4, 0.9),           # a7: Centroid
            rng.uniform(*a8_bounds),         # a8: Compat (Dynamic range)
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
        write_log("All trials failed. Using Smart Fallback.", log_file, tag="TUNING_WARN")
    
    return best_params, best_score, {}

# =====================================================
# BLOCK 3: PredictClass
# =====================================================

def predictclass(
    data, train_indices, test_indices,
    a1, a2, a3, a4, a5, a6, a7, a8,
    seed: int = 1337,
    log_prefix: str = "predictclass",
    log_every: int | None = None,
    log_file: str | None = None,
    mlp_probs: Optional[torch.Tensor] = None, 
    # Performance Knobs
    min_labeled_neighbors_for_full_weight: int = 2, # Reduced to 2 for small graphs
    **kwargs
):
    t_all = Timer()
    device = data.x.device
    N = int(data.num_nodes)
    
    # 1. Setup
    train_indices = np.asarray(train_indices, dtype=np.int64)
    test_indices  = np.asarray(test_indices,  dtype=np.int64)
    y_true = data.y.detach().cpu().numpy().astype(np.int64)
    
    all_labels = sorted(set(y_true.tolist()))
    C = len(all_labels)
    lbl2i = {lbl: j for j, lbl in enumerate(all_labels)}
    lbl_arr = np.array(all_labels, dtype=np.int64)
    
    mlp_probs_np = mlp_probs.detach().cpu().numpy() if mlp_probs is not None else None

    # 2. Build Graph & Neighbors
    edge_index = torch.cat([data.edge_index, data.edge_index.flip(0)], dim=1)
    edge_index = torch.unique(edge_index, dim=1)
    ei = edge_index.detach().cpu().numpy()
    src, dst = ei[0], ei[1]
    
    neighbors = [[] for _ in range(N)]
    for u, v in zip(src, dst): neighbors[u].append(v)
    deg = np.array([len(n) for n in neighbors], dtype=np.int32)

    # 3. Init State
    y_pred = np.full(N, -1, dtype=np.int64)
    y_pred[train_indices] = y_true[train_indices]
    
    # Initial Centroids
    train_mask_t = torch.zeros(N, device=device, dtype=torch.bool)
    train_mask_t[train_indices] = True
    
    # Feature Standardization (Standard Z-Score based on Train)
    with torch.no_grad():
        x_tr = data.x[train_mask_t]
        mu, sigma = x_tr.mean(0, keepdim=True), x_tr.std(0, keepdim=True).clamp_min(1e-6)
        xz = (data.x - mu) / sigma
        
    class_insts = defaultdict(set)
    for i in train_indices: class_insts[y_pred[i]].add(i)

    # 4. Matrices
    # Prior
    counts = np.bincount([lbl2i[y_pred[i]] for i in train_indices], minlength=C)
    prior = (counts + 1.0) / (len(train_indices) + C)
    
    # Compat (Initialize with higher smoothing for small graphs)
    compat_cts = np.ones((C, C)) * 2.0 
    tr_src, tr_dst = src[np.isin(src, train_indices)], dst[np.isin(src, train_indices)]
    # Filter to edges where both are train
    mask = np.isin(tr_dst, train_indices)
    if mask.any():
        u, v = tr_src[mask], tr_dst[mask]
        np.add.at(compat_cts, ([lbl2i[y_pred[i]] for i in v], [lbl2i[y_pred[i]] for i in u]), 1)
    compat = compat_cts / compat_cts.sum(axis=0, keepdims=True)

    # Nbr Counts
    nbr_cts = np.zeros((N, C))
    for u in train_indices:
        c = lbl2i[y_pred[u]]
        for v in neighbors[u]: nbr_cts[v, c] += 1
            
    # Sim Matrix
    D = np.zeros((N, C))
    def update_D():
        with torch.no_grad():
            for c_idx, lbl in enumerate(all_labels):
                if not class_insts[lbl]: continue
                idx_t = torch.tensor(list(class_insts[lbl]), device=device)
                # Weighted centroid: 1-a7 for train, a7 for predicted
                w = torch.where(train_mask_t[idx_t], 1.0-a7, a7)
                cent = (xz[idx_t] * w[:,None]).sum(0, keepdim=True) / w.sum()
                D[:, c_idx] = ((F.cosine_similarity(xz, cent) + 1)/2).cpu().numpy()
    update_D()

    # 5. Priority Queue
    pq = []
    def calc_prio(u):
        # WScore: Label prop pressure
        l_cnt = nbr_cts[u].sum()
        w = 0.0
        if l_cnt > 0:
            # Simple heuristic: Fraction of labeled neighbors
            w = l_cnt / max(deg[u], 1)
        
        # SScore: Similarity + MLP confidence
        s = D[u].max()
        if mlp_probs_np is not None: s += mlp_probs_np[u].max()
        
        return -(w*a4 + s), u

    for u in range(N):
        if y_pred[u] == -1: heapq.heappush(pq, calc_prio(u))

    # 6. Propagation Loop
    labeled_new = 0
    
    # DEBUG: Stats accumulators
    debug_stats = defaultdict(list)
    
    while pq:
        _, u = heapq.heappop(pq)
        if y_pred[u] != -1: continue
        
        # Calculate Term Scores
        # 1. Prior
        s_prior = a1 * prior
        
        # 2. Neighbors & Compatibility scaling
        # Lower threshold for full weight (2 instead of 6) to activate structure early
        labeled_total = nbr_cts[u].sum()
        r = min(labeled_total / float(max(min_labeled_neighbors_for_full_weight, 1)), 1.0)
        
        # Neighbors
        if deg[u] > 0: s_neigh = (a2 * r) * (nbr_cts[u] / deg[u])
        else: s_neigh = np.zeros(C)
        
        # Compatibility
        if labeled_total > 0:
            dist = nbr_cts[u] / labeled_total
            s_compat = (a8 * r) * (dist @ compat)
        else:
            s_compat = np.full(C, (a8 * r) / C)
            
        # 3. Similarity
        s_sim = a3 * D[u]

        final_scores = s_prior + s_neigh + s_sim + s_compat
        
        # Decision
        best = np.argmax(final_scores)
        lbl = lbl_arr[best]
        y_pred[u] = lbl
        class_insts[lbl].add(u)
        labeled_new += 1
        
        # Update Neighbors
        for v in neighbors[u]: nbr_cts[v, best] += 1
            
        # Logging Sample (First 5 decisions)
        if log_file and labeled_new <= 5:
            debug_stats['prior'].append(abs(s_prior[best]))
            debug_stats['neigh'].append(abs(s_neigh[best]))
            debug_stats['sim'].append(abs(s_sim[best]))
            debug_stats['compat'].append(abs(s_compat[best]))

        # Refresh
        if labeled_new % max(N//5, 1) == 0:
            update_D()
            pq = [] # Rebuild PQ lazy
            for x in range(N):
                if y_pred[x] == -1: heapq.heappush(pq, calc_prio(x))
                
    # Final Logs
    if log_file:
        p = np.mean(debug_stats['prior']) if debug_stats['prior'] else 0
        n = np.mean(debug_stats['neigh']) if debug_stats['neigh'] else 0
        s = np.mean(debug_stats['sim']) if debug_stats['sim'] else 0
        c = np.mean(debug_stats['compat']) if debug_stats['compat'] else 0
        write_log(f"Avg Term Magnitudes (First 5): Prior={p:.3f}, Neigh={n:.3f}, Sim={s:.3f}, Compat={c:.3f}", log_file, tag=f"{log_prefix}:DEBUG")
        
    test_acc = _simple_accuracy(y_true[test_indices], y_pred[test_indices])
    if log_file:
        write_log(f"Labeled {labeled_new} nodes. Test Acc: {test_acc:.4f}", log_file, tag=f"{log_prefix}:DONE")
        
    return torch.tensor(y_pred[test_indices], device=device), test_acc

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
        write_log(f"Starting MLP training for {epochs} epochs...", log_file, tag="MLP")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x)
        loss = loss_fn(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        
        if log_file and (epoch + 1) % 50 == 0:
            write_log(f"Epoch {epoch+1:03d} | Loss: {loss.item():.4f}", log_file, tag="MLP")
        
    model.eval()
    with torch.no_grad():
        logits = model(data.x)
        probs = F.softmax(logits, dim=1)
        
    return probs, model

# =====================================================
# BLOCK 5: Main Execution Loop
# =====================================================

# ---- LOG CONFIG (MODIFIED) ----
# Using time-stamped log file with 'mlp+pred' in the name
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
timestamp_str = time.strftime('%Y%m%d_%H%M%S')
LOG_FILE = os.path.join(LOG_DIR, f"{DATASET_KEY.lower()}_mlp+pred_{timestamp_str}.txt")

write_log(f"[main] starting experiment for DATASET_KEY={DATASET_KEY}", LOG_FILE, tag="RUN")

num_iterations = 10 
EXPERIMENT_REPEATS = 5 # Number of times to repeat experiment per split
CONFIDENCE_THRESHOLD = 0.85 
RS_NUM_SAMPLES = 50 
RS_SPLITS = 3
RS_REPEATS = 1

final_test_accuracies = []
strategies_chosen = []
results_summary = {'val_mlp': [], 'val_refined': [], 'test_mlp': [], 'test_refined': [], 'test_chosen': []}

num_classes = int(data.y.max().item()) + 1
device = data.x.device
N_total = int(data.num_nodes)

for run in range(num_iterations):
    run_seed = 1337 + run
    
    write_log(f"{'='*20} SPLIT {run} {'='*20}", LOG_FILE)

    # 1. Load Split
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

    log_kv(LOG_FILE, tag="SPLIT", run=run, n_train=len(train_np), n_val=len(val_np))

    # Storage for 5 repeats on this split
    split_results = {'val_mlp': [], 'val_ref': [], 'test_mlp': [], 'test_ref': [], 'test_final': [], 'strat': []}

    for repeat in range(EXPERIMENT_REPEATS):
        write_log(f"-- Repeat {repeat+1}/{EXPERIMENT_REPEATS} --", LOG_FILE)
        
        # Different seed for initialization/tuning per repeat
        current_seed = run_seed * 100 + repeat 
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)
        
        # 2. Train MLP
        mlp_probs, _ = train_mlp_and_predict(data, train_idx, hidden=64, layers=2, dropout=0.5, epochs=200, log_file=LOG_FILE)
        max_probs, preds_mlp_all = mlp_probs.max(dim=1)

        # 3. Tuning
        tuned_params, _, _ = robust_random_search(
            predictclass, data, train_np,
            num_samples=RS_NUM_SAMPLES, n_splits=RS_SPLITS, repeats=RS_REPEATS,
            seed=current_seed, log_file=LOG_FILE, mlp_probs=mlp_probs
        )
        
        best_params = {
            'a1': tuned_params[0], 'a2': tuned_params[1], 'a3': tuned_params[2], 
            'a4': tuned_params[3], 'a5': tuned_params[4], 'a6': tuned_params[5], 
            'a7': tuned_params[6], 'a8': tuned_params[7]
        }

        # 4. Validation
        acc_val_mlp = (preds_mlp_all[val_idx] == data.y[val_idx]).float().mean().item()
        
        is_unlabeled = torch.ones(N_total, dtype=torch.bool, device=device)
        is_unlabeled[train_idx] = False
        is_high_conf = (max_probs >= CONFIDENCE_THRESHOLD) & is_unlabeled
        
        # Validation Pseudo-Labels (Strict Leakage Check: NO Val nodes)
        mask_val_support = is_high_conf.clone()
        mask_val_support[val_idx] = False
        pseudo_idx_val = torch.where(mask_val_support)[0].cpu().numpy()
        
        # LOGGING ENABLED: Pseudo-label stats
        if len(pseudo_idx_val) > 0:
            avg_conf = max_probs[pseudo_idx_val].mean().item()
            write_log(f"Pseudo-Val: Added {len(pseudo_idx_val)} nodes. Avg Conf: {avg_conf:.4f}", LOG_FILE, tag="PSEUDO")
        else:
            write_log("Pseudo-Val: No support nodes found.", LOG_FILE, tag="PSEUDO")
        
        aug_train_val = np.concatenate([train_np, pseudo_idx_val])
        orig_y = data.y.clone()
        data.y[pseudo_idx_val] = preds_mlp_all[pseudo_idx_val]
        
        try:
            write_log("Running Validation Refinement...", LOG_FILE, tag="VAL-REF")
            _, acc_val_ref = predictclass(
                data, aug_train_val, val_np, **best_params, seed=current_seed,
                log_file=LOG_FILE, mlp_probs=mlp_probs
            )
            acc_val_ref = float(acc_val_ref)
        finally:
            data.y = orig_y.clone()

        use_ref = acc_val_ref > acc_val_mlp
        decision_str = 'Refined' if use_ref else 'MLP'
        write_log(f"Validation Result: MLP={acc_val_mlp:.4f} vs Refined={acc_val_ref:.4f} (Delta={acc_val_ref-acc_val_mlp:+.4f}) -> Chosen: {decision_str}", LOG_FILE, tag="DECISION")
        
        split_results['val_mlp'].append(acc_val_mlp)
        split_results['val_ref'].append(acc_val_ref)
        split_results['strat'].append(1 if use_ref else 0)

        # 5. Test
        acc_test_mlp = (preds_mlp_all[test_idx] == data.y[test_idx]).float().mean().item()
        
        # Test Pseudo-Labels (Strict Leakage Check: NO Test nodes)
        mask_test_support = is_high_conf.clone()
        mask_test_support[test_idx] = False
        pseudo_idx_test = torch.where(mask_test_support)[0].cpu().numpy()
        
        aug_train_test = np.concatenate([train_np, pseudo_idx_test])
        data.y[pseudo_idx_test] = preds_mlp_all[pseudo_idx_test]
        
        try:
            write_log("Running Test Refinement...", LOG_FILE, tag="TEST-REF")
            _, acc_test_ref = predictclass(
                data, aug_train_test, test_np, **best_params, seed=current_seed,
                log_file=LOG_FILE, mlp_probs=mlp_probs
            )
            acc_test_ref = float(acc_test_ref)
        finally:
            data.y = orig_y.clone()

        final_acc = acc_test_ref if use_ref else acc_test_mlp
        
        split_results['test_mlp'].append(acc_test_mlp)
        split_results['test_ref'].append(acc_test_ref)
        split_results['test_final'].append(final_acc)
        
        write_log(f"Test Result: MLP={acc_test_mlp:.4f} vs Refined={acc_test_ref:.4f} (Delta={acc_test_ref-acc_test_mlp:+.4f})", LOG_FILE, tag="TEST")

    # Average over 5 repeats
    avg_test_final = np.mean(split_results['test_final'])
    ref_usage = np.sum(split_results['strat'])
    
    final_test_accuracies.append(avg_test_final)
    strategies_chosen.extend(["Refined" if s else "MLP" for s in split_results['strat']])
    
    results_summary['val_mlp'].append(np.mean(split_results['val_mlp']))
    results_summary['val_refined'].append(np.mean(split_results['val_ref']))
    results_summary['test_mlp'].append(np.mean(split_results['test_mlp']))
    results_summary['test_refined'].append(np.mean(split_results['test_ref']))
    results_summary['test_chosen'].append(avg_test_final)

    write_log(f"Split {run} Avg Test Acc: {avg_test_final:.4f} (Refined used {ref_usage}/{EXPERIMENT_REPEATS})", LOG_FILE, tag="SPLIT-SUMMARY")

# Final Summary
all_test_scores = np.array(final_test_accuracies)
avg_acc = np.mean(all_test_scores)
std_acc = np.std(all_test_scores)
total_runs = num_iterations * EXPERIMENT_REPEATS
total_refined = strategies_chosen.count("Refined")

summary_msg = "\n" + "="*50 + f"\n FINAL SUMMARY ({DATASET_KEY})\n" + "="*50 + "\n"
summary_msg += f"Test Accuracy: {avg_acc:.4f} ± {std_acc:.4f}\n"
summary_msg += f"Strategy Usage: Refined={total_refined}/{total_runs} ({(total_refined/total_runs)*100:.1f}%)\n"
summary_msg += "-" * 30 + "\n"
summary_msg += f"MLP Baseline (Avg): {np.mean(results_summary['test_mlp']):.4f}\n"
summary_msg += f"Refined Only (Avg): {np.mean(results_summary['test_refined']):.4f}\n"

write_log(summary_msg, LOG_FILE, tag="SUMMARY")
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





#!/usr/bin/env python
# coding: utf-8

# In[17]:


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

def predictclass(
    data, train_indices, test_indices,
    a1, a2, a3, a4, a5, a6, a7, a8,
    seed: int = 1337,
    log_prefix: str = "predictclass",
    log_every: int | None = None,
    log_file: str | None = None,
    mlp_probs: Optional[torch.Tensor] = None, 
    min_labeled_neighbors_for_full_weight: int = 2, 
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
    
    # Updated: Centroids Weighted by Certainty
    def update_D_weighted(D_matrix, c_insts, all_lbls, dev, conf_vec, x_z):
        with torch.no_grad():
            conf_t = torch.from_numpy(conf_vec).to(dev).float()
            
            for c_idx, lbl in enumerate(all_lbls):
                if not c_insts[lbl]: continue
                # Indices of nodes currently assigned to class lbl
                idx_t = torch.tensor(list(c_insts[lbl]), device=dev)
                
                # Weights: Use the tracking confidence vector
                w = conf_t[idx_t]
                w_sum = w.sum()
                if w_sum < 1e-9: continue
                
                # Weighted Centroid
                cent = (x_z[idx_t] * w[:,None]).sum(0, keepdim=True) / w_sum
                
                # Cosine Similarity
                D_matrix[:, c_idx] = ((F.cosine_similarity(x_z, cent) + 1)/2).cpu().numpy()
                
    update_D_weighted(D, class_insts, lbl_arr, device, node_state_conf, xz)

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
        else:
            s_neigh = np.zeros(C)
            r = 0.0
            
        # Compatibility (Vectorized Soft)
        total_mass = nbr_dist[u].sum()
        if total_mass > 1e-6:
            norm_dist = nbr_dist[u] / total_mass
            s_compat = (a8 * r) * (norm_dist @ compat)
        else:
            s_compat = np.full(C, (a8 * r) / C)
            
        # 3. Similarity (Feature Vector to Weighted Centroid)
        # User Req: "predict class also should have one more parameter" -> a3 handles this
        s_sim = a3 * D[u]

        final_scores = s_prior + s_neigh + s_sim + s_compat
        
        best = np.argmax(final_scores)
        lbl = lbl_arr[best]
        y_pred[u] = lbl
        class_insts[lbl].add(u)
        labeled_new += 1
        
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
            update_D_weighted(D, class_insts, lbl_arr, device, node_state_conf, xz)
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

results = {
    'mlp_only': [],
    'prop_only': [],
    'mlp_refined': [],
    'prop_mlp_selected': [],
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
        mlp_probs, _ = train_mlp_and_predict(data, train_idx, hidden=64, layers=2, dropout=0.5, epochs=200, log_file=LOG_FILE)
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





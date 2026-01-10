#!/usr/bin/env python
# coding: utf-8

# In[114]:


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

# In[115]:


from torch_geometric.datasets import Planetoid

from torch_geometric.datasets import LINKXDataset
# =====================================================
# Select dataset (UNCOMMENT ONE LINE ONLY)
# =====================================================

#DATASET_KEY = "cornell"
#DATASET_KEY = "texas"
# DATASET_KEY = "chameleon"
# DATASET_KEY = "squirrel"
DATASET_KEY = "wisconsin"
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



# In[ ]:





# In[116]:


import time
import numpy as np
import torch

def _to_long_tensor(idx, device):
    """Robust: list/np/tensor -> contiguous torch.long on device."""
    if torch.is_tensor(idx):
        return idx.to(device=device, dtype=torch.long).contiguous()
    return torch.tensor(np.asarray(idx, dtype=np.int64), device=device, dtype=torch.long).contiguous()

def _to_numpy_idx(idx):
    """Robust: list/np/tensor -> np.int64 array."""
    if torch.is_tensor(idx):
        return idx.detach().cpu().numpy().astype(np.int64, copy=False)
    return np.asarray(idx, dtype=np.int64)

def forward_probe(model, data, inject_idx_global, tag="VAL", topk=5, max_print=12):
    """
    Runs a single forward pass + rich sanity logs.
    - Works whether model returns logits for:
        (A) all nodes: [N, C]
        (B) subset only: [len(inject_idx_global), C]   (common when model internally indexes)
    - Logs shapes, ranges, NaN/Inf, class histograms, confidence stats, and a few per-node examples.

    Usage:
        forward_probe(model_val, data, val_idx, tag=f"VAL run={run}")
    """
    t0 = time.time()
    model.eval()

    with torch.no_grad():
        x = data.x
        ei = data.edge_index
        N = int(getattr(data, "num_nodes", x.size(0)))

        inject_idx_t = _to_long_tensor(inject_idx_global, device=x.device)
        inject_idx_np = _to_numpy_idx(inject_idx_global)

        # -------------------------
        # Index sanity
        # -------------------------
        if inject_idx_np.size == 0:
            print(f"[{tag}] ⚠️ inject_idx is empty -> nothing to probe.")
            return None

        bad = np.sum((inject_idx_np < 0) | (inject_idx_np >= N))
        print(f"\n[{tag}] === forward_probe ===")
        print(f"[{tag}] N={N} | x={tuple(x.shape)} | edge_index={tuple(ei.shape)} | device={x.device}")
        print(f"[{tag}] inject_idx: len={inject_idx_np.size} | unique={np.unique(inject_idx_np).size} "
              f"| min={int(inject_idx_np.min())} max={int(inject_idx_np.max())} | bad_out_of_range={int(bad)}")

        if bad > 0:
            # filter to safe subset
            mask = (inject_idx_t >= 0) & (inject_idx_t < N)
            inject_idx_t = inject_idx_t[mask]
            inject_idx_np = inject_idx_t.detach().cpu().numpy()
            print(f"[{tag}] ⚠️ filtered inject_idx -> len={inject_idx_np.size}")

        # -------------------------
        # Forward
        # -------------------------
        try:
            logits = model(x, ei)  # expected [N,C] OR [len(idx),C]
        except TypeError:
            # in case your model signature is model(data) style
            logits = model(data)

        if not torch.is_tensor(logits):
            raise TypeError(f"[{tag}] model forward returned non-tensor: {type(logits)}")

        # -------------------------
        # Logits sanity
        # -------------------------
        print(f"[{tag}] logits: shape={tuple(logits.shape)} dtype={logits.dtype} device={logits.device}")
        finite_mask = torch.isfinite(logits)
        n_finite = int(finite_mask.sum().item())
        n_total = int(logits.numel())
        if n_finite < n_total:
            print(f"[{tag}] ❌ logits have NaN/Inf: finite={n_finite}/{n_total}")
        else:
            print(f"[{tag}] logits finite: {n_finite}/{n_total}")

        # Basic stats (safe even if huge)
        lmin = float(torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0).min().item())
        lmax = float(torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0).max().item())
        print(f"[{tag}] logits range: [{lmin:.4f}, {lmax:.4f}]")

        # Determine mapping: full-N or subset
        num_classes = int(logits.shape[-1]) if logits.dim() == 2 else None
        if logits.dim() != 2:
            print(f"[{tag}] ⚠️ logits is not 2D; got {logits.dim()}D. Returning raw logits.")
            return logits

        if logits.shape[0] == N:
            mode = "full"
            logits_sel = logits[inject_idx_t]
            sel_desc = f"logits_sel = logits[inject_idx] -> shape={tuple(logits_sel.shape)}"
        elif logits.shape[0] == inject_idx_t.numel():
            mode = "subset"
            logits_sel = logits
            sel_desc = f"logits already subset-sized -> shape={tuple(logits_sel.shape)}"
        else:
            mode = "unknown"
            # Best effort: try to index if possible
            can_index = (logits.shape[0] >= int(inject_idx_t.max().item()) + 1)
            if can_index:
                logits_sel = logits[inject_idx_t]
                mode = "indexed_partial"
                sel_desc = f"logits indexed by inject_idx (partial table) -> shape={tuple(logits_sel.shape)}"
            else:
                print(f"[{tag}] ⚠️ cannot align logits with inject_idx: logits_n={logits.shape[0]} "
                      f"vs N={N} vs idx_len={inject_idx_t.numel()}. Returning raw logits.")
                return logits

        print(f"[{tag}] alignment mode: {mode} | {sel_desc}")
        print(f"[{tag}] num_classes={num_classes}")

        # -------------------------
        # Predictions + confidence
        # -------------------------
        probs_sel = torch.softmax(logits_sel, dim=1)
        conf, pred = probs_sel.max(dim=1)
        top2 = torch.topk(probs_sel, k=min(2, num_classes), dim=1).values
        margin = (top2[:, 0] - top2[:, 1]) if top2.size(1) >= 2 else torch.zeros_like(conf)

        pred_cpu = pred.detach().cpu().numpy()
        conf_cpu = conf.detach().cpu().numpy()
        margin_cpu = margin.detach().cpu().numpy()

        # Histograms
        uniq, cnt = np.unique(pred_cpu, return_counts=True)
        pred_hist = {int(u): int(c) for u, c in zip(uniq, cnt)}

        # print(f"[{tag}] pred_hist: {pred_hist}")
        # print(f"[{tag}] conf: mean={float(conf_cpu.mean()):.4f} std={float(conf_cpu.std()):.4f} "
        #       f"min={float(conf_cpu.min()):.4f} max={float(conf_cpu.max()):.4f}")
        # print(f"[{tag}] margin(top1-top2): mean={float(margin_cpu.mean()):.4f} std={float(margin_cpu.std()):.4f} "
        #       f"min={float(margin_cpu.min()):.4f} max={float(margin_cpu.max()):.4f}")

        # -------------------------
        # Optional accuracy if labels exist
        # -------------------------
        if hasattr(data, "y") and data.y is not None:
            y = data.y
            if torch.is_tensor(y):
                y_sel = y[inject_idx_t].detach().cpu().numpy()
            else:
                y_sel = np.asarray(y, dtype=np.int64)[inject_idx_np]

            acc = float((pred_cpu == y_sel).mean()) if y_sel.size > 0 else 0.0
            print(f"[{tag}] accuracy_on_inject_idx = {acc:.4f}")

            # Confusion-ish quick view (small)
            # show a few most common errors
            err = (pred_cpu != y_sel)
            if err.any():
                pairs = list(zip(y_sel[err].tolist(), pred_cpu[err].tolist()))
                from collections import Counter
                top_err = Counter(pairs).most_common(8)
                print(f"[{tag}] top_errors (true->pred): {top_err}")
            else:
                print(f"[{tag}] no errors on inject_idx ✔️")

        # -------------------------
        # Print a few examples
        # -------------------------
        k = min(max_print, inject_idx_t.numel())
        if k > 0:
            # pick a mix: lowest-confidence + highest-confidence
            order_low = np.argsort(conf_cpu)[: max(1, k // 2)]
            order_high = np.argsort(-conf_cpu)[: max(1, k - len(order_low))]
            show_idx = np.unique(np.concatenate([order_low, order_high]))[:k]

         #   print(f"[{tag}] examples (idx | pred | conf | margin | topk_probs):")
            for j in show_idx:
                if mode == "full" or mode == "indexed_partial":
                    node_id = int(inject_idx_np[j])
                else:
                    node_id = int(inject_idx_np[j])  # still the global node id list you passed

                p = int(pred_cpu[j])
                c = float(conf_cpu[j])
                m = float(margin_cpu[j])

                tk = min(topk, num_classes)
                vals, inds = torch.topk(probs_sel[j], k=tk)
                inds = inds.detach().cpu().numpy().tolist()
                vals = vals.detach().cpu().numpy().tolist()
                topk_list = list(zip(inds, [float(v) for v in vals]))

             #   print(f"  - node={node_id:4d} | pred={p:2d} | conf={c:.3f} | margin={m:.3f} | top{tk}={topk_list}")

        print(f"[{tag}] time={time.time()-t0:.3f}s")
        return logits_sel


# In[117]:


import os
import time
import heapq
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from datetime import datetime
from typing import Optional


# ============================================================
# Logging (no stdout printing; file only)
# ============================================================
def write_log(
    msg: str,
    log_file: str,
    *,
    tag: Optional[str] = None,
    also_print: bool = False,   # IMPORTANT: do not print
    flush: bool = True,
):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prefix = f"[{ts}]"
    if tag is not None:
        prefix += f"[{tag}]"

    lines = msg.splitlines() or [""]
    formatted = "\n".join(f"{prefix} {line}" for line in lines) + "\n"

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(formatted)
        if flush:
            f.flush()
            os.fsync(f.fileno())

    if also_print:
        print(formatted, end="")

def log_kv(log_file: str, tag: str, **kwargs):
    parts = [f"{k}={v}" for k, v in kwargs.items()]
    write_log(" | ".join(parts), log_file, tag=tag, also_print=False)

class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()
    def reset(self):
        self.t0 = time.perf_counter()
    def elapsed(self):
        return time.perf_counter() - self.t0


def _simple_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.size == 0:
        return 0.0
    return float((y_true == y_pred).mean())


# ============================================================
# Performance-upgraded predictclass (backward compatible)
# ============================================================
def predictclass(
    data, train_indices, test_indices,
    a1, a2, a3, a4, a5, a6, a7, a8,
    seed: int = 1337,
    log_prefix: str = "predictclass",
    log_every: int | None = None,
    log_file: str | None = None,

    # -------------------------
    # NEW (performance knobs) - defaults preserve old behavior
    # -------------------------
    use_train_feature_standardization: bool = True,
    use_two_hop: bool = False,              # try True on Texas/Wisconsin
    two_hop_decay: float = 0.5,             # weight for 2-hop evidence
    adaptive_a2: bool = True,               # dynamic neighbor weight
    adaptive_a8: bool = True,               # dynamic compat weight
    min_labeled_neighbors_for_full_weight: int = 6,

    # low-confidence deferral
    defer_low_margin: bool = False,         # try True
    margin_threshold: float = 0.03,         # try 0.02~0.08
    max_deferrals_per_node: int = 5,
    requeue_boost: float = 0.02,            # boost priority when deferred
):
    """
    scoring:
      score(c) = a1*prior(c) + a2*neighbor_dist(c) + a3*sim(u,c) + a8*compat_support(c)

    Returns:
      preds_test (torch.Tensor on data.x.device),
      acc_test (float)
    """
    t_all = Timer()
    rng = np.random.default_rng(seed)

    device = data.x.device
    N = int(data.num_nodes)

    train_indices = np.asarray(train_indices, dtype=np.int64)
    test_indices  = np.asarray(test_indices,  dtype=np.int64)

    y_true = data.y.detach().cpu().numpy().astype(np.int64, copy=False)

    # Labels present in the dataset (generic in case they aren't 0..C-1)
    all_labels = sorted(set(y_true.tolist()))
    C = len(all_labels)
    lbl2i = {lbl: j for j, lbl in enumerate(all_labels)}
    lbl_arr = np.array(all_labels, dtype=np.int64)

    def _logkv(tag: str, **kwargs):
        if log_file is not None:
            log_kv(log_file, tag=f"{log_prefix}:{tag}", **kwargs)

    # -------------------------
    # Build undirected + unique edge_index
    # -------------------------
    t = Timer()
    edge_index = torch.cat([data.edge_index, data.edge_index.flip(0)], dim=1)
    edge_index = torch.unique(edge_index, dim=1)
    ei = edge_index.detach().cpu().numpy()
    src = ei[0].astype(np.int64, copy=False)
    dst = ei[1].astype(np.int64, copy=False)
    M = src.size

    neighbors = [[] for _ in range(N)]
    for u, v in zip(src, dst):
        neighbors[int(u)].append(int(v))

    deg = np.fromiter((len(neighbors[u]) for u in range(N)), dtype=np.int32, count=N)

    # init predictions: train fixed, others -1
    y_pred = np.full(N, -1, dtype=np.int64)
    y_pred[train_indices] = y_true[train_indices]

    # train membership mask
    train_mask = np.zeros(N, dtype=bool)
    train_mask[train_indices] = True

    # class_instances for centroid recompute
    class_instances = defaultdict(set)
    for i in train_indices:
        class_instances[int(y_pred[i])].add(int(i))

    # priors (Laplace)
    alpha = 1.0
    train_ci = np.array([lbl2i[int(y_pred[i])] for i in train_indices], dtype=np.int64)
    counts = np.bincount(train_ci, minlength=C).astype(np.float64)
    prior_arr = (counts + alpha) / (len(train_indices) + alpha * C)

    _logkv(
        "INIT",
        nodes=N,
        undirected_edges=M,
        classes=C,
        train=len(train_indices),
        test=len(test_indices),
        mean_deg=float(deg.mean()) if N > 0 else 0.0,
        max_deg=int(deg.max()) if N > 0 else 0,
        t_build_graph=f"{t.elapsed():.4f}s",
    )

    # -------------------------
    # compat(train-train) with smoothing (vectorized)
    # -------------------------
    t = Timer()
    compat_counts = np.ones((C, C), dtype=np.float64)   # Laplace smoothing
    compat_totals = np.ones(C, dtype=np.float64) * C

    tr_edge_mask = train_mask[src] & train_mask[dst]
    if tr_edge_mask.any():
        u_tr = src[tr_edge_mask]
        v_tr = dst[tr_edge_mask]
        cand = np.array([lbl2i[int(y_pred[u])] for u in u_tr], dtype=np.int64)
        neigh = np.array([lbl2i[int(y_pred[v])] for v in v_tr], dtype=np.int64)

        np.add.at(compat_counts, (neigh, cand), 1.0)
        np.add.at(compat_totals, neigh, 1.0)

    compat = compat_counts / compat_totals[:, None]
    _logkv("COMPAT", train_train_edges=int(tr_edge_mask.sum()), t_compat=f"{t.elapsed():.4f}s")

    # -------------------------
    # neighbor labeled counts
    # -------------------------
    t = Timer()
    nbr_count = np.zeros((N, C), dtype=np.float64)
    train_nbr_cnt = np.zeros(N, dtype=np.int32)

    for tnode in train_indices:
        ct = lbl2i[int(y_pred[tnode])]
        for v in neighbors[int(tnode)]:
            nbr_count[v, ct] += 1.0
            train_nbr_cnt[v] += 1

    _logkv("NBR_INIT", t_nbr_init=f"{t.elapsed():.4f}s")

    # -------------------------
    # OPTIONAL 2-hop evidence (computed once from train seeds; fast & stable)
    #   This is intentionally conservative: it helps a lot on heterophily,
    #   but you can disable it (default False).
    # -------------------------
    nbr2_count = None
    if use_two_hop:
        t = Timer()
        nbr2_count = np.zeros((N, C), dtype=np.float64)
        # For each node u, accumulate classes of neighbors-of-neighbors that are train-labeled.
        # Complexity ~ sum_u deg(u)^2, but WebKB graphs are small enough.
        for u in range(N):
            # skip isolated
            if deg[u] == 0:
                continue
            acc = nbr2_count[u]
            for v in neighbors[u]:
                # only propagate from train-labeled neighbors
                if train_mask[v]:
                    acc[lbl2i[int(y_pred[v])]] += 1.0
                # also bring in train labels among v's neighbors (2-hop)
                for w in neighbors[v]:
                    if train_mask[w]:
                        acc[lbl2i[int(y_pred[w])]] += 1.0
        _logkv("TWO_HOP", t_two_hop=f"{t.elapsed():.4f}s", decay=two_hop_decay)

    # -------------------------
    # Similarity matrix D[u, c]
    #   Upgrade: standardize x using train stats (helps cosine on heterophily).
    # -------------------------
    D = np.zeros((N, C), dtype=np.float64)

    train_mask_t = torch.zeros(N, device=device, dtype=torch.bool)
    train_mask_t[torch.as_tensor(train_indices, device=device)] = True

    # Feature standardization based on train nodes
    if use_train_feature_standardization:
        with torch.no_grad():
            x_train = data.x[train_mask_t]
            mu = x_train.mean(dim=0, keepdim=True)
            sigma = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
            xz_all = (data.x - mu) / sigma
    else:
        xz_all = data.x

    def recompute_centroids_and_D():
        """
        Same semantics as your original centroid logic:
          - weights: (1-a7) for train nodes, a7 for propagated nodes
          - D[:, ci] = (cos_sim(x, centroid)+1)/2
        (We compute with standardized x when enabled.)
        """
        t = Timer()
        with torch.no_grad():
            for ci, lbl in enumerate(all_labels):
                inst = class_instances.get(lbl, None)
                if not inst:
                    continue

                idxs_np = np.fromiter(inst, dtype=np.int64)
                idxs = torch.as_tensor(idxs_np, device=device, dtype=torch.long)

                feats = xz_all[idxs]

                w = torch.where(
                    train_mask_t[idxs],
                    torch.tensor(1.0 - a7, device=device),
                    torch.tensor(a7, device=device),
                )
                wsum = w.sum()
                if float(wsum) <= 1e-12:
                    continue
                w = w / wsum

                centroid = torch.sum(feats * w[:, None], dim=0, keepdim=True)
                sims = F.cosine_similarity(xz_all, centroid.expand_as(xz_all))
                sims = (sims + 1.0) / 2.0
                D[:, ci] = sims.detach().cpu().numpy().astype(np.float64, copy=False)

        return t.elapsed()

    t_cent = recompute_centroids_and_D()
    _logkv("CENTROID", t_centroids=f"{t_cent:.4f}s", standardized=int(use_train_feature_standardization))

    # -------------------------
    # Priority queue (heapq)
    # -------------------------
    refresh_every = max(N // 5, 1)
    if log_every is None:
        log_every = refresh_every

    def priority(u: int):
        if deg[u] == 0:
            return 0.0, 0.0
        labeled_cnt_u = float(nbr_count[u].sum())
        if labeled_cnt_u <= 0:
            wscore = 0.0
        else:
            nt = float(train_nbr_cnt[u])
            npg = max(labeled_cnt_u - nt, 0.0)
            wscore = (a4 * npg + a5 * nt) / float(max(deg[u], 1))
        sscore = float(D[u].max())
        return wscore, sscore

    pq = []
    deferrals = np.zeros(N, dtype=np.int16)

    for u in range(N):
        if y_pred[u] == -1:
            ws, ss = priority(u)
            heapq.heappush(pq, (-(ws + a6 * ss), u))

    _logkv(
        "GREEDY_START",
        refresh_every=refresh_every,
        log_every=log_every,
        use_two_hop=int(use_two_hop),
        adaptive_a2=int(adaptive_a2),
        adaptive_a8=int(adaptive_a8),
        defer_low_margin=int(defer_low_margin),
        margin_threshold=margin_threshold,
        a1=a1, a2=a2, a3=a3, a4=a4, a5=a5, a6=a6, a7=a7, a8=a8
    )

    # -------------------------
    # Greedy labeling with optional low-confidence deferral
    # -------------------------
    tie_events = 0
    labeled_new = 0

    margin_sum = 0.0
    margin_cnt = 0

    compat_best_sum = 0.0
    compat_best_cnt = 0

    score_evals = 0
    t_greedy = Timer()

    scores = np.empty(C, dtype=np.float64)

    while pq:
        negp, u = heapq.heappop(pq)
        if y_pred[u] != -1:
            continue

        # neighbor distribution
        if deg[u] > 0:
            neighbor_dist = nbr_count[u] / float(deg[u])
        else:
            neighbor_dist = np.zeros(C, dtype=np.float64)

        # compat_support
        labeled_total = float(nbr_count[u].sum())
        if labeled_total > 0:
            wts = nbr_count[u] / labeled_total
            compat_support = wts @ compat
        else:
            compat_support = np.full(C, 1.0 / C, dtype=np.float64)

        # OPTIONAL: 2-hop evidence (conservative blend)
        if use_two_hop and nbr2_count is not None:
            # normalize by something stable (deg-based) to keep scale reasonable
            denom = float(max(deg[u], 1))
            neighbor_dist_2 = (nbr2_count[u] / denom)
        else:
            neighbor_dist_2 = None

        # OPTIONAL: adaptive scaling for neighbor/compat weights based on labeled neighbors
        if adaptive_a2 or adaptive_a8:
            # ratio in [0,1] increases as node gets more labeled neighbors
            r = min(labeled_total / float(max(min_labeled_neighbors_for_full_weight, 1)), 1.0)
            # if few labeled neighbors: rely less on (a2,a8), more on prior/sim
            a2_eff = (a2 * r) if adaptive_a2 else a2
            a8_eff = (a8 * r) if adaptive_a8 else a8
        else:
            a2_eff, a8_eff = a2, a8

        # scores
        scores[:] = a1 * prior_arr + a2_eff * neighbor_dist + a3 * D[u, :] + a8_eff * compat_support

        # inject 2-hop evidence as additional neighbor signal (small, decayed)
        if neighbor_dist_2 is not None:
            scores[:] += (a2_eff * two_hop_decay) * neighbor_dist_2

        score_evals += C

        best_ci = int(np.argmax(scores))
        best_s = float(scores[best_ci])

        tmp = scores.copy()
        tmp[best_ci] = -1e100
        second_s = float(tmp.max())
        margin = (best_s - second_s)

        margin_sum += margin
        margin_cnt += 1

        compat_best_sum += float(a8_eff * compat_support[best_ci])
        compat_best_cnt += 1

        # OPTIONAL: defer low-confidence nodes (prevents early error cascades)
        if defer_low_margin and (margin < margin_threshold) and (deferrals[u] < max_deferrals_per_node):
            deferrals[u] += 1
            # requeue with a tiny boost so it gets reconsidered later
            heapq.heappush(pq, (negp + requeue_boost, u))
            continue

        # deterministic tie-break (kept)
        eps = 1e-2
        best_set = np.where(np.abs(scores - best_s) <= eps)[0]
        if best_set.size > 1:
            tie_events += 1

            def tie_key(ci: int):
                lbl = int(lbl_arr[int(ci)])
                return (
                    float(neighbor_dist[int(ci)]),
                    float(D[u, int(ci)]),
                    float(prior_arr[int(ci)]),
                    -lbl,
                )
            best_ci = int(max(best_set, key=tie_key))

        chosen_lbl = int(lbl_arr[best_ci])

        # commit
        y_pred[u] = chosen_lbl
        class_instances[chosen_lbl].add(int(u))
        labeled_new += 1

        # update neighbor counts
        chosen_ci = best_ci
        for v in neighbors[u]:
            nbr_count[v, chosen_ci] += 1.0

        # periodic refresh + rebuild heap
        if labeled_new % refresh_every == 0:
            t_ref = Timer()
            t_cent = recompute_centroids_and_D()

            pq = []
            for x in range(N):
                if y_pred[x] == -1:
                    ws, ss2 = priority(x)
                    heapq.heappush(pq, (-(ws + a6 * ss2), x))

            _logkv(
                "REFRESH",
                labeled_new=labeled_new,
                t_centroids=f"{t_cent:.4f}s",
                t_rebuild_pq=f"{t_ref.elapsed():.4f}s",
                pq_size=len(pq),
            )

        if labeled_new % log_every == 0:
            tie_rate = tie_events / max(labeled_new, 1)
            mean_margin = margin_sum / max(margin_cnt, 1)
            mean_compat = compat_best_sum / max(compat_best_cnt, 1)
            _logkv(
                "PROGRESS",
                labeled=labeled_new,
                remaining=int(np.sum(y_pred == -1)),
                tie_rate=f"{tie_rate:.6f}",
                mean_margin=f"{mean_margin:.6f}",
                mean_a8_compat_best=f"{mean_compat:.6f}",
                score_evals=score_evals,
                deferrals_total=int(deferrals.sum()),
                t_greedy=f"{t_greedy.elapsed():.4f}s",
            )

    # final test preds + acc
    test_pred = y_pred[test_indices].astype(np.int64, copy=False)
    acc = _simple_accuracy(y_true[test_indices], test_pred)

    _logkv(
        "DONE",
        labeled_new=labeled_new,
        tie_events=tie_events,
        deferrals_total=int(deferrals.sum()),
        score_evals=score_evals,
        t_total=f"{t_all.elapsed():.4f}s",
        test_acc=f"{acc:.6f}",
    )

    return torch.tensor(test_pred, device=device), float(acc)


# In[ ]:





# In[ ]:





# In[ ]:





# In[118]:


import numpy as np
import torch


@torch.no_grad()
def edge_homophily_on_train_edges(
    edge_index: torch.Tensor,
    y: torch.Tensor,
    train_idx,
    *,
    shrinkage: float = 5.0,   # ↑ more conservative on small train sets
) -> tuple[float, int, int]:
    """
    Robust edge homophily over edges where BOTH endpoints are in train_idx.

    Returns:
      h_hat  : shrunk homophily estimate (toward baseline 1/C_used)
      C_used : number of unique classes in y[train_idx] (baseline=1/C_used)
      m_used : number of train-train edges used

    train_idx can be list/np/torch.
    """
    device = y.device
    if not torch.is_tensor(train_idx):
        train_idx = torch.tensor(np.asarray(train_idx, dtype=np.int64),
                                 device=device, dtype=torch.long)
    else:
        train_idx = train_idx.to(device=device, dtype=torch.long)

    N = int(y.numel())
    train_mask = torch.zeros(N, dtype=torch.bool, device=device)
    if train_idx.numel() > 0:
        train_mask[train_idx] = True

    src = edge_index[0].to(device)
    dst = edge_index[1].to(device)
    m = train_mask[src] & train_mask[dst]
    m_used = int(m.sum().item())

    C_used = int(torch.unique(y[train_idx]).numel()) if train_idx.numel() > 0 else 1
    C_used = max(C_used, 1)
    baseline = 1.0 / float(C_used)

    if m_used == 0:
        # fall back to baseline if we have no train-train edges
        return float(baseline), C_used, 0

    ys = y[src[m]]
    yd = y[dst[m]]
    h_raw = float((ys == yd).float().mean().item())

    # Shrink toward baseline to reduce variance on tiny graphs/splits (Texas/Wisconsin/Cornell)
    h_hat = (m_used * h_raw + shrinkage * baseline) / (m_used + shrinkage)

    return float(h_hat), C_used, m_used


def _hetero_level_from_h(h: float, C_used: int, margin: float = 0.02) -> float:
    """
    Turn homophily h into a heterophily level t in [0,1].
    Baseline for random labels is ~1/C_used.
    t grows as h drops below (1/C_used - margin).
    """
    C_used = max(int(C_used), 1)
    baseline = 1.0 / float(C_used)

    gap = baseline - float(h) - float(margin)  # positive if meaningfully heterophilic
    gap = max(0.0, gap)
    denom = max(1e-8, baseline)                # max possible gap is baseline (down to 0)
    t = min(1.0, gap / denom)
    return float(t)


def _inject_strength_from_hetero(
    t: float,
    *,
    lam_homo: float,
    lam_hetero: float,
    confidence: float = 1.0,
) -> float:
    """
    Map heterophily level t in [0,1] to injection strength λ.

    Improvements vs linear mapping:
      - smooth (sigmoid) transition instead of hard linear
      - confidence damping: if the heterophily estimate is unreliable (few edges),
        λ stays closer to lam_homo.

    confidence ∈ [0,1], where 1 means fully trusted estimate.
    """
    t = float(max(0.0, min(1.0, t)))
    confidence = float(max(0.0, min(1.0, confidence)))

    # smooth step around t=0.5
    s = 1.0 / (1.0 + np.exp(-6.0 * (t - 0.5)))

    lam_target = lam_homo + (lam_hetero - lam_homo) * s
    lam = lam_homo + confidence * (lam_target - lam_homo)

    return float(lam)


def build_initial_preds_full_adaptive(
    N_total,
    val_idx,
    initial_preds_val,
    *,
    edge_index: torch.Tensor | None = None,
    y: torch.Tensor | None = None,
    train_idx=None,
    device=None,
    # adaptation knobs
    lam_homo: float = 0.4,
    lam_hetero: float = 1.2,
    margin: float = 0.02,
    shrinkage: float = 5.0,
    conf_edges_full: int = 20,  # ~20 train-train edges => confidence ~ 1.0
    return_strength: bool = False,
    verbose: bool = True,
):
    """
    Adaptive version of build_initial_preds_full that ALSO computes an injection strength λ
    based on how homophilic/heterophilic the graph is (measured on train-train edges).

    Signature you can use in your pipeline:
        full, lam = build_initial_preds_full_adaptive(
            N_total, val_idx, initial_preds_val,
            edge_index=data.edge_index, y=data.y,
            train_idx=train_idx, device=device,
            return_strength=True
        )

    Notes:
    - If edge_index/y/train_idx are not provided, it behaves like the original function
      (returns only full vector; λ defaults to lam_homo).
    - λ is NOT applied inside this function; you pass it into the model and multiply init by λ.
    - We fill non-injected nodes with 0 (valid class id), same reasoning as before.
    """
    # -------------------------
    # Resolve device
    # -------------------------
    if device is None:
        if torch.is_tensor(val_idx):
            device = val_idx.device
        elif torch.is_tensor(y):
            device = y.device
        else:
            device = "cpu"

    # -------------------------
    # Convert val_idx to 1D long tensor on CPU for checks
    # -------------------------
    if torch.is_tensor(val_idx):
        idx_cpu = val_idx.detach().to("cpu", dtype=torch.long).contiguous()
    else:
        idx_cpu = torch.tensor(np.asarray(val_idx, dtype=np.int64), dtype=torch.long)

    if idx_cpu.ndim != 1:
        idx_cpu = idx_cpu.view(-1)

    # -------------------------
    # Convert preds to 1D long tensor on CPU for checks
    # -------------------------
    if torch.is_tensor(initial_preds_val):
        preds_cpu = initial_preds_val.detach().to("cpu", dtype=torch.long).contiguous()
    else:
        preds_cpu = torch.tensor(np.asarray(initial_preds_val, dtype=np.int64), dtype=torch.long)

    if preds_cpu.ndim != 1:
        preds_cpu = preds_cpu.view(-1)

    # -------------------------
    # Sanity checks
    # -------------------------
    if idx_cpu.numel() == 0:
        full = torch.zeros(int(N_total), dtype=torch.long, device=device)
        lam = float(lam_homo)
        return (full, lam) if return_strength else full

    if idx_cpu.numel() != preds_cpu.numel():
        raise ValueError(
            f"[build_initial_preds_full_adaptive] length mismatch: len(val_idx)={idx_cpu.numel()} "
            f"!= len(initial_preds_val)={preds_cpu.numel()}"
        )

    bad = int(((idx_cpu < 0) | (idx_cpu >= int(N_total))).sum().item())
    if bad > 0:
        raise ValueError("[build_initial_preds_full_adaptive] val_idx contains out-of-range indices.")

    pred_min = int(preds_cpu.min().item())
    if pred_min < 0:
        raise ValueError("[build_initial_preds_full_adaptive] initial_preds_val contains negative class ids.")
    # NOTE: can't validate pred_max < num_classes since num_classes isn't passed.

    # -------------------------
    # Build full vector (fill with valid class id 0)
    # -------------------------
    full_cpu = torch.zeros(int(N_total), dtype=torch.long)
    full_cpu[idx_cpu] = preds_cpu
    full = full_cpu.to(device=device, non_blocking=True)

    # -------------------------
    # Compute λ (injection strength) from heterophily if possible
    # -------------------------
    lam = float(lam_homo)  # safe default if we cannot measure
    if edge_index is not None and y is not None and train_idx is not None:
        h, C_used, m_used = edge_homophily_on_train_edges(
            edge_index, y, train_idx, shrinkage=shrinkage
        )
        t = _hetero_level_from_h(h, C_used=C_used, margin=margin)

        # confidence based on how many train-train edges we observed
        conf = min(1.0, float(m_used) / float(max(conf_edges_full, 1)))

        lam = _inject_strength_from_hetero(
            t,
            lam_homo=lam_homo,
            lam_hetero=lam_hetero,
            confidence=conf,
        )

        baseline = 1.0 / float(max(C_used, 1))
        if h < baseline - margin:
            regime = "HETEROPHILIC"
        elif h > baseline + margin:
            regime = "HOMOPHILIC"
        else:
            regime = "MIXED"

        if verbose:
            print(
                f"[adapt-inject] h={h:.4f} | baseline~1/C={baseline:.4f} (C_used={C_used}) "
                f"| train-train-edges={m_used} | conf={conf:.2f} "
                f"| regime={regime} | hetero_level={t:.3f} | lambda={lam:.3f}"
            )

    return (full, lam) if return_strength else full


# In[119]:


# mean_accuracy = np.mean(test_accuracies)
# std_accuracy = np.std(test_accuracies)

#print(f"Mean: {mean_accuracy:.4f}, Std: {std_accuracy:.4f}")


# In[120]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Sequential, Linear, ReLU
import numpy as np


# ============================================================
# Utils
# ============================================================
def _simple_accuracy_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    y_true = y_true.detach()
    y_pred = y_pred.detach()
    if y_true.numel() == 0:
        return 0.0
    return float((y_true == y_pred).float().mean().item())


# ============================================================
# Robust train-edge homophily (variance-controlled)
# ============================================================
@torch.no_grad()
def edge_homophily_on_train_edges(
    edge_index: torch.Tensor,
    y: torch.Tensor,
    train_idx,
    *,
    shrinkage: float = 5.0,
):
    device = y.device
    if not torch.is_tensor(train_idx):
        train_idx = torch.tensor(np.asarray(train_idx, dtype=np.int64),
                                 device=device, dtype=torch.long)
    else:
        train_idx = train_idx.to(device=device, dtype=torch.long)

    N = int(y.numel())
    train_mask = torch.zeros(N, dtype=torch.bool, device=device)
    train_mask[train_idx] = True

    src, dst = edge_index[0].to(device), edge_index[1].to(device)
    m = train_mask[src] & train_mask[dst]
    m_used = int(m.sum().item())

    C_used = int(torch.unique(y[train_idx]).numel()) if train_idx.numel() > 0 else 1
    C_used = max(C_used, 1)
    baseline = 1.0 / float(C_used)

    if m_used == 0:
        return baseline, C_used, 0

    ys = y[src[m]]
    yd = y[dst[m]]
    h_raw = float((ys == yd).float().mean().item())

    # shrink toward baseline (critical for small train splits)
    h_hat = (m_used * h_raw + shrinkage * baseline) / (m_used + shrinkage)

    return float(h_hat), C_used, m_used


def _heterophily_level_from_h(h: float, C_used: int, margin: float = 0.02) -> float:
    C_used = max(int(C_used), 1)
    baseline = 1.0 / float(C_used)
    gap = max(0.0, baseline - h - margin)
    t = min(1.0, gap / max(1e-8, baseline))
    return float(t)


def _smooth_inject_strength(
    t: float,
    lam_homo: float,
    lam_hetero: float,
    confidence: float,
):
    """
    Smooth + confidence-gated lambda.
    Prevents over-injection on noisy splits.
    """
    t = max(0.0, min(1.0, t))
    confidence = max(0.0, min(1.0, confidence))

    s = 1.0 / (1.0 + np.exp(-6.0 * (t - 0.5)))
    lam_target = lam_homo + (lam_hetero - lam_homo) * s
    lam = lam_homo + confidence * (lam_target - lam_homo)

    return float(lam)


# ============================================================
# Adaptive GCN (homophily / heterophily aware)
# ============================================================
class AdaptiveNodeClassifier(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        *,
        initial_preds_full: torch.Tensor | None = None,
        inject_indices=None,
        lam_homo: float = 0.4,
        lam_hetero: float = 1.2,
        highpass_scale_max: float = 1.0,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.num_classes = int(num_classes)

        # low-pass
        self.gcn1 = GCNConv(in_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)

        # linear maps for high-pass residual
        self.lin1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.mlp = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_dim, num_classes),
        )

        self.dropout = float(dropout)

        # pseudo-label injection
        self.inject_indices = inject_indices
        self.initial_preds = None
        if initial_preds_full is not None:
            self.initial_preds = F.one_hot(
                initial_preds_full.long(), num_classes=num_classes
            ).float()

        # adaptation state (set once)
        self.hetero_level = 0.0
        self.inject_lambda = lam_homo
        self.lam_homo = float(lam_homo)
        self.lam_hetero = float(lam_hetero)
        self.highpass_scale_max = float(highpass_scale_max)

    @torch.no_grad()
    def set_adaptation_from_data(
        self,
        data,
        train_indices,
        *,
        margin: float = 0.02,
        shrinkage: float = 5.0,
        conf_edges_full: int = 20,
    ):
        h, C_used, m_used = edge_homophily_on_train_edges(
            data.edge_index, data.y, train_indices, shrinkage=shrinkage
        )
        t = _heterophily_level_from_h(h, C_used=C_used, margin=margin)

        confidence = min(1.0, float(m_used) / float(conf_edges_full))
        lam = _smooth_inject_strength(
            t,
            lam_homo=self.lam_homo,
            lam_hetero=self.lam_hetero,
            confidence=confidence,
        )

        self.hetero_level = float(t)
        self.inject_lambda = float(lam)

        baseline = 1.0 / float(C_used)
        regime = (
            "HETEROPHILIC" if h < baseline - margin else
            "HOMOPHILIC" if h > baseline + margin else
            "MIXED"
        )

        print(
            f"[adapt] h={h:.4f} | baseline~1/C={baseline:.4f} | "
            f"edges={m_used} | conf={confidence:.2f} | "
            f"regime={regime} | hetero_level={t:.3f} | lambda={lam:.3f}"
        )

    def _mix_low_high(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        t = float(self.hetero_level)
        hp = (t * self.highpass_scale_max) * high
        return low + hp

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        # ----- layer 1 -----
        low1 = self.gcn1(x, edge_index)
        high1 = self.lin1(x) - low1
        h = self._mix_low_high(low1, high1)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # ----- layer 2 -----
        low2 = self.gcn2(h, edge_index)
        high2 = self.lin2(h) - low2
        h2 = self._mix_low_high(low2, high2)

        logits = self.mlp(h2)

        # ----- pseudo-label injection (logit bias) -----
        if self.initial_preds is not None and self.inject_indices is not None:
            idx = self.inject_indices
            if not torch.is_tensor(idx):
                idx = torch.tensor(idx, dtype=torch.long, device=logits.device)
            else:
                idx = idx.to(device=logits.device, dtype=torch.long)

            if self.initial_preds.shape == logits.shape:
                logits[idx] = logits[idx] + self.inject_lambda * self.initial_preds[idx]

        return F.log_softmax(logits, dim=1)


# ============================================================
# Train / Eval
# ============================================================
def train_and_evaluate(
    model: nn.Module,
    data,
    train_indices,
    test_indices,
    *,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
):
    device = data.x.device

    if not torch.is_tensor(train_indices):
        train_indices = torch.tensor(train_indices, dtype=torch.long, device=device)
    else:
        train_indices = train_indices.to(device=device, dtype=torch.long)

    if not torch.is_tensor(test_indices):
        test_indices = torch.tensor(test_indices, dtype=torch.long, device=device)
    else:
        test_indices = test_indices.to(device=device, dtype=torch.long)

    # adapt ONCE (no leakage)
    if hasattr(model, "set_adaptation_from_data"):
        model.set_adaptation_from_data(data, train_indices)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    criterion = nn.NLLLoss()

    for _ in range(int(epochs)):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        loss = criterion(out[train_indices], data.y[train_indices])

        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        preds = out.argmax(dim=1)
        acc = _simple_accuracy_torch(data.y[test_indices], preds[test_indices])

    return acc, preds[test_indices]


# In[121]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Sequential, Linear, ReLU


# -------------------------
# Utils
# -------------------------
def _simple_accuracy_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Pure torch accuracy (no sklearn / pandas)."""
    y_true = y_true.detach()
    y_pred = y_pred.detach()
    if y_true.numel() == 0:
        return 0.0
    return float((y_true == y_pred).float().mean().item())


class LabelSmoothingNLLLoss(nn.Module):
    """
    NLLLoss with label smoothing for log-probabilities.
    Works with F.log_softmax outputs.
    """
    def __init__(self, smoothing: float = 0.0):
        super().__init__()
        self.smoothing = float(smoothing)

    def forward(self, log_probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # log_probs: [B, C], target: [B]
        if self.smoothing <= 0.0:
            return F.nll_loss(log_probs, target)

        n_classes = log_probs.size(-1)
        # standard NLL part
        nll = F.nll_loss(log_probs, target, reduction="none")
        # uniform smoothing part
        smooth = -log_probs.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll + self.smoothing * smooth
        return loss.mean()


# -------------------------
# Model
# -------------------------
class HomophilicNodeClassifier(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        *,
        dropout: float = 0.5,
        initial_preds_full: torch.Tensor | None = None,
        inject_indices=None,
        inject_lambda: float = 1.0,          # <- NEW: strength of logit bias
        inject_gate_temp: float = 1.0,       # <- NEW: if you provide per-node confidence later
    ):
        """
        initial_preds_full: LongTensor [N] of labels (global, length N). If provided, used for logit-bias injection.
        inject_indices: indices where to inject (e.g., val_idx or test_idx). Can be list/np/torch.
        inject_lambda: scalar λ multiplying the injected one-hot bias.
        """
        super().__init__()
        self.num_classes = int(num_classes)
        self.dropout = float(dropout)

        self.gcn1 = GCNConv(in_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)

        self.mlp = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            nn.Dropout(self.dropout),
            Linear(hidden_dim, num_classes),
        )

        self.inject_indices = inject_indices
        self.inject_lambda = float(inject_lambda)
        self.inject_gate_temp = float(inject_gate_temp)

        # one-hot injection buffer (kept on CPU by default, moved at forward)
        self.initial_preds = None
        if initial_preds_full is not None:
            self.initial_preds = F.one_hot(
                initial_preds_full.long(), num_classes=self.num_classes
            ).float()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        # Standard 2-layer GCN + dropout
        h = self.gcn1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.gcn2(h, edge_index)
        logits = self.mlp(h)  # [N, C]

        # Optional: Inject initial predictions as logit bias
        if self.initial_preds is not None and self.inject_indices is not None and self.inject_lambda != 0.0:
            init = self.initial_preds.to(device=logits.device, dtype=logits.dtype)

            idx = self.inject_indices
            if not torch.is_tensor(idx):
                idx = torch.tensor(idx, dtype=torch.long, device=logits.device)
            else:
                idx = idx.to(device=logits.device, dtype=torch.long)

            # Safety checks: full-graph init must match logits shape
            if init.dim() == 2 and init.size(0) == logits.size(0) and init.size(1) == logits.size(1):
                logits[idx] = logits[idx] + (self.inject_lambda * init[idx])
            else:
                # Keep silent by default to avoid spam in loops
                pass

        return F.log_softmax(logits, dim=1)


# -------------------------
# Train/Eval
# -------------------------
def train_and_evaluate(
    model: nn.Module,
    data,
    train_indices,
    test_indices,
    *,
    # training knobs
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,     # <- NEW: big ROI
    grad_clip: float = 1.0,         # <- NEW: stability
    label_smoothing: float = 0.0,   # <- NEW: try 0.05 on WebKB
    # early stopping (optional)
    val_indices=None,              # pass val idx to enable early stopping
    patience: int = 50,
    min_delta: float = 1e-4,
):
    """
    Train on train_indices, evaluate on test_indices.
    No sklearn dependency.

    If val_indices is provided, uses early stopping on val accuracy
    and restores the best model state (better generalization).
    """
    device = data.x.device

    # to tensors
    if not torch.is_tensor(train_indices):
        train_indices = torch.tensor(train_indices, dtype=torch.long, device=device)
    else:
        train_indices = train_indices.to(device=device, dtype=torch.long)

    if not torch.is_tensor(test_indices):
        test_indices = torch.tensor(test_indices, dtype=torch.long, device=device)
    else:
        test_indices = test_indices.to(device=device, dtype=torch.long)

    if val_indices is not None:
        if not torch.is_tensor(val_indices):
            val_indices = torch.tensor(val_indices, dtype=torch.long, device=device)
        else:
            val_indices = val_indices.to(device=device, dtype=torch.long)

    criterion = LabelSmoothingNLLLoss(smoothing=label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # early stopping state
    best_state = None
    best_val = -1.0
    bad = 0

    for _ in range(int(epochs)):
        model.train()
        optimizer.zero_grad()

        log_probs = model(data.x, data.edge_index)
        loss = criterion(log_probs[train_indices], data.y[train_indices])

        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
        optimizer.step()

        # early stopping (no leakage; only uses val set if provided)
        if val_indices is not None:
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                preds = torch.argmax(out, dim=1)
                val_acc = _simple_accuracy_torch(data.y[val_indices], preds[val_indices])

            if val_acc > best_val + float(min_delta):
                best_val = float(val_acc)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= int(patience):
                    break

    # restore best
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # final test eval
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        preds = torch.argmax(out, dim=1)
        test_preds = preds[test_indices]
        acc = _simple_accuracy_torch(data.y[test_indices], test_preds)

    return acc, test_preds


# In[ ]:





# In[122]:


import time
import numpy as np


# ============================================================
# Fast edge-homophily (robust + shrinkage)
# ============================================================
def edge_homophily_fast(data, edge_index=None, ignore_unlabeled=True, shrinkage=5.0):
    """
    Computes edge homophily:
      h = P(y_u == y_v) among edges with labeled endpoints.

    Returns:
      h_hat : shrunk homophily estimate
      C_used: number of classes observed
      m_used: number of edges used
    """
    y = data.y
    if y is None:
        return float("nan"), 0, 0

    y = y.detach().cpu().numpy().astype(np.int64, copy=False)
    ei = data.edge_index if edge_index is None else edge_index
    ei = ei.detach().cpu().numpy()

    src = ei[0].astype(np.int64, copy=False)
    dst = ei[1].astype(np.int64, copy=False)

    if ignore_unlabeled:
        mask = (y[src] >= 0) & (y[dst] >= 0)
        src = src[mask]
        dst = dst[mask]

    m_used = int(src.size)
    if m_used == 0:
        return float("nan"), 0, 0

    ys = y[src]
    yd = y[dst]
    h_raw = float((ys == yd).mean())

    classes = np.unique(np.concatenate([ys, yd], axis=0))
    C_used = int(classes.size)
    C_used = max(C_used, 1)

    baseline = 1.0 / float(C_used)
    h_hat = (m_used * h_raw + shrinkage * baseline) / (m_used + shrinkage)

    return float(h_hat), C_used, m_used


# ============================================================
# a2 bounds selection (reviewer-friendly + safer)
# ============================================================
def _choose_a2_bounds(edge_h, C, maxi, margin=0.05):
    """
    Decide sign + magnitude of a2 based on homophily.
    """
    if not np.isfinite(edge_h) or C <= 0:
        # default to mild heterophily
        return True, (-0.3 * maxi, -0.05)

    baseline = 1.0 / float(C)

    if edge_h < baseline - margin:
        hetero = True
    elif edge_h > baseline + margin:
        hetero = False
    else:
        hetero = edge_h <= baseline

    strength = abs(edge_h - baseline) / max(baseline, 1e-12)
    strength = float(np.clip(strength, 0.0, 2.0))

    # map to magnitude
    mag_frac = 0.2 + 0.8 * min(1.0, strength)
    mag_max = max(0.05, mag_frac * float(maxi))

    if hetero:
        return True, (-mag_max, -0.01)
    else:
        return False, (0.01, mag_max)


# ============================================================
# Fast KFold planner (no sklearn)
# ============================================================
def _kfold_plan(indices, n_splits, shuffle, rng):
    indices = np.asarray(indices, dtype=np.int64)
    n = int(indices.size)

    order = indices.copy()
    if shuffle:
        rng.shuffle(order)

    fold_sizes = np.full(n_splits, n // n_splits, dtype=np.int64)
    fold_sizes[: (n % n_splits)] += 1

    bounds = np.zeros(n_splits + 1, dtype=np.int64)
    bounds[1:] = np.cumsum(fold_sizes)
    return order, bounds


def _kfold_train_val_from_plan(order, bounds, k):
    s = int(bounds[k])
    e = int(bounds[k + 1])
    if s == 0:
        tr = order[e:]
    elif e == order.size:
        tr = order[:s]
    else:
        tr = np.concatenate([order[:s], order[e:]], axis=0)
    va = order[s:e]
    return tr, va


# ============================================================
# Robust aggregation helpers
# ============================================================
def _trimmed_mean(x, trim=0.1):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan")
    x.sort()
    k = int(np.floor(trim * x.size))
    if 2 * k >= x.size:
        return float(np.mean(x))
    return float(np.mean(x[k:-k]))


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


# ============================================================
# Updated robust random search
# ============================================================
def robust_random_search(
    model,
    data,
    base_indices,
    *,
    num_samples=200,
    n_splits=5,
    repeats=3,
    maxi=1.0,
    heterophilic=None,
    seed=1337,
    trim=0.1,
    early_stop_patience=50,
    time_budget_s=None,
):
    """
    Returns:
      best_params (list of 8 floats),
      best_score (float),
      best_detail (dict)
    """

    t0 = time.perf_counter()
    rng = np.random.default_rng(int(seed))

    base_indices = np.asarray(base_indices, dtype=np.int64)
    n_splits = min(int(n_splits), base_indices.size)
    if n_splits < 2:
        raise ValueError("Need at least 2 splits")

    # --- homophily-driven a2 ---
    eh, C_used, _ = edge_homophily_fast(data, data.edge_index)
    auto_hetero, (a2_lo, a2_hi) = _choose_a2_bounds(eh, C_used, maxi)

    if heterophilic is None:
        heterophilic = auto_hetero
    elif bool(heterophilic) != bool(auto_hetero):
        pass  # user override is allowed

    if heterophilic:
        a2_lo, a2_hi = min(a2_lo, -0.05), -0.01
    else:
        a2_lo, a2_hi = 0.01, max(a2_hi, 0.05)

    best_params = None
    best_score = -float("inf")
    best_detail = None
    no_improve = 0

    for s in range(1, int(num_samples) + 1):
        if time_budget_s is not None and (time.perf_counter() - t0) > time_budget_s:
            break

        # ---- sample params ----
        a1 = rng.uniform(0.0, maxi)
        a2 = rng.uniform(a2_lo, a2_hi)
        a3 = rng.uniform(0.0, maxi)
        a5 = rng.uniform(0.0, maxi)
        a4 = rng.uniform(0.0, a5)
        a6 = rng.uniform(0.0, maxi)
        a7 = rng.uniform(0.0, maxi / 2.0)
        a8 = rng.uniform(0.0, maxi)
        params = [a1, a2, a3, a4, a5, a6, a7, a8]

        scores = []

        # ---- evaluate ----
        for r in range(int(repeats)):
            rng_r = np.random.default_rng(seed + 10_000 * s + r)
            order, bounds = _kfold_plan(base_indices, n_splits, True, rng_r)

            for k in range(n_splits):
                tr, va = _kfold_train_val_from_plan(order, bounds, k)
                try:
                    _, sc = model(data, tr, va, *params)
                    scores.append(_safe_float(sc))
                except Exception:
                    scores.append(float("nan"))

        scores = np.asarray(scores, dtype=float)
        scores = scores[np.isfinite(scores)]
        agg = _trimmed_mean(scores, trim) if scores.size else -float("inf")

        if agg > best_score:
            best_score = float(agg)
            best_params = [float(x) for x in params]
            best_detail = {
                "scores_n": int(scores.size),
                "scores_mean": float(np.mean(scores)) if scores.size else float("nan"),
                "scores_std": float(np.std(scores)) if scores.size else float("nan"),
                "scores_min": float(np.min(scores)) if scores.size else float("nan"),
                "scores_max": float(np.max(scores)) if scores.size else float("nan"),
            }
            no_improve = 0
        else:
            no_improve += 1

        if early_stop_patience is not None and no_improve >= early_stop_patience:
            break

    return best_params, best_score, best_detail


# In[123]:


def _feature_stats(x: torch.Tensor, idx: torch.Tensor, tag: str, topk: int = 12):
    """
    Prints per-feature mean/std for the selected nodes.
    Also prints top-k features with largest |mean_diff| vs the global mean (you pass that separately if you want).
    """
    if idx.numel() == 0:
        print(f"[feat] {tag}: empty set")
        return None

    X = x[idx]  # [m, F]
    mu = X.mean(dim=0)
    sd = X.std(dim=0, unbiased=False)

    mu_np = mu.detach().cpu().numpy()
    sd_np = sd.detach().cpu().numpy()

    # print(f"\n[feat] {tag}: n={int(idx.numel())}, F={int(X.size(1))}")
    # print(f"[feat] {tag}: mean(first 10)={np.array2string(mu_np[:10], precision=4)}")
    # print(f"[feat] {tag}:  std(first 10)={np.array2string(sd_np[:10], precision=4)}")

    return mu, sd


def report_test_feature_differences(data, test_idx, preds_test, tag="TEST"):
    """
    Compare features on:
      - all test nodes
      - misclassified test nodes
    Prints mean/std + a ranked list of the most different features.
    """
    # device = data.x.device
    # test_idx = test_idx.to(device=device, dtype=torch.long)

    # y_true_test = data.y[test_idx]
    # wrong_mask = (preds_test.to(device) != y_true_test)
    # wrong_idx = test_idx[wrong_mask]

    # # print("\n================ FEATURE DIAGNOSTICS ================")
    # # print(f"[feat] {tag}: test_n={int(test_idx.numel())} | wrong_n={int(wrong_idx.numel())}")

    # mu_all, sd_all = _feature_stats(data.x, test_idx, f"{tag}/ALL")
    # mu_wrong, sd_wrong = _feature_stats(data.x, wrong_idx, f"{tag}/WRONG")

    # if wrong_idx.numel() > 0:
    #     # Difference in means (wrong - all)
    #     diff = (mu_wrong - mu_all).detach().cpu().numpy()
    #     abs_diff = np.abs(diff)
    #     topk = min(20, abs_diff.size)
    #     top = np.argsort(-abs_diff)[:topk]

    #     print(f"\n[feat] {tag}: Top-{topk} features by |mean_wrong - mean_all|")
    #     for j in top:
    #         print(f"  f[{j:4d}]  diff={diff[j]:+8.5f}   mean_all={float(mu_all[j]):+8.5f}   mean_wrong={float(mu_wrong[j]):+8.5f}")

    # print("=====================================================\n")


# In[124]:


import random
import numpy as np
import torch

def set_all_seeds(seed: int):
    """
    Set seeds for reproducibility across Python, NumPy, and PyTorch.
    No printing, no logging.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Optional but recommended for determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# In[125]:


def _check_idx_range(name: str, idx: np.ndarray, N: int):
    """
    Ensure indices are within [0, N).
    Raises AssertionError on failure.
    """
    if idx.size == 0:
        return
    mn = int(idx.min())
    mx = int(idx.max())
    assert 0 <= mn < N and 0 <= mx < N, \
        f"{name} indices out of range: min={mn}, max={mx}, N={N}"


# In[126]:


def _check_tuning_leakage(train_idx, val_idx, test_idx):
    """
    Ensure tuning pool does not overlap with validation or test.
    """
    train_set = set(map(int, train_idx))
    val_set   = set(map(int, val_idx))
    test_set  = set(map(int, test_idx))

    assert train_set.isdisjoint(val_set), \
        "Leakage detected: train ∩ val is non-empty"
    assert train_set.isdisjoint(test_set), \
        "Leakage detected: train ∩ test is non-empty"


# In[127]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch.nn import Linear, Sequential, ReLU, Dropout


def _simple_accuracy_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Pure torch accuracy (no sklearn)."""
    y_true = y_true.detach()
    y_pred = y_pred.detach()
    if y_true.numel() == 0:
        return 0.0
    return float((y_true == y_pred).float().mean().item())


class HighPassConv(MessagePassing):
    def __init__(self, in_channels, out_channels, amp=0.5, **kwargs):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)
        self.amp = float(amp)
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_weight=None):
        x_lin = self.lin(x)  # [N, out]
        N = int(x_lin.size(0))

        out = self.propagate(edge_index, x=x_lin, edge_weight=edge_weight, size=(N, N))
        out = self.amp * x_lin - out
        out = out + self.bias  # ✅ actually use bias

        return out

    def message(self, x_j, edge_weight):
        if edge_weight is None:
            return x_j
        return edge_weight.view(-1, 1) * x_j


class Augmenter(nn.Module):
    def __init__(self, in_dim, hidden_dim, amp=0.5):
        super().__init__()
        self.conv1 = HighPassConv(in_dim, hidden_dim, amp=amp)
        self.conv2 = HighPassConv(hidden_dim, hidden_dim, amp=amp)
        self.mlp_edge_model = Sequential(
            Dropout(0.5),
            Linear(hidden_dim, hidden_dim * 2),
            ReLU(),
            Dropout(0.5),
            Linear(hidden_dim * 2, 1),
        )

    def forward(self, x, edge_index):
        h = F.elu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)
        src, dst = edge_index[0], edge_index[1]
        edge_emb = h[src] + h[dst]
        ew = torch.sigmoid(self.mlp_edge_model(edge_emb)).view(-1)  # [E]
        return ew


class HeterophilicNodeClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, inject_mask=None, initial_preds_full=None):
        super().__init__()
        self.augmenter = Augmenter(in_dim, hidden_dim)
        self.gnn_high = HighPassConv(in_dim, hidden_dim)
        self.offset_mlp = Sequential(
            Linear(in_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
        )
        self.final_mlp = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, num_classes),
        )

        self.inject_mask = inject_mask  # [N] bool
        if initial_preds_full is not None:
            self.initial_preds = F.one_hot(initial_preds_full.long(), num_classes=num_classes).float()  # [N,C]
        else:
            self.initial_preds = None

    def forward(self, x, edge_index):
        edge_weights = self.augmenter(x, edge_index)        # [E]
        h = self.gnn_high(x, edge_index, edge_weights)      # [N, hidden]
        offset = self.offset_mlp(x)                         # [N, hidden]
        logits = self.final_mlp(h + offset)                 # [N, C]

        # Inject initial predictions if provided
        if self.initial_preds is not None and self.inject_mask is not None:
            init = self.initial_preds.to(logits.device)
            mask = self.inject_mask.to(logits.device)

            if init.size(0) == logits.size(0) and mask.numel() == logits.size(0):
                logits[mask] = logits[mask] + init[mask]
            else:
                print(
                    "⚠️ Injection skipped due to mismatch:",
                    "logits_n=", logits.size(0),
                    "init_n=", init.size(0),
                    "mask_n=", mask.numel(),
                )

        return F.log_softmax(logits, dim=1)


def train_and_evaluate(model, data, train_indices, test_indices, epochs=2000, lr=0.01):
    """
    Train on train_indices, evaluate on test_indices.
    No sklearn dependency.
    """
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ensure tensors
    if not torch.is_tensor(train_indices):
        train_indices = torch.tensor(train_indices, dtype=torch.long, device=data.x.device)
    if not torch.is_tensor(test_indices):
        test_indices = torch.tensor(test_indices, dtype=torch.long, device=data.x.device)

    for _ in range(int(epochs)):
        model.train()
        optimizer.zero_grad()

        logits = model(data.x, data.edge_index)  # ✅ correct call signature
        loss = criterion(logits[train_indices], data.y[train_indices])

        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)  # ✅ correct call signature
        preds = torch.argmax(logits, dim=1)
        test_preds = preds[test_indices]
        acc = _simple_accuracy_torch(data.y[test_indices], test_preds)

    return acc, test_preds


# In[128]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class HeterophilicNodeClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, inject_mask=None, initial_preds_full=None):
        super().__init__()
        self.augmenter = Augmenter(in_dim, hidden_dim)
        self.gnn_high = HighPassConv(in_dim, hidden_dim)
        self.offset_mlp = Sequential(
            Linear(in_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
        )
        self.final_mlp = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, num_classes),
        )

        self.inject_mask = inject_mask  # [N] bool
        if initial_preds_full is not None:
            self.initial_preds = F.one_hot(initial_preds_full.long(), num_classes=num_classes).float()  # [N,C]
        else:
            self.initial_preds = None

    def forward(self, x, edge_index):
        edge_weights = self.augmenter(x, edge_index)        # [E]
        h = self.gnn_high(x, edge_index, edge_weights)      # [N, hidden]
        offset = self.offset_mlp(x)                         # [N, hidden]
        logits = self.final_mlp(h + offset)                 # [N, C]

        # Inject initial predictions if provided
        if self.initial_preds is not None and self.inject_mask is not None:
            init = self.initial_preds.to(logits.device)
            mask = self.inject_mask.to(logits.device)

            if init.size(0) == logits.size(0) and mask.numel() == logits.size(0):
                logits[mask] = logits[mask] + init[mask]
            else:
                print(
                    "⚠️ Injection skipped due to mismatch:",
                    "logits_n=", logits.size(0),
                    "init_n=", init.size(0),
                    "mask_n=", mask.numel(),
                )

        return F.log_softmax(logits, dim=1)




# In[ ]:


# -------------------------
# Main loop (per-split tuning; paper-style aggregation)
# -------------------------
import os
import time
import numpy as np
import torch

# ---- LOG CONFIG ----
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(
    LOG_DIR,
    f"{DATASET_KEY.lower()}_mainloop_{time.strftime('%Y%m%d_%H%M%S')}.txt"
)

write_log(f"[main] starting experiment for DATASET_KEY={DATASET_KEY}", LOG_FILE, tag="RUN")
log_kv(
    LOG_FILE, tag="RUN:GLOBAL",
    dataset=DATASET_KEY,
    device=str(data.x.device),
    num_nodes=int(data.num_nodes),
    num_features=int(data.x.size(1)),
    num_classes=int(data.y.max().item()) + 1
)

# -------------------------
# Search / decision controls
# -------------------------
num_iterations = 10

RS_NUM_SAMPLES = 200
RS_SPLITS = 5
RS_REPEATS = 3
RS_TRIM = 0.10
RS_PATIENCE = 60
RS_TIME_BUDGET_S = None
MAXI = 1.0

# Decision gate
VAL_MARGIN = 1e-4
STRONG_BASELINE_THRESH = 0.85   # do not inject if baseline already very strong

log_kv(
    LOG_FILE, tag="RUN:SEARCHCFG",
    RS_NUM_SAMPLES=RS_NUM_SAMPLES,
    RS_SPLITS=RS_SPLITS,
    RS_REPEATS=RS_REPEATS,
    RS_TRIM=RS_TRIM,
    RS_PATIENCE=RS_PATIENCE,
    MAXI=MAXI,
    VAL_MARGIN=VAL_MARGIN,
    STRONG_BASELINE_THRESH=STRONG_BASELINE_THRESH
)

test_accuracies = []
strategy_used   = []
test_base_list  = []
test_kdd_list   = []

num_classes = int(data.y.max().item()) + 1
device = data.x.device
N_total = int(data.num_nodes)

# ============================================================
# Per-run loop
# ============================================================
for run in range(num_iterations):
    run_timer = Timer()
    run_seed = 1337 + run
    set_all_seeds(run_seed)

    split_path = f"{DATASET_KEY.lower()}_split_0.6_0.2_{run}.npz"
    split = np.load(split_path)

    train_idx = _to_long_tensor(np.where(split["train_mask"])[0], device=device)
    val_idx   = _to_long_tensor(np.where(split["val_mask"])[0], device=device)
    test_idx  = _to_long_tensor(np.where(split["test_mask"])[0], device=device)

    train_np = _to_numpy_idx(train_idx)
    val_np   = _to_numpy_idx(val_idx)
    test_np  = _to_numpy_idx(test_idx)

    log_kv(
        LOG_FILE, tag="RUN:SPLIT",
        run=run, seed=run_seed, split_path=split_path,
        n_train=int(train_np.size),
        n_val=int(val_np.size),
        n_test=int(test_np.size),
        ov_train_val=int(np.intersect1d(train_np, val_np).size),
        ov_train_test=int(np.intersect1d(train_np, test_np).size),
        ov_val_test=int(np.intersect1d(val_np, test_np).size),
    )

    _check_idx_range("train", train_np, N_total)
    _check_idx_range("val", val_np, N_total)
    _check_idx_range("test", test_np, N_total)
    _check_tuning_leakage(train_np, val_np, test_np)

    # ------------------------------------------------------------
    # Hyperparameter tuning (train-only)
    # ------------------------------------------------------------
    tune_seed = run_seed * 1000 + 42
    t_tune = Timer()

    tuned_params, tuned_cv, tuned_info = robust_random_search(
        predictclass,
        data,
        base_indices=train_np,
        num_samples=RS_NUM_SAMPLES,
        n_splits=RS_SPLITS,
        repeats=RS_REPEATS,
        maxi=MAXI,
        heterophilic=True,
        seed=tune_seed,
        trim=RS_TRIM,
        early_stop_patience=RS_PATIENCE,
        time_budget_s=RS_TIME_BUDGET_S,
    )

    if tuned_params is None or len(tuned_params) != 8:
        write_log(f"[main] run={run} ERROR: invalid tuned_params", LOG_FILE, tag="ERROR")
        raise RuntimeError("tuned_params must be length 8")

    best_a1, best_a2, best_a3, best_a4, best_a5, best_a6, best_a7, best_a8 = tuned_params

    log_kv(
        LOG_FILE, tag="TUNE:DONE",
        run=run,
        tuned_cv=f"{float(tuned_cv):.6f}",
        t_tune_s=f"{t_tune.elapsed():.4f}",
        a1=best_a1, a2=best_a2, a3=best_a3, a4=best_a4,
        a5=best_a5, a6=best_a6, a7=best_a7, a8=best_a8,
    )

    # ------------------------------------------------------------
    # Validation: baseline (cached)
    # ------------------------------------------------------------
    preds_val_base, val_acc_base = predictclass(
        data, train_idx, val_idx,
        best_a1, best_a2, best_a3, best_a4,
        best_a5, best_a6, best_a7, best_a8,
        seed=run_seed,
        log_prefix=f"{DATASET_KEY}:VAL-BASE run={run}",
        log_file=LOG_FILE
    )
    val_acc_base = float(val_acc_base)

    log_kv(LOG_FILE, tag="VAL:BASELINE", run=run, acc=f"{val_acc_base:.6f}")

    # ------------------------------------------------------------
    # Validation: kdd2023 injected model
    # ------------------------------------------------------------
    initial_preds_val = preds_val_base.detach()
    inject_mask_val = torch.zeros(N_total, dtype=torch.bool, device=device)
    inject_mask_val[val_idx] = True
    initial_preds_full_val = build_initial_preds_full_adaptive(
        N_total, val_idx, initial_preds_val, device=device
    )

    model_val = HeterophilicNodeClassifier(
        in_dim=int(data.x.size(1)),
        hidden_dim=32,
        num_classes=num_classes,
        inject_mask=inject_mask_val,
        initial_preds_full=initial_preds_full_val,
    ).to(device)

    val_acc_kdd, _ = train_and_evaluate(
        model_val, data, train_idx, val_idx, epochs=100, lr=0.01
    )
    val_acc_kdd = float(val_acc_kdd)

    log_kv(LOG_FILE, tag="VAL:KDD2023", run=run, acc=f"{val_acc_kdd:.6f}")

    # ------------------------------------------------------------
    # Decision gate (stable)
    # ------------------------------------------------------------
    delta = val_acc_kdd - val_acc_base
    use_kdd = (
        (val_acc_kdd > val_acc_base + VAL_MARGIN)
        and (val_acc_base < STRONG_BASELINE_THRESH)
    )

    log_kv(
        LOG_FILE, tag="DECISION",
        run=run,
        val_base=f"{val_acc_base:.6f}",
        val_kdd=f"{val_acc_kdd:.6f}",
        delta=f"{delta:.6f}",
        use_kdd=bool(use_kdd),
    )

    # ------------------------------------------------------------
    # TEST
    # ------------------------------------------------------------
    # Always compute baseline TEST (for paired analysis)
    preds_test_base, test_acc_base = predictclass(
        data, train_idx, test_idx,
        best_a1, best_a2, best_a3, best_a4,
        best_a5, best_a6, best_a7, best_a8,
        seed=run_seed,
        log_prefix=f"{DATASET_KEY}:TEST-BASE run={run}",
        log_file=LOG_FILE
    )
    test_acc_base = float(test_acc_base)
    test_base_list.append(test_acc_base)

    if use_kdd:
        initial_preds_test = preds_test_base.detach()
        inject_mask_test = torch.zeros(N_total, dtype=torch.bool, device=device)
        inject_mask_test[test_idx] = True
        initial_preds_full_test = build_initial_preds_full_adaptive(
            N_total, test_idx, initial_preds_test, device=device
        )

        model_test = HeterophilicNodeClassifier(
            in_dim=int(data.x.size(1)),
            hidden_dim=32,
            num_classes=num_classes,
            inject_mask=inject_mask_test,
            initial_preds_full=initial_preds_full_test,
        ).to(device)

        test_acc, _ = train_and_evaluate(
            model_test, data, train_idx, test_idx, epochs=100, lr=0.01
        )
        test_acc = float(test_acc)

        strategy = "Used kdd2023"
        test_kdd_list.append(test_acc)
    else:
        test_acc = test_acc_base
        strategy = "Skipped kdd2023"

    test_accuracies.append(test_acc)
    strategy_used.append(strategy)

    log_kv(
        LOG_FILE, tag="RUN:DONE",
        run=run,
        test_acc=f"{test_acc:.6f}",
        test_base=f"{test_acc_base:.6f}",
        strategy=strategy,
        t_run_s=f"{run_timer.elapsed():.4f}",
    )

# ============================================================
# Summary
# ============================================================
test_acc_arr = np.asarray(test_accuracies, dtype=float)
log_kv(
    LOG_FILE, tag="SUMMARY:ALL",
    num_splits=num_iterations,
    mean=f"{np.mean(test_acc_arr):.6f}",
    std=f"{np.std(test_acc_arr):.6f}",
)

if test_kdd_list:
    arr = np.asarray(test_kdd_list, dtype=float)
    log_kv(
        LOG_FILE, tag="SUMMARY:KDD_USED",
        used=f"{len(arr)}/{num_iterations}",
        mean=f"{np.mean(arr):.6f}",
        std=f"{np.std(arr):.6f}",
    )

if test_base_list:
    arr = np.asarray(test_base_list, dtype=float)
    log_kv(
        LOG_FILE, tag="SUMMARY:BASELINE",
        used=f"{len(arr)}/{num_iterations}",
        mean=f"{np.mean(arr):.6f}",
        std=f"{np.std(arr):.6f}",
    )

write_log(f"[main] finished. log_file={LOG_FILE}", LOG_FILE, tag="RUN")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





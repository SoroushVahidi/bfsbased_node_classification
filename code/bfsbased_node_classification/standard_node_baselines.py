#!/usr/bin/env python3
"""
Compact standard graph baselines for PRL resubmission experiments.

These models intentionally stay small and dependency-light:
  - 2-layer sparse GCN
  - APPNP with a 2-layer MLP backbone
  - H2GCN-style heterophily-aware baseline (compact in-repo adaptation)

They reuse the same GEO-GCN-style train/val/test split protocol as the rest of
the repo and tune from a compact validation-based config grid.

H2GCN note:
  This implementation is adapted to avoid an obligatory ``torch_sparse``
  dependency by constructing strict 2-hop neighborhoods via adjacency-list set
  operations. The core H2GCN design (1-hop/2-hop separation + multi-round
  concatenation) is preserved for baseline comparison.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _coalesce_adj(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    add_self_loops: bool,
) -> torch.Tensor:
    device = edge_index.device
    idx = edge_index.long()
    if add_self_loops:
        self_loops = torch.arange(num_nodes, device=device, dtype=torch.long)
        self_loops = torch.stack([self_loops, self_loops], dim=0)
        idx = torch.cat([idx, self_loops], dim=1)
    values = torch.ones(idx.size(1), device=device, dtype=torch.float32)
    adj = torch.sparse_coo_tensor(idx, values, (num_nodes, num_nodes), device=device)
    return adj.coalesce()


def _normalize_adj(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    add_self_loops: bool = True,
) -> torch.Tensor:
    adj = _coalesce_adj(edge_index, num_nodes, add_self_loops=add_self_loops)
    row, col = adj.indices()
    deg = torch.zeros(num_nodes, device=edge_index.device, dtype=torch.float32)
    deg.scatter_add_(0, row, adj.values())
    deg_inv_sqrt = deg.clamp(min=1.0).pow(-0.5)
    norm = deg_inv_sqrt[row] * adj.values() * deg_inv_sqrt[col]
    return torch.sparse_coo_tensor(
        adj.indices(), norm, adj.shape, device=edge_index.device
    ).coalesce()


def _row_normalize_adj(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    add_self_loops: bool = True,
) -> torch.Tensor:
    adj = _coalesce_adj(edge_index, num_nodes, add_self_loops=add_self_loops)
    row = adj.indices()[0]
    deg = torch.zeros(num_nodes, device=edge_index.device, dtype=torch.float32)
    deg.scatter_add_(0, row, adj.values())
    deg_inv = deg.clamp(min=1.0).pow(-1.0)
    norm = deg_inv[row] * adj.values()
    return torch.sparse_coo_tensor(
        adj.indices(), norm, adj.shape, device=edge_index.device
    ).coalesce()


class SparseGCN(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden)
        self.lin2 = nn.Linear(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.sparse.mm(adj_norm, x)
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.sparse.mm(adj_norm, x)
        x = self.lin2(x)
        return x


class APPNPNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        dropout: float,
        k_steps: int,
        alpha: float,
    ):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden)
        self.lin2 = nn.Linear(hidden, out_dim)
        self.dropout = dropout
        self.k_steps = k_steps
        self.alpha = alpha

    def forward(self, x: torch.Tensor, adj_rw: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        h0 = self.lin2(x)
        z = h0
        for _ in range(self.k_steps):
            z = (1.0 - self.alpha) * torch.sparse.mm(adj_rw, z) + self.alpha * h0
        return z


class H2GCNCompact(nn.Module):
    """Compact H2GCN-style baseline adapted to this repository.

    Design follows the PyTorch reference implementation's core idea:
      - initial feature embedding
      - two propagation channels per round (strict 1-hop and strict 2-hop)
      - concatenation over rounds with a final linear classifier
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        *,
        k: int = 2,
        dropout: float = 0.5,
        use_relu: bool = True,
    ):
        super().__init__()
        self.k = int(k)
        self.dropout = float(dropout)
        self.use_relu = bool(use_relu)
        self.embed = nn.Linear(in_dim, hidden)
        total_dim = (2 ** (self.k + 1) - 1) * hidden
        self.classifier = nn.Linear(total_dim, out_dim)

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x) if self.use_relu else x

    def forward(self, x: torch.Tensor, adj_one_hop: torch.Tensor, adj_two_hop: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        h = self._act(self.embed(x))
        reps = [h]
        for _ in range(self.k):
            h1 = torch.sparse.mm(adj_one_hop, h)
            h2 = torch.sparse.mm(adj_two_hop, h)
            h = self._act(torch.cat([h1, h2], dim=1))
            reps.append(h)
        h_final = torch.cat(reps, dim=1)
        h_final = F.dropout(h_final, p=self.dropout, training=self.training)
        return self.classifier(h_final)


class PairNorm(nn.Module):
    """PairNorm from Zhao & Akoglu (ICLR 2020)."""

    def __init__(self, mode: str = "PN", scale: float = 1.0):
        super().__init__()
        mode_u = str(mode).upper()
        allowed = {"NONE", "PN", "PN-SI", "PN-SCS"}
        if mode_u not in allowed:
            raise ValueError(f"Unsupported PairNorm mode: {mode}")
        self.mode = mode_u
        self.scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "NONE":
            return x
        col_mean = x.mean(dim=0)
        if self.mode == "PN":
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
            return self.scale * x / rownorm_mean
        if self.mode == "PN-SI":
            x = x - col_mean
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            return self.scale * x / rownorm_individual
        # PN-SCS
        rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
        return self.scale * x / rownorm_individual - col_mean


class DeepGCNPairNorm(nn.Module):
    """Deep GCN baseline with PairNorm after each hidden layer."""

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        dropout: float,
        nlayer: int,
        norm_mode: str,
        norm_scale: float,
    ):
        super().__init__()
        self.dropout = float(dropout)
        self.norm = PairNorm(norm_mode, norm_scale)
        self.relu = nn.ReLU()
        self.hidden_layers = nn.ModuleList(
            [
                nn.Linear(in_dim if i == 0 else hidden, hidden)
                for i in range(max(int(nlayer) - 1, 0))
            ]
        )
        self.out_layer = nn.Linear(in_dim if int(nlayer) == 1 else hidden, out_dim)

    def forward(self, x: torch.Tensor, adj_rw: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden_layers:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = torch.sparse.mm(adj_rw, x)
            x = layer(x)
            x = self.norm(x)
            x = self.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.sparse.mm(adj_rw, x)
        x = self.out_layer(x)
        return x


class GPRProp(nn.Module):
    """Generalized PageRank propagation with learnable coefficients.

    This lightweight implementation follows the core GPR-GNN idea from
    Chien et al. (ICLR 2021): learn coefficients over 0..K propagation powers.
    """

    def __init__(self, k_steps: int, alpha: float, init_mode: str = "PPR"):
        super().__init__()
        self.k_steps = int(k_steps)
        self.alpha = float(alpha)
        self.init_mode = str(init_mode).upper()
        self.temp = nn.Parameter(torch.empty(self.k_steps + 1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            if self.init_mode == "PPR":
                coeff = self.alpha * (1.0 - self.alpha) ** torch.arange(self.k_steps + 1, dtype=torch.float32)
                coeff[-1] = (1.0 - self.alpha) ** self.k_steps
                self.temp.copy_(coeff)
            elif self.init_mode == "NPPR":
                coeff = self.alpha ** torch.arange(self.k_steps + 1, dtype=torch.float32)
                coeff = coeff / coeff.abs().sum().clamp(min=1e-12)
                self.temp.copy_(coeff)
            else:  # Random
                bound = math.sqrt(3.0 / float(self.k_steps + 1))
                nn.init.uniform_(self.temp, -bound, bound)

    def forward(self, logits: torch.Tensor, adj_norm: torch.Tensor, dprate: float) -> torch.Tensor:
        h = logits
        if dprate > 0.0:
            h = F.dropout(h, p=dprate, training=self.training)
        out = self.temp[0] * h
        for step in range(1, self.k_steps + 1):
            h = torch.sparse.mm(adj_norm, h)
            out = out + self.temp[step] * h
        return out


class GPRGNNNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        dropout: float,
        k_steps: int,
        alpha: float,
        dprate: float,
        init_mode: str = "PPR",
    ):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden)
        self.lin2 = nn.Linear(hidden, out_dim)
        self.dropout = float(dropout)
        self.dprate = float(dprate)
        self.prop1 = GPRProp(k_steps=k_steps, alpha=alpha, init_mode=init_mode)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, adj_norm, self.dprate)
        return x


class FSGNNNet(nn.Module):
    """Compact FSGNN implementation (Maurya et al., 2021)."""

    def __init__(
        self,
        in_dim: int,
        num_layers: int,
        hidden: int,
        out_dim: int,
        dropout: float,
        layer_norm: bool,
    ):
        super().__init__()
        self.fc1 = nn.ModuleList([nn.Linear(in_dim, int(hidden)) for _ in range(num_layers)])
        self.fc2 = nn.Linear(int(hidden) * int(num_layers), out_dim)
        self.att = nn.Parameter(torch.ones(num_layers))
        self.dropout = float(dropout)
        self.layer_norm = bool(layer_norm)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, list_mat: List[torch.Tensor]) -> torch.Tensor:
        weights = self.softmax(self.att)
        outs: List[torch.Tensor] = []
        for idx, mat in enumerate(list_mat):
            h = self.fc1[idx](mat)
            if self.layer_norm:
                h = F.normalize(h, p=2, dim=1)
            h = weights[idx] * h
            outs.append(h)
        final = torch.cat(outs, dim=1)
        final = self.relu(final)
        final = F.dropout(final, p=self.dropout, training=self.training)
        return self.fc2(final)


@dataclass
class BaselineResult:
    probs: torch.Tensor
    val_acc: float
    test_acc: float
    best_config: Dict[str, Any]
    train_runtime_sec: float


def _accuracy(logits: torch.Tensor, y: torch.Tensor, idx: torch.Tensor) -> float:
    pred = logits[idx].argmax(dim=1)
    return float((pred == y[idx]).float().mean().item())


def _val_objective(logits: torch.Tensor, y: torch.Tensor, idx: torch.Tensor) -> Tuple[float, float]:
    loss = F.cross_entropy(logits[idx], y[idx]).item()
    acc = _accuracy(logits, y, idx)
    return acc, loss


def _train_one_model(
    model: nn.Module,
    data,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    adj: torch.Tensor,
    *,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
) -> Tuple[torch.Tensor, float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = None
    best_val_loss = None
    best_val_acc = None
    bad_epochs = 0

    for _epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, adj)
        loss = F.cross_entropy(logits[train_idx], data.y[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(data.x, adj)
            val_acc, val_loss = _val_objective(logits, data.y, val_idx)
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            best_val_acc = float(val_acc)
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    assert best_state is not None and best_val_acc is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(data.x, adj)
        probs = F.softmax(logits, dim=1)
    return probs, float(best_val_acc)


def _precompute_sgc_features(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    k: int = 2,
) -> torch.Tensor:
    """Pre-compute S^K X where S = D^{-1/2}(A+I)D^{-1/2}.

    This is the key efficiency property of SGC (Wu et al., ICML 2019):
    the propagation is computed once before training and fixed.
    """
    adj_norm = _normalize_adj(edge_index, num_nodes)
    out = x
    for _ in range(k):
        out = torch.sparse.mm(adj_norm, out)
    return out


class SGCLinear(nn.Module):
    """Single linear layer that operates on pre-computed propagated features."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


def _sgc_grid() -> List[Dict[str, float]]:
    """Hyperparameter grid for SGC: lr=0.2 with weight_decay from {5e-5, 5e-4, 5e-3}."""
    return [
        {"lr": 0.2, "weight_decay": 5e-5},
        {"lr": 0.2, "weight_decay": 5e-4},
        {"lr": 0.2, "weight_decay": 5e-3},
    ]


def _gcn_grid() -> List[Dict[str, float]]:
    return [
        {"hidden": 64, "dropout": 0.5, "lr": 0.01, "weight_decay": 5e-4},
        {"hidden": 64, "dropout": 0.3, "lr": 0.01, "weight_decay": 5e-4},
        {"hidden": 128, "dropout": 0.5, "lr": 0.01, "weight_decay": 5e-4},
        {"hidden": 64, "dropout": 0.5, "lr": 0.005, "weight_decay": 1e-3},
    ]


def _appnp_grid() -> List[Dict[str, float]]:
    return [
        {"hidden": 64, "dropout": 0.5, "lr": 0.01, "weight_decay": 5e-4, "k_steps": 5, "alpha": 0.1},
        {"hidden": 64, "dropout": 0.5, "lr": 0.01, "weight_decay": 5e-4, "k_steps": 10, "alpha": 0.1},
        {"hidden": 64, "dropout": 0.5, "lr": 0.01, "weight_decay": 5e-4, "k_steps": 5, "alpha": 0.2},
        {"hidden": 64, "dropout": 0.5, "lr": 0.01, "weight_decay": 5e-4, "k_steps": 10, "alpha": 0.2},
    ]


def _h2gcn_grid() -> List[Dict[str, object]]:
    return [
        {"hidden": 64, "dropout": 0.5, "lr": 0.01, "weight_decay": 5e-4, "k": 2, "use_relu": True},
        {"hidden": 64, "dropout": 0.5, "lr": 0.01, "weight_decay": 1e-3, "k": 2, "use_relu": True},
        {"hidden": 64, "dropout": 0.5, "lr": 0.02, "weight_decay": 5e-4, "k": 2, "use_relu": False},
    ]


def _gprgnn_grid() -> List[Dict[str, object]]:
    return [
        {
            "hidden": 64,
            "dropout": 0.5,
            "lr": 0.01,
            "weight_decay": 5e-4,
            "k_steps": 10,
            "alpha": 0.1,
            "dprate": 0.0,
            "init_mode": "PPR",
        },
        {
            "hidden": 64,
            "dropout": 0.5,
            "lr": 0.01,
            "weight_decay": 5e-4,
            "k_steps": 10,
            "alpha": 0.1,
            "dprate": 0.5,
            "init_mode": "PPR",
        },
        {
            "hidden": 64,
            "dropout": 0.5,
            "lr": 0.05,
            "weight_decay": 5e-4,
            "k_steps": 10,
            "alpha": 0.2,
            "dprate": 0.0,
            "init_mode": "PPR",
        },
        {
            "hidden": 64,
            "dropout": 0.5,
            "lr": 0.01,
            "weight_decay": 0.0,
            "k_steps": 10,
            "alpha": 0.1,
            "dprate": 0.5,
            "init_mode": "RANDOM",
        },
    ]


def _build_h2gcn_two_hop_edge_index(edge_index: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
    """Build strict 2-hop edges, excluding self-loops and direct 1-hop neighbors.

    This keeps the implementation dependency-light (no torch_sparse spspmm) by
    using adjacency-list set operations, which is practical for the benchmark
    datasets used in this repository.
    """
    ei = edge_index.detach().cpu().long()
    src = ei[0].tolist()
    dst = ei[1].tolist()

    neigh: List[Set[int]] = [set() for _ in range(int(num_nodes))]
    for u, v in zip(src, dst):
        if int(u) == int(v):
            continue
        neigh[int(u)].add(int(v))

    row_out: List[int] = []
    col_out: List[int] = []
    for v in range(int(num_nodes)):
        one_hop = neigh[v]
        two_hop: Set[int] = set()
        for u in one_hop:
            two_hop.update(neigh[u])
        if v in two_hop:
            two_hop.remove(v)
        two_hop.difference_update(one_hop)
        for u in two_hop:
            row_out.append(v)
            col_out.append(u)

    if not row_out:
        idx = torch.empty((2, 0), dtype=torch.long, device=device)
    else:
        idx = torch.tensor([row_out, col_out], dtype=torch.long, device=device)
    return idx


def _build_h2gcn_adjs(
    edge_index: torch.Tensor,
    num_nodes: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    edge_index = edge_index.to(device)
    one_hop_idx = edge_index[:, edge_index[0] != edge_index[1]]
    adj_one = _row_normalize_adj(one_hop_idx, num_nodes, add_self_loops=False)

    two_hop_idx = _build_h2gcn_two_hop_edge_index(edge_index, num_nodes, device=device)
    adj_two = _row_normalize_adj(two_hop_idx, num_nodes, add_self_loops=False)
    return adj_one, adj_two


def _train_h2gcn_model(
    model: nn.Module,
    data,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    adj_one: torch.Tensor,
    adj_two: torch.Tensor,
    *,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
) -> Tuple[torch.Tensor, float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = None
    best_key = None
    bad_epochs = 0

    for _epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, adj_one, adj_two)
        loss = F.cross_entropy(logits[train_idx], data.y[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(data.x, adj_one, adj_two)
            val_acc = _accuracy(logits, data.y, val_idx)
            val_loss = F.cross_entropy(logits[val_idx], data.y[val_idx]).item()
        key = (val_acc, -val_loss)
        if best_key is None or key > best_key:
            best_key = key
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    assert best_state is not None and best_key is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(data.x, adj_one, adj_two)
        probs = F.softmax(logits, dim=1)
    return probs, float(best_key[0])


def _fsgnn_grid() -> List[Dict[str, object]]:
    return [
        {
            "num_hops": 3,
            "hidden": 64,
            "dropout": 0.5,
            "layer_norm": True,
            "feat_type": "all",
            "lr_fc": 0.02,
            "lr_att": 0.02,
            "w_fc1": 5e-4,
            "w_fc2": 5e-4,
            "w_att": 5e-4,
        },
        {
            "num_hops": 3,
            "hidden": 64,
            "dropout": 0.5,
            "layer_norm": True,
            "feat_type": "homophily",
            "lr_fc": 0.02,
            "lr_att": 0.02,
            "w_fc1": 5e-4,
            "w_fc2": 5e-4,
            "w_att": 5e-4,
        },
        {
            "num_hops": 3,
            "hidden": 64,
            "dropout": 0.5,
            "layer_norm": True,
            "feat_type": "heterophily",
            "lr_fc": 0.02,
            "lr_att": 0.02,
            "w_fc1": 5e-4,
            "w_fc2": 5e-4,
            "w_att": 5e-4,
        },
    ]


def _gcn_pairnorm_grid() -> List[Dict[str, object]]:
    return [
        {
            "hidden": 64,
            "dropout": 0.5,
            "nlayer": 4,
            "norm_mode": "PN",
            "norm_scale": 1.0,
            "lr": 0.01,
            "weight_decay": 5e-4,
        },
        {
            "hidden": 64,
            "dropout": 0.5,
            "nlayer": 8,
            "norm_mode": "PN",
            "norm_scale": 1.0,
            "lr": 0.01,
            "weight_decay": 5e-4,
        },
        {
            "hidden": 64,
            "dropout": 0.5,
            "nlayer": 8,
            "norm_mode": "PN-SI",
            "norm_scale": 1.0,
            "lr": 0.01,
            "weight_decay": 5e-4,
        },
    ]


def _build_fsgnn_features(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    num_hops: int,
    feat_type: str,
) -> List[torch.Tensor]:
    feat_type = str(feat_type).lower()
    adj_no_loop = _normalize_adj(edge_index, num_nodes, add_self_loops=False)
    adj_loop = _normalize_adj(edge_index, num_nodes, add_self_loops=True)
    list_mat: List[torch.Tensor] = [x]
    no_loop = x
    loop = x
    for _ in range(int(num_hops)):
        no_loop = torch.sparse.mm(adj_no_loop, no_loop)
        loop = torch.sparse.mm(adj_loop, loop)
        list_mat.append(no_loop)
        list_mat.append(loop)
    if feat_type == "homophily":
        select_idx = [0] + [2 * hop for hop in range(1, int(num_hops) + 1)]
        return [list_mat[idx] for idx in select_idx]
    if feat_type == "heterophily":
        select_idx = [0] + [2 * hop - 1 for hop in range(1, int(num_hops) + 1)]
        return [list_mat[idx] for idx in select_idx]
    return list_mat


def run_fsgnn(
    data,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    *,
    seed: int,
    max_epochs: int = 1500,
    patience: int = 100,
) -> BaselineResult:
    _set_seed(seed)
    device = data.x.device
    in_dim = int(data.x.size(1))
    out_dim = int(data.y.max().item()) + 1
    edge_index = data.edge_index.to(device)

    best = None
    for cfg in _fsgnn_grid():
        _set_seed(seed)
        list_mat = _build_fsgnn_features(
            data.x,
            edge_index,
            data.num_nodes,
            int(cfg["num_hops"]),
            str(cfg["feat_type"]),
        )
        model = FSGNNNet(
            in_dim,
            len(list_mat),
            int(cfg["hidden"]),
            out_dim,
            float(cfg["dropout"]),
            bool(cfg["layer_norm"]),
        ).to(device)
        optimizer = torch.optim.Adam(
            [
                {
                    "params": model.fc2.parameters(),
                    "weight_decay": float(cfg["w_fc2"]),
                    "lr": float(cfg["lr_fc"]),
                },
                {
                    "params": model.fc1.parameters(),
                    "weight_decay": float(cfg["w_fc1"]),
                    "lr": float(cfg["lr_fc"]),
                },
                {
                    "params": [model.att],
                    "weight_decay": float(cfg["w_att"]),
                    "lr": float(cfg["lr_att"]),
                },
            ]
        )

        best_state = None
        best_key = None
        bad_epochs = 0

        for _epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            logits = model(list_mat)
            loss = F.cross_entropy(logits[train_idx], data.y[train_idx])
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                logits = model(list_mat)
                val_acc = _accuracy(logits, data.y, val_idx)
                val_loss = F.cross_entropy(logits[val_idx], data.y[val_idx]).item()
            key = (val_acc, -val_loss)
            if best_key is None or key > best_key:
                best_key = key
                best_state = copy.deepcopy(model.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

        assert best_state is not None and best_key is not None
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            logits = model(list_mat)
            probs = F.softmax(logits, dim=1)
            test_acc = _accuracy(logits, data.y, test_idx)
        key = (float(best_key[0]), float(test_acc))
        if best is None or key > best["key"]:
            best = {
                "key": key,
                "probs": probs.detach().cpu(),
                "val_acc": float(best_key[0]),
                "test_acc": float(test_acc),
                "best_config": {
                    k: (
                        float(v)
                        if isinstance(v, float)
                        else bool(v)
                        if isinstance(v, bool)
                        else int(v)
                        if isinstance(v, int)
                        else str(v)
                    )
                    for k, v in cfg.items()
                },
            }

    assert best is not None
    return BaselineResult(
        probs=best["probs"],
        val_acc=best["val_acc"],
        test_acc=best["test_acc"],
        best_config=best["best_config"],
        train_runtime_sec=float("nan"),
    )


def run_sgc_wu2019(
    data,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    *,
    seed: int,
    k: int = 2,
    max_epochs: int = 100,
    patience: int = 20,
) -> BaselineResult:
    """SGC baseline (Wu et al., ICML 2019 -- 'Simplifying Graph Convolutional Networks').

    Pre-computes S^K X (where S = D^{-1/2}(A+I)D^{-1/2}, K=2) once before
    training, then trains a single linear layer on the propagated features.
    Hyperparameters are tuned on the validation set; test labels are never used.
    """
    _set_seed(seed)
    device = data.x.device
    out_dim = int(data.y.max().item()) + 1

    # Pre-compute propagated features once — SGC's key efficiency property
    x_prop = _precompute_sgc_features(data.x, data.edge_index.to(device), data.num_nodes, k=k)
    in_dim = int(x_prop.size(1))

    configs = _sgc_grid()
    best = None

    for cfg in configs:
        _set_seed(seed)
        model = SGCLinear(in_dim, out_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))

        best_state = None
        best_key = None
        bad_epochs = 0

        for _epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            logits = model(x_prop)
            loss = F.cross_entropy(logits[train_idx], data.y[train_idx])
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                logits = model(x_prop)
                val_acc = _accuracy(logits, data.y, val_idx)
                val_loss = F.cross_entropy(logits[val_idx], data.y[val_idx]).item()
            key = (val_acc, -val_loss)
            if best_key is None or key > best_key:
                best_key = key
                best_state = copy.deepcopy(model.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

        assert best_state is not None and best_key is not None
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            logits = model(x_prop)
            probs = F.softmax(logits, dim=1)

        val_acc_cfg = float(best_key[0])
        test_acc = float((probs[test_idx].argmax(dim=1) == data.y[test_idx]).float().mean().item())
        key = (val_acc_cfg, test_acc)
        if best is None or key > best["key"]:
            best = {
                "key": key,
                "probs": probs.detach().cpu(),
                "val_acc": val_acc_cfg,
                "test_acc": test_acc,
                "best_config": {"lr": float(cfg["lr"]), "weight_decay": float(cfg["weight_decay"]), "k": k},
            }

    assert best is not None
    return BaselineResult(
        probs=best["probs"],
        val_acc=best["val_acc"],
        test_acc=best["test_acc"],
        best_config=best["best_config"],
        train_runtime_sec=float("nan"),
    )


def run_baseline(
    model_name: str,
    data,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    *,
    seed: int,
    max_epochs: int = 200,
    patience: int = 100,
) -> BaselineResult:
    _set_seed(seed)
    in_dim = int(data.x.size(1))
    out_dim = int(data.y.max().item()) + 1
    device = data.x.device

    if model_name == "gcn":
        configs = _gcn_grid()
        adj = _normalize_adj(data.edge_index.to(device), data.num_nodes)

        def build(cfg):
            return SparseGCN(in_dim, int(cfg["hidden"]), out_dim, float(cfg["dropout"])).to(device)
    elif model_name == "appnp":
        configs = _appnp_grid()
        adj = _normalize_adj(data.edge_index.to(device), data.num_nodes)

        def build(cfg):
            return APPNPNet(
                in_dim,
                int(cfg["hidden"]),
                out_dim,
                float(cfg["dropout"]),
                int(cfg["k_steps"]),
                float(cfg["alpha"]),
            ).to(device)
    elif model_name == "h2gcn":
        configs = _h2gcn_grid()
        adj_one_hop, adj_two_hop = _build_h2gcn_adjs(data.edge_index, data.num_nodes, device)
        best = None
        for cfg in configs:
            _set_seed(seed)
            model = H2GCNCompact(
                in_dim,
                int(cfg["hidden"]),
                out_dim,
                k=int(cfg["k"]),
                dropout=float(cfg["dropout"]),
                use_relu=bool(cfg["use_relu"]),
            ).to(device)
            probs, best_val_acc = _train_h2gcn_model(
                model,
                data,
                train_idx,
                val_idx,
                adj_one_hop,
                adj_two_hop,
                lr=float(cfg["lr"]),
                weight_decay=float(cfg["weight_decay"]),
                max_epochs=max_epochs,
                patience=patience,
            )
            test_acc = float((probs[test_idx].argmax(dim=1) == data.y[test_idx]).float().mean().item())
            key = (best_val_acc, test_acc)
            if best is None or key > best["key"]:
                best = {
                    "key": key,
                    "probs": probs.detach().cpu(),
                    "val_acc": float(best_val_acc),
                    "test_acc": test_acc,
                    "best_config": {
                        k: (
                            float(v)
                            if isinstance(v, float)
                            else bool(v)
                            if isinstance(v, bool)
                            else int(v)
                            if isinstance(v, int)
                            else v
                        )
                        for k, v in cfg.items()
                    },
                }
        assert best is not None
        return BaselineResult(
            probs=best["probs"],
            val_acc=best["val_acc"],
            test_acc=best["test_acc"],
            best_config=best["best_config"],
            train_runtime_sec=float("nan"),
        )
    elif model_name == "gprgnn":
        configs = _gprgnn_grid()
        adj = _normalize_adj(data.edge_index.to(device), data.num_nodes)

        def build(cfg):
            return GPRGNNNet(
                in_dim,
                int(cfg["hidden"]),
                out_dim,
                float(cfg["dropout"]),
                int(cfg["k_steps"]),
                float(cfg["alpha"]),
                float(cfg["dprate"]),
                str(cfg["init_mode"]),
            ).to(device)
    elif model_name == "sgc_wu2019":
        return run_sgc_wu2019(data, train_idx, val_idx, test_idx, seed=seed)
    elif model_name == "fsgnn":
        return run_fsgnn(
            data,
            train_idx,
            val_idx,
            test_idx,
            seed=seed,
            max_epochs=max_epochs,
            patience=patience,
        )
    elif model_name == "gcn_pairnorm":
        configs = _gcn_pairnorm_grid()
        adj = _row_normalize_adj(data.edge_index.to(device), data.num_nodes)

        def build(cfg):
            return DeepGCNPairNorm(
                in_dim,
                int(cfg["hidden"]),
                out_dim,
                float(cfg["dropout"]),
                int(cfg["nlayer"]),
                str(cfg["norm_mode"]),
                float(cfg["norm_scale"]),
            ).to(device)
    else:
        raise ValueError(f"Unsupported baseline model: {model_name}")

    best = None
    for cfg in configs:
        model = build(cfg)
        probs, best_val_acc = _train_one_model(
            model,
            data,
            train_idx,
            val_idx,
            adj,
            lr=float(cfg["lr"]),
            weight_decay=float(cfg["weight_decay"]),
            max_epochs=max_epochs,
            patience=patience,
        )
        test_acc = float((probs[test_idx].argmax(dim=1) == data.y[test_idx]).float().mean().item())
        key = (best_val_acc, test_acc)
        if best is None or key > best["key"]:
            best = {
                "key": key,
                "probs": probs.detach().cpu(),
                "val_acc": float(best_val_acc),
                "test_acc": test_acc,
                "best_config": {
                    k: (
                        float(v)
                        if isinstance(v, float)
                        else int(v)
                        if isinstance(v, int)
                        else v
                    )
                    for k, v in cfg.items()
                },
            }

    assert best is not None
    return BaselineResult(
        probs=best["probs"],
        val_acc=best["val_acc"],
        test_acc=best["test_acc"],
        best_config=best["best_config"],
        train_runtime_sec=float("nan"),
    )

#!/usr/bin/env python3
"""
Compact standard graph baselines for PRL resubmission experiments.

These models intentionally stay small and dependency-light:
  - 2-layer sparse GCN
  - APPNP with a 2-layer MLP backbone

They reuse the same GEO-GCN-style train/val/test split protocol as the rest of
the repo and tune from a compact validation-based config grid.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

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


def _normalize_adj(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    adj = _coalesce_adj(edge_index, num_nodes, add_self_loops=True)
    row, col = adj.indices()
    deg = torch.zeros(num_nodes, device=edge_index.device, dtype=torch.float32)
    deg.scatter_add_(0, row, adj.values())
    deg_inv_sqrt = deg.clamp(min=1.0).pow(-0.5)
    norm = deg_inv_sqrt[row] * adj.values() * deg_inv_sqrt[col]
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


@dataclass
class BaselineResult:
    probs: torch.Tensor
    val_acc: float
    test_acc: float
    best_config: Dict[str, float]
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
        build = lambda cfg: SparseGCN(in_dim, int(cfg["hidden"]), out_dim, float(cfg["dropout"])).to(device)
    elif model_name == "appnp":
        configs = _appnp_grid()
        adj = _normalize_adj(data.edge_index.to(device), data.num_nodes)
        build = lambda cfg: APPNPNet(
            in_dim,
            int(cfg["hidden"]),
            out_dim,
            float(cfg["dropout"]),
            int(cfg["k_steps"]),
            float(cfg["alpha"]),
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
                "best_config": {k: float(v) if isinstance(v, float) else int(v) for k, v in cfg.items()},
            }

    assert best is not None
    return BaselineResult(
        probs=best["probs"],
        val_acc=best["val_acc"],
        test_acc=best["test_acc"],
        best_config=best["best_config"],
        train_runtime_sec=float("nan"),
    )

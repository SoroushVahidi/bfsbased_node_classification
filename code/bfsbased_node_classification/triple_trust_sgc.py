#!/usr/bin/env python3
"""TRIPLE_TRUST_SGC: Triple-Trust Selective Graph Correction.

Feature-first selective correction with three explicit trust components:
  1) class trust (tau_c)
  2) source-node trust (s_u)
  3) target-node labelability (ell_v)

This is an experimental method path and does not modify FINAL_V3 defaults.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


def _simple_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    return float((y_true == y_pred).mean())


def _norm01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi <= lo + 1e-12:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _to_neighbors(edge_index: torch.Tensor, n_nodes: int) -> Tuple[list[list[int]], np.ndarray, np.ndarray, np.ndarray]:
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_index = torch.unique(edge_index, dim=1)
    ei = edge_index.detach().cpu().numpy()
    src = ei[0].astype(np.int64)
    dst = ei[1].astype(np.int64)
    neighbors = [[] for _ in range(n_nodes)]
    for u, v in zip(src, dst):
        neighbors[int(u)].append(int(v))
    deg = np.array([len(nbrs) for nbrs in neighbors], dtype=np.float64)
    return neighbors, deg, src, dst


def _init_compat_from_train(y_true: np.ndarray, train_mask: np.ndarray, src: np.ndarray, dst: np.ndarray, n_classes: int) -> np.ndarray:
    compat = np.ones((n_classes, n_classes), dtype=np.float64)
    for u, v in zip(src, dst):
        if train_mask[u] and train_mask[v]:
            compat[y_true[u], y_true[v]] += 1.0
    rs = compat.sum(axis=1, keepdims=True)
    return compat / np.clip(rs, 1e-12, None)


def _compute_source_trust(
    train_mask: np.ndarray,
    pseudo_mask: np.ndarray,
    yhat: np.ndarray,
    q: np.ndarray,
    tau_class: np.ndarray,
    deg: np.ndarray,
    gamma_source: float,
    lambda_deg: float,
) -> np.ndarray:
    s = np.zeros_like(q, dtype=np.float64)
    s[train_mask] = 1.0
    if pseudo_mask.any():
        lbl = yhat[pseudo_mask]
        denom = 1.0 + float(lambda_deg) * np.log1p(deg[pseudo_mask])
        s[pseudo_mask] = (np.clip(q[pseudo_mask], 0.0, 1.0) ** float(gamma_source)) * tau_class[lbl] / np.clip(denom, 1e-12, None)
    return np.clip(s, 0.0, 1.0)


def _trusted_mass_and_tau(
    yhat: np.ndarray,
    s: np.ndarray,
    train_mask: np.ndarray,
    pseudo_update_mask: np.ndarray,
    n_classes: int,
    alpha_class: float,
) -> Tuple[np.ndarray, np.ndarray]:
    trusted_mass = np.zeros(n_classes, dtype=np.float64)
    for c in range(n_classes):
        trusted_mass[c] = float(s[(yhat == c) & (train_mask | pseudo_update_mask)].sum())
    tau = (trusted_mass + float(alpha_class)) / (trusted_mass.sum() + float(alpha_class) * float(n_classes))
    return trusted_mass, np.clip(tau, 0.0, 1.0)


def _compute_prototypes(
    x_np: np.ndarray,
    yhat: np.ndarray,
    train_mask: np.ndarray,
    pseudo_mask: np.ndarray,
    pseudo_update_mask: np.ndarray,
    s: np.ndarray,
    n_classes: int,
    lambda_proto: float,
    eps: float,
) -> np.ndarray:
    d = x_np.shape[1]
    mu = np.zeros((n_classes, d), dtype=np.float64)
    for c in range(n_classes):
        mask_train = (yhat == c) & train_mask
        mask_pseudo = (yhat == c) & pseudo_mask & pseudo_update_mask
        weights = np.zeros(x_np.shape[0], dtype=np.float64)
        weights[mask_train] = 1.0
        weights[mask_pseudo] = float(lambda_proto) * s[mask_pseudo]
        wsum = float(weights.sum())
        if wsum > eps:
            mu[c] = (weights[:, None] * x_np).sum(axis=0) / wsum
    return mu


def _feature_similarity_from_prototypes(x_np: np.ndarray, prototypes: np.ndarray, eps: float) -> np.ndarray:
    x_norm = np.linalg.norm(x_np, axis=1, keepdims=True) + eps
    proto_norm = np.linalg.norm(prototypes, axis=1)
    valid_proto = proto_norm > eps

    transformed = np.zeros((x_np.shape[0], prototypes.shape[0]), dtype=np.float64)
    if not np.any(valid_proto):
        return transformed

    valid_prototypes = prototypes[valid_proto]
    valid_p_norm = proto_norm[valid_proto][np.newaxis, :]
    sim = (x_np @ valid_prototypes.T) / (x_norm * valid_p_norm)
    transformed[:, valid_proto] = np.clip((sim + 1.0) / 2.0, 0.0, 1.0)
    return transformed
def _compute_trust_neighbor_and_labelability(
    neighbors: list[list[int]],
    yhat: np.ndarray,
    s: np.ndarray,
    deg: np.ndarray,
    n_classes: int,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_nodes = yhat.shape[0]
    ns = np.zeros((n_nodes, n_classes), dtype=np.float64)
    labelability = np.zeros(n_nodes, dtype=np.float64)
    neigh_trust_mass = np.zeros(n_nodes, dtype=np.float64)
    for v in range(n_nodes):
        nbrs = neighbors[v]
        if not nbrs:
            continue
        mass = float(np.sum(s[nbrs]))
        neigh_trust_mass[v] = mass
        if mass <= eps:
            continue
        for u in nbrs:
            ns[v, yhat[u]] += s[u]
        ns[v] /= (mass + eps)
        labelability[v] = (mass / (deg[v] + eps)) * float(np.max(ns[v]))
    return ns, np.clip(labelability, 0.0, 1.0), neigh_trust_mass


def _compute_trust_compat_support(
    neighbors: list[list[int]],
    yhat: np.ndarray,
    s: np.ndarray,
    tau_class: np.ndarray,
    compat: np.ndarray,
    n_classes: int,
    eps: float,
) -> np.ndarray:
    n_nodes = yhat.shape[0]
    cs = np.zeros((n_nodes, n_classes), dtype=np.float64)
    for v in range(n_nodes):
        nbrs = neighbors[v]
        if not nbrs:
            continue
        mass = float(np.sum(s[nbrs]))
        if mass <= eps:
            continue
        base = np.zeros(n_classes, dtype=np.float64)
        for u in nbrs:
            base += s[u] * compat[yhat[u]]
        cs[v] = tau_class * (base / (mass + eps))
    return np.clip(cs, 0.0, 1.0)


def _compat_update(
    compat: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    yhat: np.ndarray,
    s: np.ndarray,
    trusted_source_mask: np.ndarray,
    eta: float,
    n_classes: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    est = np.ones((n_classes, n_classes), dtype=np.float64)
    used_edges = 0
    for u, v in zip(src, dst):
        if trusted_source_mask[u] and trusted_source_mask[v]:
            w = float(s[u] * s[v])
            if w <= 0.0:
                continue
            est[yhat[u], yhat[v]] += w
            used_edges += 1
    est /= np.clip(est.sum(axis=1, keepdims=True), 1e-12, None)
    out = (1.0 - float(eta)) * compat + float(eta) * est
    out /= np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)
    return out, {"used_edges": float(used_edges), "eta": float(eta)}


def triple_trust_sgc_predictclass(
    data,
    train_indices,
    val_indices,
    test_indices,
    *,
    mlp_probs: Optional[torch.Tensor] = None,
    seed: int = 1337,
    mod=None,
    alpha_class: float = 0.5,
    gamma_source: float = 2.0,
    lambda_deg: float = 0.2,
    lambda_proto: float = 0.3,
    tau_uncertain: Optional[float] = None,
    rho_target: float = 0.15,
    kappa_update: float = 0.65,
    q_weights: Tuple[float, float, float, float] = (0.35, 0.30, 0.25, 0.10),
    refresh_fraction: float = 0.25,
    compatibility_update_eta: float = 0.2,
    enable_target_labelability_gate: bool = True,
    b1: float = 1.0,
    b2: float = 0.6,
    b4: float = 0.5,
    b5: float = 0.3,
    max_refresh_rounds: int = 3,
    eps: float = 1e-9,
) -> Tuple[float, float, Dict[str, Any]]:
    """Run TRIPLE_TRUST_SGC on one split.

    Returns (val_acc, test_acc, info).
    """
    _ = seed  # deterministic behavior comes from caller-set seeds in current runners.

    if mod is None:
        raise ValueError(
            "mod must be provided; it is required for compute_mlp_margin and related utilities. "
            "Pass the manuscript module as mod=, and optionally supply pre-computed mlp_probs."
        )

    y_true = data.y.detach().cpu().numpy().astype(np.int64)
    n_nodes = int(data.num_nodes)
    n_classes = int(max(int(y_true.max()) + 1, int(data.y.max().item()) + 1))
    train_np = np.asarray(train_indices, dtype=np.int64)
    val_np = np.asarray(val_indices, dtype=np.int64)
    test_np = np.asarray(test_indices, dtype=np.int64)

    train_mask = np.zeros(n_nodes, dtype=bool)
    train_mask[train_np] = True
    pseudo_mask = ~train_mask

    if mlp_probs is None:
        mlp_probs, _ = mod.train_mlp_and_predict(
            data,
            train_np,
            **mod.DEFAULT_MANUSCRIPT_MLP_KWARGS,
            log_file=None,
        )

    mlp_info = mod.compute_mlp_margin(mlp_probs)
    mlp_probs_np = mlp_info["mlp_probs_np"]
    mlp_pred = mlp_info["mlp_pred_all"]
    mlp_margin = mlp_info["mlp_margin_all"]
    norm_margin = _norm01(mlp_margin)

    neighbors, deg, src, dst = _to_neighbors(data.edge_index, n_nodes)

    x_np = data.x.detach().cpu().numpy().astype(np.float64)
    if train_np.size > 1:
        mu = x_np[train_np].mean(axis=0, keepdims=True)
        sd = x_np[train_np].std(axis=0, keepdims=True)
        x_np = (x_np - mu) / np.clip(sd, 1e-6, None)

    # init state
    yhat = mlp_pred.copy()
    # Keep labeled training nodes fixed to their ground-truth labels.
    yhat[train_mask] = y_true[train_mask]
    q = norm_margin.copy()
    tau_class = np.full(n_classes, 1.0 / float(n_classes), dtype=np.float64)
    s = _compute_source_trust(train_mask, pseudo_mask, yhat, q, tau_class, deg, gamma_source, lambda_deg)

    # Training nodes must never be treated as pseudo-updated/corrected nodes.
    pseudo_update_mask = np.zeros(n_nodes, dtype=bool)
    pseudo_update_mask[train_mask] = False
    trusted_mass, tau_class = _trusted_mass_and_tau(yhat, s, train_mask, pseudo_update_mask, n_classes, alpha_class)
    compat = _init_compat_from_train(y_true, train_mask, src, dst, n_classes)

    w = np.asarray(q_weights, dtype=np.float64)
    if w.size != 4:
        raise ValueError("q_weights must have 4 entries")
    sw = float(w.sum())
    if sw <= 0:
        raise ValueError("q_weights must have positive sum")
    w = w / sw

    if tau_uncertain is None:
        v_margin = mlp_margin[val_np]
        tau_uncertain = float(np.quantile(v_margin, 0.40))

    refresh_fraction = float(np.clip(refresh_fraction, 0.05, 1.0))
    max_refresh_rounds = int(max(1, max_refresh_rounds))

    last_scores = None
    last_ns = np.zeros((n_nodes, n_classes), dtype=np.float64)
    last_ell = np.zeros(n_nodes, dtype=np.float64)
    compat_updates = []

    for _round in range(max_refresh_rounds):
        prototypes = _compute_prototypes(
            x_np,
            yhat,
            train_mask,
            pseudo_mask,
            pseudo_update_mask,
            s,
            n_classes,
            lambda_proto,
            eps,
        )
        dmat = _feature_similarity_from_prototypes(x_np, prototypes, eps)

        ns, ell, neigh_mass = _compute_trust_neighbor_and_labelability(neighbors, yhat, s, deg, n_classes, eps)
        cs = _compute_trust_compat_support(neighbors, yhat, s, tau_class, compat, n_classes, eps)

        scores = float(b1) * np.log(np.clip(mlp_probs_np, eps, 1.0)) + float(b2) * dmat + float(b4) * ns + float(b5) * cs
        pred_scores = np.argmax(scores, axis=1).astype(np.int64)
        top1 = np.max(scores, axis=1)
        tmp = scores.copy()
        tmp[np.arange(n_nodes), pred_scores] = -1e12
        top2 = np.max(tmp, axis=1)
        score_gap = _norm01(top1 - top2)

        neighbor_vote = np.argmax(ns, axis=1)
        has_trusted_neighbor_mass = neigh_mass > float(eps)
        agreement_signal = np.where(
            has_trusted_neighbor_mass,
            (mlp_pred == neighbor_vote).astype(np.float64),
            0.0,
        )
        q = np.clip(w[0] * norm_margin + w[1] * score_gap + w[2] * ell + w[3] * agreement_signal, 0.0, 1.0)

        eligible_corr = mlp_margin < float(tau_uncertain)
        if enable_target_labelability_gate:
            eligible_corr = eligible_corr & (ell >= float(rho_target))

        final_pred = mlp_pred.copy()
        final_pred[eligible_corr] = pred_scores[eligible_corr]
        yhat = final_pred

        s = _compute_source_trust(train_mask, pseudo_mask, yhat, q, tau_class, deg, gamma_source, lambda_deg)

        # Conservative state update: only top-q pseudo nodes (batch refresh) and above kappa_update.
        pseudo_ids = np.where(pseudo_mask)[0]
        pseudo_update_mask = np.zeros(n_nodes, dtype=bool)
        if pseudo_ids.size > 0:
            n_keep = max(1, int(math.ceil(refresh_fraction * pseudo_ids.size)))
            order = pseudo_ids[np.argsort(-q[pseudo_ids])]
            keep = order[:n_keep]
            pseudo_update_mask[keep] = q[keep] >= float(kappa_update)

        trusted_mass, tau_class = _trusted_mass_and_tau(
            yhat,
            s,
            train_mask,
            pseudo_update_mask,
            n_classes,
            alpha_class,
        )

        trusted_sources = train_mask | pseudo_update_mask
        compat, upd_stats = _compat_update(
            compat,
            src,
            dst,
            yhat,
            s,
            trusted_sources,
            compatibility_update_eta,
            n_classes,
        )
        compat_updates.append(upd_stats)

        last_scores = scores
        last_ns = ns
        last_ell = ell

    assert last_scores is not None

    # final pass for reporting predictions using last state
    pred_final = np.argmax(last_scores, axis=1).astype(np.int64)
    corr_mask = (mlp_margin < float(tau_uncertain))
    if enable_target_labelability_gate:
        corr_mask = corr_mask & (last_ell >= float(rho_target))
    final_pred = mlp_pred.copy()
    final_pred[corr_mask] = pred_final[corr_mask]

    val_acc = _simple_accuracy(y_true[val_np], final_pred[val_np])
    test_acc = _simple_accuracy(y_true[test_np], final_pred[test_np])
    mlp_test_acc = _simple_accuracy(y_true[test_np], mlp_pred[test_np])

    changed_test = test_np[final_pred[test_np] != mlp_pred[test_np]]
    n_helped = n_hurt = 0
    if changed_test.size > 0:
        before = mlp_pred[changed_test] == y_true[changed_test]
        after = final_pred[changed_test] == y_true[changed_test]
        n_helped = int((~before & after).sum())
        n_hurt = int((before & ~after).sum())

    pseudo_ids = np.where(pseudo_mask)[0]
    update_count = int(pseudo_update_mask[pseudo_ids].sum()) if pseudo_ids.size > 0 else 0
    ns_conc = np.max(last_ns, axis=1)

    info: Dict[str, Any] = {
        "variant": "TRIPLE_TRUST_SGC",
        "mode": "triple_trust_selective_graph_correction_v1",
        "val_acc_selective": float(val_acc),
        "test_acc_selective": float(test_acc),
        "test_acc_mlp": float(mlp_test_acc),
        "delta_vs_mlp": float(test_acc - mlp_test_acc),
        "selected_threshold_high": float(tau_uncertain),
        "rho_target": float(rho_target),
        "kappa_update": float(kappa_update),
        "enable_target_labelability_gate": bool(enable_target_labelability_gate),
        "selected_weights": {"b1": float(b1), "b2": float(b2), "b3": 0.0, "b4": float(b4), "b5": float(b5), "b6": 0.0},
        "q_weights": [float(x) for x in w.tolist()],
        "fraction_test_nodes_uncertain": float((mlp_margin[test_np] < float(tau_uncertain)).mean()),
        "fraction_test_nodes_changed_from_mlp": float((final_pred[test_np] != mlp_pred[test_np]).mean()),
        "fraction_test_nodes_corrected": float(corr_mask[test_np].mean()),
        "mean_q_corrected": float(q[test_np][corr_mask[test_np]].mean()) if corr_mask[test_np].any() else 0.0,
        "mean_source_trust_pseudo": float(s[pseudo_ids].mean()) if pseudo_ids.size > 0 else 0.0,
        "average_target_labelability": float(last_ell.mean()),
        "pseudo_update_eligible_count": int(update_count),
        "trust_neighbor_concentration_mean": float(ns_conc.mean()),
        "trust_neighbor_concentration_corrected_mean": float(ns_conc[test_np][corr_mask[test_np]].mean()) if corr_mask[test_np].any() else 0.0,
        "per_class_trusted_mass": [float(x) for x in trusted_mass.tolist()],
        "class_trust_vector": [float(x) for x in tau_class.tolist()],
        "compatibility_update_stats": compat_updates,
        "correction_analysis": {
            "n_changed": int(changed_test.size),
            "n_helped": int(n_helped),
            "n_hurt": int(n_hurt),
        },
    }
    return val_acc, test_acc, info

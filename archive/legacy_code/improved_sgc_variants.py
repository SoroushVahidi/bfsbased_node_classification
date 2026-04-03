#!/usr/bin/env python3
"""
Improved SGC (Selective Graph Correction) variants.

**Publication method:** FINAL_V3 is implemented in ``final_method_v3.py`` (not in this module).
This file retains **baselines and ablations** used for comparison: original selective correction
(SGC v1) and **V2_MULTIBRANCH** (archived multibranch design). Other tagged variants are
experimental and not part of the main PRL narrative.

Each variant builds on the existing evidence infrastructure in
bfsbased-full-investigate-homophil.py but upgrades the gating, mixing,
or scoring strategy.

Variants implemented:
  V1_RELIABILITY  – Per-node reliability-aware correction suppression
  V2_MULTIBRANCH  – 3-way branch: no-correct / feature-only / graph+feature
  V3_LEARNED_MIX  – Node-wise learned mixing weights via logistic model
  V4_ENTROPY_GATE – Entropy-based uncertainty + calibrated gate
  V5_KNN_ENHANCED – Feature-kNN fully enabled with refined integration
  V6_COMBO        – Best ideas combined: reliability + kNN + entropy gate
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def _simple_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.size == 0:
        return 0.0
    return float((y_true == y_pred).mean())


def _top1_top2_margin(scores: np.ndarray):
    scores = np.asarray(scores, dtype=np.float64)
    pred = np.argmax(scores, axis=1).astype(np.int64)
    top1 = scores[np.arange(scores.shape[0]), pred]
    tmp = scores.copy()
    tmp[np.arange(scores.shape[0]), pred] = -1e100
    top2 = tmp.max(axis=1)
    margin = top1 - top2
    return pred, top1, top2, margin


def _safe_sigmoid(x):
    x = np.clip(np.asarray(x, dtype=np.float64), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def _entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise entropy of a probability matrix [N, C]."""
    p = np.clip(probs, eps, 1.0)
    return -(p * np.log(p)).sum(axis=1)


# ---------------------------------------------------------------------------
# Per-node reliability score (used by V1, V6)
# ---------------------------------------------------------------------------

def compute_node_reliability(
    mlp_probs_np: np.ndarray,
    evidence: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    Compute a per-node reliability score in [0, 1] for graph correction.

    High reliability = graph evidence is likely trustworthy for this node.
    Low reliability = graph correction is risky (e.g. heterophilic neighborhood).

    Signals combined:
      1. neighbor_agreement: do graph neighbors agree on a class?
      2. mlp_graph_agreement: does MLP prediction match graph neighbor vote?
      3. feature_graph_agreement: does feature similarity agree with graph vote?
      4. low_entropy: is neighbor class distribution concentrated?
      5. degree_confidence: higher degree → more neighbor signal
    """
    N = mlp_probs_np.shape[0]
    C = mlp_probs_np.shape[1]

    mlp_pred = np.argmax(mlp_probs_np, axis=1)
    graph_ns = evidence["graph_neighbor_support"]
    feat_sim = evidence["feature_similarity"]
    entropy_nbr = evidence["neighbor_class_entropy"]
    deg = evidence["node_degree"]

    graph_pred = np.argmax(graph_ns, axis=1)
    graph_max = graph_ns.max(axis=1)
    feat_pred = np.argmax(feat_sim, axis=1)

    neighbor_agreement = np.clip(graph_max, 0, 1)

    mlp_graph_agree = (mlp_pred == graph_pred).astype(np.float64)

    feat_graph_agree = (feat_pred == graph_pred).astype(np.float64)

    max_entropy = np.log(max(C, 2))
    low_entropy = 1.0 - np.clip(entropy_nbr / max_entropy, 0, 1)

    deg_conf = 1.0 - np.exp(-deg / 5.0)

    reliability = (
        0.25 * neighbor_agreement
        + 0.25 * mlp_graph_agree
        + 0.15 * feat_graph_agree
        + 0.20 * low_entropy
        + 0.15 * deg_conf
    )
    return np.clip(reliability, 0.0, 1.0)


# ---------------------------------------------------------------------------
# V1: Reliability-aware selective correction
# ---------------------------------------------------------------------------

def v1_reliability_correction(
    data,
    train_indices,
    val_indices,
    test_indices,
    *,
    mlp_probs: Optional[torch.Tensor] = None,
    seed: int = 1337,
    mod=None,
    reliability_threshold_candidates: Optional[List[float]] = None,
    weight_candidates: Optional[List[Dict[str, float]]] = None,
    threshold_candidates: Optional[List[float]] = None,
    enable_feature_knn: bool = False,
    feature_knn_k: int = 5,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Like baseline SGC but adds a per-node reliability filter:
    correction is only applied if node reliability >= r_threshold.
    """
    y_true = data.y.detach().cpu().numpy().astype(np.int64)
    train_np = np.asarray(train_indices, dtype=np.int64)
    val_np = np.asarray(val_indices, dtype=np.int64)
    test_np = np.asarray(test_indices, dtype=np.int64)

    if mlp_probs is None:
        mlp_probs, _ = mod.train_mlp_and_predict(
            data, train_np,
            **mod.DEFAULT_MANUSCRIPT_MLP_KWARGS, log_file=None,
        )

    mlp_info = mod.compute_mlp_margin(mlp_probs)
    mlp_probs_np = mlp_info["mlp_probs_np"]
    mlp_pred_all = mlp_info["mlp_pred_all"]
    mlp_margin_all = mlp_info["mlp_margin_all"]

    evidence = mod._build_selective_correction_evidence(
        data, train_np, mlp_probs_np=mlp_probs_np,
        enable_feature_knn=enable_feature_knn, feature_knn_k=feature_knn_k,
    )

    reliability = compute_node_reliability(mlp_probs_np, evidence)

    if weight_candidates is None:
        weight_candidates = [
            {"b1": 1.0, "b2": 0.6, "b3": 0.0, "b4": 0.5, "b5": 0.3, "b6": 0.0},
            {"b1": 1.0, "b2": 0.9, "b3": 0.0, "b4": 0.3, "b5": 0.2, "b6": 0.0},
            {"b1": 1.0, "b2": 0.4, "b3": 0.0, "b4": 0.9, "b5": 0.5, "b6": 0.0},
            {"b1": 1.2, "b2": 0.5, "b3": 0.0, "b4": 0.4, "b5": 0.2, "b6": 0.0},
        ]

    if reliability_threshold_candidates is None:
        rq = np.quantile(reliability[val_np], [0.2, 0.3, 0.4, 0.5, 0.6])
        reliability_threshold_candidates = sorted(set(
            np.round(np.concatenate([rq, [0.3, 0.4, 0.5, 0.6, 0.7]]), 4).tolist()
        ))

    if threshold_candidates is None:
        val_margins = mlp_margin_all[val_np]
        q = np.quantile(val_margins, [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        base = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
        threshold_candidates = sorted(set(np.round(np.concatenate([q, base]), 6).tolist()))

    best_key = None
    best_cfg = None
    best_scores = None

    for w_cfg in weight_candidates:
        comps = mod.build_selective_correction_scores(
            mlp_probs_np, evidence,
            b1=w_cfg["b1"], b2=w_cfg["b2"], b3=w_cfg["b3"],
            b4=w_cfg["b4"], b5=w_cfg["b5"], b6=w_cfg.get("b6", 0.0),
        )
        combined = comps["combined_scores"]

        for t in threshold_candidates:
            uncertain_val = mlp_margin_all[val_np] < t
            for r_thr in reliability_threshold_candidates:
                reliable_val = reliability[val_np] >= r_thr
                apply_val = uncertain_val & reliable_val
                final_val = mlp_pred_all[val_np].copy()
                if apply_val.any():
                    final_val[apply_val] = np.argmax(
                        combined[val_np][apply_val], axis=1
                    ).astype(np.int64)
                val_acc = _simple_accuracy(y_true[val_np], final_val)
                changed = float((final_val != mlp_pred_all[val_np]).mean())
                unc_frac = float(apply_val.mean())
                key = (val_acc, -changed, -unc_frac)
                if best_key is None or key > best_key:
                    best_key = key
                    best_cfg = {
                        "weights": dict(w_cfg),
                        "threshold": t,
                        "reliability_threshold": r_thr,
                    }
                    best_scores = combined

    t = best_cfg["threshold"]
    r_thr = best_cfg["reliability_threshold"]
    uncertain_all = mlp_margin_all < t
    reliable_all = reliability >= r_thr
    apply_all = uncertain_all & reliable_all

    final_pred = mlp_pred_all.copy()
    if apply_all.any():
        final_pred[apply_all] = np.argmax(best_scores[apply_all], axis=1).astype(np.int64)

    test_acc = _simple_accuracy(y_true[test_np], final_pred[test_np])
    val_acc = _simple_accuracy(y_true[val_np], final_pred[val_np])
    mlp_test_acc = _simple_accuracy(y_true[test_np], mlp_pred_all[test_np])

    info = {
        "variant": "V1_RELIABILITY",
        "val_acc_selective": val_acc,
        "test_acc_mlp": mlp_test_acc,
        "test_acc_selective": test_acc,
        "selected_threshold": t,
        "selected_reliability_threshold": r_thr,
        "selected_weights": best_cfg["weights"],
        "fraction_test_uncertain": float(uncertain_all[test_np].mean()),
        "fraction_test_reliable": float(reliable_all[test_np].mean()),
        "fraction_test_corrected": float(apply_all[test_np].mean()),
        "fraction_test_changed": float((final_pred[test_np] != mlp_pred_all[test_np]).mean()),
    }
    return val_acc, test_acc, info


# ---------------------------------------------------------------------------
# V2: Multi-branch correction (no-correct / feature-only / full)
# ---------------------------------------------------------------------------

def v2_multibranch_correction(
    data,
    train_indices,
    val_indices,
    test_indices,
    *,
    mlp_probs: Optional[torch.Tensor] = None,
    seed: int = 1337,
    mod=None,
    threshold_candidates: Optional[List[float]] = None,
    enable_feature_knn: bool = False,
    feature_knn_k: int = 5,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    3-way branch:
      - confident nodes → keep MLP (no correction)
      - uncertain + low reliability → feature-only correction (b4=0, b5=0)
      - uncertain + high reliability → full graph+feature correction
    """
    y_true = data.y.detach().cpu().numpy().astype(np.int64)
    train_np = np.asarray(train_indices, dtype=np.int64)
    val_np = np.asarray(val_indices, dtype=np.int64)
    test_np = np.asarray(test_indices, dtype=np.int64)

    if mlp_probs is None:
        mlp_probs, _ = mod.train_mlp_and_predict(
            data, train_np,
            **mod.DEFAULT_MANUSCRIPT_MLP_KWARGS, log_file=None,
        )

    mlp_info = mod.compute_mlp_margin(mlp_probs)
    mlp_probs_np = mlp_info["mlp_probs_np"]
    mlp_pred_all = mlp_info["mlp_pred_all"]
    mlp_margin_all = mlp_info["mlp_margin_all"]

    evidence = mod._build_selective_correction_evidence(
        data, train_np, mlp_probs_np=mlp_probs_np,
        enable_feature_knn=enable_feature_knn, feature_knn_k=feature_knn_k,
    )
    reliability = compute_node_reliability(mlp_probs_np, evidence)

    feat_only_scores = mod.build_selective_correction_scores(
        mlp_probs_np, evidence,
        b1=1.0, b2=0.8, b3=0.3 if enable_feature_knn else 0.0,
        b4=0.0, b5=0.0, b6=0.0,
    )["combined_scores"]

    full_configs = [
        {"b1": 1.0, "b2": 0.6, "b3": 0.0, "b4": 0.5, "b5": 0.3, "b6": 0.0},
        {"b1": 1.0, "b2": 0.4, "b3": 0.0, "b4": 0.9, "b5": 0.5, "b6": 0.0},
        {"b1": 1.0, "b2": 0.9, "b3": 0.0, "b4": 0.3, "b5": 0.2, "b6": 0.0},
    ]

    if threshold_candidates is None:
        val_margins = mlp_margin_all[val_np]
        q = np.quantile(val_margins, [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        base = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
        threshold_candidates = sorted(set(np.round(np.concatenate([q, base]), 6).tolist()))

    rel_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    best_key = None
    best_cfg = None
    best_full_scores = None

    for full_w in full_configs:
        full_scores = mod.build_selective_correction_scores(
            mlp_probs_np, evidence,
            b1=full_w["b1"], b2=full_w["b2"], b3=full_w["b3"],
            b4=full_w["b4"], b5=full_w["b5"], b6=full_w.get("b6", 0.0),
        )["combined_scores"]

        for t in threshold_candidates:
            uncertain_val = mlp_margin_all[val_np] < t
            for r_thr in rel_thresholds:
                reliable_val = reliability[val_np] >= r_thr
                feat_branch = uncertain_val & ~reliable_val
                full_branch = uncertain_val & reliable_val

                final_val = mlp_pred_all[val_np].copy()
                if feat_branch.any():
                    final_val[feat_branch] = np.argmax(
                        feat_only_scores[val_np][feat_branch], axis=1
                    ).astype(np.int64)
                if full_branch.any():
                    final_val[full_branch] = np.argmax(
                        full_scores[val_np][full_branch], axis=1
                    ).astype(np.int64)

                val_acc = _simple_accuracy(y_true[val_np], final_val)
                changed = float((final_val != mlp_pred_all[val_np]).mean())
                key = (val_acc, -changed)
                if best_key is None or key > best_key:
                    best_key = key
                    best_cfg = {
                        "full_weights": dict(full_w),
                        "threshold": t,
                        "reliability_threshold": r_thr,
                    }
                    best_full_scores = full_scores

    t = best_cfg["threshold"]
    r_thr = best_cfg["reliability_threshold"]
    uncertain_all = mlp_margin_all < t
    reliable_all = reliability >= r_thr
    feat_branch_all = uncertain_all & ~reliable_all
    full_branch_all = uncertain_all & reliable_all

    final_pred = mlp_pred_all.copy()
    if feat_branch_all.any():
        final_pred[feat_branch_all] = np.argmax(
            feat_only_scores[feat_branch_all], axis=1
        ).astype(np.int64)
    if full_branch_all.any():
        final_pred[full_branch_all] = np.argmax(
            best_full_scores[full_branch_all], axis=1
        ).astype(np.int64)

    test_acc = _simple_accuracy(y_true[test_np], final_pred[test_np])
    val_acc = _simple_accuracy(y_true[val_np], final_pred[val_np])
    mlp_test_acc = _simple_accuracy(y_true[test_np], mlp_pred_all[test_np])

    info = {
        "variant": "V2_MULTIBRANCH",
        "val_acc_selective": val_acc,
        "test_acc_mlp": mlp_test_acc,
        "test_acc_selective": test_acc,
        "selected_threshold": t,
        "selected_reliability_threshold": r_thr,
        "selected_full_weights": best_cfg["full_weights"],
        "fraction_test_feat_branch": float(feat_branch_all[test_np].mean()),
        "fraction_test_full_branch": float(full_branch_all[test_np].mean()),
        "fraction_test_no_correct": float((~uncertain_all)[test_np].mean()),
    }
    return val_acc, test_acc, info


# ---------------------------------------------------------------------------
# V3: Learned node-wise mixing
# ---------------------------------------------------------------------------

def v3_learned_mixing(
    data,
    train_indices,
    val_indices,
    test_indices,
    *,
    mlp_probs: Optional[torch.Tensor] = None,
    seed: int = 1337,
    mod=None,
    enable_feature_knn: bool = False,
    feature_knn_k: int = 5,
    gate_epochs: int = 200,
    gate_lr: float = 0.02,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Train a lightweight logistic model on val data to predict
    whether graph correction improves over MLP per node.
    Then use the learned gate probability as a soft mixing weight.
    """
    y_true = data.y.detach().cpu().numpy().astype(np.int64)
    train_np = np.asarray(train_indices, dtype=np.int64)
    val_np = np.asarray(val_indices, dtype=np.int64)
    test_np = np.asarray(test_indices, dtype=np.int64)

    if mlp_probs is None:
        mlp_probs, _ = mod.train_mlp_and_predict(
            data, train_np,
            **mod.DEFAULT_MANUSCRIPT_MLP_KWARGS, log_file=None,
        )

    mlp_info = mod.compute_mlp_margin(mlp_probs)
    mlp_probs_np = mlp_info["mlp_probs_np"]
    mlp_pred_all = mlp_info["mlp_pred_all"]
    mlp_margin_all = mlp_info["mlp_margin_all"]

    evidence = mod._build_selective_correction_evidence(
        data, train_np, mlp_probs_np=mlp_probs_np,
        enable_feature_knn=enable_feature_knn, feature_knn_k=feature_knn_k,
    )
    reliability = compute_node_reliability(mlp_probs_np, evidence)

    best_w = {"b1": 1.0, "b2": 0.6, "b3": 0.0, "b4": 0.5, "b5": 0.3, "b6": 0.0}
    combined = mod.build_selective_correction_scores(
        mlp_probs_np, evidence, **best_w,
    )["combined_scores"]
    sgc_pred_all = np.argmax(combined, axis=1).astype(np.int64)

    mlp_entropy = _entropy(mlp_probs_np)
    max_ent = np.log(max(mlp_probs_np.shape[1], 2))
    mlp_entropy_norm = mlp_entropy / max_ent

    graph_ns = evidence["graph_neighbor_support"]
    graph_pred = np.argmax(graph_ns, axis=1)
    graph_max = graph_ns.max(axis=1)

    feat_sim = evidence["feature_similarity"]
    feat_pred = np.argmax(feat_sim, axis=1)

    gate_features_all = np.column_stack([
        mlp_margin_all,
        mlp_probs_np.max(axis=1),
        mlp_entropy_norm,
        reliability,
        graph_max,
        (mlp_pred_all == graph_pred).astype(np.float64),
        (mlp_pred_all == feat_pred).astype(np.float64),
        (graph_pred == feat_pred).astype(np.float64),
        evidence["neighbor_class_entropy"] / max_ent,
        np.clip(evidence["node_degree"] / 20.0, 0, 1),
        evidence["support_train_neighbors"].astype(np.float64) / max(evidence["support_train_neighbors"].max(), 1),
    ]).astype(np.float32)

    val_mlp_correct = (mlp_pred_all[val_np] == y_true[val_np])
    val_sgc_correct = (sgc_pred_all[val_np] == y_true[val_np])
    val_sgc_better = (~val_mlp_correct & val_sgc_correct).astype(np.float32)
    val_sgc_worse = (val_mlp_correct & ~val_sgc_correct).astype(np.float32)
    val_labels = np.zeros(len(val_np), dtype=np.float32)
    val_labels[val_sgc_better.astype(bool)] = 1.0
    val_labels[val_sgc_worse.astype(bool)] = 0.0
    both_right = val_mlp_correct & val_sgc_correct
    val_labels[both_right] = 0.3
    both_wrong = ~val_mlp_correct & ~val_sgc_correct
    val_labels[both_wrong] = 0.5

    X_val = gate_features_all[val_np]
    torch.manual_seed(seed)
    X_t = torch.from_numpy(X_val)
    y_t = torch.from_numpy(val_labels).unsqueeze(1)
    mean = X_t.mean(dim=0, keepdim=True)
    std = X_t.std(dim=0, keepdim=True).clamp_min(1e-6)
    Xn = (X_t - mean) / std

    gate_model = nn.Linear(Xn.size(1), 1)
    opt = optim.Adam(gate_model.parameters(), lr=gate_lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(gate_epochs):
        opt.zero_grad()
        loss = loss_fn(gate_model(Xn), y_t)
        loss.backward()
        opt.step()

    with torch.no_grad():
        X_all_t = torch.from_numpy(gate_features_all.astype(np.float32))
        Xn_all = (X_all_t - mean) / std
        gate_probs = torch.sigmoid(gate_model(Xn_all)).numpy().flatten()

    N = mlp_probs_np.shape[0]
    C = mlp_probs_np.shape[1]
    mlp_log = np.log(np.clip(mlp_probs_np, 1e-9, 1.0))
    sgc_log = np.log(np.clip(
        np.exp(combined) / np.exp(combined).sum(axis=1, keepdims=True), 1e-9, 1.0
    ))

    alpha = gate_probs[:, None]
    blended = (1.0 - alpha) * mlp_log + alpha * sgc_log
    final_pred = np.argmax(blended, axis=1).astype(np.int64)

    test_acc = _simple_accuracy(y_true[test_np], final_pred[test_np])
    val_acc = _simple_accuracy(y_true[val_np], final_pred[val_np])
    mlp_test_acc = _simple_accuracy(y_true[test_np], mlp_pred_all[test_np])

    info = {
        "variant": "V3_LEARNED_MIX",
        "val_acc_selective": val_acc,
        "test_acc_mlp": mlp_test_acc,
        "test_acc_selective": test_acc,
        "mean_gate_prob_test": float(gate_probs[test_np].mean()),
        "fraction_test_gate_above_0.5": float((gate_probs[test_np] > 0.5).mean()),
    }
    return val_acc, test_acc, info


# ---------------------------------------------------------------------------
# V4: Entropy-based uncertainty gate
# ---------------------------------------------------------------------------

def v4_entropy_gate(
    data,
    train_indices,
    val_indices,
    test_indices,
    *,
    mlp_probs: Optional[torch.Tensor] = None,
    seed: int = 1337,
    mod=None,
    weight_candidates: Optional[List[Dict[str, float]]] = None,
    enable_feature_knn: bool = False,
    feature_knn_k: int = 5,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Replace margin-based uncertainty with entropy-based uncertainty.
    Use calibrated percentile thresholds for the gate.
    """
    y_true = data.y.detach().cpu().numpy().astype(np.int64)
    train_np = np.asarray(train_indices, dtype=np.int64)
    val_np = np.asarray(val_indices, dtype=np.int64)
    test_np = np.asarray(test_indices, dtype=np.int64)

    if mlp_probs is None:
        mlp_probs, _ = mod.train_mlp_and_predict(
            data, train_np,
            **mod.DEFAULT_MANUSCRIPT_MLP_KWARGS, log_file=None,
        )

    mlp_info = mod.compute_mlp_margin(mlp_probs)
    mlp_probs_np = mlp_info["mlp_probs_np"]
    mlp_pred_all = mlp_info["mlp_pred_all"]

    mlp_ent = _entropy(mlp_probs_np)

    evidence = mod._build_selective_correction_evidence(
        data, train_np, mlp_probs_np=mlp_probs_np,
        enable_feature_knn=enable_feature_knn, feature_knn_k=feature_knn_k,
    )

    if weight_candidates is None:
        weight_candidates = [
            {"b1": 1.0, "b2": 0.6, "b3": 0.0, "b4": 0.5, "b5": 0.3, "b6": 0.0},
            {"b1": 1.0, "b2": 0.9, "b3": 0.0, "b4": 0.3, "b5": 0.2, "b6": 0.0},
            {"b1": 1.0, "b2": 0.4, "b3": 0.0, "b4": 0.9, "b5": 0.5, "b6": 0.0},
            {"b1": 1.2, "b2": 0.5, "b3": 0.0, "b4": 0.4, "b5": 0.2, "b6": 0.0},
        ]

    val_ent = mlp_ent[val_np]
    percentiles = [50, 55, 60, 65, 70, 75, 80, 85, 90]
    ent_thresholds = sorted(set(np.percentile(val_ent, percentiles).tolist()))

    best_key = None
    best_cfg = None
    best_scores = None

    for w_cfg in weight_candidates:
        comps = mod.build_selective_correction_scores(
            mlp_probs_np, evidence,
            b1=w_cfg["b1"], b2=w_cfg["b2"], b3=w_cfg["b3"],
            b4=w_cfg["b4"], b5=w_cfg["b5"], b6=w_cfg.get("b6", 0.0),
        )
        combined = comps["combined_scores"]

        for e_thr in ent_thresholds:
            uncertain_val = val_ent > e_thr
            final_val = mlp_pred_all[val_np].copy()
            if uncertain_val.any():
                final_val[uncertain_val] = np.argmax(
                    combined[val_np][uncertain_val], axis=1
                ).astype(np.int64)
            val_acc = _simple_accuracy(y_true[val_np], final_val)
            changed = float((final_val != mlp_pred_all[val_np]).mean())
            unc_frac = float(uncertain_val.mean())
            key = (val_acc, -changed, -unc_frac)
            if best_key is None or key > best_key:
                best_key = key
                best_cfg = {
                    "weights": dict(w_cfg),
                    "entropy_threshold": e_thr,
                }
                best_scores = combined

    e_thr = best_cfg["entropy_threshold"]
    uncertain_all = mlp_ent > e_thr
    final_pred = mlp_pred_all.copy()
    if uncertain_all.any():
        final_pred[uncertain_all] = np.argmax(
            best_scores[uncertain_all], axis=1
        ).astype(np.int64)

    test_acc = _simple_accuracy(y_true[test_np], final_pred[test_np])
    val_acc = _simple_accuracy(y_true[val_np], final_pred[val_np])
    mlp_test_acc = _simple_accuracy(y_true[test_np], mlp_pred_all[test_np])

    info = {
        "variant": "V4_ENTROPY_GATE",
        "val_acc_selective": val_acc,
        "test_acc_mlp": mlp_test_acc,
        "test_acc_selective": test_acc,
        "selected_entropy_threshold": e_thr,
        "selected_weights": best_cfg["weights"],
        "fraction_test_uncertain": float(uncertain_all[test_np].mean()),
        "fraction_test_changed": float((final_pred[test_np] != mlp_pred_all[test_np]).mean()),
    }
    return val_acc, test_acc, info


# ---------------------------------------------------------------------------
# V5: Feature-kNN enhanced
# ---------------------------------------------------------------------------

def v5_knn_enhanced(
    data,
    train_indices,
    val_indices,
    test_indices,
    *,
    mlp_probs: Optional[torch.Tensor] = None,
    seed: int = 1337,
    mod=None,
    feature_knn_k: int = 10,
    threshold_candidates: Optional[List[float]] = None,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Enable feature-kNN with larger k and give it meaningful weight.
    Also add a feature-graph agreement filter.
    """
    y_true = data.y.detach().cpu().numpy().astype(np.int64)
    train_np = np.asarray(train_indices, dtype=np.int64)
    val_np = np.asarray(val_indices, dtype=np.int64)
    test_np = np.asarray(test_indices, dtype=np.int64)

    if mlp_probs is None:
        mlp_probs, _ = mod.train_mlp_and_predict(
            data, train_np,
            **mod.DEFAULT_MANUSCRIPT_MLP_KWARGS, log_file=None,
        )

    mlp_info = mod.compute_mlp_margin(mlp_probs)
    mlp_probs_np = mlp_info["mlp_probs_np"]
    mlp_pred_all = mlp_info["mlp_pred_all"]
    mlp_margin_all = mlp_info["mlp_margin_all"]

    evidence = mod._build_selective_correction_evidence(
        data, train_np, mlp_probs_np=mlp_probs_np,
        enable_feature_knn=True, feature_knn_k=feature_knn_k,
    )

    weight_candidates = [
        {"b1": 1.0, "b2": 0.5, "b3": 0.4, "b4": 0.3, "b5": 0.2, "b6": 0.0},
        {"b1": 1.0, "b2": 0.6, "b3": 0.3, "b4": 0.5, "b5": 0.3, "b6": 0.0},
        {"b1": 1.0, "b2": 0.4, "b3": 0.5, "b4": 0.2, "b5": 0.1, "b6": 0.0},
        {"b1": 1.0, "b2": 0.7, "b3": 0.2, "b4": 0.4, "b5": 0.2, "b6": 0.0},
        {"b1": 1.0, "b2": 0.3, "b3": 0.6, "b4": 0.1, "b5": 0.1, "b6": 0.0},
    ]

    if threshold_candidates is None:
        val_margins = mlp_margin_all[val_np]
        q = np.quantile(val_margins, [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        base = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
        threshold_candidates = sorted(set(np.round(np.concatenate([q, base]), 6).tolist()))

    best_key = None
    best_cfg = None
    best_scores = None

    for w_cfg in weight_candidates:
        comps = mod.build_selective_correction_scores(
            mlp_probs_np, evidence,
            b1=w_cfg["b1"], b2=w_cfg["b2"], b3=w_cfg["b3"],
            b4=w_cfg["b4"], b5=w_cfg["b5"], b6=w_cfg.get("b6", 0.0),
        )
        combined = comps["combined_scores"]

        for t in threshold_candidates:
            uncertain_val = mlp_margin_all[val_np] < t
            final_val = mlp_pred_all[val_np].copy()
            if uncertain_val.any():
                final_val[uncertain_val] = np.argmax(
                    combined[val_np][uncertain_val], axis=1
                ).astype(np.int64)
            val_acc = _simple_accuracy(y_true[val_np], final_val)
            changed = float((final_val != mlp_pred_all[val_np]).mean())
            unc_frac = float(uncertain_val.mean())
            key = (val_acc, -changed, -unc_frac)
            if best_key is None or key > best_key:
                best_key = key
                best_cfg = {"weights": dict(w_cfg), "threshold": t}
                best_scores = combined

    t = best_cfg["threshold"]
    uncertain_all = mlp_margin_all < t
    final_pred = mlp_pred_all.copy()
    if uncertain_all.any():
        final_pred[uncertain_all] = np.argmax(
            best_scores[uncertain_all], axis=1
        ).astype(np.int64)

    test_acc = _simple_accuracy(y_true[test_np], final_pred[test_np])
    val_acc = _simple_accuracy(y_true[val_np], final_pred[val_np])
    mlp_test_acc = _simple_accuracy(y_true[test_np], mlp_pred_all[test_np])

    info = {
        "variant": "V5_KNN_ENHANCED",
        "val_acc_selective": val_acc,
        "test_acc_mlp": mlp_test_acc,
        "test_acc_selective": test_acc,
        "selected_threshold": t,
        "selected_weights": best_cfg["weights"],
        "feature_knn_k": feature_knn_k,
        "fraction_test_uncertain": float(uncertain_all[test_np].mean()),
        "fraction_test_changed": float((final_pred[test_np] != mlp_pred_all[test_np]).mean()),
    }
    return val_acc, test_acc, info


# ---------------------------------------------------------------------------
# V6: Combined best ideas (reliability + kNN + entropy)
# ---------------------------------------------------------------------------

def v6_combo(
    data,
    train_indices,
    val_indices,
    test_indices,
    *,
    mlp_probs: Optional[torch.Tensor] = None,
    seed: int = 1337,
    mod=None,
    feature_knn_k: int = 10,
    gate_epochs: int = 200,
    gate_lr: float = 0.02,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Combine the best ideas:
    - Feature-kNN enabled
    - Reliability score
    - Entropy-based uncertainty
    - Learned soft gate for mixing strength
    """
    y_true = data.y.detach().cpu().numpy().astype(np.int64)
    train_np = np.asarray(train_indices, dtype=np.int64)
    val_np = np.asarray(val_indices, dtype=np.int64)
    test_np = np.asarray(test_indices, dtype=np.int64)

    if mlp_probs is None:
        mlp_probs, _ = mod.train_mlp_and_predict(
            data, train_np,
            **mod.DEFAULT_MANUSCRIPT_MLP_KWARGS, log_file=None,
        )

    mlp_info = mod.compute_mlp_margin(mlp_probs)
    mlp_probs_np = mlp_info["mlp_probs_np"]
    mlp_pred_all = mlp_info["mlp_pred_all"]
    mlp_margin_all = mlp_info["mlp_margin_all"]
    mlp_ent = _entropy(mlp_probs_np)

    evidence = mod._build_selective_correction_evidence(
        data, train_np, mlp_probs_np=mlp_probs_np,
        enable_feature_knn=True, feature_knn_k=feature_knn_k,
    )
    reliability = compute_node_reliability(mlp_probs_np, evidence)

    weight_candidates = [
        {"b1": 1.0, "b2": 0.5, "b3": 0.3, "b4": 0.4, "b5": 0.2, "b6": 0.0},
        {"b1": 1.0, "b2": 0.6, "b3": 0.2, "b4": 0.5, "b5": 0.3, "b6": 0.0},
        {"b1": 1.0, "b2": 0.4, "b3": 0.4, "b4": 0.3, "b5": 0.2, "b6": 0.0},
    ]

    best_val_acc = -1.0
    best_combined = None
    best_w = None
    for w_cfg in weight_candidates:
        comps = mod.build_selective_correction_scores(
            mlp_probs_np, evidence, **w_cfg,
        )
        combined = comps["combined_scores"]
        sgc_pred = np.argmax(combined, axis=1).astype(np.int64)
        val_acc_sgc = _simple_accuracy(y_true[val_np], sgc_pred[val_np])
        if val_acc_sgc > best_val_acc:
            best_val_acc = val_acc_sgc
            best_combined = combined
            best_w = w_cfg

    combined = best_combined
    sgc_pred_all = np.argmax(combined, axis=1).astype(np.int64)

    max_ent = np.log(max(mlp_probs_np.shape[1], 2))
    mlp_entropy_norm = mlp_ent / max_ent
    C = mlp_probs_np.shape[1]

    graph_ns = evidence["graph_neighbor_support"]
    graph_pred = np.argmax(graph_ns, axis=1)
    graph_max = graph_ns.max(axis=1)
    feat_sim = evidence["feature_similarity"]
    feat_pred = np.argmax(feat_sim, axis=1)
    knn_vote = evidence["feature_knn_vote"]
    knn_pred = np.argmax(knn_vote, axis=1)

    gate_features_all = np.column_stack([
        mlp_margin_all,
        mlp_probs_np.max(axis=1),
        mlp_entropy_norm,
        reliability,
        graph_max,
        (mlp_pred_all == graph_pred).astype(np.float64),
        (mlp_pred_all == feat_pred).astype(np.float64),
        (mlp_pred_all == knn_pred).astype(np.float64),
        (graph_pred == feat_pred).astype(np.float64),
        (graph_pred == knn_pred).astype(np.float64),
        evidence["neighbor_class_entropy"] / max_ent,
        np.clip(evidence["node_degree"] / 20.0, 0, 1),
    ]).astype(np.float32)

    val_mlp_correct = (mlp_pred_all[val_np] == y_true[val_np])
    val_sgc_correct = (sgc_pred_all[val_np] == y_true[val_np])
    val_labels = np.zeros(len(val_np), dtype=np.float32)
    val_labels[~val_mlp_correct & val_sgc_correct] = 1.0
    val_labels[val_mlp_correct & ~val_sgc_correct] = 0.0
    val_labels[val_mlp_correct & val_sgc_correct] = 0.3
    val_labels[~val_mlp_correct & ~val_sgc_correct] = 0.5

    X_val = gate_features_all[val_np]
    torch.manual_seed(seed)
    X_t = torch.from_numpy(X_val)
    y_t = torch.from_numpy(val_labels).unsqueeze(1)
    mean = X_t.mean(dim=0, keepdim=True)
    std = X_t.std(dim=0, keepdim=True).clamp_min(1e-6)
    Xn = (X_t - mean) / std

    gate_model = nn.Linear(Xn.size(1), 1)
    opt = optim.Adam(gate_model.parameters(), lr=gate_lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(gate_epochs):
        opt.zero_grad()
        loss = loss_fn(gate_model(Xn), y_t)
        loss.backward()
        opt.step()

    with torch.no_grad():
        X_all_t = torch.from_numpy(gate_features_all.astype(np.float32))
        Xn_all = (X_all_t - mean) / std
        gate_probs = torch.sigmoid(gate_model(Xn_all)).numpy().flatten()

    mlp_log = np.log(np.clip(mlp_probs_np, 1e-9, 1.0))
    combined_softmax = np.exp(combined)
    combined_softmax = combined_softmax / combined_softmax.sum(axis=1, keepdims=True)
    sgc_log = np.log(np.clip(combined_softmax, 1e-9, 1.0))

    alpha = gate_probs[:, None]
    blended = (1.0 - alpha) * mlp_log + alpha * sgc_log
    final_pred = np.argmax(blended, axis=1).astype(np.int64)

    test_acc = _simple_accuracy(y_true[test_np], final_pred[test_np])
    val_acc = _simple_accuracy(y_true[val_np], final_pred[val_np])
    mlp_test_acc = _simple_accuracy(y_true[test_np], mlp_pred_all[test_np])

    info = {
        "variant": "V6_COMBO",
        "val_acc_selective": val_acc,
        "test_acc_mlp": mlp_test_acc,
        "test_acc_selective": test_acc,
        "selected_weights": best_w,
        "feature_knn_k": feature_knn_k,
        "mean_gate_prob_test": float(gate_probs[test_np].mean()),
        "fraction_test_gate_above_0.5": float((gate_probs[test_np] > 0.5).mean()),
    }
    return val_acc, test_acc, info


# ---------------------------------------------------------------------------
# Registry for the experiment runner
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# V7: Adaptive regime-aware correction (refined combo from Phase 1 insights)
# ---------------------------------------------------------------------------

def v7_adaptive(
    data,
    train_indices,
    val_indices,
    test_indices,
    *,
    mlp_probs: Optional[torch.Tensor] = None,
    seed: int = 1337,
    mod=None,
    feature_knn_k: int = 10,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Adaptive variant that detects the dataset regime (homophilic vs heterophilic)
    from validation data and adapts the correction strategy:

    - High homophily: standard graph+feature correction (like baseline SGC)
      with reliability filtering to avoid rare bad corrections.
    - Low homophily: feature-heavy correction with kNN enabled,
      graph terms downweighted, stronger reliability filtering.

    Also uses a validation-calibrated approach: measures correction harm rate
    on validation to decide the aggressiveness of correction.
    """
    y_true = data.y.detach().cpu().numpy().astype(np.int64)
    train_np = np.asarray(train_indices, dtype=np.int64)
    val_np = np.asarray(val_indices, dtype=np.int64)
    test_np = np.asarray(test_indices, dtype=np.int64)

    if mlp_probs is None:
        mlp_probs, _ = mod.train_mlp_and_predict(
            data, train_np,
            **mod.DEFAULT_MANUSCRIPT_MLP_KWARGS, log_file=None,
        )

    mlp_info = mod.compute_mlp_margin(mlp_probs)
    mlp_probs_np = mlp_info["mlp_probs_np"]
    mlp_pred_all = mlp_info["mlp_pred_all"]
    mlp_margin_all = mlp_info["mlp_margin_all"]

    evidence = mod._build_selective_correction_evidence(
        data, train_np, mlp_probs_np=mlp_probs_np,
        enable_feature_knn=True, feature_knn_k=feature_knn_k,
    )
    reliability = compute_node_reliability(mlp_probs_np, evidence)

    homophily, _, _ = mod.edge_homophily_fast(data)
    is_homophilic = homophily >= 0.5

    if is_homophilic:
        weight_candidates = [
            {"b1": 1.0, "b2": 0.6, "b3": 0.0, "b4": 0.5, "b5": 0.3, "b6": 0.0},
            {"b1": 1.0, "b2": 0.9, "b3": 0.0, "b4": 0.3, "b5": 0.2, "b6": 0.0},
            {"b1": 1.0, "b2": 0.4, "b3": 0.0, "b4": 0.9, "b5": 0.5, "b6": 0.0},
            {"b1": 1.2, "b2": 0.5, "b3": 0.0, "b4": 0.4, "b5": 0.2, "b6": 0.0},
            {"b1": 1.0, "b2": 0.6, "b3": 0.2, "b4": 0.5, "b5": 0.3, "b6": 0.0},
        ]
        rel_thresholds = [0.3, 0.4, 0.5]
    else:
        weight_candidates = [
            {"b1": 1.0, "b2": 0.7, "b3": 0.4, "b4": 0.1, "b5": 0.1, "b6": 0.0},
            {"b1": 1.0, "b2": 0.5, "b3": 0.5, "b4": 0.2, "b5": 0.1, "b6": 0.0},
            {"b1": 1.0, "b2": 0.6, "b3": 0.3, "b4": 0.0, "b5": 0.0, "b6": 0.0},
            {"b1": 1.0, "b2": 0.4, "b3": 0.6, "b4": 0.1, "b5": 0.1, "b6": 0.0},
            {"b1": 1.0, "b2": 0.8, "b3": 0.2, "b4": 0.0, "b5": 0.0, "b6": 0.0},
        ]
        rel_thresholds = [0.5, 0.6, 0.7, 0.8]

    val_margins = mlp_margin_all[val_np]
    q = np.quantile(val_margins, [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    base = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    threshold_candidates = sorted(set(np.round(np.concatenate([q, base]), 6).tolist()))

    best_key = None
    best_cfg = None
    best_scores = None

    for w_cfg in weight_candidates:
        comps = mod.build_selective_correction_scores(
            mlp_probs_np, evidence, **w_cfg,
        )
        combined = comps["combined_scores"]

        for t in threshold_candidates:
            uncertain_val = mlp_margin_all[val_np] < t
            for r_thr in rel_thresholds:
                reliable_val = reliability[val_np] >= r_thr
                apply_val = uncertain_val & reliable_val
                final_val = mlp_pred_all[val_np].copy()
                if apply_val.any():
                    final_val[apply_val] = np.argmax(
                        combined[val_np][apply_val], axis=1
                    ).astype(np.int64)

                val_acc = _simple_accuracy(y_true[val_np], final_val)
                changed = float((final_val != mlp_pred_all[val_np]).mean())

                n_apply = apply_val.sum()
                if n_apply > 0:
                    corrections = final_val[apply_val]
                    mlp_on_apply = mlp_pred_all[val_np][apply_val]
                    true_on_apply = y_true[val_np][apply_val]
                    actually_changed = corrections != mlp_on_apply
                    if actually_changed.any():
                        harm_rate = float((
                            (mlp_on_apply[actually_changed] == true_on_apply[actually_changed]) &
                            (corrections[actually_changed] != true_on_apply[actually_changed])
                        ).mean())
                    else:
                        harm_rate = 0.0
                else:
                    harm_rate = 0.0

                penalty = harm_rate * 0.5
                adjusted_acc = val_acc - penalty

                key = (adjusted_acc, -changed)
                if best_key is None or key > best_key:
                    best_key = key
                    best_cfg = {
                        "weights": dict(w_cfg),
                        "threshold": t,
                        "reliability_threshold": r_thr,
                        "val_harm_rate": harm_rate,
                    }
                    best_scores = combined

    t = best_cfg["threshold"]
    r_thr = best_cfg["reliability_threshold"]
    uncertain_all = mlp_margin_all < t
    reliable_all = reliability >= r_thr
    apply_all = uncertain_all & reliable_all

    final_pred = mlp_pred_all.copy()
    if apply_all.any():
        final_pred[apply_all] = np.argmax(best_scores[apply_all], axis=1).astype(np.int64)

    test_acc = _simple_accuracy(y_true[test_np], final_pred[test_np])
    val_acc = _simple_accuracy(y_true[val_np], final_pred[val_np])
    mlp_test_acc = _simple_accuracy(y_true[test_np], mlp_pred_all[test_np])

    info = {
        "variant": "V7_ADAPTIVE",
        "val_acc_selective": val_acc,
        "test_acc_mlp": mlp_test_acc,
        "test_acc_selective": test_acc,
        "homophily": homophily,
        "is_homophilic": is_homophilic,
        "selected_threshold": t,
        "selected_reliability_threshold": r_thr,
        "selected_weights": best_cfg["weights"],
        "val_harm_rate": best_cfg["val_harm_rate"],
        "fraction_test_corrected": float(apply_all[test_np].mean()),
        "fraction_test_changed": float((final_pred[test_np] != mlp_pred_all[test_np]).mean()),
    }
    return val_acc, test_acc, info


VARIANT_REGISTRY = {
    "V1_RELIABILITY": v1_reliability_correction,
    "V2_MULTIBRANCH": v2_multibranch_correction,
    "V3_LEARNED_MIX": v3_learned_mixing,
    "V4_ENTROPY_GATE": v4_entropy_gate,
    "V5_KNN_ENHANCED": v5_knn_enhanced,
    "V6_COMBO": v6_combo,
    "V7_ADAPTIVE": v7_adaptive,
}

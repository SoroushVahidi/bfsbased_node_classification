#!/usr/bin/env python3
"""
FINAL METHOD v3: Reliability-Gated Selective Graph Correction.

A clean, principled simplification of V2_MULTIBRANCH, designed for
a research paper submission.

Method summary:
  1. Train feature-only MLP as base classifier
  2. Build graph/feature evidence for all nodes
  3. Compute per-node graph reliability score R(v) from 3 signals
  4. Select uncertainty threshold τ and reliability threshold ρ on validation
  5. For each test node:
       - If MLP is confident (margin ≥ τ): keep MLP prediction
       - If MLP is uncertain AND graph is reliable (R(v) ≥ ρ): correct with
         combined evidence scores
       - If MLP is uncertain AND graph is unreliable: keep MLP prediction

Key design vs V2:
  - **Two evidence weight profiles** (`WEIGHT_PROFILES`: balanced vs graph-heavy). The
    best profile is chosen **on the validation split**, jointly with τ and ρ (not fixed a priori).
  - **Three tuned choices on validation:** weight profile, uncertainty threshold τ, and
    reliability threshold ρ, searched over ~|τ|×|ρ|×2 profiles (see implementation).
  - 3-signal reliability formula R(v) with equal weights (no ad-hoc coefficients).
  - No feature-only branch (dropped — the reliability gate provides safety).
  - Cleaner search space than V2’s multibranch / multi-config grid.
"""
from __future__ import annotations

import csv
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _simple_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.size == 0:
        return 0.0
    return float((y_true == y_pred).mean())


# ---------------------------------------------------------------------------
# Core: Graph Reliability Score
# ---------------------------------------------------------------------------

def compute_graph_reliability(
    mlp_probs_np: np.ndarray,
    evidence: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    Per-node graph reliability score R(v) in [0, 1].

    Three interpretable signals, equally weighted:

    1. Neighbor concentration: max class probability among graph neighbors.
       High → neighbors agree on a class → graph signal is clear.

    2. MLP-graph agreement: binary indicator that the MLP prediction
       matches the plurality vote of graph neighbors.
       Agreement → graph evidence is consistent with feature evidence.

    3. Degree signal: saturating function of node degree.
       More neighbors → more evidence → more reliable aggregate.

    R(v) = (concentration + agreement + degree_signal) / 3
    """
    mlp_pred = np.argmax(mlp_probs_np, axis=1)
    graph_ns = evidence["graph_neighbor_support"]
    deg = evidence["node_degree"]

    graph_pred = np.argmax(graph_ns, axis=1)
    concentration = np.clip(graph_ns.max(axis=1), 0.0, 1.0)
    agreement = (mlp_pred == graph_pred).astype(np.float64)
    degree_signal = 1.0 - np.exp(-deg / 5.0)

    R = (concentration + agreement + degree_signal) / 3.0
    return np.clip(R, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Core: Final Method v3
# ---------------------------------------------------------------------------

# Two evidence weight profiles: one feature-leaning, one graph-leaning.
# The best profile is selected on validation alongside (τ, ρ).
# b1: MLP log-probability (anchor)
# b2: feature-prototype cosine similarity
# b4: 1-hop graph neighbor class support
# b5: train-train compatibility prior
WEIGHT_PROFILES = [
    {"b1": 1.0, "b2": 0.6, "b3": 0.0, "b4": 0.5, "b5": 0.3, "b6": 0.0},
    {"b1": 1.0, "b2": 0.4, "b3": 0.0, "b4": 0.9, "b5": 0.5, "b6": 0.0},
]


def compute_soft_local_agreement(
    data,
    train_indices: np.ndarray,
    mlp_probs_np: np.ndarray,
    mlp_pred_all: np.ndarray,
    y_true: np.ndarray,
    *,
    degree0_default: float = 0.0,
) -> np.ndarray:
    """
    Soft local agreement score h_v_soft in [0,1].

    For node v with predicted class c_v:
      h_v_soft(v) = mean_{u in N(v)} contribution(u -> v)

    contribution(u -> v):
      - if u is train-labeled: 1[y_u == c_v]
      - else: p_MLP(u, c_v)
    """
    num_nodes = int(data.num_nodes)
    edge_index = data.edge_index.detach().cpu().numpy()
    src = edge_index[0].astype(np.int64)
    dst = edge_index[1].astype(np.int64)

    train_mask = np.zeros(num_nodes, dtype=bool)
    train_mask[np.asarray(train_indices, dtype=np.int64)] = True

    dst_pred = mlp_pred_all[dst]
    contrib = mlp_probs_np[src, dst_pred].astype(np.float64)
    train_src = train_mask[src]
    if train_src.any():
        contrib[train_src] = (y_true[src[train_src]] == dst_pred[train_src]).astype(np.float64)

    accum = np.zeros(num_nodes, dtype=np.float64)
    deg = np.zeros(num_nodes, dtype=np.float64)
    np.add.at(accum, dst, contrib)
    np.add.at(deg, dst, 1.0)

    out = np.full(num_nodes, float(degree0_default), dtype=np.float64)
    nz = deg > 0
    out[nz] = accum[nz] / deg[nz]
    return np.clip(out, 0.0, 1.0)


def local_agreement_multiplier(
    local_agreement_scores: np.ndarray,
    *,
    mode: str,
    eta: Optional[float] = None,
) -> np.ndarray:
    """Return g(h_v_soft) used to attenuate graph-dependent correction terms."""
    h = np.clip(local_agreement_scores.astype(np.float64), 0.0, 1.0)
    mode = str(mode).lower()
    if mode == "none":
        return np.ones_like(h, dtype=np.float64)
    if mode == "linear":
        return h
    if mode == "hard_threshold":
        thr = 0.5 if eta is None else float(eta)
        return (h >= thr).astype(np.float64)
    if mode == "shifted_linear":
        thr = 0.5 if eta is None else float(eta)
        if thr >= 1.0:
            return np.zeros_like(h, dtype=np.float64)
        if thr <= 0.0:
            return h
        return np.clip((h - thr) / (1.0 - thr), 0.0, 1.0)
    raise ValueError(f"Unknown local agreement mode: {mode}")


def final_method_v3(
    data,
    train_indices,
    val_indices,
    test_indices,
    *,
    mlp_probs: Optional[torch.Tensor] = None,
    seed: int = 1337,
    mod=None,
    weights: Optional[Dict[str, float]] = None,
    include_node_arrays: bool = False,
    use_local_agreement_gate: bool = False,
    local_agreement_mode: str = "none",
    local_agreement_eta: Optional[float] = None,
    local_agreement_search_modes: Optional[List[str]] = None,
    local_agreement_eta_candidates: Optional[List[float]] = None,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Reliability-Gated Selective Graph Correction (v3).

    Parameters
    ----------
    data : PyG Data object
    train_indices, val_indices, test_indices : array-like
    mlp_probs : optional pre-computed MLP probabilities
    seed : random seed, retained for API compatibility with older runners.
        When `mlp_probs` is supplied, RNG control is handled by the caller.
    mod : the loaded bfsbased-full-investigate module
    weights : optional single weight dict. If set, **only** this profile is evaluated
        (ablation: single-profile / fixed-weight runs). Default: both `WEIGHT_PROFILES`
        are searched on validation together with τ and ρ.
    include_node_arrays : if True, include per-node arrays for the test split in
        info["node_arrays"]. Useful for downstream bucket analysis. Default: False.
    use_local_agreement_gate : if True, attenuate graph-dependent correction terms
        using soft local agreement g(h_v_soft). Default: False (canonical FINAL_V3).
    local_agreement_mode : one of {"none","linear","hard_threshold","shifted_linear"}.
        Used when `use_local_agreement_gate=True` and no search list is provided.
    local_agreement_eta : threshold parameter for thresholded modes.
    local_agreement_search_modes : optional list of modes to search on validation.
        If set, modes are searched jointly with (weights, τ, ρ).
    local_agreement_eta_candidates : eta grid for thresholded modes.

    Returns
    -------
    val_acc : float
    test_acc : float
    info : dict with full diagnostics
    """
    t_start = time.perf_counter()
    y_true = data.y.detach().cpu().numpy().astype(np.int64)
    train_np = np.asarray(train_indices, dtype=np.int64)
    val_np = np.asarray(val_indices, dtype=np.int64)
    test_np = np.asarray(test_indices, dtype=np.int64)

    # --- Step 1: MLP baseline ---
    t_mlp = time.perf_counter()
    if mlp_probs is None:
        mlp_probs, _ = mod.train_mlp_and_predict(
            data, train_np,
            **mod.DEFAULT_MANUSCRIPT_MLP_KWARGS, log_file=None,
        )
    mlp_time = time.perf_counter() - t_mlp

    mlp_info = mod.compute_mlp_margin(mlp_probs)
    mlp_probs_np = mlp_info["mlp_probs_np"]
    mlp_pred_all = mlp_info["mlp_pred_all"]
    mlp_margin_all = mlp_info["mlp_margin_all"]

    # --- Step 2: Build evidence ---
    t_ev = time.perf_counter()
    evidence = mod._build_selective_correction_evidence(
        data, train_np, mlp_probs_np=mlp_probs_np,
        enable_feature_knn=False,
    )
    evidence_time = time.perf_counter() - t_ev

    # --- Step 3: Reliability score ---
    reliability = compute_graph_reliability(mlp_probs_np, evidence)
    local_agreement_scores = compute_soft_local_agreement(
        data, train_np, mlp_probs_np, mlp_pred_all, y_true
    )

    # --- Step 4–5: Select (weights, τ, ρ) jointly on validation ---
    t_sel = time.perf_counter()
    val_margins = mlp_margin_all[val_np]

    tau_candidates = sorted(set(np.round(np.concatenate([
        np.quantile(val_margins, [0.25, 0.40, 0.50, 0.60, 0.75]),
        np.array([0.05, 0.10, 0.20, 0.30]),
    ]), 5).tolist()))

    rho_candidates = [0.3, 0.4, 0.5, 0.6]

    profiles = [weights] if weights else WEIGHT_PROFILES
    eta_candidates = local_agreement_eta_candidates or [0.3, 0.4, 0.5, 0.6, 0.7]
    if local_agreement_search_modes is not None:
        modes = [str(m).lower() for m in local_agreement_search_modes]
    elif use_local_agreement_gate:
        modes = [str(local_agreement_mode).lower()]
    else:
        modes = ["none"]

    local_configs: List[Tuple[str, Optional[float]]] = []
    for mode in modes:
        if mode in {"hard_threshold", "shifted_linear"}:
            if use_local_agreement_gate and local_agreement_search_modes is None and local_agreement_eta is not None:
                local_configs.append((mode, float(local_agreement_eta)))
            else:
                for eta in eta_candidates:
                    local_configs.append((mode, float(eta)))
        else:
            local_configs.append((mode, None))

    best_key = None
    best_tau = None
    best_rho = None
    best_combined = None
    best_w = None
    best_local_mode = None
    best_local_eta = None
    best_local_multiplier = None

    for w in profiles:
        comps = mod.build_selective_correction_scores(
            mlp_probs_np, evidence,
            b1=w["b1"], b2=w["b2"], b3=w["b3"],
            b4=w["b4"], b5=w["b5"], b6=w.get("b6", 0.0),
        )
        base_terms = (
            comps["mlp_term"]
            + comps["feature_similarity_term"]
            + comps["feature_knn_term"]
            + comps["structural_far_term"]
        )
        graph_terms = comps["graph_neighbor_term"] + comps["compatibility_term"]

        for local_mode, local_eta in local_configs:
            local_multiplier = local_agreement_multiplier(
                local_agreement_scores, mode=local_mode, eta=local_eta
            )
            combined_scores = base_terms + local_multiplier[:, None] * graph_terms

            for tau in tau_candidates:
                uncertain_val = val_margins < tau
                for rho in rho_candidates:
                    reliable_val = reliability[val_np] >= rho
                    apply_val = uncertain_val & reliable_val

                    final_val = mlp_pred_all[val_np].copy()
                    if apply_val.any():
                        final_val[apply_val] = np.argmax(
                            combined_scores[val_np][apply_val], axis=1
                        ).astype(np.int64)

                    val_acc = _simple_accuracy(y_true[val_np], final_val)
                    changed_frac = float((final_val != mlp_pred_all[val_np]).mean())
                    key = (val_acc, -changed_frac)

                    if best_key is None or key > best_key:
                        best_key = key
                        best_tau = tau
                        best_rho = rho
                        best_combined = combined_scores
                        best_w = dict(w)
                        best_local_mode = local_mode
                        best_local_eta = local_eta
                        best_local_multiplier = local_multiplier

    selection_time = time.perf_counter() - t_sel
    combined_scores = best_combined
    w = best_w
    local_multiplier_all = best_local_multiplier

    # --- Step 6: Apply correction ---
    uncertain_all = mlp_margin_all < best_tau
    reliable_all = reliability >= best_rho
    correct_mask = uncertain_all & reliable_all

    final_pred = mlp_pred_all.copy()
    if correct_mask.any():
        final_pred[correct_mask] = np.argmax(
            combined_scores[correct_mask], axis=1
        ).astype(np.int64)

    # --- Metrics ---
    test_acc = _simple_accuracy(y_true[test_np], final_pred[test_np])
    val_acc = _simple_accuracy(y_true[val_np], final_pred[val_np])
    mlp_test_acc = _simple_accuracy(y_true[test_np], mlp_pred_all[test_np])
    mlp_val_acc = _simple_accuracy(y_true[val_np], mlp_pred_all[val_np])

    # --- Branch analysis ---
    test_confident = (~uncertain_all)[test_np]
    test_uncertain_unreliable = (uncertain_all & ~reliable_all)[test_np]
    test_corrected = correct_mask[test_np]
    test_changed = (final_pred[test_np] != mlp_pred_all[test_np])
    test_local_multiplier = local_multiplier_all[test_np]

    # Correction precision: among actually changed nodes, what fraction improved?
    changed_test_idx = test_np[test_changed]
    if changed_test_idx.size > 0:
        was_correct_before = (mlp_pred_all[changed_test_idx] == y_true[changed_test_idx])
        is_correct_after = (final_pred[changed_test_idx] == y_true[changed_test_idx])
        n_helped = int((~was_correct_before & is_correct_after).sum())
        n_hurt = int((was_correct_before & ~is_correct_after).sum())
        n_neutral = int(changed_test_idx.size) - n_helped - n_hurt
        correction_precision = n_helped / max(n_helped + n_hurt, 1)
    else:
        n_helped = n_hurt = n_neutral = 0
        correction_precision = 1.0

    total_time = time.perf_counter() - t_start

    info = {
        "variant": "FINAL_V3",
        "val_acc_mlp": float(mlp_val_acc),
        "val_acc_v3": float(val_acc),
        "test_acc_mlp": float(mlp_test_acc),
        "test_acc_v3": float(test_acc),
        "delta_vs_mlp": float(test_acc - mlp_test_acc),
        "selected_tau": float(best_tau),
        "selected_rho": float(best_rho),
        "weights": {k: float(v) for k, v in w.items()},
        "local_agreement": {
            "enabled": bool(best_local_mode and best_local_mode != "none"),
            "selected_mode": str(best_local_mode),
            "selected_eta": None if best_local_eta is None else float(best_local_eta),
            "mean_score_all": float(local_agreement_scores.mean()),
            "mean_score_test": float(local_agreement_scores[test_np].mean()),
            "mean_multiplier_test": float(test_local_multiplier.mean()),
        },
        "branch_fractions": {
            "confident_keep_mlp": float(test_confident.mean()),
            "uncertain_unreliable_keep_mlp": float(test_uncertain_unreliable.mean()),
            "uncertain_reliable_corrected": float(test_corrected.mean()),
        },
        "correction_analysis": {
            "n_changed": int(test_changed.sum()),
            "n_helped": n_helped,
            "n_hurt": n_hurt,
            "n_neutral": n_neutral,
            "correction_precision": float(correction_precision),
        },
        "reliability_stats": {
            "mean_all": float(reliability.mean()),
            "mean_test": float(reliability[test_np].mean()),
            "mean_corrected": float(reliability[test_np][test_corrected].mean()) if test_corrected.any() else None,
            "mean_kept": float(reliability[test_np][~test_corrected].mean()) if (~test_corrected).any() else None,
        },
        "runtime_sec": {
            "mlp": float(mlp_time),
            "evidence": float(evidence_time),
            "selection": float(selection_time),
            "total": float(total_time),
        },
        "search_space_size": len(tau_candidates) * len(rho_candidates) * len(profiles),
    }
    if local_configs:
        info["search_space_size"] *= len(local_configs)
    if include_node_arrays:
        test_idx_local = test_np
        was_correct_before_test = (mlp_pred_all[test_idx_local] == y_true[test_idx_local])
        is_correct_after_test = (final_pred[test_idx_local] == y_true[test_idx_local])
        changed_test = (final_pred[test_idx_local] != mlp_pred_all[test_idx_local])
        changed_helpful = changed_test & (~was_correct_before_test) & is_correct_after_test
        changed_harmful = changed_test & was_correct_before_test & (~is_correct_after_test)
        info["node_arrays"] = {
            "test_indices": test_np.copy(),
            "mlp_predictions": mlp_pred_all[test_np].copy(),
            "final_predictions": final_pred[test_np].copy(),
            "mlp_margins": mlp_margin_all[test_np].copy(),
            "true_labels": y_true[test_np].copy(),
            "local_agreement_scores": local_agreement_scores[test_np].copy(),
            "local_graph_multiplier": local_multiplier_all[test_np].copy(),
            "is_uncertain": uncertain_all[test_np].copy(),
            "passes_reliability_gate": reliable_all[test_np].copy(),
            "is_corrected_branch": correct_mask[test_np].copy(),
            "is_changed": changed_test.copy(),
            "is_changed_helpful": changed_helpful.copy(),
            "is_changed_harmful": changed_harmful.copy(),
        }
    return val_acc, test_acc, info

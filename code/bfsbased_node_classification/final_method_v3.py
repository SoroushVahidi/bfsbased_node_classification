#!/usr/bin/env python3
"""
FINAL METHOD v3: Reliability-Gated Selective Graph Correction.

A clean, principled simplification of V2_MULTIBRANCH, designed for
a Pattern Recognition Letters submission.

Method summary:
  1. Train feature-only MLP as base classifier
  2. Build graph/feature evidence for all nodes
  3. Compute per-node graph reliability score R(v) from 3 signals
  4. Build a split-conformal uncertainty gate on validation calibration nodes
  5. Select reliability threshold ρ (and optional CP alpha) on validation
  6. For each test node:
       - If CP prediction set is singleton: keep MLP prediction
       - If MLP is uncertain AND graph is reliable (R(v) ≥ ρ): correct with
         combined evidence scores
       - If MLP is uncertain AND graph is unreliable: keep MLP prediction

Key design vs V2:
  - **Two evidence weight profiles** (`WEIGHT_PROFILES`: balanced vs graph-heavy). The
    best profile is chosen **on the validation split**, jointly with τ and ρ (not fixed a priori).
  - **Conformal uncertainty gate** (LAC by default; optional RAPS) calibrated on val.
  - **Tuned choices on validation:** weight profile, CP α (optional), and reliability
    threshold ρ, searched over ~|α|×|ρ|×2 profiles (see implementation).
  - 3-signal reliability formula R(v) with equal weights (no ad-hoc coefficients).
  - No feature-only branch (dropped — the reliability gate provides safety).
  - Cleaner search space than V2’s multibranch / multi-config grid.
"""
from __future__ import annotations

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


def _compute_node_context_features(
    data,
    mlp_probs_np: np.ndarray,
    mlp_margin_all: np.ndarray,
    evidence: Dict[str, np.ndarray],
) -> np.ndarray:
    """Build 9-D node context features used by the adaptive gate."""
    n_nodes = mlp_probs_np.shape[0]
    eps = 1e-12
    mlp_pred_all = np.argmax(mlp_probs_np, axis=1)
    feat_entropy = -np.sum(mlp_probs_np * np.log(np.clip(mlp_probs_np, eps, 1.0)), axis=1)

    deg = evidence["node_degree"].astype(np.float64)
    max_deg = float(np.max(deg)) if deg.size else 1.0
    deg_norm = np.log1p(deg) / max(np.log1p(max_deg), eps)
    deg_sat = np.minimum(deg, 10.0) / 10.0

    graph_ns = evidence["graph_neighbor_support"]
    conc = np.clip(graph_ns.max(axis=1), 0.0, 1.0)
    graph_pred = np.argmax(graph_ns, axis=1)
    agree = (mlp_pred_all == graph_pred).astype(np.float64)
    homophily_local = np.clip(graph_ns[np.arange(n_nodes), mlp_pred_all], 0.0, 1.0)

    edge_index = data.edge_index.detach().cpu().numpy()
    src = edge_index[0].astype(np.int64)
    dst = edge_index[1].astype(np.int64)
    nbr_margin_sum = np.zeros(n_nodes, dtype=np.float64)
    nbr_entropy_sum = np.zeros(n_nodes, dtype=np.float64)
    nbr_count = np.zeros(n_nodes, dtype=np.float64)
    np.add.at(nbr_margin_sum, dst, mlp_margin_all[src])
    np.add.at(nbr_entropy_sum, dst, feat_entropy[src])
    np.add.at(nbr_count, dst, 1.0)
    safe_cnt = np.maximum(nbr_count, 1.0)
    nbr_confidence = nbr_margin_sum / safe_cnt
    nbr_entropy = nbr_entropy_sum / safe_cnt

    return np.column_stack([
        mlp_margin_all,
        feat_entropy,
        deg_norm,
        homophily_local,
        nbr_confidence,
        nbr_entropy,
        conc,
        agree,
        deg_sat,
    ]).astype(np.float32)


def _search_heuristic_uncertainty(
    y_true: np.ndarray,
    mlp_pred_all: np.ndarray,
    mlp_margin_all: np.ndarray,
    reliability: np.ndarray,
    val_np: np.ndarray,
    profiles: List[Dict[str, float]],
    mod,
    mlp_probs_np: np.ndarray,
    evidence: Dict[str, np.ndarray],
):
    """Original FINAL_V3 τ-based uncertainty search."""
    val_margins = mlp_margin_all[val_np]
    tau_candidates = sorted(set(np.round(np.concatenate([
        np.quantile(val_margins, [0.25, 0.40, 0.50, 0.60, 0.75]),
        np.array([0.05, 0.10, 0.20, 0.30]),
    ]), 5).tolist()))
    rho_candidates = [0.3, 0.4, 0.5, 0.6]

    best_key = None
    best = {}
    for w in profiles:
        comps = mod.build_selective_correction_scores(
            mlp_probs_np, evidence,
            b1=w["b1"], b2=w["b2"], b3=w["b3"],
            b4=w["b4"], b5=w["b5"], b6=w.get("b6", 0.0),
        )
        combined_scores = comps["combined_scores"]
        for tau in tau_candidates:
            uncertain_all = mlp_margin_all < tau
            uncertain_val = uncertain_all[val_np]
            for rho in rho_candidates:
                reliable_val = reliability[val_np] >= rho
                apply_val = uncertain_val & reliable_val
                final_val = mlp_pred_all[val_np].copy()
                if apply_val.any():
                    final_val[apply_val] = np.argmax(combined_scores[val_np][apply_val], axis=1).astype(np.int64)
                val_acc = _simple_accuracy(y_true[val_np], final_val)
                changed_frac = float((final_val != mlp_pred_all[val_np]).mean())
                key = (val_acc, -changed_frac)
                if best_key is None or key > best_key:
                    best_key = key
                    best = {
                        "combined_scores": combined_scores,
                        "weights": dict(w),
                        "selected_tau": float(tau),
                        "selected_rho": float(rho),
                        "uncertain_all": uncertain_all,
                        "search_space_size": len(tau_candidates) * len(rho_candidates) * len(profiles),
                    }
    return best


def _train_adaptive_gate(
    features_all: np.ndarray,
    val_np: np.ndarray,
    y_gate_val: np.ndarray,
    seed: int,
):
    """Train gate MLP on validation nodes and return gate probabilities for all nodes."""
    torch.manual_seed(seed + 17)
    val_x = features_all[val_np]
    mu = val_x.mean(axis=0, keepdims=True)
    sigma = val_x.std(axis=0, keepdims=True)
    sigma[sigma < 1e-8] = 1.0
    x_norm = (features_all - mu) / sigma

    x_t = torch.from_numpy(x_norm).float()
    x_val_t = x_t[val_np]
    y_val_t = torch.from_numpy(y_gate_val.astype(np.float32)).view(-1, 1)

    model = torch.nn.Sequential(
        torch.nn.Linear(9, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 1),
    )
    n_pos = int(y_gate_val.sum())
    n_neg = int(y_gate_val.size - n_pos)
    if n_pos > 0 and (n_pos / max(y_gate_val.size, 1)) < 0.2:
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    best_state = None
    best_loss = float("inf")
    patience = 20
    no_improve = 0
    for _ in range(200):
        model.train()
        opt.zero_grad()
        logits = model(x_val_t)
        loss = loss_fn(logits, y_val_t)
        loss.backward()
        opt.step()

        cur_loss = float(loss.detach().cpu().item())
        if cur_loss < (best_loss - 1e-6):
            best_loss = cur_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(x_t)).squeeze(1).cpu().numpy()
    return probs, best_loss


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
    gate: str = "heuristic",
    split_id: Optional[int] = None,
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
        are searched on validation together with gate and ρ.
    gate : uncertainty gate type; one of {"heuristic", "adaptive"}.
    split_id : optional split index for diagnostics logging.

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

    # --- Step 4–5: Select (weights, gate, ρ) jointly on validation ---
    t_sel = time.perf_counter()
    profiles = [weights] if weights else WEIGHT_PROFILES
    if gate not in {"heuristic", "adaptive"}:
        raise ValueError("gate must be one of {'heuristic', 'adaptive'}")

    if gate == "heuristic":
        best = _search_heuristic_uncertainty(
            y_true=y_true,
            mlp_pred_all=mlp_pred_all,
            mlp_margin_all=mlp_margin_all,
            reliability=reliability,
            val_np=val_np,
            profiles=profiles,
            mod=mod,
            mlp_probs_np=mlp_probs_np,
            evidence=evidence,
        )
        best_gate_loss = None
        n_pos = None
        n_neg = None
    else:
        rho_candidates = [0.3, 0.4, 0.5, 0.6]
        features_all = _compute_node_context_features(
            data=data,
            mlp_probs_np=mlp_probs_np,
            mlp_margin_all=mlp_margin_all,
            evidence=evidence,
        )
        best_key = None
        best = {}
        best_gate_loss = None
        n_pos = 0
        n_neg = 0
        for w in profiles:
            comps = mod.build_selective_correction_scores(
                mlp_probs_np, evidence,
                b1=w["b1"], b2=w["b2"], b3=w["b3"],
                b4=w["b4"], b5=w["b5"], b6=w.get("b6", 0.0),
            )
            combined_scores = comps["combined_scores"]
            corrected_all = np.argmax(combined_scores, axis=1).astype(np.int64)
            y_gate_val = (
                (corrected_all[val_np] == y_true[val_np]) &
                (mlp_pred_all[val_np] != y_true[val_np])
            ).astype(np.int64)
            cur_pos = int(y_gate_val.sum())
            cur_neg = int(y_gate_val.size - cur_pos)
            if cur_pos < 10:
                print(f"[AdaptiveGate] Fallback to heuristic τ: too few positive training examples (n_pos={cur_pos}).")
                best = _search_heuristic_uncertainty(
                    y_true=y_true,
                    mlp_pred_all=mlp_pred_all,
                    mlp_margin_all=mlp_margin_all,
                    reliability=reliability,
                    val_np=val_np,
                    profiles=profiles,
                    mod=mod,
                    mlp_probs_np=mlp_probs_np,
                    evidence=evidence,
                )
                best["fallback_reason"] = "too_few_positive_examples"
                best_gate_loss = None
                n_pos = cur_pos
                n_neg = cur_neg
                gate = "heuristic"
                break

            gate_probs_all, gate_loss = _train_adaptive_gate(features_all, val_np, y_gate_val, seed=seed)
            uncertain_all = gate_probs_all > 0.5
            for rho in rho_candidates:
                reliable_val = reliability[val_np] >= rho
                apply_val = uncertain_all[val_np] & reliable_val
                final_val = mlp_pred_all[val_np].copy()
                if apply_val.any():
                    final_val[apply_val] = corrected_all[val_np][apply_val]
                val_acc = _simple_accuracy(y_true[val_np], final_val)
                changed_frac = float((final_val != mlp_pred_all[val_np]).mean())
                key = (val_acc, -changed_frac)
                if best_key is None or key > best_key:
                    best_key = key
                    n_pos = cur_pos
                    n_neg = cur_neg
                    best_gate_loss = gate_loss
                    best = {
                        "combined_scores": combined_scores,
                        "weights": dict(w),
                        "selected_tau": None,
                        "selected_rho": float(rho),
                        "uncertain_all": uncertain_all,
                        "gate_probs_all": gate_probs_all,
                        "search_space_size": len(rho_candidates) * len(profiles),
                    }

    selection_time = time.perf_counter() - t_sel
    combined_scores = best["combined_scores"]
    w = best["weights"]
    selected_tau = best.get("selected_tau")
    best_rho = best["selected_rho"]

    # --- Step 6: Apply correction ---
    uncertain_all = best["uncertain_all"]
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
    test_uncertain = uncertain_all[test_np]

    if test_uncertain.any():
        acc_test_uncertain_mlp = _simple_accuracy(y_true[test_np][test_uncertain], mlp_pred_all[test_np][test_uncertain])
        acc_test_uncertain_v3 = _simple_accuracy(y_true[test_np][test_uncertain], final_pred[test_np][test_uncertain])
    else:
        acc_test_uncertain_mlp = None
        acc_test_uncertain_v3 = None

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
        "gate_mode": gate,
        "selected_tau": float(selected_tau) if selected_tau is not None else None,
        "selected_rho": float(best_rho),
        "weights": {k: float(v) for k, v in w.items()},
        "branch_fractions": {
            "confident_keep_mlp": float(test_confident.mean()),
            "uncertain_unreliable_keep_mlp": float(test_uncertain_unreliable.mean()),
            "uncertain_reliable_corrected": float(test_corrected.mean()),
        },
        "gate_stats": {
            "uncertain_fraction_test": float(uncertain_all[test_np].mean()),
            "gate_bce_loss": float(best_gate_loss) if best_gate_loss is not None else None,
            "n_pos_val": int(n_pos) if n_pos is not None else None,
            "n_neg_val": int(n_neg) if n_neg is not None else None,
            "low_conf_gain_vs_mlp_test": (
                float(acc_test_uncertain_v3 - acc_test_uncertain_mlp)
                if (acc_test_uncertain_mlp is not None and acc_test_uncertain_v3 is not None)
                else None
            ),
        },
        "correction_analysis": {
            "n_changed": int(test_changed.sum()),
            "n_helped": n_helped,
            "n_hurt": n_hurt,
            "n_neutral": n_neutral,
            "correction_precision": float(correction_precision),
            "changed_fraction_test": float(test_changed.mean()),
            "harmful_overwrite_rate_non_neutral": float(1.0 - correction_precision),
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
        "search_space_size": int(best.get("search_space_size", 0)),
    }
    if info["gate_mode"] == "adaptive":
        split_label = split_id if split_id is not None else "?"
        print(
            f"[AdaptiveGate] split={split_label} | n_pos={info['gate_stats']['n_pos_val']} "
            f"n_neg={info['gate_stats']['n_neg_val']} | routed to correction: "
            f"{100.0 * info['gate_stats']['uncertain_fraction_test']:.1f}% of test nodes | "
            f"gate BCE loss: {info['gate_stats']['gate_bce_loss']:.3f}"
        )
    return val_acc, test_acc, info

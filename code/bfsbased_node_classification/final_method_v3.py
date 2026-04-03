#!/usr/bin/env python3
"""
FINAL METHOD v3: Reliability-Gated Selective Graph Correction.

A clean, principled simplification of V2_MULTIBRANCH, designed for
a research paper submission.

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
  - Cleaner search space than V2's multibranch / multi-config grid.
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
# Core: Final Method v3 — local agreement helpers
# ---------------------------------------------------------------------------

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


def compute_local_homophily_from_labels(
    data,
    train_indices: np.ndarray,
    working_labels: np.ndarray,
    y_true: np.ndarray,
    *,
    degree0_default: float = 0.0,
) -> np.ndarray:
    """
    Exact local homophily score h_v in [0,1] from a current label assignment.

    For node v:
      h_v = |{u in N(v): yhat_u == yhat_v}| / |N(v)|

    Label source:
      - train-labeled nodes are forced to true labels y_true
      - all other nodes use `working_labels` (current propagated/working labels)
    """
    num_nodes = int(data.num_nodes)
    edge_index = data.edge_index.detach().cpu().numpy()
    src = edge_index[0].astype(np.int64)
    dst = edge_index[1].astype(np.int64)

    train_mask = np.zeros(num_nodes, dtype=bool)
    train_mask[np.asarray(train_indices, dtype=np.int64)] = True

    labels = np.asarray(working_labels, dtype=np.int64).copy()
    labels[train_mask] = y_true[train_mask]

    matches = (labels[src] == labels[dst]).astype(np.float64)
    accum = np.zeros(num_nodes, dtype=np.float64)
    deg = np.zeros(num_nodes, dtype=np.float64)
    np.add.at(accum, dst, matches)
    np.add.at(deg, dst, 1.0)

    out = np.full(num_nodes, float(degree0_default), dtype=np.float64)
    nz = deg > 0
    out[nz] = accum[nz] / deg[nz]
    return np.clip(out, 0.0, 1.0)


def _is_exact_local_mode(mode: str) -> bool:
    return str(mode).lower().startswith("exact_local_")


def _mode_to_score_family(mode: str) -> str:
    return "exact" if _is_exact_local_mode(mode) else "soft"


def _canonical_local_mode(mode: str) -> str:
    mode_l = str(mode).lower()
    if mode_l.startswith("exact_local_"):
        return mode_l[len("exact_local_"):]
    return mode_l


def local_agreement_multiplier(
    local_agreement_scores: np.ndarray,
    *,
    mode: str,
    eta: Optional[float] = None,
) -> np.ndarray:
    """Return multiplier g(h_v) used to attenuate graph-dependent correction terms."""
    h = np.clip(local_agreement_scores.astype(np.float64), 0.0, 1.0)
    mode = _canonical_local_mode(mode)
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
    include_node_arrays: bool = False,
    enable_lowconf_structural_term: bool = False,
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
        are searched on validation together with gate and ρ.
    gate : uncertainty gate type; one of {"heuristic", "adaptive"}.
    split_id : optional split index for diagnostics logging.
    include_node_arrays : if True, include per-node arrays for the test split in
        info["node_arrays"] and info["test_node_outputs"]. Default: False.
    enable_lowconf_structural_term : if True, search over b6 candidates
        (low-confidence structural term weight). Default: False (canonical FINAL_V3).
    use_local_agreement_gate : if True, attenuate graph-dependent correction terms
        using a local reliability score g(h_v). Default: False (canonical FINAL_V3).
        The canonical soft local agreement path remains the default family unless an
        exact mode is explicitly selected.
    local_agreement_mode : supports:
        {"none","linear","hard_threshold","shifted_linear",
         "exact_local_linear","exact_local_hard_threshold","exact_local_shifted_linear"}.
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
    if local_agreement_search_modes is not None:
        modes = [str(m).lower() for m in local_agreement_search_modes]
    elif use_local_agreement_gate:
        modes = [str(local_agreement_mode).lower()]
    else:
        modes = ["none"]

    need_soft_local_agreement = bool(
        include_node_arrays or any(_mode_to_score_family(m) == "soft" for m in modes)
    )
    if need_soft_local_agreement:
        local_agreement_scores = compute_soft_local_agreement(
            data, train_np, mlp_probs_np, mlp_pred_all, y_true
        )
    else:
        local_agreement_scores = np.ones_like(reliability, dtype=np.float32)

    # --- Step 4–5: Select (weights, gate, ρ) jointly on validation ---
    t_sel = time.perf_counter()
    val_margins = mlp_margin_all[val_np]
    tau_candidates = sorted(set(np.round(np.concatenate([
        np.quantile(val_margins, [0.25, 0.40, 0.50, 0.60, 0.75]),
        np.array([0.05, 0.10, 0.20, 0.30]),
    ]), 5).tolist()))
    rho_candidates = [0.3, 0.4, 0.5, 0.6]
    profiles = [weights] if weights else WEIGHT_PROFILES

    # b6 candidates (low-confidence structural term; canonical FINAL_V3 uses b6=0)
    b6_candidates = [0.0, 0.2, 0.4] if enable_lowconf_structural_term else [0.0]

    # local agreement configurations
    eta_candidates = local_agreement_eta_candidates or [0.3, 0.4, 0.5, 0.6, 0.7]
    local_configs: List[Tuple[str, Optional[float]]] = []
    for mode in modes:
        canonical_mode = _canonical_local_mode(mode)
        if canonical_mode in {"hard_threshold", "shifted_linear"}:
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
    best_b6 = 0.0
    best_local_mode = None
    best_local_eta = None
    best_local_multiplier = None
    best_local_scores = None
    best_local_score_family = None

    for w in profiles:
        for b6_val in b6_candidates:
            w_eff = dict(w)
            w_eff["b6"] = float(b6_val)
            comps = mod.build_selective_correction_scores(
                mlp_probs_np, evidence,
                b1=w_eff["b1"], b2=w_eff["b2"], b3=w_eff["b3"],
                b4=w_eff["b4"], b5=w_eff["b5"], b6=w_eff["b6"],
            )
            base_terms = (
                comps["mlp_term"]
                + comps["feature_similarity_term"]
                + comps["feature_knn_term"]
                + comps["structural_far_term"]
            )
            graph_terms = comps["graph_neighbor_term"] + comps["compatibility_term"]
            working_labels = np.argmax(base_terms + graph_terms, axis=1).astype(np.int64)
            working_labels[train_np] = y_true[train_np]
            local_homophily_exact = compute_local_homophily_from_labels(
                data, train_np, working_labels, y_true
            )

            for local_mode, local_eta in local_configs:
                score_family = _mode_to_score_family(local_mode)
                local_scores = local_homophily_exact if score_family == "exact" else local_agreement_scores
                local_multiplier = local_agreement_multiplier(
                    local_scores, mode=local_mode, eta=local_eta
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
                            best_w = w_eff
                            best_b6 = float(b6_val)
                            best_local_mode = local_mode
                            best_local_eta = local_eta
                            best_local_multiplier = local_multiplier
                            best_local_scores = local_scores
                            best_local_score_family = score_family

    selection_time = time.perf_counter() - t_sel
    combined_scores = best_combined
    w = best_w
    local_multiplier_all = best_local_multiplier
    selected_local_scores_all = best_local_scores if best_local_scores is not None else local_agreement_scores
    selected_local_score_family = best_local_score_family or "soft"

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
    test_local_scores = selected_local_scores_all[test_np]
    test_uncertain = uncertain_all[test_np]

    if test_uncertain.any():
        acc_test_uncertain_mlp = _simple_accuracy(
            y_true[test_np][test_uncertain], mlp_pred_all[test_np][test_uncertain]
        )
        acc_test_uncertain_v3 = _simple_accuracy(
            y_true[test_np][test_uncertain], final_pred[test_np][test_uncertain]
        )
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

    search_space_size = (
        len(tau_candidates) * len(rho_candidates) * len(profiles)
        * len(b6_candidates) * len(local_configs)
    )

    info = {
        "variant": "FINAL_V3",
        "split_id": int(split_id) if split_id is not None else None,
        "gate": gate,
        "val_acc_mlp": float(mlp_val_acc),
        "val_acc_v3": float(val_acc),
        "test_acc_mlp": float(mlp_test_acc),
        "test_acc_v3": float(test_acc),
        "delta_vs_mlp": float(test_acc - mlp_test_acc),
        "gate_mode": gate,
        "selected_tau": float(best_tau),
        "selected_rho": float(best_rho),
        "weights": {k: float(v) for k, v in w.items()},
        "enable_lowconf_structural_term": bool(enable_lowconf_structural_term),
        "selected_b6": float(best_b6),
        "local_agreement": {
            "enabled": bool(best_local_mode and best_local_mode != "none"),
            "selected_mode": str(best_local_mode),
            "selected_score_family": str(selected_local_score_family),
            "selected_eta": None if best_local_eta is None else float(best_local_eta),
            "mean_score_all": float(selected_local_scores_all.mean()),
            "mean_score_test": float(test_local_scores.mean()),
            "std_score_test": float(test_local_scores.std()),
            "mean_score_corrected_test": float(test_local_scores[test_corrected].mean()) if test_corrected.any() else None,
            "mean_score_kept_test": float(test_local_scores[~test_corrected].mean()) if (~test_corrected).any() else None,
            "mean_multiplier_test": float(test_local_multiplier.mean()),
        },
        "branch_fractions": {
            "confident_keep_mlp": float(test_confident.mean()),
            "uncertain_unreliable_keep_mlp": float(test_uncertain_unreliable.mean()),
            "uncertain_reliable_corrected": float(test_corrected.mean()),
        },
        "gate_stats": {
            "uncertain_fraction_test": float(uncertain_all[test_np].mean()),
            "gate_bce_loss": None,
            "n_pos_val": None,
            "n_neg_val": None,
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
        "search_space_size": search_space_size,
    }
    if include_node_arrays:
        test_idx_local = test_np
        was_correct_before_test = (mlp_pred_all[test_idx_local] == y_true[test_idx_local])
        is_correct_after_test = (final_pred[test_idx_local] == y_true[test_idx_local])
        changed_test = (final_pred[test_idx_local] != mlp_pred_all[test_idx_local])
        changed_helpful = changed_test & (~was_correct_before_test) & is_correct_after_test
        changed_harmful = changed_test & was_correct_before_test & (~is_correct_after_test)
        # node_arrays: numpy arrays for downstream use
        info["node_arrays"] = {
            "test_indices": test_np.copy(),
            "mlp_predictions": mlp_pred_all[test_np].copy(),
            "final_predictions": final_pred[test_np].copy(),
            "mlp_margins": mlp_margin_all[test_np].copy(),
            "true_labels": y_true[test_np].copy(),
            "local_agreement_scores": local_agreement_scores[test_np].copy(),
            "selected_local_scores": selected_local_scores_all[test_np].copy(),
            "local_graph_multiplier": local_multiplier_all[test_np].copy(),
            "is_uncertain": uncertain_all[test_np].copy(),
            "passes_reliability_gate": reliable_all[test_np].copy(),
            "is_corrected_branch": correct_mask[test_np].copy(),
            "is_changed": changed_test.copy(),
            "is_changed_helpful": changed_helpful.copy(),
            "is_changed_harmful": changed_harmful.copy(),
        }
        # test_node_outputs: JSON-serializable version for CSV/JSONL export
        info["test_node_outputs"] = {
            "node_id": test_np.astype(np.int64).tolist(),
            "true_label": y_true[test_np].astype(np.int64).tolist(),
            "mlp_pred": mlp_pred_all[test_np].astype(np.int64).tolist(),
            "final_pred": final_pred[test_np].astype(np.int64).tolist(),
            "changed_from_mlp": (final_pred[test_np] != mlp_pred_all[test_np]).astype(np.int64).tolist(),
            "beneficial_change": changed_helpful.astype(np.int64).tolist(),
            "harmful_change": changed_harmful.astype(np.int64).tolist(),
            "mlp_margin": mlp_margin_all[test_np].astype(np.float64).tolist(),
            "passed_uncertainty_gate": uncertain_all[test_np].astype(np.int64).tolist(),
            "passed_reliability_gate": reliable_all[test_np].astype(np.int64).tolist(),
            "reliability": reliability[test_np].astype(np.float64).tolist(),
            "local_homophily_score": selected_local_scores_all[test_np].astype(np.float64).tolist(),
            "combined_top_score": combined_scores[test_np].max(axis=1).astype(np.float64).tolist(),
            "combined_top_label": np.argmax(combined_scores[test_np], axis=1).astype(np.int64).tolist(),
            "selected_tau": float(best_tau),
            "selected_rho": float(best_rho),
            "selected_weights": {k: float(v) for k, v in w.items()},
        }
    return val_acc, test_acc, info

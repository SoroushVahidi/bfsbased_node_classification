#!/usr/bin/env python3
"""
MS_HSGC: Multi-Scale Heterophily-aware Selective Graph Correction.

Experimental method that extends FINAL_V3 with:
  - 2-hop graph support (soft, sparse-matrix based)
  - Per-node heterophily measures H1(v), H2(v), DeltaH(v)
  - Multi-scale routing: confident → MLP, uncertain+low_H1+reliable_1hop → 1-hop
    correction, uncertain+high_H1+reliable_2hop+pos_DeltaH → 2-hop correction
  - Validation-based grid search over ~96 configurations

Status: EXPERIMENTAL. Does not affect canonical FINAL_V3 outputs.
"""
from __future__ import annotations

import importlib.util
import os
import time
import types
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch


# ---------------------------------------------------------------------------
# Weight profiles
# ---------------------------------------------------------------------------

WEIGHT_PROFILES_1HOP: List[Dict[str, float]] = [
    {"a1": 1.0, "a2": 0.6, "a3": 0.5, "a4": 0.3},  # balanced
    {"a1": 1.0, "a2": 0.4, "a3": 0.9, "a4": 0.5},  # graph_heavy
]

WEIGHT_PROFILES_2HOP: List[Dict[str, float]] = [
    {"b1": 1.0, "b2": 0.5, "b3": 0.5, "b4": 0.2},  # balanced_2hop
    {"b1": 0.8, "b2": 0.4, "b3": 1.0, "b4": 0.2},  # 2hop_trusting
]


# ---------------------------------------------------------------------------
# Module loader (mirrors run_final_evaluation._load_module)
# ---------------------------------------------------------------------------

def _load_module():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "bfsbased-full-investigate-homophil.py")
    with open(path, "r", encoding="utf-8") as fh:
        source = fh.read()

    start = "dataset = load_dataset(DATASET_KEY, root)"
    end = "# In[21]:"
    legacy = 'LOG_DIR = "logs"'
    if start in source and end in source:
        a = source.index(start)
        b = source.index(end, a)
        source = source[:a] + "\n" + source[b:]
    if legacy in source:
        source = source[: source.index(legacy)]

    mod = importlib.util.module_from_spec(
        importlib.util.spec_from_loader("bfs_full_investigate", loader=None)
    )
    setattr(mod, "__file__", path)
    setattr(mod, "DATASET_KEY", "texas")
    exec(compile(source, path, "exec"), mod.__dict__)
    return mod


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _to_np_to_tensor(np_arr: np.ndarray, device) -> torch.Tensor:
    return torch.as_tensor(np_arr, dtype=torch.long, device=device)


def _simple_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.size == 0:
        return 0.0
    return float((y_true == y_pred).mean())


# ---------------------------------------------------------------------------
# Sparse 1-hop and 2-hop support computation
# ---------------------------------------------------------------------------

def _compute_multiscale_support(
    data,
    train_indices: np.ndarray,
    mlp_probs_np: np.ndarray,
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute 1-hop and 2-hop class support for all nodes.

    Uses train labels where available, MLP probabilities otherwise.
    Both returned arrays have shape (N, num_classes), rows summing to 1.
    """
    N = data.num_nodes
    edge_index = data.edge_index.detach().cpu().numpy()
    row_idx, col_idx = edge_index[0].astype(np.int64), edge_index[1].astype(np.int64)

    vals = np.ones(len(row_idx), dtype=np.float32)
    A = sp.csr_matrix((vals, (row_idx, col_idx)), shape=(N, N))

    # Build node label matrix: one-hot for train nodes, MLP probs otherwise
    node_labels = mlp_probs_np.copy().astype(np.float32)
    for idx in train_indices:
        idx_int = int(idx)
        node_labels[idx_int] = 0.0
        node_labels[idx_int, int(data.y[idx_int])] = 1.0

    # 1-hop support
    support_1hop_raw = A.dot(node_labels)  # (N, C)
    row_sums_1 = np.array(support_1hop_raw.sum(axis=1)).flatten()
    row_sums_1 = np.where(row_sums_1 < 1e-9, 1.0, row_sums_1)
    support_1hop = np.asarray(support_1hop_raw / row_sums_1[:, None], dtype=np.float32)

    # 2-hop support: A @ (A @ node_labels)  — avoids materialising A^2
    support_2hop_raw = A.dot(A.dot(node_labels))  # (N, C)
    row_sums_2 = np.array(support_2hop_raw.sum(axis=1)).flatten()
    row_sums_2 = np.where(row_sums_2 < 1e-9, 1.0, row_sums_2)
    support_2hop = np.asarray(support_2hop_raw / row_sums_2[:, None], dtype=np.float32)

    return support_1hop, support_2hop


# ---------------------------------------------------------------------------
# Heterophily measures
# ---------------------------------------------------------------------------

def _compute_heterophily(
    data,
    mlp_pred_all: np.ndarray,
    support_1hop: np.ndarray,
    support_2hop: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-node heterophily measures H1, H2, DeltaH.

    H1(v) = 1 - support_1hop[v, mlp_pred[v]]
    H2(v) = 1 - support_2hop[v, mlp_pred[v]]
    DeltaH(v) = H1(v) - H2(v)  (positive → 2-hop more homophilic than 1-hop)
    """
    N = data.num_nodes
    node_range = np.arange(N)
    pred_support_1hop = support_1hop[node_range, mlp_pred_all]
    pred_support_2hop = support_2hop[node_range, mlp_pred_all]

    H1 = 1.0 - pred_support_1hop
    H2 = 1.0 - pred_support_2hop
    DeltaH = H1 - H2

    H1 = np.clip(H1, 0.0, 1.0).astype(np.float32)
    H2 = np.clip(H2, 0.0, 1.0).astype(np.float32)
    DeltaH = DeltaH.astype(np.float32)
    return H1, H2, DeltaH


# ---------------------------------------------------------------------------
# Reliability scores
# ---------------------------------------------------------------------------

def _compute_reliability_1hop(
    data,
    mlp_pred_all: np.ndarray,
    support_1hop: np.ndarray,
) -> np.ndarray:
    """1-hop reliability R1(v) ∈ [0, 1]."""
    N = data.num_nodes
    edge_index = data.edge_index.detach().cpu().numpy()
    row_idx = edge_index[0].astype(np.int64)

    deg = np.bincount(row_idx, minlength=N).astype(np.float32)
    concentration = support_1hop.max(axis=1)
    nbr_pred = support_1hop.argmax(axis=1)
    agreement = (nbr_pred == mlp_pred_all).astype(np.float32)
    degree_signal = 1.0 - np.exp(-deg / 5.0)

    R1 = (concentration + agreement + degree_signal) / 3.0
    return R1.astype(np.float32)


def _compute_reliability_2hop(
    mlp_pred_all: np.ndarray,
    support_2hop: np.ndarray,
    deg: np.ndarray,
) -> np.ndarray:
    """2-hop reliability R2(v) ∈ [0, 1]."""
    concentration = support_2hop.max(axis=1)
    nbr_pred_2hop = support_2hop.argmax(axis=1)
    agreement = (nbr_pred_2hop == mlp_pred_all).astype(np.float32)
    availability = 1.0 - np.exp(-deg / 10.0)

    R2 = (concentration + agreement + availability) / 3.0
    return R2.astype(np.float32)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _route_nodes(
    uncertain_mask: np.ndarray,
    R1: np.ndarray,
    R2: np.ndarray,
    H1: np.ndarray,
    DeltaH: np.ndarray,
    rho1: float,
    rho2: float,
    h1_max: float,
    delta_min: float,
) -> np.ndarray:
    """Assign each node to a routing bucket (vectorised).

    Buckets:
      "confident"     – MLP margin above tau
      "correct_1hop"  – uncertain, R1 ≥ rho1, H1 ≤ h1_max
      "correct_2hop"  – uncertain, 1-hop criteria not met, R2 ≥ rho2, DeltaH ≥ delta_min
      "mlp_only"      – uncertain, no reliable correction available

    Returns an integer array:
      0 = confident, 1 = correct_1hop, 2 = correct_2hop, 3 = mlp_only
    """
    N = len(uncertain_mask)
    # Nodes where uncertain_mask=False stay at 0 (confident); uncertain nodes are routed below.
    route = np.zeros(N, dtype=np.int8)

    unc = uncertain_mask

    can_1hop = unc & (R1 >= rho1) & (H1 <= h1_max)
    can_2hop = unc & ~can_1hop & (R2 >= rho2) & (DeltaH >= delta_min)
    fallback = unc & ~can_1hop & ~can_2hop

    route[can_1hop] = 1
    route[can_2hop] = 2
    route[fallback] = 3
    return route


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

_EPS = 1e-12


def _compute_scores_1hop(
    mlp_probs_np: np.ndarray,
    feature_sim: np.ndarray,
    support_1hop: np.ndarray,
    compat: np.ndarray,
    profile: Dict[str, float],
) -> np.ndarray:
    """S1(v, c) = a1*log(p)+a2*feat_sim+a3*1hop_support+a4*compat."""
    log_p = np.log(np.clip(mlp_probs_np, _EPS, 1.0))
    S = (profile["a1"] * log_p
         + profile["a2"] * feature_sim
         + profile["a3"] * support_1hop
         + profile["a4"] * compat)
    return S


def _compute_scores_2hop(
    mlp_probs_np: np.ndarray,
    feature_sim: np.ndarray,
    support_2hop: np.ndarray,
    compat: np.ndarray,
    profile: Dict[str, float],
) -> np.ndarray:
    """S2(v, c) = b1*log(p)+b2*feat_sim+b3*2hop_support+b4*compat."""
    log_p = np.log(np.clip(mlp_probs_np, _EPS, 1.0))
    S = (profile["b1"] * log_p
         + profile["b2"] * feature_sim
         + profile["b3"] * support_2hop
         + profile["b4"] * compat)
    return S


# ---------------------------------------------------------------------------
# Single-config prediction
# ---------------------------------------------------------------------------

def _predict_with_config(
    mlp_pred_all: np.ndarray,
    mlp_margin_all: np.ndarray,
    mlp_probs_np: np.ndarray,
    R1: np.ndarray,
    R2: np.ndarray,
    H1: np.ndarray,
    DeltaH: np.ndarray,
    support_1hop: np.ndarray,
    support_2hop: np.ndarray,
    feature_sim: np.ndarray,
    compat: np.ndarray,
    tau: float,
    rho1: float,
    rho2: float,
    h1_max: float,
    delta_min: float,
    profile_1hop: Dict[str, float],
    profile_2hop: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (predictions, route_array) for all nodes."""
    uncertain_mask = mlp_margin_all < tau
    route = _route_nodes(uncertain_mask, R1, R2, H1, DeltaH, rho1, rho2, h1_max, delta_min)

    preds = mlp_pred_all.copy()

    mask_1hop = route == 1
    if mask_1hop.any():
        S1 = _compute_scores_1hop(mlp_probs_np, feature_sim, support_1hop, compat, profile_1hop)
        preds[mask_1hop] = S1[mask_1hop].argmax(axis=1)

    mask_2hop = route == 2
    if mask_2hop.any():
        S2 = _compute_scores_2hop(mlp_probs_np, feature_sim, support_2hop, compat, profile_2hop)
        preds[mask_2hop] = S2[mask_2hop].argmax(axis=1)

    return preds, route


# ---------------------------------------------------------------------------
# Validation grid search
# ---------------------------------------------------------------------------

TAU_CANDIDATES = [0.05, 0.10, 0.20]
RHO1_CANDIDATES = [0.3, 0.5]
RHO2_CANDIDATES = [0.3, 0.5]
H1_MAX_CANDIDATES = [0.4, 0.6]
DELTA_MIN_CANDIDATES = [0.0, 0.15]
PROFILE_1HOP_CANDIDATES = [0, 1]
PROFILE_2HOP_CANDIDATES = [0, 1]
# Grid size: product of candidate list lengths (verified at import time below).
_EXPECTED_GRID_SIZE = (
    len(TAU_CANDIDATES) * len(RHO1_CANDIDATES) * len(RHO2_CANDIDATES)
    * len(H1_MAX_CANDIDATES) * len(DELTA_MIN_CANDIDATES)
    * len(PROFILE_1HOP_CANDIDATES) * len(PROFILE_2HOP_CANDIDATES)
)  # = 96


def _grid_search_val(
    y_true: np.ndarray,
    val_np: np.ndarray,
    mlp_pred_all: np.ndarray,
    mlp_margin_all: np.ndarray,
    mlp_probs_np: np.ndarray,
    R1: np.ndarray,
    R2: np.ndarray,
    H1: np.ndarray,
    DeltaH: np.ndarray,
    support_1hop: np.ndarray,
    support_2hop: np.ndarray,
    feature_sim: np.ndarray,
    compat: np.ndarray,
    light_grid_override: Optional[Dict[str, List]] = None,
) -> Dict[str, Any]:
    """Search over configs and return the best val-accuracy config.

    If ``light_grid_override`` is provided, its lists replace the default
    candidate lists for a cheaper search (useful for debugging / light runs).
    """
    if light_grid_override is not None:
        tau_list = light_grid_override.get("tau", TAU_CANDIDATES)
        rho1_list = light_grid_override.get("rho1", RHO1_CANDIDATES)
        rho2_list = light_grid_override.get("rho2", RHO2_CANDIDATES)
        h1_max_list = light_grid_override.get("h1_max", H1_MAX_CANDIDATES)
        delta_min_list = light_grid_override.get("delta_min", DELTA_MIN_CANDIDATES)
        pi1_list = light_grid_override.get("pi1", PROFILE_1HOP_CANDIDATES)
        pi2_list = light_grid_override.get("pi2", PROFILE_2HOP_CANDIDATES)
    else:
        tau_list = TAU_CANDIDATES
        rho1_list = RHO1_CANDIDATES
        rho2_list = RHO2_CANDIDATES
        h1_max_list = H1_MAX_CANDIDATES
        delta_min_list = DELTA_MIN_CANDIDATES
        pi1_list = PROFILE_1HOP_CANDIDATES
        pi2_list = PROFILE_2HOP_CANDIDATES

    best_val_acc = -1.0
    best_cfg: Dict[str, Any] = {}

    for tau in tau_list:
        for rho1 in rho1_list:
            for rho2 in rho2_list:
                for h1_max in h1_max_list:
                    for delta_min in delta_min_list:
                        for pi1 in pi1_list:
                            for pi2 in pi2_list:
                                preds, route = _predict_with_config(
                                    mlp_pred_all, mlp_margin_all, mlp_probs_np,
                                    R1, R2, H1, DeltaH,
                                    support_1hop, support_2hop,
                                    feature_sim, compat,
                                    tau, rho1, rho2, h1_max, delta_min,
                                    WEIGHT_PROFILES_1HOP[pi1],
                                    WEIGHT_PROFILES_2HOP[pi2],
                                )
                                val_acc = _simple_accuracy(y_true[val_np], preds[val_np])
                                if val_acc > best_val_acc:
                                    best_val_acc = val_acc
                                    best_cfg = {
                                        "tau": tau, "rho1": rho1, "rho2": rho2,
                                        "h1_max": h1_max, "delta_min": delta_min,
                                        "pi1": pi1, "pi2": pi2,
                                        "val_acc": val_acc,
                                    }

    return best_cfg


# ---------------------------------------------------------------------------
# Main callable
# ---------------------------------------------------------------------------

def ms_hsgc(
    data,
    train_indices,
    val_indices,
    test_indices,
    *,
    mlp_probs=None,
    seed: int = 1337,
    mod=None,
    light_grid_override: Optional[Dict[str, List]] = None,
    include_node_arrays: bool = False,
) -> Tuple[float, float, Dict[str, Any]]:
    """MS_HSGC: Multi-Scale Heterophily-aware Selective Graph Correction.

    Parameters
    ----------
    data : torch_geometric Data
        Graph data object with .x, .y, .edge_index.
    train_indices, val_indices, test_indices : array-like of int
        Node index arrays.
    mlp_probs : torch.Tensor or None
        Pre-computed MLP output probabilities (N, C). If None, MLP is trained.
    seed : int
        Random seed used for MLP training.
    include_node_arrays : bool
        If True, include per-node arrays for the test split in info["node_arrays"].
        Default False (canonical behaviour unchanged).
    mod : module or None
        Pre-loaded legacy module (avoids reloading on every call).
    light_grid_override : dict or None
        If provided, overrides the candidate lists used in the validation grid
        search. Keys: 'tau', 'rho1', 'rho2', 'h1_max', 'delta_min', 'pi1',
        'pi2'. Useful for lightweight/debug runs. Non-canonical.

    Returns
    -------
    val_acc, test_acc : float
    info_dict : dict
        Routing diagnostics and selected hyperparameters.
    """
    t_start = time.perf_counter()

    # Normalise indices to numpy
    def _to_np(idx):
        if isinstance(idx, torch.Tensor):
            return idx.detach().cpu().numpy().astype(np.int64)
        return np.asarray(idx, dtype=np.int64)

    train_np = _to_np(train_indices)
    val_np = _to_np(val_indices)
    test_np = _to_np(test_indices)

    if mod is None:
        mod = _load_module()

    # ------------------------------------------------------------------
    # 1. MLP probabilities
    # ------------------------------------------------------------------
    if mlp_probs is None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        mlp_probs, _ = mod.train_mlp_and_predict(
            data, _to_np_to_tensor(train_np, data.x.device),
            **mod.DEFAULT_MANUSCRIPT_MLP_KWARGS, log_file=None,
        )

    mlp_probs_np = mlp_probs.detach().cpu().numpy().astype(np.float32)
    num_classes = mlp_probs_np.shape[1]
    N = data.num_nodes
    y_true = data.y.detach().cpu().numpy().astype(np.int64)

    margin_info = mod.compute_mlp_margin(mlp_probs)
    mlp_pred_all = margin_info["mlp_pred_all"].astype(np.int64)
    mlp_margin_all = margin_info["mlp_margin_all"].astype(np.float32)

    # ------------------------------------------------------------------
    # 2. Evidence (feature similarity, compatibility)
    # ------------------------------------------------------------------
    evidence = mod._build_selective_correction_evidence(
        data, _to_np_to_tensor(train_np, data.x.device),
        mlp_probs_np=mlp_probs_np,
    )
    feature_sim = evidence["feature_similarity"].astype(np.float32)   # (N, C)
    compat = evidence["compatibility_support"].astype(np.float32)      # (N, C)

    # ------------------------------------------------------------------
    # 3. Multi-scale support
    # ------------------------------------------------------------------
    support_1hop, support_2hop = _compute_multiscale_support(
        data, train_np, mlp_probs_np, num_classes
    )

    # ------------------------------------------------------------------
    # 4. Heterophily measures
    # ------------------------------------------------------------------
    H1, H2, DeltaH = _compute_heterophily(data, mlp_pred_all, support_1hop, support_2hop)

    # ------------------------------------------------------------------
    # 5. Reliability scores
    # ------------------------------------------------------------------
    R1 = _compute_reliability_1hop(data, mlp_pred_all, support_1hop)

    edge_index_np = data.edge_index.detach().cpu().numpy()
    deg = np.bincount(edge_index_np[0].astype(np.int64), minlength=N).astype(np.float32)
    R2 = _compute_reliability_2hop(mlp_pred_all, support_2hop, deg)

    # ------------------------------------------------------------------
    # 6. Validation grid search
    # ------------------------------------------------------------------
    best_cfg = _grid_search_val(
        y_true, val_np,
        mlp_pred_all, mlp_margin_all, mlp_probs_np,
        R1, R2, H1, DeltaH,
        support_1hop, support_2hop, feature_sim, compat,
        light_grid_override=light_grid_override,
    )

    tau = best_cfg["tau"]
    rho1 = best_cfg["rho1"]
    rho2 = best_cfg["rho2"]
    h1_max = best_cfg["h1_max"]
    delta_min = best_cfg["delta_min"]
    pi1 = best_cfg["pi1"]
    pi2 = best_cfg["pi2"]
    val_acc = float(best_cfg["val_acc"])

    # ------------------------------------------------------------------
    # 7. Apply best config to test set
    # ------------------------------------------------------------------
    preds, route = _predict_with_config(
        mlp_pred_all, mlp_margin_all, mlp_probs_np,
        R1, R2, H1, DeltaH,
        support_1hop, support_2hop, feature_sim, compat,
        tau, rho1, rho2, h1_max, delta_min,
        WEIGHT_PROFILES_1HOP[pi1], WEIGHT_PROFILES_2HOP[pi2],
    )

    test_acc = _simple_accuracy(y_true[test_np], preds[test_np])

    # ------------------------------------------------------------------
    # 8. Diagnostics
    # ------------------------------------------------------------------
    route_test = route[test_np]
    n_test = len(test_np)

    frac_confident = float((route_test == 0).sum()) / max(n_test, 1)
    frac_1hop = float((route_test == 1).sum()) / max(n_test, 1)
    frac_2hop = float((route_test == 2).sum()) / max(n_test, 1)
    frac_mlp_unc = float((route_test == 3).sum()) / max(n_test, 1)

    n_uncertain_test = int((route_test != 0).sum())
    n_corrected_1hop = int((route_test == 1).sum())
    n_corrected_2hop = int((route_test == 2).sum())

    # Correction quality on test
    mlp_pred_test = mlp_pred_all[test_np]
    preds_test = preds[test_np]
    y_test = y_true[test_np]

    corrected_mask = (route_test == 1) | (route_test == 2)
    helped_mask_arr = np.zeros(n_test, dtype=bool)
    hurt_mask_arr = np.zeros(n_test, dtype=bool)
    if corrected_mask.any():
        helped_mask_arr = corrected_mask & (preds_test == y_test) & (mlp_pred_test != y_test)
        hurt_mask_arr = corrected_mask & (preds_test != y_test) & (mlp_pred_test == y_test)
        changed = preds_test[corrected_mask] != mlp_pred_test[corrected_mask]
        n_helped = int(helped_mask_arr.sum())
        n_hurt = int(hurt_mask_arr.sum())
        n_changed = int(changed.sum())
        correction_precision = float(n_helped) / max(n_changed, 1)
    else:
        n_helped = n_hurt = 0
        correction_precision = 1.0

    # ------------------------------------------------------------------
    # 8b. Routing block diagnostics (per-test-set, using best config)
    # ------------------------------------------------------------------
    R1_test = R1[test_np]
    R2_test = R2[test_np]
    H1_test = H1[test_np]
    DeltaH_test = DeltaH[test_np]
    unc_test = route_test != 0
    can_1hop_test = route_test == 1
    not_1hop = unc_test & ~can_1hop_test
    can_2hop_test = route_test == 2
    not_1hop_not_2hop = not_1hop & ~can_2hop_test

    routing_blocks: Dict[str, int] = {
        "uncertain_total": int(unc_test.sum()),
        "routed_1hop": int(can_1hop_test.sum()),
        "routed_2hop": int(can_2hop_test.sum()),
        "fallback_mlp_only": int((route_test == 3).sum()),
        # Among uncertain & not routed to 1-hop:
        "blocked_by_h1": int((not_1hop & (R1_test >= rho1) & (H1_test > h1_max)).sum()),
        "blocked_by_r1": int((not_1hop & (R1_test < rho1)).sum()),
        # Among uncertain & not routed to 1-hop or 2-hop:
        "blocked_by_r2": int((not_1hop_not_2hop & (R2_test < rho2)).sum()),
        "blocked_by_delta": int(
            (not_1hop_not_2hop & (R2_test >= rho2) & (DeltaH_test < delta_min)).sum()
        ),
    }

    runtime_sec = time.perf_counter() - t_start

    info_dict: Dict[str, Any] = {
        "val_acc": val_acc,
        "test_acc": test_acc,
        "frac_confident": frac_confident,
        "frac_mlp_only_uncertain": frac_mlp_unc,
        "frac_corrected_1hop": frac_1hop,
        "frac_corrected_2hop": frac_2hop,
        "mean_H1": float(H1.mean()),
        "mean_H2": float(H2.mean()),
        "mean_DeltaH": float(DeltaH.mean()),
        "selected_tau": tau,
        "selected_rho1": rho1,
        "selected_rho2": rho2,
        "selected_h1_max": h1_max,
        "selected_delta_min": delta_min,
        "selected_profile_1hop": pi1,
        "selected_profile_2hop": pi2,
        "n_uncertain": n_uncertain_test,
        "n_corrected_1hop": n_corrected_1hop,
        "n_corrected_2hop": n_corrected_2hop,
        "n_helped": n_helped,
        "n_hurt": n_hurt,
        "correction_precision": correction_precision,
        "routing_blocks": routing_blocks,
        "runtime_sec": runtime_sec,
    }

    if include_node_arrays:
        edge_index_np = data.edge_index.detach().cpu().numpy()
        deg_all = np.bincount(edge_index_np[0].astype(np.int64), minlength=N).astype(np.int64)
        info_dict["node_arrays"] = {
            "test_indices": test_np.copy(),
            "mlp_predictions": mlp_pred_test.copy(),
            "final_predictions": preds_test.copy(),
            "mlp_margins": mlp_margin_all[test_np].copy(),
            "true_labels": y_test.copy(),
            "route": route_test.copy(),
            "H1": H1_test.copy(),
            "H2": H2_test.copy(),
            "DeltaH": DeltaH_test.copy(),
            "R1": R1_test.copy(),
            "R2": R2_test.copy(),
            "degree": deg_all[test_np].copy(),
            "is_changed_helpful": helped_mask_arr.copy(),
            "is_changed_harmful": hurt_mask_arr.copy(),
        }

    return val_acc, test_acc, info_dict

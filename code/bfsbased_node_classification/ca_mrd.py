#!/usr/bin/env python3
"""
CA-MRD: Compatibility-Aware Multi-hop Residual Diffusion.

An experimental, non-canonical node-classification correction method that:
  1. Trains a feature-only MLP (frozen afterwards).
  2. Computes a per-node residual from MLP probabilities and training labels.
  3. Diffuses the residual through a multi-hop graph kernel.
  4. Modulates the diffusion by a label-compatibility matrix H estimated from
     training-labeled edges only.
  5. Applies the resulting correction to MLP logits selectively via a simple
     uncertainty + correction-magnitude gate.

Design constraints:
  - Feature-first: MLP is the primary predictor; the graph adds a post-hoc
    correction only.
  - No end-to-end retraining, no learned H, no nonlinear correction.
  - Lightweight and interpretable (low parameter count, clear algebra).
  - Non-canonical: does NOT touch canonical FINAL_V3 outputs.

Mathematical summary:
  R_i = Y_i - 1/K   for training nodes (one-hot minus uniform prior)
  R_i = 0            for non-training nodes (residual unknown at inference)

  P = row-normalised adjacency (transition matrix)
  K(R) = a1*(P@R) + a2*(P²@R) + a3*(P³@R) + a4*(P⁴@R)

  H[a,b] = fraction of training edges (u->v) where y_u=a and y_v=b
           (estimated from training-labeled edges with Laplace smoothing)
  H_reg  = (1-rho_H)*I + rho_H*H

  R_diff     = K(R) @ H_reg
  Z_tilde[i] = Z[i] + lambda_corr * R_diff[i]

  Gate:
    gate_i = 1  if uncertainty_i > tau_u  AND  corr_mag_i > tau_r
    gate_i = 0  otherwise

  Z_final[i] = gate_i * Z_tilde[i] + (1-gate_i) * Z[i]

Status: EXPERIMENTAL. Safe to run alongside canonical methods.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch


# ---------------------------------------------------------------------------
# Hop-weight profiles
# ---------------------------------------------------------------------------

HOP_PROFILES: Dict[str, Tuple[float, float, float, float]] = {
    "profile_A": (1.0, 0.0,  0.0,   0.0),    # 1-hop only (sanity check)
    "profile_B": (1.0, 0.5,  0.0,   0.0),    # 1-2 hop
    "profile_C": (1.0, 0.5,  0.25,  0.0),    # 1-3 hop
    "profile_D": (1.0, 0.5,  0.25,  0.125),  # 1-4 hop / geometric
}

# Default compact validation grid
DEFAULT_GRID = {
    "rho_H":        [0.0, 0.25, 0.5, 0.75],
    "lambda_corr":  [0.25, 0.5, 1.0],
    "hop_profile":  ["profile_A", "profile_B", "profile_C", "profile_D"],
    "tau_u":        [0.1, 0.2, 0.3],
    "tau_r":        [0.0, 0.05, 0.1],
    "tau_e":        [0.0],           # entropy-gate disabled by default in compact grid
}

# Minimal light grid for fast reconnaissance runs
LIGHT_GRID = {
    "rho_H":        [0.0, 0.5],
    "lambda_corr":  [0.5, 1.0],
    "hop_profile":  ["profile_A", "profile_B", "profile_C"],
    "tau_u":        [0.1, 0.2],
    "tau_r":        [0.0, 0.05],
    "tau_e":        [0.0],
}

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Utility: accuracy
# ---------------------------------------------------------------------------

def _simple_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.size == 0:
        return 0.0
    return float((y_true == y_pred).mean())


def _entropy(probs: np.ndarray) -> np.ndarray:
    """Per-node Shannon entropy, shape (N,)."""
    p = np.clip(probs, _EPS, 1.0)
    return -float(np.sum(p * np.log(p), axis=1).mean())   # scalar for logging

def _per_node_entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, _EPS, 1.0)
    return -np.sum(p * np.log(p), axis=1)


# ---------------------------------------------------------------------------
# Step 1 – Compatibility matrix
# ---------------------------------------------------------------------------

def estimate_compatibility_matrix(
    data,
    train_indices: np.ndarray,
    y_true: np.ndarray,
    num_classes: int,
    laplace_alpha: float = 1.0,
) -> np.ndarray:
    """Estimate label-compatibility matrix H from training-labeled edges.

    H[a, b] = P(neighbor label = b | node label = a) estimated over all
              directed training edges (u, v) where both endpoints are labeled.

    Only training-labeled edges are used so as not to leak test information.

    Parameters
    ----------
    data : PyG Data object with .edge_index
    train_indices : int array of training node indices
    y_true : full label array (N,)
    num_classes : K
    laplace_alpha : Laplace smoothing weight (default 1.0)

    Returns
    -------
    H : (K, K) float32 array, rows sum to 1.
    """
    train_set = set(int(i) for i in train_indices)
    edge_index = data.edge_index.detach().cpu().numpy()
    src = edge_index[0].astype(np.int64)
    dst = edge_index[1].astype(np.int64)

    # Count transitions among training-labeled edges
    H_counts = np.full((num_classes, num_classes), laplace_alpha, dtype=np.float64)
    for u, v in zip(src, dst):
        if int(u) in train_set and int(v) in train_set:
            H_counts[int(y_true[u]), int(y_true[v])] += 1.0

    row_sums = H_counts.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < _EPS, 1.0, row_sums)
    H = (H_counts / row_sums).astype(np.float32)
    return H


def regularise_compatibility(H: np.ndarray, rho_H: float) -> np.ndarray:
    """H_reg = (1 - rho_H) * I + rho_H * H.

    rho_H = 0.0 → pure identity (graph compatibility ignored)
    rho_H = 1.0 → full compatibility matrix
    """
    K = H.shape[0]
    I = np.eye(K, dtype=np.float32)
    return ((1.0 - rho_H) * I + rho_H * H).astype(np.float32)


# ---------------------------------------------------------------------------
# Step 2 – Residual construction
# ---------------------------------------------------------------------------

def build_residual(
    mlp_probs_np: np.ndarray,
    train_indices: np.ndarray,
    y_true: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Build per-node residual R.

    For training nodes:
        R_i = Y_i - 1/K
        i.e. the one-hot label minus the uniform prior.
        This gives a robust, non-zero signal (+[K-1]/K on true class,
        -1/K on all others) that does not collapse to zero when the MLP
        memorises its training data.

    For non-training nodes:  R_i = 0  (residual is unknown at inference time)

    Note: using (Y_i - P_base_i) instead would produce near-zero residuals
    on training nodes when the MLP is well-trained, which makes diffusion
    uninformative.  The uniform-prior deviation is the standard choice in
    label-diffusion / Correct-and-Smooth style methods.

    Shape: (N, K)
    """
    N = mlp_probs_np.shape[0]
    R = np.zeros((N, num_classes), dtype=np.float32)
    uniform_prior = np.full(num_classes, 1.0 / num_classes, dtype=np.float32)
    for idx in train_indices:
        i = int(idx)
        one_hot = np.zeros(num_classes, dtype=np.float32)
        one_hot[int(y_true[i])] = 1.0
        R[i] = one_hot - uniform_prior
    return R


# ---------------------------------------------------------------------------
# Step 3 – Multi-hop diffusion kernel
# ---------------------------------------------------------------------------

def build_transition_matrix(data) -> sp.csr_matrix:
    """Row-normalised adjacency (transition / random-walk matrix)."""
    N = data.num_nodes
    edge_index = data.edge_index.detach().cpu().numpy()
    src = edge_index[0].astype(np.int64)
    dst = edge_index[1].astype(np.int64)
    vals = np.ones(len(src), dtype=np.float32)
    A = sp.csr_matrix((vals, (src, dst)), shape=(N, N))
    row_sums = np.array(A.sum(axis=1)).flatten()
    row_sums = np.where(row_sums < _EPS, 1.0, row_sums)
    # Row-normalise: D^{-1} A
    D_inv = sp.diags(1.0 / row_sums)
    P = D_inv.dot(A)
    return P


def diffuse_residual(
    P: sp.csr_matrix,
    R: np.ndarray,
    a1: float,
    a2: float,
    a3: float,
    a4: float,
) -> np.ndarray:
    """K(R) = a1*(P@R) + a2*(P²@R) + a3*(P³@R) + a4*(P⁴@R).

    Computed iteratively: each hop applies P once more to the previous result.
    No dense materialisation of P^k.
    """
    PR = P.dot(R)              # P  @ R
    K_R = a1 * PR
    if a2 != 0.0:
        P2R = P.dot(PR)        # P² @ R
        K_R = K_R + a2 * P2R
        if a3 != 0.0:
            P3R = P.dot(P2R)   # P³ @ R
            K_R = K_R + a3 * P3R
            if a4 != 0.0:
                P4R = P.dot(P3R)   # P⁴ @ R
                K_R = K_R + a4 * P4R
    return K_R.astype(np.float32)


# ---------------------------------------------------------------------------
# Step 4 – Gate
# ---------------------------------------------------------------------------

def apply_gate(
    uncertainty: np.ndarray,
    corr_mag: np.ndarray,
    delta_entropy: Optional[np.ndarray],
    tau_u: float,
    tau_r: float,
    tau_e: float,
) -> np.ndarray:
    """Boolean gate mask (N,).

    gate_i = 1  if  uncertainty_i > tau_u  AND  corr_mag_i > tau_r
                    [AND delta_entropy_i > tau_e if tau_e > 0]
    """
    mask = (uncertainty > tau_u) & (corr_mag > tau_r)
    if tau_e > 0.0 and delta_entropy is not None:
        mask = mask & (delta_entropy > tau_e)
    return mask


# ---------------------------------------------------------------------------
# Core callable: ca_mrd
# ---------------------------------------------------------------------------

def ca_mrd(
    data,
    train_indices,
    val_indices,
    test_indices,
    *,
    mlp_probs: Optional[torch.Tensor] = None,
    seed: int = 1337,
    mod=None,
    light: bool = False,
    grid_override: Optional[Dict[str, List]] = None,
) -> Tuple[float, float, Dict[str, Any]]:
    """Compatibility-Aware Multi-hop Residual Diffusion (CA-MRD).

    Parameters
    ----------
    data : PyG Data object (.x, .y, .edge_index)
    train_indices, val_indices, test_indices : array-like of int
    mlp_probs : optional pre-computed MLP softmax probabilities (N, K).
        If None, the MLP is trained using mod utilities.
    seed : random seed
    mod : pre-loaded legacy module (avoids reloading on every call)
    light : use the compact LIGHT_GRID instead of DEFAULT_GRID
    grid_override : custom grid dict overriding the chosen default

    Returns
    -------
    val_acc : float
    test_acc : float
    info : dict with diagnostics
    """
    t_start = time.perf_counter()

    # --- normalise indices ---
    def _to_np(idx):
        if isinstance(idx, torch.Tensor):
            return idx.detach().cpu().numpy().astype(np.int64)
        return np.asarray(idx, dtype=np.int64)

    train_np = _to_np(train_indices)
    val_np = _to_np(val_indices)
    test_np = _to_np(test_indices)

    if mod is None:
        raise ValueError("mod must be provided (pre-loaded legacy module)")

    # --- MLP ---
    t_mlp = time.perf_counter()
    if mlp_probs is None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        mlp_probs, _ = mod.train_mlp_and_predict(
            data, _to_np(train_indices),
            **mod.DEFAULT_MANUSCRIPT_MLP_KWARGS, log_file=None,
        )
    mlp_time = time.perf_counter() - t_mlp

    mlp_probs_np = mlp_probs.detach().cpu().numpy().astype(np.float32)
    N, K = mlp_probs_np.shape
    y_true = data.y.detach().cpu().numpy().astype(np.int64)

    mlp_info = mod.compute_mlp_margin(mlp_probs)
    mlp_pred_all = mlp_info["mlp_pred_all"].astype(np.int64)
    mlp_test_acc = _simple_accuracy(y_true[test_np], mlp_pred_all[test_np])

    # Uncertainty: 1 - max_class_prob
    uncertainty = 1.0 - mlp_probs_np.max(axis=1)   # (N,)

    # Per-node baseline entropy
    base_entropy = _per_node_entropy(mlp_probs_np)  # (N,)

    # MLP logits (log-odds; recompute as log(p) for stability)
    Z = np.log(np.clip(mlp_probs_np, _EPS, 1.0))   # (N, K)

    # --- Compatibility matrix (fixed, estimated from training edges) ---
    H_raw = estimate_compatibility_matrix(data, train_np, y_true, K)

    # --- Residual (training nodes only) ---
    R = build_residual(mlp_probs_np, train_np, y_true, K)

    # --- Transition matrix (fixed, computed once) ---
    P = build_transition_matrix(data)

    # --- Grid search on validation ---
    grid = grid_override if grid_override is not None else (
        LIGHT_GRID if light else DEFAULT_GRID
    )
    tau_list = grid.get("tau_u", DEFAULT_GRID["tau_u"])
    tau_r_list = grid.get("tau_r", DEFAULT_GRID["tau_r"])
    tau_e_list = grid.get("tau_e", DEFAULT_GRID["tau_e"])
    rho_H_list = grid.get("rho_H", DEFAULT_GRID["rho_H"])
    lambda_list = grid.get("lambda_corr", DEFAULT_GRID["lambda_corr"])
    profile_list = grid.get("hop_profile", DEFAULT_GRID["hop_profile"])

    # Pre-compute diffused residuals per hop profile (shared across rho_H / lambda)
    diffused_per_profile: Dict[str, np.ndarray] = {}
    for pname in profile_list:
        a1, a2, a3, a4 = HOP_PROFILES[pname]
        diffused_per_profile[pname] = diffuse_residual(P, R, a1, a2, a3, a4)

    best_val_acc = -1.0
    best_key = None
    best_cfg: Dict[str, Any] = {}

    for rho_H in rho_H_list:
        H_reg = regularise_compatibility(H_raw, rho_H)
        for pname in profile_list:
            KR = diffused_per_profile[pname]
            # Compatibility-aware diffused residual: (N, K) @ (K, K)
            R_diff = KR.dot(H_reg)  # (N, K)

            # Correction magnitude per node
            corr_mag = np.abs(R_diff).sum(axis=1)   # (N,) L1

            for lambda_corr in lambda_list:
                Z_tilde = Z + lambda_corr * R_diff

                # Corrected probabilities (softmax of Z_tilde)
                Z_tilde_shifted = Z_tilde - Z_tilde.max(axis=1, keepdims=True)
                exp_Z = np.exp(Z_tilde_shifted)
                P_tilde = (exp_Z / exp_Z.sum(axis=1, keepdims=True)).astype(np.float32)

                # Entropy reduction
                delta_entropy = base_entropy - _per_node_entropy(P_tilde)   # (N,)

                for tau_u in tau_list:
                    for tau_r in tau_r_list:
                        for tau_e in tau_e_list:
                            gate = apply_gate(
                                uncertainty, corr_mag, delta_entropy,
                                tau_u, tau_r, tau_e,
                            )
                            # Apply on validation
                            Z_final_val = Z.copy()
                            if gate.any():
                                Z_final_val[gate] = Z_tilde[gate]

                            pred_val = np.argmax(Z_final_val[val_np], axis=1).astype(np.int64)
                            val_acc = _simple_accuracy(y_true[val_np], pred_val)
                            changed_frac = float(
                                (np.argmax(Z_final_val[val_np], axis=1)
                                 != mlp_pred_all[val_np]).mean()
                            )
                            key = (val_acc, -changed_frac)
                            if best_key is None or key > best_key:
                                best_key = key
                                best_val_acc = val_acc
                                best_cfg = {
                                    "rho_H": rho_H, "lambda_corr": lambda_corr,
                                    "hop_profile": pname,
                                    "tau_u": tau_u, "tau_r": tau_r, "tau_e": tau_e,
                                    "gate_mask": gate.copy(),
                                    "Z_tilde": Z_tilde.copy(),
                                    "H_reg": H_reg.copy(),
                                    "R_diff": R_diff.copy(),
                                    "P_tilde": P_tilde.copy(),
                                    "delta_entropy": delta_entropy.copy(),
                                    "corr_mag": corr_mag.copy(),
                                }

    # --- Apply best config to full graph / test ---
    gate = best_cfg["gate_mask"]
    Z_final = Z.copy()
    if gate.any():
        Z_final[gate] = best_cfg["Z_tilde"][gate]

    final_pred = np.argmax(Z_final, axis=1).astype(np.int64)
    test_acc = _simple_accuracy(y_true[test_np], final_pred[test_np])
    val_acc_final = _simple_accuracy(y_true[val_np], final_pred[val_np])

    # --- Diagnostics ---
    test_gate = gate[test_np]
    test_changed = (final_pred[test_np] != mlp_pred_all[test_np])
    test_mlp_correct = (mlp_pred_all[test_np] == y_true[test_np])
    test_final_correct = (final_pred[test_np] == y_true[test_np])
    test_changed_idx = np.where(test_changed)[0]

    n_helped = int((test_changed & (~test_mlp_correct) & test_final_correct).sum())
    n_hurt = int((test_changed & test_mlp_correct & (~test_final_correct)).sum())
    net_help = n_helped - n_hurt
    n_changed = int(test_changed.sum())
    correction_precision = float(n_helped) / max(n_changed, 1)

    frac_corrected = float(test_gate.sum()) / max(len(test_np), 1)
    mean_unc_corrected = float(
        uncertainty[test_np][test_gate].mean()
    ) if test_gate.any() else float("nan")
    mean_corr_mag = float(
        best_cfg["corr_mag"][test_np][test_gate].mean()
    ) if test_gate.any() else float("nan")
    mean_delta_entropy_corrected = float(
        best_cfg["delta_entropy"][test_np][test_gate].mean()
    ) if test_gate.any() else float("nan")

    total_time = time.perf_counter() - t_start

    # H_reg as dict for logging
    H_reg_list = best_cfg["H_reg"].tolist()

    info: Dict[str, Any] = {
        "method": "CA_MRD",
        "mlp_test_acc": float(mlp_test_acc),
        "val_acc": float(val_acc_final),
        "test_acc": float(test_acc),
        "delta_vs_mlp": float(test_acc - mlp_test_acc),
        # Selected hyperparameters
        "selected_rho_H": best_cfg["rho_H"],
        "selected_lambda_corr": best_cfg["lambda_corr"],
        "selected_hop_profile": best_cfg["hop_profile"],
        "selected_tau_u": best_cfg["tau_u"],
        "selected_tau_r": best_cfg["tau_r"],
        "selected_tau_e": best_cfg["tau_e"],
        # Gate diagnostics
        "frac_corrected": frac_corrected,
        "mean_uncertainty_corrected": mean_unc_corrected,
        "mean_corr_mag": mean_corr_mag,
        "mean_delta_entropy_corrected": mean_delta_entropy_corrected,
        # Correction quality
        "n_helped": n_helped,
        "n_hurt": n_hurt,
        "net_help": net_help,
        "n_changed": n_changed,
        "correction_precision": correction_precision,
        # Compatibility matrix (serialisable)
        "H_reg": H_reg_list,
        # Timing
        "runtime_sec": float(total_time),
        "mlp_time_sec": float(mlp_time),
    }

    return float(val_acc_final), float(test_acc), info

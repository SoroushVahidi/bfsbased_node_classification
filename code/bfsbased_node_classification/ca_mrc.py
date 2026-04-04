#!/usr/bin/env python3
"""
CA-MRC: Compatibility-Aware Multi-hop Residual Correction.

EXPERIMENTAL — does NOT affect canonical FINAL_V3 outputs.

Method summary:
  1. Train a feature-only MLP as the base classifier (reuses repo utilities).
  2. Estimate a compatibility matrix C ∈ R^{K×K} from training edges/labels only.
     C[a,b] ≈ P(class b in neighborhood | node class is a).
  3. Compute a per-node residual r from labeled training nodes
     (r_i = y_i_one_hot - p_i for train nodes, 0 elsewhere).
  4. Diffuse the residual through the graph:
        r_hat = alpha1 * (P @ r @ C) + alpha2 * (P2 @ r @ C)
     where P is the row-stochastic transition matrix and P2 = P @ P.
  5. Add a scaled correction to the MLP logits:
        z_tilde = z + lambda_corr * r_hat
  6. Apply a selective gate:
     gate_i = 1  iff  delta_entropy_i > tau_H  and  corr_mag_i > tau_R
        delta_entropy_i = H(softmax(z_i)) - H(softmax(z_tilde_i))
        corr_mag_i      = ||r_hat_i||_1
  7. Final prediction:
        z_final_i = gate_i * z_tilde_i + (1 - gate_i) * z_i

Ablation modes (controlled via `ablation` parameter):
  - "full"         : compatibility-aware, gated (default)
  - "no_compat"    : skip C (use identity), diffuse raw residual
  - "always_apply" : gate always passes (gate_i = 1 for all uncertain nodes)
  - "entropy_only" : gate uses delta_entropy > tau_H only
  - "mag_only"     : gate uses corr_mag > tau_R only
"""
from __future__ import annotations

import importlib.util
import os
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch


# ---------------------------------------------------------------------------
# Constants / Grid
# ---------------------------------------------------------------------------

LIGHT_GRID: Dict[str, List] = {
    "alpha1":     [1.0],
    "alpha2":     [0.25, 0.5, 1.0],
    "lambda_corr":[0.25, 0.5, 1.0],
    "tau_H":      [0.0, 0.01, 0.02],
    "tau_R":      [0.05, 0.1, 0.2],
}
# Total: 1 × 3 × 3 × 3 × 3 = 81 configs (vectorised, fast)

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Module loader (reuses pattern from run_ms_hsgc_evaluation.py)
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
# Utilities
# ---------------------------------------------------------------------------

def _simple_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.size == 0:
        return 0.0
    return float((y_true == y_pred).mean())


def _entropy(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy per row (nats). probs: (N, K)."""
    clipped = np.clip(probs, _EPS, 1.0)
    return float(-1.0) * (clipped * np.log(clipped)).sum(axis=1)


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax."""
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# 1. Compatibility matrix
# ---------------------------------------------------------------------------

def _estimate_compatibility_matrix(
    data,
    train_indices: np.ndarray,
    num_classes: int,
    smoothing: float = 1.0,
) -> np.ndarray:
    """Estimate C ∈ R^{K×K} from training edges only.

    C[a, b] ≈ P(neighbor has class b | node has class a), estimated
    from (train_node, neighbor) pairs where the neighbor is also a
    training node.  Laplace-smoothed and row-normalised.

    Parameters
    ----------
    data        : PyG Data object with .y and .edge_index
    train_indices : 1-D int array of training node indices
    num_classes : K
    smoothing   : Laplace smoothing pseudo-count per cell

    Returns
    -------
    C : (K, K) float32 row-normalised matrix
    """
    train_set = set(train_indices.tolist())
    y_all = data.y.detach().cpu().numpy().astype(np.int64)

    ei = data.edge_index.detach().cpu().numpy()
    row, col = ei[0].astype(np.int64), ei[1].astype(np.int64)

    C = np.full((num_classes, num_classes), smoothing, dtype=np.float64)
    for u, v in zip(row, col):
        if int(u) in train_set and int(v) in train_set:
            a = int(y_all[u])
            b = int(y_all[v])
            C[a, b] += 1.0

    row_sums = C.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < _EPS, 1.0, row_sums)
    return (C / row_sums).astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Row-stochastic transition matrix (sparse)
# ---------------------------------------------------------------------------

def _build_transition_matrix(data) -> sp.csr_matrix:
    """Build row-stochastic adjacency P (self-loops included)."""
    N = data.num_nodes
    ei = data.edge_index.detach().cpu().numpy()
    row_idx = ei[0].astype(np.int64)
    col_idx = ei[1].astype(np.int64)

    # add self-loops
    self_loop_nodes = np.arange(N, dtype=np.int64)
    row_idx = np.concatenate([row_idx, self_loop_nodes])
    col_idx = np.concatenate([col_idx, self_loop_nodes])
    vals = np.ones(len(row_idx), dtype=np.float32)

    A = sp.csr_matrix((vals, (row_idx, col_idx)), shape=(N, N))
    # row-normalise
    row_sums = np.array(A.sum(axis=1)).flatten()
    row_sums = np.where(row_sums < _EPS, 1.0, row_sums)
    D_inv = sp.diags(1.0 / row_sums, format="csr")
    return D_inv.dot(A)


# ---------------------------------------------------------------------------
# 3. Residual construction
# ---------------------------------------------------------------------------

def _build_residual(
    train_indices: np.ndarray,
    y_all: np.ndarray,
    mlp_probs_np: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Residual r ∈ R^{N×K}.

    r_i = one_hot(y_i) - p_i   for i in train_indices
    r_i = 0                     otherwise
    """
    N = mlp_probs_np.shape[0]
    r = np.zeros((N, num_classes), dtype=np.float32)
    for idx in train_indices:
        i = int(idx)
        one_hot = np.zeros(num_classes, dtype=np.float32)
        one_hot[int(y_all[i])] = 1.0
        r[i] = one_hot - mlp_probs_np[i]
    return r


# ---------------------------------------------------------------------------
# 4. Residual diffusion
# ---------------------------------------------------------------------------

def _diffuse_residual(
    P: sp.csr_matrix,
    r: np.ndarray,
    C: np.ndarray,
    alpha1: float,
    alpha2: float,
    use_compat: bool = True,
) -> np.ndarray:
    """r_hat = alpha1*(P@r@C) + alpha2*(P2@r@C).

    If use_compat=False, C is treated as identity (plain multi-hop diffusion).
    """
    Pr = P.dot(r)          # (N, K) — 1-hop diffused residual
    P2r = P.dot(Pr)        # (N, K) — 2-hop diffused residual

    if use_compat:
        r_hat = alpha1 * Pr.dot(C) + alpha2 * P2r.dot(C)
    else:
        r_hat = alpha1 * Pr + alpha2 * P2r

    return r_hat.astype(np.float32)


# ---------------------------------------------------------------------------
# 5. Selective gate
# ---------------------------------------------------------------------------

def _apply_gate(
    mlp_logits: np.ndarray,
    r_hat: np.ndarray,
    lambda_corr: float,
    tau_H: float,
    tau_R: float,
    mode: str = "full",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply selective gate and return final logits + diagnostics.

    Parameters
    ----------
    mlp_logits : (N, K) float — raw MLP output logits (before softmax)
    r_hat      : (N, K) float — diffused residual
    lambda_corr: scaling factor for correction
    tau_H      : entropy-reduction threshold
    tau_R      : correction-magnitude threshold
    mode       : "full" | "no_compat" | "always_apply" | "entropy_only" | "mag_only"

    Returns
    -------
    final_logits   : (N, K)
    gate           : (N,) bool
    delta_entropy  : (N,) float
    corr_mag       : (N,) float
    """
    z_tilde = mlp_logits + lambda_corr * r_hat

    p_base = _softmax(mlp_logits)
    p_corr = _softmax(z_tilde)

    H_base = _entropy(p_base)
    H_corr = _entropy(p_corr)
    delta_entropy = H_base - H_corr       # positive = entropy decreased = more confident
    corr_mag = np.abs(r_hat).sum(axis=1)   # L1 magnitude of correction

    if mode == "always_apply":
        gate = np.ones(len(mlp_logits), dtype=bool)
    elif mode == "entropy_only":
        gate = delta_entropy > tau_H
    elif mode == "mag_only":
        gate = corr_mag > tau_R
    else:  # "full" or "no_compat"
        gate = (delta_entropy > tau_H) & (corr_mag > tau_R)

    gate_f = gate.astype(np.float32)[:, None]
    final_logits = gate_f * z_tilde + (1.0 - gate_f) * mlp_logits

    return final_logits, gate, delta_entropy, corr_mag


# ---------------------------------------------------------------------------
# 6. Single-config evaluation
# ---------------------------------------------------------------------------

def _eval_config(
    mlp_logits: np.ndarray,
    r_hat: np.ndarray,
    y_true: np.ndarray,
    eval_idx: np.ndarray,
    lambda_corr: float,
    tau_H: float,
    tau_R: float,
    mode: str = "full",
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    final_logits, gate, delta_entropy, corr_mag = _apply_gate(
        mlp_logits, r_hat, lambda_corr, tau_H, tau_R, mode
    )
    preds = final_logits.argmax(axis=1)
    acc = _simple_accuracy(y_true[eval_idx], preds[eval_idx])
    return acc, final_logits, gate, delta_entropy, corr_mag


# ---------------------------------------------------------------------------
# 7. Validation grid search
# ---------------------------------------------------------------------------

def _grid_search(
    mlp_logits: np.ndarray,
    r_hat: np.ndarray,
    y_true: np.ndarray,
    val_idx: np.ndarray,
    mlp_preds: np.ndarray,
    grid: Optional[Dict[str, List]] = None,
    mode: str = "full",
) -> Dict[str, Any]:
    """Search over (lambda_corr, tau_H, tau_R) on validation set.

    alpha1/alpha2 are already baked into r_hat.  Grid search is over
    correction strength and gate thresholds.

    Tie-break: higher val_acc → lower changed_frac.
    """
    if grid is None:
        grid = LIGHT_GRID

    best_val_acc = -1.0
    best_changed_frac = 1.0
    best_cfg: Dict[str, Any] = {}

    for lambda_corr in grid.get("lambda_corr", [0.5]):
        for tau_H in grid.get("tau_H", [0.01]):
            for tau_R in grid.get("tau_R", [0.1]):
                acc, final_logits, gate, _, _ = _eval_config(
                    mlp_logits, r_hat, y_true, val_idx,
                    lambda_corr, tau_H, tau_R, mode
                )
                preds = final_logits.argmax(axis=1)
                changed_frac = float((preds != mlp_preds).mean())

                if (acc > best_val_acc or
                        (acc == best_val_acc and changed_frac < best_changed_frac)):
                    best_val_acc = acc
                    best_changed_frac = changed_frac
                    best_cfg = {
                        "lambda_corr": lambda_corr,
                        "tau_H": tau_H,
                        "tau_R": tau_R,
                        "val_acc": acc,
                        "changed_frac": changed_frac,
                    }

    return best_cfg


# ---------------------------------------------------------------------------
# 8. Main CA-MRC callable
# ---------------------------------------------------------------------------

def ca_mrc(
    data,
    train_indices,
    val_indices,
    test_indices,
    *,
    mlp_probs: Optional[torch.Tensor] = None,
    seed: int = 1337,
    mod=None,
    mlp_kwargs: Optional[Dict[str, Any]] = None,
    grid: Optional[Dict[str, List]] = None,
    ablation: str = "full",
    save_compat_matrix_path: Optional[str] = None,
) -> Tuple[float, float, Dict[str, Any]]:
    """Compatibility-Aware Multi-hop Residual Correction.

    Parameters
    ----------
    data           : PyG Data with .x, .y, .edge_index
    train_indices,
    val_indices,
    test_indices   : array-like of int
    mlp_probs      : pre-computed MLP probabilities (N, K). If None, MLP is trained.
    seed           : random seed for MLP training
    mod            : pre-loaded legacy module (avoids reload)
    mlp_kwargs     : MLP training kwargs; defaults to mod.DEFAULT_MANUSCRIPT_MLP_KWARGS
    grid           : validation grid dict; defaults to LIGHT_GRID
    ablation       : "full" | "no_compat" | "always_apply" | "entropy_only" | "mag_only"
    save_compat_matrix_path : if given, write C to this CSV path

    Returns
    -------
    val_acc, test_acc : float
    info_dict         : dict with diagnostics
    """
    t_start = time.perf_counter()

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
    # 1. MLP
    # ------------------------------------------------------------------
    if mlp_probs is None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        kw = mlp_kwargs or mod.DEFAULT_MANUSCRIPT_MLP_KWARGS
        train_t = torch.as_tensor(train_np, dtype=torch.long, device=data.x.device)
        mlp_probs, _ = mod.train_mlp_and_predict(
            data, train_t, **kw, log_file=None,
        )

    mlp_probs_np = mlp_probs.detach().cpu().numpy().astype(np.float32)
    num_classes = mlp_probs_np.shape[1]
    N = data.num_nodes
    y_all = data.y.detach().cpu().numpy().astype(np.int64)

    # MLP logits (log-prob, sufficient as logits for argmax / entropy)
    mlp_logits = np.log(np.clip(mlp_probs_np, _EPS, 1.0)).astype(np.float32)
    mlp_preds = mlp_probs_np.argmax(axis=1).astype(np.int64)

    # ------------------------------------------------------------------
    # 2. Compatibility matrix
    # ------------------------------------------------------------------
    C = _estimate_compatibility_matrix(data, train_np, num_classes, smoothing=1.0)
    if save_compat_matrix_path is not None:
        try:
            os.makedirs(os.path.dirname(save_compat_matrix_path) or ".", exist_ok=True)
            np.savetxt(save_compat_matrix_path, C, delimiter=",", fmt="%.6f")
        except Exception as exc:
            warnings.warn(f"Could not save compatibility matrix: {exc}")

    # ------------------------------------------------------------------
    # 3. Transition matrix
    # ------------------------------------------------------------------
    P = _build_transition_matrix(data)

    # ------------------------------------------------------------------
    # 4. Residual
    # ------------------------------------------------------------------
    r = _build_residual(train_np, y_all, mlp_probs_np, num_classes)

    # ------------------------------------------------------------------
    # 5. Diffuse over alpha grid → pick best (alpha1, alpha2) on val
    # ------------------------------------------------------------------
    use_compat = (ablation != "no_compat")
    _grid = grid if grid is not None else LIGHT_GRID

    # Pre-compute r_hat for each (alpha1, alpha2) pair, then search
    best_val_acc_outer = -1.0
    best_changed_outer = 1.0
    best_cfg_outer: Dict[str, Any] = {}
    best_r_hat: Optional[np.ndarray] = None

    for alpha1 in _grid.get("alpha1", [1.0]):
        for alpha2 in _grid.get("alpha2", [0.5]):
            r_hat = _diffuse_residual(P, r, C, alpha1, alpha2, use_compat=use_compat)
            cfg = _grid_search(
                mlp_logits, r_hat, y_all, val_np, mlp_preds,
                grid=_grid, mode=ablation,
            )
            va = cfg.get("val_acc", -1.0)
            cf = cfg.get("changed_frac", 1.0)
            if va > best_val_acc_outer or (va == best_val_acc_outer and cf < best_changed_outer):
                best_val_acc_outer = va
                best_changed_outer = cf
                best_cfg_outer = dict(cfg, alpha1=alpha1, alpha2=alpha2)
                best_r_hat = r_hat

    if best_r_hat is None:
        # Fallback — should not happen if grid is non-empty
        best_r_hat = np.zeros_like(mlp_logits)
        best_cfg_outer = {"alpha1": 1.0, "alpha2": 0.5, "lambda_corr": 0.5,
                          "tau_H": 0.01, "tau_R": 0.1, "val_acc": 0.0}

    # ------------------------------------------------------------------
    # 6. Apply best config to all nodes
    # ------------------------------------------------------------------
    alpha1 = best_cfg_outer["alpha1"]
    alpha2 = best_cfg_outer["alpha2"]
    lambda_corr = best_cfg_outer["lambda_corr"]
    tau_H = best_cfg_outer["tau_H"]
    tau_R = best_cfg_outer["tau_R"]
    val_acc = float(best_cfg_outer["val_acc"])

    final_logits, gate, delta_entropy, corr_mag = _apply_gate(
        mlp_logits, best_r_hat, lambda_corr, tau_H, tau_R, mode=ablation
    )
    final_preds = final_logits.argmax(axis=1).astype(np.int64)
    test_acc = _simple_accuracy(y_all[test_np], final_preds[test_np])

    # ------------------------------------------------------------------
    # 7. Diagnostics
    # ------------------------------------------------------------------
    gate_test = gate[test_np]
    n_test = len(test_np)
    frac_corrected = float(gate_test.sum()) / max(n_test, 1)
    mean_delta_entropy = float(delta_entropy[test_np].mean())
    mean_corr_mag = float(corr_mag[test_np].mean())

    preds_test = final_preds[test_np]
    mlp_preds_test = mlp_preds[test_np]
    y_test = y_all[test_np]

    corrected_test = gate_test
    if corrected_test.any():
        helped = int(((preds_test == y_test) & (mlp_preds_test != y_test) & corrected_test).sum())
        hurt = int(((preds_test != y_test) & (mlp_preds_test == y_test) & corrected_test).sum())
    else:
        helped = hurt = 0
    net_help = helped - hurt

    # Margin-bin analysis on corrected test nodes (low/medium/high MLP confidence)
    mlp_margin_test = np.sort(mlp_probs_np[test_np], axis=1)[:, -1] - np.sort(mlp_probs_np[test_np], axis=1)[:, -2]
    if corrected_test.any():
        margins_corrected = mlp_margin_test[corrected_test]
        frac_low = float((margins_corrected < 0.2).mean())
        frac_med = float(((margins_corrected >= 0.2) & (margins_corrected < 0.5)).mean())
        frac_high = float((margins_corrected >= 0.5).mean())
    else:
        frac_low = frac_med = frac_high = float("nan")

    runtime_sec = time.perf_counter() - t_start

    info_dict: Dict[str, Any] = {
        "val_acc": val_acc,
        "test_acc": test_acc,
        "ablation": ablation,
        "alpha1": alpha1,
        "alpha2": alpha2,
        "lambda_corr": lambda_corr,
        "tau_H": tau_H,
        "tau_R": tau_R,
        "frac_corrected": frac_corrected,
        "mean_delta_entropy": mean_delta_entropy,
        "mean_corr_mag": mean_corr_mag,
        "helped_count": helped,
        "hurt_count": hurt,
        "net_help": net_help,
        "frac_corrected_low_margin": frac_low,
        "frac_corrected_med_margin": frac_med,
        "frac_corrected_high_margin": frac_high,
        "compat_matrix": C,
        "runtime_sec": runtime_sec,
    }

    return val_acc, test_acc, info_dict

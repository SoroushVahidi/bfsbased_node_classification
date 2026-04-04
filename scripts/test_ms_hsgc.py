#!/usr/bin/env python3
"""
Lightweight validation tests for MS_HSGC.

Tests:
  1. py_compile check on ms_hsgc.py
  2. Routing fractions sum to 1 on synthetic data
  3. H1, H2 in [0, 1]
  4. DeltaH == H1 - H2
  5. Expected info_dict keys are present
  6. Smoke test on tiny synthetic graph

Run with:
  python3 scripts/test_ms_hsgc.py

Exit code 0 = all tests passed; non-zero = at least one failure.
"""
from __future__ import annotations

import os
import py_compile
import sys
import traceback

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CODE_DIR = os.path.join(REPO_ROOT, "code", "bfsbased_node_classification")
MS_HSGC_PATH = os.path.join(CODE_DIR, "ms_hsgc.py")

if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

_FAILURES: list = []
_PASSES: list = []


def _ok(name: str):
    _PASSES.append(name)
    print(f"  PASS: {name}")


def _fail(name: str, msg: str):
    _FAILURES.append(name)
    print(f"  FAIL: {name} — {msg}")


# ---------------------------------------------------------------------------
# Minimal torch_geometric-style Data stub
# ---------------------------------------------------------------------------

class _FakeData:
    """Tiny stand-in for torch_geometric.data.Data."""

    def __init__(self, N: int, C: int, num_edges: int, seed: int = 0):
        rng = np.random.RandomState(seed)
        self.num_nodes = N

        # Features (N, F)
        F = 8
        self.x = torch.tensor(rng.randn(N, F).astype(np.float32))

        # Labels (N,)
        self.y = torch.tensor(rng.randint(0, C, size=N).astype(np.int64))

        # Random edge_index (2, E) — undirected
        src = rng.randint(0, N, size=num_edges)
        dst = rng.randint(0, N, size=num_edges)
        # Ensure undirected
        src_both = np.concatenate([src, dst])
        dst_both = np.concatenate([dst, src])
        edge_index = np.stack([src_both, dst_both], axis=0)
        self.edge_index = torch.tensor(edge_index.astype(np.int64))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_py_compile():
    """Test 1: ms_hsgc.py is syntactically valid."""
    name = "py_compile ms_hsgc.py"
    try:
        py_compile.compile(MS_HSGC_PATH, doraise=True)
        _ok(name)
    except py_compile.PyCompileError as e:
        _fail(name, str(e))


def test_imports():
    """Test 2: ms_hsgc module imports without error."""
    name = "import ms_hsgc"
    try:
        import ms_hsgc  # noqa: F401
        _ok(name)
        return True
    except Exception as e:
        _fail(name, str(e))
        return False


def test_multiscale_support_shapes():
    """Test 3: _compute_multiscale_support returns correct shapes."""
    name = "_compute_multiscale_support shapes"
    try:
        from ms_hsgc import _compute_multiscale_support
        N, C = 20, 4
        data = _FakeData(N=N, C=C, num_edges=40)
        train_np = np.arange(5, dtype=np.int64)
        mlp_probs = np.random.dirichlet(np.ones(C), size=N).astype(np.float32)
        s1, s2 = _compute_multiscale_support(data, train_np, mlp_probs, C)
        assert s1.shape == (N, C), f"s1 shape {s1.shape}"
        assert s2.shape == (N, C), f"s2 shape {s2.shape}"
        _ok(name)
    except Exception as e:
        _fail(name, traceback.format_exc())


def test_support_row_sums():
    """Test 4: Support rows sum to ~1."""
    name = "support rows sum to 1"
    try:
        from ms_hsgc import _compute_multiscale_support
        N, C = 30, 3
        data = _FakeData(N=N, C=C, num_edges=60)
        train_np = np.arange(6, dtype=np.int64)
        mlp_probs = np.random.dirichlet(np.ones(C), size=N).astype(np.float32)
        s1, s2 = _compute_multiscale_support(data, train_np, mlp_probs, C)
        assert np.allclose(s1.sum(axis=1), 1.0, atol=1e-5), "s1 row sums off"
        assert np.allclose(s2.sum(axis=1), 1.0, atol=1e-5), "s2 row sums off"
        _ok(name)
    except Exception as e:
        _fail(name, traceback.format_exc())


def test_heterophily_range():
    """Test 5: H1, H2 in [0, 1] and DeltaH = H1 - H2."""
    name = "H1 H2 in [0,1] and DeltaH = H1 - H2"
    try:
        from ms_hsgc import _compute_multiscale_support, _compute_heterophily
        N, C = 25, 5
        data = _FakeData(N=N, C=C, num_edges=50)
        train_np = np.arange(5, dtype=np.int64)
        mlp_probs = np.random.dirichlet(np.ones(C), size=N).astype(np.float32)
        mlp_pred_all = mlp_probs.argmax(axis=1).astype(np.int64)
        s1, s2 = _compute_multiscale_support(data, train_np, mlp_probs, C)
        H1, H2, DeltaH = _compute_heterophily(data, mlp_pred_all, s1, s2)
        assert H1.min() >= -1e-6, f"H1 min {H1.min()}"
        assert H1.max() <= 1.0 + 1e-6, f"H1 max {H1.max()}"
        assert H2.min() >= -1e-6, f"H2 min {H2.min()}"
        assert H2.max() <= 1.0 + 1e-6, f"H2 max {H2.max()}"
        assert np.allclose(DeltaH, H1 - H2, atol=1e-5), "DeltaH != H1 - H2"
        _ok(name)
    except Exception as e:
        _fail(name, traceback.format_exc())


def test_route_fractions_sum():
    """Test 6: Routing fractions sum to 1 across all nodes."""
    name = "routing fractions sum to 1"
    try:
        from ms_hsgc import _route_nodes
        N = 50
        rng = np.random.RandomState(42)
        uncertain_mask = rng.rand(N) > 0.5
        R1 = rng.rand(N).astype(np.float32)
        R2 = rng.rand(N).astype(np.float32)
        H1 = rng.rand(N).astype(np.float32)
        DeltaH = rng.randn(N).astype(np.float32) * 0.2
        route = _route_nodes(uncertain_mask, R1, R2, H1, DeltaH,
                             rho1=0.4, rho2=0.4, h1_max=0.5, delta_min=0.0)
        counts = np.array([
            (route == 0).sum(), (route == 1).sum(),
            (route == 2).sum(), (route == 3).sum(),
        ])
        assert counts.sum() == N, f"Sum of route buckets {counts.sum()} != {N}"
        # Fractions must sum to 1
        fracs = counts / N
        assert abs(fracs.sum() - 1.0) < 1e-9, f"Fractions sum {fracs.sum()}"
        _ok(name)
    except Exception as e:
        _fail(name, traceback.format_exc())


def test_info_dict_keys():
    """Test 7: Smoke test on tiny synthetic graph — check info_dict keys."""
    name = "smoke test info_dict keys"
    expected_keys = {
        "val_acc", "test_acc",
        "frac_confident", "frac_mlp_only_uncertain",
        "frac_corrected_1hop", "frac_corrected_2hop",
        "mean_H1", "mean_H2", "mean_DeltaH",
        "selected_tau", "selected_rho1", "selected_rho2",
        "selected_h1_max", "selected_delta_min",
        "selected_profile_1hop", "selected_profile_2hop",
        "n_uncertain", "n_corrected_1hop", "n_corrected_2hop",
        "n_helped", "n_hurt", "correction_precision",
        "runtime_sec",
    }
    try:
        import ms_hsgc as _ms
        # Build a tiny but realistic synthetic graph
        N, C, num_edges = 40, 3, 80
        data = _FakeData(N=N, C=C, num_edges=num_edges, seed=7)

        # Fake MLP probs (N, C) — row-normalised random
        rng = np.random.RandomState(99)
        probs_np = np.random.dirichlet(np.ones(C), size=N).astype(np.float32)
        mlp_probs = torch.tensor(probs_np)

        # Simple index split
        train_np = np.arange(0, 10, dtype=np.int64)
        val_np = np.arange(10, 20, dtype=np.int64)
        test_np = np.arange(20, N, dtype=np.int64)

        # Build a minimal mod stub (avoids loading the full legacy module)
        import types
        mod = types.SimpleNamespace()

        def fake_train_mlp(data, train_idx, **kwargs):
            return mlp_probs, None

        def fake_compute_margin(probs):
            p = probs.numpy()
            sorted_p = np.sort(p, axis=1)[:, ::-1]
            margin = sorted_p[:, 0] - sorted_p[:, 1]
            return {
                "mlp_pred_all": p.argmax(axis=1).astype(np.int64),
                "mlp_top1_all": sorted_p[:, 0].astype(np.float32),
                "mlp_top2_all": sorted_p[:, 1].astype(np.float32),
                "mlp_margin_all": margin.astype(np.float32),
            }

        def fake_build_evidence(data, train_idx, mlp_probs_np=None, log_file=None):
            nc = mlp_probs_np.shape[1]
            n = data.num_nodes
            return {
                "feature_similarity": np.random.dirichlet(np.ones(nc), size=n).astype(np.float32),
                "graph_neighbor_support": np.random.dirichlet(np.ones(nc), size=n).astype(np.float32),
                "structural_far_support": np.random.dirichlet(np.ones(nc), size=n).astype(np.float32),
                "compatibility_support": np.random.dirichlet(np.ones(nc), size=n).astype(np.float32),
                "node_degree": np.random.randint(1, 10, size=n).astype(np.float32),
                "support_train_neighbors": np.random.dirichlet(np.ones(nc), size=n).astype(np.float32),
                "neighbor_class_entropy": np.random.rand(n).astype(np.float32),
            }

        mod.train_mlp_and_predict = fake_train_mlp
        mod.compute_mlp_margin = fake_compute_margin
        mod._build_selective_correction_evidence = fake_build_evidence
        mod.DEFAULT_MANUSCRIPT_MLP_KWARGS = {}

        val_acc, test_acc, info = _ms.ms_hsgc(
            data, train_np, val_np, test_np,
            mlp_probs=mlp_probs, seed=42, mod=mod,
        )

        missing = expected_keys - set(info.keys())
        if missing:
            _fail(name, f"Missing keys: {missing}")
            return
        assert 0.0 <= val_acc <= 1.0, f"val_acc out of range: {val_acc}"
        assert 0.0 <= test_acc <= 1.0, f"test_acc out of range: {test_acc}"
        frac_sum = (info["frac_confident"] + info["frac_corrected_1hop"]
                    + info["frac_corrected_2hop"] + info["frac_mlp_only_uncertain"])
        assert abs(frac_sum - 1.0) < 1e-5, f"Routing fracs sum {frac_sum}"
        _ok(name)
    except Exception as e:
        _fail(name, traceback.format_exc())


def test_csv_columns_stable():
    """Test 8: run_ms_hsgc_evaluation MS_HSGC_RESULT_FIELDS match expected set."""
    name = "CSV column set is stable"
    expected_cols = {
        "dataset", "split_id", "method", "test_acc", "val_acc", "delta_vs_mlp",
        "frac_confident", "frac_corrected_1hop", "frac_corrected_2hop",
        "frac_mlp_only_uncertain",
        "mean_H1", "mean_H2", "mean_DeltaH",
        "n_helped", "n_hurt", "correction_precision",
        "selected_tau", "selected_rho1", "selected_rho2",
        "selected_h1_max", "selected_delta_min",
        "selected_profile_1hop", "selected_profile_2hop",
        "runtime_sec",
    }
    try:
        run_eval_path = os.path.join(CODE_DIR, "run_ms_hsgc_evaluation.py")
        # Read the field list from the module without executing it fully
        with open(run_eval_path, "r", encoding="utf-8") as fh:
            src = fh.read()
        for col in expected_cols:
            if f'"{col}"' not in src:
                _fail(name, f"Column {col!r} not found in run_ms_hsgc_evaluation.py")
                return
        _ok(name)
    except Exception as e:
        _fail(name, traceback.format_exc())


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 60)
    print("MS_HSGC Validation Tests")
    print("=" * 60)

    test_py_compile()
    ok = test_imports()
    if not ok:
        print("\n[ABORT] Cannot import ms_hsgc — skipping runtime tests.")
    else:
        test_multiscale_support_shapes()
        test_support_row_sums()
        test_heterophily_range()
        test_route_fractions_sum()
        test_info_dict_keys()
    test_csv_columns_stable()

    print()
    if _FAILURES:
        print(f"FAILED: {len(_FAILURES)} test(s): {', '.join(_FAILURES)}")
        return 1
    print(f"All {len(_PASSES)} tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

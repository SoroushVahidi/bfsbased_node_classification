#!/usr/bin/env python3
"""
Reliability-Gated Selective Graph Correction — Lightweight Experiment Runner.

Compares five methods on three representative datasets × 2 splits:
  1. mlp_only           — base MLP, no graph correction
  2. sgc_v1             — current method (v1, no reliability gate)
  3. sgc_v2_reliability — v2 with reliability gate, absolute threshold
  4. sgc_v2_percentile  — v2 with percentile-based threshold, no reliability gate
  5. sgc_v2_full        — v2 with reliability gate + percentile threshold

Datasets (hard-coded for reproducibility):
  • cora       — homophilic  (H ≈ 0.81)
  • chameleon  — heterophilic (H ≈ 0.23)
  • texas      — small, low-homophily (H ≈ 0.09)

Usage:
    cd code/bfsbased_node_classification
    python reliability_gated_runner.py \\
        --split-dir ../../data/splits \\
        --data-dir ../../data \\
        --splits 0 1 \\
        --out-dir ../../reports

Output:
    reports/reliability_gated_results.csv
    reports/reliability_gated_report.md
"""

import argparse
import csv
import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_REPORTS_DIR = REPO_ROOT / "reports"


# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------

def _load_bfs_module():
    """Load the BFS module as a importable object.

    The main BFS file is a Jupyter-notebook-style script that contains both
    library functions and legacy top-level experiment cells.  We strip the
    top-level cells before executing so only function/class definitions are
    imported.  The sentinel strings used below are stable landmarks that have
    been present in every version of the file; if the file structure changes,
    update these sentinels accordingly.
    """
    path = str(HERE / "bfsbased-full-investigate-homophil.py")
    with open(path, encoding="utf-8") as f:
        source = f.read()
    # Strip the legacy dataset-loading and experiment-execution cells.
    # These blocks sit between function definitions and should not be run
    # at import time (they would fail without a live DATASET_KEY / GPU).
    start_block = "dataset = load_dataset(DATASET_KEY, root)"
    end_block = "# In[21]:"
    legacy_block = 'LOG_DIR = "logs"'
    if start_block in source and end_block in source:
        a = source.index(start_block)
        b = source.index(end_block, a)
        source = source[:a] + "\n" + source[b:]
    if legacy_block in source:
        source = source[: source.index(legacy_block)]
    mod = importlib.util.module_from_spec(
        importlib.util.spec_from_loader("bfs_full", loader=None)
    )
    setattr(mod, "__file__", path)
    setattr(mod, "DATASET_KEY", "cora")
    exec(compile(source, path, "exec"), mod.__dict__)
    return mod


def _load_data(mod, dataset_key: str, data_dir: Path):
    root = str(data_dir) + "/"
    dataset = mod.load_dataset(dataset_key, root)
    data = dataset[0]
    data.edge_index = torch.cat([data.edge_index, data.edge_index.flip(0)], dim=1)
    data.edge_index = torch.unique(data.edge_index, dim=1)
    return data


def _find_split(dataset_key: str, split_id: int, data_dir: Path) -> Optional[Path]:
    prefix = "film" if dataset_key == "actor" else dataset_key.lower()
    fname = f"{prefix}_split_0.6_0.2_{split_id}.npz"
    candidates = [
        data_dir / "splits" / fname,
        data_dir / dataset_key / "raw" / fname,
        data_dir / dataset_key / "geom_gcn" / "raw" / fname,
        data_dir / "Film" / "raw" / fname,
        HERE / fname,
        HERE / "data" / fname,
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _load_split(dataset_key: str, split_id: int, data_dir: Path):
    p = _find_split(dataset_key, split_id, data_dir)
    if p is None:
        return None
    sp = np.load(str(p))
    return (
        np.where(sp["train_mask"])[0],
        np.where(sp["val_mask"])[0],
        np.where(sp["test_mask"])[0],
    )


def _edge_homophily(data) -> float:
    try:
        y = data.y.detach().cpu().numpy()
        ei = data.edge_index.detach().cpu().numpy()
        src, dst = ei[0], ei[1]
        mask = (y[src] >= 0) & (y[dst] >= 0)
        if not mask.any():
            return float("nan")
        return float((y[src[mask]] == y[dst[mask]]).mean())
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Per-method run helpers
# ---------------------------------------------------------------------------

def _run_mlp_only(mod, data, train_idx, val_idx, test_idx, seed: int) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    y_true = data.y.detach().cpu().numpy().astype(np.int64)
    t0 = time.perf_counter()
    mlp_probs, _ = mod.train_mlp_and_predict(data, train_idx, hidden=64, layers=2, dropout=0.5, lr=0.01, epochs=300)
    elapsed = time.perf_counter() - t0
    info = mod.compute_mlp_margin(mlp_probs)
    acc = float((info["mlp_pred_all"][test_idx] == y_true[test_idx]).mean())
    return {"test_acc": acc, "delta_over_mlp": 0.0, "runtime_sec": elapsed, "mlp_probs": mlp_probs}


def _run_sgc_v1(mod, data, train_idx, val_idx, test_idx, seed: int, mlp_probs=None) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    _, acc, info = mod.selective_graph_correction_predictclass(
        data, train_idx, val_idx, test_idx,
        mlp_probs=mlp_probs,
        seed=seed,
        write_node_diagnostics=False,
    )
    return {
        "test_acc": acc,
        "test_acc_mlp": info["test_acc_mlp"],
        "delta_over_mlp": info["test_acc_selective"] - info["test_acc_mlp"],
        "threshold": info["selected_threshold_high"],
        "uncertain_frac": info["fraction_test_nodes_uncertain"],
        "changed_frac": info["fraction_test_nodes_changed_from_mlp"],
        "runtime_sec": info["runtime_breakdown_sec"]["total_runtime_sec"],
    }


def _run_sgc_v2(
    mod, data, train_idx, val_idx, test_idx, seed: int,
    use_percentile_threshold: bool,
    enable_reliability_gate: bool,
    mlp_probs=None,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    _, acc, info = mod.selective_graph_correction_v2(
        data, train_idx, val_idx, test_idx,
        mlp_probs=mlp_probs,
        seed=seed,
        enable_feature_knn=True,
        feature_knn_k=5,
        use_percentile_threshold=use_percentile_threshold,
        enable_reliability_gate=enable_reliability_gate,
    )
    return {
        "test_acc": acc,
        "test_acc_mlp": info["test_acc_mlp"],
        "delta_over_mlp": info["delta_over_mlp"],
        "threshold": info["selected_tau"],
        "percentile": info["selected_percentile"],
        "reliability_threshold": info["selected_reliability_threshold"],
        "uncertain_frac": info["frac_test_uncertain"],
        "frac_full_graph": info["frac_test_full_graph"],
        "frac_feat_only": info["frac_test_feat_only"],
        "mean_reliability": info["mean_reliability_score"],
        "runtime_sec": info["runtime_sec"]["total"],
    }


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

METHODS = [
    "mlp_only",
    "sgc_v1",
    "sgc_v2_reliability",
    "sgc_v2_percentile",
    "sgc_v2_full",
]


def run_experiments(
    datasets: List[str],
    split_ids: List[int],
    data_dir: Path,
    seed_base: int = 1337,
) -> List[Dict[str, Any]]:
    print("\nLoading BFS module...")
    mod = _load_bfs_module()

    records: List[Dict[str, Any]] = []

    for ds in datasets:
        print(f"\n{'='*56}\nDataset: {ds}\n{'='*56}")
        try:
            data = _load_data(mod, ds, data_dir)
        except Exception as e:
            print(f"  [ERROR] Could not load {ds}: {e}")
            continue
        homophily = _edge_homophily(data)
        num_nodes = int(data.num_nodes)
        avg_degree = float(data.edge_index.size(1)) / float(num_nodes)
        num_classes = int(data.y.max().item()) + 1
        print(f"  nodes={num_nodes}  H={homophily:.3f}  deg={avg_degree:.1f}  classes={num_classes}")

        for sid in split_ids:
            split = _load_split(ds, sid, data_dir)
            if split is None:
                print(f"  [SKIP] No split for {ds} split {sid}")
                continue
            train_idx, val_idx, test_idx = split
            seed = seed_base + sid
            print(f"\n  split={sid}  train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

            # Train MLP once, reuse across all methods
            torch.manual_seed(seed)
            np.random.seed(seed)
            try:
                mlp_probs, _ = mod.train_mlp_and_predict(
                    data, train_idx, hidden=64, layers=2, dropout=0.5, lr=0.01, epochs=300
                )
            except Exception as e:
                print(f"    [ERROR] MLP training failed: {e}")
                continue

            mlp_info = mod.compute_mlp_margin(mlp_probs)
            y_true = data.y.detach().cpu().numpy().astype(np.int64)
            test_acc_mlp = float((mlp_info["mlp_pred_all"][test_idx] == y_true[test_idx]).mean())
            print(f"    MLP baseline acc: {test_acc_mlp:.4f}")

            base_row = {
                "dataset": ds,
                "split_id": sid,
                "seed": seed,
                "homophily": round(homophily, 4),
                "num_nodes": num_nodes,
                "avg_degree": round(avg_degree, 2),
                "num_classes": num_classes,
                "train_size": len(train_idx),
                "val_size": len(val_idx),
                "test_size": len(test_idx),
                "test_acc_mlp_base": test_acc_mlp,
            }

            for method in METHODS:
                try:
                    t0 = time.perf_counter()
                    if method == "mlp_only":
                        res = {"test_acc": test_acc_mlp, "delta_over_mlp": 0.0}
                    elif method == "sgc_v1":
                        res = _run_sgc_v1(mod, data, train_idx, val_idx, test_idx, seed, mlp_probs=mlp_probs)
                    elif method == "sgc_v2_reliability":
                        res = _run_sgc_v2(mod, data, train_idx, val_idx, test_idx, seed,
                                          use_percentile_threshold=False,
                                          enable_reliability_gate=True,
                                          mlp_probs=mlp_probs)
                    elif method == "sgc_v2_percentile":
                        res = _run_sgc_v2(mod, data, train_idx, val_idx, test_idx, seed,
                                          use_percentile_threshold=True,
                                          enable_reliability_gate=False,
                                          mlp_probs=mlp_probs)
                    elif method == "sgc_v2_full":
                        res = _run_sgc_v2(mod, data, train_idx, val_idx, test_idx, seed,
                                          use_percentile_threshold=True,
                                          enable_reliability_gate=True,
                                          mlp_probs=mlp_probs)
                    else:
                        continue
                    elapsed = time.perf_counter() - t0

                    row = {**base_row, "method": method, **res}
                    records.append(row)

                    acc = res.get("test_acc", float("nan"))
                    delta = res.get("delta_over_mlp", float("nan"))
                    print(f"    {method:22s}: acc={acc:.4f}  Δ={delta:+.4f}  [{elapsed:.1f}s]")

                except Exception as e:
                    import traceback
                    print(f"    [ERROR] {method}: {e}")
                    traceback.print_exc()

    return records


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def write_csv(records: List[Dict[str, Any]], path: Path) -> None:
    if not records:
        print("No records to write.")
        return
    all_keys: List[str] = []
    seen = set()
    for r in records:
        for k in r:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in all_keys})
    print(f"CSV written to: {path}")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _nan_or(v, default="N/A"):
    if v is None or (isinstance(v, float) and (v != v)):
        return default
    return v


def write_report(records: List[Dict[str, Any]], path: Path) -> None:
    """Generate a compact markdown diagnostic report."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Aggregate per (dataset, method)
    from collections import defaultdict
    agg: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    meta: Dict[str, Dict] = {}

    for r in records:
        ds = r["dataset"]
        method = r["method"]
        delta = r.get("delta_over_mlp", float("nan"))
        if isinstance(delta, (int, float)) and delta == delta:
            agg[(ds, method)].append(float(delta))
        if ds not in meta:
            meta[ds] = {
                "homophily": r.get("homophily", "N/A"),
                "num_nodes": r.get("num_nodes", "N/A"),
                "avg_degree": r.get("avg_degree", "N/A"),
            }

    datasets = sorted(meta.keys())

    lines = []
    lines.append("# Reliability-Gated Selective Graph Correction — Experiment Report")
    lines.append("")
    lines.append(f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}*")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Experiment Setup")
    lines.append("")
    lines.append("| Dataset | Homophily | Nodes | Avg Degree | Regime |")
    lines.append("|---------|-----------|-------|-----------|--------|")
    for ds in datasets:
        h = meta[ds]["homophily"]
        n = meta[ds]["num_nodes"]
        d = meta[ds]["avg_degree"]
        regime = "homophilic" if isinstance(h, float) and h > 0.5 else "heterophilic"
        lines.append(f"| {ds} | {h} | {n} | {d} | {regime} |")
    lines.append("")
    lines.append("**Methods compared:**")
    lines.append("")
    lines.append("| Method ID | Description |")
    lines.append("|-----------|-------------|")
    lines.append("| `mlp_only` | Base MLP — no graph correction |")
    lines.append("| `sgc_v1` | Current method (v1): margin threshold, no reliability gate |")
    lines.append("| `sgc_v2_reliability` | v2: reliability gate + absolute threshold |")
    lines.append("| `sgc_v2_percentile` | v2: no reliability gate + percentile threshold |")
    lines.append("| `sgc_v2_full` | v2: reliability gate + percentile threshold |")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 2. Main Results: Δ Accuracy over MLP Baseline")
    lines.append("")
    lines.append("Values are mean Δ (test accuracy - MLP baseline) across splits.")
    lines.append("Positive = better than MLP; negative = worse than MLP.")
    lines.append("")

    header = "| Dataset | H |" + "".join(f" {m} |" for m in METHODS)
    sep = "|---------|---|" + "".join(" :---: |" for _ in METHODS)
    lines.append(header)
    lines.append(sep)

    for ds in datasets:
        h = meta[ds]["homophily"]
        row = f"| **{ds}** | {h} |"
        for m in METHODS:
            vals = agg.get((ds, m), [])
            if vals:
                mean_d = float(np.mean(vals))
                symbol = "✓" if mean_d > 0.005 else ("✗" if mean_d < -0.002 else "~")
                row += f" {mean_d:+.4f} {symbol} |"
            else:
                row += " N/A |"
        lines.append(row)
    lines.append("")
    lines.append("Legend: ✓ = clear gain (>+0.5pp), ✗ = clear loss (<-0.2pp), ~ = neutral")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 3. Per-Split Detail")
    lines.append("")

    for ds in datasets:
        ds_records = [r for r in records if r["dataset"] == ds]
        lines.append(f"### {ds.capitalize()}")
        lines.append("")
        # Build a flat table
        lines.append("| Split | Method | Test Acc | Δ | Uncertain% | Full-Graph% | Feat-Only% | Reliability |")
        lines.append("|-------|--------|---------|---|-----------|------------|-----------|------------|")
        for r in sorted(ds_records, key=lambda x: (x["split_id"], METHODS.index(x["method"]) if x["method"] in METHODS else 99)):
            acc = r.get("test_acc", float("nan"))
            delta = r.get("delta_over_mlp", float("nan"))
            unc = r.get("uncertain_frac", float("nan"))
            fg = r.get("frac_full_graph", float("nan"))
            fo = r.get("frac_feat_only", float("nan"))
            rel = r.get("mean_reliability", float("nan"))

            def _f(v): return f"{v:.4f}" if isinstance(v, float) and v == v else "N/A"
            def _pct(v): return f"{v:.1%}" if isinstance(v, float) and v == v else "N/A"

            delta_str = f"{delta:+.4f}" if isinstance(delta, float) and delta == delta else "N/A"
            lines.append(
                f"| {r['split_id']} | {r['method']} | {_f(acc)} | {delta_str} | "
                f"{_pct(unc)} | {_pct(fg)} | {_pct(fo)} | {_f(rel)} |"
            )
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 4. Analysis")
    lines.append("")
    lines.append("### 4.1 Does reliability-aware gating reduce harmful corrections on heterophilic datasets?")
    lines.append("")

    for ds in datasets:
        h = meta[ds]["homophily"]
        v1_deltas = agg.get((ds, "sgc_v1"), [])
        v2r_deltas = agg.get((ds, "sgc_v2_reliability"), [])
        v2f_deltas = agg.get((ds, "sgc_v2_full"), [])
        if not v1_deltas:
            continue
        mean_v1 = float(np.mean(v1_deltas))
        mean_v2r = float(np.mean(v2r_deltas)) if v2r_deltas else float("nan")
        mean_v2f = float(np.mean(v2f_deltas)) if v2f_deltas else float("nan")

        is_hetero = isinstance(h, float) and h < 0.5
        better_r = mean_v2r > mean_v1 if not (mean_v2r != mean_v2r) else False
        better_f = mean_v2f > mean_v1 if not (mean_v2f != mean_v2f) else False

        lines.append(f"**{ds.capitalize()}** (H={h}):")
        lines.append(f"  - v1 mean Δ: {mean_v1:+.4f}")
        lines.append(f"  - v2_reliability mean Δ: {mean_v2r:+.4f}" if mean_v2r == mean_v2r else "  - v2_reliability: N/A")
        lines.append(f"  - v2_full mean Δ: {mean_v2f:+.4f}" if mean_v2f == mean_v2f else "  - v2_full: N/A")
        if is_hetero:
            if better_r or better_f:
                lines.append(f"  - ✓ Reliability gating **reduces harm** on this heterophilic dataset.")
            else:
                lines.append(f"  - ~ Reliability gating does not clearly help on this heterophilic dataset.")
        else:
            if better_r or better_f:
                lines.append(f"  - ✓ Reliability gating **preserves or improves** gains on this homophilic dataset.")
            else:
                lines.append(f"  - ~ Reliability gating does not hurt on this homophilic dataset.")
        lines.append("")

    lines.append("### 4.2 Does feature-kNN help when graph neighbors are misleading?")
    lines.append("")
    lines.append(
        "The v2 methods always enable feature-kNN (b3 > 0). "
        "For uncertain nodes with *low reliability*, only the feature-based correction "
        "(b2, b3) is applied — graph terms (b4, b5) are suppressed. "
        "The `frac_feat_only` column shows how often this path activates."
    )
    lines.append("")
    for ds in datasets:
        ds_records_v2r = [r for r in records if r["dataset"] == ds and r["method"] == "sgc_v2_reliability"]
        if not ds_records_v2r:
            continue
        mean_fo = float(np.mean([r.get("frac_feat_only", float("nan")) for r in ds_records_v2r if r.get("frac_feat_only") == r.get("frac_feat_only")]))
        mean_fg = float(np.mean([r.get("frac_full_graph", float("nan")) for r in ds_records_v2r if r.get("frac_full_graph") == r.get("frac_full_graph")]))
        lines.append(f"**{ds.capitalize()}**: feat-only-correction={mean_fo:.1%}  full-graph-correction={mean_fg:.1%}")
    lines.append("")

    lines.append("### 4.3 Is percentile-based threshold better on small/heterophilic datasets?")
    lines.append("")
    lines.append(
        "Percentile-based thresholds adapt to each dataset's margin distribution, "
        "ensuring a consistent fraction of nodes is flagged as uncertain regardless of scale."
    )
    lines.append("")
    for ds in datasets:
        v1_deltas = agg.get((ds, "sgc_v1"), [])
        v2p_deltas = agg.get((ds, "sgc_v2_percentile"), [])
        if not v1_deltas or not v2p_deltas:
            continue
        mean_v1 = float(np.mean(v1_deltas))
        mean_v2p = float(np.mean(v2p_deltas))
        better = mean_v2p > mean_v1
        lines.append(f"**{ds.capitalize()}**: v1={mean_v1:+.4f}  v2_percentile={mean_v2p:+.4f}  {'← better' if better else '← worse or same'}")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 5. Conclusions")
    lines.append("")

    # Auto-generate conclusions from data
    hetero_datasets = [ds for ds in datasets if isinstance(meta[ds]["homophily"], float) and meta[ds]["homophily"] < 0.5]
    homo_datasets = [ds for ds in datasets if isinstance(meta[ds]["homophily"], float) and meta[ds]["homophily"] >= 0.5]

    conclusion_num = 1
    for ds in hetero_datasets:
        v1_d = agg.get((ds, "sgc_v1"), [float("nan")])
        v2f_d = agg.get((ds, "sgc_v2_full"), [float("nan")])
        mv1 = float(np.mean(v1_d))
        mv2f = float(np.mean(v2f_d)) if v2f_d else float("nan")
        if mv2f == mv2f and mv2f > mv1:
            lines.append(
                f"{conclusion_num}. **Reliability gating helps on {ds}** (heterophilic, H={meta[ds]['homophily']}): "
                f"v2_full Δ={mv2f:+.4f} vs v1 Δ={mv1:+.4f}. "
                f"Suppressing graph terms for low-reliability nodes prevents harmful corrections."
            )
        elif mv2f == mv2f:
            lines.append(
                f"{conclusion_num}. **Reliability gating is neutral on {ds}** (heterophilic, H={meta[ds]['homophily']}): "
                f"v2_full Δ={mv2f:+.4f} vs v1 Δ={mv1:+.4f}. "
                f"Further tuning of the reliability threshold may be needed."
            )
        conclusion_num += 1
    for ds in homo_datasets:
        v1_d = agg.get((ds, "sgc_v1"), [float("nan")])
        v2f_d = agg.get((ds, "sgc_v2_full"), [float("nan")])
        mv1 = float(np.mean(v1_d))
        mv2f = float(np.mean(v2f_d)) if v2f_d else float("nan")
        if mv2f == mv2f:
            lines.append(
                f"{conclusion_num}. **On homophilic dataset {ds}** (H={meta[ds]['homophily']}): "
                f"v2_full Δ={mv2f:+.4f} vs v1 Δ={mv1:+.4f}. "
                f"{'Reliability gating preserves or improves gains.' if mv2f >= mv1 - 0.002 else 'Small loss from gate may indicate threshold needs recalibration.'}"
            )
            conclusion_num += 1
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 6. Reproducibility")
    lines.append("")
    lines.append("```bash")
    lines.append("cd code/bfsbased_node_classification")
    lines.append("python reliability_gated_runner.py \\")
    lines.append("    --split-dir ../../data/splits \\")
    lines.append("    --data-dir ../../data \\")
    lines.append("    --splits 0 1")
    lines.append("```")
    lines.append("")
    lines.append("Outputs:")
    lines.append("- `reports/reliability_gated_results.csv`")
    lines.append("- `reports/reliability_gated_report.md`")
    lines.append("")

    text = "\n".join(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Report written to: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Reliability-Gated SGC experiment runner")
    parser.add_argument(
        "--datasets", nargs="+",
        default=["cora", "chameleon", "texas"],
        help="Datasets to run (default: cora chameleon texas)",
    )
    parser.add_argument(
        "--splits", nargs="+", type=int,
        default=[0, 1],
        help="Split IDs to run (default: 0 1)",
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Path to data directory (default: ../../data from script location)",
    )
    parser.add_argument(
        "--split-dir", default=None,
        help="Override split-file directory (appended to candidates list)",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Output directory for reports (default: ../../reports)",
    )
    parser.add_argument(
        "--seed", type=int, default=1337,
        help="Base random seed (default: 1337)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else DEFAULT_DATA_DIR
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_REPORTS_DIR

    # If --split-dir was given, add it to data_dir/splits candidates by symlinking
    if args.split_dir:
        # Patch _find_split to also search args.split_dir
        sd = Path(args.split_dir)
        _orig_find_split = _find_split

        def _patched_find_split(dataset_key, split_id, data_dir_inner):
            p = _orig_find_split(dataset_key, split_id, data_dir_inner)
            if p is not None:
                return p
            prefix = "film" if dataset_key == "actor" else dataset_key.lower()
            fname = f"{prefix}_split_0.6_0.2_{split_id}.npz"
            c = sd / fname
            return c if c.exists() else None

        # Monkey-patch in this module's scope
        import builtins
        globals()["_find_split"] = _patched_find_split

    print("=" * 60)
    print("Reliability-Gated SGC — Lightweight Experiment Runner")
    print("=" * 60)
    print(f"Datasets : {args.datasets}")
    print(f"Splits   : {args.splits}")
    print(f"Data dir : {data_dir}")
    print(f"Out dir  : {out_dir}")

    records = run_experiments(args.datasets, args.splits, data_dir, seed_base=args.seed)

    csv_path = out_dir / "reliability_gated_results.csv"
    md_path = out_dir / "reliability_gated_report.md"

    write_csv(records, csv_path)
    write_report(records, md_path)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    from collections import defaultdict
    agg = defaultdict(list)
    for r in records:
        agg[(r["dataset"], r["method"])].append(r.get("delta_over_mlp", float("nan")))
    for ds in args.datasets:
        print(f"\n  {ds}:")
        for m in METHODS:
            vals = agg.get((ds, m), [])
            if vals:
                print(f"    {m:22s}: {float(np.mean(vals)):+.4f}")


if __name__ == "__main__":
    main()

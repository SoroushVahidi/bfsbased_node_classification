#!/usr/bin/env python3
"""
Margin-Bucket Safety Experiment.

Tests the core principle of FINAL_V3:
  "Does FINAL_V3 help mainly on low-confidence nodes while leaving
   high-confidence MLP predictions mostly undisturbed, and is it safer
   than a graph-wide baseline?"

Bucketing rule
--------------
  Test nodes are split into 3 equal-sized buckets by MLP margin
  (top probability − second probability) on the test split:
    low    : margin < Q33  (bottom third)
    medium : Q33 <= margin < Q67  (middle third)
    high   : margin >= Q67  (top third)
  Q33 / Q67 are the 33rd / 67th percentiles of the MLP test-node margins
  for that dataset-split combination.

Methods
-------
  - MLP        (feature-only MLP, fixed architecture)
  - FINAL_V3   (reliability-gated selective correction)
  - GCN        (graph-wide 2-layer GCN baseline)

Datasets / splits
-----------------
  Defaults: cora, chameleon, texas  ×  splits 0, 1, 2

Main hypotheses
---------------
  H1: On high-confidence nodes, FINAL_V3 should stay close to MLP.
  H2: On low-confidence nodes, FINAL_V3 should improve more vs MLP.
  H3: FINAL_V3 should be more conservative (fewer hurt nodes) than GCN,
      especially on heterophilous datasets (chameleon, texas).

Output files
------------
  logs/margin_bucket_safety_per_bucket.csv
  logs/margin_bucket_safety_per_split.csv
  reports/margin_bucket_safety_summary.md
  reports/margin_bucket_delta_by_bucket.png  (if matplotlib available)

Usage
-----
  python3 code/bfsbased_node_classification/run_margin_bucket_safety_experiment.py \
      --split-dir data/splits \
      --datasets cora chameleon texas \
      --splits 0 1 2
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup: allow sibling imports when run from repo root
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import manuscript_runner as mr  # noqa: E402 (after sys.path edit)
from standard_node_baselines import run_baseline  # noqa: E402
from final_method_v3 import final_method_v3  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_LOGS_DIR = os.path.join(_REPO_ROOT, "logs")
_REPORTS_DIR = os.path.join(_REPO_ROOT, "reports")

EXPERIMENT_NAME = "margin_bucket_safety"

# Default experiment configuration
DEFAULT_DATASETS = ["cora", "chameleon", "texas"]
DEFAULT_SPLITS = [0, 1, 2]


# ---------------------------------------------------------------------------
# Bucket helpers
# ---------------------------------------------------------------------------

BUCKET_NAMES = ["low", "medium", "high"]


def compute_buckets(margins: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Split test nodes into 3 equal-sized margin buckets.

    Parameters
    ----------
    margins : 1-D array of MLP margins for test nodes.

    Returns
    -------
    bucket_ids : integer array (0=low, 1=medium, 2=high) per test node.
    q33, q67   : the 33rd and 67th percentile cut-points.
    """
    q33 = float(np.percentile(margins, 100 / 3))
    q67 = float(np.percentile(margins, 200 / 3))
    bucket_ids = np.where(margins < q33, 0, np.where(margins < q67, 1, 2))
    return bucket_ids.astype(np.int64), q33, q67


def _bucket_accuracy_metrics(
    bucket_mask: np.ndarray,
    true_labels: np.ndarray,
    mlp_preds: np.ndarray,
    method_preds: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute bucket-level metrics comparing method_preds against mlp_preds.

    Parameters
    ----------
    bucket_mask   : boolean array over test nodes selecting this bucket.
    true_labels   : true class per test node.
    mlp_preds     : MLP predictions per test node.
    method_preds  : method predictions per test node.

    Returns
    -------
    Metrics dict (all counts/fractions are over the bucket).
    """
    bm = bucket_mask
    bucket_size = int(bm.sum())
    if bucket_size == 0:
        return {
            "bucket_size": 0,
            "bucket_accuracy": float("nan"),
            "mlp_accuracy": float("nan"),
            "delta_vs_mlp": float("nan"),
            "corrected_fraction": float("nan"),
            "changed_count": 0,
            "correction_precision": float("nan"),
            "hurt_rate": float("nan"),
            "n_helped": 0,
            "n_hurt": 0,
        }

    mlp_correct = mlp_preds[bm] == true_labels[bm]
    method_correct = method_preds[bm] == true_labels[bm]

    mlp_acc = float(mlp_correct.mean())
    method_acc = float(method_correct.mean())
    delta = method_acc - mlp_acc

    changed = method_preds[bm] != mlp_preds[bm]
    changed_count = int(changed.sum())
    corrected_fraction = float(changed.mean())

    if changed_count > 0:
        helped = changed & method_correct & ~mlp_correct
        hurt = changed & ~method_correct & mlp_correct
        n_helped = int(helped.sum())
        n_hurt = int(hurt.sum())
        # Among changed nodes, fraction that were net improvements (helped / (helped+hurt))
        correction_precision = n_helped / max(n_helped + n_hurt, 1)
        # Among changed nodes, fraction that hurt accuracy
        hurt_rate = n_hurt / changed_count
    else:
        n_helped = n_hurt = 0
        correction_precision = float("nan")
        hurt_rate = 0.0

    return {
        "bucket_size": bucket_size,
        "bucket_accuracy": method_acc,
        "mlp_accuracy": mlp_acc,
        "delta_vs_mlp": delta,
        "corrected_fraction": corrected_fraction,
        "changed_count": changed_count,
        "correction_precision": correction_precision,
        "hurt_rate": hurt_rate,
        "n_helped": n_helped,
        "n_hurt": n_hurt,
    }


# ---------------------------------------------------------------------------
# Per-split runner
# ---------------------------------------------------------------------------

def _run_one_split(
    mod,
    data,
    dataset_key: str,
    split_id: int,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
    seed: int,
) -> Dict[str, Any]:
    """
    Run MLP + FINAL_V3 + GCN on one dataset-split and return raw per-node results.
    """
    train_np = mod._to_numpy_idx(train_idx)
    val_np = mod._to_numpy_idx(val_idx)
    test_np = mod._to_numpy_idx(test_idx)

    # ------------------------------------------------------------------
    # 1. MLP training (shared for MLP and FINAL_V3)
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    mlp_cfg = getattr(mod, "DEFAULT_MANUSCRIPT_MLP_KWARGS")
    mlp_probs, _ = mod.train_mlp_and_predict(
        data,
        train_idx,
        hidden=mlp_cfg["hidden"],
        layers=mlp_cfg["layers"],
        dropout=mlp_cfg["dropout"],
        lr=mlp_cfg["lr"],
        epochs=mlp_cfg["epochs"],
        log_file=None,
    )
    mlp_time = time.perf_counter() - t0

    # Per-node MLP predictions for all nodes
    mlp_preds_all = mlp_probs.argmax(dim=1).cpu().numpy().astype(np.int64)
    true_labels_all = data.y.cpu().numpy().astype(np.int64)

    # Per-test-node MLP predictions
    mlp_preds_test = mlp_preds_all[test_np]
    true_labels_test = true_labels_all[test_np]

    mlp_acc_test = float((mlp_preds_test == true_labels_test).mean())

    # ------------------------------------------------------------------
    # 2. FINAL_V3
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    _val_acc_v3, _test_acc_v3, v3_info = final_method_v3(
        data,
        train_np,
        val_np,
        test_np,
        mlp_probs=mlp_probs,
        seed=seed,
        mod=mod,
        include_node_arrays=True,
    )
    v3_time = time.perf_counter() - t0

    na = v3_info["node_arrays"]
    # Sanity check: the test indices inside info should match ours
    assert np.array_equal(na["test_indices"], test_np), "test_indices mismatch"

    v3_preds_test = na["final_predictions"]
    mlp_margins_test = na["mlp_margins"]

    # ------------------------------------------------------------------
    # 3. GCN baseline
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    gcn_result = run_baseline(
        "gcn",
        data,
        train_idx,
        val_idx,
        test_idx,
        seed=seed,
        max_epochs=500,
        patience=100,
    )
    gcn_time = time.perf_counter() - t0

    # GCN probs are for all nodes
    gcn_preds_test = gcn_result.probs[test_idx].argmax(dim=1).numpy().astype(np.int64)

    return {
        "dataset": dataset_key,
        "split_id": split_id,
        "seed": seed,
        "test_np": test_np,
        "true_labels_test": true_labels_test,
        "mlp_preds_test": mlp_preds_test,
        "mlp_margins_test": mlp_margins_test,
        "v3_preds_test": v3_preds_test,
        "gcn_preds_test": gcn_preds_test,
        "mlp_acc_test": mlp_acc_test,
        "v3_acc_test": _test_acc_v3,
        "gcn_acc_test": gcn_result.test_acc,
        "v3_info": v3_info,
        "runtime": {
            "mlp_sec": mlp_time,
            "v3_sec": v3_time,
            "gcn_sec": gcn_time,
        },
    }


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

PER_BUCKET_FIELDS = [
    "experiment_name",
    "dataset",
    "split_id",
    "method",
    "bucket_name",
    "bucket_start",
    "bucket_end",
    "bucket_size",
    "bucket_accuracy",
    "delta_vs_mlp",
    "corrected_fraction",
    "changed_count",
    "correction_precision",
    "hurt_rate",
]

PER_SPLIT_FIELDS = [
    "dataset",
    "split_id",
    "mlp_acc",
    "final_v3_acc",
    "baseline_acc",
    "final_v3_harmful_count",
    "baseline_harmful_count",
    "notes",
]


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def write_per_bucket_csv(rows: List[Dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PER_BUCKET_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row.get(k)) for k in PER_BUCKET_FIELDS})


def write_per_split_csv(rows: List[Dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PER_SPLIT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row.get(k)) for k in PER_SPLIT_FIELDS})


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------

def _describe_bucket_obs(
    dataset: str,
    per_bucket_rows: List[Dict],
) -> str:
    """Generate a short per-dataset observation string."""
    lines = []
    by_bucket: Dict[str, Dict[str, Dict]] = {}  # method -> bucket -> metrics
    for row in per_bucket_rows:
        if row["dataset"] != dataset:
            continue
        m = row["method"]
        b = row["bucket_name"]
        if m not in by_bucket:
            by_bucket[m] = {}
        # Average across splits
        for k in ["bucket_accuracy", "delta_vs_mlp", "hurt_rate", "corrected_fraction"]:
            val = row.get(k)
            if isinstance(val, str):
                try:
                    val = float(val) if val else float("nan")
                except ValueError:
                    val = float("nan")
            if b not in by_bucket[m]:
                by_bucket[m][b] = {}
            if k not in by_bucket[m][b]:
                by_bucket[m][b][k] = []
            if not (isinstance(val, float) and np.isnan(val)):
                by_bucket[m][b][k].append(val)

    for bname in BUCKET_NAMES:
        v3 = by_bucket.get("FINAL_V3", {}).get(bname, {})
        gcn = by_bucket.get("GCN", {}).get(bname, {})
        v3_delta = np.mean(v3.get("delta_vs_mlp", [float("nan")]))
        gcn_delta = np.mean(gcn.get("delta_vs_mlp", [float("nan")]))
        v3_hurt = np.mean(v3.get("hurt_rate", [float("nan")]))
        gcn_hurt = np.mean(gcn.get("hurt_rate", [float("nan")]))
        v3_corr = np.mean(v3.get("corrected_fraction", [float("nan")]))
        lines.append(
            f"  {bname:6s}: FINAL_V3 Δ={v3_delta:+.3f}  GCN Δ={gcn_delta:+.3f}"
            f"  | V3 hurt_rate={v3_hurt:.3f}  V3 corr_frac={v3_corr:.3f}"
        )
    return "\n".join(lines)


def write_markdown_report(
    per_bucket_rows: List[Dict],
    per_split_rows: List[Dict],
    datasets: List[str],
    splits: List[int],
    path: str,
) -> None:
    """Write a concise markdown summary report."""

    # Compute per-dataset, per-bucket averages across splits
    def avg_field(rows, dataset, method, bucket, field):
        vals = []
        for r in rows:
            if r["dataset"] != dataset or r["method"] != method or r["bucket_name"] != bucket:
                continue
            v = r.get(field, "")
            if isinstance(v, str):
                try:
                    v = float(v) if v else float("nan")
                except ValueError:
                    v = float("nan")
            if not np.isnan(float(v if v is not None else float("nan"))):
                vals.append(float(v))
        return float(np.mean(vals)) if vals else float("nan")

    def avg_split_field(rows, dataset, field):
        vals = []
        for r in rows:
            if r["dataset"] != dataset:
                continue
            v = r.get(field, "")
            if isinstance(v, str):
                try:
                    v = float(v) if v else float("nan")
                except ValueError:
                    v = float("nan")
            if not np.isnan(float(v if v is not None else float("nan"))):
                vals.append(float(v))
        return float(np.mean(vals)) if vals else float("nan")

    lines = [
        "# Margin-Bucket Safety Experiment — Summary Report",
        "",
        "## Overview",
        "",
        "This experiment tests whether FINAL_V3 (reliability-gated selective graph",
        "correction) concentrates its gains on low-confidence nodes and remains more",
        "conservative than a graph-wide baseline (GCN).",
        "",
        "## Setup",
        "",
        f"- **Methods**: MLP, FINAL_V3, GCN",
        f"- **Datasets**: {', '.join(datasets)}",
        f"- **Splits**: {', '.join(str(s) for s in splits)}",
        "",
        "## Bucketing Rule",
        "",
        "Test nodes are divided into three equal-sized buckets based on the MLP margin",
        "(top predicted probability − second predicted probability) on each test split.",
        "",
        "| Bucket | Rule |",
        "|--------|------|",
        "| low    | margin < 33rd percentile of test-node margins |",
        "| medium | 33rd ≤ margin < 67th percentile |",
        "| high   | margin ≥ 67th percentile |",
        "",
        "Each bucket contains approximately 1/3 of test nodes.",
        "",
        "## Hypotheses",
        "",
        "| ID | Statement |",
        "|----|-----------|",
        "| H1 | On high-confidence nodes, FINAL_V3 stays close to MLP (small Δ). |",
        "| H2 | On low-confidence nodes, FINAL_V3 shows larger improvement vs MLP. |",
        "| H3 | FINAL_V3 has fewer hurt nodes than GCN, especially on heterophilous datasets. |",
        "",
    ]

    # Per-dataset tables
    for ds in datasets:
        lines += [
            f"## Dataset: {ds}",
            "",
            "### Overall accuracy (averaged over splits)",
            "",
        ]
        mlp_acc = avg_split_field(per_split_rows, ds, "mlp_acc")
        v3_acc = avg_split_field(per_split_rows, ds, "final_v3_acc")
        gcn_acc = avg_split_field(per_split_rows, ds, "baseline_acc")
        v3_harmful = avg_split_field(per_split_rows, ds, "final_v3_harmful_count")
        gcn_harmful = avg_split_field(per_split_rows, ds, "baseline_harmful_count")

        lines += [
            "| Method   | Test Acc | Harmful count |",
            "|----------|----------|---------------|",
            f"| MLP      | {mlp_acc:.3f}    | —             |",
            f"| FINAL_V3 | {v3_acc:.3f}    | {v3_harmful:.1f}          |",
            f"| GCN      | {gcn_acc:.3f}    | {gcn_harmful:.1f}          |",
            "",
            "### Per-bucket metrics (averaged over splits)",
            "",
            "| Bucket | MLP acc | V3 acc | V3 Δ | GCN Δ | V3 hurt_rate | V3 corr_frac |",
            "|--------|---------|--------|------|-------|-------------|-------------|",
        ]
        for bname in BUCKET_NAMES:
            mlp_b_acc = avg_field(per_bucket_rows, ds, "MLP", bname, "bucket_accuracy")
            v3_b_acc = avg_field(per_bucket_rows, ds, "FINAL_V3", bname, "bucket_accuracy")
            v3_b_delta = avg_field(per_bucket_rows, ds, "FINAL_V3", bname, "delta_vs_mlp")
            gcn_b_delta = avg_field(per_bucket_rows, ds, "GCN", bname, "delta_vs_mlp")
            v3_hurt = avg_field(per_bucket_rows, ds, "FINAL_V3", bname, "hurt_rate")
            v3_corr = avg_field(per_bucket_rows, ds, "FINAL_V3", bname, "corrected_fraction")

            def _s(v, signed=False):
                if np.isnan(v):
                    return "n/a"
                return (f"{v:+.3f}" if signed else f"{v:.3f}")

            lines.append(
                f"| {bname:6s} | {_s(mlp_b_acc)}   | {_s(v3_b_acc)}  | {_s(v3_b_delta, True)} "
                f"| {_s(gcn_b_delta, True)} | {_s(v3_hurt)}       | {_s(v3_corr)}       |"
            )
        lines.append("")

    # Hypothesis assessments
    lines += [
        "## Hypothesis Assessment",
        "",
        "The table below summarises the evidence from the averaged per-bucket metrics.",
        "Conclusions are necessarily tentative given the small number of splits (3 per dataset).",
        "",
        "| Hypothesis | Assessment |",
        "|------------|------------|",
    ]

    h1_evidence = []
    h2_evidence = []
    h3_evidence = []

    for ds in datasets:
        v3_high_delta = avg_field(per_bucket_rows, ds, "FINAL_V3", "high", "delta_vs_mlp")
        v3_low_delta = avg_field(per_bucket_rows, ds, "FINAL_V3", "low", "delta_vs_mlp")
        v3_hurt_all = avg_split_field(per_split_rows, ds, "final_v3_harmful_count")
        gcn_hurt_all = avg_split_field(per_split_rows, ds, "baseline_harmful_count")

        if not np.isnan(v3_high_delta):
            h1_evidence.append(abs(v3_high_delta) < 0.02)
        if not np.isnan(v3_low_delta) and not np.isnan(v3_high_delta):
            h2_evidence.append(v3_low_delta > v3_high_delta)
        if not np.isnan(v3_hurt_all) and not np.isnan(gcn_hurt_all):
            h3_evidence.append(v3_hurt_all <= gcn_hurt_all)

    def _verdict(evidence):
        if not evidence:
            return "insufficient data"
        frac = sum(evidence) / len(evidence)
        if frac >= 0.67:
            return "supported"
        if frac >= 0.34:
            return "mixed"
        return "not supported"

    lines += [
        f"| H1 | {_verdict(h1_evidence)} — high-confidence delta close to zero |",
        f"| H2 | {_verdict(h2_evidence)} — low-confidence delta larger than high |",
        f"| H3 | {_verdict(h3_evidence)} — FINAL_V3 hurt count ≤ GCN hurt count |",
        "",
        "## Caveats",
        "",
        "- Only 3 splits per dataset are used; results may be noisy.",
        "- Bucket boundaries vary per split (quantile-based, not fixed).",
        "- GCN is trained with full graph information from the outset; FINAL_V3 starts",
        "  from MLP and selectively uses graph evidence for uncertain nodes only.",
        "- Correction precision and hurt rate are undefined when no predictions are",
        "  changed (reported as n/a).",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "python3 code/bfsbased_node_classification/run_margin_bucket_safety_experiment.py \\",
        "    --split-dir data/splits \\",
        "    --datasets cora chameleon texas \\",
        "    --splits 0 1 2",
        "```",
        "",
        "*Generated automatically by run_margin_bucket_safety_experiment.py.*",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Optional plots
# ---------------------------------------------------------------------------

def _try_make_plots(
    per_bucket_rows: List[Dict],
    datasets: List[str],
    output_dir: str,
) -> None:
    """Create simple per-bucket delta plots. Skipped silently if matplotlib missing."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [INFO] matplotlib not available — skipping plots.")
        return

    buckets = BUCKET_NAMES
    x = np.arange(len(buckets))
    width = 0.35

    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    def _mean_across_splits(rows, dataset, method, bucket, field):
        vals = []
        for r in rows:
            if r["dataset"] != dataset or r["method"] != method or r["bucket_name"] != bucket:
                continue
            v = r.get(field, "")
            try:
                v = float(v) if v != "" else float("nan")
            except (ValueError, TypeError):
                v = float("nan")
            if not np.isnan(v):
                vals.append(v)
        return float(np.mean(vals)) if vals else 0.0

    for ax, ds in zip(axes, datasets):
        v3_deltas = [_mean_across_splits(per_bucket_rows, ds, "FINAL_V3", b, "delta_vs_mlp") for b in buckets]
        gcn_deltas = [_mean_across_splits(per_bucket_rows, ds, "GCN", b, "delta_vs_mlp") for b in buckets]

        bars1 = ax.bar(x - width / 2, v3_deltas, width, label="FINAL_V3", color="#2196F3", alpha=0.85)
        bars2 = ax.bar(x + width / 2, gcn_deltas, width, label="GCN", color="#FF9800", alpha=0.85)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(ds, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(buckets)
        ax.set_xlabel("Margin bucket")
        ax.set_ylabel("Δ accuracy vs MLP")
        ax.legend(fontsize=9)

        # Annotate bars with values
        for bar in bars1:
            h = bar.get_height()
            ax.annotate(f"{h:+.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3 if h >= 0 else -10), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7)
        for bar in bars2:
            h = bar.get_height()
            ax.annotate(f"{h:+.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3 if h >= 0 else -10), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7)

    fig.suptitle("Per-bucket Δ accuracy vs MLP (averaged over splits)", fontsize=12)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "margin_bucket_delta_by_bucket.png")
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {plot_path}")

    # Second plot: corrected_fraction per bucket for FINAL_V3
    fig2, axes2 = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), sharey=False)
    if len(datasets) == 1:
        axes2 = [axes2]

    for ax, ds in zip(axes2, datasets):
        v3_corr = [_mean_across_splits(per_bucket_rows, ds, "FINAL_V3", b, "corrected_fraction") for b in buckets]
        ax.bar(x, v3_corr, color="#4CAF50", alpha=0.85)
        ax.set_title(ds, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(buckets)
        ax.set_xlabel("Margin bucket")
        ax.set_ylabel("Corrected fraction (FINAL_V3)")
        ax.set_ylim(0, 1)
        for i, v in enumerate(v3_corr):
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)

    fig2.suptitle("FINAL_V3 corrected fraction per margin bucket", fontsize=12)
    plt.tight_layout()
    plot2_path = os.path.join(output_dir, "margin_bucket_corrected_fraction.png")
    plt.savefig(plot2_path, dpi=120, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved plot: {plot2_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_margin_bucket_experiment(
    datasets: Optional[List[str]] = None,
    split_ids: Optional[List[int]] = None,
    split_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the margin-bucket safety experiment.

    Parameters
    ----------
    datasets  : list of dataset keys (default: cora, chameleon, texas)
    split_ids : list of split IDs to run (default: [0, 1, 2])
    split_dir : directory containing split .npz files

    Returns
    -------
    paths dict with output file paths.
    """
    if datasets is None:
        datasets = DEFAULT_DATASETS[:]
    if split_ids is None:
        split_ids = DEFAULT_SPLITS[:]

    os.makedirs(_LOGS_DIR, exist_ok=True)
    os.makedirs(_REPORTS_DIR, exist_ok=True)

    per_bucket_path = os.path.join(_LOGS_DIR, "margin_bucket_safety_per_bucket.csv")
    per_split_path = os.path.join(_LOGS_DIR, "margin_bucket_safety_per_split.csv")
    report_path = os.path.join(_REPORTS_DIR, "margin_bucket_safety_summary.md")

    # ------------------------------------------------------------------
    # Load the BFS module once (needed for MLP + evidence)
    # ------------------------------------------------------------------
    print("Loading bfsbased-full-investigate module...")
    mod = mr._load_full_investigate_module()

    # ------------------------------------------------------------------
    # Check split coverage
    # ------------------------------------------------------------------
    coverage = mr.check_split_coverage(datasets, split_ids, split_dir)
    runnable = [ds for ds in datasets if coverage[ds]]
    if not runnable:
        raise SystemExit("No runnable datasets found. Check --split-dir.")
    print(f"Runnable datasets: {runnable}")

    per_bucket_rows: List[Dict] = []
    per_split_rows: List[Dict] = []

    t_wall = time.perf_counter()

    for dataset_key in runnable:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_key}")
        print(f"{'='*60}")

        dataset, data = mr._load_dataset_and_data(mod, dataset_key)
        device = data.x.device

        for split_id in coverage[dataset_key]:
            print(f"  split={split_id}")
            seed = (1337 + split_id) * 100

            torch.manual_seed(seed)
            np.random.seed(seed)

            try:
                train_idx, val_idx, test_idx, split_path = mr._load_split_npz(
                    dataset_key, split_id, device, split_dir
                )
            except FileNotFoundError as e:
                print(f"    [SKIP] {e}")
                continue

            try:
                result = _run_one_split(
                    mod, data, dataset_key, split_id,
                    train_idx, val_idx, test_idx, seed,
                )
            except Exception as e:
                import traceback
                print(f"    [ERROR] split={split_id}: {e}")
                traceback.print_exc()
                continue

            # --------------------------------------------------------------
            # Bucket analysis
            # --------------------------------------------------------------
            margins = result["mlp_margins_test"]
            true_labels = result["true_labels_test"]
            mlp_preds = result["mlp_preds_test"]
            v3_preds = result["v3_preds_test"]
            gcn_preds = result["gcn_preds_test"]

            bucket_ids, q33, q67 = compute_buckets(margins)

            bucket_ranges = {
                "low": (0.0, q33),
                "medium": (q33, q67),
                "high": (q67, 1.0),
            }

            methods_map = {
                "MLP": mlp_preds,
                "FINAL_V3": v3_preds,
                "GCN": gcn_preds,
            }

            for b_idx, bname in enumerate(BUCKET_NAMES):
                bm = bucket_ids == b_idx
                b_start, b_end = bucket_ranges[bname]

                for method_name, method_preds in methods_map.items():
                    m = _bucket_accuracy_metrics(bm, true_labels, mlp_preds, method_preds)
                    per_bucket_rows.append({
                        "experiment_name": EXPERIMENT_NAME,
                        "dataset": dataset_key,
                        "split_id": split_id,
                        "method": method_name,
                        "bucket_name": bname,
                        "bucket_start": b_start,
                        "bucket_end": b_end,
                        "bucket_size": m["bucket_size"],
                        "bucket_accuracy": m["bucket_accuracy"],
                        "delta_vs_mlp": m["delta_vs_mlp"],
                        "corrected_fraction": m["corrected_fraction"],
                        "changed_count": m["changed_count"],
                        "correction_precision": m["correction_precision"],
                        "hurt_rate": m["hurt_rate"],
                    })

            # --------------------------------------------------------------
            # Per-split summary
            # --------------------------------------------------------------
            # Harmful count: nodes where method is wrong but MLP was correct
            v3_harmful = int(((v3_preds != true_labels) & (mlp_preds == true_labels)).sum())
            gcn_harmful = int(((gcn_preds != true_labels) & (mlp_preds == true_labels)).sum())

            notes = (
                f"tau={result['v3_info']['selected_tau']:.3f} "
                f"rho={result['v3_info']['selected_rho']:.2f}"
            )

            per_split_rows.append({
                "dataset": dataset_key,
                "split_id": split_id,
                "mlp_acc": result["mlp_acc_test"],
                "final_v3_acc": result["v3_acc_test"],
                "baseline_acc": result["gcn_acc_test"],
                "final_v3_harmful_count": v3_harmful,
                "baseline_harmful_count": gcn_harmful,
                "notes": notes,
            })

            rt = result["runtime"]
            print(
                f"    MLP={result['mlp_acc_test']:.3f}  "
                f"V3={result['v3_acc_test']:.3f}  "
                f"GCN={result['gcn_acc_test']:.3f}  "
                f"[mlp:{rt['mlp_sec']:.1f}s v3:{rt['v3_sec']:.1f}s gcn:{rt['gcn_sec']:.1f}s]"
            )

    wall_time = time.perf_counter() - t_wall
    print(f"\nTotal wall time: {wall_time:.1f}s")

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    print(f"\nWriting {per_bucket_path}")
    write_per_bucket_csv(per_bucket_rows, per_bucket_path)

    print(f"Writing {per_split_path}")
    write_per_split_csv(per_split_rows, per_split_path)

    print(f"Writing {report_path}")
    write_markdown_report(per_bucket_rows, per_split_rows, runnable, split_ids, report_path)

    print("Generating plots...")
    _try_make_plots(per_bucket_rows, runnable, _REPORTS_DIR)

    print("\nDone.")
    return {
        "per_bucket_csv": per_bucket_path,
        "per_split_csv": per_split_path,
        "report_md": report_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Margin-Bucket Safety Experiment for FINAL_V3."
    )
    parser.add_argument(
        "--split-dir",
        default=None,
        help="Directory containing split .npz files (default: auto-detected).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=f"Datasets to run (default: {DEFAULT_DATASETS}).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        type=int,
        default=None,
        help=f"Split IDs to run (default: {DEFAULT_SPLITS}).",
    )
    args = parser.parse_args()

    out = run_margin_bucket_experiment(
        datasets=args.datasets,
        split_ids=args.splits,
        split_dir=args.split_dir,
    )
    print("\nOutput files:")
    for k, v in out.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

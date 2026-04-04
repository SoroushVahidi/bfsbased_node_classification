#!/usr/bin/env python3
"""
Reconnaissance behavior experiment: lightweight multi-dataset sweep for
FINAL_V3 and MS_HSGC.

Status: NON-CANONICAL diagnostic runner. Does NOT write to canonical paths.

Usage:
  python3 code/bfsbased_node_classification/run_recon_behavior.py \\
    --light \\
    --datasets chameleon squirrel texas wisconsin cornell citeseer cora pubmed \\
    --splits 0 \\
    --split-dir data/splits \\
    --output-tag recon_sweep

Outputs (all non-canonical):
  reports/recon_behavior_results.csv
  reports/recon_behavior_binned.csv
  reports/recon_ms_hsgc_vs_final_v3.csv
  reports/recon_behavior_summary.md
  tables/recon_behavior_comparison.md
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from final_method_v3 import final_method_v3, _simple_accuracy  # noqa: E402
from ms_hsgc import ms_hsgc  # noqa: E402


# ---------------------------------------------------------------------------
# Light-mode settings
# ---------------------------------------------------------------------------

LIGHT_MLP_KWARGS = {
    "hidden": 64, "layers": 2, "dropout": 0.5, "lr": 0.01, "epochs": 100,
}

# Light MS_HSGC grid: 3 tau × 1 rho1 × 1 rho2 × 1 h1_max × 1 delta_min × 1 × 1 = 3 configs
LIGHT_GRID = {
    "tau":       [0.05, 0.10, 0.20],
    "rho1":      [0.4],
    "rho2":      [0.4],
    "h1_max":    [0.5],
    "delta_min": [0.0],
    "pi1":       [0],
    "pi2":       [0],
}

# Binning helpers
MARGIN_BINS = [
    ("very_low",  0.00, 0.10),
    ("low",       0.10, 0.25),
    ("medium",    0.25, 0.50),
    ("high",      0.50, 2.00),
]

DEGREE_BINS = [
    ("0-2",  0,  2),
    ("3-5",  3,  5),
    ("6-10", 6, 10),
    (">10", 11, 10**9),
]


def _margin_bin_label(m: float) -> str:
    for label, lo, hi in MARGIN_BINS:
        if lo <= m < hi:
            return label
    return "high"


def _degree_bin_label(d: int) -> str:
    for label, lo, hi in DEGREE_BINS:
        if lo <= d <= hi:
            return label
    return ">10"


# ---------------------------------------------------------------------------
# Module and data loading (mirrors run_ms_hsgc_evaluation.py)
# ---------------------------------------------------------------------------

def _load_module():
    path = os.path.join(HERE, "bfsbased-full-investigate-homophil.py")
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


def _load_data(mod, ds: str, root: str = "data/"):
    dataset = mod.load_dataset(ds, root)
    data = dataset[0]
    data.edge_index = torch.cat([data.edge_index, data.edge_index.flip(0)], dim=1)
    data.edge_index = torch.unique(data.edge_index, dim=1)
    return data


def _load_split(ds: str, sid: int, device, split_dir: str):
    from split_paths import is_official_single_split_dataset, split_npz_prefix

    if is_official_single_split_dataset(ds) and int(sid) != 0:
        raise FileNotFoundError(
            f"Dataset {ds!r} uses the official split only (split_id=0 required, got {sid})."
        )
    prefix = split_npz_prefix(ds)
    fname = f"{prefix}_split_0.6_0.2_{sid}.npz"
    path = os.path.join(split_dir, fname)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Split not found: {path}")
    sp = np.load(path)

    def to_t(mask):
        return torch.as_tensor(np.where(mask)[0], dtype=torch.long, device=device)

    return to_t(sp["train_mask"]), to_t(sp["val_mask"]), to_t(sp["test_mask"])


# ---------------------------------------------------------------------------
# Per-node binning helpers
# ---------------------------------------------------------------------------

def _build_degree_array(data) -> np.ndarray:
    N = data.num_nodes
    edge_np = data.edge_index.detach().cpu().numpy()
    return np.bincount(edge_np[0].astype(np.int64), minlength=N).astype(np.int64)


def _compute_binned_rows(
    ds: str,
    sid: int,
    method: str,
    margins: np.ndarray,
    degrees: np.ndarray,
    helped: np.ndarray,
    hurt: np.ndarray,
    corrected: np.ndarray,
    bin_type: str,   # "margin" or "degree"
    extra: Optional[Dict[str, np.ndarray]] = None,
) -> List[Dict]:
    """Return one row per bin for the given method and bin_type."""
    rows = []
    bins = MARGIN_BINS if bin_type == "margin" else DEGREE_BINS

    for bin_label, lo, hi in bins:
        if bin_type == "margin":
            mask = (margins >= lo) & (margins < hi)
        else:
            mask = (degrees >= lo) & (degrees <= hi)

        n_bin = int(mask.sum())
        if n_bin == 0:
            continue

        n_corr = int(corrected[mask].sum())
        n_help = int(helped[mask].sum())
        n_hurt_bin = int(hurt[mask].sum())
        net = n_help - n_hurt_bin
        frac_corr = n_corr / n_bin if n_bin > 0 else 0.0

        row: Dict[str, Any] = {
            "dataset": ds,
            "split_id": sid,
            "method": method,
            "bin_type": bin_type,
            "bin_label": bin_label,
            "n_nodes": n_bin,
            "frac_corrected": round(frac_corr, 4),
            "n_helped": n_help,
            "n_hurt": n_hurt_bin,
            "net_help": net,
        }

        # Heterophily profile bins for MS_HSGC
        if extra and bin_type == "margin":
            for k in ("H1", "H2", "DeltaH", "routed_1hop", "routed_2hop"):
                if k in extra:
                    row[k] = round(float(extra[k][mask].mean()), 4) if n_bin > 0 else ""
        rows.append(row)

    return rows


def _compute_hetero_bins(
    ds: str,
    sid: int,
    method: str,
    H1: np.ndarray,
    H2: np.ndarray,
    DeltaH: np.ndarray,
    route: np.ndarray,   # MS_HSGC route array (0/1/2/3)
    helped: np.ndarray,
    hurt: np.ndarray,
) -> List[Dict]:
    """Quantile bins over H1, H2, DeltaH for MS_HSGC routing diagnostics."""
    rows = []
    for feat_name, arr in [("H1", H1), ("H2", H2), ("DeltaH", DeltaH)]:
        q25, q75 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
        q_median = float(np.percentile(arr, 50))
        hetero_bins = [
            ("Q1_low",  -99.0, q25),
            ("Q2_mid",  q25,   q_median),
            ("Q3_mid",  q_median, q75),
            ("Q4_high", q75,  99.0),
        ]
        for bin_label, lo, hi in hetero_bins:
            mask = (arr >= lo) & (arr < hi) if lo > -99 else (arr < hi)
            if bin_label == "Q4_high":
                mask = arr >= q75
            n_bin = int(mask.sum())
            if n_bin == 0:
                continue
            rows.append({
                "dataset": ds,
                "split_id": sid,
                "method": method,
                "bin_type": f"hetero_{feat_name}",
                "bin_label": bin_label,
                "n_nodes": n_bin,
                "frac_corrected": round(float(((route[mask] == 1) | (route[mask] == 2)).mean()), 4),
                "frac_routed_1hop": round(float((route[mask] == 1).mean()), 4),
                "frac_routed_2hop": round(float((route[mask] == 2).mean()), 4),
                "n_helped": int(helped[mask].sum()),
                "n_hurt": int(hurt[mask].sum()),
                "net_help": int(helped[mask].sum()) - int(hurt[mask].sum()),
            })
    return rows


# ---------------------------------------------------------------------------
# Result record fields
# ---------------------------------------------------------------------------

RESULT_FIELDS = [
    "dataset", "split_id", "method",
    "test_acc", "val_acc", "delta_vs_mlp", "runtime_sec",
    # FINAL_V3 / shared
    "frac_confident", "frac_unreliable", "frac_corrected",
    "tau", "rho", "n_helped", "n_hurt", "net_help", "correction_precision",
    # MS_HSGC specific
    "frac_corrected_1hop", "frac_corrected_2hop",
    "frac_mlp_only_uncertain", "mean_H1", "mean_H2", "mean_DeltaH",
    "selected_tau", "selected_rho1", "selected_rho2",
    "selected_h1_max", "selected_delta_min",
    # routing blocks
    "uncertain_total", "blocked_by_h1", "blocked_by_r1",
    "blocked_by_delta", "blocked_by_r2", "routed_1hop", "routed_2hop",
]

BINNED_FIELDS = [
    "dataset", "split_id", "method", "bin_type", "bin_label",
    "n_nodes", "frac_corrected", "n_helped", "n_hurt", "net_help",
    "frac_routed_1hop", "frac_routed_2hop",
    "H1", "H2", "DeltaH",
]

DELTA_FIELDS = [
    "dataset", "split_id",
    "mlp_test_acc", "final_v3_test_acc", "ms_hsgc_test_acc",
    "delta_final_v3_vs_mlp", "delta_ms_hsgc_vs_mlp",
    "delta_ms_hsgc_vs_final_v3",
]


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_recon_sweep(
    datasets: List[str],
    splits: List[int],
    split_dir: str,
    output_tag: str,
    light: bool = True,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Run MLP_ONLY + FINAL_V3 + MS_HSGC for all datasets/splits.

    Returns (result_records, binned_records, delta_records).
    """
    print("\n" + "=" * 70)
    print("RECONNAISSANCE BEHAVIOR EXPERIMENT")
    print("Status: NON-CANONICAL diagnostic run")
    if light:
        print(f"Mode: LIGHT  (MLP epochs={LIGHT_MLP_KWARGS['epochs']}, "
              f"MS_HSGC grid={len(LIGHT_GRID['tau'])} tau configs)")
    print(f"Datasets: {datasets}")
    print(f"Splits:   {splits}")
    print("=" * 70 + "\n")

    mlp_kwargs = LIGHT_MLP_KWARGS if light else None
    mod = _load_module()

    result_rows: List[Dict] = []
    binned_rows: List[Dict] = []
    delta_rows: List[Dict] = []

    for ds in datasets:
        print(f"\n{'='*60}\nDataset: {ds}\n{'='*60}")
        try:
            data = _load_data(mod, ds)
        except Exception as exc:
            print(f"  [SKIP loading {ds}] {exc}")
            continue

        deg_all = _build_degree_array(data)

        for sid in splits:
            print(f"\n  split={sid}")
            try:
                train_idx, val_idx, test_idx = _load_split(
                    ds, sid, data.x.device, split_dir
                )
            except FileNotFoundError as exc:
                print(f"  [SKIP split] {exc}")
                continue

            train_np = mod._to_numpy_idx(train_idx)
            val_np = mod._to_numpy_idx(val_idx)
            test_np = mod._to_numpy_idx(test_idx)
            seed = (1337 + sid) * 100
            torch.manual_seed(seed)
            np.random.seed(seed)

            y_true = data.y.detach().cpu().numpy().astype(np.int64)
            deg_test = deg_all[test_np]

            # ---- MLP ----
            effective_mlp_kwargs = mlp_kwargs if mlp_kwargs else mod.DEFAULT_MANUSCRIPT_MLP_KWARGS
            t0 = time.perf_counter()
            mlp_probs, _ = mod.train_mlp_and_predict(
                data, train_idx,
                **effective_mlp_kwargs, log_file=None,
            )
            mlp_rt = time.perf_counter() - t0
            mlp_info = mod.compute_mlp_margin(mlp_probs)
            mlp_pred = mlp_info["mlp_pred_all"]
            mlp_test_acc = _simple_accuracy(y_true[test_np], mlp_pred[test_np])

            print(f"    MLP: {mlp_test_acc:.4f}  ({mlp_rt:.1f}s)")
            result_rows.append(_make_row(ds, sid, "MLP_ONLY", mlp_test_acc, None, 0.0, mlp_rt))

            # ---- FINAL_V3 ----
            t0 = time.perf_counter()
            v3_val, v3_acc, v3_info = final_method_v3(
                data, train_np, val_np, test_np,
                mlp_probs=mlp_probs, seed=seed, mod=mod,
                gate="heuristic", split_id=sid,
                include_node_arrays=True,
            )
            v3_rt = time.perf_counter() - t0
            v3_delta = v3_acc - mlp_test_acc

            bf = v3_info["branch_fractions"]
            ca = v3_info["correction_analysis"]
            na_v3 = v3_info.get("node_arrays", {})

            v3_row = _make_row(ds, sid, "FINAL_V3", v3_acc, v3_val, v3_delta, v3_rt)
            v3_row.update({
                "frac_confident": bf.get("confident_keep_mlp"),
                "frac_unreliable": bf.get("uncertain_unreliable_keep_mlp"),
                "frac_corrected": bf.get("uncertain_reliable_corrected"),
                "tau": v3_info.get("selected_tau"),
                "rho": v3_info.get("selected_rho"),
                "n_helped": ca.get("n_helped"),
                "n_hurt": ca.get("n_hurt"),
                "net_help": (ca.get("n_helped") or 0) - (ca.get("n_hurt") or 0),
                "correction_precision": ca.get("correction_precision"),
            })
            result_rows.append(v3_row)
            print(f"    FINAL_V3: {v3_acc:.4f}  (delta={v3_delta:+.4f}  "
                  f"corrected={bf.get('uncertain_reliable_corrected', 0):.2f}  "
                  f"helped={ca.get('n_helped')} hurt={ca.get('n_hurt')})  "
                  f"[{v3_rt:.1f}s]")

            # Binned diagnostics for FINAL_V3
            if na_v3:
                v3_margins = na_v3["mlp_margins"].astype(np.float32)
                v3_helped = na_v3["is_changed_helpful"].astype(bool)
                v3_hurt = na_v3["is_changed_harmful"].astype(bool)
                v3_corrected = na_v3["is_corrected_branch"].astype(bool)

                binned_rows.extend(_compute_binned_rows(
                    ds, sid, "FINAL_V3", v3_margins, deg_test,
                    v3_helped, v3_hurt, v3_corrected, "margin",
                ))
                binned_rows.extend(_compute_binned_rows(
                    ds, sid, "FINAL_V3", v3_margins, deg_test,
                    v3_helped, v3_hurt, v3_corrected, "degree",
                ))

            # ---- MS_HSGC ----
            t0 = time.perf_counter()
            ms_val, ms_acc, ms_info = ms_hsgc(
                data, train_np, val_np, test_np,
                mlp_probs=mlp_probs, seed=seed, mod=mod,
                light_grid_override=LIGHT_GRID if light else None,
                include_node_arrays=True,
            )
            ms_rt = time.perf_counter() - t0
            ms_delta = ms_acc - mlp_test_acc

            rb = ms_info.get("routing_blocks", {})
            ms_row = _make_row(ds, sid, "MS_HSGC", ms_acc, ms_val, ms_delta, ms_rt)
            ms_row.update({
                "frac_confident": ms_info["frac_confident"],
                "frac_corrected_1hop": ms_info["frac_corrected_1hop"],
                "frac_corrected_2hop": ms_info["frac_corrected_2hop"],
                "frac_mlp_only_uncertain": ms_info["frac_mlp_only_uncertain"],
                "mean_H1": ms_info["mean_H1"],
                "mean_H2": ms_info["mean_H2"],
                "mean_DeltaH": ms_info["mean_DeltaH"],
                "n_helped": ms_info["n_helped"],
                "n_hurt": ms_info["n_hurt"],
                "net_help": ms_info["n_helped"] - ms_info["n_hurt"],
                "correction_precision": ms_info["correction_precision"],
                "selected_tau": ms_info["selected_tau"],
                "selected_rho1": ms_info["selected_rho1"],
                "selected_rho2": ms_info["selected_rho2"],
                "selected_h1_max": ms_info["selected_h1_max"],
                "selected_delta_min": ms_info["selected_delta_min"],
                "uncertain_total": rb.get("uncertain_total"),
                "blocked_by_h1": rb.get("blocked_by_h1"),
                "blocked_by_r1": rb.get("blocked_by_r1"),
                "blocked_by_delta": rb.get("blocked_by_delta"),
                "blocked_by_r2": rb.get("blocked_by_r2"),
                "routed_1hop": rb.get("routed_1hop"),
                "routed_2hop": rb.get("routed_2hop"),
            })
            result_rows.append(ms_row)
            print(f"    MS_HSGC:  {ms_acc:.4f}  (delta={ms_delta:+.4f}  "
                  f"1h={ms_info['frac_corrected_1hop']:.2f} "
                  f"2h={ms_info['frac_corrected_2hop']:.2f}  "
                  f"helped={ms_info['n_helped']} hurt={ms_info['n_hurt']})  "
                  f"[{ms_rt:.1f}s]")

            # Binned diagnostics for MS_HSGC
            na_ms = ms_info.get("node_arrays", {})
            if na_ms:
                ms_margins = na_ms["mlp_margins"].astype(np.float32)
                ms_helped = na_ms["is_changed_helpful"].astype(bool)
                ms_hurt = na_ms["is_changed_harmful"].astype(bool)
                ms_route = na_ms["route"].astype(np.int8)
                ms_corrected = (ms_route == 1) | (ms_route == 2)
                H1_test = na_ms["H1"]
                H2_test = na_ms["H2"]
                DeltaH_test = na_ms["DeltaH"]

                extra_ms = {
                    "H1": H1_test, "H2": H2_test, "DeltaH": DeltaH_test,
                    "routed_1hop": (ms_route == 1).astype(np.float32),
                    "routed_2hop": (ms_route == 2).astype(np.float32),
                }
                binned_rows.extend(_compute_binned_rows(
                    ds, sid, "MS_HSGC", ms_margins, deg_test,
                    ms_helped, ms_hurt, ms_corrected, "margin", extra_ms,
                ))
                binned_rows.extend(_compute_binned_rows(
                    ds, sid, "MS_HSGC", ms_margins, deg_test,
                    ms_helped, ms_hurt, ms_corrected, "degree",
                ))
                # Heterophily profile bins
                binned_rows.extend(_compute_hetero_bins(
                    ds, sid, "MS_HSGC",
                    H1_test, H2_test, DeltaH_test, ms_route,
                    ms_helped, ms_hurt,
                ))

            # ---- Delta record ----
            delta_rows.append({
                "dataset": ds,
                "split_id": sid,
                "mlp_test_acc": round(mlp_test_acc, 4),
                "final_v3_test_acc": round(v3_acc, 4),
                "ms_hsgc_test_acc": round(ms_acc, 4),
                "delta_final_v3_vs_mlp": round(v3_delta, 4),
                "delta_ms_hsgc_vs_mlp": round(ms_delta, 4),
                "delta_ms_hsgc_vs_final_v3": round(ms_acc - v3_acc, 4),
            })

    return result_rows, binned_rows, delta_rows


def _make_row(ds, sid, method, test_acc, val_acc, delta, runtime):
    return {
        "dataset": ds, "split_id": sid, "method": method,
        "test_acc": round(test_acc, 4) if test_acc is not None else "",
        "val_acc": round(val_acc, 4) if val_acc is not None else "",
        "delta_vs_mlp": round(delta, 4) if delta is not None else "",
        "runtime_sec": round(runtime, 2) if runtime is not None else "",
        # defaults (overwritten as needed)
        "frac_confident": "", "frac_unreliable": "", "frac_corrected": "",
        "tau": "", "rho": "", "n_helped": "", "n_hurt": "", "net_help": "",
        "correction_precision": "",
        "frac_corrected_1hop": "", "frac_corrected_2hop": "",
        "frac_mlp_only_uncertain": "", "mean_H1": "", "mean_H2": "", "mean_DeltaH": "",
        "selected_tau": "", "selected_rho1": "", "selected_rho2": "",
        "selected_h1_max": "", "selected_delta_min": "",
        "uncertain_total": "", "blocked_by_h1": "", "blocked_by_r1": "",
        "blocked_by_delta": "", "blocked_by_r2": "", "routed_1hop": "", "routed_2hop": "",
    }


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def _write_csv(path: str, rows: List[Dict], fields: List[str]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"  Written: {path}")


# ---------------------------------------------------------------------------
# Markdown summary generator
# ---------------------------------------------------------------------------

def _fmt(v, fmt=".4f"):
    if v is None or v == "":
        return "N/A"
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return str(v)


def _build_summary(
    result_rows: List[Dict],
    binned_rows: List[Dict],
    delta_rows: List[Dict],
    light: bool,
    datasets: List[str],
    splits: List[int],
) -> str:
    """Construct the full markdown summary."""
    now_str = time.strftime("%Y-%m-%d")
    mode_str = "LIGHT (reconnaissance)" if light else "FULL"

    lines = []
    lines.append("# Reconnaissance Behavior Experiment — Summary")
    lines.append("")
    lines.append("**Status:** NON-CANONICAL diagnostic run — does not overwrite canonical FINAL_V3 outputs.")
    lines.append(f"**Date:** {now_str}")
    lines.append(f"**Mode:** {mode_str}")
    lines.append(f"**Datasets:** {', '.join(datasets)}")
    lines.append(f"**Splits:** {splits}")
    if light:
        lines.append(f"**Light settings:** MLP epochs=100, MS_HSGC grid={len(LIGHT_GRID['tau'])} tau configs")
    lines.append("")

    # --- Per-dataset accuracy table ---
    lines.append("## 1. Per-Dataset Test Accuracy")
    lines.append("")

    by_ds_method: Dict = defaultdict(lambda: defaultdict(list))
    for r in result_rows:
        ta = r.get("test_acc")
        if ta not in (None, ""):
            by_ds_method[r["dataset"]][r["method"]].append(float(ta))

    methods = ["MLP_ONLY", "FINAL_V3", "MS_HSGC"]
    header = "| Dataset | " + " | ".join(methods) + " | V3-MLP | MS-MLP | MS-V3 |"
    sep = "|" + "---|" * (len(methods) + 4)
    lines.append(header)
    lines.append(sep)

    for ds in datasets:
        row_vals = {}
        for m in methods:
            accs = by_ds_method[ds][m]
            row_vals[m] = float(np.mean(accs)) if accs else None

        mlp_v = row_vals.get("MLP_ONLY")
        v3_v = row_vals.get("FINAL_V3")
        ms_v = row_vals.get("MS_HSGC")

        def _d(a, b):
            if a is None or b is None:
                return "N/A"
            return f"{a - b:+.4f}"

        lines.append(
            f"| {ds} | "
            + " | ".join(_fmt(row_vals[m]) for m in methods)
            + f" | {_d(v3_v, mlp_v)} | {_d(ms_v, mlp_v)} | {_d(ms_v, v3_v)} |"
        )
    lines.append("")

    # --- FINAL_V3 diagnostics ---
    lines.append("## 2. FINAL_V3 Diagnostics")
    lines.append("")
    lines.append("| Dataset | frac_confident | frac_unreliable | frac_corrected | "
                 "tau | rho | n_helped | n_hurt | net_help |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    for ds in datasets:
        rows_ds = [r for r in result_rows if r["dataset"] == ds and r["method"] == "FINAL_V3"]
        if not rows_ds:
            continue
        r = rows_ds[0]
        lines.append(
            f"| {ds} | {_fmt(r.get('frac_confident'))} | {_fmt(r.get('frac_unreliable'))} "
            f"| {_fmt(r.get('frac_corrected'))} | {_fmt(r.get('tau'))} "
            f"| {_fmt(r.get('rho'))} | {r.get('n_helped', 'N/A')} "
            f"| {r.get('n_hurt', 'N/A')} | {r.get('net_help', 'N/A')} |"
        )
    lines.append("")

    # --- MS_HSGC routing diagnostics ---
    lines.append("## 3. MS_HSGC Routing Diagnostics")
    lines.append("")
    lines.append("| Dataset | frac_conf | frac_1hop | frac_2hop | frac_mlp_unc | "
                 "mean_H1 | mean_H2 | mean_DeltaH | helped | hurt |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")

    for ds in datasets:
        rows_ds = [r for r in result_rows if r["dataset"] == ds and r["method"] == "MS_HSGC"]
        if not rows_ds:
            continue
        r = rows_ds[0]
        lines.append(
            f"| {ds} | {_fmt(r.get('frac_confident'))} "
            f"| {_fmt(r.get('frac_corrected_1hop'))} "
            f"| {_fmt(r.get('frac_corrected_2hop'))} "
            f"| {_fmt(r.get('frac_mlp_only_uncertain'))} "
            f"| {_fmt(r.get('mean_H1'))} "
            f"| {_fmt(r.get('mean_H2'))} "
            f"| {_fmt(r.get('mean_DeltaH'))} "
            f"| {r.get('n_helped', 'N/A')} "
            f"| {r.get('n_hurt', 'N/A')} |"
        )
    lines.append("")

    # --- MS_HSGC routing block diagnostics ---
    lines.append("## 4. MS_HSGC Routing Block Diagnostics")
    lines.append("")
    lines.append("| Dataset | uncertain_total | routed_1hop | routed_2hop | "
                 "fallback_mlp | blocked_by_h1 | blocked_by_r1 | blocked_by_r2 | blocked_by_delta |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    for ds in datasets:
        rows_ds = [r for r in result_rows if r["dataset"] == ds and r["method"] == "MS_HSGC"]
        if not rows_ds:
            continue
        r = rows_ds[0]
        lines.append(
            f"| {ds} | {r.get('uncertain_total', 'N/A')} "
            f"| {r.get('routed_1hop', 'N/A')} "
            f"| {r.get('routed_2hop', 'N/A')} "
            f"| {r.get('frac_mlp_only_uncertain', 'N/A')} "
            f"| {r.get('blocked_by_h1', 'N/A')} "
            f"| {r.get('blocked_by_r1', 'N/A')} "
            f"| {r.get('blocked_by_r2', 'N/A')} "
            f"| {r.get('blocked_by_delta', 'N/A')} |"
        )
    lines.append("")

    # --- Binned diagnostics summary (margin bins) ---
    lines.append("## 5. Margin-Binned Diagnostics")
    lines.append("")

    for method in ["FINAL_V3", "MS_HSGC"]:
        lines.append(f"### {method}")
        lines.append("")
        lines.append("| Dataset | margin_bin | n_nodes | frac_corrected | n_helped | n_hurt | net_help |")
        lines.append("|---|---|---|---|---|---|---|")
        for ds in datasets:
            ds_rows = [b for b in binned_rows
                       if b["dataset"] == ds and b["method"] == method
                       and b.get("bin_type") == "margin"]
            for b in ds_rows:
                lines.append(
                    f"| {ds} | {b['bin_label']} | {b['n_nodes']} "
                    f"| {b['frac_corrected']} | {b['n_helped']} "
                    f"| {b['n_hurt']} | {b['net_help']} |"
                )
        lines.append("")

    # --- Degree-binned diagnostics summary ---
    lines.append("## 6. Degree-Binned Diagnostics")
    lines.append("")

    for method in ["FINAL_V3", "MS_HSGC"]:
        lines.append(f"### {method}")
        lines.append("")
        lines.append("| Dataset | degree_bin | n_nodes | frac_corrected | n_helped | n_hurt | net_help |")
        lines.append("|---|---|---|---|---|---|---|")
        for ds in datasets:
            ds_rows = [b for b in binned_rows
                       if b["dataset"] == ds and b["method"] == method
                       and b.get("bin_type") == "degree"]
            for b in ds_rows:
                lines.append(
                    f"| {ds} | {b['bin_label']} | {b['n_nodes']} "
                    f"| {b['frac_corrected']} | {b['n_helped']} "
                    f"| {b['n_hurt']} | {b['net_help']} |"
                )
        lines.append("")

    # --- Answer summary questions ---
    lines.append("## 7. Summary Questions")
    lines.append("")

    # Build per-dataset delta dictionaries from delta_rows
    by_ds_delta: Dict[str, Dict] = {}
    for r in delta_rows:
        ds = r["dataset"]
        if ds not in by_ds_delta:
            by_ds_delta[ds] = r
    v3_gains = [(ds, float(by_ds_delta[ds].get("delta_final_v3_vs_mlp", 0) or 0))
                for ds in datasets if ds in by_ds_delta]
    ms_gains = [(ds, float(by_ds_delta[ds].get("delta_ms_hsgc_vs_mlp", 0) or 0))
                for ds in datasets if ds in by_ds_delta]
    ms_1hop_usage = {}
    ms_2hop_usage = {}
    for r in result_rows:
        if r["method"] == "MS_HSGC":
            ms_1hop_usage[r["dataset"]] = float(r.get("frac_corrected_1hop") or 0)
            ms_2hop_usage[r["dataset"]] = float(r.get("frac_corrected_2hop") or 0)

    v3_best = sorted(v3_gains, key=lambda x: -x[1])
    ms_best = sorted(ms_gains, key=lambda x: -x[1])
    ms_1hop_sorted = sorted(ms_1hop_usage.items(), key=lambda x: -x[1])
    ms_2hop_sorted = sorted(ms_2hop_usage.items(), key=lambda x: -x[1])

    def _ds_gain_str(lst):
        return ", ".join(f"{ds} ({v:+.4f})" for ds, v in lst[:3])

    lines.append("**Q1: On which datasets does FINAL_V3 help most?**")
    if v3_best:
        lines.append(f"Top datasets by delta vs MLP: {_ds_gain_str(v3_best)}")
    else:
        lines.append("Not enough data.")
    lines.append("")

    lines.append("**Q2: On which datasets does MS_HSGC engage routing meaningfully?**")
    if ms_1hop_sorted:
        lines.append(
            "Top datasets by frac_1hop: "
            + ", ".join(f"{ds} ({v:.3f})" for ds, v in ms_1hop_sorted[:3])
        )
    lines.append("")

    lines.append("**Q3: On which datasets is MS_HSGC mostly collapsing back to MLP?**")
    ms_collapse = [(ds, v) for ds, v in ms_1hop_usage.items()
                   if v < 0.01 and ms_2hop_usage.get(ds, 0) < 0.01]
    if ms_collapse:
        lines.append("Datasets with frac_1hop < 0.01 and frac_2hop < 0.01: "
                     + ", ".join(ds for ds, _ in ms_collapse))
    else:
        lines.append("No dataset fully collapsed to MLP (some routing active).")
    lines.append("")

    lines.append("**Q4: Is 2-hop routing being used at all, and where?**")
    if ms_2hop_sorted:
        active_2hop = [(ds, v) for ds, v in ms_2hop_sorted if v > 0.005]
        if active_2hop:
            lines.append("Datasets with frac_2hop > 0.5%: "
                         + ", ".join(f"{ds} ({v:.3f})" for ds, v in active_2hop))
        else:
            lines.append("2-hop routing is not meaningfully active on any dataset (<0.5%).")
    lines.append("")

    lines.append("**Q5: In which margin bins do corrections help most?**")
    best_bin = _best_margin_bin(binned_rows, "FINAL_V3")
    if best_bin:
        lines.append(f"FINAL_V3: Best margin bin overall is `{best_bin}` (highest net_help).")
    best_bin_ms = _best_margin_bin(binned_rows, "MS_HSGC")
    if best_bin_ms:
        lines.append(f"MS_HSGC: Best margin bin overall is `{best_bin_ms}` (highest net_help).")
    lines.append("")

    lines.append("**Q6: In which degree bins do corrections help or hurt?**")
    lines.append(_degree_bin_summary(binned_rows, "FINAL_V3"))
    lines.append(_degree_bin_summary(binned_rows, "MS_HSGC"))
    lines.append("")

    lines.append("**Q7: Does DeltaH correlate with useful 2-hop routing?**")
    delta_h_insight = _deltah_routing_insight(binned_rows)
    lines.append(delta_h_insight)
    lines.append("")

    lines.append("**Q8: What appears to be the main failure mode of MS_HSGC?**")
    failure_mode = _diagnose_ms_failure(result_rows)
    lines.append(failure_mode)
    lines.append("")

    # --- Key insights ---
    lines.append("## 8. Key Insights")
    lines.append("")
    insights = _build_key_insights(result_rows, binned_rows, datasets)
    for i, ins in enumerate(insights, 1):
        lines.append(f"{i}. {ins}")
    lines.append("")

    lines.append("## 9. Next Hypotheses")
    lines.append("")
    next_hyp = _build_next_hypotheses(result_rows, binned_rows, datasets)
    for i, hyp in enumerate(next_hyp, 1):
        lines.append(f"{i}. {hyp}")
    lines.append("")

    lines.append("---")
    lines.append("*Reconnaissance sweep — non-canonical diagnostic. "
                 "Do not cite in the paper without re-running with canonical settings.*")

    return "\n".join(lines)


def _best_margin_bin(binned_rows, method):
    bin_net: Dict[str, int] = defaultdict(int)
    bin_n: Dict[str, int] = defaultdict(int)
    for b in binned_rows:
        if b.get("method") == method and b.get("bin_type") == "margin":
            lbl = b.get("bin_label", "")
            try:
                bin_net[lbl] += int(b.get("net_help", 0))
                bin_n[lbl] += int(b.get("n_nodes", 0))
            except (TypeError, ValueError):
                pass
    if not bin_net:
        return None
    return max(bin_net, key=lambda k: bin_net[k])


def _degree_bin_summary(binned_rows, method):
    bin_net: Dict[str, int] = defaultdict(int)
    for b in binned_rows:
        if b.get("method") == method and b.get("bin_type") == "degree":
            lbl = b.get("bin_label", "")
            try:
                bin_net[lbl] += int(b.get("net_help", 0))
            except (TypeError, ValueError):
                pass
    if not bin_net:
        return f"  {method}: No degree-bin data."
    best = max(bin_net, key=lambda k: bin_net[k])
    worst = min(bin_net, key=lambda k: bin_net[k])
    return (f"  {method}: Best degree bin = `{best}` (net_help={bin_net[best]}), "
            f"worst = `{worst}` (net_help={bin_net[worst]}).")


def _deltah_routing_insight(binned_rows):
    high_delta_2hop = []
    low_delta_2hop = []
    for b in binned_rows:
        if b.get("method") == "MS_HSGC" and b.get("bin_type") == "hetero_DeltaH":
            try:
                f2 = float(b.get("frac_routed_2hop", 0) or 0)
                lbl = b.get("bin_label", "")
                if lbl in ("Q4_high",):
                    high_delta_2hop.append(f2)
                elif lbl in ("Q1_low",):
                    low_delta_2hop.append(f2)
            except (TypeError, ValueError):
                pass
    if high_delta_2hop and low_delta_2hop:
        avg_high = float(np.mean(high_delta_2hop))
        avg_low = float(np.mean(low_delta_2hop))
        if avg_high > avg_low + 0.01:
            return (f"Yes: High-DeltaH nodes (Q4) have mean frac_2hop={avg_high:.3f} vs "
                    f"low-DeltaH (Q1) frac_2hop={avg_low:.3f}. "
                    "2-hop routing activates more where DeltaH is positive.")
        else:
            return (f"Weak signal: High-DeltaH frac_2hop={avg_high:.3f} "
                    f"vs low-DeltaH frac_2hop={avg_low:.3f}. "
                    "DeltaH does not strongly predict 2-hop usage in this run.")
    return "Insufficient heterophily-bin data to conclude."


def _diagnose_ms_failure(result_rows):
    ms_rows = [r for r in result_rows if r.get("method") == "MS_HSGC"]
    if not ms_rows:
        return "No MS_HSGC data available."

    total_unc = sum(int(r.get("uncertain_total") or 0) for r in ms_rows)
    total_1hop = sum(int(r.get("routed_1hop") or 0) for r in ms_rows)
    total_2hop = sum(int(r.get("routed_2hop") or 0) for r in ms_rows)
    total_bh1 = sum(int(r.get("blocked_by_h1") or 0) for r in ms_rows)
    total_br1 = sum(int(r.get("blocked_by_r1") or 0) for r in ms_rows)
    total_br2 = sum(int(r.get("blocked_by_r2") or 0) for r in ms_rows)
    total_bd = sum(int(r.get("blocked_by_delta") or 0) for r in ms_rows)

    if total_unc == 0:
        return ("Very few uncertain nodes detected overall — over-conservative confidence "
                "gate (tau) may be the dominant bottleneck.")

    routed_frac = (total_1hop + total_2hop) / max(total_unc, 1)
    if routed_frac < 0.1:
        # Very few corrections
        dominant_block = max(
            [("blocked_by_r1", total_br1), ("blocked_by_h1", total_bh1),
             ("blocked_by_r2", total_br2), ("blocked_by_delta", total_bd)],
            key=lambda x: x[1],
        )
        return (
            f"Only {routed_frac:.1%} of uncertain nodes get corrected. "
            f"Dominant routing block: `{dominant_block[0]}` "
            f"({dominant_block[1]} nodes). "
            "Main failure mode: over-conservative routing thresholds "
            f"(especially {dominant_block[0].replace('blocked_by_', '')}-related gates)."
        )
    elif total_2hop < total_1hop * 0.1:
        return (
            "1-hop corrections are active but 2-hop is barely used. "
            "The 2-hop route is gated out, likely by R2/delta-min thresholds. "
            "Main failure mode: weak 2-hop reliability signal or delta_min too strict."
        )
    else:
        return (
            f"Routing is moderately active (routed_frac={routed_frac:.1%}). "
            "No single dominant failure mode — review net_help per dataset."
        )


def _build_key_insights(result_rows, binned_rows, datasets):
    insights = []

    # Insight 1: Where does FINAL_V3 help most?
    v3_gains = []
    mlp_accs = {}
    v3_accs = {}
    ms_accs = {}
    for r in result_rows:
        ds = r["dataset"]
        m = r["method"]
        ta = r.get("test_acc")
        if ta not in (None, ""):
            if m == "MLP_ONLY":
                mlp_accs[ds] = float(ta)
            elif m == "FINAL_V3":
                v3_accs[ds] = float(ta)
            elif m == "MS_HSGC":
                ms_accs[ds] = float(ta)

    for ds in datasets:
        if ds in mlp_accs and ds in v3_accs:
            v3_gains.append((ds, v3_accs[ds] - mlp_accs[ds]))
    v3_gains.sort(key=lambda x: -x[1])

    if v3_gains:
        top_ds = v3_gains[0]
        bot_ds = v3_gains[-1]
        insights.append(
            f"FINAL_V3 helps most on **{top_ds[0]}** (delta={top_ds[1]:+.4f}) "
            f"and hurts/is neutral on **{bot_ds[0]}** (delta={bot_ds[1]:+.4f})."
        )

    # Insight 2: MS_HSGC routing engagement
    ms_1hop = {r["dataset"]: float(r.get("frac_corrected_1hop") or 0)
               for r in result_rows if r["method"] == "MS_HSGC"}
    ms_2hop = {r["dataset"]: float(r.get("frac_corrected_2hop") or 0)
               for r in result_rows if r["method"] == "MS_HSGC"}

    active_ds = [ds for ds in datasets
                 if ms_1hop.get(ds, 0) + ms_2hop.get(ds, 0) > 0.02]
    dormant_ds = [ds for ds in datasets
                  if ms_1hop.get(ds, 0) + ms_2hop.get(ds, 0) <= 0.01]
    if active_ds or dormant_ds:
        insights.append(
            "MS_HSGC routing is active (>2% corrected) on: ["
            + ", ".join(active_ds) + "]. "
            "Mostly dormant (≤1% corrected) on: [" + ", ".join(dormant_ds) + "]."
        )

    # Insight 3: 2-hop vs 1-hop usage
    total_1hop = sum(ms_1hop.get(ds, 0) for ds in datasets)
    total_2hop = sum(ms_2hop.get(ds, 0) for ds in datasets)
    if total_2hop < 0.01 * len(datasets):
        insights.append(
            "2-hop routing is effectively inactive in this light run. "
            "The h1_max and delta_min gates appear over-conservative or 2-hop reliability "
            "is insufficient. This is the most critical open question for MS_HSGC."
        )
    else:
        insights.append(
            f"2-hop routing is non-trivially active (mean_frac_2hop ≈ "
            f"{total_2hop / max(len(datasets), 1):.3f}). "
            "DeltaH signal may be informative — investigate hetero-bin diagnostics."
        )

    return insights[:5]  # cap at 5


def _build_next_hypotheses(result_rows, binned_rows, datasets):
    hyps = [
        "Lower the routing thresholds (rho1, rho2, h1_max) in a dedicated light grid "
        "to see if increased engagement improves accuracy, or hurts it. "
        "This tests whether the bottleneck is gate conservatism.",

        "Run a 3-split (splits 0, 1, 2) version of this sweep on the top-3 datasets "
        "where FINAL_V3 showed the largest deltas, to build statistical confidence "
        "before any full 10-split benchmark.",
    ]
    return hyps


# ---------------------------------------------------------------------------
# Comparison table (for tables/ directory)
# ---------------------------------------------------------------------------

def _build_comparison_table(result_rows, delta_rows, datasets, splits):
    lines = []
    lines.append("# Reconnaissance Behavior — Comparison Table")
    lines.append("")
    lines.append("**Status:** Non-canonical diagnostic run.")
    lines.append("")
    lines.append("| Dataset | Split | MLP | FINAL_V3 | MS_HSGC | V3-MLP | MS-MLP | MS-V3 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in delta_rows:
        ds = r["dataset"]
        sid = r["split_id"]
        lines.append(
            f"| {ds} | {sid} "
            f"| {_fmt(r.get('mlp_test_acc'))} "
            f"| {_fmt(r.get('final_v3_test_acc'))} "
            f"| {_fmt(r.get('ms_hsgc_test_acc'))} "
            f"| {_fmt(r.get('delta_final_v3_vs_mlp'), '+.4f')} "
            f"| {_fmt(r.get('delta_ms_hsgc_vs_mlp'), '+.4f')} "
            f"| {_fmt(r.get('delta_ms_hsgc_vs_final_v3'), '+.4f')} |"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Reconnaissance behavior sweep")
    parser.add_argument("--split-dir", default="data/splits")
    parser.add_argument(
        "--datasets", nargs="+",
        default=[
            "chameleon", "squirrel", "texas", "wisconsin",
            "cornell", "citeseer", "cora", "pubmed",
        ],
    )
    parser.add_argument("--splits", nargs="+", type=int, default=[0])
    parser.add_argument("--output-tag", default="recon_sweep")
    parser.add_argument(
        "--light", action="store_true", default=True,
        help="Lightweight reconnaissance mode (reduced MLP epochs + minimal grid). Non-canonical.",
    )
    parser.add_argument(
        "--mode", choices=["recon", "full"], default="recon",
        help="'recon' = light mode (default), 'full' = canonical-equivalent settings.",
    )
    args = parser.parse_args()

    # --mode recon implies --light
    light = args.light or (args.mode == "recon")

    result_rows, binned_rows, delta_rows = run_recon_sweep(
        datasets=args.datasets,
        splits=args.splits,
        split_dir=args.split_dir,
        output_tag=args.output_tag,
        light=light,
    )

    tag = f"_{args.output_tag}" if args.output_tag else ""
    results_path = f"reports/recon_behavior_results{tag}.csv"
    binned_path = f"reports/recon_behavior_binned{tag}.csv"
    delta_path = f"reports/recon_ms_hsgc_vs_final_v3{tag}.csv"
    summary_path = f"reports/recon_behavior_summary{tag}.md"
    table_path = f"tables/recon_behavior_comparison{tag}.md"

    print("\n--- Writing outputs ---")
    _write_csv(results_path, result_rows, RESULT_FIELDS)
    _write_csv(binned_path, binned_rows, BINNED_FIELDS)
    _write_csv(delta_path, delta_rows, DELTA_FIELDS)

    summary_md = _build_summary(
        result_rows, binned_rows, delta_rows,
        light=light,
        datasets=args.datasets,
        splits=args.splits,
    )
    os.makedirs("reports", exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write(summary_md)
    print(f"  Written: {summary_path}")

    os.makedirs("tables", exist_ok=True)
    comparison_md = _build_comparison_table(result_rows, delta_rows, args.datasets, args.splits)
    with open(table_path, "w", encoding="utf-8") as fh:
        fh.write(comparison_md)
    print(f"  Written: {table_path}")

    print("\n--- Command used ---")
    print(
        f"python3 code/bfsbased_node_classification/run_recon_behavior.py "
        f"{'--light ' if light else ''}"
        f"--datasets {' '.join(args.datasets)} "
        f"--splits {' '.join(str(s) for s in args.splits)} "
        f"--split-dir {args.split_dir} "
        f"--output-tag {args.output_tag}"
    )


if __name__ == "__main__":
    main()

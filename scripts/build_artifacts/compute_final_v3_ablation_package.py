#!/usr/bin/env python3
"""
Compute a compact FINAL_V3 ablation package for the frozen six-dataset PRL setup.

This script retrains the MLP once per split, then evaluates several lightweight
FINAL_V3 variants using the same cached MLP probabilities:

- MLP_ONLY
- FINAL_V3
- ALWAYS_CORRECT_ALL            (no uncertainty/reliability gating)
- UNCERTAINTY_ONLY              (uncertainty gate only; no reliability gate)
- NO_FEATURE_SIMILARITY         (b2 = 0)
- NO_GRAPH_NEIGHBOR             (b4 = 0)
- NO_COMPATIBILITY              (b5 = 0)
- PROFILE_A_ONLY
- PROFILE_B_ONLY

Outputs:
- logs/prl_ablation_results.csv
- logs/prl_ablation_summary.csv
- logs/prl_gate_analysis.csv
- reports/prl_ablation_interpretation.md
- reports/prl_gate_analysis.md
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code" / "bfsbased_node_classification"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from final_method_v3 import WEIGHT_PROFILES, compute_graph_reliability, _simple_accuracy

RESULTS_PATH = ROOT / "logs" / "prl_ablation_results.csv"
SUMMARY_PATH = ROOT / "logs" / "prl_ablation_summary.csv"
GATE_PATH = ROOT / "logs" / "prl_gate_analysis.csv"
ABLATION_REPORT = ROOT / "reports" / "prl_ablation_interpretation.md"
GATE_REPORT = ROOT / "reports" / "prl_gate_analysis.md"

DEFAULT_DATASETS = ["cora", "citeseer", "pubmed", "chameleon", "texas", "wisconsin"]
DEFAULT_SPLITS = list(range(10))
RHO_CANDIDATES = [0.3, 0.4, 0.5, 0.6]


def _load_module():
    path = CODE / "bfsbased-full-investigate-homophil.py"
    source = path.read_text(encoding="utf-8")
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
    setattr(mod, "__file__", str(path))
    setattr(mod, "DATASET_KEY", "texas")
    exec(compile(source, str(path), "exec"), mod.__dict__)
    return mod


def _load_data(mod, ds: str):
    dataset = mod.load_dataset(ds, "data/")
    data = dataset[0]
    data.edge_index = torch.cat([data.edge_index, data.edge_index.flip(0)], dim=1)
    data.edge_index = torch.unique(data.edge_index, dim=1)
    return data


def _load_split(ds: str, sid: int, device, split_dir: str):
    prefix = "film" if ds == "actor" else ds.lower()
    fname = f"{prefix}_split_0.6_0.2_{sid}.npz"
    path = Path(split_dir) / fname
    if not path.is_file():
        raise FileNotFoundError(path)
    sp = np.load(path)
    to_t = lambda m: torch.as_tensor(np.where(m)[0], dtype=torch.long, device=device)
    return to_t(sp["train_mask"]), to_t(sp["val_mask"]), to_t(sp["test_mask"])


def _build_tau_candidates(val_margins: np.ndarray) -> list[float]:
    return sorted(
        set(
            np.round(
                np.concatenate(
                    [
                        np.quantile(val_margins, [0.25, 0.40, 0.50, 0.60, 0.75]),
                        np.array([0.05, 0.10, 0.20, 0.30]),
                    ]
                ),
                5,
            ).tolist()
        )
    )


def _correction_stats(y_true: np.ndarray, mlp_pred: np.ndarray, final_pred: np.ndarray, test_np: np.ndarray):
    changed = final_pred[test_np] != mlp_pred[test_np]
    idx = test_np[changed]
    if idx.size == 0:
        return 0, 0, 0, 1.0
    was_ok = mlp_pred[idx] == y_true[idx]
    now_ok = final_pred[idx] == y_true[idx]
    n_helped = int((~was_ok & now_ok).sum())
    n_hurt = int((was_ok & ~now_ok).sum())
    n_changed = int(idx.size)
    precision = float(n_helped / max(n_helped + n_hurt, 1))
    return n_helped, n_hurt, n_changed, precision


def _variant_profiles(name: str) -> list[dict[str, float]]:
    if name == "FINAL_V3":
        return [dict(p) for p in WEIGHT_PROFILES]
    if name == "NO_FEATURE_SIMILARITY":
        return [{**dict(p), "b2": 0.0} for p in WEIGHT_PROFILES]
    if name == "NO_GRAPH_NEIGHBOR":
        return [{**dict(p), "b4": 0.0} for p in WEIGHT_PROFILES]
    if name == "NO_COMPATIBILITY":
        return [{**dict(p), "b5": 0.0} for p in WEIGHT_PROFILES]
    if name == "PROFILE_A_ONLY":
        return [dict(WEIGHT_PROFILES[0])]
    if name == "PROFILE_B_ONLY":
        return [dict(WEIGHT_PROFILES[1])]
    if name in {"ALWAYS_CORRECT_ALL", "UNCERTAINTY_ONLY"}:
        return [dict(p) for p in WEIGHT_PROFILES]
    raise ValueError(f"Unknown variant {name}")


def _evaluate_variant(
    variant: str,
    *,
    y_true: np.ndarray,
    val_np: np.ndarray,
    test_np: np.ndarray,
    mlp_probs_np: np.ndarray,
    mlp_pred_all: np.ndarray,
    mlp_margin_all: np.ndarray,
    evidence: dict,
    reliability: np.ndarray,
    mod,
):
    profiles = _variant_profiles(variant)
    tau_candidates = _build_tau_candidates(mlp_margin_all[val_np])
    best_key = None
    best_cfg = None
    best_scores = None

    for profile in profiles:
        scores = mod.build_selective_correction_scores(
            mlp_probs_np,
            evidence,
            b1=profile["b1"],
            b2=profile["b2"],
            b3=profile["b3"],
            b4=profile["b4"],
            b5=profile["b5"],
            b6=profile.get("b6", 0.0),
        )["combined_scores"]

        if variant == "ALWAYS_CORRECT_ALL":
            final_val = np.argmax(scores[val_np], axis=1).astype(np.int64)
            val_acc = _simple_accuracy(y_true[val_np], final_val)
            changed_frac = float((final_val != mlp_pred_all[val_np]).mean())
            key = (val_acc, -changed_frac)
            if best_key is None or key > best_key:
                best_key = key
                best_cfg = {"profile": dict(profile), "tau": float("nan"), "rho": float("nan")}
                best_scores = scores
            continue

        for tau in tau_candidates:
            uncertain_val = mlp_margin_all[val_np] < tau
            rho_values = [float("nan")] if variant == "UNCERTAINTY_ONLY" else RHO_CANDIDATES
            for rho in rho_values:
                if variant == "UNCERTAINTY_ONLY":
                    apply_val = uncertain_val
                else:
                    apply_val = uncertain_val & (reliability[val_np] >= rho)
                final_val = mlp_pred_all[val_np].copy()
                if apply_val.any():
                    final_val[apply_val] = np.argmax(scores[val_np][apply_val], axis=1).astype(np.int64)
                val_acc = _simple_accuracy(y_true[val_np], final_val)
                changed_frac = float((final_val != mlp_pred_all[val_np]).mean())
                key = (val_acc, -changed_frac)
                if best_key is None or key > best_key:
                    best_key = key
                    best_cfg = {"profile": dict(profile), "tau": float(tau), "rho": float(rho)}
                    best_scores = scores

    final_pred = mlp_pred_all.copy()
    if variant == "ALWAYS_CORRECT_ALL":
        final_pred = np.argmax(best_scores, axis=1).astype(np.int64)
        applied_all = np.ones_like(mlp_pred_all, dtype=bool)
        uncertain_all = np.ones_like(mlp_pred_all, dtype=bool)
        reliable_all = np.ones_like(mlp_pred_all, dtype=bool)
    else:
        uncertain_all = mlp_margin_all < best_cfg["tau"]
        if variant == "UNCERTAINTY_ONLY":
            reliable_all = np.ones_like(uncertain_all, dtype=bool)
            applied_all = uncertain_all
        else:
            reliable_all = reliability >= best_cfg["rho"]
            applied_all = uncertain_all & reliable_all
        if applied_all.any():
            final_pred[applied_all] = np.argmax(best_scores[applied_all], axis=1).astype(np.int64)

    test_acc = _simple_accuracy(y_true[test_np], final_pred[test_np])
    n_helped, n_hurt, n_changed, precision = _correction_stats(y_true, mlp_pred_all, final_pred, test_np)
    return {
        "test_acc": float(test_acc),
        "delta_vs_mlp": float(test_acc - _simple_accuracy(y_true[test_np], mlp_pred_all[test_np])),
        "selected_tau": best_cfg["tau"],
        "selected_rho": best_cfg["rho"],
        "weights": best_cfg["profile"],
        "n_helped": n_helped,
        "n_hurt": n_hurt,
        "n_changed": n_changed,
        "correction_precision": precision,
        "frac_changed": float((final_pred[test_np] != mlp_pred_all[test_np]).mean()),
        "frac_corrected": float(applied_all[test_np].mean()),
        "frac_uncertain": float(uncertain_all[test_np].mean()),
        "frac_reliable": float(reliable_all[test_np].mean()),
    }


def run_package(datasets: list[str], splits: list[int], split_dir: str):
    mod = _load_module()
    rows: list[dict] = []
    variants = [
        "FINAL_V3",
        "ALWAYS_CORRECT_ALL",
        "UNCERTAINTY_ONLY",
        "NO_FEATURE_SIMILARITY",
        "NO_GRAPH_NEIGHBOR",
        "NO_COMPATIBILITY",
        "PROFILE_A_ONLY",
        "PROFILE_B_ONLY",
    ]

    for ds in datasets:
        print(f"\n{'=' * 60}\nAblation dataset: {ds}\n{'=' * 60}")
        data = _load_data(mod, ds)
        y_true = data.y.detach().cpu().numpy().astype(np.int64)

        for sid in splits:
            train_idx, val_idx, test_idx = _load_split(ds, sid, data.x.device, split_dir)
            train_np = mod._to_numpy_idx(train_idx)
            val_np = mod._to_numpy_idx(val_idx)
            test_np = mod._to_numpy_idx(test_idx)
            seed = (1337 + sid) * 100
            torch.manual_seed(seed)
            np.random.seed(seed)

            mlp_probs, _ = mod.train_mlp_and_predict(
                data,
                train_idx,
                **mod.DEFAULT_MANUSCRIPT_MLP_KWARGS,
                log_file=None,
            )
            mlp_info = mod.compute_mlp_margin(mlp_probs)
            mlp_probs_np = mlp_info["mlp_probs_np"]
            mlp_pred_all = mlp_info["mlp_pred_all"]
            mlp_margin_all = mlp_info["mlp_margin_all"]
            mlp_test_acc = _simple_accuracy(y_true[test_np], mlp_pred_all[test_np])

            evidence = mod._build_selective_correction_evidence(
                data,
                train_np,
                mlp_probs_np=mlp_probs_np,
                enable_feature_knn=False,
            )
            reliability = compute_graph_reliability(mlp_probs_np, evidence)

            rows.append(
                {
                    "dataset": ds,
                    "split_id": sid,
                    "seed": seed,
                    "variant": "MLP_ONLY",
                    "test_acc": float(mlp_test_acc),
                    "delta_vs_mlp": 0.0,
                    "selected_tau": float("nan"),
                    "selected_rho": float("nan"),
                    "n_helped": 0,
                    "n_hurt": 0,
                    "n_changed": 0,
                    "correction_precision": float("nan"),
                    "frac_changed": 0.0,
                    "frac_corrected": 0.0,
                    "frac_uncertain": 0.0,
                    "frac_reliable": 1.0,
                    "weights_json": "",
                }
            )

            for variant in variants:
                out = _evaluate_variant(
                    variant,
                    y_true=y_true,
                    val_np=val_np,
                    test_np=test_np,
                    mlp_probs_np=mlp_probs_np,
                    mlp_pred_all=mlp_pred_all,
                    mlp_margin_all=mlp_margin_all,
                    evidence=evidence,
                    reliability=reliability,
                    mod=mod,
                )
                rows.append(
                    {
                        "dataset": ds,
                        "split_id": sid,
                        "seed": seed,
                        "variant": variant,
                        "test_acc": out["test_acc"],
                        "delta_vs_mlp": out["delta_vs_mlp"],
                        "selected_tau": out["selected_tau"],
                        "selected_rho": out["selected_rho"],
                        "n_helped": out["n_helped"],
                        "n_hurt": out["n_hurt"],
                        "n_changed": out["n_changed"],
                        "correction_precision": out["correction_precision"],
                        "frac_changed": out["frac_changed"],
                        "frac_corrected": out["frac_corrected"],
                        "frac_uncertain": out["frac_uncertain"],
                        "frac_reliable": out["frac_reliable"],
                        "weights_json": str(out["weights"]),
                    }
                )

            print(
                f"  split={sid} mlp={mlp_test_acc:.4f} "
                f"v3={rows[-8]['test_acc']:.4f} "
                f"always={rows[-7]['test_acc']:.4f} "
                f"u-only={rows[-6]['test_acc']:.4f}"
            )
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(rows: list[dict]):
    by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_key[(row["dataset"], row["variant"])].append(row)

    summary_rows = []
    for (dataset, variant), group in sorted(by_key.items()):
        full_group = by_key.get((dataset, "FINAL_V3"), [])
        mlp_group = by_key.get((dataset, "MLP_ONLY"), [])
        mean_acc = float(np.mean([r["test_acc"] for r in group]))
        std_acc = float(np.std([r["test_acc"] for r in group]))
        mean_delta = float(np.mean([r["delta_vs_mlp"] for r in group]))
        mean_changed = float(np.mean([r["frac_changed"] for r in group]))
        mean_corrected = float(np.mean([r["frac_corrected"] for r in group]))
        prec_vals = np.asarray([r["correction_precision"] for r in group], dtype=float)
        mean_prec = float(np.nanmean(prec_vals)) if np.isfinite(prec_vals).any() else float("nan")
        wins_vs_v3 = losses_vs_v3 = ties_vs_v3 = 0
        wins_vs_mlp = losses_vs_mlp = ties_vs_mlp = 0
        if full_group and len(full_group) == len(group):
            for r, f in zip(sorted(group, key=lambda x: x["split_id"]), sorted(full_group, key=lambda x: x["split_id"])):
                diff = r["test_acc"] - f["test_acc"]
                if diff > 1e-12:
                    wins_vs_v3 += 1
                elif diff < -1e-12:
                    losses_vs_v3 += 1
                else:
                    ties_vs_v3 += 1
        if mlp_group and len(mlp_group) == len(group):
            for r, m in zip(sorted(group, key=lambda x: x["split_id"]), sorted(mlp_group, key=lambda x: x["split_id"])):
                diff = r["test_acc"] - m["test_acc"]
                if diff > 1e-12:
                    wins_vs_mlp += 1
                elif diff < -1e-12:
                    losses_vs_mlp += 1
                else:
                    ties_vs_mlp += 1
        summary_rows.append(
            {
                "dataset": dataset,
                "variant": variant,
                "mean_test_acc": f"{mean_acc:.4f}",
                "std_test_acc": f"{std_acc:.4f}",
                "mean_delta_vs_mlp": f"{mean_delta:+.4f}",
                "mean_frac_changed": f"{mean_changed:.4f}",
                "mean_frac_corrected": f"{mean_corrected:.4f}",
                "mean_correction_precision": "" if math.isnan(mean_prec) else f"{mean_prec:.4f}",
                "wins_vs_final_v3": wins_vs_v3,
                "losses_vs_final_v3": losses_vs_v3,
                "ties_vs_final_v3": ties_vs_v3,
                "wins_vs_mlp": wins_vs_mlp,
                "losses_vs_mlp": losses_vs_mlp,
                "ties_vs_mlp": ties_vs_mlp,
                "n_runs": len(group),
            }
        )
    return summary_rows


def write_gate_analysis(summary_rows: list[dict]) -> list[dict]:
    keep = {"MLP_ONLY", "ALWAYS_CORRECT_ALL", "UNCERTAINTY_ONLY", "FINAL_V3"}
    gate_rows = [row for row in summary_rows if row["variant"] in keep]
    gate_rows.sort(key=lambda r: (r["dataset"], ["MLP_ONLY", "ALWAYS_CORRECT_ALL", "UNCERTAINTY_ONLY", "FINAL_V3"].index(r["variant"])))
    return gate_rows


def write_reports(summary_rows: list[dict], gate_rows: list[dict]) -> None:
    by_dataset = defaultdict(dict)
    for row in summary_rows:
        by_dataset[row["dataset"]][row["variant"]] = row

    ablation_lines = [
        "# FINAL_V3 ablation interpretation",
        "",
        "This report was generated from `logs/prl_ablation_results.csv` using the",
        "current frozen six-dataset, 10-split FINAL_V3 benchmark protocol.",
        "",
        "## Main readings",
        "",
    ]
    for ds in DEFAULT_DATASETS:
        rows = by_dataset.get(ds, {})
        if not rows:
            continue
        full = rows["FINAL_V3"]
        no_graph = rows.get("NO_GRAPH_NEIGHBOR")
        no_compat = rows.get("NO_COMPATIBILITY")
        no_feat = rows.get("NO_FEATURE_SIMILARITY")
        always = rows.get("ALWAYS_CORRECT_ALL")
        u_only = rows.get("UNCERTAINTY_ONLY")
        ablation_lines.append(f"### {ds.capitalize()}")
        ablation_lines.append(
            f"- `FINAL_V3` mean accuracy: `{full['mean_test_acc']}` with mean Δ vs MLP `{full['mean_delta_vs_mlp']}`."
        )
        if no_graph:
            ablation_lines.append(
                f"- Removing graph-neighbor support changes mean accuracy to `{no_graph['mean_test_acc']}` "
                f"({no_graph['mean_delta_vs_mlp']} vs MLP; wins/losses vs FINAL_V3: "
                f"{no_graph['wins_vs_final_v3']}/{no_graph['losses_vs_final_v3']}/{no_graph['ties_vs_final_v3']})."
            )
        if no_compat:
            ablation_lines.append(
                f"- Removing compatibility changes mean accuracy to `{no_compat['mean_test_acc']}` "
                f"({no_compat['mean_delta_vs_mlp']} vs MLP)."
            )
        if no_feat:
            ablation_lines.append(
                f"- Removing feature similarity changes mean accuracy to `{no_feat['mean_test_acc']}` "
                f"({no_feat['mean_delta_vs_mlp']} vs MLP)."
            )
        if always and u_only:
            ablation_lines.append(
                f"- Gate comparison: always-correct `{always['mean_test_acc']}`, uncertainty-only `{u_only['mean_test_acc']}`, "
                f"FINAL_V3 `{full['mean_test_acc']}`."
            )
        ablation_lines.append("")

    ablation_lines.extend(
        [
            "## Global notes",
            "",
            "- `feature-kNN` is not part of the canonical FINAL_V3 score profiles; its weight remains `0.0` in `WEIGHT_PROFILES`.",
            "- The highest-value controlled ablations here are no-graph-neighbor, no-compatibility, no-feature-similarity, and the gate comparisons.",
            "- Interpret these results as manuscript-support evidence for the frozen FINAL_V3 package, not as a new benchmark line.",
            "",
        ]
    )
    ABLATION_REPORT.write_text("\n".join(ablation_lines), encoding="utf-8")

    gate_lines = [
        "# FINAL_V3 gate analysis",
        "",
        "This report compares the no-correction baseline (`MLP_ONLY`),",
        "indiscriminate correction (`ALWAYS_CORRECT_ALL`), uncertainty-only",
        "correction (`UNCERTAINTY_ONLY`), and the full reliability-gated method",
        "(`FINAL_V3`) on the frozen six-dataset, 10-split benchmark.",
        "",
        "| Dataset | MLP | Always-correct | Uncertainty-only | FINAL_V3 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for ds in DEFAULT_DATASETS:
        rows = {r["variant"]: r for r in gate_rows if r["dataset"] == ds}
        if not rows:
            continue
        gate_lines.append(
            f"| {ds.capitalize()} | {rows['MLP_ONLY']['mean_test_acc']} | {rows['ALWAYS_CORRECT_ALL']['mean_test_acc']} | "
            f"{rows['UNCERTAINTY_ONLY']['mean_test_acc']} | {rows['FINAL_V3']['mean_test_acc']} |"
        )
    gate_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `ALWAYS_CORRECT_ALL` tests whether indiscriminate graph correction helps without any selectivity.",
            "- `UNCERTAINTY_ONLY` keeps the uncertainty gate but removes the reliability filter.",
            "- `FINAL_V3` adds the reliability filter on top of uncertainty gating.",
            "- Use `logs/prl_gate_analysis.csv` for manuscript-ready dataset-level values.",
            "",
        ]
    )
    GATE_REPORT.write_text("\n".join(gate_lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--splits", nargs="+", type=int, default=DEFAULT_SPLITS)
    parser.add_argument("--split-dir", default=str(ROOT / "data" / "splits"))
    args = parser.parse_args()

    rows = run_package(args.datasets, args.splits, args.split_dir)
    fieldnames = [
        "dataset",
        "split_id",
        "seed",
        "variant",
        "test_acc",
        "delta_vs_mlp",
        "selected_tau",
        "selected_rho",
        "n_helped",
        "n_hurt",
        "n_changed",
        "correction_precision",
        "frac_changed",
        "frac_corrected",
        "frac_uncertain",
        "frac_reliable",
        "weights_json",
    ]
    write_csv(RESULTS_PATH, fieldnames, rows)
    summary_rows = summarize(rows)
    write_csv(
        SUMMARY_PATH,
        [
            "dataset",
            "variant",
            "mean_test_acc",
            "std_test_acc",
            "mean_delta_vs_mlp",
            "mean_frac_changed",
            "mean_frac_corrected",
            "mean_correction_precision",
            "wins_vs_final_v3",
            "losses_vs_final_v3",
            "ties_vs_final_v3",
            "wins_vs_mlp",
            "losses_vs_mlp",
            "ties_vs_mlp",
            "n_runs",
        ],
        summary_rows,
    )
    gate_rows = write_gate_analysis(summary_rows)
    write_csv(
        GATE_PATH,
        [
            "dataset",
            "variant",
            "mean_test_acc",
            "std_test_acc",
            "mean_delta_vs_mlp",
            "mean_frac_changed",
            "mean_frac_corrected",
            "mean_correction_precision",
            "wins_vs_final_v3",
            "losses_vs_final_v3",
            "ties_vs_final_v3",
            "wins_vs_mlp",
            "losses_vs_mlp",
            "ties_vs_mlp",
            "n_runs",
        ],
        gate_rows,
    )
    write_reports(summary_rows, gate_rows)
    print("Wrote", RESULTS_PATH)
    print("Wrote", SUMMARY_PATH)
    print("Wrote", GATE_PATH)
    print("Wrote", ABLATION_REPORT)
    print("Wrote", GATE_REPORT)


if __name__ == "__main__":
    main()

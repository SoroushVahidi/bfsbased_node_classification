#!/usr/bin/env python3
"""Lightweight node-level correction explanation audit for FINAL_V3.

This script is manuscript-supporting (non-canonical by default): it reruns FINAL_V3
on a compact split set with `include_node_arrays=True`, reconstructs additive score
components for changed nodes, and exports node-level + aggregate interpretability
artifacts.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import final_method_v3 as fm  # noqa: E402


def _load_module():
    path = HERE / "bfsbased-full-investigate-homophil.py"
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
    spec = importlib.util.spec_from_loader("bfs_full_investigate", loader=None)
    mod = importlib.util.module_from_spec(spec)
    setattr(mod, "__file__", str(path))
    setattr(mod, "DATASET_KEY", "texas")
    exec(compile(source, str(path), "exec"), mod.__dict__)
    return mod


def _load_data(mod, ds: str, root="data/"):
    dataset = mod.load_dataset(ds, root)
    data = dataset[0]
    data.edge_index = torch.cat([data.edge_index, data.edge_index.flip(0)], dim=1)
    data.edge_index = torch.unique(data.edge_index, dim=1)
    return data


def _load_split(ds, sid, device, split_dir):
    from split_paths import split_npz_prefix

    prefix = split_npz_prefix(ds)
    path = Path(split_dir) / f"{prefix}_split_0.6_0.2_{sid}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Split not found: {path}")
    sp = np.load(path)

    def to_t(mask):
        return torch.as_tensor(np.where(mask)[0], dtype=torch.long, device=device)

    return to_t(sp["train_mask"]), to_t(sp["val_mask"]), to_t(sp["test_mask"])


def _safe_mean(values):
    vals = [float(v) for v in values if v is not None and not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _dominant_driver(gaps: dict[str, float]) -> str:
    return max(gaps.items(), key=lambda kv: kv[1])[0]


def run_audit(datasets, splits, split_dir, tag: str):
    mod = _load_module()
    rows = []

    for ds in datasets:
        print(f"\n=== {ds} ===")
        data = _load_data(mod, ds)
        y_true = data.y.detach().cpu().numpy().astype(np.int64)

        for sid in splits:
            try:
                train_idx, val_idx, test_idx = _load_split(ds, sid, data.x.device, split_dir)
            except FileNotFoundError as exc:
                print(f"  [SKIP] split={sid}: {exc}")
                continue

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

            _, _, info = fm.final_method_v3(
                data,
                train_np,
                val_np,
                test_np,
                mlp_probs=mlp_probs,
                seed=seed,
                mod=mod,
                gate="heuristic",
                split_id=sid,
                include_node_arrays=True,
            )

            evidence = mod._build_selective_correction_evidence(
                data,
                train_np,
                mlp_probs_np=mlp_probs_np,
                enable_feature_knn=False,
            )
            w = info["weights"]
            comps = mod.build_selective_correction_scores(
                mlp_probs_np,
                evidence,
                b1=w["b1"],
                b2=w["b2"],
                b3=w["b3"],
                b4=w["b4"],
                b5=w["b5"],
                b6=w["b6"],
            )

            local_mult_all = np.ones_like(mlp_pred_all, dtype=np.float64)
            if info.get("local_agreement", {}).get("enabled", False):
                mode = info["local_agreement"].get("selected_mode", "none")
                eta = info["local_agreement"].get("selected_eta")
                if info["local_agreement"].get("selected_score_family") == "exact":
                    base_terms = (
                        comps["mlp_term"]
                        + comps["feature_similarity_term"]
                        + comps["feature_knn_term"]
                        + comps["structural_far_term"]
                    )
                    graph_terms = comps["graph_neighbor_term"] + comps["compatibility_term"]
                    working_labels = np.argmax(base_terms + graph_terms, axis=1).astype(np.int64)
                    working_labels[train_np] = y_true[train_np]
                    local_scores = fm.compute_local_homophily_from_labels(
                        data,
                        train_np,
                        working_labels,
                        y_true,
                    )
                else:
                    local_scores = fm.compute_soft_local_agreement(
                        data,
                        train_np,
                        mlp_probs_np,
                        mlp_pred_all,
                        y_true,
                    )
                local_mult_all = fm.local_agreement_multiplier(local_scores, mode=mode, eta=eta)

            out = info["test_node_outputs"]
            for i, nid in enumerate(out["node_id"]):
                routed = bool(out["passed_uncertainty_gate"][i] and out["passed_reliability_gate"][i])
                changed = bool(out["changed_from_mlp"][i])
                if not changed:
                    continue

                node_id = int(nid)
                orig_cls = int(out["mlp_pred"][i])
                final_cls = int(out["final_pred"][i])
                true_cls = int(out["true_label"][i])

                m = float(local_mult_all[node_id])
                terms = {
                    "mlp_anchor": comps["mlp_term"],
                    "prototype": comps["feature_similarity_term"],
                    "neighbor_support": m * comps["graph_neighbor_term"],
                    "compatibility": m * comps["compatibility_term"],
                    "feature_knn": comps["feature_knn_term"],
                    "structural_far": comps["structural_far_term"],
                }

                term_orig = {k: float(v[node_id, orig_cls]) for k, v in terms.items()}
                term_final = {k: float(v[node_id, final_cls]) for k, v in terms.items()}
                gaps = {k: term_final[k] - term_orig[k] for k in terms}
                dominant = _dominant_driver(gaps)
                largest_gap = float(gaps[dominant])

                score_orig = float(sum(term_orig.values()))
                score_final = float(sum(term_final.values()))
                score_gap = score_final - score_orig
                concentration = float(max(gaps.values()) / max(sum(abs(v) for v in gaps.values()), 1e-12))

                beneficial = bool(out["beneficial_change"][i])
                harmful = bool(out["harmful_change"][i])

                rows.append(
                    {
                        "dataset": ds,
                        "split": int(sid),
                        "node_id": node_id,
                        "mlp_pred": orig_cls,
                        "final_pred": final_cls,
                        "true_label": true_cls,
                        "beneficial": int(beneficial),
                        "harmful": int(harmful),
                        "routed_to_correction": int(routed),
                        "reliability": float(out["reliability"][i]),
                        "mlp_margin": float(out["mlp_margin"][i]),
                        "score_orig": score_orig,
                        "score_final": score_final,
                        "score_gap": score_gap,
                        "gap_concentration": concentration,
                        "dominant_reason": dominant,
                        "dominant_gap": largest_gap,
                        "mlp_anchor_orig": term_orig["mlp_anchor"],
                        "mlp_anchor_final": term_final["mlp_anchor"],
                        "prototype_orig": term_orig["prototype"],
                        "prototype_final": term_final["prototype"],
                        "neighbor_support_orig": term_orig["neighbor_support"],
                        "neighbor_support_final": term_final["neighbor_support"],
                        "compatibility_orig": term_orig["compatibility"],
                        "compatibility_final": term_final["compatibility"],
                        "feature_knn_orig": term_orig["feature_knn"],
                        "feature_knn_final": term_final["feature_knn"],
                        "structural_far_orig": term_orig["structural_far"],
                        "structural_far_final": term_final["structural_far"],
                        "mlp_anchor_gap": gaps["mlp_anchor"],
                        "prototype_gap": gaps["prototype"],
                        "neighbor_support_gap": gaps["neighbor_support"],
                        "compatibility_gap": gaps["compatibility"],
                        "feature_knn_gap": gaps["feature_knn"],
                        "structural_far_gap": gaps["structural_far"],
                    }
                )

            print(f"  split={sid}: changed nodes logged")

    if not rows:
        raise RuntimeError("No changed nodes found; cannot build explanation audit.")

    reports_dir = ROOT / "reports"
    tables_dir = ROOT / "tables"
    figures_dir = ROOT / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    node_csv = reports_dir / f"correction_explanation_nodes_{tag}.csv"
    with node_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    ben = [r for r in rows if r["beneficial"] == 1]
    harm = [r for r in rows if r["harmful"] == 1]

    def cmean(rows_in, key):
        return _safe_mean([r[key] for r in rows_in])

    driver_set = ["prototype", "neighbor_support", "compatibility", "mlp_anchor", "feature_knn", "structural_far"]

    summary_rows = []
    for label, subset in [("beneficial", ben), ("harmful", harm), ("all_changed", rows)]:
        cnt = Counter(r["dominant_reason"] for r in subset)
        top = cnt.most_common(1)[0][0] if cnt else "none"
        agree = [
            r for r in subset
            if (r["prototype_gap"] > 0 and r["neighbor_support_gap"] > 0)
            or (r["prototype_gap"] < 0 and r["neighbor_support_gap"] < 0)
        ]
        summary_rows.append(
            {
                "group": label,
                "n_nodes": len(subset),
                "most_common_dominant_driver": top,
                "mean_reliability": cmean(subset, "reliability"),
                "mean_score_gap": cmean(subset, "score_gap"),
                "mean_gap_concentration": cmean(subset, "gap_concentration"),
                "prototype_graph_agreement_rate": (len(agree) / len(subset)) if subset else float("nan"),
                **{f"driver_share_{d}": (cnt[d] / len(subset)) if subset else float("nan") for d in driver_set},
            }
        )

    summary_csv = tables_dir / f"correction_explanation_summary_{tag}.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    summary_md = tables_dir / f"correction_explanation_summary_{tag}.md"
    lines = [
        f"# FINAL_V3 correction explanation summary ({tag})",
        "",
        f"Source changed-node export: `{node_csv.relative_to(ROOT)}`.",
        "",
        "| Group | N | Most common dominant driver | Mean reliability | "
        "Mean score gap | Mean dominance concentration | "
        "Prototype/graph agreement |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['group']} | {r['n_nodes']} | {r['most_common_dominant_driver']} | {r['mean_reliability']:.3f} | "
            f"{r['mean_score_gap']:.3f} | {r['mean_gap_concentration']:.3f} | "
            f"{100.0*r['prototype_graph_agreement_rate']:.1f}% |"
        )

    lines.extend([
        "",
        "## Dominant-driver shares",
        "",
        "| Group | " + " | ".join(driver_set) + " |",
        "| --- | " + " | ".join(["---:"] * len(driver_set)) + " |",
    ])
    for r in summary_rows:
        vals = " | ".join(f"{100.0*r[f'driver_share_{d}']:.1f}%" for d in driver_set)
        lines.append(f"| {r['group']} | {vals} |")
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report_md = reports_dir / f"correction_explanation_audit_{tag}.md"
    harmful_low_rel = cmean(harm, "reliability") < cmean(ben, "reliability")
    harmful_small_gap = cmean(harm, "score_gap") < cmean(ben, "score_gap")
    harmful_more_concentrated = cmean(harm, "gap_concentration") > cmean(ben, "gap_concentration")

    report_lines = [
        f"# FINAL_V3 node-level correction explanation audit ({tag})",
        "",
        "This is a manuscript-supporting, non-canonical interpretability audit over changed FINAL_V3 test nodes.",
        "",
        "## What was reconstructed",
        "- Additive evidence components from FINAL_V3 score: MLP anchor, "
        "prototype similarity, graph neighbor support, compatibility, "
        "and zero-weight residual terms (feature_kNN, structural_far).",
        "- Per-node class-wise contribution deltas between final corrected class and original MLP class.",
        "- Dominant correction reason defined as the component with largest contribution gap.",
        "",
        "## Findings",
        "- Beneficial corrections: most common dominant driver = "
        f"**{summary_rows[0]['most_common_dominant_driver']}**.",
        f"- Harmful corrections: most common dominant driver = **{summary_rows[1]['most_common_dominant_driver']}**.",
        f"- Harmful corrections have lower reliability than beneficial: **{harmful_low_rel}**.",
        f"- Harmful corrections have smaller score gaps than beneficial: **{harmful_small_gap}**.",
        f"- Harmful corrections are more single-component concentrated: **{harmful_more_concentrated}**.",
        "- Beneficial prototype/graph agreement rate: "
        f"**{100.0 * summary_rows[0]['prototype_graph_agreement_rate']:.1f}%**.",
        "- Harmful prototype/graph agreement rate: "
        f"**{100.0 * summary_rows[1]['prototype_graph_agreement_rate']:.1f}%**.",
        "",
        "## Outputs",
        f"- Node-level export: `{node_csv.relative_to(ROOT)}`",
        f"- Summary table (CSV): `{summary_csv.relative_to(ROOT)}`",
        f"- Summary table (MD): `{summary_md.relative_to(ROOT)}`",
    ]
    report_md.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    case_md = reports_dir / f"correction_case_studies_{tag}.md"
    ben_cases = sorted(ben, key=lambda r: r["score_gap"], reverse=True)[:3]
    harm_cases = sorted(harm, key=lambda r: r["score_gap"])[:2]

    def _case_block(title, case):
        return [
            f"### {title}: {case['dataset']} split {case['split']} node {case['node_id']}",
            f"- MLP → FINAL_V3: {case['mlp_pred']} → {case['final_pred']} (true={case['true_label']})",
            "- Reliability="
            f"{case['reliability']:.3f}, margin={case['mlp_margin']:.3f}, "
            f"score_gap={case['score_gap']:.3f}",
            f"- Gaps: anchor={case['mlp_anchor_gap']:.3f}, prototype={case['prototype_gap']:.3f}, "
            f"neighbor={case['neighbor_support_gap']:.3f}, compatibility={case['compatibility_gap']:.3f}",
            f"- Dominant reason: **{case['dominant_reason']}** ({case['dominant_gap']:.3f})",
            "- Interpretation: correction shifted to the final class because "
            "its combined structural/feature evidence outweighed the "
            "original class anchor.",
            "",
        ]

    case_lines = [
        f"# FINAL_V3 correction case studies ({tag})",
        "",
        "Small illustrative bundle: 3 beneficial + 2 harmful changed-node examples.",
        "",
    ]
    for idx, c in enumerate(ben_cases, 1):
        case_lines.extend(_case_block(f"Beneficial {idx}", c))
    for idx, c in enumerate(harm_cases, 1):
        case_lines.extend(_case_block(f"Harmful {idx}", c))
    case_md.write_text("\n".join(case_lines), encoding="utf-8")

    fig_path = figures_dir / f"correction_explanation_driver_stacked_{tag}.pdf"
    try:
        import matplotlib.pyplot as plt

        labels = ["beneficial", "harmful"]
        subsets = [ben, harm]
        vals_by_driver = defaultdict(list)
        for subset in subsets:
            cnt = Counter(r["dominant_reason"] for r in subset)
            denom = max(len(subset), 1)
            for d in driver_set:
                vals_by_driver[d].append(cnt[d] / denom)

        x = np.arange(len(labels))
        bottom = np.zeros(len(labels), dtype=np.float64)
        plt.figure(figsize=(7.0, 4.2))
        for d in driver_set:
            vals = np.array(vals_by_driver[d])
            plt.bar(x, vals, bottom=bottom, label=d)
            bottom += vals
        plt.xticks(x, labels)
        plt.ylabel("Share of changed nodes")
        plt.title("FINAL_V3 dominant correction drivers")
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=180)
        plt.close()
    except Exception as exc:
        print(f"[WARN] Could not render figure: {exc}")
        fig_path = None

    return {
        "node_csv": node_csv,
        "summary_csv": summary_csv,
        "summary_md": summary_md,
        "report_md": report_md,
        "case_md": case_md,
        "figure": fig_path,
    }


def main():
    p = argparse.ArgumentParser(description="Run lightweight FINAL_V3 correction explanation audit.")
    p.add_argument("--datasets", nargs="+", default=["cora", "citeseer", "pubmed"], help="Datasets to run")
    p.add_argument("--splits", nargs="+", type=int, default=[0, 1], help="Split IDs to run")
    p.add_argument("--split-dir", default="data/splits", help="Directory containing split .npz files")
    p.add_argument("--tag", default=datetime.now(UTC).strftime("%Y%m%d"), help="Output tag/date")
    args = p.parse_args()

    outputs = run_audit(args.datasets, args.splits, args.split_dir, args.tag)
    print("\nCreated artifacts:")
    for key, path in outputs.items():
        if path is not None:
            print(f"- {key}: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

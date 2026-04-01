#!/usr/bin/env python3
"""
Analyze focused PRL-resubmission runs.

Outputs:
  - tables/prl_resubmission/*.csv
  - reports/prl_resubmission/*.md
  - protocol summary JSON/MD for manuscript rewriting
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

BASELINE_METHODS = [
    "mlp_only",
    "prop_only",
    "gcn",
    "appnp",
    "sgc_wu2019",
    "selective_graph_correction",
    "selective_graph_correction_structural",
    "gated_mlp_prop",
]

ABLATION_METHODS = [
    "selective_graph_correction",
    "sgc_no_gate",
    "sgc_no_feature_similarity",
    "sgc_no_graph_neighbor",
    "sgc_no_compatibility",
    "sgc_mlp_plus_graph",
    "sgc_knn_enabled",
    "selective_graph_correction_structural",
    "sgcs_margin_gate_only",
    "sgcs_no_far",
    "sgcs_far_only",
    "sgcs_no_gate",
]

DISPLAY = {
    "mlp_only": "MLP",
    "prop_only": "Propagation",
    "gcn": "GCN",
    "appnp": "APPNP",
    "sgc_wu2019": "SGC (Wu et al., 2019)",
    "selective_graph_correction": "Selective correction",
    "selective_graph_correction_structural": "UG-SGC-S structural",
    "gated_mlp_prop": "Learned MLP/prop gate",
    "sgc_no_gate": "SGC without gate",
    "sgc_no_feature_similarity": "SGC w/o feature similarity",
    "sgc_no_graph_neighbor": "SGC w/o graph-neighbor support",
    "sgc_no_compatibility": "SGC w/o compatibility",
    "sgc_mlp_plus_graph": "SGC MLP+graph only",
    "sgc_knn_enabled": "SGC with feature-kNN enabled",
    "sgcs_margin_gate_only": "UG-SGC-S without structure-aware gate",
    "sgcs_no_far": "UG-SGC-S near only",
    "sgcs_far_only": "UG-SGC-S far only",
    "sgcs_no_gate": "UG-SGC-S without gate",
}


def _load_records(path: str) -> List[Dict[str, Any]]:
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def _write_csv(path: str, header: List[str], rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(h, "")) for h in header) + "\n")


def _pairwise_counts(recs_a: List[Dict[str, Any]], recs_b: List[Dict[str, Any]]) -> Tuple[int, int, int]:
    map_a = {(r["dataset"], r["split_id"], r["repeat_id"]): float(r["test_acc"]) for r in recs_a if r.get("success")}
    map_b = {(r["dataset"], r["split_id"], r["repeat_id"]): float(r["test_acc"]) for r in recs_b if r.get("success")}
    common = set(map_a) & set(map_b)
    wins = sum(1 for k in common if map_a[k] > map_b[k] + 1e-6)
    losses = sum(1 for k in common if map_a[k] < map_b[k] - 1e-6)
    ties = len(common) - wins - losses
    return wins, losses, ties


def analyze(runs_path: str, output_tag: str) -> Dict[str, str]:
    records = _load_records(runs_path)
    ok = [r for r in records if r.get("success", False)]
    by_ds_method: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    all_methods = sorted({r["method"] for r in ok})
    datasets = sorted({r["dataset"] for r in ok})

    for r in ok:
        by_ds_method[r["dataset"]][r["method"]].append(r)

    tables_dir = os.path.join(REPO_ROOT, "tables", "prl_resubmission")
    reports_dir = os.path.join(REPO_ROOT, "reports", "prl_resubmission")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # 1) Protocol summary
    protocol = {
        "output_tag": output_tag,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runs_file": runs_path,
        "datasets": datasets,
        "methods": all_methods,
        "n_records_total": len(records),
        "n_records_successful": len(ok),
        "splits": sorted({int(r["split_id"]) for r in ok}),
        "repeats": sorted({int(r["repeat_id"]) for r in ok}),
        "parameter_sources": sorted({str(r.get("parameter_source")) for r in ok}),
    }
    protocol_json = os.path.join(reports_dir, f"protocol_{output_tag}.json")
    with open(protocol_json, "w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2)

    protocol_md = os.path.join(reports_dir, f"protocol_{output_tag}.md")
    with open(protocol_md, "w", encoding="utf-8") as f:
        f.write("# Protocol summary\n\n")
        f.write(f"- output_tag: `{output_tag}`\n")
        f.write(f"- runs_file: `{runs_path}`\n")
        f.write(f"- datasets: `{', '.join(datasets)}`\n")
        f.write(f"- methods: `{', '.join(all_methods)}`\n")
        f.write(f"- successful records: `{len(ok)}` / `{len(records)}`\n")
        f.write(f"- splits: `{', '.join(map(str, protocol['splits']))}`\n")
        f.write(f"- repeats: `{', '.join(map(str, protocol['repeats']))}`\n")
        f.write("- selective correction and all ablations use validation-only threshold/weight selection.\n")
        f.write("- GCN and APPNP use compact validation-based config grids with early stopping.\n")

    # 2) Baseline comparison table
    baseline_rows: List[Dict[str, Any]] = []
    for ds in datasets:
        mlp_recs = by_ds_method[ds].get("mlp_only", [])
        mlp_mean = float(np.mean([r["test_acc"] for r in mlp_recs])) if mlp_recs else float("nan")
        for method in BASELINE_METHODS:
            recs = by_ds_method[ds].get(method, [])
            if not recs:
                continue
            mean = float(np.mean([r["test_acc"] for r in recs]))
            std = float(np.std([r["test_acc"] for r in recs]))
            wins, losses, ties = _pairwise_counts(recs, mlp_recs) if mlp_recs else (0, 0, 0)
            baseline_rows.append(
                {
                    "dataset": ds,
                    "method": method,
                    "display_name": DISPLAY.get(method, method),
                    "mean_test_acc": f"{mean:.4f}",
                    "std_test_acc": f"{std:.4f}",
                    "delta_vs_mlp": f"{(mean - mlp_mean):+.4f}" if not np.isnan(mlp_mean) else "",
                    "wins_vs_mlp": wins,
                    "losses_vs_mlp": losses,
                    "ties_vs_mlp": ties,
                    "n_runs": len(recs),
                }
            )
    baseline_csv = os.path.join(tables_dir, f"baseline_comparison_{output_tag}.csv")
    _write_csv(
        baseline_csv,
        ["dataset", "method", "display_name", "mean_test_acc", "std_test_acc", "delta_vs_mlp", "wins_vs_mlp", "losses_vs_mlp", "ties_vs_mlp", "n_runs"],
        baseline_rows,
    )

    # 3) Ablation comparison vs default SGC
    ablation_rows: List[Dict[str, Any]] = []
    for ds in datasets:
        ref_recs = by_ds_method[ds].get("selective_graph_correction", [])
        ref_mean = float(np.mean([r["test_acc"] for r in ref_recs])) if ref_recs else float("nan")
        for method in ABLATION_METHODS:
            recs = by_ds_method[ds].get(method, [])
            if not recs:
                continue
            mean = float(np.mean([r["test_acc"] for r in recs]))
            std = float(np.std([r["test_acc"] for r in recs]))
            wins, losses, ties = _pairwise_counts(recs, ref_recs) if ref_recs else (0, 0, 0)
            ablation_rows.append(
                {
                    "dataset": ds,
                    "method": method,
                    "display_name": DISPLAY.get(method, method),
                    "mean_test_acc": f"{mean:.4f}",
                    "std_test_acc": f"{std:.4f}",
                    "delta_vs_default_sgc": f"{(mean - ref_mean):+.4f}" if not np.isnan(ref_mean) else "",
                    "wins_vs_default_sgc": wins,
                    "losses_vs_default_sgc": losses,
                    "ties_vs_default_sgc": ties,
                    "n_runs": len(recs),
                }
            )
    ablation_csv = os.path.join(tables_dir, f"ablation_comparison_{output_tag}.csv")
    _write_csv(
        ablation_csv,
        ["dataset", "method", "display_name", "mean_test_acc", "std_test_acc", "delta_vs_default_sgc", "wins_vs_default_sgc", "losses_vs_default_sgc", "ties_vs_default_sgc", "n_runs"],
        ablation_rows,
    )

    # 4) Method specification / claims-facing markdown
    summary_md = os.path.join(reports_dir, f"summary_{output_tag}.md")
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# PRL resubmission experiment summary\n\n")
        f.write("## Datasets\n\n")
        for ds in datasets:
            f.write(f"- `{ds}`\n")
        f.write("\n## Baseline package\n\n")
        f.write("- Existing baselines retained: `mlp_only`, `prop_only`.\n")
        f.write("- Newly added standard graph baselines: `gcn`, `appnp`.\n")
        f.write("- Alternative gate baseline: `gated_mlp_prop`.\n")
        f.write("\n## Ablations\n\n")
        for method in ABLATION_METHODS[1:]:
            f.write(f"- `{method}` = {DISPLAY.get(method, method)}\n")
        f.write("\n## Main caveat\n\n")
        f.write("- This package strengthens baselines and ablations, but it does not convert the paper into a universal multi-regime claim.\n")

    claims_md = os.path.join(reports_dir, f"claims_and_risks_{output_tag}.md")
    with open(claims_md, "w", encoding="utf-8") as f:
        f.write("# Claims and risks for PRL rewrite\n\n")
        f.write("## Supportable if results are favorable\n\n")
        f.write("- Selective graph correction can be compared against a small but standard graph-baseline set.\n")
        f.write("- The uncertainty gate and score terms can now be examined with real controlled variants.\n")
        f.write("- The method remains feature-first and only uses graph information selectively.\n")
        f.write("\n## Still avoid\n\n")
        f.write("- Universal robustness claims.\n")
        f.write("- Overinterpreting descriptive gate diagnostics as causal without the ablation table.\n")

    return {
        "protocol_json": protocol_json,
        "protocol_md": protocol_md,
        "baseline_csv": baseline_csv,
        "ablation_csv": ablation_csv,
        "summary_md": summary_md,
        "claims_md": claims_md,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze PRL resubmission runs.")
    parser.add_argument("--runs-file", required=True)
    parser.add_argument("--output-tag", required=True)
    args = parser.parse_args()
    out = analyze(args.runs_file, args.output_tag)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

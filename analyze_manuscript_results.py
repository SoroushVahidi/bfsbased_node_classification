#!/usr/bin/env python3
"""
analyze_manuscript_results.py

Standalone post-processing script.
Reads existing JSONL run files and regenerates all 5 manuscript analysis tables.

Usage:
    python analyze_manuscript_results.py \
        --runs-file logs/comparison_runs_manuscript_phase.jsonl \
        [--output-tag manuscript_phase]
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np


# -----------------------------------------------------------------------
# Table generation (mirrors manuscript_runner._generate_summary but
# runs standalone without importing torch or the BFS module)
# -----------------------------------------------------------------------

def load_records(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_csv(path: str, header: List[str], rows: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(h, "")) for h in header) + "\n")


def analyze(runs_path: str, output_tag: str, diagnostics_dir: str = "logs"):
    """Main analysis entry point."""
    import time

    if not os.path.isfile(runs_path):
        print(f"[ERROR] Runs file not found: {runs_path}")
        return

    records = load_records(runs_path)
    print(f"Loaded {len(records)} records from {runs_path}")

    # Group successful records
    by_ds_method: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    by_ds_method_rec: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    by_ds_method_rt: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for rec in records:
        if not rec.get("success", False):
            continue
        ds = rec["dataset"]
        method = rec["method"]
        acc = rec.get("test_acc")
        if acc is not None:
            by_ds_method[ds][method].append(float(acc))
            by_ds_method_rec[ds][method].append(rec)
            by_ds_method_rt[ds][method].append(float(rec.get("total_runtime_sec", 0.0)))

    methods_all = ["mlp_only", "prop_only", "selective_graph_correction", "prop_mlp_select"]
    os.makedirs(diagnostics_dir, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")

    # ------------------------------------------------------------------
    # 1. Core result table (print + JSON)
    # ------------------------------------------------------------------
    summary_path = os.path.join(diagnostics_dir, f"comparison_summary_{output_tag}.json")
    summary: Dict[str, Any] = {
        "output_tag": output_tag,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "datasets": {},
    }

    print("\n" + "=" * 80)
    print(f"CORE RESULT TABLE — {output_tag}")
    print("=" * 80)
    col_w = 26
    print(f"{'Dataset':<14}", end="")
    for m in methods_all:
        short = {"selective_graph_correction": "sgc", "prop_mlp_select": "pms"}.get(m, m)
        print(f"{short:>{col_w}}", end="")
    print()
    print("-" * (14 + col_w * len(methods_all)))

    for ds in sorted(by_ds_method.keys()):
        ds_entry = {}
        row_parts = [f"{ds:<14}"]
        for method in methods_all:
            accs = by_ds_method[ds].get(method, [])
            rts = by_ds_method_rt[ds].get(method, [])
            if accs:
                ds_entry[method] = {
                    "mean": float(np.mean(accs)),
                    "std": float(np.std(accs)),
                    "n": len(accs),
                    "runs": accs,
                    "mean_runtime_sec": float(np.mean(rts)) if rts else 0.0,
                }
                row_parts.append(f"{np.mean(accs):.4f}±{np.std(accs):.4f}".rjust(col_w))
            else:
                ds_entry[method] = None
                row_parts.append("N/A".rjust(col_w))
        summary["datasets"][ds] = ds_entry
        print("".join(row_parts))

    print("=" * 80)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_path}")

    # ------------------------------------------------------------------
    # 2. Gain-over-MLP table
    # ------------------------------------------------------------------
    gain_path = os.path.join(diagnostics_dir, f"gain_over_mlp_{output_tag}.csv")
    gain_header = [
        "dataset", "sgc_mean", "mlp_mean", "delta",
        "n_sgc_wins", "n_mlp_wins", "n_ties",
        "avg_uncertain_frac", "avg_changed_frac",
    ]
    gain_rows = []

    print("\n=== Gain Over MLP (SGC - MLP) ===")
    for ds in sorted(by_ds_method.keys()):
        mlp_accs = by_ds_method[ds].get("mlp_only", [])
        sgc_accs = by_ds_method[ds].get("selective_graph_correction", [])
        sgc_recs = by_ds_method_rec[ds].get("selective_graph_correction", [])
        if not mlp_accs or not sgc_accs:
            continue

        sgc_m = float(np.mean(sgc_accs))
        mlp_m = float(np.mean(mlp_accs))
        delta = sgc_m - mlp_m

        # Per-run win/loss/tie
        mlp_map = {(r["split_id"], r["repeat_id"]): r.get("test_acc")
                   for r in by_ds_method_rec[ds].get("mlp_only", [])}
        sgc_map = {(r["split_id"], r["repeat_id"]): r.get("test_acc") for r in sgc_recs}
        common = set(mlp_map) & set(sgc_map)
        wins = sum(1 for k in common if sgc_map[k] > mlp_map[k] + 1e-6)
        losses = sum(1 for k in common if sgc_map[k] < mlp_map[k] - 1e-6)
        ties = len(common) - wins - losses

        unc_fracs = []
        chg_fracs = []
        for r in sgc_recs:
            nt = r.get("test_count", 1) or 1
            nu = r.get("sgc_n_uncertain") or 0
            nc = r.get("sgc_n_changed") or 0
            unc_fracs.append(nu / nt)
            chg_fracs.append(nc / max(nu, 1))

        sign = "+" if delta > 0.005 else ("-" if delta < -0.005 else "~")
        print(f"  {ds:<12}: delta={delta:+.4f} {sign}  wins={wins} losses={losses} ties={ties}")

        gain_rows.append({
            "dataset": ds,
            "sgc_mean": f"{sgc_m:.4f}",
            "mlp_mean": f"{mlp_m:.4f}",
            "delta": f"{delta:+.4f}",
            "n_sgc_wins": wins,
            "n_mlp_wins": losses,
            "n_ties": ties,
            "avg_uncertain_frac": f"{np.mean(unc_fracs):.3f}" if unc_fracs else "N/A",
            "avg_changed_frac": f"{np.mean(chg_fracs):.3f}" if chg_fracs else "N/A",
        })

    _write_csv(gain_path, gain_header, gain_rows)
    print(f"Gain-over-MLP table saved: {gain_path}")

    # ------------------------------------------------------------------
    # 3. Correction-behavior table
    # ------------------------------------------------------------------
    cb_path = os.path.join(diagnostics_dir, f"correction_behavior_{output_tag}.csv")
    cb_header = [
        "dataset", "n_runs",
        "avg_threshold", "avg_b1", "avg_b2", "avg_b3", "avg_b4",
        "avg_uncertain_frac", "avg_changed_of_unc",
        "zero_uncertain_runs", "zero_changed_runs",
    ]
    cb_rows = []

    for ds in sorted(by_ds_method_rec.keys()):
        sgc_recs = by_ds_method_rec[ds].get("selective_graph_correction", [])
        if not sgc_recs:
            continue
        thresholds = [r["sgc_threshold"] for r in sgc_recs if r.get("sgc_threshold") is not None]
        bw = [r["sgc_b_weights"] for r in sgc_recs if r.get("sgc_b_weights") is not None]
        unc_f, chg_f = [], []
        zero_u = zero_c = 0
        for r in sgc_recs:
            nt = r.get("test_count", 1) or 1
            nu = r.get("sgc_n_uncertain") or 0
            nc = r.get("sgc_n_changed") or 0
            unc_f.append(nu / nt)
            chg_f.append(nc / max(nu, 1))
            if nu == 0:
                zero_u += 1
            if nc == 0:
                zero_c += 1
        cb_rows.append({
            "dataset": ds,
            "n_runs": len(sgc_recs),
            "avg_threshold": f"{np.mean(thresholds):.3f}" if thresholds else "N/A",
            "avg_b1": f"{np.mean([w[0] for w in bw]):.3f}" if bw else "N/A",
            "avg_b2": f"{np.mean([w[1] for w in bw]):.3f}" if bw else "N/A",
            "avg_b3": f"{np.mean([w[2] for w in bw]):.3f}" if bw else "N/A",
            "avg_b4": f"{np.mean([w[3] for w in bw]):.3f}" if bw else "N/A",
            "avg_uncertain_frac": f"{np.mean(unc_f):.3f}",
            "avg_changed_of_unc": f"{np.mean(chg_f):.3f}",
            "zero_uncertain_runs": zero_u,
            "zero_changed_runs": zero_c,
        })
    _write_csv(cb_path, cb_header, cb_rows)
    print(f"Correction-behavior table saved: {cb_path}")

    # ------------------------------------------------------------------
    # 4. Regime-analysis table
    # ------------------------------------------------------------------
    regime_path = os.path.join(diagnostics_dir, f"regime_analysis_{output_tag}.csv")
    regime_header = [
        "dataset",
        "mlp_mean", "mlp_std",
        "prop_mean", "prop_std",
        "sgc_mean", "sgc_std",
        "sgc_delta",
        "sgc_better_than_prop",
    ]
    regime_rows = []
    for ds in sorted(by_ds_method.keys()):
        mlp_accs = by_ds_method[ds].get("mlp_only", [])
        prop_accs = by_ds_method[ds].get("prop_only", [])
        sgc_accs = by_ds_method[ds].get("selective_graph_correction", [])
        mlp_m = float(np.mean(mlp_accs)) if mlp_accs else float("nan")
        prop_m = float(np.mean(prop_accs)) if prop_accs else float("nan")
        sgc_m = float(np.mean(sgc_accs)) if sgc_accs else float("nan")
        delta = sgc_m - mlp_m
        regime_rows.append({
            "dataset": ds,
            "mlp_mean": f"{mlp_m:.4f}" if not np.isnan(mlp_m) else "N/A",
            "mlp_std": f"{float(np.std(mlp_accs)):.4f}" if mlp_accs else "N/A",
            "prop_mean": f"{prop_m:.4f}" if not np.isnan(prop_m) else "N/A",
            "prop_std": f"{float(np.std(prop_accs)):.4f}" if prop_accs else "N/A",
            "sgc_mean": f"{sgc_m:.4f}" if not np.isnan(sgc_m) else "N/A",
            "sgc_std": f"{float(np.std(sgc_accs)):.4f}" if sgc_accs else "N/A",
            "sgc_delta": f"{delta:+.4f}" if not (np.isnan(delta)) else "N/A",
            "sgc_better_than_prop": (
                "yes" if (not np.isnan(sgc_m) and not np.isnan(prop_m) and sgc_m > prop_m + 1e-4)
                else ("no" if (not np.isnan(sgc_m) and not np.isnan(prop_m) and sgc_m < prop_m - 1e-4)
                      else "tie")
            ),
        })
    _write_csv(regime_path, regime_header, regime_rows)
    print(f"Regime-analysis table saved: {regime_path}")

    # ------------------------------------------------------------------
    # 5. Case studies from diagnostics JSON
    # ------------------------------------------------------------------
    cs_path = os.path.join(diagnostics_dir, f"case_studies_{output_tag}.md")
    _write_case_studies(by_ds_method_rec, diagnostics_dir, cs_path)
    print(f"Case studies saved: {cs_path}")

    # ------------------------------------------------------------------
    # Fairness report (Markdown)
    # ------------------------------------------------------------------
    fairness_path = os.path.join(diagnostics_dir, f"comparison_fairness_report_{output_tag}.md")
    _write_fairness_report(by_ds_method, gain_rows, methods_all, fairness_path, output_tag)
    print(f"Fairness report saved: {fairness_path}")

    # ------------------------------------------------------------------
    # Manuscript readiness assessment
    # ------------------------------------------------------------------
    _manuscript_readiness_assessment(by_ds_method, gain_rows)


def _write_case_studies(
    by_ds_method_rec: Dict,
    diagnostics_dir: str,
    output_path: str,
):
    lines = [
        "# Case Studies — Selective Graph Correction\n\n",
        "Per dataset, corrected test nodes from split 0, repeat 0.\n\n",
    ]

    for ds in sorted(by_ds_method_rec.keys()):
        sgc_recs = by_ds_method_rec[ds].get("selective_graph_correction", [])
        if not sgc_recs:
            continue
        # Use split_id=0, repeat=0 if available
        target = next(
            (r for r in sgc_recs if r.get("split_id") == 0 and r.get("repeat_id") == 0),
            sgc_recs[0],
        )
        run_id = f"{ds}_s{target['split_id']}_r{target['repeat_id']}"
        diag_path = os.path.join(diagnostics_dir, f"sgc_{run_id}_diagnostics.json")

        lines.append(f"## {ds.upper()}\n\n")

        if os.path.isfile(diag_path):
            with open(diag_path, encoding="utf-8") as df:
                diag = json.load(df)
            node_diags = diag.get("node_diagnostics", [])
            changed = [nd for nd in node_diags if nd.get("changed")]

            lines.append(
                f"threshold={diag['threshold_high']:.2f}  "
                f"b=({diag['b1']:.2f},{diag['b2']:.2f},{diag['b3']:.2f},{diag['b4']:.2f})  "
                f"test_acc={diag['test_acc']:.4f}\n\n"
            )
            lines.append(
                f"Uncertain nodes: {diag['n_uncertain']}/{diag['n_test']}  "
                f"Changed: {diag['n_changed']}\n\n"
            )

            if changed:
                lines.append(
                    "| node_id | mlp | corrected | true | mlp✓ | sgc✓ | margin | dom_term |\n"
                )
                lines.append(
                    "|---------|-----|-----------|------|------|------|--------|----------|\n"
                )
                for nd in changed[:15]:
                    comps = nd.get("components", {})
                    dom = max(comps, key=lambda k: abs(float(comps[k]))) if comps else "?"
                    lines.append(
                        f"| {nd['node_id']} | {nd['mlp_pred']} | {nd['corrected_pred']} | "
                        f"{nd['true_label']} | {'✓' if nd['mlp_correct'] else '✗'} | "
                        f"{'✓' if nd['corrected_correct'] else '✗'} | "
                        f"{nd['mlp_margin']:.3f} | {dom} |\n"
                    )
                lines.append("\n")

                # Correction utility analysis
                improvements = sum(1 for nd in changed if nd["corrected_correct"] and not nd["mlp_correct"])
                degradations = sum(1 for nd in changed if not nd["corrected_correct"] and nd["mlp_correct"])
                neutral = len(changed) - improvements - degradations
                lines.append(
                    f"Correction utility: {improvements} improved, "
                    f"{degradations} degraded, {neutral} neutral\n\n"
                )
            else:
                lines.append("_No changed nodes in this run._\n\n")
        else:
            lines.append(f"_Diagnostics file not found: {diag_path}_\n\n")
            lines.append(
                f"Run-level: test_acc={target.get('test_acc', 'N/A')}  "
                f"uncertain={target.get('sgc_n_uncertain', 'N/A')}  "
                f"changed={target.get('sgc_n_changed', 'N/A')}\n\n"
            )

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def _write_fairness_report(by_ds_method, gain_rows, methods_all, path, output_tag):
    import time
    lines = [
        f"# Manuscript Fairness Report — {output_tag}\n",
        f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%S')}\n\n",
        "## Core Result Table\n\n",
        "| Dataset | mlp_only | prop_only | selective_graph_correction | prop_mlp_select |\n",
        "|---------|----------|-----------|---------------------------|-----------------|\n",
    ]
    for ds in sorted(by_ds_method.keys()):
        row = [ds]
        for method in methods_all:
            accs = by_ds_method[ds].get(method, [])
            if accs:
                row.append(f"{np.mean(accs):.4f}±{np.std(accs):.4f}")
            else:
                row.append("N/A")
        lines.append("| " + " | ".join(row) + " |\n")

    lines.append("\n## Gain Over MLP\n\n")
    lines.append("| Dataset | Delta | SGC wins | MLP wins | Ties |\n")
    lines.append("|---------|-------|----------|----------|------|\n")
    for r in gain_rows:
        lines.append(
            f"| {r['dataset']} | {r['delta']} | {r['n_sgc_wins']} | "
            f"{r['n_mlp_wins']} | {r['n_ties']} |\n"
        )

    lines.append(
        "\n## Fairness Notes\n\n"
        "- Canonical GEO-GCN splits only (no random fallback).\n"
        "- MLP uses early stopping on val set.\n"
        "- SGC threshold and b-weights tuned on val set; test set not seen during tuning.\n"
        "- prop_only params tuned via robust_random_search on train k-fold only.\n"
        "- prop_mlp_select: val accuracy selects between MLP and prop_only (no test leakage).\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def _manuscript_readiness_assessment(by_ds_method, gain_rows):
    print("\n" + "=" * 70)
    print("MANUSCRIPT READINESS ASSESSMENT")
    print("=" * 70)

    # Classify datasets
    positive = [r for r in gain_rows if float(r["delta"]) > 0.005]
    neutral = [r for r in gain_rows if -0.005 <= float(r["delta"]) <= 0.005]
    negative = [r for r in gain_rows if float(r["delta"]) < -0.005]

    print(f"\n1. Datasets where SGC > MLP by >0.5%: {[r['dataset'] for r in positive]}")
    print(f"   Datasets neutral (±0.5%):            {[r['dataset'] for r in neutral]}")
    print(f"   Datasets where SGC < MLP by >0.5%:  {[r['dataset'] for r in negative]}")

    # Determine paper framing
    n_pos = len(positive)
    n_neg = len(negative)
    n_neu = len(neutral)
    n_total = len(gain_rows)

    print(f"\n2. Summary: {n_pos} positive / {n_neu} neutral / {n_neg} negative out of {n_total} datasets")

    if n_total == 0:
        print("\nNo data available yet. Run the experiment first.")
        return

    frac_positive = n_pos / n_total
    frac_negative = n_neg / n_total

    if frac_positive >= 0.6:
        framing = "positive-method paper"
        claim = (
            "SGC provides consistent, meaningful improvements over MLP-only across most datasets. "
            "The method is particularly effective when MLP predictions are uncertain."
        )
        not_supported = "Strong claims for specific heterophilic datasets without stability analysis."
    elif n_pos > 0 and n_neg > 0:
        framing = "regime-analysis paper"
        claim = (
            "SGC selectively improves on MLP in specific graph regimes. "
            "The method is most effective on [insert datasets]. "
            "The paper documents when graph evidence is helpful vs. redundant."
        )
        not_supported = "Universal improvement claim across all graph types."
    else:
        framing = "diagnostic/negative paper"
        claim = (
            "Selective graph correction via BFS-based propagation provides minimal or no improvement "
            "over strong MLP baselines on most benchmark datasets. "
            "The paper contributes a diagnostic analysis of when graph structure helps."
        )
        not_supported = "Positive-method claim."

    print(f"\n3. Paper framing most supported: **{framing}**")
    print(f"\n4. Strongest supported claim:\n   {claim}")
    print(f"\n5. Claim to AVOID:\n   {not_supported}")
    print(
        "\n6. Minimum next results needed before drafting seriously:\n"
        "   - repeats=3 on Wulver for stability estimates (current: repeats=1)\n"
        "   - Per-node diagnostics for at least 2 datasets (case studies)\n"
        "   - Regime correlation analysis (SGC delta vs. homophily, avg degree)\n"
        "   - Ablation: b2=0 (no feature sim) and b3=0 (no graph) to isolate gains\n"
    )
    print(
        "7. Recommended paper angle:\n"
        f"   {framing}. Focus on '{claim[:80]}...'\n"
    )
    print("=" * 70)


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Standalone analysis: regenerate manuscript tables from existing JSONL runs."
    )
    parser.add_argument(
        "--runs-file",
        default=os.path.join("logs", "comparison_runs_manuscript_phase.jsonl"),
        help="Path to the JSONL runs file produced by manuscript_runner.py.",
    )
    parser.add_argument(
        "--output-tag",
        default="manuscript_phase",
        help="Tag for output file names.",
    )
    parser.add_argument(
        "--diagnostics-dir",
        default="logs",
        help="Directory containing diagnostics JSON files and where outputs are written.",
    )
    args = parser.parse_args()
    analyze(args.runs_file, args.output_tag, args.diagnostics_dir)


if __name__ == "__main__":
    main()

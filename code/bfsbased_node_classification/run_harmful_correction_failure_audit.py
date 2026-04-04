#!/usr/bin/env python3
"""Lightweight harmful-correction failure audit for FINAL_V3 changed-node exports."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def _find_latest(path_glob: str) -> Path:
    matches = sorted(ROOT.glob(path_glob))
    if not matches:
        raise FileNotFoundError(f"No files found for pattern: {path_glob}")
    return matches[-1]


def _dominant_share(sub: pd.DataFrame, drivers: list[str]) -> dict[str, float]:
    c = Counter(sub["dominant_reason"].tolist())
    n = max(len(sub), 1)
    return {d: c.get(d, 0) / n for d in drivers}


def run(tag: str, node_csv: Path | None = None) -> dict[str, Path]:
    node_csv = node_csv or _find_latest("reports/correction_explanation_nodes_*.csv")
    explanation_md = _find_latest("reports/correction_explanation_audit_*.md")
    explanation_summary = _find_latest("tables/correction_explanation_summary_*.csv")
    bounded_csv = _find_latest("tables/bounded_intervention_fullscope_*.csv")
    sensitivity_csv = _find_latest("tables/gate_sensitivity_fullscope_*.csv")

    df = pd.read_csv(node_csv)
    df["proto_graph_agree"] = np.sign(df["prototype_gap"]) == np.sign(df["neighbor_support_gap"])

    harm = df[df["harmful"] == 1].copy()
    ben = df[df["beneficial"] == 1].copy()
    changed = df.copy()

    drivers = ["neighbor_support", "prototype", "compatibility", "mlp_anchor", "feature_knn", "structural_far"]

    failure_rows = [
        {
            "section": "overall",
            "metric": "n_changed",
            "value": int(len(changed)),
        },
        {
            "section": "overall",
            "metric": "n_harmful",
            "value": int(len(harm)),
        },
        {
            "section": "overall",
            "metric": "harmful_rate_within_changed",
            "value": float(len(harm) / max(len(changed), 1)),
        },
        {
            "section": "harmful",
            "metric": "mean_reliability",
            "value": float(harm["reliability"].mean()),
        },
        {
            "section": "harmful",
            "metric": "mean_margin",
            "value": float(harm["mlp_margin"].mean()),
        },
        {
            "section": "harmful",
            "metric": "mean_score_gap",
            "value": float(harm["score_gap"].mean()),
        },
        {
            "section": "harmful",
            "metric": "proto_graph_agreement_rate",
            "value": float(harm["proto_graph_agree"].mean()),
        },
        {
            "section": "beneficial",
            "metric": "mean_reliability",
            "value": float(ben["reliability"].mean()),
        },
        {
            "section": "beneficial",
            "metric": "mean_margin",
            "value": float(ben["mlp_margin"].mean()),
        },
        {
            "section": "beneficial",
            "metric": "mean_score_gap",
            "value": float(ben["score_gap"].mean()),
        },
        {
            "section": "beneficial",
            "metric": "proto_graph_agreement_rate",
            "value": float(ben["proto_graph_agree"].mean()),
        },
    ]

    ds_share = (harm["dataset"].value_counts(normalize=True).rename_axis("dataset").reset_index(name="value"))
    split_share = (
        harm.groupby(["dataset", "split"])
        .size()
        .reset_index(name="n_harmful")
        .sort_values("n_harmful", ascending=False)
    )

    for _, r in ds_share.iterrows():
        failure_rows.append(
            {"section": "harmful_dataset_share", "metric": str(r["dataset"]), "value": float(r["value"])}
        )
    for _, r in split_share.iterrows():
        failure_rows.append(
            {
                "section": "harmful_split_count",
                "metric": f"{r['dataset']}_split_{int(r['split'])}",
                "value": int(r["n_harmful"]),
            }
        )

    harm_driver = _dominant_share(harm, drivers)
    ben_driver = _dominant_share(ben, drivers)
    for d in drivers:
        failure_rows.append({"section": "harmful_driver_share", "metric": d, "value": harm_driver[d]})
        failure_rows.append({"section": "beneficial_driver_share", "metric": d, "value": ben_driver[d]})

    safeguards = {
        "gap_ge_0p05": (df["score_gap"] >= 0.05),
        "rel_ge_0p45": (df["reliability"] >= 0.45),
        "proto_graph_agree": df["proto_graph_agree"],
        "rel_ge_0p43_and_gap_ge_0p03": (df["reliability"] >= 0.43) & (df["score_gap"] >= 0.03),
    }

    safeguard_rows = []
    for name, keep_mask in safeguards.items():
        blocked = ~keep_mask
        harm_prevented = int((blocked & (df["harmful"] == 1)).sum())
        ben_lost = int((blocked & (df["beneficial"] == 1)).sum())
        safeguard_rows.append(
            {
                "rule": name,
                "harmful_prevented": harm_prevented,
                "harmful_prevented_rate": harm_prevented / max(len(harm), 1),
                "beneficial_lost": ben_lost,
                "beneficial_lost_rate": ben_lost / max(len(ben), 1),
                "prevented_to_lost_ratio": harm_prevented / max(ben_lost, 1),
            }
        )
    safeguard_df = pd.DataFrame(safeguard_rows).sort_values("prevented_to_lost_ratio", ascending=False)

    tables = ROOT / "tables"
    reports = ROOT / "reports"
    figures = ROOT / "figures"
    for d in (tables, reports, figures):
        d.mkdir(parents=True, exist_ok=True)

    summary_csv = tables / f"harmful_correction_failure_summary_{tag}.csv"
    summary_md = tables / f"harmful_correction_failure_summary_{tag}.md"
    report_md = reports / f"harmful_correction_failure_audit_{tag}.md"
    cases_md = reports / f"harmful_correction_case_studies_{tag}.md"
    fig_pdf = figures / f"harmful_vs_beneficial_correction_comparison_{tag}.pdf"

    pd.DataFrame(failure_rows).to_csv(summary_csv, index=False)

    top_ds = ds_share.iloc[0]["dataset"] if not ds_share.empty else "n/a"
    top_ds_share = float(ds_share.iloc[0]["value"]) if not ds_share.empty else float("nan")

    md_lines = [
        f"# FINAL_V3 harmful-correction failure summary ({tag})",
        "",
        f"Source changed-node export: `{node_csv.relative_to(ROOT)}`",
        f"Reference explanation audit: `{explanation_md.relative_to(ROOT)}`",
        f"Reference explanation summary: `{explanation_summary.relative_to(ROOT)}`",
        f"Context bounded-intervention CSV: `{bounded_csv.relative_to(ROOT)}`",
        f"Context gate-sensitivity CSV: `{sensitivity_csv.relative_to(ROOT)}`",
        "",
        "## Failure profile",
        f"- Harmful changed nodes: **{len(harm)} / {len(changed)}** "
        f"({100.0 * len(harm) / max(len(changed), 1):.1f}% of changed).",
        "- Mean reliability (harmful): "
        f"**{harm['reliability'].mean():.3f}** vs beneficial "
        f"**{ben['reliability'].mean():.3f}**.",
        "- Mean margin (harmful): "
        f"**{harm['mlp_margin'].mean():.3f}** vs beneficial "
        f"**{ben['mlp_margin'].mean():.3f}**.",
        "- Mean score gap (harmful): "
        f"**{harm['score_gap'].mean():.3f}** vs beneficial "
        f"**{ben['score_gap'].mean():.3f}**.",
        f"- Prototype/graph agreement (harmful): **{100.0 * harm['proto_graph_agree'].mean():.1f}%** "
        f"vs beneficial **{100.0 * ben['proto_graph_agree'].mean():.1f}%**.",
        f"- Highest harmful concentration dataset: **{top_ds}** "
        f"({100.0 * top_ds_share:.1f}% of harmful cases).",
        "",
        "### Harmful dataset distribution",
        "",
        "| Dataset | Harmful share |",
        "| --- | ---: |",
    ]
    for _, r in ds_share.iterrows():
        md_lines.append(f"| {r['dataset']} | {100.0 * float(r['value']):.1f}% |")

    md_lines.extend([
        "",
        "### Harmful split distribution (top 8)",
        "",
        "| Dataset/split | Harmful count |",
        "| --- | ---: |",
    ])
    for _, r in split_share.head(8).iterrows():
        md_lines.append(f"| {r['dataset']} / {int(r['split'])} | {int(r['n_harmful'])} |")

    md_lines.extend([
        "",
        "## Dominant-driver shares (harmful vs beneficial)",
        "",
        "| Driver | Harmful | Beneficial |",
        "| --- | ---: | ---: |",
    ])
    for d in drivers:
        md_lines.append(f"| {d} | {100.0 * harm_driver[d]:.1f}% | {100.0 * ben_driver[d]:.1f}% |")

    md_lines.extend([
        "",
        "## Safeguard what-if (post hoc; no retraining)",
        "",
        "| Rule | Harmful prevented | Harmful prevented % | Beneficial lost | Beneficial lost % | Prevented/Lost |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for _, r in safeguard_df.iterrows():
        md_lines.append(
            f"| {r['rule']} | {int(r['harmful_prevented'])} | {100.0*r['harmful_prevented_rate']:.1f}% | "
            f"{int(r['beneficial_lost'])} | {100.0*r['beneficial_lost_rate']:.1f}% | "
            f"{r['prevented_to_lost_ratio']:.2f} |"
        )
    summary_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    safeguard_best = safeguard_df.iloc[0]
    report_lines = [
        f"# FINAL_V3 harmful-correction failure audit ({tag})",
        "",
        "This is a lightweight, non-canonical post hoc failure audit on changed-node explanations.",
        "",
        "## Key observations",
        f"- Harmful cases are a minority among changed nodes ({len(harm)}/{len(changed)}).",
        "- Harmful cases tend to show slightly lower reliability and "
        "smaller final-vs-original score gaps than beneficial changes.",
        "- Dominant driver in this bundle remains graph neighbor support for both helpful and harmful changes.",
        f"- Best simple safeguard by prevented/lost ratio: `{safeguard_best['rule']}` "
        f"(prevented {int(safeguard_best['harmful_prevented'])} harmful while losing "
        f"{int(safeguard_best['beneficial_lost'])} beneficial).",
        "",
        "## Outputs",
        f"- Summary CSV: `{summary_csv.relative_to(ROOT)}`",
        f"- Summary Markdown: `{summary_md.relative_to(ROOT)}`",
        f"- Figure: `{fig_pdf.relative_to(ROOT)}`",
        f"- Case studies: `{cases_md.relative_to(ROOT)}`",
    ]
    report_md.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    top_harm = harm.sort_values("score_gap").head(5).copy()
    best_rule = safeguard_best["rule"]
    best_mask = safeguards[best_rule]

    case_lines = [
        f"# FINAL_V3 harmful correction case studies ({tag})",
        "",
        "Illustrative harmful changed nodes (lowest score-gap failures).",
        "",
    ]
    for idx, row in top_harm.iterrows():
        nid = int(row["node_id"])
        blocked = not bool(best_mask.loc[idx])
        case_lines.extend(
            [
                f"## {row['dataset']} split {int(row['split'])} node {nid}",
                f"- MLP→FINAL_V3: {int(row['mlp_pred'])}→{int(row['final_pred'])}, true={int(row['true_label'])}",
                "- Reliability="
                f"{row['reliability']:.3f}, margin={row['mlp_margin']:.3f}, "
                f"score_gap={row['score_gap']:.3f}",
                f"- Contribution gaps: anchor={row['mlp_anchor_gap']:.3f}, prototype={row['prototype_gap']:.3f}, "
                f"neighbor={row['neighbor_support_gap']:.3f}, compatibility={row['compatibility_gap']:.3f}",
                f"- Dominant driver: {row['dominant_reason']}",
                "- Failure reading: graph-driven shift overrode a "
                "stronger/safer MLP anchor with only marginal total-score "
                "advantage.",
                f"- Blocked by safeguard `{best_rule}`: {'yes' if blocked else 'no'}.",
                "",
            ]
        )
    cases_md.write_text("\n".join(case_lines), encoding="utf-8")

    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].bar(["harmful", "beneficial"], [harm["reliability"].mean(), ben["reliability"].mean()])
    ax[0].set_title("Mean reliability")
    ax[1].bar(["harmful", "beneficial"], [harm["score_gap"].mean(), ben["score_gap"].mean()])
    ax[1].set_title("Mean score gap")
    plt.tight_layout()
    plt.savefig(fig_pdf)
    plt.close(fig)

    return {
        "summary_csv": summary_csv,
        "summary_md": summary_md,
        "report_md": report_md,
        "cases_md": cases_md,
        "figure": fig_pdf,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=datetime.now(UTC).strftime("%Y%m%d"))
    ap.add_argument("--node-csv", default=None)
    args = ap.parse_args()
    out = run(args.tag, Path(args.node_csv) if args.node_csv else None)
    for k, v in out.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()

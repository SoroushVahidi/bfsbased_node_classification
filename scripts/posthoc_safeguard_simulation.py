#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class Rule:
    name: str
    description: str
    block_fn: Callable[[pd.DataFrame], pd.Series]


def newest_file(patterns: list[str]) -> Path:
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(Path('.').glob(pattern))
    if not candidates:
        raise FileNotFoundError(f'No files found for patterns: {patterns}')
    return max(candidates, key=lambda p: p.stat().st_mtime)


def ensure_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')


def compute_metrics(df: pd.DataFrame, blocked: pd.Series) -> dict[str, float]:
    harmful = df['harmful'] == 1
    beneficial = df['beneficial'] == 1
    n_total = len(df)
    h_total = int(harmful.sum())
    b_total = int(beneficial.sum())

    harmful_prevented = int((harmful & blocked).sum())
    beneficial_lost = int((beneficial & blocked).sum())

    kept = ~blocked
    harmful_remaining = int((harmful & kept).sum())
    beneficial_remaining = int((beneficial & kept).sum())
    kept_total = harmful_remaining + beneficial_remaining

    baseline_precision = b_total / n_total if n_total else 0.0
    new_precision = beneficial_remaining / kept_total if kept_total else 0.0

    return {
        'n_changed_total': n_total,
        'n_harmful_total': h_total,
        'n_beneficial_total': b_total,
        'n_blocked_total': int(blocked.sum()),
        'harmful_prevented': harmful_prevented,
        'beneficial_lost': beneficial_lost,
        'harmful_prevention_rate': harmful_prevented / h_total if h_total else 0.0,
        'beneficial_retention_rate': (b_total - beneficial_lost) / b_total if b_total else 0.0,
        'changed_precision_baseline': baseline_precision,
        'changed_precision_after': new_precision,
        'changed_precision_delta_pp': 100.0 * (new_precision - baseline_precision),
    }


def to_markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    out = df[cols].copy()
    return out.to_markdown(index=False)


def main() -> None:
    today = datetime.now(timezone.utc).strftime('%Y%m%d')

    explanation_path = newest_file(['reports/correction_explanation_nodes_*.csv'])
    harmful_summary_path = newest_file(['tables/harmful_correction_failure_summary_*.csv'])
    bounded_path = newest_file(['tables/bounded_intervention_fullscope_*.csv'])
    sensitivity_path = newest_file(['tables/gate_sensitivity_fullscope_*.csv'])

    df = pd.read_csv(explanation_path)
    required = [
        'dataset', 'split', 'node_id', 'harmful', 'beneficial', 'reliability', 'score_gap',
        'dominant_reason', 'prototype_gap', 'neighbor_support_gap', 'mlp_margin', 'gap_concentration',
    ]
    ensure_columns(df, required)

    df = df[(df['harmful'] == 1) | (df['beneficial'] == 1)].copy()

    rules = [
        Rule(
            'R1_reliability_lt_0p40',
            'Block correction if reliability < 0.40.',
            lambda x: x['reliability'] < 0.40,
        ),
        Rule(
            'R2_score_gap_lt_0p30',
            'Block correction if score_gap < 0.30.',
            lambda x: x['score_gap'] < 0.30,
        ),
        Rule(
            'R3_no_proto_graph_agreement',
            'Block correction if prototype_gap <= 0 or neighbor_support_gap <= 0.',
            lambda x: (x['prototype_gap'] <= 0) | (x['neighbor_support_gap'] <= 0),
        ),
        Rule(
            'R4_graph_dominant_low_rel',
            'Block if dominant_reason is neighbor_support and reliability < 0.45.',
            lambda x: (x['dominant_reason'] == 'neighbor_support') & (x['reliability'] < 0.45),
        ),
        Rule(
            'R5_require_rel_and_gap',
            'Block unless reliability >= 0.40 and score_gap >= 0.20.',
            lambda x: ~((x['reliability'] >= 0.40) & (x['score_gap'] >= 0.20)),
        ),
    ]

    overall_rows: list[dict[str, float | str]] = []
    dataset_rows: list[dict[str, float | str]] = []

    for rule in rules:
        blocked = rule.block_fn(df)
        base = compute_metrics(df, blocked)
        overall_rows.append({'level': 'overall', 'rule': rule.name, 'description': rule.description, **base})

        for dataset, ddf in df.groupby('dataset'):
            blocked_dataset = rule.block_fn(ddf)
            drow = compute_metrics(ddf, blocked_dataset)
            dataset_rows.append(
                {
                    'level': 'dataset',
                    'dataset': dataset,
                    'rule': rule.name,
                    'description': rule.description,
                    **drow,
                }
            )

    overall_df = pd.DataFrame(overall_rows)
    ranked_df = overall_df.sort_values(
        ['harmful_prevention_rate', 'beneficial_retention_rate'],
        ascending=[False, False],
    )
    conservative_pool = overall_df[overall_df['beneficial_retention_rate'] >= 0.70]
    if not conservative_pool.empty:
        selected_df = conservative_pool.sort_values(
            ['harmful_prevention_rate', 'changed_precision_delta_pp'], ascending=[False, False]
        )
        best_rule_name = selected_df.iloc[0]['rule']
    else:
        best_rule_name = ranked_df.iloc[0]['rule']

    overall_df = overall_df.sort_values(
        ['harmful_prevention_rate', 'beneficial_retention_rate'],
        ascending=[False, False],
    )
    best_rule = next(r for r in rules if r.name == best_rule_name)
    best_blocked = best_rule.block_fn(df)

    case_pool = df[(df['harmful'] == 1) & best_blocked].copy()
    case_pool = case_pool.sort_values(['reliability', 'score_gap'], ascending=[True, True]).head(3)

    out_csv = Path(f'tables/posthoc_safeguard_simulation_{today}.csv')
    out_md_table = Path(f'tables/posthoc_safeguard_simulation_{today}.md')
    out_report = Path(f'reports/posthoc_safeguard_simulation_{today}.md')
    out_cases = Path(f'reports/posthoc_safeguard_case_studies_{today}.md')
    out_fig = Path(f'figures/posthoc_safeguard_tradeoff_{today}.pdf')

    combined_df = pd.concat([overall_df, pd.DataFrame(dataset_rows)], ignore_index=True)
    combined_df.to_csv(out_csv, index=False)

    table_cols = [
        'rule', 'harmful_prevented', 'beneficial_lost', 'harmful_prevention_rate', 'beneficial_retention_rate',
        'changed_precision_baseline', 'changed_precision_after', 'changed_precision_delta_pp', 'n_blocked_total',
    ]
    out_md_table.write_text(
        '# Post hoc safeguard simulation summary\n\n'
        + to_markdown_table(overall_df, table_cols)
        + '\n',
        encoding='utf-8',
    )

    fig_df = overall_df.copy()
    plt.figure(figsize=(7, 5))
    plt.scatter(fig_df['harmful_prevention_rate'], fig_df['beneficial_retention_rate'])
    for _, row in fig_df.iterrows():
        plt.annotate(row['rule'].replace('R', ''), (row['harmful_prevention_rate'], row['beneficial_retention_rate']))
    plt.xlabel('Harmful prevention rate')
    plt.ylabel('Beneficial retention rate')
    plt.title('Post hoc safeguard tradeoff (FINAL_V3 changed nodes)')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_fig, format='pdf')
    plt.close()

    lines = [
        f'# Post hoc safeguard simulation ({today})',
        '',
        '## Scope and method',
        'This is a post hoc simulation on existing FINAL_V3 changed-node exports only; no method redesign was run.',
        f'- Node-level explanation export used: `{explanation_path}`',
        f'- Harmful correction summary used: `{harmful_summary_path}`',
        f'- Bounded intervention table used: `{bounded_path}`',
        f'- Gate sensitivity table used: `{sensitivity_path}`',
        '',
        '## Overall safeguard tradeoff',
        to_markdown_table(overall_df, table_cols),
        '',
        f'Primary conservative rule (retention-aware): **{best_rule_name}**.',
        f"Secondary stronger-but-costlier option: **{ranked_df.iloc[0]['rule']}**.",
        '',
        '## Interpretation',
        '- Harmful changed-node corrections are the minority in the changed-node pool and can be partially targeted.',
        '- Simple threshold safeguards show clear tradeoffs between harmful prevention and beneficial retention.',
        '- A conservative score-gap rule preserves most beneficial corrections while still blocking a subset of harmful'
        ' ones.',
        '- A stricter reliability + graph-dominant rule blocks more harmful cases but at a substantial retention cost.',
        '',
        f'Figure: `{out_fig}`',
    ]
    out_report.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    case_lines = [
        f'# Post hoc safeguard case studies ({today})',
        '',
        f'Best safeguard selected: **{best_rule_name}** ({best_rule.description})',
        '',
        'The following harmful changed-node cases would be blocked by the selected safeguard:',
        '',
    ]
    for _, row in case_pool.iterrows():
        case_lines.extend(
            [
                f"- Dataset `{row['dataset']}`, split `{int(row['split'])}`, node `{int(row['node_id'])}`: "
                f"reliability={row['reliability']:.3f}, score_gap={row['score_gap']:.3f}, "
                f"dominant_reason={row['dominant_reason']}, mlp_margin={row['mlp_margin']:.3f}, "
                f"prototype_gap={row['prototype_gap']:.3f}, neighbor_support_gap={row['neighbor_support_gap']:.3f}.",
            ]
        )
    out_cases.write_text('\n'.join(case_lines) + '\n', encoding='utf-8')

    print(f'Wrote {out_csv}')
    print(f'Wrote {out_md_table}')
    print(f'Wrote {out_report}')
    print(f'Wrote {out_cases}')
    print(f'Wrote {out_fig}')


if __name__ == '__main__':
    main()

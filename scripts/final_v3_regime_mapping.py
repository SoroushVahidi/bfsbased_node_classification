#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def newest(patterns: list[str]) -> Path:
    paths: list[Path] = []
    for pat in patterns:
        paths.extend(Path('.').glob(pat))
    if not paths:
        raise FileNotFoundError(f'No files matched: {patterns}')
    return max(paths, key=lambda p: p.stat().st_mtime)


def main() -> None:
    today = datetime.now(timezone.utc).strftime('%Y%m%d')

    regime_path = newest(['tables/final_v3_regime_summary.csv'])
    setup_path = newest(['tables/experimental_setup_selective_correction.csv'])
    bounded_path = newest(['tables/bounded_intervention_fullscope_*.csv'])
    sensitivity_path = newest(['tables/gate_sensitivity_fullscope_*.csv'])
    diagnostics_path = newest(['reports/final_v3_correction_diagnostics_*_fullscope.csv'])
    explanation_path = newest(['reports/correction_explanation_nodes_*.csv'])

    regime = pd.read_csv(regime_path)
    setup = pd.read_csv(setup_path)
    bounded = pd.read_csv(bounded_path)
    sensitivity = pd.read_csv(sensitivity_path)
    diagnostics = pd.read_csv(diagnostics_path)
    explanation = pd.read_csv(explanation_path)

    explanation = explanation[(explanation['harmful'] == 1) | (explanation['beneficial'] == 1)].copy()
    explanation_agg = (
        explanation.groupby('dataset', as_index=False)
        .agg(
            n_changed_nodes=('node_id', 'count'),
            harmful_rate_changed_nodes=('harmful', 'mean'),
            mean_score_gap_changed_nodes=('score_gap', 'mean'),
            mean_reliability_changed_nodes=('reliability', 'mean'),
            mean_gap_concentration=('gap_concentration', 'mean'),
        )
    )

    sensitivity_ds = sensitivity[sensitivity['dataset'] != 'ALL'].copy()
    sens_agg = (
        sensitivity_ds.groupby('dataset', as_index=False)
        .agg(
            gate_delta_span_pp=('mean_delta_vs_mlp_pp', lambda s: float(s.max() - s.min())),
            gate_delta_std_pp=('mean_delta_vs_mlp_pp', 'std'),
            gate_mean_routed_pct_across_rho=('mean_fraction_routed_to_correction_pct', 'mean'),
        )
    )

    out = regime[
        [
            'dataset',
            'n_splits',
            'edge_homophily',
            'final_minus_mlp_mean',
            'changed_fraction_mean',
            'correction_precision_mean',
            'harmful_overwrite_rate_mean',
            'wins_vs_mlp',
            'losses_vs_mlp',
            'ties_vs_mlp',
        ]
    ].copy()

    out = out.merge(
        bounded[
            [
                'dataset',
                'mean_fraction_routed_to_correction_pct',
                'mean_fraction_changed_vs_mlp_pct',
                'mean_changed_precision',
                'harmful_split_level_departures',
            ]
        ],
        on='dataset',
        how='left',
    )
    out = out.merge(
        diagnostics[['dataset', 'mean_reliability_corrected', 'mean_delta_vs_mlp_pp']], on='dataset', how='left'
    )
    out = out.merge(setup[['dataset', 'num_nodes', 'num_classes']], on='dataset', how='left')
    out = out.merge(sens_agg, on='dataset', how='left')
    out = out.merge(explanation_agg, on='dataset', how='left')

    out = out.sort_values('final_minus_mlp_mean', ascending=False)

    rename_map = {
        'final_minus_mlp_mean': 'improvement_vs_mlp_pp',
        'changed_fraction_mean': 'changed_fraction',
        'correction_precision_mean': 'changed_precision',
        'harmful_overwrite_rate_mean': 'harmful_overwrite_rate',
        'mean_fraction_routed_to_correction_pct': 'routed_fraction_pct',
        'mean_fraction_changed_vs_mlp_pct': 'changed_fraction_pct',
        'mean_changed_precision': 'changed_precision_bounded',
        'mean_reliability_corrected': 'mean_reliability_corrected_nodes',
        'mean_delta_vs_mlp_pp': 'improvement_vs_mlp_pp_diagnostics',
    }
    out = out.rename(columns=rename_map)

    numeric_cols = [
        'improvement_vs_mlp_pp',
        'edge_homophily',
        'routed_fraction_pct',
        'changed_fraction_pct',
        'changed_precision',
        'changed_precision_bounded',
        'harmful_overwrite_rate',
        'mean_reliability_corrected_nodes',
        'mean_score_gap_changed_nodes',
        'harmful_rate_changed_nodes',
        'gate_delta_span_pp',
        'gate_delta_std_pp',
    ]
    corr_rows: list[dict[str, float | str | int]] = []
    for factor in numeric_cols:
        if factor == 'improvement_vs_mlp_pp':
            continue
        pair = out[['improvement_vs_mlp_pp', factor]].dropna()
        n_used = len(pair)
        if n_used >= 4:
            corr_value = pair['improvement_vs_mlp_pp'].corr(pair[factor], method='spearman')
        else:
            corr_value = float('nan')
        corr_rows.append(
            {
                'factor': factor,
                'n_used': n_used,
                'spearman_corr_with_improvement': corr_value,
            }
        )
    corr_df = pd.DataFrame(corr_rows).sort_values(
        'spearman_corr_with_improvement',
        ascending=False,
        na_position='last',
    )

    out_csv = Path(f'tables/final_v3_regime_mapping_{today}.csv')
    out_md = Path(f'tables/final_v3_regime_mapping_{today}.md')
    report_md = Path(f'reports/final_v3_regime_mapping_{today}.md')
    fig_map = Path(f'figures/final_v3_regime_map_{today}.pdf')
    fig_precision = Path(f'figures/final_v3_improvement_vs_precision_{today}.pdf')

    out.to_csv(out_csv, index=False)

    out_md.write_text(
        '# FINAL_V3 regime mapping summary\n\n'
        '## Dataset-level regime table\n\n'
        + out.to_markdown(index=False)
        + '\n\n## Spearman correlations with improvement_vs_mlp_pp\n\n'
        + corr_df.to_markdown(index=False)
        + '\n',
        encoding='utf-8',
    )

    fig_df = out.copy()
    plt.figure(figsize=(7, 5))
    sizes = 30 + fig_df['routed_fraction_pct'].fillna(0).clip(lower=0) * 8
    plt.scatter(
        fig_df['edge_homophily'],
        fig_df['improvement_vs_mlp_pp'],
        s=sizes,
        c=fig_df['changed_precision'],
        cmap='viridis',
    )
    for _, row in fig_df.iterrows():
        plt.annotate(row['dataset'], (row['edge_homophily'], row['improvement_vs_mlp_pp']))
    cbar = plt.colorbar()
    cbar.set_label('Changed precision')
    plt.xlabel('Edge homophily')
    plt.ylabel('FINAL_V3 improvement vs MLP (pp)')
    plt.title('FINAL_V3 regime map (bubble size = routed fraction %)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_map, format='pdf')
    plt.close()

    plt.figure(figsize=(7, 5))
    x = fig_df['changed_precision']
    y = fig_df['improvement_vs_mlp_pp']
    plt.scatter(x, y)
    for _, row in fig_df.iterrows():
        plt.annotate(row['dataset'], (row['changed_precision'], row['improvement_vs_mlp_pp']))
    if len(fig_df) >= 2:
        coeffs = pd.Series(y).corr(x, method='pearson')
        fit = np.polyfit(x, y, 1)
        xline = pd.Series([x.min(), x.max()])
        yline = fit[0] * xline + fit[1]
        plt.plot(xline, yline, linestyle='--')
        plt.text(float(x.min()), float(y.max()), f'Pearson r={coeffs:.3f}')
    plt.xlabel('Changed precision')
    plt.ylabel('FINAL_V3 improvement vs MLP (pp)')
    plt.title('Improvement tracks correction quality more than volume')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_precision, format='pdf')
    plt.close()

    corr_valid = corr_df.dropna(subset=['spearman_corr_with_improvement']).copy()
    top_positive = corr_valid.head(4).to_markdown(index=False)
    top_negative = (
        corr_valid.sort_values('spearman_corr_with_improvement', ascending=True).head(4).to_markdown(index=False)
    )

    report_lines = [
        f'# FINAL_V3 regime mapping ({today})',
        '',
        '## Source artifacts (auto-detected newest)',
        f'- Regime summary: `{regime_path}`',
        f'- Experimental setup: `{setup_path}`',
        f'- Bounded intervention: `{bounded_path}`',
        f'- Gate sensitivity: `{sensitivity_path}`',
        f'- Correction diagnostics: `{diagnostics_path}`',
        f'- Node-level explanation: `{explanation_path}`',
        '',
        '## Key descriptive findings',
        '- FINAL_V3 gains are heterogeneous across datasets; they are not uniform.',
        '- Larger gains align more with correction quality metrics (changed precision, reliability) than with'
        ' high routing volume.',
        '- Harmful-overwrite signals trend negatively with gains; changed-node harmful-rate signals are limited'
        ' to datasets with node-level exports.',
        '- Sensitivity-span indicators suggest that highly unstable gate-response regimes are less consistently'
        ' favorable.',
        '',
        '## Correlation snapshot (Spearman, dataset-level)',
        'Top positive correlates:',
        top_positive,
        '',
        'Top negative correlates:',
        top_negative,
        '',
        '## Regime-oriented interpretation',
        'This is a descriptive regime map, not a causal claim. The pattern supports selective correction as a bounded',
        'intervention that works best when corrections are high quality and safety indicators remain controlled.',
        '',
        f'Figures: `{fig_map}`, `{fig_precision}`',
    ]
    report_md.write_text('\n'.join(report_lines) + '\n', encoding='utf-8')

    print(f'Wrote {out_csv}')
    print(f'Wrote {out_md}')
    print(f'Wrote {report_md}')
    print(f'Wrote {fig_map}')
    print(f'Wrote {fig_precision}')


if __name__ == '__main__':
    main()

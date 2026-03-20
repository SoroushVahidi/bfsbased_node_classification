#!/usr/bin/env bash
set -euo pipefail
python3 - <<'PY'
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

import comparison_runner as cr

out_root = Path('/workspace/logs/prl_local_20260319/exp01_term_ablation')
split_dir = '/workspace/data/geom-gcn-splits-broad'
datasets = ['cora', 'citeseer']
splits = list(range(10))
repeats = [0]

configs = {
    'full': {'b1': 1.0, 'b2': 0.4, 'b3': 0.0, 'b4': 0.9, 'b5': 0.5},
    'no_graph': {'b1': 1.0, 'b2': 0.4, 'b3': 0.0, 'b4': 0.0, 'b5': 0.0},
    'no_feature_sim': {'b1': 1.0, 'b2': 0.0, 'b3': 0.0, 'b4': 0.9, 'b5': 0.5},
    'no_compatibility': {'b1': 1.0, 'b2': 0.4, 'b3': 0.0, 'b4': 0.9, 'b5': 0.0},
}

mod = cr._load_full_investigate_module()
records = []

for ds in datasets:
    dataset, data = cr._load_dataset_and_data(mod, ds)
    device = data.x.device
    for split_id in splits:
        for repeat_id in repeats:
            seed = (1337 + split_id) * 100 + repeat_id
            torch.manual_seed(seed)
            np.random.seed(seed)

            train_idx, val_idx, test_idx, split_path = cr._load_split_npz(ds, split_id, device=device, split_dir=split_dir)
            train_np = mod._to_numpy_idx(train_idx)
            val_np = mod._to_numpy_idx(val_idx)
            test_np = mod._to_numpy_idx(test_idx)

            t0 = time.perf_counter()
            mlp_probs, _ = mod.train_mlp_and_predict(
                data,
                train_idx,
                hidden=64,
                layers=2,
                dropout=0.5,
                lr=0.01,
                epochs=200,
                log_file=None,
            )
            mlp_runtime = time.perf_counter() - t0
            preds = mlp_probs.argmax(dim=1)
            mlp_test_acc = float((preds[test_idx] == data.y[test_idx]).float().mean().item())
            mlp_val_acc = float((preds[val_idx] == data.y[val_idx]).float().mean().item())

            for config_name, weights in configs.items():
                run_id = f'{config_name}_split{split_id}_rep{repeat_id}'
                _, test_acc_sel, info = mod.selective_graph_correction_predictclass(
                    data,
                    train_np,
                    val_np,
                    test_np,
                    mlp_probs=mlp_probs,
                    seed=seed,
                    log_prefix=f'{ds}:PRL-EXP01',
                    log_file=None,
                    diagnostics_run_id=run_id,
                    dataset_key_for_logs=ds,
                    mlp_runtime_sec=mlp_runtime,
                    threshold_candidates=None,
                    weight_candidates=[weights],
                    enable_feature_knn=False,
                    feature_knn_k=5,
                    write_node_diagnostics=False,
                )
                rec = {
                    'dataset': ds,
                    'split_id': split_id,
                    'repeat_id': repeat_id,
                    'seed': seed,
                    'config_name': config_name,
                    'weights': weights,
                    'split_file': split_path,
                    'mlp_val_acc': mlp_val_acc,
                    'mlp_test_acc': mlp_test_acc,
                    'selective_test_acc': float(test_acc_sel),
                    'delta_vs_mlp': float(test_acc_sel - mlp_test_acc),
                    'selected_threshold_high': info.get('selected_threshold_high'),
                    'fraction_test_nodes_uncertain': info.get('fraction_test_nodes_uncertain'),
                    'fraction_test_nodes_changed_from_mlp': info.get('fraction_test_nodes_changed_from_mlp'),
                    'acc_test_confident_nodes': info.get('acc_test_confident_nodes'),
                    'acc_test_uncertain_nodes': info.get('acc_test_uncertain_nodes'),
                    'runtime_breakdown_sec': info.get('runtime_breakdown_sec', {}),
                }
                records.append(rec)

raw_path = out_root / 'raw' / 'term_ablation_runs.jsonl'
with raw_path.open('w', encoding='utf-8') as f:
    for r in records:
        f.write(json.dumps(r) + '\n')

summary_rows = []
by = defaultdict(list)
for r in records:
    by[(r['dataset'], r['config_name'])].append(r)

for (ds, cfg), items in sorted(by.items()):
    deltas = np.array([x['delta_vs_mlp'] for x in items], dtype=float)
    sels = np.array([x['selective_test_acc'] for x in items], dtype=float)
    wins = int((deltas > 1e-12).sum())
    losses = int((deltas < -1e-12).sum())
    ties = int((np.abs(deltas) <= 1e-12).sum())
    summary_rows.append({
        'dataset': ds,
        'config_name': cfg,
        'n_runs': len(items),
        'mean_selective_test_acc': float(sels.mean()),
        'std_selective_test_acc': float(sels.std()),
        'mean_delta_vs_mlp': float(deltas.mean()),
        'std_delta_vs_mlp': float(deltas.std()),
        'wins_vs_mlp': wins,
        'losses_vs_mlp': losses,
        'ties_vs_mlp': ties,
        'avg_uncertain': float(np.mean([x['fraction_test_nodes_uncertain'] for x in items])),
        'avg_changed': float(np.mean([x['fraction_test_nodes_changed_from_mlp'] for x in items])),
        'avg_selected_threshold': float(np.mean([x['selected_threshold_high'] for x in items])),
    })

summary_csv = out_root / 'tables' / 'term_ablation_summary.csv'
with summary_csv.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    writer.writeheader()
    writer.writerows(summary_rows)

summary_lookup = {(r['dataset'], r['config_name']): r for r in summary_rows}
gain_rows = []
for ds in datasets:
    full = summary_lookup[(ds, 'full')]
    for cfg in configs.keys():
        cur = summary_lookup[(ds, cfg)]
        gain_rows.append({
            'dataset': ds,
            'config_name': cfg,
            'mean_delta_vs_mlp': cur['mean_delta_vs_mlp'],
            'mean_selective_test_acc': cur['mean_selective_test_acc'],
            'mean_selective_minus_full': cur['mean_selective_test_acc'] - full['mean_selective_test_acc'],
            'mean_delta_vs_mlp_minus_full': cur['mean_delta_vs_mlp'] - full['mean_delta_vs_mlp'],
        })

gain_csv = out_root / 'tables' / 'term_ablation_gain_vs_full.csv'
with gain_csv.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(gain_rows[0].keys()))
    writer.writeheader()
    writer.writerows(gain_rows)

cfg_order = list(configs.keys())
x = np.arange(len(datasets))
width = 0.18
fig, ax = plt.subplots(figsize=(8, 4.8))
for i, cfg in enumerate(cfg_order):
    vals = [summary_lookup[(ds, cfg)]['mean_delta_vs_mlp'] for ds in datasets]
    ax.bar(x + (i - 1.5) * width, vals, width=width, label=cfg)
ax.axhline(0.0, color='black', linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.set_ylabel('Mean selective - MLP test acc')
ax.set_title('Exp01 term ablation (Cora/Citeseer, splits 0..9, repeats=1)')
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(out_root / 'plots' / 'term_ablation_bar_cora_citeseer.png', dpi=220)
plt.close(fig)

notes = out_root / 'notes' / 'term_ablation_notes.md'
lines = [
    '# Exp01 term ablation notes',
    '',
    '- Scope: datasets={cora,citeseer}, splits=0..9, repeats=1',
    '- Configs: full, no_graph, no_feature_sim, no_compatibility',
    '- Interpretation uses mean delta vs MLP and mean selective acc vs full.',
    '',
    '## Quick summary',
]
for ds in datasets:
    rows_ds = [r for r in gain_rows if r['dataset'] == ds]
    rows_ds.sort(key=lambda r: r['mean_selective_minus_full'], reverse=True)
    best = rows_ds[0]
    worst = rows_ds[-1]
    lines.append(f"- {ds}: best config by selective acc = {best['config_name']} ({best['mean_selective_minus_full']:+.4f} vs full), worst = {worst['config_name']} ({worst['mean_selective_minus_full']:+.4f} vs full)")

notes.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(raw_path)
print(summary_csv)
print(gain_csv)
print(out_root / 'plots' / 'term_ablation_bar_cora_citeseer.png')
print(notes)
PY

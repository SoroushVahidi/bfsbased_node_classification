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

out_root = Path('/workspace/logs/prl_local_20260319/exp06_threshold_control_mini_sweep')
split_dir = '/workspace/data/geom-gcn-splits-broad'
datasets = ['chameleon', 'wisconsin']
splits = list(range(10))
repeats = [0]
threshold_grid = [0.05, 0.25, 0.65]
full_weights = {'b1': 1.0, 'b2': 0.4, 'b3': 0.0, 'b4': 0.9, 'b5': 0.5}

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

            for thr in threshold_grid:
                run_id = f'thr{thr:.2f}_split{split_id}_rep{repeat_id}'
                _, test_acc_sel, info = mod.selective_graph_correction_predictclass(
                    data,
                    train_np,
                    val_np,
                    test_np,
                    mlp_probs=mlp_probs,
                    seed=seed,
                    log_prefix=f'{ds}:PRL-EXP06',
                    log_file=None,
                    diagnostics_run_id=run_id,
                    dataset_key_for_logs=ds,
                    mlp_runtime_sec=mlp_runtime,
                    threshold_candidates=[float(thr)],
                    weight_candidates=[full_weights],
                    enable_feature_knn=False,
                    feature_knn_k=5,
                    write_node_diagnostics=False,
                )
                rec = {
                    'dataset': ds,
                    'split_id': split_id,
                    'repeat_id': repeat_id,
                    'seed': seed,
                    'threshold_requested': float(thr),
                    'threshold_selected': info.get('selected_threshold_high'),
                    'weights': full_weights,
                    'split_file': split_path,
                    'mlp_val_acc': mlp_val_acc,
                    'mlp_test_acc': mlp_test_acc,
                    'selective_test_acc': float(test_acc_sel),
                    'delta_vs_mlp': float(test_acc_sel - mlp_test_acc),
                    'fraction_test_nodes_uncertain': info.get('fraction_test_nodes_uncertain'),
                    'fraction_test_nodes_changed_from_mlp': info.get('fraction_test_nodes_changed_from_mlp'),
                    'acc_test_confident_nodes': info.get('acc_test_confident_nodes'),
                    'acc_test_uncertain_nodes': info.get('acc_test_uncertain_nodes'),
                    'runtime_breakdown_sec': info.get('runtime_breakdown_sec', {}),
                }
                records.append(rec)

raw_path = out_root / 'raw' / 'threshold_control_mini_sweep_runs.jsonl'
with raw_path.open('w', encoding='utf-8') as f:
    for r in records:
        f.write(json.dumps(r) + '\n')

summary_rows = []
by = defaultdict(list)
for r in records:
    by[(r['dataset'], r['threshold_requested'])].append(r)

for (ds, thr), items in sorted(by.items()):
    deltas = np.array([x['delta_vs_mlp'] for x in items], dtype=float)
    sels = np.array([x['selective_test_acc'] for x in items], dtype=float)
    wins = int((deltas > 1e-12).sum())
    losses = int((deltas < -1e-12).sum())
    ties = int((np.abs(deltas) <= 1e-12).sum())
    summary_rows.append({
        'dataset': ds,
        'threshold': float(thr),
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
    })

summary_csv = out_root / 'tables' / 'threshold_control_mini_sweep_summary.csv'
with summary_csv.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    writer.writeheader()
    writer.writerows(summary_rows)

uc_rows = []
for r in summary_rows:
    uc_rows.append({
        'dataset': r['dataset'],
        'threshold': r['threshold'],
        'avg_uncertain': r['avg_uncertain'],
        'avg_changed': r['avg_changed'],
        'mean_delta_vs_mlp': r['mean_delta_vs_mlp'],
    })

uc_csv = out_root / 'tables' / 'threshold_control_vs_uncertain_changed.csv'
with uc_csv.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(uc_rows[0].keys()))
    writer.writeheader()
    writer.writerows(uc_rows)

for ds in datasets:
    rows_ds = sorted([r for r in summary_rows if r['dataset'] == ds], key=lambda z: z['threshold'])
    x = [r['threshold'] for r in rows_ds]
    y = [r['mean_delta_vs_mlp'] for r in rows_ds]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(x, y, marker='o')
    ax.axhline(0.0, color='black', linewidth=1)
    ax.set_xlabel('Threshold (fixed)')
    ax.set_ylabel('Mean selective - MLP test acc')
    ax.set_title(f'Exp06 threshold control mini-sweep: {ds}')
    fig.tight_layout()
    fig.savefig(out_root / 'plots' / f'threshold_gain_curve_{ds}.png', dpi=220)
    plt.close(fig)

notes = out_root / 'notes' / 'threshold_control_mini_sweep_notes.md'
lines = [
    '# Exp06 threshold control mini-sweep notes',
    '',
    '- Scope: datasets={chameleon,wisconsin}, splits=0..9, repeats=1',
    '- Fixed weights: full (b1=1.0,b2=0.4,b3=0.0,b4=0.9,b5=0.5)',
    f'- Threshold grid: {threshold_grid}',
    '',
    '## Summary by dataset',
]
for ds in datasets:
    rows_ds = [r for r in summary_rows if r['dataset'] == ds]
    best = max(rows_ds, key=lambda z: z['mean_delta_vs_mlp'])
    worst = min(rows_ds, key=lambda z: z['mean_delta_vs_mlp'])
    lines.append(
        f"- {ds}: best thr={best['threshold']:.2f} delta={best['mean_delta_vs_mlp']:+.4f}, "
        f"worst thr={worst['threshold']:.2f} delta={worst['mean_delta_vs_mlp']:+.4f}"
    )

notes.write_text('\n'.join(lines) + '\n', encoding='utf-8')

print(raw_path)
print(summary_csv)
print(uc_csv)
print(out_root / 'plots' / 'threshold_gain_curve_chameleon.png')
print(out_root / 'plots' / 'threshold_gain_curve_wisconsin.png')
print(notes)
PY

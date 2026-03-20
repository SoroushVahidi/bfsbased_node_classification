#!/usr/bin/env bash
set -euo pipefail
python3 - <<'PY'
import csv
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

out_root = Path('/workspace/logs/prl_local_20260319/exp05_case_study')
source = Path('/workspace/logs/manuscript_case_studies_local_cycle.json')
blocks = json.loads(source.read_text())

wanted = ['citeseer', 'chameleon', 'wisconsin']
selected = [b for b in blocks if b.get('dataset') in wanted]

rows = []
for b in selected:
    ds = b['dataset']
    kind = b.get('kind', '')
    split_id = b.get('split_id')
    repeat_id = b.get('repeat_id')
    for ex in b.get('examples', []):
        rows.append({
            'dataset': ds,
            'kind': kind,
            'source_split_id': split_id,
            'source_repeat_id': repeat_id,
            'source_diagnostics_file': b.get('diagnostics_file'),
            'node_id': ex.get('node_id'),
            'mlp_pred': ex.get('mlp_pred'),
            'final_pred': ex.get('final_pred'),
            'true_label': ex.get('true_label'),
            'outcome': ex.get('outcome'),
            'mlp_margin': ex.get('mlp_margin'),
            'dominant_term': ex.get('dominant_term'),
            'graph_support': ex.get('graph_support'),
            'feature_support': ex.get('feature_support'),
            'interpretation': ex.get('interpretation'),
        })

csv_path = out_root / 'tables' / 'case_examples.csv'
if rows:
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
else:
    csv_path.write_text('dataset,kind,source_split_id,source_repeat_id,source_diagnostics_file,node_id,mlp_pred,final_pred,true_label,outcome,mlp_margin,dominant_term,graph_support,feature_support,interpretation\n')

json_path = out_root / 'tables' / 'case_examples.json'
json_path.write_text(json.dumps({'source': str(source), 'selected_blocks': selected, 'flat_rows': rows}, indent=2), encoding='utf-8')

md_path = out_root / 'tables' / 'case_examples.md'
md = ['# Case examples (refresh)', '']
for b in selected:
    ds = b['dataset']
    md.append(f"## {ds} ({b.get('kind','')})")
    md.append(f"- source diagnostics: `{b.get('diagnostics_file')}`")
    md.append(f"- selected run split={b.get('split_id')}, repeat={b.get('repeat_id')}")
    md.append(f"- changed/helped/hurt: {b.get('n_changed_in_selected_run')}/{b.get('n_helped_in_selected_run')}/{b.get('n_hurt_in_selected_run')}")
    md.append('| node | mlp | final | true | outcome | margin | dominant term | graph support | feature support |')
    md.append('|---:|---:|---:|---:|---|---:|---|---:|---:|')
    examples = b.get('examples', [])
    if examples:
        for ex in examples:
            md.append(f"| {ex.get('node_id')} | {ex.get('mlp_pred')} | {ex.get('final_pred')} | {ex.get('true_label')} | {ex.get('outcome')} | {float(ex.get('mlp_margin',0.0)):.4f} | {ex.get('dominant_term')} | {float(ex.get('graph_support',0.0)):.4f} | {float(ex.get('feature_support',0.0)):.4f} |")
    else:
        md.append('| - | - | - | - | no changed nodes in selected run | - | - | - | - |')
    md.append('')
md_path.write_text('\n'.join(md)+'\n', encoding='utf-8')

note_path = out_root / 'notes' / 'case_interpretation_notes.md'
notes = ['# Case-study interpretation notes', '']
for b in selected:
    ds = b['dataset']
    notes.append(f"- {ds}: changed={b.get('n_changed_in_selected_run',0)}, helped={b.get('n_helped_in_selected_run',0)}, hurt={b.get('n_hurt_in_selected_run',0)}")
notes.append('')
notes.append('Interpretation focus: cite helpful low-margin corrections in citeseer; mixed helped/hurt behavior in chameleon; fallback/no-change behavior in wisconsin if present.')
note_path.write_text('\n'.join(notes)+'\n', encoding='utf-8')

labels=[]
helped=[]
hurt=[]
other=[]
for b in selected:
    labels.append(b['dataset'])
    h = int(b.get('n_helped_in_selected_run', 0) or 0)
    u = int(b.get('n_hurt_in_selected_run', 0) or 0)
    c = int(b.get('n_changed_in_selected_run', 0) or 0)
    o = max(0, c - h - u)
    helped.append(h); hurt.append(u); other.append(o)

if labels:
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.2,4.6))
    ax.bar(x, helped, label='helped')
    ax.bar(x, hurt, bottom=np.array(helped), label='hurt')
    ax.bar(x, other, bottom=np.array(helped)+np.array(hurt), label='other_changed')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Count in selected run')
    ax.set_title('Case-study component outcomes')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_root / 'plots' / 'case_component_stackedbars.png', dpi=220)
    plt.close(fig)

print(csv_path)
print(json_path)
print(md_path)
print(note_path)
print(out_root / 'plots' / 'case_component_stackedbars.png')
PY

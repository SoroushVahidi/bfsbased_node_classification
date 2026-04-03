"""
LINKX runner - Large Scale Learning on Non-Homophilous Graphs
Inline reimplementation. Source: https://github.com/CUAI/Non-Homophily-Large-Scale
"""
import sys
import os
import argparse
import json
import time

_BENCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_BENCH_DIR)
sys.path.insert(0, _BENCH_DIR)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'code', 'bfsbased_node_classification'))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.utils import add_self_loops, degree
except ImportError as e:
    print(f"ERROR: Required packages not available: {e}", file=sys.stderr)
    sys.exit(1)

from benchmark.data_adapter import build_data_bundle

SOURCE_REPO = "https://github.com/CUAI/Non-Homophily-Large-Scale"
METHOD = "linkx"


def sparse_row_norm(edge_index, n_nodes):
    """Row-normalized adjacency (no self-loops) as sparse tensor."""
    from torch_geometric.utils import remove_self_loops
    edge_index, _ = remove_self_loops(edge_index)
    row, col = edge_index
    deg = degree(row, n_nodes, dtype=torch.float).clamp(min=1)
    ew = 1.0 / deg[row]
    return torch.sparse_coo_tensor(edge_index, ew, (n_nodes, n_nodes)).coalesce()


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        self.dropout = dropout
        layers = []
        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LINKX(nn.Module):
    """
    LINKX: separate feature and structure branches combined additively.
    - Feature branch: MLP_f(x)
    - Structure branch: MLP_a(A@x)  [1-hop aggregated features]
    - Combined: MLP_f(x) + MLP_a(A@x) -> Linear -> C
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.mlp_f = MLP(in_channels, hidden_channels, hidden_channels, num_layers, dropout)
        self.mlp_a = MLP(in_channels, hidden_channels, hidden_channels, num_layers, dropout)
        self.lin   = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, adj):
        h_f = self.mlp_f(x)
        ax  = torch.sparse.mm(adj, x)
        h_a = self.mlp_a(ax)
        h   = F.relu(h_f + h_a)
        h   = F.dropout(h, p=self.dropout, training=self.training)
        return self.lin(h)


def train_epoch(model, optimizer, adj, x, y, train_mask):
    model.train()
    optimizer.zero_grad()
    out = model(x, adj)
    loss = F.cross_entropy(out[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, adj, x, y, mask):
    model.eval()
    out = model(x, adj)
    pred = out[mask].argmax(dim=1)
    return (pred == y[mask]).float().mean().item()


def run(args):
    import yaml

    config_path = args.config or os.path.join(_BENCH_DIR, 'configs', 'linkx.yaml')
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    hidden     = cfg.get('hidden_channels', 256)
    dropout    = cfg.get('dropout', 0.5)
    lr         = cfg.get('lr', 0.01)
    wd         = cfg.get('weight_decay', 5e-4)
    epochs     = cfg.get('epochs', 1000)
    patience   = cfg.get('patience', 100)
    num_layers = cfg.get('num_layers', 3)

    output_dir = args.output_dir or os.path.join(_BENCH_DIR, 'results', 'per_run')
    os.makedirs(output_dir, exist_ok=True)
    tag = args.output_tag or 'default'
    out_file = os.path.join(
        output_dir,
        f"{METHOD}_{args.dataset}_split{args.split_id}_seed{args.seed}_{tag}.json"
    )

    if not args.force_rerun and os.path.exists(out_file):
        print(f"[{METHOD}] Output exists, skipping: {out_file}")
        with open(out_file) as f:
            return json.load(f)

    torch.manual_seed(args.seed)

    bundle = build_data_bundle(args.dataset, args.split_id,
                               data_root=args.data_root, split_dir=args.split_dir)
    x          = bundle['x']
    y          = bundle['y']
    edge_index = bundle['edge_index']
    train_mask = bundle['train_mask']
    val_mask   = bundle['val_mask']
    test_mask  = bundle['test_mask']
    n_nodes    = bundle['n_nodes']

    adj = sparse_row_norm(edge_index, n_nodes)

    model = LINKX(bundle['n_features'], hidden, bundle['n_classes'],
                  num_layers=num_layers, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best_val_acc  = 0.0
    best_test_acc = 0.0
    best_epoch    = 0
    no_improve    = 0

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        train_epoch(model, optimizer, adj, x, y, train_mask)
        val_acc = evaluate(model, adj, x, y, val_mask)

        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_test_acc = evaluate(model, adj, x, y, test_mask)
            best_epoch    = epoch
            no_improve    = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    runtime = time.time() - t0

    result = {
        "method": METHOD,
        "dataset": args.dataset,
        "split_id": args.split_id,
        "seed": args.seed,
        "success": True,
        "val_acc": round(best_val_acc, 6),
        "test_acc": round(best_test_acc, 6),
        "best_epoch": best_epoch,
        "runtime_sec": round(runtime, 3),
        "source_repo": SOURCE_REPO,
        "config_used": cfg,
        "stdout_log_path": None,
        "stderr_log_path": None,
        "notes": f"Inline reimplementation. Best epoch: {best_epoch}.",
        "commit_or_version_if_available": "inline-reimplementation",
    }

    tmp = out_file + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(result, f, indent=2)
    os.replace(tmp, out_file)

    print(f"[{METHOD}] {args.dataset} split={args.split_id} seed={args.seed} "
          f"val={best_val_acc:.4f} test={best_test_acc:.4f} ({runtime:.1f}s) -> {out_file}")
    return result


def main():
    parser = argparse.ArgumentParser(description='Run LINKX benchmark')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--split-id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-tag', default='default')
    parser.add_argument('--config', default=None)
    parser.add_argument('--split-dir', default=None)
    parser.add_argument('--data-root', default=None)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--force-rerun', action='store_true')
    args = parser.parse_args()

    try:
        run(args)
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

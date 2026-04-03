"""
ACM-GNN runner - Adaptive Channel Mixing GNN
Inline reimplementation. Source: https://github.com/SitaoLuan/ACM-GNN
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

SOURCE_REPO = "https://github.com/SitaoLuan/ACM-GNN"
METHOD = "acmgnn"


def get_sym_norm_adj(edge_index, n_nodes):
    """Symmetric normalized adjacency with self-loops as sparse tensor."""
    edge_index, _ = add_self_loops(edge_index, num_nodes=n_nodes)
    row, col = edge_index
    deg = degree(row, n_nodes, dtype=torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
    ew = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return torch.sparse_coo_tensor(edge_index, ew, (n_nodes, n_nodes)).coalesce()


class ACMGNNLayer(nn.Module):
    """
    Single ACM layer:
    - LP: A_sym @ H  (low-pass)
    - HP: H - A_sym @ H  (high-pass)
    - ID: H  (identity)
    - Mix: per-node softmax attention over the 3 channels
    """
    def __init__(self, hidden_channels):
        super().__init__()
        self.attn = nn.Linear(hidden_channels, 3, bias=False)

    def forward(self, h, adj):
        lp  = torch.sparse.mm(adj, h)
        hp  = h - lp
        id_ = h

        w = F.softmax(self.attn(h), dim=1)  # n x 3
        out = (w[:, 0:1] * lp) + (w[:, 1:2] * hp) + (w[:, 2:3] * id_)
        return out


class ACMGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.embed = nn.Linear(in_channels, hidden_channels)
        self.acm   = ACMGNNLayer(hidden_channels)
        self.out   = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, adj):
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = F.relu(self.embed(h))
        h = self.acm(h, adj)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.out(h)


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

    config_path = args.config or os.path.join(_BENCH_DIR, 'configs', 'acmgnn.yaml')
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    hidden   = cfg.get('hidden_channels', 64)
    dropout  = cfg.get('dropout', 0.5)
    lr       = cfg.get('lr', 0.01)
    wd       = cfg.get('weight_decay', 5e-4)
    epochs   = cfg.get('epochs', 1000)
    patience = cfg.get('patience', 100)

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

    adj = get_sym_norm_adj(edge_index, n_nodes)

    model = ACMGNN(bundle['n_features'], hidden, bundle['n_classes'], dropout=dropout)
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
        "notes": f"Inline reimplementation. LP+HP+ID mixing. Best epoch: {best_epoch}.",
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
    parser = argparse.ArgumentParser(description='Run ACM-GNN benchmark')
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

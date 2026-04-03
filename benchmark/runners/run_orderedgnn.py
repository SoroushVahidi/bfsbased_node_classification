"""
OrderedGNN runner - Ordered GNN: Ordering Message Passing to Deal with Heterophily and Over-smoothing
STATUS: PLANNED - Phase C integration
Source: https://github.com/LUMIA-Group/OrderedGNN
"""
import sys
import os
import argparse
import json

_BENCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_BENCH_DIR)
sys.path.insert(0, _BENCH_DIR)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'code', 'bfsbased_node_classification'))

SOURCE_REPO = "https://github.com/LUMIA-Group/OrderedGNN"
METHOD = "orderedgnn"

PLANNED_NOTES = (
    "OrderedGNN requires chunk-based ordered message passing with a specialized "
    "propagation scheme that partitions hidden dimensions into ordered chunks. "
    "Full integration is planned for Phase C. To run this baseline, clone the "
    "official repo and follow external_baselines/README.md instructions."
)


def run(args):
    output_dir = args.output_dir or os.path.join(_BENCH_DIR, 'results', 'per_run')
    os.makedirs(output_dir, exist_ok=True)
    tag = args.output_tag or 'default'
    out_file = os.path.join(
        output_dir,
        f"{METHOD}_{args.dataset}_split{args.split_id}_seed{args.seed}_{tag}.json"
    )

    result = {
        "method": METHOD,
        "dataset": args.dataset,
        "split_id": args.split_id,
        "seed": args.seed,
        "success": False,
        "val_acc": None,
        "test_acc": None,
        "runtime_sec": 0.0,
        "source_repo": SOURCE_REPO,
        "config_used": {},
        "stdout_log_path": None,
        "stderr_log_path": None,
        "notes": PLANNED_NOTES,
        "commit_or_version_if_available": "not-integrated",
    }

    tmp = out_file + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(result, f, indent=2)
    os.replace(tmp, out_file)

    print(f"[{METHOD}] PLANNED - not yet integrated. See notes. Output: {out_file}")
    print(f"  {PLANNED_NOTES}")
    return result


def main():
    parser = argparse.ArgumentParser(description='Run OrderedGNN benchmark (PLANNED STUB)')
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

    run(args)
    sys.exit(1)


if __name__ == '__main__':
    main()

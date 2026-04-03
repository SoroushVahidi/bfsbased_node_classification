"""
Master benchmark suite runner.
Runs all or selected baseline methods across datasets and splits.
"""
import sys
import os
import argparse
import json
import subprocess
import csv
import time
from datetime import datetime

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(_BENCH_DIR)

ALL_METHODS  = ["gprgnn", "gcnii", "h2gcn", "linkx", "fsgnn", "acmgnn", "orderedgnn", "djgnn"]
ALL_DATASETS = ["cora", "citeseer", "pubmed", "chameleon", "squirrel", "texas", "cornell", "wisconsin", "actor"]

METHOD_SCRIPTS = {
    m: os.path.join(_BENCH_DIR, "runners", f"run_{m}.py")
    for m in ALL_METHODS
}


def result_file_path(output_dir, method, dataset, split_id, seed, tag):
    return os.path.join(
        output_dir,
        f"{method}_{dataset}_split{split_id}_seed{seed}_{tag}.json"
    )


def run_one(method, dataset, split_id, seed, tag, split_dir, data_root, output_dir, force_rerun):
    script = METHOD_SCRIPTS[method]
    cmd = [
        sys.executable, script,
        "--dataset", dataset,
        "--split-id", str(split_id),
        "--seed", str(seed),
        "--output-tag", tag,
        "--output-dir", output_dir,
    ]
    if split_dir:
        cmd += ["--split-dir", split_dir]
    if data_root:
        cmd += ["--data-root", data_root]
    if force_rerun:
        cmd.append("--force-rerun")

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    out_file = result_file_path(output_dir, method, dataset, split_id, seed, tag)
    if os.path.exists(out_file):
        with open(out_file) as f:
            result = json.load(f)
    else:
        result = {
            "method": method,
            "dataset": dataset,
            "split_id": split_id,
            "seed": seed,
            "success": False,
            "val_acc": None,
            "test_acc": None,
            "runtime_sec": elapsed,
            "notes": f"Runner exited with code {proc.returncode}. stderr: {proc.stderr[:500]}",
        }

    if proc.stdout:
        print(proc.stdout, end="")
    if proc.returncode != 0 and proc.stderr:
        print(f"  STDERR: {proc.stderr[:300]}", file=sys.stderr)

    return result


def write_summary(results, summaries_dir, tag):
    os.makedirs(summaries_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(summaries_dir, f"benchmark_summary_{tag}_{ts}.json")
    tmp = json_path + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, json_path)

    csv_path = os.path.join(summaries_dir, f"benchmark_summary_{tag}_{ts}.csv")
    fields = ["method", "dataset", "split_id", "seed", "success", "val_acc", "test_acc", "runtime_sec", "notes"]
    tmp = csv_path + ".tmp"
    with open(tmp, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    os.replace(tmp, csv_path)

    return json_path, csv_path


def print_summary_table(results):
    print("\n" + "=" * 80)
    print(f"{'METHOD':<12} {'DATASET':<12} {'SPLIT':>5} {'SEED':>5} {'VAL':>7} {'TEST':>7} {'OK':>4} {'SEC':>7}")
    print("-" * 80)
    for r in results:
        ok   = "Y" if r.get("success") else "N"
        val  = f"{r['val_acc']:.4f}"  if r.get("val_acc")  is not None else "  N/A"
        test = f"{r['test_acc']:.4f}" if r.get("test_acc") is not None else "  N/A"
        sec  = f"{r.get('runtime_sec', 0):.1f}"
        print(f"{r['method']:<12} {r['dataset']:<12} {r['split_id']:>5} {r['seed']:>5} "
              f"{val:>7} {test:>7} {ok:>4} {sec:>7}")
    print("=" * 80)

    successes = [r for r in results if r.get("success")]
    print(f"\n{len(successes)}/{len(results)} runs succeeded.")


def main():
    parser = argparse.ArgumentParser(description='Run full benchmark suite')
    parser.add_argument('--methods',   nargs='+', default=ALL_METHODS,
                        choices=ALL_METHODS, help='Methods to run')
    parser.add_argument('--datasets',  nargs='+', default=['cora'],
                        choices=ALL_DATASETS, help='Datasets to run')
    parser.add_argument('--splits',    nargs='+', type=int, default=[0],
                        help='Split IDs to run')
    parser.add_argument('--seed',      type=int, default=42)
    parser.add_argument('--output-tag', default='suite')
    parser.add_argument('--split-dir', default=None)
    parser.add_argument('--data-root', default=None)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--force-rerun', action='store_true')
    args = parser.parse_args()

    output_dir    = args.output_dir or os.path.join(_BENCH_DIR, 'results', 'per_run')
    summaries_dir = os.path.join(_BENCH_DIR, 'results', 'summaries')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(summaries_dir, exist_ok=True)

    tag = args.output_tag

    combos = [
        (method, dataset, split_id)
        for method   in args.methods
        for dataset  in args.datasets
        for split_id in args.splits
    ]

    print(f"Running {len(combos)} combinations: "
          f"{len(args.methods)} methods x {len(args.datasets)} datasets x {len(args.splits)} splits")
    print(f"Methods:  {args.methods}")
    print(f"Datasets: {args.datasets}")
    print(f"Splits:   {args.splits}")
    print(f"Seed:     {args.seed}")
    print()

    results = []
    for i, (method, dataset, split_id) in enumerate(combos, 1):
        out_file = result_file_path(output_dir, method, dataset, split_id, args.seed, tag)
        if not args.force_rerun and os.path.exists(out_file):
            print(f"[{i}/{len(combos)}] SKIP {method} {dataset} split={split_id} (exists)")
            with open(out_file) as f:
                results.append(json.load(f))
            continue

        print(f"[{i}/{len(combos)}] Running {method} on {dataset} split={split_id} seed={args.seed} ...")
        result = run_one(
            method, dataset, split_id, args.seed, tag,
            args.split_dir, args.data_root, output_dir, args.force_rerun
        )
        results.append(result)

    json_path, csv_path = write_summary(results, summaries_dir, tag)
    print(f"\nSummary written to:\n  {json_path}\n  {csv_path}")
    print_summary_table(results)


if __name__ == '__main__':
    main()

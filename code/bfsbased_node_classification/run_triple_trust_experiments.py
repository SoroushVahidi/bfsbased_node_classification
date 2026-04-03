#!/usr/bin/env python3
"""Run TRIPLE_TRUST_SGC experiments with canonical split protocol.

This runner is non-destructive: it writes under a dedicated output tag.
"""
from __future__ import annotations

import argparse
import json

from prl_resubmission_runner import run_prl_resubmission


DEFAULT_METHODS = ["mlp_only", "selective_graph_correction", "triple_trust_sgc"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TRIPLE_TRUST_SGC experiments.")
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--datasets", nargs="+", default=["cora"])
    parser.add_argument("--splits", nargs="+", type=int, default=[0])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output-tag", default="triple_trust_sgc_smoke")
    args = parser.parse_args()

    out = run_prl_resubmission(
        datasets=args.datasets,
        split_ids=args.splits,
        repeats=args.repeats,
        methods=DEFAULT_METHODS,
        split_dir=args.split_dir,
        output_tag=args.output_tag,
    )
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()

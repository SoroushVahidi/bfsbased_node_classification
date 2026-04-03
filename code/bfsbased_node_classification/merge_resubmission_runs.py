#!/usr/bin/env python3
"""Merge multiple resubmission JSONL files and regenerate analysis outputs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from analyze_resubmission_results import analyze


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def merge_runs(inputs: List[Path], output_path: Path) -> Tuple[int, int]:
    merged: List[Dict[str, Any]] = []
    seen = set()
    for path in inputs:
        for rec in _read_jsonl(path):
            key = (
                rec.get("dataset"),
                rec.get("split_id"),
                rec.get("repeat_id"),
                rec.get("method"),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(rec)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for rec in merged:
            f.write(json.dumps(rec) + "\n")
    return len(merged), len(seen)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge resubmission run files.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-tag", required=True)
    args = parser.parse_args()

    inputs = [Path(p) for p in args.inputs]
    repo_root = Path(__file__).resolve().parents[2]
    out_path = repo_root / "logs" / "resubmission_runs" / f"comparison_runs_{args.output_tag}.jsonl"
    n_rows, _ = merge_runs(inputs, out_path)
    print(f"Merged {n_rows} records into {out_path}")
    artifacts = analyze(str(out_path), args.output_tag)
    print(json.dumps(artifacts, indent=2))


if __name__ == "__main__":
    main()

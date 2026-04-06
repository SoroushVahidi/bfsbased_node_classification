#!/usr/bin/env python3
"""Create a lightweight manifest for off-Git sweep artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Iterable, Tuple


def _iso_utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _git_commit(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"], text=True
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _scan_path(path: Path) -> Tuple[int, int]:
    if not path.exists():
        return 0, 0
    if path.is_file():
        return 1, path.stat().st_size
    file_count = 0
    total_bytes = 0
    for root, _, files in os.walk(path):
        for name in files:
            fpath = Path(root) / name
            try:
                file_count += 1
                total_bytes += fpath.stat().st_size
            except OSError:
                continue
    return file_count, total_bytes


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _split_csv(value: str) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _split_int_csv(value: str) -> list[int]:
    out: list[int] = []
    for part in _split_csv(value):
        out.append(int(part))
    return out


def _default_manifest_name(sweep_name: str) -> str:
    date_str = dt.datetime.now().strftime("%Y%m%d")
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in sweep_name)
    return f"manifest_{safe}_{date_str}.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a JSON manifest for raw sweep artifacts.")
    p.add_argument("--sweep-name", required=True)
    p.add_argument("--runner-script", required=True)
    p.add_argument("--raw-output-dir", action="append", required=True, help="Can be passed multiple times.")
    p.add_argument("--tarball", default="", help="Optional archive path to hash and size.")
    p.add_argument("--datasets", default="", help="Comma-separated list.")
    p.add_argument("--methods", default="", help="Comma-separated list.")
    p.add_argument("--split-ids", default="", help="Comma-separated ints.")
    p.add_argument("--job-ids", default="", help="Comma-separated job IDs.")
    p.add_argument("--storage-location", default="TBD")
    p.add_argument("--notes", default="")
    p.add_argument("--manifest-out", default="", help="Optional explicit output manifest path.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    manifests_dir = repo_root / "run_metadata" / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    raw_dirs = [Path(p) for p in args.raw_output_dir]
    total_files = 0
    total_bytes = 0
    for path in raw_dirs:
        files, size = _scan_path(path)
        total_files += files
        total_bytes += size

    tarball = Path(args.tarball) if args.tarball else None
    archive_filename = "TBD"
    archive_sha256 = "TBD"
    archive_size_bytes = None
    if tarball:
        archive_filename = str(tarball)
        if tarball.exists() and tarball.is_file():
            archive_sha256 = _sha256_file(tarball)
            archive_size_bytes = tarball.stat().st_size
        else:
            archive_sha256 = "MISSING"

    manifest = {
        "sweep_name": args.sweep_name,
        "created_at": _iso_utc_now(),
        "git_commit": _git_commit(repo_root),
        "runner_script": args.runner_script,
        "datasets": _split_csv(args.datasets),
        "methods": _split_csv(args.methods),
        "split_ids": _split_int_csv(args.split_ids),
        "job_ids": _split_csv(args.job_ids),
        "raw_output_dir": str(raw_dirs[0]) if raw_dirs else "",
        "raw_output_dirs": [str(p) for p in raw_dirs],
        "raw_file_count": total_files,
        "raw_total_bytes": total_bytes,
        "archive_filename": archive_filename,
        "archive_sha256": archive_sha256,
        "archive_size_bytes": archive_size_bytes,
        "storage_location": args.storage_location,
        "notes": args.notes,
    }

    out_path = Path(args.manifest_out) if args.manifest_out else (manifests_dir / _default_manifest_name(args.sweep_name))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote manifest: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

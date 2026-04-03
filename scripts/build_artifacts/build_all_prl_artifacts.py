#!/usr/bin/env python3
"""
Regenerate all cheap PRL manuscript artifacts from canonical logs/ and results_prl/.

Run from repo root:
  python3 scripts/build_artifacts/build_all_prl_artifacts.py
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _run_script(path: Path) -> None:
    r = subprocess.run([sys.executable, str(path)], cwd=str(ROOT), check=False)
    if r.returncode != 0:
        raise SystemExit(f"Script failed ({r.returncode}): {path}")


def mirror_prl_final_additions() -> None:
    """Copy prl_final_additions/{tables,figures} into root tables/figures for zip-friendly layout."""
    src_tables = ROOT / "prl_final_additions" / "tables"
    dst_tables = ROOT / "tables" / "prl_final_additions"
    dst_tables.mkdir(parents=True, exist_ok=True)
    if src_tables.exists():
        for p in src_tables.glob("*"):
            if p.is_file():
                shutil.copy2(p, dst_tables / p.name)

    src_fig = ROOT / "prl_final_additions" / "figures"
    dst_fig = ROOT / "figures" / "prl_final_additions"
    dst_fig.mkdir(parents=True, exist_ok=True)
    if src_fig.exists():
        for p in src_fig.glob("*"):
            if p.is_file():
                shutil.copy2(p, dst_fig / p.name)


def write_artifact_inventory() -> Path:
    """Machine-readable inventory of canonical PRL-related paths."""
    globs = [
        "logs/manuscript_*final_validation*.csv",
        "logs/manuscript_*final_validation*.md",
        "logs/comparison_*final_validation*.jsonl",
        "logs/comparison_summary_manuscript_final_validation.json",
        "results_prl/*",
        "tables/prl_final_additions/*",
        "figures/prl_final_additions/*",
        "reports/prl_final_additions/*",
    ]
    entries = []
    seen: set[str] = set()

    def add_path(p: Path, category: str | None = None) -> None:
        if not p.is_file() or ".git" in str(p):
            return
        rp = str(p.relative_to(ROOT))
        if rp in seen:
            return
        seen.add(rp)
        try:
            st = p.stat()
            entries.append(
                {
                    "path": rp,
                    "bytes": st.st_size,
                    "category": category or rp.split("/")[0],
                }
            )
        except OSError:
            pass

    for pat in globs:
        for p in sorted(ROOT.glob(pat)):
            add_path(p)

    base = ROOT / "prl_final_additions"
    if base.exists():
        for p in sorted(base.rglob("*")):
            add_path(p, "prl_final_additions")

    out = ROOT / "reports" / "prl_final_additions" / "artifact_inventory.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(ROOT),
        "entries": sorted(entries, key=lambda e: e["path"]),
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    csv_path = out.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "bytes", "category"])
        for e in payload["entries"]:
            w.writerow([e["path"], e["bytes"], e["category"]])
    return out


def main() -> None:
    scripts = ROOT / "prl_final_additions" / "scripts"
    order = [
        "build_prl_tables.py",
        "compute_subsection3_crossrun_correlations.py",
        "generate_prl_benchmark_figure.py",
        "generate_correction_behavior_priority_figure.py",
    ]
    for name in order:
        path = scripts / name
        if path.exists():
            print("Running", path.relative_to(ROOT))
            _run_script(path)
        else:
            print("Skip missing", path)

    mirror_prl_final_additions()
    inv = write_artifact_inventory()
    print("Wrote inventory:", inv.relative_to(ROOT))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate a compact markdown summary from Slurm .out/.err files."""

from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path


ERROR_HINTS = (
    "traceback",
    "error",
    "exception",
    "failed",
    "segmentation fault",
    "oom",
    "out of memory",
    "killed",
    "timeout",
)
WARN_HINTS = ("warning", "warn", "userwarning", "deprecated")
SUCCESS_HINTS = ("[done]", "completed", "all checks passed", "finished")
RUNNING_HINTS = ("running",)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize Slurm stdout/stderr files into markdown.")
    p.add_argument("--slurm-dir", default="slurm")
    p.add_argument("--glob", default="*.out,*.err", help="Comma-separated globs.")
    p.add_argument("--max-lines", type=int, default=6, help="Max key lines per file.")
    p.add_argument(
        "--include-success",
        action="store_true",
        help="Include success files in the summary table (default: show non-success only).",
    )
    p.add_argument("--output", default="", help="Optional output markdown path.")
    return p.parse_args()


def detect_job_id(name: str, text: str) -> str:
    m = re.search(r"_(\d{5,})_(\d+)\.(out|err)$", name)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    m = re.search(r"\bSLURM_JOB_ID[:=]\s*(\d+)", text)
    if m:
        return m.group(1)
    return "unknown"


def classify(text_lower: str) -> str:
    if any(k in text_lower for k in ERROR_HINTS):
        if "timeout" in text_lower:
            return "timeout"
        return "error"
    if any(k in text_lower for k in WARN_HINTS):
        return "warning"
    if any(k in text_lower for k in SUCCESS_HINTS):
        return "success"
    if any(k in text_lower for k in RUNNING_HINTS):
        return "running"
    return "unknown"


def recommend(status: str) -> str:
    if status == "success":
        return "No action."
    if status == "warning":
        return "Review warnings; rerun only if warnings affect metrics."
    if status == "timeout":
        return "Backfill timed-out shard with same or follow-up output tag."
    if status == "running":
        return "Wait and re-run summary after completion."
    if status == "error":
        return "Inspect full log, fix root cause, and rerun shard."
    return "Inspect manually."


def extract_key_lines(lines: list[str], max_lines: int) -> list[str]:
    picked: list[str] = []
    for raw in lines:
        s = raw.strip()
        if not s:
            continue
        low = s.lower()
        if any(k in low for k in ERROR_HINTS + WARN_HINTS + SUCCESS_HINTS):
            picked.append(s)
        if len(picked) >= max_lines:
            break
    if not picked and lines:
        picked = [ln.strip() for ln in lines[-max_lines:] if ln.strip()]
    return picked[:max_lines]


def main() -> int:
    args = parse_args()
    slurm_dir = Path(args.slurm_dir)
    patterns = [g.strip() for g in args.glob.split(",") if g.strip()]

    files: list[Path] = []
    for pat in patterns:
        files.extend(sorted(slurm_dir.glob(pat)))
    files = sorted(set(files))

    ts = dt.datetime.now().strftime("%Y%m%d")
    default_out = Path("reports") / f"slurm_runtime_notes_{ts}.md"
    out_path = Path(args.output) if args.output else default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lines = text.splitlines()
        text_lower = text.lower()
        status = classify(text_lower)
        key_lines = extract_key_lines(lines, args.max_lines)
        job_id = detect_job_id(path.name, text)
        rows.append(
            {
                "job_id": job_id,
                "file": str(path),
                "status": status,
                "key_lines": key_lines,
                "action": recommend(status),
            }
        )

    non_success = [r for r in rows if r["status"] != "success"]
    visible = rows if args.include_success else non_success

    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"# Slurm runtime notes ({dt.date.today().isoformat()})\n\n")
        f.write(f"- Source directory: `{slurm_dir}`\n")
        f.write(f"- Files scanned: {len(rows)}\n")
        f.write(f"- Rows shown: {len(visible)} (`--include-success` to show all)\n\n")
        f.write("| Job ID | File | Likely status | Recommended action |\n")
        f.write("|---|---|---|---|\n")
        for r in visible:
            f.write(f"| `{r['job_id']}` | `{Path(r['file']).name}` | {r['status']} | {r['action']} |\n")
        f.write("\n## Key warning/error lines\n\n")
        for r in visible:
            if not r["key_lines"]:
                continue
            f.write(f"### `{Path(r['file']).name}` ({r['status']})\n\n")
            for line in r["key_lines"]:
                safe = line.replace("|", "\\|")
                f.write(f"- {safe}\n")
            f.write("\n")

    print(f"Wrote report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

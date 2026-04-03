#!/usr/bin/env python3
"""Pearson correlations from results_prl/threshold_sensitivity_crossruns.jsonl (cross-run only)."""
from __future__ import annotations

import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JSONL = ROOT / "results_prl" / "threshold_sensitivity_crossruns.jsonl"
OUT = ROOT / "prl_final_additions" / "tables" / "prl_subsection3_crossrun_correlations.csv"
MIRROR = ROOT / "tables" / "prl_final_additions"


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = sum((x - mx) ** 2 for x in xs) ** 0.5
    deny = sum((y - my) ** 2 for y in ys) ** 0.5
    if denx == 0.0 or deny == 0.0:
        return float("nan")
    return num / (denx * deny)


def main() -> None:
    by_ds: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    for line in JSONL.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        by_ds[r["dataset"]].append(
            (
                float(r["threshold"]),
                float(r["delta_vs_mlp"]),
                float(r["uncertain_fraction_test"]),
                float(r.get("changed_fraction_test", float("nan"))),
            )
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "dataset",
                "n_runs",
                "corr_threshold_vs_uncertain_fraction",
                "corr_threshold_vs_delta_vs_mlp",
                "corr_uncertain_fraction_vs_delta_vs_mlp",
                "corr_threshold_vs_changed_fraction_test",
            ]
        )
        for ds in sorted(by_ds):
            pts = by_ds[ds]
            thr = [p[0] for p in pts]
            dlt = [p[1] for p in pts]
            unc = [p[2] for p in pts]
            chg = [p[3] for p in pts]
            c_tc = (
                f"{pearson(thr, chg):.3f}"
                if not any(math.isnan(x) for x in chg)
                else ""
            )
            w.writerow(
                [
                    ds,
                    len(pts),
                    f"{pearson(thr, unc):.3f}",
                    f"{pearson(thr, dlt):.3f}",
                    f"{pearson(unc, dlt):.3f}",
                    c_tc,
                ]
            )
    MIRROR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT, MIRROR / OUT.name)
    print("Wrote", OUT)


if __name__ == "__main__":
    main()

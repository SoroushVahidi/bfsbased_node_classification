#!/usr/bin/env python3
"""Regenerate PRL tables from canonical logs/ CSVs (non-destructive to logs)."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "logs"
OUT = ROOT / "prl_final_additions"
TBL = OUT / "tables"


def main() -> None:
    gain = LOGS / "manuscript_gain_over_mlp_final_validation.csv"
    regime = LOGS / "manuscript_regime_analysis_final_validation.csv"
    corr = LOGS / "manuscript_correction_behavior_final_validation.csv"

    g_rows = {r["dataset"]: r for r in csv.DictReader(gain.open())}
    r_rows = {r["dataset"]: r for r in csv.DictReader(regime.open())}

    keys = sorted(set(g_rows) & set(r_rows))
    merged_path = TBL / "prl_main_benchmark_merged.csv"
    TBL.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset",
        "mlp_mean",
        "prop_mean",
        "sgc_mean",
        "delta_sgc_minus_mlp",
        "n_splits_sgc_wins",
        "n_splits_mlp_wins",
        "n_splits_ties",
        "avg_uncertain_fraction",
        "avg_changed_fraction",
    ]
    with merged_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for ds in keys:
            g, r = g_rows[ds], r_rows[ds]
            d_raw = g["delta"].replace("+", "")
            w.writerow(
                {
                    "dataset": ds,
                    "mlp_mean": r["mlp_mean"],
                    "prop_mean": r["prop_mean"],
                    "sgc_mean": g["sgc_mean"],
                    "delta_sgc_minus_mlp": d_raw,
                    "n_splits_sgc_wins": g["n_splits_sgc_wins"],
                    "n_splits_mlp_wins": g["n_splits_mlp_wins"],
                    "n_splits_ties": g["n_splits_ties"],
                    "avg_uncertain_fraction": g["avg_uncertain_fraction"],
                    "avg_changed_fraction": g["avg_changed_fraction"],
                }
            )
    rows = list(csv.DictReader(merged_path.open()))
    for row in rows:
        d = float(row["delta_sgc_minus_mlp"])
        if abs(d) < 1e-9:
            row["delta_sgc_minus_mlp"] = "+0.0000"
        else:
            row["delta_sgc_minus_mlp"] = f"{d:+.4f}"
    with merged_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    wtl_path = TBL / "prl_split_win_tie_loss.csv"
    with wtl_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "sgc_wins", "mlp_wins", "ties", "n_comparisons"])
        for ds in keys:
            g = g_rows[ds]
            w.writerow(
                [
                    ds,
                    g["n_splits_sgc_wins"],
                    g["n_splits_mlp_wins"],
                    g["n_splits_ties"],
                    int(g["n_splits_sgc_wins"])
                    + int(g["n_splits_mlp_wins"])
                    + int(g["n_splits_ties"]),
                ]
            )

    compact = TBL / "prl_correction_behavior_compact.csv"
    shutil.copy2(corr, compact)

    kn = {
        "benchmark_delta_vs_mlp": {
            r["dataset"]: f"{float(r['delta_sgc_minus_mlp']):+.4f}" for r in rows
        },
        "win_tie_loss": {
            r["dataset"]: {
                "sgc": g_rows[r["dataset"]]["n_splits_sgc_wins"],
                "mlp": g_rows[r["dataset"]]["n_splits_mlp_wins"],
                "ties": g_rows[r["dataset"]]["n_splits_ties"],
            }
            for r in rows
        },
    }
    (OUT / "key_numbers.json").write_text(json.dumps(kn, indent=2) + "\n")

    print("Wrote", merged_path)
    print("Wrote", wtl_path)
    print("Wrote", compact)
    print("Wrote", OUT / "key_numbers.json")

    mirror = ROOT / "tables" / "prl_final_additions"
    mirror.mkdir(parents=True, exist_ok=True)
    for fn in (
        "prl_main_benchmark_merged.csv",
        "prl_split_win_tie_loss.csv",
        "prl_correction_behavior_compact.csv",
    ):
        shutil.copy2(TBL / fn, mirror / fn)
    shutil.copy2(merged_path, mirror / "prl_benchmark_summary.csv")
    print("Mirrored to", mirror)


if __name__ == "__main__":
    main()

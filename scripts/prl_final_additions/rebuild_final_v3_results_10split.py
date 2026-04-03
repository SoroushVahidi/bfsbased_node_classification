#!/usr/bin/env python3
"""
Rebuild reports/final_method_v3_results.csv from the frozen 10-split table export.

Source: reports/archive/superseded_final_tables_prl/final_tables_prl.csv

Optional enrichment for FINAL_V3 splits 0–4: merge frac_confident, frac_unreliable,
tau, rho, runtime_sec from reports/archive/final_method_v3_results_5split.csv when present.

No PyTorch / training.
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_TABLE = ROOT / "reports/archive/superseded_final_tables_prl/final_tables_prl.csv"
ENRICH = ROOT / "reports/archive/final_method_v3_results_5split.csv"
OUT = ROOT / "reports/final_method_v3_results.csv"

METHOD_IN = {"SGC_V1": "BASELINE_SGC_V1", "V2_MULTI": "V2_MULTIBRANCH", "FINAL_V3": "FINAL_V3"}
ROW_ORDER = ["MLP_ONLY", "BASELINE_SGC_V1", "V2_MULTIBRANCH", "FINAL_V3"]
FIELDS = [
    "dataset",
    "split_id",
    "seed",
    "method",
    "test_acc",
    "delta_vs_mlp",
    "n_helped",
    "n_hurt",
    "correction_precision",
    "frac_confident",
    "frac_unreliable",
    "frac_corrected",
    "tau",
    "rho",
    "runtime_sec",
]


def _float(s: str) -> float:
    return float(s) if s not in ("", None) else float("nan")


def _int_or_empty(s: str):
    s = (s or "").strip()
    if not s:
        return ""
    return int(float(s))


def load_enrichment() -> dict[tuple[str, int], dict]:
    if not ENRICH.is_file():
        return {}
    out: dict[tuple[str, int], dict] = {}
    with ENRICH.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["method"] != "FINAL_V3":
                continue
            key = (row["dataset"].lower(), int(row["split_id"]))
            out[key] = row
    return out


def main() -> None:
    if not ARCHIVE_TABLE.is_file():
        raise SystemExit(f"Missing {ARCHIVE_TABLE}")

    enrich = load_enrichment()
    raw_rows = list(csv.DictReader(ARCHIVE_TABLE.open(newline="", encoding="utf-8")))

    def sort_key(r):
        m = METHOD_IN[r["method"]]
        return (
            r["dataset"].lower(),
            int(r["split_id"]),
            int(r["seed"]),
            ROW_ORDER.index(m),
        )

    raw_rows.sort(key=sort_key)

    records: list[dict] = []
    seen_mlp: set[tuple[str, int, int]] = set()

    for row in raw_rows:
        ds = row["dataset"].strip().lower()
        sid = int(row["split_id"])
        seed = int(row["seed"])
        key = (ds, sid, seed)

        if key not in seen_mlp:
            mlp_acc = _float(row["mlp_acc"])
            records.append(
                {
                    "dataset": ds,
                    "split_id": sid,
                    "seed": seed,
                    "method": "MLP_ONLY",
                    "test_acc": mlp_acc,
                    "delta_vs_mlp": 0.0,
                    "n_helped": 0,
                    "n_hurt": 0,
                    "correction_precision": 1.0,
                    "frac_confident": 1.0,
                    "frac_unreliable": 0.0,
                    "frac_corrected": 0.0,
                    "tau": "",
                    "rho": "",
                    "runtime_sec": 0.0,
                }
            )
            seen_mlp.add(key)

        raw_m = row["method"]
        method = METHOD_IN[raw_m]
        test_acc = _float(row["test_acc"])
        delta = _float(row["delta"])
        fc_raw = row.get("frac_corrected", "").strip()
        fc = float(fc_raw) if fc_raw else 0.0

        nh = _int_or_empty(row.get("n_helped", ""))
        hh = _int_or_empty(row.get("n_hurt", ""))
        pr_raw = row.get("precision", "").strip()
        prec = float(pr_raw) if pr_raw else ""

        rec = {
            "dataset": ds,
            "split_id": sid,
            "seed": seed,
            "method": method,
            "test_acc": test_acc,
            "delta_vs_mlp": delta,
            "n_helped": nh,
            "n_hurt": hh,
            "correction_precision": prec,
            "frac_confident": "",
            "frac_unreliable": "",
            "frac_corrected": fc,
            "tau": "",
            "rho": "",
            "runtime_sec": "",
        }

        if method != "FINAL_V3":
            records.append(rec)
            continue

        ek = enrich.get((ds, sid))
        if ek:
            for fld in ("frac_confident", "frac_unreliable", "tau", "rho", "runtime_sec"):
                rec[fld] = ek.get(fld, "") or rec[fld]
            if rec["correction_precision"] == "" and ek.get("correction_precision"):
                rec["correction_precision"] = float(ek["correction_precision"])
        records.append(rec)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in FIELDS})

    print(f"Wrote {len(records)} rows to {OUT}")


if __name__ == "__main__":
    main()

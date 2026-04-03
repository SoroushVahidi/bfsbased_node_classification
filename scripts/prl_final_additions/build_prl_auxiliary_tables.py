#!/usr/bin/env python3
"""
Manuscript-facing auxiliary tables (no training).

Writes:
  tables/experimental_setup_prl.{csv,md}
  tables/ablation_prl.{csv,md}
  tables/sensitivity_prl.{csv,md}

Experimental setup: dataset stats + homophily from logs/manuscript_regime_analysis_final_validation_main.csv.

Ablation / sensitivity: FINAL_V3 row recomputed from reports/final_method_v3_results.csv;
  other ablation rows and the ρ-sweep from the historical snapshot
  reports/archive/superseded_final_tables_prl/final_tables_prl.md (Table 7–8),
  clearly labeled — not re-run here.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "reports" / "final_method_v3_results.csv"
REGIME_MAIN = ROOT / "logs" / "manuscript_regime_analysis_final_validation_main.csv"
REGIME_JOB2 = ROOT / "logs" / "regime_analysis_manuscript_final_validation_job2.csv"
REGIME_FALLBACK = ROOT / "logs" / "manuscript_regime_analysis_final_validation.csv"
TBL = ROOT / "tables"

PRL_DATASETS = ["chameleon", "citeseer", "cora", "pubmed", "texas", "wisconsin"]
ABLATION_SUBSET = ["chameleon", "cora", "texas"]

# Table 7–8 in reports/archive/superseded_final_tables_prl/final_tables_prl.md (historical export).
ARCHIVED_ABLATION = [
    ("A1_NO_RELIABILITY", 0.22, 2.58, 0.54, 1.11),
    ("A2_PROFILE_A_ONLY", 0.04, 1.41, 1.08, 0.84),
    ("A3_PROFILE_B_ONLY", -0.09, 2.49, 1.08, 1.16),
]
ARCHIVED_SENSITIVITY = [
    # rho, chameleon_dpp, cora_dpp, texas_dpp, mean_dpp, total_hurt
    (0.0, 0.15, 2.21, 0.90, 1.09, 11),
    (0.2, 0.15, 2.21, 0.90, 1.09, 11),
    (0.3, 0.15, 2.08, 0.90, 1.04, 10),
    (0.4, 0.15, 1.48, 0.00, 0.54, 5),
    (0.5, -0.07, 0.47, 0.00, 0.13, 1),
    (0.6, 0.00, 0.00, 0.00, 0.00, 0),
    (0.7, 0.00, 0.00, 0.00, 0.00, 0),
]


def load_regime() -> dict[str, dict[str, str]]:
    """Merge regime rows: main CSV first; fill missing datasets / N/A homophily from job2 and fallback."""
    out: dict[str, dict[str, str]] = {}

    def ingest(path: Path, overwrite_homophily: bool = False) -> None:
        if not path.is_file():
            return
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ds = row["dataset"].strip().lower()
                h = (row.get("homophily") or "").strip()
                if ds not in out:
                    out[ds] = dict(row)
                    continue
                cur = out[ds]
                cur_h = (cur.get("homophily") or "").strip()
                if overwrite_homophily or cur_h.upper() == "N/A" or cur_h == "":
                    if h and h.upper() != "N/A":
                        cur["homophily"] = h
                for k, v in row.items():
                    if k == "homophily":
                        continue
                    if not (cur.get(k) or "").strip() and (v or "").strip():
                        cur[k] = v

    ingest(REGIME_MAIN)
    ingest(REGIME_JOB2)
    ingest(REGIME_FALLBACK)
    return out


def mean_delta_v3(rows: list[dict], datasets: list[str]) -> dict[str, float]:
    by_ds: dict[str, list[float]] = {d: [] for d in datasets}
    for r in rows:
        if r["method"] != "FINAL_V3":
            continue
        if r["dataset"] in by_ds:
            by_ds[r["dataset"]].append(float(r["delta_vs_mlp"]))
    return {d: float(np.mean(by_ds[d]) * 100.0) for d in datasets}


def write_experimental_setup(regime: dict[str, dict[str, str]]) -> None:
    split_txt = "60%/20%/20% train/val/test; 10 fixed splits (indices 0–9); GEO-GCN `.npz` in `data/splits/`"
    methods = "MLP_ONLY; BASELINE_SGC_V1; V2_MULTIBRANCH; FINAL_V3"
    header = [
        "dataset",
        "num_nodes",
        "num_classes",
        "edge_homophily",
        "split_protocol",
        "compared_methods",
    ]
    csv_path = TBL / "experimental_setup_prl.csv"
    md_path = TBL / "experimental_setup_prl.md"
    TBL.mkdir(parents=True, exist_ok=True)

    lines_md = [
        "# Experimental setup (PRL main benchmark)",
        "",
        "Dataset statistics and edge homophily are merged from "
        "`logs/manuscript_regime_analysis_final_validation_main.csv` and, where needed, "
        "`logs/regime_analysis_manuscript_final_validation_job2.csv` (PubMed homophily), "
        "using the same homophily column as `figures/correction_rate_vs_homophily.png`.",
        "",
        f"**Split protocol:** {split_txt}",
        "",
        f"**Compared methods (main table):** {methods.replace('; ', ', ')}",
        "",
        "| Dataset | # nodes | # classes | Edge homophily | Split protocol | Compared methods |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(header)
        for ds in sorted(PRL_DATASETS):
            row = regime[ds]
            nn = row["num_nodes"]
            nc = row["num_classes"]
            h = row["homophily"]
            w.writerow([ds, nn, nc, h, split_txt, methods])
            lines_md.append(
                f"| {ds.capitalize()} | {nn} | {nc} | {h} | {split_txt} | {methods} |"
            )
    lines_md.append("")
    md_path.write_text("\n".join(lines_md), encoding="utf-8")
    print("Wrote", csv_path, md_path)


def write_ablation(rows: list[dict]) -> None:
    dpp = mean_delta_v3(rows, ABLATION_SUBSET)
    mean3 = float(np.mean(list(dpp.values())))
    csv_path = TBL / "ablation_prl.csv"
    md_path = TBL / "ablation_prl.md"
    header = [
        "variant",
        "chameleon_delta_pp",
        "cora_delta_pp",
        "texas_delta_pp",
        "mean_delta_pp_cham_cora_texas",
        "source",
    ]
    lines_md = [
        "# Ablation-style variants (PRL supplementary)",
        "",
        "**FINAL_V3** row: mean test Δ vs MLP (percentage points) over **10 splits**, recomputed from "
        "`reports/final_method_v3_results.csv` for Chameleon, Cora, and Texas.",
        "",
        "**Other rows:** frozen snapshot from `reports/archive/superseded_final_tables_prl/final_tables_prl.md` "
        "(Table 7) from the same historical validation campaign — **not** re-executed against the current "
        "rebuild merge. Use for qualitative discussion; a full 10-split ablation re-run would require new experiments.",
        "",
        "| Variant | Chameleon Δ (pp) | Cora Δ (pp) | Texas Δ (pp) | Mean (3 ds) | Source |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(header)
        src_full = "recomputed from final_method_v3_results.csv (10 splits)"
        w.writerow(
            [
                "FINAL_V3",
                f"{dpp['chameleon']:.2f}",
                f"{dpp['cora']:.2f}",
                f"{dpp['texas']:.2f}",
                f"{mean3:.2f}",
                src_full,
            ]
        )
        lines_md.append(
            f"| FINAL_V3 | {dpp['chameleon']:+.2f} | {dpp['cora']:+.2f} | {dpp['texas']:+.2f} | {mean3:+.2f} | {src_full} |"
        )
        src_arch = "archived final_tables_prl.md Table 7 (historical)"
        for name, a, b, c, m in ARCHIVED_ABLATION:
            w.writerow([name, f"{a:.2f}", f"{b:.2f}", f"{c:.2f}", f"{m:.2f}", src_arch])
            lines_md.append(
                f"| {name} | {a:+.2f} | {b:+.2f} | {c:+.2f} | {m:+.2f} | {src_arch} |"
            )
    lines_md.append("")
    md_path.write_text("\n".join(lines_md), encoding="utf-8")
    print("Wrote", csv_path, md_path)


def write_sensitivity() -> None:
    csv_path = TBL / "sensitivity_prl.csv"
    md_path = TBL / "sensitivity_prl.md"
    header = [
        "rho",
        "chameleon_delta_pp",
        "cora_delta_pp",
        "texas_delta_pp",
        "mean_delta_pp",
        "total_hurt_nodes",
        "source",
    ]
    src = "archived final_tables_prl.md Table 8 (3-split ρ grid; historical)"
    lines_md = [
        "# Reliability threshold ρ — sensitivity (PRL supplementary)",
        "",
        "Mean test Δ vs MLP (percentage points) on **Chameleon, Cora, Texas** when ρ is fixed in a grid "
        "and other hyperparameters follow the historical evaluation protocol.",
        "",
        "**Source:** `reports/archive/superseded_final_tables_prl/final_tables_prl.md` (Table 8). "
        "This is a **coarse 3-split** sensitivity snapshot, **not** the full 10-split aggregate.",
        "",
        "**Reading:** Very low ρ (e.g. 0.0) allows almost all uncertain nodes through the reliability check, "
        "increasing harmful corrections (`total_hurt_nodes`). Around **ρ ≈ 0.3–0.4** mean Δ stays strongest "
        "before collapsing; **ρ ≥ 0.5** suppresses graph correction heavily (means approach 0).",
        "",
        "| ρ | Chameleon Δ (pp) | Cora Δ (pp) | Texas Δ (pp) | Mean | Total hurt |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(header)
        for row in ARCHIVED_SENSITIVITY:
            rho, a, b, c, m, h = row
            w.writerow([rho, f"{a:.2f}", f"{b:.2f}", f"{c:.2f}", f"{m:.2f}", h, src])
            lines_md.append(
                f"| {rho} | {a:+.2f} | {b:+.2f} | {c:+.2f} | {m:+.2f} | {h} |"
            )
    lines_md.append("")
    md_path.write_text("\n".join(lines_md), encoding="utf-8")
    print("Wrote", csv_path, md_path)


def main() -> None:
    if not RESULTS.is_file():
        raise SystemExit(f"Missing {RESULTS}")
    if not REGIME_MAIN.is_file():
        raise SystemExit(f"Missing {REGIME_MAIN}")

    regime = load_regime()
    for ds in PRL_DATASETS:
        if ds not in regime:
            raise SystemExit(f"Regime merge missing dataset {ds}")
        h = (regime[ds].get("homophily") or "").strip()
        if not h or h.upper() == "N/A":
            raise SystemExit(f"Regime merge missing homophily for {ds}")

    with RESULTS.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    write_experimental_setup(regime)
    write_ablation(rows)
    write_sensitivity()


if __name__ == "__main__":
    main()

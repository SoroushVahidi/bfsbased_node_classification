#!/usr/bin/env python3
"""
Generate markdown summary and comparison tables from MS_HSGC targeted experiment CSVs.

Usage (after running run_ms_hsgc_evaluation.py with --output-tag targeted_10split):
    python3 code/bfsbased_node_classification/generate_ms_hsgc_summary.py \
        --tag targeted_10split

Outputs:
    reports/ms_hsgc_targeted_10split_summary.md
    tables/ms_hsgc_targeted_10split_comparison.md
    tables/ms_hsgc_h1max_sensitivity.md

Status: EXPERIMENTAL — does NOT touch canonical FINAL_V3 paths.
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from datetime import date
from typing import Dict, List, Optional


def _read_csv(path: str) -> List[Dict]:
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _flt(v, default=float("nan")):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _fmt(v: float, digits: int = 4, signed: bool = False) -> str:
    if v != v:  # nan check
        return "N/A"
    fmt = f"{{:+.{digits}f}}" if signed else f"{{:.{digits}f}}"
    return fmt.format(v)


def _mean_std(vals: List[float]):
    import math
    if not vals:
        return float("nan"), float("nan")
    n = len(vals)
    mu = sum(vals) / n
    if n < 2:
        return mu, 0.0
    var = sum((x - mu) ** 2 for x in vals) / (n - 1)
    return mu, math.sqrt(var)


def _classify_outcome(datasets_delta: Dict[str, float], datasets_hurt: Dict[str, float]) -> str:
    """Classify overall outcome of MS_HSGC vs FINAL_V3."""
    deltas = [v for v in datasets_delta.values() if v == v]
    if not deltas:
        return "insufficient data"
    mean_delta = sum(deltas) / len(deltas)
    # Check for safety issues (large negative on homophilic datasets)
    max_hurt = max(datasets_hurt.values(), default=0.0)
    if mean_delta < -0.005 and max_hurt > 3:
        return "over-correcting / unsafe"
    elif mean_delta > 0.003:
        return "balanced / ready for broader benchmark"
    elif mean_delta > -0.003:
        return "promising but still over-conservative"
    else:
        return "not promising"


def generate_summary(
    results_path: str,
    delta_path: str,
    sensitivity_path: str,
    summary_out: str,
    comparison_table_out: str,
    sensitivity_table_out: str,
    tag: str,
) -> None:
    records = _read_csv(results_path)
    delta_records = _read_csv(delta_path)
    sensitivity_records = _read_csv(sensitivity_path)

    # -------------------------------------------------------------------
    # Organise records by method / dataset
    # -------------------------------------------------------------------
    by_md: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    by_md_delta_mlp: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    ms_diag: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for r in records:
        method = r.get("method", "")
        ds = r.get("dataset", "")
        ta = _flt(r.get("test_acc"))
        if ta == ta:
            by_md[method][ds].append(ta)
        dm = _flt(r.get("delta_vs_mlp"))
        if dm == dm:
            by_md_delta_mlp[method][ds].append(dm)
        if method == "MS_HSGC":
            for field in [
                "frac_corrected_1hop", "frac_corrected_2hop", "frac_mlp_only_uncertain",
                "n_helped", "n_hurt", "net_help", "correction_precision",
                "blocked_by_h1_only", "blocked_by_r1", "blocked_by_r2", "blocked_by_delta",
                "mean_H1", "mean_H2", "mean_DeltaH",
            ]:
                v = _flt(r.get(field))
                if v == v:
                    ms_diag[field][ds].append(v)

    # MS_HSGC vs FINAL_V3 deltas
    ds_delta_v3: Dict[str, List[float]] = defaultdict(list)
    for r in delta_records:
        d = _flt(r.get("delta_ms_hsgc_vs_final_v3"))
        if d == d:
            ds_delta_v3[r["dataset"]].append(d)

    # Datasets in deterministic order
    datasets = sorted(set(
        r["dataset"] for r in records if r.get("dataset")
    ))
    methods = ["MLP_ONLY", "FINAL_V3", "MS_HSGC"]

    # -------------------------------------------------------------------
    # Summary statistics
    # -------------------------------------------------------------------
    acc_stats: Dict[str, Dict[str, tuple]] = {}
    delta_v3_stats: Dict[str, tuple] = {}
    for m in methods:
        acc_stats[m] = {}
        for ds in datasets:
            acc_stats[m][ds] = _mean_std(by_md[m][ds])
    for ds in datasets:
        delta_v3_stats[ds] = _mean_std(ds_delta_v3[ds])

    # Hurt counts
    ds_hurt_total: Dict[str, float] = {
        ds: sum(ms_diag["n_hurt"].get(ds, [0.0])) for ds in datasets
    }
    ds_helped_total: Dict[str, float] = {
        ds: sum(ms_diag["n_helped"].get(ds, [0.0])) for ds in datasets
    }

    mean_delta_by_ds: Dict[str, float] = {
        ds: _mean_std(ds_delta_v3.get(ds, []))[0] for ds in datasets
    }

    outcome = _classify_outcome(mean_delta_by_ds, ds_hurt_total)

    # -------------------------------------------------------------------
    # Sensitivity statistics
    # -------------------------------------------------------------------
    sens_by_h1_ds: Dict[float, Dict[str, Dict]] = defaultdict(lambda: defaultdict(lambda: {
        "accs": [], "delta_v3": [], "n_helped": [], "n_hurt": [],
        "blocked_h1_only": [], "frac_1hop": [], "frac_2hop": [],
    }))
    for r in sensitivity_records:
        h1v = _flt(r.get("h1_max_value"))
        ds = r.get("dataset", "")
        if h1v != h1v:
            continue
        d = sens_by_h1_ds[h1v][ds]
        for src, key in [
            ("ms_hsgc_test_acc", "accs"), ("delta_vs_final_v3", "delta_v3"),
            ("n_helped", "n_helped"), ("n_hurt", "n_hurt"),
            ("blocked_by_h1_only", "blocked_h1_only"),
            ("frac_corrected_1hop", "frac_1hop"), ("frac_corrected_2hop", "frac_2hop"),
        ]:
            v = _flt(r.get(src))
            if v == v:
                d[key].append(v)

    h1_values = sorted(sens_by_h1_ds.keys())

    # -------------------------------------------------------------------
    # Build the summary markdown
    # -------------------------------------------------------------------
    lines: List[str] = []
    L = lines.append

    L("# MS_HSGC Targeted 10-Split Experiment — Summary\n")
    L(f"**Status:** EXPERIMENTAL, non-canonical.  ")
    L(f"**Date:** {date.today()}  ")
    L(f"**Tag:** `{tag}`  ")
    L(f"**Datasets:** {', '.join(datasets)}  ")
    L(f"**Splits:** 0–9 (10 splits each)  ")
    L(f"**Configuration:** Full 96-config validation grid (canonical MLP epochs=300)  ")
    L(f"")
    L(f"Output files:")
    for p in [results_path, delta_path, sensitivity_path]:
        if os.path.isfile(p):
            L(f"- `{p}`")
    L("")

    # ----- A. Main 10-split comparison -----
    L("---\n")
    L("## A. Main 10-Split Comparison\n")
    L("### Accuracy (mean ± std over 10 splits)\n")
    hdr = "| Dataset | " + " | ".join(f"{m}" for m in methods) + " | Δ(MS−V3) |"
    sep = "|-" + "-|-".join(["--------"] * (len(methods) + 2)) + "-|"
    L(hdr)
    L(sep)
    for ds in datasets:
        row = f"| {ds:<12s} |"
        for m in methods:
            mu, sd = acc_stats[m][ds]
            row += f" {_fmt(mu)} ± {_fmt(sd, 4)} |"
        mu_d, sd_d = delta_v3_stats[ds]
        row += f" {_fmt(mu_d, 4, signed=True)} ± {_fmt(sd_d, 4)} |"
        L(row)
    L("")

    L("### MS_HSGC Routing Statistics (mean over 10 splits)\n")
    L("| Dataset | confident | 1hop | 2hop | unc_fallback | helped | hurt | net | precision |")
    L("|---------|-----------|------|------|--------------|--------|------|-----|-----------|")
    for ds in datasets:
        def _m(f): return _fmt(_mean_std(ms_diag[f].get(ds, []))[0])
        def _i(f): return str(int(round(_mean_std(ms_diag[f].get(ds, []))[0]))) if ms_diag[f].get(ds) else "N/A"
        conf = 1.0 - _flt(_mean_std(ms_diag["frac_corrected_1hop"].get(ds, []))[0]) \
               - _flt(_mean_std(ms_diag["frac_corrected_2hop"].get(ds, []))[0]) \
               - _flt(_mean_std(ms_diag["frac_mlp_only_uncertain"].get(ds, []))[0])
        L(f"| {ds:<12s} | {_fmt(conf)} | {_m('frac_corrected_1hop')} | "
          f"{_m('frac_corrected_2hop')} | {_m('frac_mlp_only_uncertain')} | "
          f"{_i('n_helped')} | {_i('n_hurt')} | {_i('net_help')} | {_m('correction_precision')} |")
    L("")

    L("### Routing Bottleneck (mean fractions of uncertain test nodes)\n")
    L("| Dataset | blocked_h1_only | blocked_r1 | blocked_r2 | blocked_delta |")
    L("|---------|-----------------|------------|------------|---------------|")
    for ds in datasets:
        def _m(f): return _fmt(_mean_std(ms_diag[f].get(ds, []))[0])
        L(f"| {ds:<12s} | {_m('blocked_by_h1_only')} | {_m('blocked_by_r1')} | "
          f"{_m('blocked_by_r2')} | {_m('blocked_by_delta')} |")
    L("")

    L("### Heterophily Measures (mean)\n")
    L("| Dataset | mean_H1 | mean_H2 | mean_ΔH |")
    L("|---------|---------|---------|---------|")
    for ds in datasets:
        def _m(f): return _fmt(_mean_std(ms_diag[f].get(ds, []))[0])
        L(f"| {ds:<12s} | {_m('mean_H1')} | {_m('mean_H2')} | {_m('mean_DeltaH')} |")
    L("")

    # ----- B. h1_max sensitivity -----
    if h1_values:
        L("---\n")
        L("## B. h1_max Sensitivity Sweep\n")
        for ds in datasets:
            L(f"### {ds}\n")
            L("| h1_max | MS_HSGC | Δ(MS−V3) | helped | hurt | net | "
              "frac_1hop | frac_2hop | blocked_h1_only |")
            L("|--------|---------|----------|--------|------|-----|"
              "----------|-----------|-----------------|")
            for h1v in h1_values:
                d = sens_by_h1_ds[h1v][ds]
                acc_mu = _mean_std(d["accs"])[0]
                dv3_mu = _mean_std(d["delta_v3"])[0]
                nh = _mean_std(d["n_helped"])[0]
                nhu = _mean_std(d["n_hurt"])[0]
                net = nh - nhu
                f1 = _mean_std(d["frac_1hop"])[0]
                f2 = _mean_std(d["frac_2hop"])[0]
                bh = _mean_std(d["blocked_h1_only"])[0]
                L(f"| {h1v:.2f}   | {_fmt(acc_mu)} | {_fmt(dv3_mu, 4, True)} | "
                  f"{int(round(nh)) if nh==nh else 'N/A'} | "
                  f"{int(round(nhu)) if nhu==nhu else 'N/A'} | "
                  f"{int(round(net)) if net==net else 'N/A'} | "
                  f"{_fmt(f1)} | {_fmt(f2)} | {_fmt(bh)} |")
            L("")
    else:
        L("---\n")
        L("## B. h1_max Sensitivity Sweep\n")
        L("_No sensitivity sweep was run. Use `--h1max-sensitivity 0.60 0.70 0.80` to add it._\n")

    # ----- C. Final diagnosis -----
    L("---\n")
    L("## C. Final Diagnosis\n")
    L(f"**Outcome classification:** **{outcome}**\n")

    # Answer the 7 key questions
    L("### Answers to key questions\n")

    # Q1: Chameleon
    if "chameleon" in datasets:
        mu, _ = delta_v3_stats.get("chameleon", (float("nan"), float("nan")))
        n_s = len(ds_delta_v3.get("chameleon", []))
        if mu > 0.001:
            q1 = f"**YES** — MS_HSGC beats FINAL_V3 on Chameleon (mean Δ={_fmt(mu, 4, True)}, n={n_s} splits)."
        elif mu > -0.001:
            q1 = f"**TIE** — MS_HSGC ties FINAL_V3 on Chameleon (mean Δ={_fmt(mu, 4, True)}, n={n_s} splits)."
        else:
            q1 = f"**NO** — MS_HSGC does not beat FINAL_V3 on Chameleon (mean Δ={_fmt(mu, 4, True)}, n={n_s} splits)."
    else:
        q1 = "_Chameleon not in this run._"
    L(f"**1. Does MS_HSGC beat FINAL_V3 on Chameleon over 10 splits?**  \n{q1}\n")

    # Q2: Citeseer
    if "citeseer" in datasets:
        mu, _ = delta_v3_stats.get("citeseer", (float("nan"), float("nan")))
        if mu > -0.005:
            q2 = f"**YES** — close to FINAL_V3 on Citeseer (mean Δ={_fmt(mu, 4, True)}), acceptable degradation."
        else:
            q2 = f"**CONCERN** — Citeseer shows notable degradation vs FINAL_V3 (mean Δ={_fmt(mu, 4, True)})."
    else:
        q2 = "_Citeseer not in this run._"
    L(f"**2. Does MS_HSGC stay close to FINAL_V3 on Citeseer?**  \n{q2}\n")

    # Q3: Texas
    if "texas" in datasets:
        mu, _ = delta_v3_stats.get("texas", (float("nan"), float("nan")))
        hurt = ds_hurt_total.get("texas", 0)
        if hurt == 0:
            q3 = f"**YES — SAFE** — Zero hurt corrections on Texas (mean Δ vs V3={_fmt(mu, 4, True)})."
        elif hurt <= 2:
            q3 = f"**MOSTLY SAFE** — Minimal hurt on Texas ({int(hurt)} total across splits, Δ={_fmt(mu, 4, True)})."
        else:
            q3 = f"**SAFETY CONCERN** — Texas shows {int(hurt)} hurt corrections across splits (Δ={_fmt(mu, 4, True)})."
    else:
        q3 = "_Texas not in this run._"
    L(f"**3. Does MS_HSGC preserve safety on Texas?**  \n{q3}\n")

    # Q4: 2-hop channel dominance
    if datasets:
        ds_1hop = {ds: _mean_std(ms_diag["frac_corrected_1hop"].get(ds, []))[0] for ds in datasets}
        ds_2hop = {ds: _mean_std(ms_diag["frac_corrected_2hop"].get(ds, []))[0] for ds in datasets}
        two_dom = sum(1 for ds in datasets if ds_2hop.get(ds, 0) > ds_1hop.get(ds, 0))
        detail = ", ".join(
            f"{ds}: 1h={_fmt(ds_1hop.get(ds,float('nan')))} 2h={_fmt(ds_2hop.get(ds,float('nan')))}"
            for ds in datasets
        )
        q4 = f"2-hop dominates 1-hop on {two_dom}/{len(datasets)} datasets. ({detail})"
    else:
        q4 = "_No routing data available._"
    L(f"**4. Is 2-hop still the dominant helpful channel?**  \n{q4}\n")

    # Q5: h1_max sensitivity — Chameleon/Citeseer
    if h1_values and datasets:
        improvements = []
        for ds in ["chameleon", "citeseer"]:
            if ds not in datasets:
                continue
            base_d = _mean_std(sens_by_h1_ds[h1_values[0]][ds]["delta_v3"])[0] if h1_values else float("nan")
            best_d = max(
                (_mean_std(sens_by_h1_ds[h1v][ds]["delta_v3"])[0] for h1v in h1_values),
                default=float("nan"),
            )
            if best_d > base_d + 0.002:
                improvements.append(f"{ds} gains {_fmt(best_d - base_d, 4, True)} by loosening h1_max")
        if improvements:
            q5 = "YES — " + "; ".join(improvements) + "."
        else:
            q5 = "NO — Loosening h1_max does not meaningfully improve Chameleon or Citeseer in this sweep."
    else:
        q5 = "_Sensitivity sweep not available._"
    L(f"**5. Does loosening h1_max improve Chameleon/Citeseer meaningfully?**  \n{q5}\n")

    # Q6: When does loosening hurt safety?
    if h1_values and "texas" in datasets:
        safety_breaks = []
        for h1v in h1_values:
            nhu = _mean_std(sens_by_h1_ds[h1v]["texas"]["n_hurt"])[0]
            if nhu == nhu and nhu > 0.5:
                safety_breaks.append(f"h1_max={h1v:.2f} (mean hurt={_fmt(nhu, 1)})")
        if safety_breaks:
            q6 = "Safety deteriorates at: " + ", ".join(safety_breaks) + "."
        else:
            q6 = "Safety (Texas) remains intact across all tested h1_max values."
    else:
        q6 = "_Sensitivity sweep or Texas data not available._"
    L(f"**6. At what point does loosening h1_max hurt safety?**  \n{q6}\n")

    # Q7: Dominant bottleneck
    if datasets:
        all_h1_only = [v for ds in datasets for v in ms_diag["blocked_by_h1_only"].get(ds, [])]
        all_r1 = [v for ds in datasets for v in ms_diag["blocked_by_r1"].get(ds, [])]
        all_r2 = [v for ds in datasets for v in ms_diag["blocked_by_r2"].get(ds, [])]
        all_delta = [v for ds in datasets for v in ms_diag["blocked_by_delta"].get(ds, [])]
        mu_h1 = _mean_std(all_h1_only)[0]
        mu_r1 = _mean_std(all_r1)[0]
        mu_r2 = _mean_std(all_r2)[0]
        mu_d = _mean_std(all_delta)[0]
        bottlenecks = sorted(
            [("h1_only", mu_h1), ("r1", mu_r1), ("r2", mu_r2), ("delta", mu_d)],
            key=lambda x: -x[1] if x[1] == x[1] else 0,
        )
        top = bottlenecks[0]
        if top[1] > 0.15:
            q7 = f"**YES** — `blocked_by_{top[0]}` is still the dominant bottleneck (mean={_fmt(top[1])})."
        else:
            q7 = f"Bottlenecks are distributed: {', '.join(f'{b[0]}={_fmt(b[1])}' for b in bottlenecks)}."
    else:
        q7 = "_No routing data available._"
    L(f"**7. Is blocked_by_h1_only still the dominant bottleneck?**  \n{q7}\n")

    # Interpretive conclusions
    L("---\n")
    L("## D. Interpretive Analysis\n")

    # 2-hop hypothesis
    if datasets:
        mu_dh = _mean_std([v for ds in datasets for v in ms_diag["mean_DeltaH"].get(ds, [])])[0]
        if mu_dh > 0.02:
            L("**Core hypothesis (2-hop helps where 1-hop evidence is weak):** "
              f"SUPPORTED — mean ΔH={_fmt(mu_dh)} > 0 across datasets, suggesting 2-hop "
              f"consistently provides softer evidence in heterophilic regimes.\n")
        else:
            L("**Core hypothesis (2-hop helps where 1-hop evidence is weak):** "
              f"WEAK — mean ΔH={_fmt(mu_dh)} is near zero, suggesting limited 2-hop advantage.\n")

    # Chameleon signal
    if "chameleon" in datasets:
        mu_cham, _ = delta_v3_stats.get("chameleon", (float("nan"), float("nan")))
        n_pos = sum(1 for v in ds_delta_v3.get("chameleon", []) if v > 0)
        n_all = len(ds_delta_v3.get("chameleon", []))
        if mu_cham > 0.002 and n_pos >= 6:
            L(f"**Chameleon:** Genuine improvement — mean Δ={_fmt(mu_cham, 4, True)}, "
              f"positive on {n_pos}/{n_all} splits. Signal is real.\n")
        elif mu_cham > -0.002:
            L(f"**Chameleon:** Mixed/noise — mean Δ={_fmt(mu_cham, 4, True)}, "
              f"positive on {n_pos}/{n_all} splits. Not conclusive yet.\n")
        else:
            L(f"**Chameleon:** No improvement — mean Δ={_fmt(mu_cham, 4, True)}. "
              f"MS_HSGC does not help on Chameleon.\n")

    # Citeseer degradation
    if "citeseer" in datasets:
        mu_cit, _ = delta_v3_stats.get("citeseer", (float("nan"), float("nan")))
        if mu_cit > -0.005:
            L(f"**Citeseer:** Degradation is acceptable (mean Δ={_fmt(mu_cit, 4, True)}). "
              f"The method is not overly heterophily-specialized.\n")
        else:
            L(f"**Citeseer:** Degradation is concerning (mean Δ={_fmt(mu_cit, 4, True)}). "
              f"Method may be over-specialized toward heterophily.\n")

    # Texas safety
    if "texas" in datasets:
        hurt_tx = ds_hurt_total.get("texas", 0)
        if hurt_tx == 0:
            L("**Texas:** Safety fully preserved — zero hurt corrections. "
              "The conservatism mechanism is working correctly.\n")
        else:
            L(f"**Texas:** Safety partially compromised — {int(hurt_tx)} total hurt corrections. "
              f"Review h1_max and rho thresholds.\n")

    # Next step recommendation
    L("### Recommended next step\n")
    if outcome == "balanced / ready for broader benchmark":
        L("→ **Option 1: Broader all-dataset run.** "
          "The method shows balanced behavior and is ready for a full benchmark comparison.\n")
    elif outcome == "promising but still over-conservative":
        L("→ **Option 2: More tuning on h1_max.** "
          "The method is over-conservative. Consider relaxing h1_max further "
          "or reviewing the routing logic to allow more 2-hop corrections.\n")
    elif outcome == "over-correcting / unsafe":
        L("→ **Option 3: Move to compatibility-aware residual diffusion.** "
          "The current routing is causing safety violations. Consider a new method branch.\n")
    else:
        L("→ **Option 2: More tuning on h1_max or routing logic.** "
          "Results are not yet promising enough for a broader benchmark.\n")

    summary_text = "\n".join(lines)

    # -------------------------------------------------------------------
    # Build comparison table
    # -------------------------------------------------------------------
    tlines: List[str] = []
    T = tlines.append
    T("# MS_HSGC vs FINAL_V3 — Targeted 10-Split Comparison Table\n")
    T(f"> **Non-canonical.** Tag: `{tag}`. Datasets: {', '.join(datasets)} — 10 splits each.\n")
    T("## Accuracy (test set, mean ± std)\n")
    T("| Dataset | MLP_ONLY | FINAL_V3 | MS_HSGC | Δ(MS−V3) |")
    T("|---------|----------|----------|---------|----------|")
    for ds in datasets:
        row = f"| {ds:<12s} |"
        for m in methods:
            mu, sd = acc_stats[m][ds]
            row += f" {_fmt(mu)} ± {_fmt(sd)} |"
        mu_d, sd_d = delta_v3_stats[ds]
        row += f" {_fmt(mu_d, 4, True)} ± {_fmt(sd_d)} |"
        T(row)
    T("")
    T("## Routing Statistics (MS_HSGC, mean)\n")
    T("| Dataset | 1hop | 2hop | unc_fallback | helped | hurt | net | prec | "
      "blk_h1_only | blk_r1 | blk_r2 | blk_delta |")
    T("|---------|------|------|--------------|--------|------|-----|------|"
      "------------|--------|--------|-----------|")
    for ds in datasets:
        def _m(f): return _fmt(_mean_std(ms_diag[f].get(ds, []))[0])
        def _i(f):
            vs = ms_diag[f].get(ds, [])
            return str(int(round(_mean_std(vs)[0]))) if vs else "N/A"
        T(f"| {ds:<12s} | {_m('frac_corrected_1hop')} | {_m('frac_corrected_2hop')} | "
          f"{_m('frac_mlp_only_uncertain')} | {_i('n_helped')} | {_i('n_hurt')} | "
          f"{_i('net_help')} | {_m('correction_precision')} | "
          f"{_m('blocked_by_h1_only')} | {_m('blocked_by_r1')} | "
          f"{_m('blocked_by_r2')} | {_m('blocked_by_delta')} |")
    T("")
    T(f"**Outcome:** {outcome}")
    comparison_text = "\n".join(tlines)

    # -------------------------------------------------------------------
    # Build sensitivity table
    # -------------------------------------------------------------------
    slines: List[str] = []
    S = slines.append
    S("# MS_HSGC h1_max Sensitivity — Targeted 10-Split Table\n")
    S(f"> **Non-canonical.** Tag: `{tag}`. Datasets: {', '.join(datasets)} — 10 splits each.\n")
    if h1_values:
        for ds in datasets:
            S(f"## {ds}\n")
            S("| h1_max | MS_HSGC | Δ(MS−V3) | helped | hurt | net | "
              "frac_1hop | frac_2hop | blk_h1_only |")
            S("|--------|---------|----------|--------|------|-----|"
              "----------|-----------|------------|")
            for h1v in h1_values:
                d = sens_by_h1_ds[h1v][ds]
                acc_mu = _mean_std(d["accs"])[0]
                dv3_mu = _mean_std(d["delta_v3"])[0]
                nh_mu = _mean_std(d["n_helped"])[0]
                nhu_mu = _mean_std(d["n_hurt"])[0]
                net = nh_mu - nhu_mu if (nh_mu == nh_mu and nhu_mu == nhu_mu) else float("nan")
                f1 = _mean_std(d["frac_1hop"])[0]
                f2 = _mean_std(d["frac_2hop"])[0]
                bh = _mean_std(d["blocked_h1_only"])[0]
                S(f"| {h1v:.2f}   | {_fmt(acc_mu)} | {_fmt(dv3_mu, 4, True)} | "
                  f"{int(round(nh_mu)) if nh_mu == nh_mu else 'N/A'} | "
                  f"{int(round(nhu_mu)) if nhu_mu == nhu_mu else 'N/A'} | "
                  f"{int(round(net)) if net == net else 'N/A'} | "
                  f"{_fmt(f1)} | {_fmt(f2)} | {_fmt(bh)} |")
            S("")
    else:
        S("_No sensitivity sweep data. Use `--h1max-sensitivity 0.60 0.70 0.80` when running "
          "run_ms_hsgc_evaluation.py to populate this table._\n")
    sensitivity_text = "\n".join(slines)

    # -------------------------------------------------------------------
    # Write files
    # -------------------------------------------------------------------
    for path, text in [
        (summary_out, summary_text),
        (comparison_table_out, comparison_text),
        (sensitivity_table_out, sensitivity_text),
    ]:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        print(f"Written: {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate MS_HSGC summary markdown")
    parser.add_argument("--tag", default="targeted_10split",
                        help="Output tag used when running run_ms_hsgc_evaluation.py")
    args = parser.parse_args()

    tag = args.tag
    results_path = f"reports/ms_hsgc_results_{tag}.csv"
    delta_path = f"reports/ms_hsgc_vs_final_v3_{tag}.csv"
    sensitivity_path = f"reports/ms_hsgc_h1max_sensitivity_{tag}.csv"
    summary_out = f"reports/ms_hsgc_{tag}_summary.md"
    comparison_out = f"tables/ms_hsgc_{tag}_comparison.md"
    sensitivity_table_out = f"tables/ms_hsgc_h1max_sensitivity.md"

    generate_summary(
        results_path=results_path,
        delta_path=delta_path,
        sensitivity_path=sensitivity_path,
        summary_out=summary_out,
        comparison_table_out=comparison_out,
        sensitivity_table_out=sensitivity_table_out,
        tag=tag,
    )


if __name__ == "__main__":
    main()

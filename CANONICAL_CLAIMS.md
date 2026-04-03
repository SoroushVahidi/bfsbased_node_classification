# Canonical Claims — FINAL_V3

This file states precisely what the paper claims, what files support those
claims, and what is explicitly excluded from the main claim.

---

## Canonical method

**FINAL_V3** — reliability-gated selective graph correction on top of a strong
MLP.

- Train a feature-only MLP; derive a per-node reliability score from MLP output
  confidence.
- Apply selective feature or graph correction **only** to nodes whose
  reliability score falls below a threshold `ρ`.
- High-confidence nodes are left unchanged by the graph.
- The method is conservative by design: it never reduces the MLP's benefit on
  reliably-classified nodes.

**Not claimed:** universal heterophily robustness; replacement of the MLP with
graph propagation; superiority on every dataset or split.

---

## Canonical entry points

| Role | Path |
|------|------|
| Stable import | `code/final_method_v3.py` |
| Core implementation | `code/bfsbased_node_classification/final_method_v3.py` |
| 10-split benchmark driver | `code/bfsbased_node_classification/run_final_evaluation.py` |

---

## Canonical result files (frozen — do not overwrite)

| Artifact | Path | Description |
|----------|------|-------------|
| Per-split numerical log | `reports/final_method_v3_results.csv` | 10 splits × 6 datasets, FROZEN |
| Main benchmark table (MD) | `tables/main_results_selective_correction.md` | Regenerated from CSV |
| Main benchmark table (CSV) | `tables/main_results_selective_correction.csv` | Regenerated from CSV |
| Experimental setup table | `tables/experimental_setup_selective_correction.{md,csv}` | |
| Ablation table | `tables/ablation_selective_correction.{md,csv}` | |
| Sensitivity table | `tables/sensitivity_selective_correction.{md,csv}` | |
| Graphical abstract | `figures/graphical_abstract_selective_correction_v3.{png,pdf,svg}` | |
| Correction-rate figure | `figures/correction_rate_vs_homophily.png` | |
| Safety figure | `figures/safety_comparison.png` | |
| Reliability figure | `figures/reliability_vs_accuracy.png` | |
| Method analysis | `reports/final_method_v3_analysis.md` | |
| Safety/harmful-split analysis | `reports/safety_analysis.md` | |
| Regime analysis | `reports/final_v3_regime_analysis.md` | |
| GCN baseline log | `logs/gcn_runs_gcn_table1.jsonl` | |

---

## Canonical reproduction command

Regenerate all tables and figures (no model training, ~5 seconds):

```bash
bash scripts/run_all_selective_correction_results.sh
```

See `scripts/README_REPRODUCE.md` for details, including how to re-run
experiments from scratch without overwriting frozen evidence.

---

## Evaluation protocol

- **Datasets:** Cora, CiteSeer, PubMed (homophilic); Chameleon, Texas, Wisconsin
  (heterophilic)
- **Splits:** 10 GEO-GCN splits per dataset (indices 0–9, pre-bundled in
  `data/splits/`)
- **Baselines in Table 1:** MLP (feature-only), GCN, APPNP, FINAL_V3 (ours)
- **GCN baseline log:** `logs/gcn_runs_gcn_table1.jsonl`

---

## What is excluded from the main claim

| Material | Location | Reason excluded |
|----------|----------|-----------------|
| UG-SGC (uncertainty-gated SGC) | `code/bfsbased_node_classification/manuscript_runner.py`, `archive/legacy_venue_specific/` | Older line; uses confidence thresholding, not reliability gating; not the paper's main evidence |
| UG-SGC-S (structural extension) | `code/bfsbased_node_classification/resubmission_runner.py`, `archive/legacy_venue_specific/` | Exploratory; evidence is promising but mixed; not a replacement for FINAL_V3 |
| Benchmark baseline suite | `archive/exploratory/benchmark/` | Exploratory additional baselines; not part of the main paper comparison |
| Legacy diagnostic scripts | `archive/legacy_code/`, `archive/exploratory_code/` | Superseded by FINAL_V3 pipeline |
| Legacy reports and logs | `archive/legacy_reports/`, `archive/legacy_logs/` | From earlier development phases; not canonical evidence |

---

## Caveats

- Some split-level diagnostic columns (`frac_confident`, `frac_unreliable`,
  `tau`, `rho`, `runtime_sec`) are present only for splits 0–4 in the frozen
  export. The main accuracy results cover all 10 splits. See
  `reports/final_method_v3_analysis.md` for details.
- This is an evidence/code package only; there is no `.tex` manuscript file in
  this repository.

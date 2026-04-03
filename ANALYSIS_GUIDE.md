# Analysis Guide

> **Best single entry point** for an external reader or reviewer.  
> Read this file before anything else.

---

## One-sentence summary

This repository is a frozen evidence package for **FINAL_V3** — a
reliability-gated selective graph correction method that wraps a strong MLP —
submitted as a short paper to Pattern Recognition Letters (PRL).

---

## Canonical method: FINAL_V3

**What it does:**  
Trains a strong MLP; identifies nodes whose reliability score (derived from MLP
output confidence) falls below a threshold `ρ`; applies selective feature or
graph correction only to those low-reliability nodes.

**Implementation paths:**

| Role | Path |
|------|------|
| Stable import | `code/final_method_v3.py` |
| Core implementation | `code/bfsbased_node_classification/final_method_v3.py` |
| 10-split benchmark driver | `code/bfsbased_node_classification/run_final_evaluation.py` |

**Do not confuse with:**
- `UG-SGC` (older uncertainty-gated line; uses confidence thresholding, not
  reliability gating; lives in `manuscript_runner.py`)
- `UG-SGC-S` (structural extension; exploratory; lives in
  `prl_resubmission_runner.py`)

---

## Canonical result files

| Artifact | Path | Note |
|----------|------|------|
| Per-split numerical log | `reports/final_method_v3_results.csv` | **FROZEN** — do not overwrite |
| Main results table (MD) | `tables/main_results_prl.md` | Regenerated from CSV |
| Main results table (CSV) | `tables/main_results_prl.csv` | Regenerated from CSV |
| Experimental setup table | `tables/experimental_setup_prl.{md,csv}` | |
| Ablation table | `tables/ablation_prl.{md,csv}` | |
| Sensitivity table | `tables/sensitivity_prl.{md,csv}` | |
| Safety analysis | `reports/safety_analysis.md` | |
| Method analysis narrative | `reports/final_method_v3_analysis.md` | |
| Graphical abstract | `figures/prl_graphical_abstract_v3.{png,pdf,svg}` | |

Regenerate all tables and figures (no training, ~5 s):

```bash
bash scripts/run_all_prl_results.sh
```

---

## Frozen vs historical vs exploratory artifacts

| Category | Location | Description |
|----------|----------|-------------|
| **Canonical / frozen** | `reports/final_method_v3_results.csv`, `tables/main_results_prl.*`, `figures/prl_graphical_abstract_v3.*` | Do not edit; reproduce only from the frozen CSV |
| **Legacy / supplementary** | `logs/*manuscript_final_validation*`, `tables/prl_final_additions/`, `reports/prl_final_additions/`, `results_prl/` | UG-SGC line; useful provenance but not the paper's main evidence |
| **Exploratory / partial** | `logs/prl_resubmission/`, `tables/prl_resubmission/`, `reports/prl_resubmission/` | UG-SGC-S structural extension; some runs may still be incomplete |
| **Historical / superseded** | `reports/archive/` | Older exports; kept for provenance only |

---

## Exact paths for main tables, figures, logs, and scripts

### Main benchmark (Table 1 equivalent)

- Source data: `reports/final_method_v3_results.csv`
- Rendered table: `tables/main_results_prl.md`
- Build script: `scripts/prl_final_additions/build_prl_v3_figures_and_tables.py`

### Graphical abstract / correction-rate figure

- `figures/prl_graphical_abstract_v3.png`
- `figures/correction_rate_vs_homophily.png`
- Build script: `scripts/prl_final_additions/build_prl_v3_figures_and_tables.py`

### Safety analysis

- `reports/safety_analysis.md` (harmful-split counts, table)
- Build script: `scripts/prl_final_additions/build_prl_v3_figures_and_tables.py`

### GCN baseline (Table 1 comparison row)

- Log: `logs/gcn_runs_gcn_table1.jsonl`
- Summary CSV: `reports/gcn_results_gcn_table1.csv`
- Runner: `code/bfsbased_node_classification/gcn_baseline_runner.py`

### Threshold sensitivity (supplementary)

- `results_prl/threshold_sensitivity_crossruns.jsonl`
- `results_prl/threshold_sensitivity_mini_table_tertiles.csv`
- Build script: `results_prl/build_threshold_sensitivity_artifacts.py`

---

## Common mistakes a reviewer might make

1. **Treating UG-SGC results as the paper's main numbers.**  
   The canonical results are in `reports/final_method_v3_results.csv` and
   `tables/main_results_prl.*`, not in `tables/prl_final_additions/` or
   `logs/manuscript_*`.

2. **Confusing `results_prl/` with FINAL_V3 outputs.**  
   `results_prl/` contains a threshold-sensitivity cross-run package from the
   older UG-SGC line. It is supplementary context, not the main evidence.

3. **Thinking the structural extension (UG-SGC-S) replaces FINAL_V3.**  
   It does not. UG-SGC-S is exploratory and some grouped jobs may be
   incomplete. The frozen package is FINAL_V3.

4. **Running an experiment without `--output-tag` and overwriting canonical files.**  
   Always pass a unique `--output-tag` when re-running. See the safety note in
   [`scripts/README_REPRODUCE.md`](scripts/README_REPRODUCE.md).

5. **Assuming all 10 diagnostic columns are present for all 10 splits.**  
   The main benchmark (test accuracy) covers all 10 splits. Some auxiliary
   diagnostic columns (`frac_confident`, `tau`, `rho`, etc.) are only populated
   for splits 0–4 in the frozen export. This is documented in
   `reports/final_method_v3_analysis.md`.

6. **Looking for a LaTeX manuscript or bibliography in this repository.**  
   There is none. This is an evidence/code package only.

---

## Read this first (ordered reading list)

1. `README.md` — high-level overview and quick start
2. `REPO_STATUS.md` — what is canonical, legacy, and exploratory
3. `reports/final_method_v3_analysis.md` — frozen PRL submission analysis
4. `tables/main_results_prl.md` — main benchmark table
5. `reports/safety_analysis.md` — harmful-split analysis
6. `README_PRL_MANUSCRIPT.md` — full manuscript artifact index

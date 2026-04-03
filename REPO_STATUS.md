# Repository Status

This file describes the current state of the repository: which packages are
canonical, which are legacy, and which are still exploratory.

> **Last updated:** repository reorganization — all legacy and exploratory
> material has been moved to `archive/`. See `archive/README.md` for the full
> inventory of archived material.

**Navigation for tooling / AI agents:** start with [`AGENTS.md`](AGENTS.md) and [`docs/INDEX.md`](docs/INDEX.md).

---

## Canonical package (FINAL_V3)

**Status: ✅ Frozen — do not modify result files**

The canonical package for the research paper submission is **FINAL_V3** (reliability-gated
selective graph correction on top of a strong MLP).

### Frozen evidence files

| File | Description |
|------|-------------|
| `reports/final_method_v3_results.csv` | Per-split log, 10 splits × 6 datasets |
| `tables/main_results_selective_correction.{md,csv}` | Main benchmark table |
| `tables/experimental_setup_selective_correction.{md,csv}` | Experimental setup table |
| `tables/ablation_selective_correction.{md,csv}` | Ablation table |
| `tables/sensitivity_selective_correction.{md,csv}` | Sensitivity table |
| `figures/graphical_abstract_selective_correction_v3.*` | Graphical abstract |
| `figures/correction_rate_vs_homophily.png` | Correction-rate vs homophily |
| `figures/safety_comparison.png` | Safety analysis figure |
| `figures/reliability_vs_accuracy.png` | Reliability vs accuracy figure |
| `reports/safety_analysis.md` | Harmful-split analysis |
| `reports/final_method_v3_analysis.md` | Method analysis narrative |
| `reports/final_v3_regime_analysis.md` | Regime analysis |

### Canonical scripts

| Script | Role |
|--------|------|
| `code/final_method_v3.py` | Stable import path |
| `code/bfsbased_node_classification/final_method_v3.py` | Core implementation |
| `code/bfsbased_node_classification/run_final_evaluation.py` | 10-split benchmark driver |
| `code/bfsbased_node_classification/gcn_baseline_runner.py` | GCN Table 1 baseline |
| `code/bfsbased_node_classification/analyze_final_v3_regimes.py` | Regime analysis |
| `code/bfsbased_node_classification/build_final_v3_regime_analysis.py` | Regime analysis builder |
| `bash scripts/run_all_selective_correction_results.sh` | One-command table/figure refresh |

### Canonical log files

| File | Description |
|------|-------------|
| `logs/final_v3_regime_analysis_per_dataset.csv` | FINAL_V3 per-dataset regime summary |
| `logs/final_v3_regime_analysis_per_split.csv` | FINAL_V3 per-split regime detail |
| `logs/gcn_runs_gcn_table1.jsonl` | GCN baseline per-split runs |
| `logs/manuscript_regime_analysis_final_validation_main.csv` | Homophily values used by build_auxiliary_tables.py |
| `logs/regime_analysis_manuscript_final_validation_job2.csv` | PubMed homophily used by build_auxiliary_tables.py |

---

## Legacy package (UG-SGC)

**Status: 📁 Preserved for provenance — supplementary use only**

The original uncertainty-gated selective correction line. Useful as background
and supplementary context but **not** the paper's main evidence.

### Key locations

- Runner: `code/bfsbased_node_classification/manuscript_runner.py`
- Core module (all variants): `code/bfsbased_node_classification/bfsbased-full-investigate-homophil.py`
- Tables: `archive/legacy_venue_specific/tables_prl_final_additions/`
- Figures: `archive/legacy_venue_specific/figures_prl_final_additions/`
- Reports: `archive/legacy_venue_specific/reports_prl_final_additions/`
- Per-run logs: `archive/legacy_logs/ugsgc_per_split_runs/`
- Threshold sensitivity: `archive/legacy_venue_specific/results_prl/`

---

## Exploratory package (UG-SGC-S)

**Status: 🔬 Partial / exploratory — some grouped jobs may be incomplete**

The structural extension adds a far-structural support term to the selective
correction. Evidence is promising but mixed; not a replacement for FINAL_V3.

### Key locations

- Runner: `code/bfsbased_node_classification/resubmission_runner.py`
- Tables: `archive/legacy_venue_specific/tables_prl_resubmission/`
- Logs: `archive/legacy_venue_specific/logs_prl_resubmission/`
- Reports: `archive/legacy_venue_specific/reports_resubmission_protocols/`

---

## GCN / APPNP baselines

**Status: ✅ Complete (6 datasets × 10 splits)**

Standalone GCN baseline for Table 1 comparison.

- Runner: `code/bfsbased_node_classification/gcn_baseline_runner.py`
- Log: `logs/gcn_runs_gcn_table1.jsonl`

Additional standard and paper baselines (APPNP, H2GCN, SGC, GraphSAGE, Correct & Smooth, CLP, Begga-style models) are implemented in `code/bfsbased_node_classification/standard_node_baselines.py` and selectable from `resubmission_runner.py`. See [`docs/BASELINES_SUITE.md`](docs/BASELINES_SUITE.md) for names, references, and fidelity notes.

---

## Regenerated cluster outputs (`outputs/`)

**Status: 🔁 Not versioned — safe to delete locally**

Large JSONL sweep results land under `outputs/baseline_comparison/`, `outputs/label_budget/`, and `outputs/graph_noise_robustness/` (see root `.gitignore`). Regenerate with the drivers and Slurm launch scripts in `scripts/baseline_comparison_wulver/`. They do **not** replace frozen canonical CSVs in `reports/` and `tables/`.

---

## Archived / superseded

**Status: 🗄️ Kept for provenance — do not use for paper claims**

| Location | Description |
|----------|-------------|
| `archive/legacy_code/` | Superseded legacy code (BFS runner, comparison runner, old GNN scripts) |
| `archive/exploratory_code/` | Experimental scripts (local agreement, margin safety, resubmission analysis) |
| `archive/legacy_reports/` | Diagnostic and exploratory reports from development phases |
| `archive/legacy_logs/` | UG-SGC per-run logs and old analysis files |
| `archive/legacy_material/` | Run metadata, HPC output logs, broken job backups |
| `archive/exploratory/` | Benchmark baseline suite and external baselines (not part of main claim) |
| `archive/legacy_venue_specific/` | Full UG-SGC and UG-SGC-S venue-specific materials |
| `archive/legacy_docs/` | Superseded documentation (README_MANUSCRIPT_ARTIFACTS.md) |
| `reports/archive/` | Superseded exports and older result files |
| `code/bfsbased_node_classification/experimental_archived/` | Legacy diagnostic scripts |

See `archive/README.md` for the complete inventory and justification.

---

## Summary

| Package | Label | Evidence state |
|---------|-------|---------------|
| Reliability-gated SGC | **FINAL_V3** | ✅ Frozen canonical |
| Uncertainty-gated SGC | **UG-SGC** | 📁 Legacy / supplementary |
| Structural extension | **UG-SGC-S** | 🔬 Exploratory / partial |
| GCN/APPNP baselines | **baselines** | ✅ Complete |
| Benchmark baseline suite | — | 🔬 Exploratory / archive |
| Superseded exports | — | 🗄️ Archived |

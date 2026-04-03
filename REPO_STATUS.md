# Repository Status

This repository uses a strict three-label status system.

## Canonical

**Navigation for tooling / AI agents:** start with [`AGENTS.md`](AGENTS.md) and [`docs/INDEX.md`](docs/INDEX.md).

---
- Method: `FINAL_V3`
- Entry point: `code/final_method_v3.py`
- Canonical results source: `reports/final_method_v3_results.csv`
- Canonical result products: `tables/`, `figures/`, and canonical analyses in `reports/`.

## Archived

- Venue-specific historical bundles and manuscript-era assets in `archive/legacy_venue_specific/`.
- Legacy UG-SGC logs and supporting files in `archive/legacy_logs/`.
- Legacy code and docs in `archive/legacy_code/` and `archive/legacy_docs/`.

Archived material is preserved for provenance and should not be used as primary evidence for canonical claims.

## Exploratory

- Structural-extension and non-canonical exploratory lines.
- Locations: `archive/exploratory/`, `archive/exploratory_code/`, and exploratory runners retained in `code/bfsbased_node_classification/`.

Exploratory material is informative but not claim-bearing for the canonical method.

## Reader path

1. `README.md`
2. `docs/OVERVIEW.md`
3. `docs/METHOD_STATUS.md`
4. `docs/REPRODUCIBILITY.md`
5. `docs/RESULTS_GUIDE.md`

For a link hub (baselines, Slurm, agents): [`docs/INDEX.md`](docs/INDEX.md).

---

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

## Archived / superseded (inventory)

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

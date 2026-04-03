# Archive — Inventory and Justification

For the active codebase map, see [`../AGENTS.md`](../AGENTS.md) and [`../docs/INDEX.md`](../docs/INDEX.md).

This directory contains all non-canonical material that has been moved out of
the main repository surface to keep the top-level layout clean and
reviewer-focused.

**Nothing in this directory is part of the paper's main evidence.** The
canonical evidence lives in `reports/`, `tables/`, `figures/`, and `logs/` at
the repository root.

---

## Directory map

| Directory | Contents | Reason archived |
|-----------|----------|-----------------|
| `legacy_venue_specific/` | Full UG-SGC and UG-SGC-S materials from earlier submission phases | Venue-specific; superseded by FINAL_V3; kept for provenance |
| `legacy_code/` | Superseded Python scripts | Replaced by FINAL_V3 pipeline; no longer part of canonical flow |
| `exploratory_code/` | Experimental scripts not part of the main claim | Exploratory; some incomplete; not paper evidence |
| `legacy_reports/` | Diagnostic and exploratory reports from development | Superseded by canonical reports; internal development records |
| `legacy_logs/` | UG-SGC per-run logs and old analysis CSV/MD files | UG-SGC evidence, not FINAL_V3; archived for provenance |
| `exploratory/` | Benchmark baseline suite and external baselines stub | Exploratory; not part of main paper comparison |
| `legacy_material/` | Run metadata, HPC job output logs, broken job backups | Operational/provenance; not evidence |
| `legacy_docs/` | Superseded documentation files | Content merged into README.md / CANONICAL_CLAIMS.md |

---

## `legacy_venue_specific/`

Material created during an earlier submission to a specific venue (PRL).
Includes UG-SGC per-run logs, threshold sensitivity cross-run package,
structural extension (UG-SGC-S) exploratory results, venue-specific figures and
tables, HPC scripts, and run protocols.

See `legacy_venue_specific/README.md` for the full sub-inventory.

---

## `legacy_code/`

Python scripts that were part of earlier development phases and are no longer
part of the canonical FINAL_V3 flow:

| File | Reason archived |
|------|-----------------|
| `bfsbased-gnn-predictclass.py` | Very old GNN prediction script; pre-dates manuscript phase |
| `bfs_runner.py` | Original BFS-based runner; superseded by `run_final_evaluation.py` |
| `comparison_runner.py` | Old per-dataset comparison runner; superseded |
| `compare_with_begga.py` | One-off comparison with BEGGA baseline; not in paper |
| `improved_sgc_variants.py` | Exploratory SGC variant experiments; none canonical |
| `analyze_manuscript_results.py` | UG-SGC analysis helper; not relevant to FINAL_V3 |
| `run_sgc_baseline.py` | Standalone SGC baseline runner; superseded |

---

## `exploratory_code/`

Experimental scripts that explore ideas adjacent to the main paper but are not
part of the canonical claim:

| File | Reason archived |
|------|-----------------|
| `analyze_resubmission_results.py` | UG-SGC-S structural extension analysis |
| `run_final_v3_local_agreement_experiment.py` | Local agreement gating experiment; promising but not canonical |
| `run_margin_bucket_safety_experiment.py` | Margin-bucket safety diagnostic; internal development |
| `appnp_baseline_runner.py` | APPNP baseline runner used by `resubmission_runner.py` |
| `merge_resubmission_runs.py` | Grouped-job merge helper for UG-SGC-S experiments |

---

## `legacy_reports/`

Reports generated during development phases that are no longer canonical:

| File | Reason archived |
|------|-----------------|
| `diagnostic_why_not_beating_baselines.*` | Internal diagnostic; superseded |
| `method_improvement_exploration*` | Exploratory improvement experiments |
| `reliability_gated_report.*` | v2 reliability-gated variant; superseded by FINAL_V3 |
| `reliability_gated_v2_next_phase.*` | v2 next-phase experiments; superseded |
| `manuscript_repo_consistency_audit.md` | Internal audit document |
| `naming_migration_map.md` | Internal tracking of file renames |
| `final_repo_polish_audit.md` | Internal polish checklist |
| `repository_status.md` | Old status file; superseded by `REPO_STATUS.md` at root |
| `final_method_story.txt` | Internal narrative for paper drafting |
| `manuscript_section_support.md` | Internal manuscript-section-to-evidence mapping |
| `gcn_results_smoke_test.csv` | Smoke-test output only; not canonical evidence |

---

## `legacy_logs/`

Log files from earlier development phases:

| Pattern | Description |
|---------|-------------|
| `ugsgc_per_split_runs/` | ~270 per-split per-repeat JSON files from UG-SGC runs (manuscript_runner.py output) |
| `comparison_runs_*` | UG-SGC cross-run comparison JSONL files |
| `comparison_summary_*` | UG-SGC comparison summary JSON files |
| `comparison_fairness_report_*` | UG-SGC fairness report MD files |
| `manuscript_case_studies_*` | UG-SGC case study notes |
| `manuscript_correction_behavior_*` | UG-SGC correction behavior CSVs |
| `manuscript_gain_over_mlp_*` | UG-SGC gain-over-MLP CSVs |
| `correction_behavior_manuscript_final_validation_job2.csv` | Job2 UG-SGC correction data |
| `gain_over_mlp_manuscript_final_validation_job2.csv` | Job2 UG-SGC gain data |
| `manuscript_regime_analysis_final_validation.csv` | UG-SGC regime analysis (job1 only) |
| `case_studies_manuscript_final_validation_job2.md` | UG-SGC case study notes (job2) |
| `comparison_runs_manuscript_final_validation_job1.jsonl` | Job1 comparison runs |
| `texas_full-investigate-v2_*.txt` | Very old debug log files |
| `gcn_runs_smoke_test.jsonl` | GCN smoke test; not canonical evidence |

**Note:** The two regime analysis CSV files needed by `build_auxiliary_tables.py`
(`logs/manuscript_regime_analysis_final_validation_main.csv` and
`logs/regime_analysis_manuscript_final_validation_job2.csv`) are **kept in
`logs/`** at the root and were not archived, as they are required for
generating the canonical experimental setup table.

---

## `exploratory/`

### `benchmark/`

A benchmark baseline suite implementing GPRGNN, GCNII, H2GCN, LINKX, FSGNN,
ACM-GNN (inline) plus stubs for OrderedGNN and DJ-GNN. Archived because:
- These baselines are not part of the main paper comparison in FINAL_V3.
- Some runners are stubs (DJ-GNN, OrderedGNN).
- The benchmark suite was developed in parallel with the main paper and was not
  used to generate any canonical result file.

### `external_baselines/`

A stub directory for vendored third-party baseline repositories (planned but
not cloned). Archived because no external code was ever added.

---

## `legacy_material/`

### `run_metadata/`

Snapshots and hash files from specific cluster runs (job1/job3). Contains code
snapshots and run commands used to reproduce specific HPC jobs. Archived for
provenance; not needed for canonical reproduction from frozen CSVs.

### `broken_job1_backup/`

7 backup files from job1 (ID 883410) that was interrupted. These are copies of
`manuscript_final_validation_main` files. Archived because the canonical
evidence comes from the completed job2/merged runs.

### `logs_slurm_output/`

Slurm `.err` and `.out` output files from HPC job runs. Operational logs only;
no numerical evidence.

---

## `legacy_docs/`

### `README_MANUSCRIPT_ARTIFACTS.md`

A manuscript-oriented artifact index that was superseded by:
- `README.md` (repository overview)
- `CANONICAL_CLAIMS.md` (explicit claims and evidence mapping)
- `REPO_STATUS.md` (canonical/legacy/exploratory classification)

Kept in archive for historical reference only.

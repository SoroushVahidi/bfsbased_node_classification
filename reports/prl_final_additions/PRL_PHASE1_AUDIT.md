# Phase 1 — Audit (PRL manuscript package)

**Repository root:** same directory as this file’s parent’s parent (`…/bfsbased_node_classification/`).  
**Generated:** part of the PRL finalization pass.

## 1. Manuscript source files

| Item | Status |
|------|--------|
| LaTeX / Word | **Not in repo** — no `.tex` found; author adds manuscript separately. |
| Narrative support | `logs/manuscript_case_studies_final_validation.md`, `logs/comparison_fairness_report_manuscript_final_validation.md` |

## 2. Figures referenced by manuscript

Publication-oriented outputs:

| Figure | Path |
|--------|------|
| Δ (SGC − MLP) bar | `figures/prl_final_additions/fig_delta_vs_mlp_bar.pdf` |
| Gate stats (4 datasets) | `figures/prl_final_additions/fig_gate_stats_priority_four.pdf` |
| Threshold **cross-run** (not a τ sweep) | `results_prl/fig_threshold_*.pdf` |

## 3. Tables / data

| Table | Path |
|-------|------|
| Merged benchmark | `tables/prl_final_additions/prl_benchmark_summary.csv` |
| Win/tie/loss | `tables/prl_final_additions/prl_split_win_tie_loss.csv` |
| Correction behavior | `tables/prl_final_additions/prl_correction_behavior_compact.csv` |
| Gate correlations (cross-run) | `tables/prl_final_additions/prl_subsection3_crossrun_correlations.csv` |

**Canonical sources:** `logs/manuscript_*_final_validation.csv`.

## 4. Final manuscript-ready logs

- `logs/manuscript_gain_over_mlp_final_validation.csv`
- `logs/manuscript_correction_behavior_final_validation.csv`
- `logs/manuscript_regime_analysis_final_validation.csv`
- `logs/comparison_runs_manuscript_final_validation.jsonl`
- `logs/comparison_summary_manuscript_final_validation.json`

## 5. Older / exploratory outputs

- `code/bfsbased_node_classification/diagnosis/preserve_*` — point-in-time snapshots; **prefer `logs/`** for citations.
- `logs/broken_job1_*` — superseded failed job.
- `logs/*_final_validation_main.csv` — earlier pipeline stage where superseded by `*_final_validation.csv`.

## 6. Scripts

| Role | Path |
|------|------|
| **Orchestrator** | `scripts/prl_final_additions/build_all_prl_artifacts.py` |
| Tables from logs | `scripts/prl_final_additions/build_prl_tables.py` |
| Gate correlations | `scripts/prl_final_additions/compute_subsection3_crossrun_correlations.py` |
| Figures | `scripts/prl_final_additions/generate_prl_benchmark_figure.py`, `generate_correction_behavior_priority_figure.py` |
| Threshold package builder | `results_prl/build_threshold_sensitivity_artifacts.py` |
| Slurm | `slurm/run_selective_validation_*.sbatch` |

## 7. Artifact coverage checklist

| Artifact | Status |
|----------|--------|
| Main benchmark table | **Yes** — gain CSV + merged summary |
| Score weighting / correction behavior | **Yes** — correction CSV + compact copy |
| Threshold / gate behavior | **Yes** — `results_prl/` + correlation CSV |
| Win/tie/loss | **Yes** — in gain CSV + `prl_split_win_tie_loss.csv` |
| Qualitative examples | **Yes** — case studies MD (split 0) |
| Compact PRL figures | **Yes** — under `figures/prl_final_additions/` |

## Machine-readable inventory

- `reports/prl_final_additions/artifact_inventory.json`
- `reports/prl_final_additions/artifact_inventory.csv`

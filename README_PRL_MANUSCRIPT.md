# PRL manuscript — repository evidence package

Start with the [root README](README.md) for the high-level repository map.

**Paper working title:** *Uncertainty-Gated Selective Graph Correction for Node Classification*  
**Target:** Pattern Recognition Letters (7 pages incl. references)

This repository currently contains multiple manuscript-related packages. This
file explains which artifacts are canonical, which are supplementary, and which
belong to the newer resubmission / structural-upgrade line.

## Start here

If you are reading the repository for the first time:

1. Read `README.md` for the package overview.
2. Read `reports/final_method_v3_analysis.md` for the frozen PRL submission line.
3. Read `reports/prl_resubmission/PRL_METHOD_SPEC.md` for the resubmission / `UG-SGC` method line.
4. Read `reports/prl_resubmission/RUN_STATUS.md` for current grouped-job status.

## Canonical artifacts (FINAL_V3)

Evaluation uses **10 GEO-GCN splits** per dataset (indices 0–9 in `data/splits/`).

| Purpose | Location |
|--------|----------|
| **Per-split numerical log** | `reports/final_method_v3_results.csv` |
| **Frozen 10-split table export (rebuild source)** | `reports/archive/superseded_final_tables_prl/final_tables_prl.csv` |
| **Rebuild CSV from frozen export (no training)** | `python3 scripts/prl_final_additions/rebuild_final_v3_results_10split.py` |
| **Main benchmark table** | `tables/main_results_prl.md`, `tables/main_results_prl.csv` |
| **Experimental setup table** | `tables/experimental_setup_prl.md`, `tables/experimental_setup_prl.csv` |
| **Ablation snapshot (supplementary)** | `tables/ablation_prl.md`, `tables/ablation_prl.csv` |
| **ρ sensitivity snapshot (supplementary)** | `tables/sensitivity_prl.md`, `tables/sensitivity_prl.csv` |
| **Analysis writeup** | `reports/final_method_v3_analysis.md` |
| **Narrative for drafting** | `reports/final_method_story.txt` |
| **Safety / harmful-split summary** | `reports/safety_analysis.md` |
| **Figures** | `figures/prl_graphical_abstract_v3.png`, `correction_rate_vs_homophily.png`, `safety_comparison.png`, `reliability_vs_accuracy.png` |
| **Regenerate tables + figures (no training)** | `bash scripts/run_all_prl_results.sh` — details in [scripts/README_REPRODUCE.md](scripts/README_REPRODUCE.md) |

## Avoiding package confusion

The repository contains **three** distinct evidence lines:

1. **FINAL_V3 (frozen submission package)**  
   `reports/final_method_v3_results.csv` -> `tables/main_results_prl.*`  
   This is the cleanest submission-facing package currently present on `main`.

2. **Original UG-SGC / manuscript-runner line**  
   `logs/manuscript_gain_over_mlp_final_validation.csv`,
   `tables/prl_final_additions/prl_benchmark_summary.csv`, and related
   `prl_final_additions/` artifacts.  
   This is still useful for manuscript analysis and gate-behavior evidence, but
   should not be mixed blindly with the frozen `FINAL_V3` numbers.

3. **Structural extension line (`UG-SGC-S`)**  
   `tables/prl_resubmission/`, `logs/prl_resubmission/`,
   `reports/prl_resubmission/STRUCTURAL_UPGRADE_SMOKE_NOTE.md`  
   This is the newer stronger-method package. Some grouped-job outputs may still
   be partial or in progress.

## Current repo state

The repository currently contains:

- a frozen PRL submission package (`FINAL_V3`)
- a manuscript-analysis package under `prl_final_additions/`
- a newer PRL resubmission package with:
  - stronger baselines
  - true ablations
  - the structural selective-correction extension (`UG-SGC-S`)
  - smoke-test artifacts
  - grouped Wulver job outputs / partial logs

Because some grouped Wulver jobs may still be running when this repository is
inspected, `reports/prl_resubmission/RUN_STATUS.md` should be treated as the
source of truth for what is complete versus in-progress.

## Supplementary PRL materials (resubmission / analysis bundle)

| Purpose | Location |
|--------|----------|
| Regenerate extended tables/figures + inventory | `python3 scripts/prl_final_additions/build_all_prl_artifacts.py` |
| Phase 1 audit + inventory | `reports/prl_final_additions/PRL_PHASE1_AUDIT.md`, `artifact_inventory.json` |
| Phase 2 gap analysis | `reports/prl_final_additions/PRL_PHASE2_GAP_ANALYSIS.md` |
| Results §3 (gate / threshold) notes | `reports/prl_final_additions/PRL_RESULTS_SUBSECTION_3.md` |
| Safe claims & caveats | `reports/prl_final_additions/CLAIMS_AND_CAVEATS.md` |
| PRL resubmission method spec / readiness | `reports/prl_resubmission/PRL_METHOD_SPEC.md`, `REPO_READINESS_AUDIT.md` |
| Structural selective-correction smoke note | `reports/prl_resubmission/STRUCTURAL_UPGRADE_SMOKE_NOTE.md` |
| Focused resubmission runner | `python3 code/bfsbased_node_classification/prl_resubmission_runner.py ...` |
| Merged benchmark CSV | `tables/prl_final_additions/prl_benchmark_summary.csv` |
| Win/tie/loss | `tables/prl_final_additions/prl_split_win_tie_loss.csv` |
| Correction behavior | `tables/prl_final_additions/prl_correction_behavior_compact.csv` |
| Gate cross-run correlations | `tables/prl_final_additions/prl_subsection3_crossrun_correlations.csv` |
| Graphical abstract (legacy PRL bundle) | `figures/prl_final_additions/graphical_abstract_prl.{pdf,png,svg}` |
| Threshold cross-run package | `results_prl/`, `results_prl/README_threshold_sensitivity.md` |
| Focused resubmission outputs | `logs/prl_resubmission/`, `tables/prl_resubmission/`, `reports/prl_resubmission/` |

## LaTeX manuscript

There is **no** `.tex` file in this repository; keep the paper in a separate writing repo if needed.

## Drivers

- Core manuscript driver: `code/bfsbased_node_classification/manuscript_runner.py`
- **FINAL_V3** benchmark driver: `code/bfsbased_node_classification/run_final_evaluation.py`
- Focused resubmission driver: `code/bfsbased_node_classification/prl_resubmission_runner.py`
- Analysis helper: `code/bfsbased_node_classification/analyze_prl_resubmission.py`
- Structural grouped merge helper: `code/bfsbased_node_classification/merge_prl_resubmission_runs.py`
- Slurm templates: `slurm/*.sbatch`

## Framing (careful claims)

- Feature-first selective graph correction, not a global graph learner.
- Do not describe the repository as universally heterophily-robust.
- Threshold evidence is mostly cross-run validation-choice evidence, not a full
  fixed-split per-threshold sweep.

## Archived / exploratory

- `code/bfsbased_node_classification/experimental_archived/`
- `reports/archive/`
- older exploratory reports under `reports/*.md`

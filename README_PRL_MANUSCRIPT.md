# PRL manuscript — repository evidence package

**Paper:** *Uncertainty-Gated Selective Graph Correction for Node Classification*  
**Target:** Pattern Recognition Letters (7 pages incl. references)

This README lists **manuscript-ready** artifacts. Prefer these over older exploratory copies under `code/.../diagnosis/preserve_*`.

## Quick navigation

| Purpose | Location |
|--------|----------|
| **Regenerate tables/figures + inventory** | `python3 scripts/prl_final_additions/build_all_prl_artifacts.py` |
| **Phase 1 audit + inventory** | `reports/prl_final_additions/PRL_PHASE1_AUDIT.md`, `artifact_inventory.json` |
| **Phase 2 gap analysis** | `reports/prl_final_additions/PRL_PHASE2_GAP_ANALYSIS.md` |
| **Results §3 (gate / threshold) notes** | `reports/prl_final_additions/PRL_RESULTS_SUBSECTION_3.md` |
| **Safe claims & caveats** | `reports/prl_final_additions/CLAIMS_AND_CAVEATS.md` |
| **Merged benchmark CSV** | `tables/prl_final_additions/prl_benchmark_summary.csv` (copy of merged table) |
| **Win/tie/loss** | `tables/prl_final_additions/prl_split_win_tie_loss.csv` |
| **Correction behavior** | `tables/prl_final_additions/prl_correction_behavior_compact.csv` |
| **Gate cross-run correlations** | `tables/prl_final_additions/prl_subsection3_crossrun_correlations.csv` |
| **Figures (Δ vs MLP, gate stats)** | `figures/prl_final_additions/` |
| **Graphical abstract (submission, separate from main PDF)** | `figures/prl_final_additions/graphical_abstract_prl.{pdf,png,svg}` — regenerate: `python3 scripts/prl_final_additions/make_graphical_abstract.py` |
| **Threshold cross-run package** | `results_prl/` + `results_prl/README_threshold_sensitivity.md` |
| **Canonical logs** | `logs/*_manuscript_final_validation*` |

## LaTeX manuscript

There is **no** `.tex` file in this repository; add your paper separately or submodule your writing repo.

## Reproducibility

- Per-run JSONL: `logs/comparison_runs_manuscript_final_validation.jsonl`
- Training/eval driver (under `code/`): `code/bfsbased_node_classification/manuscript_runner.py`
- Slurm templates: `slurm/*.sbatch`

## Framing (do not overclaim)

- **Not** a universal heterophily-robust method.
- **Main story:** strong MLP → **selective** graph correction for **uncertain** nodes.
- **No** formal component-removal ablations unless you add experiments that truly implement them.
- **Threshold:** evidence is primarily **cross-run** (validation-selected τ vs test uncertain/changed fractions), **not** full per-τ sweeps on a fixed split — see `results_prl/README_threshold_sensitivity.md`.

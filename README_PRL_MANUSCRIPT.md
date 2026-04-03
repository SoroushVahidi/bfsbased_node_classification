# PRL manuscript — repository evidence package

> **For the fastest orientation, read [`ANALYSIS_GUIDE.md`](ANALYSIS_GUIDE.md) first.**  
> See [`REPO_STATUS.md`](REPO_STATUS.md) for a concise canonical/legacy/exploratory inventory.

Start with the [root README](README.md) for the high-level repository map.

**Paper working title:** *Uncertainty-Gated Selective Graph Correction for Node Classification*  
**Target:** Pattern Recognition Letters (short paper, ~7 pages incl. references)  
**Canonical package:** `FINAL_V3` (reliability-gated selective graph correction)

This file is a **manuscript-oriented artifact index** — a reference for where
each piece of manuscript evidence lives. For a reviewer-facing overview use
`ANALYSIS_GUIDE.md`. For the canonical/legacy/exploratory breakdown use
`REPO_STATUS.md`.

---

## Canonical artifacts (FINAL_V3)

Evaluation uses **10 GEO-GCN splits** per dataset (indices 0–9 in `data/splits/`).

> ⚠️ The files below are **frozen**. Do not overwrite them when re-running
> experiments; use `--output-tag` instead.

| Purpose | Location |
|--------|----------|
| **Per-split numerical log** | `reports/final_method_v3_results.csv` |
| **Frozen 10-split export (rebuild source)** | `reports/archive/superseded_final_tables_prl/final_tables_prl.csv` |
| **Rebuild CSV from frozen export (no training)** | `python3 scripts/prl_final_additions/rebuild_final_v3_results_10split.py` |
| **Main benchmark table** | `tables/main_results_prl.md`, `tables/main_results_prl.csv` |
| **Experimental setup table** | `tables/experimental_setup_prl.{md,csv}` |
| **Ablation snapshot (supplementary)** | `tables/ablation_prl.{md,csv}` |
| **ρ sensitivity snapshot (supplementary)** | `tables/sensitivity_prl.{md,csv}` |
| **Analysis writeup** | `reports/final_method_v3_analysis.md` |
| **Narrative for drafting** | `reports/final_method_story.txt` |
| **Safety / harmful-split summary** | `reports/safety_analysis.md` |
| **Figures** | `figures/prl_graphical_abstract_v3.{png,pdf,svg}`, `figures/correction_rate_vs_homophily.png`, `figures/safety_comparison.png`, `figures/reliability_vs_accuracy.png` |
| **GCN baseline log** | `logs/gcn_runs_gcn_table1.jsonl`, `reports/gcn_results_gcn_table1.csv` |
| **Regenerate tables + figures (no training)** | `bash scripts/run_all_prl_results.sh` — details in [scripts/README_REPRODUCE.md](scripts/README_REPRODUCE.md) |

---

## The three evidence lines

| Line | Label | Status | Main locations |
|------|-------|--------|----------------|
| Reliability-gated SGC | **FINAL_V3** | ✅ Frozen canonical | `reports/final_method_v3_results.csv`, `tables/main_results_prl.*` |
| Uncertainty-gated SGC | **UG-SGC** | 📁 Legacy / supplementary | `tables/prl_final_additions/`, `logs/*manuscript_final_validation*` |
| Structural extension | **UG-SGC-S** | 🔬 Exploratory / partial | `tables/prl_resubmission/`, `logs/prl_resubmission/` |

Do not substitute UG-SGC numbers for FINAL_V3 numbers in the main manuscript
table. See [`REPO_STATUS.md`](REPO_STATUS.md) for the full boundary description.

---

## Supplementary PRL materials (UG-SGC legacy + analysis bundle)

| Purpose | Location |
|--------|----------|
| Extended tables/figures + inventory | `python3 scripts/prl_final_additions/build_all_prl_artifacts.py` |
| Phase 1 audit + inventory | `reports/prl_final_additions/PRL_PHASE1_AUDIT.md`, `artifact_inventory.json` |
| Phase 2 gap analysis | `reports/prl_final_additions/PRL_PHASE2_GAP_ANALYSIS.md` |
| Results §3 (gate / threshold) notes | `reports/prl_final_additions/PRL_RESULTS_SUBSECTION_3.md` |
| Safe claims & caveats | `reports/prl_final_additions/CLAIMS_AND_CAVEATS.md` |
| PRL resubmission method spec | `reports/prl_resubmission/PRL_METHOD_SPEC.md` |
| Structural smoke note | `reports/prl_resubmission/STRUCTURAL_UPGRADE_SMOKE_NOTE.md` |
| UG-SGC merged benchmark CSV | `tables/prl_final_additions/prl_benchmark_summary.csv` |
| Win/tie/loss | `tables/prl_final_additions/prl_split_win_tie_loss.csv` |
| Correction behavior | `tables/prl_final_additions/prl_correction_behavior_compact.csv` |
| Gate cross-run correlations | `tables/prl_final_additions/prl_subsection3_crossrun_correlations.csv` |
| Threshold cross-run package | `results_prl/`, `results_prl/README_threshold_sensitivity.md` |
| UG-SGC-S outputs | `logs/prl_resubmission/`, `tables/prl_resubmission/`, `reports/prl_resubmission/` |

---

## Script inventory

| Script | Role |
|--------|------|
| `code/bfsbased_node_classification/run_final_evaluation.py` | FINAL_V3 benchmark driver |
| `code/bfsbased_node_classification/manuscript_runner.py` | UG-SGC runner (legacy) |
| `code/bfsbased_node_classification/prl_resubmission_runner.py` | UG-SGC-S + GCN/APPNP baselines |
| `code/bfsbased_node_classification/gcn_baseline_runner.py` | Standalone GCN baseline |
| `code/bfsbased_node_classification/analyze_prl_resubmission.py` | Analysis helper |
| `code/bfsbased_node_classification/merge_prl_resubmission_runs.py` | Grouped-job merge helper |
| `slurm/*.sbatch` | Wulver HPC templates |

---

## Framing notes (careful claims)

- This is a feature-first selective graph correction method, not a global graph learner.
- Do not describe the repository as universally heterophily-robust; the evidence
  does not support that claim.
- Threshold evidence is mostly cross-run validation-choice evidence, not a full
  fixed-split per-threshold sweep.

---

## Archived / exploratory

- `code/bfsbased_node_classification/experimental_archived/` — legacy diagnostic scripts
- `reports/archive/` — superseded exports
- Older exploratory analysis under `reports/*.md`

---

## LaTeX manuscript

There is **no** `.tex` file in this repository; the paper lives in a separate writing environment.

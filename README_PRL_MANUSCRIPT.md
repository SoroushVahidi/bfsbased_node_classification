# PRL manuscript — repository evidence package

**This repository contains the evidence package for a Pattern Recognition Letters (PRL) submission.**

Start with the [root README](README.md) for install, layout, and smoke runs.

**Paper working title:** *Uncertainty-Gated Selective Graph Correction for Node Classification*  
**Target:** Pattern Recognition Letters (7 pages incl. references)

**Main method (submission):** **FINAL_V3** — reliability-gated selective graph correction (`code/bfsbased_node_classification/final_method_v3.py`, stable import: `code/final_method_v3.py`).  
**Baselines / ablations in the main benchmark table:** MLP-only, **SGC v1** (original selective correction), **V2_MULTIBRANCH** (archived multibranch ablation). These are **not** presented as competing final methods.

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

## Avoiding dual-package confusion (critical for drafting)

The repository contains **two different numerical stories**:

1. **FINAL_V3 (submission):** `reports/final_method_v3_results.csv` → `tables/main_results_prl.*` — **10 splits**, **six datasets**, reliability gate — **use this for the main paper.**
2. **Legacy UG-SGC / manuscript_runner line:** `logs/manuscript_gain_over_mlp_final_validation.csv`, `prl_final_additions/prl_writing_numbers.md`, `tables/prl_final_additions/prl_benchmark_summary.csv` — **nine datasets**, **30 comparisons per dataset**, **SGC v1-style** correction — **do not paste into the FINAL_V3 results section.**

Structural **UG-SGC-S** tables under `tables/prl_resubmission/` are a **third** line (supplementary smoke runs). See `reports/manuscript_repo_consistency_audit.md` for a full section-by-section map.

## Framing (careful claims)

- **Feature-first, reliability-aware selective graph correction** — not a claim to universally beat GNNs on all benchmarks.
- **Not** a dedicated heterophily “booster”: on strongly heterophilic Chameleon the method stays near the MLP on average; see `reports/safety_analysis.md` for split-level detail.
- **Threshold evidence:** cross-run validation choices vs test behavior; see `results_prl/README_threshold_sensitivity.md` where applicable.

## Supplementary PRL materials (logs / resubmission bundle)

| Purpose | Location |
|--------|----------|
| Regenerate extended tables/figures + inventory | `python3 scripts/prl_final_additions/build_all_prl_artifacts.py` |
| Phase 1 audit + inventory | `reports/prl_final_additions/PRL_PHASE1_AUDIT.md`, `artifact_inventory.json` |
| Phase 2 gap analysis | `reports/prl_final_additions/PRL_PHASE2_GAP_ANALYSIS.md` |
| Results §3 (gate / threshold) notes | `reports/prl_final_additions/PRL_RESULTS_SUBSECTION_3.md` |
| Safe claims & caveats | `reports/prl_final_additions/CLAIMS_AND_CAVEATS.md` |
| PRL resubmission method spec / readiness | `reports/prl_resubmission/PRL_METHOD_SPEC.md`, `REPO_READINESS_AUDIT.md` |
| Resubmission runner (GCN, APPNP, ablations) | `python3 code/bfsbased_node_classification/prl_resubmission_runner.py ...` |
| Merged benchmark CSV (supplementary) | `tables/prl_final_additions/prl_benchmark_summary.csv` |
| Older graphical abstract (legacy bundle) | `figures/prl_final_additions/graphical_abstract_prl.{pdf,png,svg}` |

## Archived / exploratory (do not treat as “main method”)

- **Legacy 5-split CSV snapshot:** `reports/archive/final_method_v3_results_5split.csv` (superseded by the 10-split log).
- **Historical table bundle:** `reports/archive/superseded_final_tables_prl/` — contains the frozen **per-split** export (`final_tables_prl.csv`) used to rebuild the canonical log; the pre-rendered `final_tables_prl.md` is a snapshot — cite `tables/main_results_prl.md` for the paper.
- **Legacy runners:** `code/bfsbased_node_classification/experimental_archived/README.md`
- **Exploration reports** (diagnostics, v2 phases, improvement sweeps): under `reports/*.md` — historical context only.

## LaTeX manuscript

There is **no** `.tex` file in this repository; keep the paper in a separate writing repo if needed.

## Drivers

- Core manuscript driver: `code/bfsbased_node_classification/manuscript_runner.py`
- **FINAL_V3** benchmark driver: `code/bfsbased_node_classification/run_final_evaluation.py`
- Focused resubmission driver: `code/bfsbased_node_classification/prl_resubmission_runner.py`
- Analysis helper: `code/bfsbased_node_classification/analyze_prl_resubmission.py`
- Slurm templates: `slurm/*.sbatch`

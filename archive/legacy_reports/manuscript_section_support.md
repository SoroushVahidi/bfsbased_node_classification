# Manuscript section support map

This file maps likely PRL manuscript sections to the repository artifacts that
support them.

**Canonical method/package for the paper:** `FINAL_V3`

**Do not use as the main paper evidence unless explicitly marked supplementary:**
- `UG-SGC` / `manuscript_runner` artifacts under `logs/*manuscript_final_validation*`
- structural / resubmission artifacts under `reports/prl_resubmission/`,
  `tables/prl_resubmission/`, and `logs/prl_resubmission/`

## 1. Title and framing

- Supporting files:
  - `README.md`
  - `README_PRL_MANUSCRIPT.md`
  - `reports/final_method_story.txt`
- What to cite/use:
  - feature-first framing
  - reliability-aware selective correction
  - conservative graph use rather than global propagation
- Caution:
  - the repository title still uses the older working title
    "Uncertainty-Gated Selective Graph Correction for Node Classification"
  - in the body of the paper, explicitly introduce `FINAL_V3` as the canonical
    method name

## 2. Method overview

- Supporting files:
  - `code/final_method_v3.py`
  - `code/bfsbased_node_classification/final_method_v3.py`
  - `reports/final_method_v3_analysis.md`
- What to cite/use:
  - MLP baseline
  - reliability score `R(v)`
  - uncertainty threshold `τ`
  - reliability threshold `ρ`
  - two weight profiles searched on validation
- Key method facts:
  - reliability signals: neighbor concentration, MLP-graph agreement, degree signal
  - final action rule: keep MLP if confident or unreliable, correct only when uncertain and reliable
- Caution:
  - `reports/prl_resubmission/PRL_METHOD_SPEC.md` is for the older `UG-SGC`
    line, not the canonical `FINAL_V3` method definition

## 3. Experimental setup

- Supporting files:
  - `tables/experimental_setup_prl.md`
  - `tables/experimental_setup_prl.csv`
  - `scripts/prl_final_additions/build_prl_auxiliary_tables.py`
  - `scripts/README_REPRODUCE.md`
- What to cite/use:
  - six datasets: Chameleon, Citeseer, Cora, Pubmed, Texas, Wisconsin
  - 60/20/20 train/val/test split protocol
  - 10 fixed GEO-GCN splits (`0–9`)
  - compared methods: `MLP_ONLY`, `BASELINE_SGC_V1`, `V2_MULTIBRANCH`, `FINAL_V3`
- Caution:
  - GCN/APPNP and structural-upgrade experiments live in the resubmission package
    and should be treated as supplementary unless the paper explicitly includes them

## 4. Main results table

- Supporting files:
  - `tables/main_results_prl.md`
  - `tables/main_results_prl.csv`
  - `reports/final_method_v3_results.csv`
  - `scripts/prl_final_additions/build_prl_v3_figures_and_tables.py`
- What to cite/use:
  - mean test accuracy and standard deviation over 10 splits
  - `FINAL_V3` mean dataset gain vs MLP:
    - Chameleon `+0.02 pp`
    - Citeseer `+2.59 pp`
    - Cora `+2.90 pp`
    - Pubmed `+1.29 pp`
    - Texas `+0.54 pp`
    - Wisconsin `+0.20 pp`
    - mean across six datasets `+1.26 pp`
- Caution:
  - do not mix in `tables/prl_final_additions/prl_benchmark_summary.csv`; that
    belongs to the older `UG-SGC` package

## 5. Safety / reduced harmful graph influence

- Supporting files:
  - `reports/safety_analysis.md`
  - `figures/safety_comparison.png`
  - `reports/final_method_v3_analysis.md`
- What to cite/use:
  - harmful split counts relative to MLP:
    - `FINAL_V3`: `2 / 60`
    - `SGC v1`: `4 / 60`
    - `V2_MULTIBRANCH`: `4 / 60`
  - exemplar cases:
    - Texas split 4: `SGC v1` harmful, `FINAL_V3` matches MLP
    - Texas split 5: all correction-family methods remain below MLP in the frozen log
- Caution:
  - keep the claim about safety narrow: reduced harmful graph influence, not universal immunity

## 6. Behavior / interpretability subsection

- Supporting files:
  - `reports/final_method_v3_analysis.md`
  - `figures/correction_rate_vs_homophily.png`
  - `figures/reliability_vs_accuracy.png`
  - `reports/final_method_v3_results.csv`
- What to cite/use:
  - mean corrected fraction across 10 splits:
    - Cora `23.7%`
    - Citeseer `20.0%`
    - Pubmed `23.9%`
    - Chameleon `5.8%`
    - Texas `2.2%`
    - Wisconsin `1.8%`
  - correction precision:
    - Cora `0.92`
    - Citeseer `0.84`
    - Pubmed `0.72`
    - Chameleon `0.53`
    - Texas `0.75`
    - Wisconsin `1.00`
- Caution:
  - branch fractions (`frac_confident`, `frac_unreliable`, `τ`, `ρ`) are only
    fully logged for splits `0–4` in the shipped frozen rebuild
  - the report already handles this correctly by restricting those specific
    aggregates to splits `0–4`

## 7. Ablation subsection

- Supporting files:
  - `tables/ablation_prl.md`
  - `tables/ablation_prl.csv`
  - `scripts/prl_final_additions/build_prl_auxiliary_tables.py`
- What to cite/use:
  - `FINAL_V3` row is recomputed from the current 10-split frozen log
  - archived rows are historical qualitative context only
- Caution:
  - this is not a full current-code 10-split ablation benchmark
  - if the manuscript needs code-faithful ablations, that would require new runs

## 8. Sensitivity subsection

- Supporting files:
  - `tables/sensitivity_prl.md`
  - `tables/sensitivity_prl.csv`
  - `results_prl/README_threshold_sensitivity.md`
- What to cite/use:
  - `tables/sensitivity_prl.*` for the historical coarse `ρ` snapshot
  - `results_prl/` only for older-line descriptive gate/threshold behavior
- Caution:
  - there is no full fixed-split threshold sweep for `FINAL_V3` in this repo
  - the `results_prl/` package is not the canonical `FINAL_V3` line

## 9. Limitations / discussion

- Supporting files:
  - `reports/final_method_v3_analysis.md`
  - `reports/final_method_story.txt`
  - `reports/safety_analysis.md`
- What to cite/use:
  - Chameleon stays near the MLP on average
  - Texas split 5 remains difficult
  - correction precision is not perfect even where net gains are positive
  - method is conservative and feature-first rather than a universal GNN replacement

## 10. Figures

- Canonical paper-facing figures:
  - `figures/correction_rate_vs_homophily.png`
  - `figures/safety_comparison.png`
  - `figures/reliability_vs_accuracy.png`
- Submission-only graphical abstract:
  - `figures/prl_graphical_abstract_v3.png`
- Caution:
  - `figures/prl_final_additions/graphical_abstract_prl.*` is older legacy art,
    not the preferred `FINAL_V3` graphical abstract

## 11. Reproducibility / data availability

- Supporting files:
  - `scripts/run_all_prl_results.sh`
  - `scripts/README_REPRODUCE.md`
  - `scripts/prl_final_additions/rebuild_final_v3_results_10split.py`
  - `code/bfsbased_node_classification/run_final_evaluation.py`
  - `data/splits/`
- What to cite/use:
  - one-command lightweight refresh from frozen CSVs
  - rebuild path from frozen 10-split export
  - heavy full rerun command if needed
- Caution:
  - a full rerun is not required for the current manuscript-facing package
  - Wulver is not needed for the repo’s current final-paper standardization

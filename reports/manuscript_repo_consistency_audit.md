# Manuscript ↔ repository consistency audit (PRL, FINAL_V3, 10-split)

**Audit date:** 2025-03-22 (repository state: branch `cursor/final-10-split-package-e4ab` or equivalent).  
**Scope:** Map each typical manuscript section to repo evidence; flag OK / inconsistent / missing / outdated and fix type.  
**Final protocol:** **10 GEO-GCN splits** (0–9) per dataset, six core datasets; submission method **FINAL_V3** (reliability-gated selective graph correction).

---

## Global findings (read first)

| Issue | Detail | Fix type |
|-------|--------|----------|
| **Dual evidence packages** | The repo still contains a rich **manuscript_runner / “UG-SGC”** trail (`logs/manuscript_gain_over_mlp_final_validation.csv`, `prl_final_additions/prl_writing_numbers.md`, `tables/prl_final_additions/prl_benchmark_summary.csv`, `reports/prl_final_additions/CLAIMS_AND_CAVEATS.md`) optimized for the **older selective correction** story. **Main PRL benchmark for FINAL_V3** is **`tables/main_results_prl.*` + `reports/final_method_v3_results.csv`.** | **Metadata / writing discipline** — banners added in this pass; do not paste old numbers into the FINAL_V3 paper. |
| **UG-SGC-S / structural** | `tables/prl_resubmission/*`, `analyze_prl_resubmission.py`, and related summaries use **UG-SGC-S** naming — **not** FINAL_V3. | **Label as supplementary / alternate line** only. |
| **`PRL_METHOD_SPEC.md`** | Describes **SGC v1** (margin gate + `threshold_high`), not the **R(v), τ, ρ** FINAL_V3 procedure. | **Banner + cross-ref** (done in this pass). |
| **Ablation table in archive** | `reports/archive/superseded_final_tables_prl/final_tables_prl.md` includes an “ablation study” table — **not** regenerated from current FINAL_V3 code in this repo state. | **Heavy recomputation** if that exact table must match current code; otherwise **omit or rewrite** as qualitative. |
| **No `.tex` in repo** | All section audits assume the LaTeX lives elsewhere; this audit is **repo evidence only**. | **Writing-only** for prose in the paper repo. |

---

## Phase 1 — Section-by-section audit

### 1. Title

| Field | Content |
|-------|---------|
| **Manuscript item** | Working title in `README_PRL_MANUSCRIPT.md`: *Uncertainty-Gated Selective Graph Correction for Node Classification* |
| **Required repo evidence** | Positioning note that “uncertainty-gated” in the title historically referred to margin gating; **body text should introduce reliability gating (FINAL_V3)** explicitly. |
| **Existing supporting files** | `README_PRL_MANUSCRIPT.md`, `reports/final_method_story.txt` |
| **Status** | **OK** for repo naming; **writing-only** risk if title implies old UG-SGC mechanism only. |
| **Fix type** | **Writing-only** (optional title tweak in external manuscript). |
| **Notes** | Repo README already states FINAL_V3 as submission method. |

### 2. Abstract

| Field | Content |
|-------|---------|
| **Manuscript item** | Abstract numbers and claims |
| **Required repo evidence** | 10-split means ± std; mean Δ vs MLP; one-line safety; no “beats all GNNs.” |
| **Existing supporting files** | `tables/main_results_prl.md`, `reports/final_method_story.txt`, `reports/safety_analysis.md` |
| **Status** | **OK** if abstract is drafted from those files; **inconsistent** if copied from `prl_writing_numbers.md` (9 datasets, SGC v1 story). |
| **Fix type** | **Metadata cleanup** (warnings on legacy number files — done). |
| **Notes** | Mean Δ FINAL_V3 vs MLP across six datasets ≈ **+1.26 pp** (unweighted mean of dataset means). |

### 3. Keywords

| Field | Content |
|-------|---------|
| **Manuscript item** | Keywords |
| **Required repo evidence** | None in repo |
| **Existing supporting files** | — |
| **Status** | **N/A** |
| **Fix type** | **Writing-only** |
| **Notes** | Suggest: node classification, graph correction, reliability, heterophily, MLP. |

### 4. Introduction

| Field | Content |
|-------|---------|
| **Manuscript item** | Problem, contribution list |
| **Required repo evidence** | Feature-first + selective correction + **reliability gate**; safety motivation |
| **Existing supporting files** | `reports/final_method_story.txt`, `reports/final_method_v3_analysis.md` §6 |
| **Status** | **OK** |
| **Fix type** | **Writing-only** |
| **Notes** | Avoid implying the paper’s method is only margin thresholding (that is SGC v1). |

### 5. Related work / positioning

| Field | Content |
|-------|---------|
| **Manuscript item** | GNN vs MLP, heterophily, post-hoc correction |
| **Required repo evidence** | Caveats: not claiming SOTA over all GNNs |
| **Existing supporting files** | `README_PRL_MANUSCRIPT.md` (Framing), `reports/final_method_v3_analysis.md` §6, `reports/prl_final_additions/CLAIMS_AND_CAVEATS.md` |
| **Status** | **OK** for claims discipline; **outdated** if `CLAIMS_AND_CAVEATS` cited without scope note (addresses older log claims). |
| **Fix type** | **Metadata cleanup** (banner on CLAIMS — done). |
| **Notes** | **Do not rewrite bib** per task; repo has no `.bib`. |

### 6. Method overview

| Field | Content |
|-------|---------|
| **Manuscript item** | High-level pipeline: MLP → uncertain nodes → gated graph correction |
| **Required repo evidence** | FINAL_V3: **τ** (margin), **ρ** (reliability R(v)), two weight profiles |
| **Existing supporting files** | `reports/final_method_v3_analysis.md` §2, `code/bfsbased_node_classification/final_method_v3.py` |
| **Status** | **OK** |
| **Fix type** | **Writing-only** |
| **Notes** | Contrast explicitly with SGC v1 (margin only). |

### 7. Detailed method subsection(s)

| Field | Content |
|-------|---------|
| **Manuscript item** | R(v) components, combined score S(v,c), search grid |
| **Required repo evidence** | Algorithm text aligned with code |
| **Existing supporting files** | `reports/final_method_v3_analysis.md` §2, `final_method_v3.py` |
| **Status** | **OK** |
| **Fix type** | **Writing-only** |
| **Notes** | `PRL_METHOD_SPEC.md` = **SGC v1** spec; use for “prior / baseline” or ignore for FINAL_V3 body (banner added). |

### 8. Inference / validation / implementation details

| Field | Content |
|-------|---------|
| **Manuscript item** | Validation-only selection; test evaluation |
| **Required repo evidence** | `run_final_evaluation.py`, split files |
| **Existing supporting files** | `code/bfsbased_node_classification/run_final_evaluation.py`, `data/splits/*.npz`, `scripts/README_REPRODUCE.md` |
| **Status** | **OK** — default **10 splits** |
| **Fix type** | None |
| **Notes** | Rebuild CSV: `scripts/prl_final_additions/rebuild_final_v3_results_10split.py` |

### 9. Experimental setup

| Field | Content |
|-------|---------|
| **Manuscript item** | Datasets, splits, baselines |
| **Required repo evidence** | 6 datasets; 10 splits; MLP, SGC v1, V2 multibranch, FINAL_V3 |
| **Existing supporting files** | `README_PRL_MANUSCRIPT.md`, `reports/final_method_v3_analysis.md` §3 opening, `tables/main_results_prl.md` header |
| **Status** | **OK** |
| **Fix type** | None |
| **Notes** | `prl_resubmission_runner` adds GCN/APPNP — **heavy**; optional supplementary. |

### 10. Main benchmark results

| Field | Content |
|-------|---------|
| **Manuscript item** | Main table |
| **Required repo evidence** | Aggregated test accuracy mean ± std |
| **Existing supporting files** | **`tables/main_results_prl.md`**, **`tables/main_results_prl.csv`**, source **`reports/final_method_v3_results.csv`** |
| **Status** | **OK** — internally consistent (10-split). |
| **Fix type** | **Lightweight rebuild:** `bash scripts/run_all_prl_results.sh` |
| **Notes** | **Do not** use `prl_benchmark_summary.csv` or `prl_writing_numbers.md` for this table. |

### 11. Safety / harmful-correction results

| Field | Content |
|-------|---------|
| **Manuscript item** | Harmful split counts; exemplar splits |
| **Required repo evidence** | Per-split `delta_vs_mlp` for each method |
| **Existing supporting files** | **`reports/safety_analysis.md`**, **`figures/safety_comparison.png`**, CSV above |
| **Status** | **OK** — FINAL_V3 **2/60**, SGC v1 **4/60**, etc. |
| **Fix type** | **Lightweight rebuild** of figure via `run_all_prl_results.sh` |
| **Notes** | Texas **split 4**: V1 harmful, V3 not; Texas **split 5**: all methods harmful vs MLP in frozen log — narrative must stay honest. |

### 12. Behavior / interpretability analysis

| Field | Content |
|-------|---------|
| **Manuscript item** | Branch fractions, correction rates, precision |
| **Required repo evidence** | Aggregates from CSV; note splits 5–9 may omit some diagnostic columns |
| **Existing supporting files** | `reports/final_method_v3_analysis.md` §4 (explicit 0–4 vs 10-split note), `figures/correction_rate_vs_homophily.png`, `figures/reliability_vs_accuracy.png` |
| **Status** | **OK** with documented limitation |
| **Fix type** | **Heavy** only if full per-split diagnostics for splits 5–9 are required without using archived export enrichment. |

### 13. Ablation study

| Field | Content |
|-------|---------|
| **Manuscript item** | Remove reliability / profiles / branches |
| **Required repo evidence** | Runner + table |
| **Existing supporting files** | **Archive** `final_tables_prl.md` Table 7 (historical); **no** current FINAL_V3-matched ablation CSV in `tables/` |
| **Status** | **Missing / outdated** for a code-faithful FINAL_V3 ablation table |
| **Fix type** | **Heavy recomputation** (extend `run_final_evaluation.py` or dedicated ablation runner × 10 splits × datasets) |
| **Notes** | Paper can cite **V2_MULTIBRANCH** in main table as architectural predecessor only. |

### 14. Sensitivity analysis

| Field | Content |
|-------|---------|
| **Manuscript item** | Threshold / gate sensitivity |
| **Required repo evidence** | Cross-run associations |
| **Existing supporting files** | `results_prl/README_threshold_sensitivity.md`, `reports/prl_final_additions/PRL_RESULTS_SUBSECTION_3.md`, related CSVs under `tables/prl_final_additions/` |
| **Status** | **OK** for descriptive cross-run narrative; protocol is **10 splits × 3 repeats** for that bundle (SGC lineage — check README there). |
| **Fix type** | **Writing-only** to align wording with that bundle’s definition; **heavy** for full τ sweep on fixed split. |

### 15. Discussion

| Field | Content |
|-------|---------|
| **Manuscript item** | Interpret results, relate to limitations |
| **Required repo evidence** | Same as §10–12 |
| **Status** | **OK** |
| **Fix type** | **Writing-only** |

### 16. Limitations

| Field | Content |
|-------|---------|
| **Manuscript item** | Chameleon near-MLP; split 5 Texas; no universal GNN beating |
| **Required repo evidence** | Numbers in analysis + safety |
| **Existing supporting files** | `reports/final_method_v3_analysis.md` §5, `reports/final_method_story.txt` |
| **Status** | **OK** |
| **Fix type** | **Writing-only** |

### 17. Conclusion

| Field | Content |
|-------|---------|
| **Manuscript item** | Summary contributions |
| **Required supporting files** | `reports/final_method_story.txt` |
| **Status** | **OK** |
| **Fix type** | **Writing-only** |

### 18. Graphical abstract

| Field | Content |
|-------|---------|
| **Manuscript item** | Single overview figure |
| **Required repo evidence** | FINAL_V3 concept figure |
| **Existing supporting files** | **`figures/prl_graphical_abstract_v3.png`** (current); `figures/prl_final_additions/graphical_abstract_prl.svg` (**legacy** UG-SGC-era art) |
| **Status** | **OK** |
| **Fix type** | **Lightweight rebuild** via `build_prl_v3_figures_and_tables.py` |
| **Notes** | Prefer **v3** PNG for submission. |

### 19. Main tables

| Field | Content |
|-------|---------|
| **Manuscript item** | Table 1 benchmark |
| **Required repo evidence** | `main_results_prl.*` |
| **Status** | **OK** |
| **Fix type** | Regenerate from CSV if needed |

### 20. Figures

| Field | Content |
|-------|---------|
| **Manuscript item** | Homophily, safety bars, reliability scatter |
| **Existing supporting files** | `figures/correction_rate_vs_homophily.png`, `figures/safety_comparison.png`, `figures/reliability_vs_accuracy.png` (from `logs/manuscript_regime_analysis_final_validation_main.csv` etc.) |
| **Status** | **OK** |
| **Fix type** | **Lightweight rebuild** |
| **Notes** | Homophily plot uses regime CSVs — not re-estimating homophily. |

### 21. Appendix / supplementary reproducibility

| Field | Content |
|-------|---------|
| **Manuscript item** | Commands, hardware, seeds |
| **Required repo evidence** | Reproduce scripts |
| **Existing supporting files** | `scripts/README_REPRODUCE.md`, `README_PRL_MANUSCRIPT.md`, `slurm/*.sbatch`, `run_metadata/` (Wulver snapshots — paths may differ) |
| **Status** | **OK** with caveat that `run_metadata/` reflects another filesystem |
| **Fix type** | **Metadata** — authors should paste commands from `scripts/README_REPRODUCE.md` |

### 22. Data / code / AI disclosure

| Field | Content |
|-------|---------|
| **Manuscript item** | Availability statements |
| **Required repo evidence** | Repo URL, license, splits bundled |
| **Existing supporting files** | `README.md`, `LICENSE`, `data/splits/` |
| **Status** | **OK** |
| **Fix type** | **Writing-only** |

---

## Phase 2 — Actions taken (low / moderate computation)

- Added **`reports/manuscript_repo_consistency_audit.md`** (this file).
- **Scope banners / cross-refs** so legacy UG-SGC materials are not mistaken for FINAL_V3 main evidence:
  - `prl_final_additions/prl_writing_numbers.md`
  - `reports/prl_final_additions/CLAIMS_AND_CAVEATS.md`
  - `reports/prl_resubmission/PRL_METHOD_SPEC.md`
  - `reports/prl_resubmission/README.md`
  - `README_PRL_MANUSCRIPT.md` (new subsection: **avoiding dual-package confusion**)
  - `tables/prl_final_additions/README.md` (new)
- **AGENTS.md**: overview now mentions **FINAL_V3** and **10-split** canonical results path.

No heavy experiments were run.

---

## Wulver plan (only if you need missing pieces)

| Goal | Command sketch | Rough runtime | Outputs to refresh |
|------|----------------|---------------|-------------------|
| Full **FINAL_V3** recompute (10 splits, 6 datasets) | `python3 code/bfsbased_node_classification/run_final_evaluation.py --split-dir data/splits --datasets cora citeseer pubmed chameleon texas wisconsin` | Many minutes CPU | `reports/final_method_v3_results.csv` → then `bash scripts/run_all_prl_results.sh` |
| **GCN/APPNP** supplementary benchmark | `python3 code/bfsbased_node_classification/prl_resubmission_runner.py` with 10 splits, full dataset list | **Heavy** (grid search) | `logs/prl_resubmission/*.jsonl`, `tables/prl_resubmission/*.csv` |
| **FINAL_V3 component ablations** | Not a single canned script in repo; would require new runner or extending `run_final_evaluation.py` | **Heavy** | New CSV + new table |

**Manuscript sections blocked** until those complete: optional **Related work** comparisons to GCN/APPNP; **Ablation** subsection (if required to be code-faithful).

---

## Final decision

**Outcome A — manuscript package fixed now (repo evidence layer)**

- **10-split FINAL_V3** canonical path is consistent: `reports/final_method_v3_results.csv` ↔ `tables/main_results_prl.*` ↔ `reports/final_method_v3_analysis.md` ↔ `reports/safety_analysis.md` ↔ `reports/final_method_story.txt` ↔ `README_PRL_MANUSCRIPT.md`.
- Remaining gaps are **writing** (external `.tex`) and **optional heavy** extras (GCN/APPNP grid, fresh ablations), not blockers for drafting the core FINAL_V3 paper from the repo.

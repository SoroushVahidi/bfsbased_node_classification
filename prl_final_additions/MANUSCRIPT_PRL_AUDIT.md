# PRL manuscript — repository & artifact audit

**Root:** `/home/sv96/node_classification_manuscript`  
**Date:** 2026-03-21 (audit); user date context 2026-03-18  
**Venue target:** Pattern Recognition Letters (~7 pages incl. references)

---

## 1. Executive summary

| Area | Status |
|------|--------|
| **Canonical benchmark + split evidence** | **Sufficient** — `logs/manuscript_gain_over_mlp_final_validation.csv` + `comparison_runs_manuscript_final_validation.jsonl` |
| **Correction-behavior table (descriptive)** | **Sufficient** — `logs/manuscript_correction_behavior_final_validation.csv` |
| **Regime / propagation baseline context** | **Sufficient** — `logs/manuscript_regime_analysis_final_validation.csv` |
| **Threshold “sensitivity” (honest scope)** | **Exists** — cross-run co-variation in `results_prl/` (not a fixed-split τ sweep) |
| **Case studies / fairness text** | **Exists** — MD under `logs/` |
| **LaTeX manuscript in repo** | **Not found** — no `.tex`; treat external editor as source of truth |
| **New Wulver jobs this session** | **None submitted** — not required for PRL package after inventory + low-cost merges |

**Canonical “final validation” prefix:** `*_manuscript_final_validation*` under `logs/` (merged job outputs).

**Obsolete / duplicate content:** Multiple **frozen snapshots** under `code/bfsbased_node_classification/diagnosis/preserve_*` duplicate `logs/` artifacts for archival; prefer **`logs/`** for citations. `logs/broken_job1_883410_backup_2026-03-19/` is explicitly superseded.

---

## 2. Manuscript files

| Item | Path | Note |
|------|------|------|
| LaTeX / elsarticle | *none in workspace* | Search: `**/*.tex` → 0 hits under project root |
| Narrative case studies | `/home/sv96/node_classification_manuscript/logs/manuscript_case_studies_final_validation.md` | Use for qualitative / examples |
| Fairness note | `/home/sv96/node_classification_manuscript/logs/comparison_fairness_report_manuscript_final_validation.md` | Supplement or trim for PRL |
| Job2-specific case studies | `/home/sv96/node_classification_manuscript/logs/case_studies_manuscript_final_validation_job2.md` | Subset / auxiliary |

---

## 3. Canonical manuscript-ready outputs (prefer these)

| Role | Exact path |
|------|------------|
| Per-dataset means + **win/tie/loss over 30 runs** | `/home/sv96/node_classification_manuscript/logs/manuscript_gain_over_mlp_final_validation.csv` |
| Correction weights / threshold / kNN flag (aggregated) | `/home/sv96/node_classification_manuscript/logs/manuscript_correction_behavior_final_validation.csv` |
| Dataset stats + MLP / prop / SGC means | `/home/sv96/node_classification_manuscript/logs/manuscript_regime_analysis_final_validation.csv` |
| Per-run JSONL (full reproducibility) | `/home/sv96/node_classification_manuscript/logs/comparison_runs_manuscript_final_validation.jsonl` |
| Aggregated summary JSON | `/home/sv96/node_classification_manuscript/logs/comparison_summary_manuscript_final_validation.json` |

**PRL-oriented merged table (generated, non-destructive):**  
`/home/sv96/node_classification_manuscript/prl_final_additions/tables/prl_main_benchmark_merged.csv`

---

## 4. Figures

### 4.1 Threshold cross-run package (`results_prl/`)

| File | Path |
|------|------|
| README (scope + caveats) | `/home/sv96/node_classification_manuscript/results_prl/README_threshold_sensitivity.md` |
| Builder script | `/home/sv96/node_classification_manuscript/results_prl/build_threshold_sensitivity_artifacts.py` |
| JSONL source rows | `/home/sv96/node_classification_manuscript/results_prl/threshold_sensitivity_crossruns.jsonl` |
| Summary CSVs | `/home/sv96/node_classification_manuscript/results_prl/threshold_uncertainty_crossrun_summary.csv`, `threshold_sensitivity_mini_table_tertiles.csv`, `threshold_sensitivity_summary.csv` |
| Figures PDF/PNG | `/home/sv96/node_classification_manuscript/results_prl/fig_threshold_vs_uncertain_fraction_crossruns.{pdf,png}`, `fig_threshold_vs_delta_crossruns.{pdf,png}`, `fig_threshold_sensitivity_panels_crossruns.{pdf,png}` |

**Claim discipline:** These support **descriptive** relationships across runs; they are **not** a controlled single-split threshold sweep (see README).

### 4.2 PRL additions — Δ vs MLP bar chart

| File | Path |
|------|------|
| PDF | `/home/sv96/node_classification_manuscript/prl_final_additions/figures/fig_delta_vs_mlp_bar.pdf` |
| PNG | `/home/sv96/node_classification_manuscript/prl_final_additions/figures/fig_delta_vs_mlp_bar.png` |
| Generator | `/home/sv96/node_classification_manuscript/prl_final_additions/scripts/generate_prl_benchmark_figure.py` |

---

## 5. Tables & machine-readable summaries

| Artifact | Path |
|----------|------|
| Merged benchmark (gain + regime `prop_mean`) | `/home/sv96/node_classification_manuscript/prl_final_additions/tables/prl_main_benchmark_merged.csv` |
| Win/tie/loss only | `/home/sv96/node_classification_manuscript/prl_final_additions/tables/prl_split_win_tie_loss.csv` |
| Correction behavior (copy of canonical) | `/home/sv96/node_classification_manuscript/prl_final_additions/tables/prl_correction_behavior_compact.csv` |
| Key numbers for writing | `/home/sv96/node_classification_manuscript/prl_final_additions/key_numbers.json` |
| Human-readable number sheet | `/home/sv96/node_classification_manuscript/prl_final_additions/prl_writing_numbers.md` |
| Gap analysis (A–H CSV) | `/home/sv96/node_classification_manuscript/prl_final_additions/prl_gap_analysis.csv` |
| Canonical inventory (preferred JSON) | `/home/sv96/node_classification_manuscript/prl_final_additions/prl_canonical_inventory.json` |
| Legacy pattern inventory (may list duplicates) | `/home/sv96/node_classification_manuscript/prl_final_additions/inventory_manifest.json` |
| Table rebuild script | `/home/sv96/node_classification_manuscript/prl_final_additions/scripts/build_prl_tables.py` |

**Metric caveat:** `manuscript_gain_over_mlp_final_validation.csv` uses `avg_changed_fraction` (global); `manuscript_correction_behavior_final_validation.csv` uses `avg_changed_of_uncertain_frac` (among uncertain). Do not mix without defining both.

---

## 6. Logs & Slurm

| Type | Paths |
|------|--------|
| Active Slurm templates | `/home/sv96/node_classification_manuscript/slurm/run_selective_validation_main_r3.sbatch`, `run_selective_validation_large_controls.sbatch`, `run_selective_validation_analysis.sbatch` |
| Historical copies | Under `code/bfsbased_node_classification/diagnosis/preserve_*/slurm/` (archival) |

---

## 7. Experiments inventory (what was run)

- **9 datasets** × **30 SGC runs** (10 splits × 3 repeats) as reflected in `manuscript_gain_over_mlp_final_validation.csv` (30 comparisons per dataset for win/tie/loss).
- **Correction-behavior aggregation** over the same 30 runs per dataset.
- **Threshold package** derived from `comparison_runs_manuscript_final_validation.jsonl` (see `results_prl/README_threshold_sensitivity.md`).

---

## 8. Obsolete / duplicate outputs (do not cite unless documenting history)

| Location | Reason |
|----------|--------|
| `logs/*_final_validation_main.csv` / `job1` / `job2` split artifacts | Earlier pipeline stages; **superseded** by `*_final_validation.csv` without `_main` where merged |
| `logs/broken_job1_883410_backup_2026-03-19/` | Broken job backup |
| `code/bfsbased_node_classification/diagnosis/preserve_*` | Point-in-time copies; use `logs/` for paper |

---

## 9. Gap analysis (A–H) — labels

| ID | Topic | Label | Rationale |
|----|-------|-------|-----------|
| A | Main benchmark table | **EXISTS AND SUFFICIENT** | Canonical CSV + merged PRL table |
| B | Score weighting / correction behavior | **EXISTS AND SUFFICIENT** | `manuscript_correction_behavior_final_validation.csv`; kNN **no** everywhere — align text |
| C | Threshold sensitivity | **EXISTS BUT NEEDS CLEANUP** | Strong **descriptive** cross-run evidence in `results_prl/`; **cannot** claim fixed-split τ sweep without new instrumentation |
| D | Qualitative correction analysis | **EXISTS BUT NEEDS CLEANUP** | `manuscript_case_studies_final_validation.md` — trim to 7 pages |
| E | Statistical stability / win-tie-loss | **EXISTS AND SUFFICIENT** | Per-dataset counts in gain CSV; **not** formal paired tests unless computed separately |
| F | Compact figure | **EXISTS** — bar chart + optional threshold panel | Minimize: **1** main figure (Δ bar) **or** threshold supplement |
| G | Appendix / supplement artifacts | **OPTIONAL** | `results_prl/` figures, fairness MD, full JSONL |
| H | Reproducibility | **EXISTS** — `comparison_runs_manuscript_final_validation.jsonl` + `manuscript_runner.py` | Bundle hash / env optional for camera-ready |

---

## 10. Phase 4 — New experiments needed?

| Candidate | Verdict | Notes |
|-----------|---------|-------|
| True τ sweep on validation | **NICE TO HAVE** | **Expensive** + needs logging change; not essential if claims stay descriptive |
| Repeated-split summary | **EXISTS** (30 runs) | Already the stability story |
| Per-dataset corrected-node fraction | **PARTIALLY EXISTS** | `avg_uncertain_fraction`, `avg_changed_fraction` in gain CSV |
| Qualitative case studies | **EXISTS** | Edit down |
| Component “ablation” | **NOT WORTH IT FOR PRL** unless precomputed — user forbids fake removals |
| Propagation-only variant | **ONLY IF** already in JSONL with clear label — verify before claiming |

**Conclusion:** **No essential new cluster runs** for a compact PRL version given current artifacts.

---

## 11. Files created/updated in `prl_final_additions/` (this pass)

- `MANUSCRIPT_PRL_AUDIT.md` (this file)
- `prl_canonical_inventory.json`
- `prl_gap_analysis.csv`
- `prl_writing_numbers.md`
- `scripts/build_prl_tables.py`
- `scripts/generate_prl_benchmark_figure.py`
- `figures/fig_delta_vs_mlp_bar.{pdf,png}`
- Tables / `key_numbers.json` (regenerated if `build_prl_tables.py` run)

---

## 12. Suggested PRL package (minimal)

**Tables:** (1) Main results: MLP mean, SGC mean, Δ, optional win/tie/loss line in caption.  
**Figures:** (1) Horizontal bar Δ(SGC−MLP). Optional supplement: one threshold cross-run panel.  
**Claims:** Selective graph correction **can** help on **citation-style** benchmarks; **not** universal; **descriptive** regime commentary only.

# Final repo polish audit

This report records the final manuscript-facing cleanup pass for the repository.

## Scope

- Repository only
- No manuscript `.tex` or PDF editing
- Goal: make the frozen PRL package clear, consistent, reproducible, and easy to
  cite from a separate writing repository

## Canonical decision

The repository is now standardized around the following rule:

- **Canonical short-paper method:** `FINAL_V3`
- **Historical / supplementary only:** `UG-SGC`, `V2`, `UG-SGC-S`, and the
  broader resubmission package

## What was inconsistent

1. Mixed method naming could still confuse a reader.
- Root documentation explained several method lines, but did not state early
  enough that `FINAL_V3` is the canonical short-paper package.
- `README_PRL_MANUSCRIPT.md` still described the structural line as a "newer
  stronger-method package", which could be read as replacing the frozen paper story.
- Some older manuscript-support files still used generic phrases like
  "proposed method" without clearly anchoring them to the older `UG-SGC` line.

2. `FINAL_V3` code/docs had a few stale or inaccurate details.
- `run_final_evaluation.py` documented `--output-tag` but did not use it.
- `final_method_v3.py` reported `search_space_size` without multiplying by the
  number of evaluated weight profiles.
- `final_method_v3.py` documented `seed` as if it always controlled the run,
  even though callers may already supply `mlp_probs`.
- `run_final_evaluation.py` kept dead v1 reconstruction variables and did not
  explain that legacy `frac_corrected` stores uncertain-node coverage for v1.

3. One figure-generation script treated missing diagnostics as zeros.
- In `build_prl_v3_figures_and_tables.py`, empty `FINAL_V3` diagnostic fields
  were coerced to `0.0`, which could distort the reliability scatter plot.

4. 10-split standardization needed clearer wording.
- The core benchmark is fully standardized on **10 splits**.
- However, the frozen rebuilt CSV carries branch-fraction / threshold / runtime
  diagnostics only for splits `0–4`; this caveat existed in the analysis report
  but was not surfaced clearly in the reproducibility instructions.

5. The repo needed a direct manuscript-support map.
- There was already a detailed consistency audit, but not a concise writer-facing
  map from manuscript sections to exact files, figures, tables, and caveats.

## What was fixed

### Documentation and naming

- Updated `README.md` to state explicitly that `FINAL_V3` is the canonical PRL package.
- Updated `README_PRL_MANUSCRIPT.md` to:
  - mark `FINAL_V3` as canonical
  - demote `UG-SGC` / structural materials to supplementary context
  - fix figure-path inconsistencies
- Added/strengthened scope notes in:
  - `reports/prl_resubmission/REPO_READINESS_AUDIT.md`
  - `prl_final_additions/MANUSCRIPT_PRL_AUDIT.md`
  - `reports/prl_final_additions/CLAIMS_AND_CAVEATS.md`

### FINAL_V3 code and reproducibility path

- Updated `code/bfsbased_node_classification/final_method_v3.py`:
  - clarified `seed` semantics
  - fixed `search_space_size`
- Updated `code/bfsbased_node_classification/run_final_evaluation.py`:
  - removed misleading dead v1 variables
  - documented the legacy `frac_corrected` caveat
  - made `--output-tag` functional for non-canonical side outputs
- Updated `scripts/run_all_prl_results.sh` messaging to reflect both
  `py_compile` checks accurately.
- Updated `scripts/README_REPRODUCE.md` with the exact 10-split caveat for
  branch/threshold/runtime diagnostics and an example of side-output reruns.

### Figures / tables

- Updated `scripts/prl_final_additions/build_prl_v3_figures_and_tables.py` so
  missing `FINAL_V3` diagnostics are treated as `NaN` rather than `0.0`.
- This keeps the reliability figure aligned with the shipped frozen CSV instead
  of silently inventing low unreliability values for splits without those fields.

### Writer support

- Added `reports/manuscript_section_support.md` as a section-by-section map from
  likely manuscript sections to exact repository evidence.

## 10-split standardization status

### Fully standardized now

- `reports/final_method_v3_results.csv`
- `tables/main_results_prl.md`
- `tables/main_results_prl.csv`
- `tables/experimental_setup_prl.md`
- `tables/experimental_setup_prl.csv`
- `reports/safety_analysis.md`
- `figures/safety_comparison.png`

These support the core `FINAL_V3` benchmark on **6 datasets × 10 splits**.

### Standardized with documented caveat

- `reports/final_method_v3_analysis.md` behavior subsection
- `figures/reliability_vs_accuracy.png`
- branch-fraction / threshold summaries derived from `reports/final_method_v3_results.csv`

These remain usable because the report explicitly restricts the affected
diagnostic aggregates to splits `0–4`.

### Historical / supplementary only

- `tables/ablation_prl.*`
- `tables/sensitivity_prl.*`
- `reports/prl_final_additions/*`
- `results_prl/*`
- `reports/prl_resubmission/*`
- `tables/prl_resubmission/*`

These are retained for provenance and optional supplementary discussion, but
they should not replace the frozen `FINAL_V3` paper package.

## Was Wulver needed?

No.

The remaining issues were documentation, metadata, and lightweight artifact
generation issues. No new training or HPC reruns were required to make the
canonical short-paper package consistent.

## Remaining historical but non-blocking items

1. `tables/ablation_prl.*` still mixes one live `FINAL_V3` row with historical archived rows.
- This is acceptable only if labeled as qualitative supplementary context.

2. `tables/sensitivity_prl.*` is still a historical 3-split snapshot.
- This is acceptable only if explicitly labeled as historical sensitivity context.

3. The repository still contains resubmission / structural-upgrade materials.
- They are now more clearly labeled, but they remain present by design for provenance.

## Manuscript-readiness decision

**Outcome A — repo polished and manuscript-ready**

The repository is now in a good state for writing the manuscript from a
separate repo:

- canonical method/package is clear
- reproducibility path is lightweight and documented
- frozen 10-split benchmark package is consistent
- older method lines are more clearly marked as historical or supplementary
- manuscript writers now have a direct support map to the exact artifacts

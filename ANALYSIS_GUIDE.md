# Analysis guide

This file is for external readers, including LLMs, who need to understand the
repository quickly without reverse-engineering the full project history.

## Read this first

If you only read five files, read them in this order:

1. `README.md`
2. `README_PRL_MANUSCRIPT.md`
3. `reports/final_method_v3_analysis.md`
4. `reports/manuscript_section_support.md`
5. `reports/prl_submission_readiness.md`

## Canonical answer to "what is the paper method?"

For the short PRL paper, the canonical method is:

- `FINAL_V3`

Primary implementation:

- `code/final_method_v3.py`
- `code/bfsbased_node_classification/final_method_v3.py`

Primary frozen result artifacts:

- `reports/final_method_v3_results.csv`
- `tables/main_results_prl.md`
- `tables/main_results_prl.csv`
- `reports/safety_analysis.md`

## Repository structure in one page

### Canonical short-paper package

Use these for the main manuscript:

- `reports/final_method_v3_results.csv`
- `tables/main_results_prl.*`
- `tables/experimental_setup_prl.*`
- `reports/final_method_v3_analysis.md`
- `reports/final_method_story.txt`
- `reports/safety_analysis.md`
- `figures/prl_graphical_abstract_v3.{png,pdf,svg}`
- `figures/correction_rate_vs_homophily.png`
- `figures/safety_comparison.png`
- `figures/reliability_vs_accuracy.png`

### Writer-support / audit layer

Use these to understand the repo and draft the paper cleanly:

- `reports/manuscript_section_support.md`
- `reports/final_repo_polish_audit.md`
- `reports/prl_repo_audit.md`
- `reports/prl_gap_analysis.md`
- `reports/prl_submission_readiness.md`
- `reports/prl_insertable_text.md`
- `reports/prl_propagation_baseline_definition.md`
- `reports/prl_related_work_patch.md`
- `reports/prl_statistical_summary.md`

### Older `UG-SGC` / manuscript-runner line

These are useful historical or supplementary artifacts, but they are **not**
the canonical `FINAL_V3` paper package:

- `logs/*manuscript_final_validation*`
- `tables/prl_final_additions/*`
- `reports/prl_final_additions/*`
- `results_prl/*`

### Resubmission / structural-upgrade line

These are supplementary exploratory packages, not the frozen short-paper line:

- `logs/prl_resubmission/*`
- `tables/prl_resubmission/*`
- `reports/prl_resubmission/*`
- `code/bfsbased_node_classification/prl_resubmission_runner.py`

## What not to confuse

### Do not confuse `FINAL_V3` with `UG-SGC`

- `FINAL_V3` is reliability-gated and is the canonical short-paper method.
- `UG-SGC` is the older uncertainty-gated / manuscript-runner line.

### Do not treat all tables as equal

For the main paper, prefer:

- `tables/main_results_prl.*`

Do not substitute:

- `tables/prl_final_additions/prl_benchmark_summary.csv`

unless you explicitly want the older `UG-SGC` package.

### Do not assume all threshold artifacts are `FINAL_V3`

- `results_prl/*` is an older cross-run threshold package from the `UG-SGC` line.
- It is useful for supplementary gate behavior context, but it is not the main
  `FINAL_V3` evidence package.

## Reproducibility

### Refresh canonical figures/tables without retraining

```bash
bash scripts/run_all_prl_results.sh
```

### Rebuild the frozen canonical CSV from archived export

```bash
python3 scripts/prl_final_additions/rebuild_final_v3_results_10split.py
```

### Full heavy rerun of the canonical benchmark

```bash
python3 code/bfsbased_node_classification/run_final_evaluation.py \
  --split-dir data/splits \
  --datasets cora citeseer pubmed chameleon texas wisconsin
```

## Important caveats

- There is **no** LaTeX manuscript in this repository.
- There is **no** bibliography file in this repository.
- Some auxiliary `FINAL_V3` diagnostics are only available for splits `0–4`,
  although the core benchmark itself is standardized on 10 splits.
- The current short-paper framing is intentionally narrow:
  - strong MLP baseline
  - selective graph correction
  - reduced harmful graph influence
  - conservative regime-aware behavior

## Best single-sentence summary

This repository contains a frozen short-paper package for `FINAL_V3`
(reliability-gated selective graph correction on top of a strong MLP), plus
older `UG-SGC` and broader resubmission/structural packages that should be read
as supplementary provenance rather than the main PRL evidence line.

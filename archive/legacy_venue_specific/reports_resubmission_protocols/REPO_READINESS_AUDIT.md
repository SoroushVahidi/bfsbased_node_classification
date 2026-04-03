# PRL resubmission readiness audit

> **Relation to the frozen paper package:** this document tracks the separate
> resubmission / extension workflow. The canonical short-paper package on the
> default branch is still `FINAL_V3`; use this audit only for supplementary
> baseline/ablation planning and provenance.

## What already existed

- Canonical final-validation manuscript package under `logs/*manuscript_final_validation*`
- PRL-facing tables/figures/reports under:
  - `tables/prl_final_additions/`
  - `figures/prl_final_additions/`
  - `reports/prl_final_additions/`
- Threshold/gate **cross-run** package under `results_prl/`
- Graphical abstract under `figures/prl_final_additions/graphical_abstract_prl.*`

## What was still missing for a stronger PRL rewrite

1. **Standard graph baselines**
- The live repo had `MLP`, `prop_only`, `prop_mlp_select`, `gated_mlp_prop`, and the main selective-correction method.
- It did **not** have a compact standard GNN package such as `GCN` / `APPNP`.

2. **True ablations**
- The repo had descriptive behavior summaries, but not controlled SGC variants that disable the gate or remove individual score terms.

3. **Protocol-facing outputs for the new package**
- The existing repo already documented the main manuscript-final-validation protocol.
- It did not yet have a separate focused resubmission package for:
  - stronger baselines on the core datasets
  - ablation comparisons against default SGC
  - report files oriented around “what PRL claims are now supportable”

## New resubmission pipeline added

- `code/bfsbased_node_classification/standard_node_baselines.py`
- `code/bfsbased_node_classification/prl_resubmission_runner.py`
- `code/bfsbased_node_classification/analyze_prl_resubmission.py`
- `slurm/run_prl_resubmission_core.sbatch`
- `slurm/run_prl_resubmission_analysis.sbatch`
- `reports/prl_resubmission/PRL_METHOD_SPEC.md`

## Focused scope for the new experiments

Datasets prioritized for the new package:
- `cora`
- `citeseer`
- `pubmed`
- `chameleon`
- `wisconsin`

Rationale:
- three clear positives from the current manuscript story,
- one near-neutral contrast,
- one small-negative/contrast regime.

## Claim discipline

- Keep the paper narrow and feature-first.
- Use the new ablations to support statements about what matters **within this method**.
- Do **not** convert the paper into a universal heterophily-robust claim.

# PRL resubmission package

**Relation to FINAL_V3:** The **short PRL manuscript** should treat **FINAL_V3** as the proposed method (`README_PRL_MANUSCRIPT.md`, `tables/main_results_prl.*`). This directory and `prl_resubmission_runner.py` focus on **extended baselines** (GCN, APPNP) and **structural / UG-SGC-S** smoke experiments — **supplementary**, not a replacement for the FINAL_V3 evidence package.

This directory holds the **new** evidence package intended to close the main
technical gaps for a PRL rewrite:

- stronger baseline comparison (`GCN`, `APPNP`, plus existing `MLP` / propagation)
- true SGC ablations
- method-spec extraction from code
- protocol / claims-facing reports

## Main files

- `PRL_METHOD_SPEC.md`
- `REPO_READINESS_AUDIT.md`
- `protocol_<output_tag>.json`
- `protocol_<output_tag>.md`
- `summary_<output_tag>.md`
- `claims_and_risks_<output_tag>.md`
- `STRUCTURAL_UPGRADE_SMOKE_NOTE.md`
- `RUN_STATUS.md`

## Matching raw / tabular outputs

- `logs/prl_resubmission/comparison_runs_<output_tag>.jsonl`
- `tables/prl_resubmission/baseline_comparison_<output_tag>.csv`
- `tables/prl_resubmission/ablation_comparison_<output_tag>.csv`

## Runner commands

Local / interactive:

```bash
python3 code/bfsbased_node_classification/prl_resubmission_runner.py \
  --datasets cora citeseer pubmed chameleon wisconsin \
  --splits 0 1 2 3 4 5 6 7 8 9 \
  --repeats 1 \
  --output-tag prl_resubmission_core_r1
```

Wulver:

```bash
sbatch slurm/run_prl_resubmission_core.sbatch
```

## Structural-upgrade jobs

The structural selective-correction extension has its own grouped Wulver
submission scripts:

- `slurm/run_prl_structural_group1.sbatch`
- `slurm/run_prl_structural_group2.sbatch`
- `slurm/run_prl_structural_merge_analysis.sbatch`

Current status is tracked in:

- `reports/prl_resubmission/RUN_STATUS.md`

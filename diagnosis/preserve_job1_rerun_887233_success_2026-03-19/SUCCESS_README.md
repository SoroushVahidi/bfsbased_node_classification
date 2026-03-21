# JOB 1 — fixed successful rerun (SLURM 887233)

**Created:** 2026-03-20T11:53:26

This bundle preserves the **successful** manuscript-validation rerun after fixing the `mlp_only` call
(`train_mlp_and_predict`: removed unsupported `val_indices` / `patience` kwargs).

## Key outcomes (from this run)
- **630** JSONL records: **210** each for `mlp_only`, `prop_only`, `selective_graph_correction`; all **`success=true`**.
- **7** datasets × **10** splits × **3** repeats × **3** methods.
- Manuscript tables under `logs/`: gain, correction behavior, regime analysis, case studies; plus summary JSON and fairness report.

## Contents
- Patched `manuscript_runner.py`, rerun patch note, JOB 1 sbatch script, SLURM stdout/stderr, and full `logs/` artifacts listed in the tarball.

**Note:** Do not treat this as replacing the earlier broken-run preservation branch; this is a separate success snapshot.

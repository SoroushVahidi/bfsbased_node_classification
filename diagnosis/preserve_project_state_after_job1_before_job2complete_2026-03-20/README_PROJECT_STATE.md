# Project state preservation — after repaired JOB 1, before JOB 2 complete (2026-03-20)

**Purpose:** Off-Wulver inspection, drafting, and continuity while clusters are unavailable.

## JOB 1 (repaired rerun)
- **Status:** Succeeded; outputs are **manuscript-usable** (full 630-run grid; all methods `success=true`).
- **Original failure root cause:** `manuscript_runner.py` called `train_mlp_and_predict(...)` with unsupported kwargs (`val_indices`, `patience`), breaking all `mlp_only` runs and blocking `selective_graph_correction`.
- **Fix applied:** Removed `val_indices=...` and `patience=...` from the `mlp_only` call site in `manuscript_runner.py`.

## JOB 2
- **Status:** **Still running or incomplete at bundle time** (see `JOB2_STATUS.md` and `logs/job2_in_progress/`).
- **Caveat:** Any JOB 2 result files in this bundle may be **partial** (e.g., growing JSONL only).

## PRL local assets
- See **PRL_ASSETS_NOTE.md**.

## Layout
- `code/` — pipeline sources
- `slurm/` — JOB 1 / JOB 2 / analysis sbatch scripts
- `run_metadata/` — reproducibility snapshots + JOB 1 patch note
- `logs/job1_*` — successful JOB 1 manuscript artifacts + JOB 1 SLURM logs + symlink documentation
- `logs/job2_in_progress/` — current JOB 2 SLURM logs and any partial outputs

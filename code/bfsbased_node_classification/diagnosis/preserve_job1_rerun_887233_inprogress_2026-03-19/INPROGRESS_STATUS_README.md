# JOB 1 rerun — in-progress preservation snapshot

**Status:** SLURM job **887233** was **still RUNNING** when this bundle was created.

**Timestamp (UTC/local):** 2026-03-19T18:00:10-04:00

## Important caveats
- All artifacts here may be **partial or incomplete** until the job finishes.
- Final manuscript tables / summary JSON may not exist yet.
- SLURM `.out` may appear small/stale while the job runs because Python often **buffers stdout** when not attached to a TTY; check `logs/comparison_runs_manuscript_final_validation_main.jsonl` growth for live progress.

## Previous failure (broken JOB 1)
- `mlp_only` failed for all runs with:
  - `train_mlp_and_predict() got an unexpected keyword argument 'val_indices'`
- Because `mlp_probs` never materialized, `selective_graph_correction` was never dispatched.

## Fix applied for this rerun
- Removed unsupported kwargs `val_indices=val_idx` and `patience=30` from the `train_mlp_and_predict(...)` call in `manuscript_runner.py`.

## Partial progress signal (at bundle time)
- See `logs/comparison_runs_manuscript_final_validation_main.jsonl` (if included) and `MANIFEST_INCLUDED.txt`.

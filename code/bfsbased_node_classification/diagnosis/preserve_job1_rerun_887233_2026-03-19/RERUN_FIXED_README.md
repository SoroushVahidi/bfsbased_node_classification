RERUN FIXED README (JOB 1 rerun 887233)
========================================
This archive was created after a fixed re-run of JOB 1 (SLURM job id: 887233).

Root cause of the original JOB 1 failure:
- Argument mismatch: `manuscript_runner.py` passed unsupported keyword args into
  `train_mlp_and_predict(...)` in `bfsbased-full-investigate-homophil.py`, causing
  `mlp_only` to crash for all records.

What was changed:
- In `manuscript_runner.py` (mlp_only block), removed unsupported keyword arguments:
  - `val_indices=val_idx`
  - `patience=30`

What to verify after the rerun completes:
- `mlp_only` now appears in the main JSONL with successful records.
- `selective_graph_correction` now appears (i.e., it is dispatched once `mlp_probs` exists).

After rerun success, the bundle should include:
- rerun stdout/stderr
- rerun main JSONL + summary + fairness report
- rerun gain/correction/regime/case-study outputs
- updated symlink info (created only after success)

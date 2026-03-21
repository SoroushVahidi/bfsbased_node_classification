# Post–JOB 2 preservation (2026-03-21)

**Original JOB 1 failure:** `train_mlp_and_predict()` received unsupported kwargs (`val_indices`, `patience`) → all `mlp_only` failed; SGC never ran.
**Fix:** removed those kwargs from `manuscript_runner.py` (`mlp_only` block).

**JOB 1 (repaired):** succeeded — full 7-dataset grid; manuscript tables + JSONL complete.
**JOB 2:** succeeded — `pubmed`, `squirrel`; 180 JSONL rows; all methods `success=true`.

**Analysis merge inputs:** `comparison_runs_manuscript_final_validation_job1.jsonl` (symlink to `..._main.jsonl`) and `..._job2.jsonl` both present under `logs/` on Wulver.

**Next:** run analysis SLURM when ready (not executed in this bundle).

Bundle for off-Wulver inspection and drafting.

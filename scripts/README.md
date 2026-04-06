# Scripts index

Short map of `scripts/` — for full reproduction steps see [`README_REPRODUCE.md`](README_REPRODUCE.md).

| Path | Role |
|------|------|
| [`README_REPRODUCE.md`](README_REPRODUCE.md) | End-to-end benchmark re-run, tags, frozen outputs |
| [`run_all_selective_correction_results.sh`](run_all_selective_correction_results.sh) | Regenerate canonical tables/figures from existing CSVs (no training) |
| [`check_canonical.py`](check_canonical.py) | Verify frozen canonical files are present and consistent |
| [`prepare_extended_dataset_splits.py`](prepare_extended_dataset_splits.py) | Prepare extended / OGB-style split `.npz` files |
| [`smoke_begga_baselines.py`](smoke_begga_baselines.py) | Quick smoke test for heterophily-style baselines |
| [`baseline_comparison_wulver/`](baseline_comparison_wulver/) | Slurm launchers, aggregation, and docs for large comparison / robustness sweeps |
| [`artifacts/`](artifacts/) | Artifact-retention helpers (`make_archive_manifest.py`, `summarize_slurm_logs.py`) |
| [`build_artifacts/`](build_artifacts/) | Python builders for paper tables and figures (called from `run_all_selective_correction_results.sh`) |

Cluster job definitions live in [`../slurm/README.md`](../slurm/README.md).
Artifact policy and manifest convention: [`../docs/ARTIFACT_RETENTION_POLICY.md`](../docs/ARTIFACT_RETENTION_POLICY.md), [`../run_metadata/manifests/README.md`](../run_metadata/manifests/README.md).

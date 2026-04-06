# Sweep Manifest Convention

This directory stores JSON manifests for experiment sweeps whose raw outputs are kept outside Git.

## Purpose

Each manifest preserves provenance for bulky, gitignored artifacts by recording:

- what was run,
- where raw outputs lived,
- which job IDs produced them,
- where archives are stored,
- how to verify integrity (`sha256`, size).

## Naming convention

Recommended filename pattern:

- `manifest_<sweep_name>_<YYYYMMDD>.json`

Examples:

- `manifest_wulver_baseline_comparison_20260404.json`
- `manifest_wulver_label_budget_20260404.json`

## Required fields

Use the schema example in `manifest_schema_example.json`.

Minimum required keys:

- `sweep_name`
- `created_at`
- `git_commit`
- `runner_script`
- `datasets`
- `methods`
- `split_ids`
- `job_ids`
- `raw_output_dir`
- `archive_filename`
- `archive_sha256`
- `archive_size_bytes`
- `storage_location`
- `notes`

## Draft manifests

If an external archive is not created yet, draft values are allowed:

- `archive_filename`: `"TBD"`
- `archive_sha256`: `"TBD"`
- `archive_size_bytes`: `null`
- `storage_location`: `"TBD"`

Mark this clearly in `notes`.

## Tooling

Generate a manifest with:

```bash
python3 scripts/artifacts/make_archive_manifest.py --help
```

Summarize Slurm logs with:

```bash
python3 scripts/artifacts/summarize_slurm_logs.py --help
```

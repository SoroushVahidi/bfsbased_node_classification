# Artifact Retention Policy

This policy defines how experiment artifacts are retained in this repository without changing canonical scientific claims.

## Scope

Applies to:

- cluster sweeps and local benchmark runs,
- generated outputs under `outputs/`,
- Slurm runtime logs under `slurm/*.out` and `slurm/*.err`,
- sweep manifests under `run_metadata/manifests/`,
- runtime caches and temporary data.

## MUST be committed

- Code, launch scripts, and aggregation scripts used to run sweeps.
- Small, claim-bearing summaries that are reviewed by humans:
  - markdown status notes in `reports/`,
  - compact summary tables intended for interpretation.
- Sweep provenance manifests in `run_metadata/manifests/`:
  - include run identity, inputs, job IDs, archive pointer/checksum, and notes.
- Slurm runtime note summaries (markdown) that capture important warnings/errors and actions.

## MUST stay gitignored

- Bulky raw sweep outputs:
  - `outputs/baseline_comparison/*`
  - `outputs/label_budget/*`
  - `outputs/graph_noise_robustness/*`
- Raw Slurm logs:
  - `slurm/*.out`
  - `slurm/*.err`
- Transient caches and runtime temp data:
  - PyG/torch caches,
  - virtualenv/build/editor swap artifacts,
  - other reproducible machine-local runtime files.

## SHOULD be archived outside Git

Archive raw sweep outputs outside Git when they are needed for re-analysis or auditing:

- preferred form: compressed tarball per sweep family/tag,
- include checksum (`sha256`) and byte size,
- record external storage pointer (path/URI/object key),
- register archive metadata in a committed manifest JSON.

Use `scripts/artifacts/make_archive_manifest.py` to generate/update manifests.

## Can be deleted after extraction

After summaries and manifests are committed, it is acceptable to delete local raw artifacts:

- raw `outputs/...` directories,
- `slurm/*.out` and `slurm/*.err`,
- transient caches.

Condition: key outcomes are already captured in:

1. status/report markdown in `reports/`,
2. manifest JSON in `run_metadata/manifests/`,
3. checksums/pointers for any off-Git archive.

## Sweep-family workflow (repeatable)

For each sweep family (for example baseline comparison, label budget, graph noise):

1. Run jobs and collect raw outputs under `outputs/<family>/<tag>/`.
2. Run aggregation scripts and produce compact summary artifacts.
3. Generate Slurm runtime notes:
   - `python3 scripts/artifacts/summarize_slurm_logs.py ...`
4. Generate/update manifest:
   - `python3 scripts/artifacts/make_archive_manifest.py ...`
5. Commit only:
   - code/docs,
   - compact report notes,
   - manifest JSON.
6. Optionally archive raw outputs outside Git and record checksum/pointer in manifest.
7. Optionally delete local raw outputs and raw Slurm logs.

## Guardrails

- Do not modify frozen canonical evidence files unless explicitly approved.
- Do not commit large regenerated `outputs/...` payloads.
- Keep provenance fields complete enough that another person can locate and verify archived raw data.

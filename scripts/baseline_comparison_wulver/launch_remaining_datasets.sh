#!/usr/bin/env bash
# Submit the 3-dataset follow-on (actor, cornell, squirrel) with the SAME output tag
# as the main baseline_comparison_array job so aggregate_and_report sees 9 shards.
#
# Usage:
#   export BASELINE_OUTPUT_TAG=wulver_baseline_cmp_20260403_150529   # copy from running job
#   ./scripts/baseline_comparison_wulver/launch_remaining_datasets.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"

if [[ -z "${BASELINE_OUTPUT_TAG:-}" ]]; then
  echo "ERROR: export BASELINE_OUTPUT_TAG to match your running 6-dataset job (see outputs/baseline_comparison/)." >&2
  exit 1
fi

export REPO_ROOT
export BASELINE_CONDA_ENV="${BASELINE_CONDA_ENV:-feedback-weighted-maximization}"

SBATCH_FILE="$REPO_ROOT/slurm/baseline_comparison_remaining.sbatch"
echo "Submitting remaining 3 datasets with OUT_TAG=$BASELINE_OUTPUT_TAG"
JOB_OUTPUT=$(sbatch --export=ALL,REPO_ROOT,BASELINE_OUTPUT_TAG,BASELINE_CONDA_ENV "$SBATCH_FILE" 2>&1)
echo "$JOB_OUTPUT"

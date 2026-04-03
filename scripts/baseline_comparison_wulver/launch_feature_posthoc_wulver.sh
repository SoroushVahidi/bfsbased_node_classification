#!/usr/bin/env bash
# Launch feature-first / post-hoc baseline suite (Applied Intelligence competitors).
# One Slurm array task per dataset (9 tasks). CPU-only; 64G RAM for large graphs.
#
# Usage:
#   ./scripts/baseline_comparison_wulver/launch_feature_posthoc_wulver.sh
# Optional:
#   export REPO_ROOT=/mmfs1/home/sv96/bfsbased_node_classification_git
#   export FEATURE_POSTHOC_OUTPUT_TAG=my_tag_20260403
#   export BASELINE_CONDA_ENV=feedback-weighted-maximization

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"

export FEATURE_POSTHOC_OUTPUT_TAG="${FEATURE_POSTHOC_OUTPUT_TAG:-wulver_feature_posthoc_applied_intel_$(date +%Y%m%d_%H%M%S)}"
export REPO_ROOT
export BASELINE_CONDA_ENV="${BASELINE_CONDA_ENV:-feedback-weighted-maximization}"

mkdir -p "$REPO_ROOT/slurm" "$REPO_ROOT/outputs/baseline_comparison/$FEATURE_POSTHOC_OUTPUT_TAG/per_run"

echo "=== Feature-first / post-hoc baseline launch ==="
echo "REPO_ROOT=$REPO_ROOT"
echo "FEATURE_POSTHOC_OUTPUT_TAG=$FEATURE_POSTHOC_OUTPUT_TAG"
echo "Per-dataset JSONL: outputs/baseline_comparison/$FEATURE_POSTHOC_OUTPUT_TAG/per_run/runs_<dataset>.jsonl"
echo "MLP_ONLY + externals: sgc, correct_and_smooth, clp, graphsage, appnp (--external-baselines-only)"
echo "Datasets: 9 array tasks (cora … squirrel), splits 0-9"
echo ""

SBATCH_FILE="$REPO_ROOT/slurm/baseline_comparison_feature_posthoc_array.sbatch"
JOB_OUTPUT=$(sbatch --export=ALL,REPO_ROOT,FEATURE_POSTHOC_OUTPUT_TAG,BASELINE_CONDA_ENV "$SBATCH_FILE" 2>&1)
echo "$JOB_OUTPUT"

JOB_ID=$(echo "$JOB_OUTPUT" | awk '/Submitted batch job/ {print $4}')
if [[ -n "${JOB_ID:-}" ]]; then
  echo ""
  echo "JOB_ID=$JOB_ID"
  echo "Monitor: squeue -u \"\$USER\" -j $JOB_ID"
  echo "After completion:"
  echo "  python3 scripts/baseline_comparison_wulver/aggregate_and_report.py \\"
  echo "    --glob \"outputs/baseline_comparison/$FEATURE_POSTHOC_OUTPUT_TAG/per_run/runs_*.jsonl\" \\"
  echo "    --output-dir \"outputs/baseline_comparison/$FEATURE_POSTHOC_OUTPUT_TAG\""
fi

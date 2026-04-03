#!/usr/bin/env bash
# Label-budget sweep: 5 datasets × (10%, 20%, 40%, full train) × 10 splits.
# Subsamples only from official train indices; val/test unchanged.
#
# Usage:
#   ./scripts/baseline_comparison_wulver/launch_label_budget_wulver.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"

export LABEL_BUDGET_OUTPUT_TAG="${LABEL_BUDGET_OUTPUT_TAG:-wulver_label_budget_applied_intel_$(date +%Y%m%d_%H%M%S)}"
export REPO_ROOT
export BASELINE_CONDA_ENV="${BASELINE_CONDA_ENV:-feedback-weighted-maximization}"

mkdir -p "$REPO_ROOT/outputs/label_budget/$LABEL_BUDGET_OUTPUT_TAG/per_run" "$REPO_ROOT/slurm"

echo "=== Label-budget sweep (Applied Intelligence) ==="
echo "LABEL_BUDGET_OUTPUT_TAG=$LABEL_BUDGET_OUTPUT_TAG"
echo "Shards: outputs/label_budget/$LABEL_BUDGET_OUTPUT_TAG/per_run/runs_<dataset>_p{10,20,40,100}.jsonl"
echo "Datasets: cora citeseer pubmed chameleon texas | fractions: 0.1 0.2 0.4 1.0 | splits 0-9"
echo "Methods: MLP_ONLY, FINAL_V3, CLP, CORRECT_AND_SMOOTH, SGC, GRAPHSAGE, APPNP"
echo ""

SBATCH_FILE="$REPO_ROOT/slurm/label_budget_sweep_array.sbatch"
JOB_OUTPUT=$(sbatch --export=ALL,REPO_ROOT,LABEL_BUDGET_OUTPUT_TAG,BASELINE_CONDA_ENV "$SBATCH_FILE" 2>&1)
echo "$JOB_OUTPUT"

JOB_ID=$(echo "$JOB_OUTPUT" | awk '/Submitted batch job/ {print $4}')
if [[ -n "${JOB_ID:-}" ]]; then
  echo ""
  echo "JOB_ID=$JOB_ID"
  echo "Monitor: squeue -u \"\$USER\" -j $JOB_ID"
  echo "After completion:"
  echo "  python3 scripts/baseline_comparison_wulver/aggregate_label_budget.py \\"
  echo "    --glob \"outputs/label_budget/$LABEL_BUDGET_OUTPUT_TAG/per_run/runs_*.jsonl\" \\"
  echo "    --output-dir \"outputs/label_budget/$LABEL_BUDGET_OUTPUT_TAG\""
fi

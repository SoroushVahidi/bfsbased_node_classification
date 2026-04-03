#!/usr/bin/env bash
# Graph edge-noise robustness (equal-count rewire). See graph_corruption.py for protocol.
#
# Usage:
#   ./scripts/baseline_comparison_wulver/launch_graph_noise_wulver.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"

export GRAPH_NOISE_OUTPUT_TAG="${GRAPH_NOISE_OUTPUT_TAG:-wulver_graph_noise_robustness_applied_intel_$(date +%Y%m%d_%H%M%S)}"
export REPO_ROOT
export BASELINE_CONDA_ENV="${BASELINE_CONDA_ENV:-feedback-weighted-maximization}"

mkdir -p "$REPO_ROOT/outputs/graph_noise_robustness/$GRAPH_NOISE_OUTPUT_TAG/per_run" "$REPO_ROOT/slurm"

echo "=== Graph noise / edge corruption robustness ==="
echo "GRAPH_NOISE_OUTPUT_TAG=$GRAPH_NOISE_OUTPUT_TAG"
echo "Shards: outputs/graph_noise_robustness/$GRAPH_NOISE_OUTPUT_TAG/per_run/runs_<ds>_n{0,10,20,30}.jsonl"
echo "Datasets: cora citeseer pubmed chameleon texas wisconsin | noise: 0.0 0.1 0.2 0.3 | splits 0-9"
echo "Methods: MLP_ONLY, FINAL_V3, CLP, CORRECT_AND_SMOOTH, SGC, GRAPHSAGE, APPNP"
echo ""

SBATCH_FILE="$REPO_ROOT/slurm/graph_noise_robustness_array.sbatch"
JOB_OUTPUT=$(sbatch --export=ALL,REPO_ROOT,GRAPH_NOISE_OUTPUT_TAG,BASELINE_CONDA_ENV "$SBATCH_FILE" 2>&1)
echo "$JOB_OUTPUT"

JOB_ID=$(echo "$JOB_OUTPUT" | awk '/Submitted batch job/ {print $4}')
if [[ -n "${JOB_ID:-}" ]]; then
  echo ""
  echo "JOB_ID=$JOB_ID"
  echo "Monitor: squeue -u \"\$USER\" -j $JOB_ID"
  echo "After completion:"
  echo "  python3 scripts/baseline_comparison_wulver/aggregate_graph_noise.py \\"
  echo "    --glob \"outputs/graph_noise_robustness/$GRAPH_NOISE_OUTPUT_TAG/per_run/runs_*.jsonl\" \\"
  echo "    --output-dir \"outputs/graph_noise_robustness/$GRAPH_NOISE_OUTPUT_TAG\""
fi

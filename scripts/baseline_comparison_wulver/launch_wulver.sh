#!/usr/bin/env bash
# Launch Wulver array job (one dataset per task). Usage:
#   ./scripts/baseline_comparison_wulver/launch_wulver.sh
# Optional:
#   export REPO_ROOT=/mmfs1/home/sv96/bfsbased_node_classification_git
#   export BASELINE_OUTPUT_TAG=my_cmp_20260403

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"

export BASELINE_OUTPUT_TAG="${BASELINE_OUTPUT_TAG:-wulver_baseline_cmp_$(date +%Y%m%d_%H%M%S)}"
export REPO_ROOT
# Conda env on Wulver compute nodes must provide torch + torch_geometric (base Anaconda does not).
export BASELINE_CONDA_ENV="${BASELINE_CONDA_ENV:-feedback-weighted-maximization}"

mkdir -p "$REPO_ROOT/slurm" "$REPO_ROOT/outputs/baseline_comparison/$BASELINE_OUTPUT_TAG"

echo "=== Dry audit ==="
echo "REPO_ROOT=$REPO_ROOT"
echo "BASELINE_OUTPUT_TAG=$BASELINE_OUTPUT_TAG"
echo "Per-run JSONL: $REPO_ROOT/outputs/baseline_comparison/$BASELINE_OUTPUT_TAG/per_run/runs_<dataset>.jsonl"
echo "Methods: MLP_ONLY, BASELINE_SGC_V1, V2_MULTIBRANCH, FINAL_V3 + H2GCN, GCNII, GeomGCN, LINKX, ACMII_GCN_PLUSPLUS"
echo "Datasets (6 array tasks): cora citeseer pubmed chameleon texas wisconsin"
echo "Splits: 0-9"
echo ""

if [[ ! -d "$REPO_ROOT/data/splits" ]]; then
  echo "WARNING: data/splits missing — jobs will skip or fail." >&2
fi

SBATCH_FILE="$REPO_ROOT/slurm/baseline_comparison_array.sbatch"
echo "Submitting: $SBATCH_FILE"
# Propagate tag + repo; Slurm account/qos/partition are set in the sbatch file.
JOB_OUTPUT=$(sbatch --export=ALL,REPO_ROOT,BASELINE_OUTPUT_TAG,BASELINE_CONDA_ENV "$SBATCH_FILE" 2>&1)
echo "$JOB_OUTPUT"

JOB_ID=$(echo "$JOB_OUTPUT" | awk '/Submitted batch job/ {print $4}')
if [[ -n "${JOB_ID:-}" ]]; then
  echo ""
  echo "JOB_ID=$JOB_ID"
  echo "Monitor: squeue -u \"\$USER\" -j $JOB_ID"
  echo "After completion, merge + report:"
  echo "  python3 scripts/baseline_comparison_wulver/aggregate_and_report.py \\"
  echo "    --glob \"outputs/baseline_comparison/$BASELINE_OUTPUT_TAG/per_run/runs_*.jsonl\" \\"
  echo "    --output-dir \"outputs/baseline_comparison/$BASELINE_OUTPUT_TAG\""
  echo "Slurm logs: $REPO_ROOT/slurm/baseline_comparison_*_${JOB_ID}_*.out"
  echo "sacct: sacct -j $JOB_ID --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode"
fi

#!/bin/bash
#SBATCH --job-name=djgnn_smoke
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --account=ikoutis
#SBATCH --output=slurm/djgnn_benchmark_%x_%j.out
#SBATCH --error=slurm/djgnn_benchmark_%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# Quick path check: small graph, few epochs (not for reporting).

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO_ROOT"

CONDA_SH="/apps/easybuild/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh"
if [[ -f "$CONDA_SH" ]]; then
  # shellcheck source=/dev/null
  source "$CONDA_SH"
fi
CONDA_ENV="${DJGNN_CONDA_ENV:-${BASELINE_CONDA_ENV:-feedback-weighted-maximization}}"
conda activate "$CONDA_ENV"

CODE_DIR="$REPO_ROOT/code/bfsbased_node_classification"
OUT_TAG="${DJGNN_OUTPUT_TAG:-djgnn_smoke_${SLURM_JOB_ID:-local}}"

python3 "$CODE_DIR/run_djgnn_benchmark.py" \
  --datasets texas \
  --splits 0 \
  --split-dir "$REPO_ROOT/data/splits" \
  --output-tag "$OUT_TAG" \
  --epochs 5 \
  --device cpu

echo "[done] smoke tag=$OUT_TAG"

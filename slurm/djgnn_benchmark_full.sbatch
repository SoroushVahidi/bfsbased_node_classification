#!/bin/bash
#SBATCH --job-name=djgnn_full
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --account=ikoutis
#SBATCH --output=slurm/djgnn_benchmark_%x_%j.out
#SBATCH --error=slurm/djgnn_benchmark_%x_%j.err
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# Full benchmark: all canonical datasets × 10 splits in one process (long walltime).
# For GPU nodes, set DJGNN_DEVICE=cuda:0 and uncomment:
# #SBATCH --gres=gpu:1

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
SPLIT_DIR="$REPO_ROOT/data/splits"
OUT_TAG="${DJGNN_OUTPUT_TAG:-djgnn_wulver_full_$(date +%Y%m%d_%H%M%S)}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

DEVICE="${DJGNN_DEVICE:-}"
if [[ -z "$DEVICE" ]]; then
  if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    DEVICE="cuda:0"
  else
    DEVICE="cpu"
  fi
fi

echo "===== djgnn_benchmark_full OUT_TAG=$OUT_TAG DEVICE=$DEVICE ====="

python3 "$CODE_DIR/run_djgnn_benchmark.py" \
  --datasets cora citeseer pubmed chameleon texas wisconsin actor cornell squirrel \
  --splits 0 1 2 3 4 5 6 7 8 9 \
  --split-dir "$SPLIT_DIR" \
  --output-tag "$OUT_TAG" \
  --device "$DEVICE"

echo "[done] tag=$OUT_TAG"

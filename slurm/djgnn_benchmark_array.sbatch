#!/bin/bash
#SBATCH --job-name=djgnn_arr
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --account=ikoutis
#SBATCH --output=slurm/djgnn_benchmark_%x_%A_%a.out
#SBATCH --error=slurm/djgnn_benchmark_%x_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# One dataset per array task (9 tasks): canonical six + actor, cornell, squirrel
#SBATCH --array=0-8
#
# Dense DJ-GNN uses full N×N adjacency; large graphs (e.g. PubMed) need RAM and/or GPU.
# Uncomment on GPU partitions if available:
# #SBATCH --gres=gpu:1
# #SBATCH --partition=gpu

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
python3 -c "import torch; import torch_geometric" || {
  echo "ERROR: need torch and torch_geometric in conda env '$CONDA_ENV'" >&2
  exit 1
}

CODE_DIR="$REPO_ROOT/code/bfsbased_node_classification"
SPLIT_DIR="$REPO_ROOT/data/splits"
BASE_TAG="${DJGNN_OUTPUT_TAG:-djgnn_wulver_$(date +%Y%m%d_%H%M%S)}"

DATASETS=(cora citeseer pubmed chameleon texas wisconsin actor cornell squirrel)
DS="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
# Unique tag per array task so JSONL/CSV are not truncated by concurrent jobs.
OUT_TAG="${BASE_TAG}__${DS}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

DEVICE="${DJGNN_DEVICE:-}"
if [[ -z "$DEVICE" ]]; then
  if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    DEVICE="cuda:0"
  else
    DEVICE="cpu"
  fi
fi

echo "===== djgnn_benchmark_array ====="
echo "REPO_ROOT=$REPO_ROOT dataset=$DS OUT_TAG=$OUT_TAG DEVICE=$DEVICE"
echo "================================="

python3 "$CODE_DIR/run_djgnn_benchmark.py" \
  --datasets "$DS" \
  --splits 0 1 2 3 4 5 6 7 8 9 \
  --split-dir "$SPLIT_DIR" \
  --output-tag "$OUT_TAG" \
  --device "$DEVICE"

echo "[done] dataset=$DS tag=$OUT_TAG"

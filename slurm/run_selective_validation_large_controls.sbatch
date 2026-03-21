#!/bin/bash
#SBATCH --job-name=sv_controls
#SBATCH --output=/mmfs1/home/sv96/node_classification_manuscript/logs/slurm/%x_%j.out
#SBATCH --error=/mmfs1/home/sv96/node_classification_manuscript/logs/slurm/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

ROOT="/mmfs1/home/sv96/node_classification_manuscript"
CODE_DIR="$ROOT/code/bfsbased_node_classification"
SPLIT_DIR="$ROOT/data/splits"

source /apps/easybuild/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate feedback-weighted-maximization

cd "$ROOT"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python3 "$CODE_DIR/manuscript_runner.py" \
  --datasets pubmed squirrel \
  --splits 0 1 2 3 4 5 6 7 8 9 \
  --repeats 3 \
  --methods mlp_only prop_only selective_graph_correction \
  --split-dir "$SPLIT_DIR" \
  --output-tag "manuscript_final_validation_job2"


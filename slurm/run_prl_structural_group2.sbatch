#!/bin/bash
#SBATCH --job-name=prl_struct_g2
#SBATCH --output=/mmfs1/home/sv96/node_classification_manuscript/logs/slurm/%x_%j.out
#SBATCH --error=/mmfs1/home/sv96/node_classification_manuscript/logs/slurm/%x_%j.err
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

ROOT="/mmfs1/home/sv96/node_classification_manuscript"
CODE_DIR="$ROOT/code/bfsbased_node_classification"
SPLIT_DIR="$ROOT/data/splits"
OUTPUT_TAG="prl_structural_group2_r1"

source /apps/easybuild/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate feedback-weighted-maximization

mkdir -p "$ROOT/logs" "$ROOT/logs/slurm" "$ROOT/logs/prl_resubmission" "$ROOT/reports/prl_resubmission" "$ROOT/tables/prl_resubmission"
cd "$ROOT"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python3 "$CODE_DIR/prl_resubmission_runner.py" \
  --datasets chameleon wisconsin \
  --splits 0 1 2 3 4 5 6 7 8 9 \
  --repeats 1 \
  --methods mlp_only prop_only gcn appnp selective_graph_correction selective_graph_correction_structural sgcs_margin_gate_only sgcs_no_far sgcs_far_only sgcs_no_gate \
  --split-dir "$SPLIT_DIR" \
  --output-tag "$OUTPUT_TAG"

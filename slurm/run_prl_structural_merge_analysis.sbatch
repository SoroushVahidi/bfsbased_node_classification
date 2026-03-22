#!/bin/bash
#SBATCH --job-name=prl_struct_merge
#SBATCH --output=/mmfs1/home/sv96/node_classification_manuscript/logs/slurm/%x_%j.out
#SBATCH --error=/mmfs1/home/sv96/node_classification_manuscript/logs/slurm/%x_%j.err
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

ROOT="/mmfs1/home/sv96/node_classification_manuscript"
CODE_DIR="$ROOT/code/bfsbased_node_classification"

source /apps/easybuild/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate feedback-weighted-maximization

cd "$ROOT"
python3 "$CODE_DIR/merge_prl_resubmission_runs.py" \
  --inputs \
    "$ROOT/logs/prl_resubmission/comparison_runs_prl_structural_group1_r1.jsonl" \
    "$ROOT/logs/prl_resubmission/comparison_runs_prl_structural_group2_r1.jsonl" \
  --output-tag prl_structural_merged_r1

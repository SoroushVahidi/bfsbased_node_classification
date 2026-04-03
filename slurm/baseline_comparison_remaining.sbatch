#!/bin/bash
#SBATCH --job-name=bl_cmp_rem
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --account=ikoutis
#SBATCH --output=slurm/baseline_comparison_rem_%x_%A_%a.out
#SBATCH --error=slurm/baseline_comparison_rem_%x_%A_%a.err
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
# Three datasets not in baseline_comparison_array.sbatch (0–5):
#   actor, cornell, squirrel  (same protocol: 10 splits, all methods)
#SBATCH --array=0-2

# Slurm opens slurm/ logs before this script; ensure slurm/ exists (launch_*.sh or mkdir -p slurm).

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO_ROOT"
if [[ ! -f "$REPO_ROOT/code/bfsbased_node_classification/baseline_comparison_suite.py" ]]; then
  echo "ERROR: REPO_ROOT=$REPO_ROOT is not this repository (missing baseline_comparison_suite.py)." >&2
  exit 1
fi

CONDA_SH="/apps/easybuild/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh"
if [[ -f "$CONDA_SH" ]]; then
  # shellcheck source=/dev/null
  source "$CONDA_SH"
fi
CONDA_ENV="${BASELINE_CONDA_ENV:-feedback-weighted-maximization}"
conda activate "$CONDA_ENV"
python3 -c "import torch; import torch_geometric" || {
  echo "ERROR: need torch and torch_geometric." >&2
  exit 1
}

CODE_DIR="$REPO_ROOT/code/bfsbased_node_classification"
SPLIT_DIR="$REPO_ROOT/data/splits"
# MUST match the tag used by the main 6-dataset array so per_run/ merges cleanly.
OUT_TAG="${BASELINE_OUTPUT_TAG:?Set BASELINE_OUTPUT_TAG to the same value as the running bl_cmp job}"

DATASETS=(actor cornell squirrel)
DS="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

OUT_BASE="$REPO_ROOT/outputs/baseline_comparison/${OUT_TAG}"
PER_RUN="$OUT_BASE/per_run"
mkdir -p "$PER_RUN" "$REPO_ROOT/slurm"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "===== baseline_comparison_remaining ($DS) ====="
echo "OUT_TAG=$OUT_TAG (must match main array)"
echo "dataset=$DS  array_id=${SLURM_ARRAY_TASK_ID:-}"
echo "jsonl=$PER_RUN/runs_${DS}.jsonl"
echo "================================================"

META="$PER_RUN/job_${DS}_meta.json"
python3 - <<PY
import json, os, socket, subprocess, sys
m = {
  "hostname": socket.gethostname(),
  "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
  "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
  "dataset": "${DS}",
  "output_tag": "${OUT_TAG}",
  "batch": "remaining_actor_cornell_squirrel",
  "repo_root": r"${REPO_ROOT}",
  "python": sys.executable,
}
try:
    m["git_rev"] = subprocess.check_output(["git", "-C", r"${REPO_ROOT}", "rev-parse", "HEAD"], text=True).strip()
except Exception:
    m["git_rev"] = None
open(r"${META}", "w").write(json.dumps(m, indent=2))
PY

python3 "$CODE_DIR/baseline_comparison_suite.py" \
  --datasets "$DS" \
  --splits 0 1 2 3 4 5 6 7 8 9 \
  --split-dir "$SPLIT_DIR" \
  --output-tag "$OUT_TAG" \
  --output-dir "$PER_RUN" \
  --output-jsonl "$PER_RUN/runs_${DS}.jsonl" \
  --gate heuristic \
  --baseline-max-epochs 500 \
  --baseline-patience 100

echo "[done] dataset=$DS tag=$OUT_TAG"

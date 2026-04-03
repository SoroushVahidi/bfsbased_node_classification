#!/bin/bash
#SBATCH --job-name=bl_cmp
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --account=ikoutis
#SBATCH --output=slurm/baseline_comparison_%x_%A_%a.out
#SBATCH --error=slurm/baseline_comparison_%x_%A_%a.err
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
# One dataset per array task (6 tasks): 0=cora ... 5=wisconsin
#SBATCH --array=0-5

set -euo pipefail

# Override with: export REPO_ROOT=/path/to/bfsbased_node_classification_git
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO_ROOT"

# Compute nodes use bare Anaconda python3 — it does NOT include PyTorch.
# Set BASELINE_CONDA_ENV to any env that has torch + torch_geometric (and this repo's deps).
CONDA_SH="/apps/easybuild/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh"
if [[ -f "$CONDA_SH" ]]; then
  # shellcheck source=/dev/null
  source "$CONDA_SH"
fi
CONDA_ENV="${BASELINE_CONDA_ENV:-feedback-weighted-maximization}"
conda activate "$CONDA_ENV"
# torch_geometric may live in ~/.local while torch lives in conda; do not set
# PYTHONNOUSERSITE=1 unless both stacks are installed inside the same env.
python3 -c "import torch; import torch_geometric" || {
  echo "ERROR: need torch and torch_geometric importable (conda env '$CONDA_ENV' + user site)." >&2
  echo "Fix: pip install torch_geometric in the conda env, or pip install -r requirements.txt" >&2
  exit 1
}

CODE_DIR="$REPO_ROOT/code/bfsbased_node_classification"
SPLIT_DIR="$REPO_ROOT/data/splits"
OUT_TAG="${BASELINE_OUTPUT_TAG:-wulver_baseline_cmp_$(date +%Y%m%d_%H%M%S)}"

DATASETS=(cora citeseer pubmed chameleon texas wisconsin)
DS="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

OUT_BASE="$REPO_ROOT/outputs/baseline_comparison/${OUT_TAG}"
PER_RUN="$OUT_BASE/per_run"
SUMM="$OUT_BASE/summaries"
SLURM_LOG="$REPO_ROOT/slurm"
mkdir -p "$PER_RUN" "$SUMM" "$SLURM_LOG"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "===== baseline_comparison_array ====="
echo "hostname: $(hostname)"
echo "REPO_ROOT: $REPO_ROOT"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID:-}"
echo "dataset: $DS"
echo "OUT_TAG: $OUT_TAG"
echo "split_dir: $SPLIT_DIR"
echo "jsonl: $PER_RUN/runs_${DS}.jsonl"
echo "python: $(command -v python3)"
python3 -c "import sys; print(sys.version)" || true
echo "====================================="

META="$PER_RUN/job_${DS}_meta.json"
python3 - <<PY
import json, os, socket, subprocess, sys
m = {
  "hostname": socket.gethostname(),
  "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
  "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
  "dataset": "${DS}",
  "output_tag": "${OUT_TAG}",
  "repo_root": r"${REPO_ROOT}",
  "python": sys.executable,
}
try:
    m["git_rev"] = subprocess.check_output(["git", "-C", r"${REPO_ROOT}", "rev-parse", "HEAD"], text=True).strip()
except Exception:
    m["git_rev"] = None
print(json.dumps(m, indent=2))
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

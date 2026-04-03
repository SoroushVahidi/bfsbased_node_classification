#!/bin/bash
#SBATCH --job-name=lb_budget
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --account=ikoutis
#SBATCH --output=slurm/label_budget_%x_%A_%a.out
#SBATCH --error=slurm/label_budget_%x_%A_%a.err
# 5 datasets × 4 train fractions = 20 tasks. Each: 10 splits × (MLP + FINAL_V3 + 5 baselines).
#SBATCH --time=56:00:00
#SBATCH --cpus-per-task=8
# 72G: PubMed + multiple heavy baselines (C&S / CLP grids) per split.
#SBATCH --mem=72G
#SBATCH --array=0-19

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO_ROOT"

CONDA_SH="/apps/easybuild/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh"
if [[ -f "$CONDA_SH" ]]; then
  # shellcheck source=/dev/null
  source "$CONDA_SH"
fi
CONDA_ENV="${BASELINE_CONDA_ENV:-feedback-weighted-maximization}"
conda activate "$CONDA_ENV"
python3 -c "import torch; import torch_geometric" || {
  echo "ERROR: need torch and torch_geometric in conda env '$CONDA_ENV'" >&2
  exit 1
}

CODE_DIR="$REPO_ROOT/code/bfsbased_node_classification"
SPLIT_DIR="$REPO_ROOT/data/splits"
OUT_TAG="${LABEL_BUDGET_OUTPUT_TAG:?Set LABEL_BUDGET_OUTPUT_TAG (use launch_label_budget_wulver.sh)}"

DATASETS=(cora citeseer pubmed chameleon texas)
declare -a FRACS=(0.1 0.2 0.4 1.0)
declare -a SUFFIX=(p10 p20 p40 p100)
NFR=4
TID="${SLURM_ARRAY_TASK_ID:?}"
DS_IDX=$((TID / NFR))
FR_IDX=$((TID % NFR))
DS="${DATASETS[$DS_IDX]}"
FR="${FRACS[$FR_IDX]}"
SUF="${SUFFIX[$FR_IDX]}"

OUT_BASE="$REPO_ROOT/outputs/label_budget/${OUT_TAG}/per_run"
mkdir -p "$OUT_BASE" "$REPO_ROOT/slurm"
JSONL="$OUT_BASE/runs_${DS}_${SUF}.jsonl"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "===== label_budget_sweep ====="
echo "REPO_ROOT=$REPO_ROOT dataset=$DS train_fraction=$FR shard=$SUF"
echo "OUT_TAG=$OUT_TAG"
echo "jsonl=$JSONL"
echo "=============================="

python3 - <<PY
import json, os, socket, subprocess, sys
m = {
  "hostname": socket.gethostname(),
  "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
  "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
  "dataset": "${DS}",
  "train_fraction": float("${FR}"),
  "output_tag": "${OUT_TAG}",
  "jsonl": r"${JSONL}",
  "repo_root": r"${REPO_ROOT}",
  "python": sys.executable,
}
try:
    m["git_rev"] = subprocess.check_output(["git", "-C", r"${REPO_ROOT}", "rev-parse", "HEAD"], text=True).strip()
except Exception:
    m["git_rev"] = None
open(r"${OUT_BASE}/job_${DS}_${SUF}_meta.json", "w").write(json.dumps(m, indent=2))
PY

python3 "$CODE_DIR/label_budget_sweep.py" \
  --dataset "$DS" \
  --train-fraction "$FR" \
  --splits 0 1 2 3 4 5 6 7 8 9 \
  --split-dir "$SPLIT_DIR" \
  --output-tag "$OUT_TAG" \
  --output-jsonl "$JSONL" \
  --baseline-max-epochs 500 \
  --baseline-patience 100 \
  --gate heuristic

echo "[done] dataset=$DS fraction=$FR tag=$OUT_TAG"

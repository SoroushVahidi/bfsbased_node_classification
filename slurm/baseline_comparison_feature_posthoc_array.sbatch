#!/bin/bash
#SBATCH --job-name=bl_featph
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --account=ikoutis
#SBATCH --output=slurm/baseline_comparison_featph_%x_%A_%a.out
#SBATCH --error=slurm/baseline_comparison_featph_%x_%A_%a.err
# 48h: MLP + 5 baselines (incl. C&S / CLP grids) x 10 splits per dataset; PubMed/Squirrel are large.
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
# 64G: full-batch GraphSAGE / sparse propagation on PubMed & Squirrel (safer than 48G default).
#SBATCH --mem=64G
# Nine datasets (same list as DJ-GNN / extended Wulver comparison).
#SBATCH --array=0-8

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
OUT_TAG="${FEATURE_POSTHOC_OUTPUT_TAG:?Set FEATURE_POSTHOC_OUTPUT_TAG before sbatch (use launch_feature_posthoc_wulver.sh)}"

DATASETS=(cora citeseer pubmed chameleon texas wisconsin actor cornell squirrel)
DS="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

OUT_BASE="$REPO_ROOT/outputs/baseline_comparison/${OUT_TAG}"
PER_RUN="$OUT_BASE/per_run"
mkdir -p "$PER_RUN" "$REPO_ROOT/slurm"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "===== baseline_comparison_feature_posthoc ====="
echo "REPO_ROOT=$REPO_ROOT dataset=$DS OUT_TAG=$OUT_TAG"
echo "Methods: MLP_ONLY + sgc correct_and_smooth clp graphsage appnp (external-baselines-only)"
echo "=============================================="

META="$PER_RUN/job_${DS}_meta.json"
python3 - <<PY
import json, os, socket, subprocess, sys
m = {
  "hostname": socket.gethostname(),
  "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
  "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
  "dataset": "${DS}",
  "output_tag": "${OUT_TAG}",
  "batch": "feature_posthoc_applied_intel",
  "external_baselines": ["sgc", "correct_and_smooth", "clp", "graphsage", "appnp"],
  "external_baselines_only": True,
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
  --baseline-patience 100 \
  --external-baselines-only \
  --external-baselines sgc correct_and_smooth clp graphsage appnp

echo "[done] dataset=$DS tag=$OUT_TAG"

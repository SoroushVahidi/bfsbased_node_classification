#!/bin/bash
#SBATCH --job-name=sv_main_r3
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

OUTDIR="$ROOT/logs"
mkdir -p "$OUTDIR"

HOST="$(hostname)"
DT="$(date -Iseconds)"
PITCH_DIR="$(pwd)"

output_tag="manuscript_final_validation_main"
final_suffix="final_validation_main"

# Provenance note (matches the exact GitHub commit blob hashes we verified locally).
PROV_BFS_BLOB="622036cab9ba33b34190ba88ef2ac88c52f5972e (bfsbased-full-investigate-homophil.py)"
PROV_CMP_BLOB="756447e7117a0b017a5f73e496ba05e28fd41272 (comparison_runner.py)"
PROV_GITHUB_REF="4252d292c84ae5d0a5db55d6ae94a01a96009357"

trap 'rc=$?; echo "[job_error] rc=$rc"; echo "[job_error] failed_cmd=${BASH_COMMAND}"; exit $rc' ERR

echo "===== JOB SELF REPORT ====="
echo "hostname: $HOST"
echo "date: $DT"
echo "working_directory: $PITCH_DIR"
echo "python_executable: $(which python3 || true)"
echo "python_version: $(python3 -c 'import sys; print(sys.version.replace(\"\\n\",\" \"))' 2>/dev/null || true)"
echo "conda_default_env: ${CONDA_DEFAULT_ENV:-}"
echo "conda_prefix: ${CONDA_PREFIX:-}"
echo "git_repo_dir: $CODE_DIR"
git -C "$CODE_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || true
git -C "$CODE_DIR" rev-parse HEAD 2>/dev/null || true
echo "provenance_github_ref: $PROV_GITHUB_REF"
echo "provenance_blobs: $PROV_BFS_BLOB"
echo "provenance_blobs: $PROV_CMP_BLOB"
echo "datasets: cora citeseer chameleon texas actor cornell wisconsin"
echo "splits: 0 1 2 3 4 5 6 7 8 9"
echo "repeats: 3"
echo "methods: mlp_only prop_only selective_graph_correction"
echo "output_tag: $output_tag"
echo "split_root: $SPLIT_DIR"
echo "data_root: $ROOT/data"
echo "logs_root: $OUTDIR"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NAME: ${SLURM_JOB_NAME:-}"
echo "============================="

cd "$ROOT"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# Resume-safe: if the main raw records already exist, do not overwrite.
expected_runs="$OUTDIR/comparison_runs_${output_tag}.jsonl"
if [ -f "$expected_runs" ]; then
  echo "[resume_safe] Found existing $expected_runs"
  echo "[resume_safe] Skipping JOB 1 to avoid overwriting important results."
  exit 0
fi

echo "[run] python3 $CODE_DIR/manuscript_runner.py (output-tag: $output_tag)"

python3 "$CODE_DIR/manuscript_runner.py" \
  --datasets cora citeseer chameleon texas actor cornell wisconsin \
  --splits 0 1 2 3 4 5 6 7 8 9 \
  --repeats 3 \
  --methods mlp_only prop_only selective_graph_correction \
  --split-dir "$SPLIT_DIR" \
  --output-tag "$output_tag"

# Normalize output filenames to the stable "manuscript_*_final_validation_main.*" naming.
intermediate_gain="$OUTDIR/gain_over_mlp_${output_tag}.csv"
intermediate_corr="$OUTDIR/correction_behavior_${output_tag}.csv"
intermediate_regime="$OUTDIR/regime_analysis_${output_tag}.csv"
intermediate_case="$OUTDIR/case_studies_${output_tag}.md"

dest_gain="$OUTDIR/manuscript_gain_over_mlp_${final_suffix}.csv"
dest_corr="$OUTDIR/manuscript_correction_behavior_${final_suffix}.csv"
dest_regime="$OUTDIR/manuscript_regime_analysis_${final_suffix}.csv"
dest_case="$OUTDIR/manuscript_case_studies_${final_suffix}.md"

rename_if_present () {
  src="$1"
  dst="$2"
  if [ ! -f "$src" ]; then
    echo "[rename] Missing intermediate file: $src"
    exit 1
  fi
  if [ -f "$dst" ]; then
    echo "[rename] Destination already exists (not overwriting): $dst"
    exit 1
  fi
  mv "$src" "$dst"
}

echo "[rename] Normalizing gain/correction/regime/case-study filenames..."
rename_if_present "$intermediate_gain" "$dest_gain"
rename_if_present "$intermediate_corr" "$dest_corr"
rename_if_present "$intermediate_regime" "$dest_regime"
rename_if_present "$intermediate_case" "$dest_case"

echo "[done] JOB 1 outputs normalized."


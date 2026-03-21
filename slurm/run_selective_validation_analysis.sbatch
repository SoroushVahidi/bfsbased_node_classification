#!/bin/bash
#SBATCH --job-name=sv_analysis
#SBATCH --output=/mmfs1/home/sv96/node_classification_manuscript/logs/slurm/%x_%j.out
#SBATCH --error=/mmfs1/home/sv96/node_classification_manuscript/logs/slurm/%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

ROOT="/mmfs1/home/sv96/node_classification_manuscript"
CODE_DIR="$ROOT/code/bfsbased_node_classification"

source /apps/easybuild/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate feedback-weighted-maximization

cd "$ROOT"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python3 - <<'PY'
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path("/mmfs1/home/sv96/node_classification_manuscript")
CODE_DIR = ROOT / "code" / "bfsbased_node_classification"
MR_PATH = CODE_DIR / "manuscript_runner.py"

spec = importlib.util.spec_from_file_location("manuscript_runner", str(MR_PATH))
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(mod)

# Merge raw records produced by job1 + job2.
runs_files = [
    ROOT / "logs" / "comparison_runs_manuscript_final_validation_job1.jsonl",
    ROOT / "logs" / "comparison_runs_manuscript_final_validation_job2.jsonl",
]

merged_runs_path = ROOT / "logs" / "comparison_runs_manuscript_final_validation.jsonl"
merged_runs_path.parent.mkdir(parents=True, exist_ok=True)

all_records = []
for rf in runs_files:
    if not rf.is_file():
        continue
    with open(rf, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            all_records.append(json.loads(line))

if not all_records:
    raise SystemExit("No records found for merge. Ensure job1/job2 completed successfully.")

# Write merged raw file (needed by case-study generation).
with open(merged_runs_path, "w", encoding="utf-8") as f:
    for rec in all_records:
        f.write(json.dumps(rec) + "\n")

# Reconstruct minimal dataset_meta from records.
dataset_meta = {}
for rec in all_records:
    if not rec.get("success", False):
        continue
    ds = rec["dataset"]
    if ds in dataset_meta:
        continue
    num_nodes = int(rec.get("num_nodes", 0) or 0)
    num_edges = int(rec.get("num_edges", 0) or 0)
    num_classes = int(rec.get("num_classes", 0) or 0)
    avg_degree = (num_edges / num_nodes) if num_nodes > 0 else float("nan")
    dataset_meta[ds] = {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_classes": num_classes,
        "avg_degree": float(avg_degree),
        "homophily": float("nan"),
    }

output_tag = "manuscript_final_validation"
logs_dir = ROOT / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)

summary_path = str(logs_dir / "comparison_summary_manuscript_final_validation.json")
fairness_path = str(logs_dir / "comparison_fairness_report_manuscript_final_validation.md")
gain_path = str(logs_dir / "manuscript_gain_over_mlp_final_validation.csv")
correction_path = str(logs_dir / "manuscript_correction_behavior_final_validation.csv")
regime_path = str(logs_dir / "manuscript_regime_analysis_final_validation.csv")

# Final case studies (stable expected name).
case_studies_path = str(logs_dir / "manuscript_case_studies_final_validation.md")

mod._generate_summary(
    all_records,
    dataset_meta,
    summary_path,
    fairness_path,
    gain_path,
    correction_path,
    regime_path,
    output_tag,
)

mod.generate_case_studies(
    str(merged_runs_path),
    case_studies_path,
    target_datasets=None,
)

# Print a recommendation/verdict to stdout by reading the just-written summary.
mod._print_manuscript_readiness({"summary_path": summary_path})

print("Analysis artifacts written under:", str(logs_dir))
PY


# Uncertainty-Gated Selective Graph Correction for Node Classification

This repository is the working code-and-artifact package for the current
Pattern Recognition Letters (PRL) manuscript line.

It contains both:

- the newer frozen PRL submission package centered on **FINAL_V3**
  (reliability-gated selective graph correction), and
- the broader manuscript / resubmission evidence package around
  **uncertainty-gated selective graph correction** (`UG-SGC`) plus the newer
  structural extension (`UG-SGC-S`)

Manuscript-oriented index: `README_PRL_MANUSCRIPT.md`  
Frozen table/figure refresh instructions: `scripts/README_REPRODUCE.md`

## Repository identity

Across all variants, the scientific story is intentionally narrow:

- start from a strong feature-only `MLP`
- identify uncertain nodes from model confidence or margin
- use graph information only as a **selective correction** mechanism
- keep the method interpretable by exposing score terms and correction behavior

This repository is **not** a claim of universal heterophily robustness or a
global replacement of the MLP with graph propagation.

## Main method / package lines

### 1. Frozen submission package: `FINAL_V3`

This is the current stable PRL submission-facing package in the remote repo.

- Stable import: `code/final_method_v3.py`
- Implementation: `code/bfsbased_node_classification/final_method_v3.py`
- Main 10-split driver: `code/bfsbased_node_classification/run_final_evaluation.py`
- Canonical tables: `tables/main_results_prl.md`, `tables/main_results_prl.csv`
- Main analysis: `reports/final_method_v3_analysis.md`

### 2. Original uncertainty-gated selective correction: `UG-SGC`

Implemented in:
- `code/bfsbased_node_classification/bfsbased-full-investigate-homophil.py`
- `code/bfsbased_node_classification/manuscript_runner.py`

Main ingredients:
- MLP log-probability term
- feature-prototype similarity
- optional feature-kNN support
- local graph-neighbor support
- compatibility support
- validation-selected uncertainty threshold

### 3. Structural selective correction extension: `UG-SGC-S`

Also implemented in:
- `code/bfsbased_node_classification/bfsbased-full-investigate-homophil.py`
- `code/bfsbased_node_classification/prl_resubmission_runner.py`

This extension keeps the same MLP-first selective-correction identity, but adds
a **far structural support** term based on confidence-weighted multi-hop
diffusion. Current repository evidence suggests it is **promising but mixed**,
not yet a universally stronger replacement.

## Repository layout

| Path | Purpose |
|------|---------|
| `code/final_method_v3.py` | Stable import path for the frozen `FINAL_V3` submission method |
| `code/bfsbased_node_classification/` | Core implementations, runners, baselines, and analysis scripts |
| `code/bfsbased_node_classification/experimental_archived/` | Legacy diagnostic and pre-final exploration scripts |
| `data/splits/` | Pre-bundled dataset splits (`.npz`) |
| `logs/` | Canonical JSONL/CSV run logs |
| `figures/` | PRL figures, including `prl_final_additions/` |
| `tables/` | Manuscript tables and supplementary PRL tables |
| `reports/` | Audits, method notes, result narratives, and run-status docs |
| `results_prl/` | Threshold/gate behavior package |
| `scripts/` | Artifact regeneration scripts |
| `slurm/` | Wulver batch templates |
| `AGENTS.md` | Automation and repository-maintenance notes |

## Most important documentation

- Manuscript artifact guide: `README_PRL_MANUSCRIPT.md`
- Frozen-method analysis: `reports/final_method_v3_analysis.md`
- PRL resubmission method spec: `reports/prl_resubmission/PRL_METHOD_SPEC.md`
- Repo readiness audit: `reports/prl_resubmission/REPO_READINESS_AUDIT.md`
- Current run/job status: `reports/prl_resubmission/RUN_STATUS.md`
- Structural-upgrade smoke note:
  `reports/prl_resubmission/STRUCTURAL_UPGRADE_SMOKE_NOTE.md`

## Quick commands

Refresh frozen PRL tables and figures from packaged CSVs:

```bash
bash scripts/run_all_prl_results.sh
```

Smoke test the original selective-correction line:

```bash
python3 code/bfsbased_node_classification/manuscript_runner.py \
  --split-dir data/splits --datasets cora --splits 0 --repeats 1 \
  --methods mlp_only selective_graph_correction --output-tag dev_test
```

Run the heavier PRL resubmission package:

```bash
python3 code/bfsbased_node_classification/prl_resubmission_runner.py \
  --split-dir data/splits --datasets cora --splits 0 --repeats 1 \
  --methods mlp_only gcn appnp selective_graph_correction --output-tag dev_test
```

Run the frozen `FINAL_V3` 10-split benchmark driver:

```bash
python3 code/bfsbased_node_classification/run_final_evaluation.py \
  --split-dir data/splits \
  --datasets cora citeseer pubmed chameleon texas wisconsin
```

## Reading guidance for external analysis

- Start with `README_PRL_MANUSCRIPT.md`.
- Treat `reports/prl_resubmission/RUN_STATUS.md` as the source of truth for
  which structural-upgrade jobs are complete versus still running.
- Treat `code/bfsbased_node_classification/experimental_archived/` and older
  diagnostic material as provenance, not the current paper story.

## License

This project is released under the [MIT License](LICENSE). A copy is also kept
under `code/bfsbased_node_classification/LICENSE` for convenience.

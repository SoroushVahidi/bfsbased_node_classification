# Uncertainty-Gated Selective Graph Correction for Node Classification

**This repository contains the evidence package for a Pattern Recognition Letters (PRL) submission.**

The work studies **feature-first, reliability-aware selective graph correction**: a strong MLP baseline plus **selective** graph-based correction that is applied mainly when the model is uncertain **and** local graph evidence appears reliable. The publication method is **FINAL_V3** (reliability-gated selective graph correction). Older gated variants and exploratory runners are **archived** under `code/bfsbased_node_classification/experimental_archived/` or labeled as baselines/ablations in reports.

Manuscript-oriented index: **[README_PRL_MANUSCRIPT.md](README_PRL_MANUSCRIPT.md)**  
Reproduce tables and figures from frozen CSVs: **[scripts/README_REPRODUCE.md](scripts/README_REPRODUCE.md)**

**Suggested GitHub “About” blurb:**

> PyTorch research code: feature-first, reliability-aware selective graph correction for node classification; PRL submission evidence package.

## License

This project is released under the [MIT License](LICENSE). A copy is also kept under `code/bfsbased_node_classification/LICENSE` for convenience when browsing only the implementation folder.

## Repository layout

| Path | Purpose |
|------|---------|
| `code/final_method_v3.py` | Stable import path for **FINAL_V3** (loads `code/bfsbased_node_classification/final_method_v3.py`) |
| `code/bfsbased_node_classification/` | Runners, models, dynamically loaded core study script |
| `code/bfsbased_node_classification/experimental_archived/` | Legacy diagnostic / v1–v2 exploration scripts (not the main method) |
| `data/splits/` | Pre-bundled dataset splits (`.npz`) |
| `logs/` | Canonical JSONL/CSV run logs |
| `figures/` | PRL figures (FINAL_V3 set + optional `prl_final_additions/`) |
| `tables/` | Merged tables (`main_results_prl.*` + optional `prl_final_additions/`) |
| `reports/` | Audits, **FINAL_V3** analysis, safety note, narrative story |
| `results_prl/` | Threshold-sensitivity and related packaged outputs |
| `scripts/` | Regeneration scripts; **`scripts/run_all_prl_results.sh`** refreshes main tables/figures |
| `prl_final_additions/` | Supplementary PRL tables/scripts bundle (mirrored under `tables/` / `figures/` by build scripts) |
| `slurm/` | Batch job templates |
| `AGENTS.md` | Tooling and validation notes for automation |

## Dependencies

- Python 3 with **PyTorch** (CPU is enough) for **training** runs
- **PyTorch Geometric**, NumPy, NetworkX, sortedcontainers, Matplotlib

Install example (CPU wheels):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric numpy networkx sortedcontainers matplotlib
```

Refreshing **tables and figures only** needs NumPy + Matplotlib (see `scripts/README_REPRODUCE.md`).

## Reproduce PRL tables and figures (lightweight)

```bash
bash scripts/run_all_prl_results.sh
```

## Quick start (smoke experiment, from repo root)

```bash
python3 code/bfsbased_node_classification/manuscript_runner.py \
  --split-dir data/splits --datasets cora --splits 0 --repeats 1 \
  --methods mlp_only selective_graph_correction --output-tag dev_test
```

PRL resubmission runner (GCN, APPNP, structural ablations — heavier):

```bash
python3 code/bfsbased_node_classification/prl_resubmission_runner.py \
  --split-dir data/splits --datasets cora --splits 0 --repeats 1 \
  --methods mlp_only gcn appnp selective_graph_correction --output-tag dev_test
```

FINAL_V3 comparison driver (same setting as `reports/final_method_v3_results.csv`):

```bash
python3 code/bfsbased_node_classification/run_final_evaluation.py \
  --split-dir data/splits \
  --datasets cora citeseer pubmed chameleon texas wisconsin \
  --splits 0 1 2 3 4
```

Check emitted JSONL under `logs/` for `"success": true` where applicable.

## Related writing

- **Earlier arXiv line** (heterophilic interpretability): [arxiv.org/abs/2512.22221](https://arxiv.org/abs/2512.22221) — see `code/bfsbased_node_classification/README.md` for the historical one-line summary.

## Citation

If you use this code, please cite the work you are building on (e.g. the PRL submission and/or the arXiv preprint above) using the citation your venue requires once DOI/page numbers are fixed.

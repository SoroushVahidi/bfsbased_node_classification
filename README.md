# Uncertainty-Gated Selective Graph Correction for Node Classification

Research code for **selective graph correction**: a strong MLP baseline plus uncertainty-gated propagation (SGC-style) on heterophilic and standard graph benchmarks. This repository supports the Pattern Recognition Letters manuscript *Uncertainty-Gated Selective Graph Correction for Node Classification* and related experiments.

**Suggested GitHub repository description** (paste into the repository “About” field):

> PyTorch research code: MLP + uncertainty-gated selective graph correction (SGC) for node classification on heterophilic graphs; PRL manuscript materials.

## License

This project is released under the [MIT License](LICENSE). A copy is also kept under `code/bfsbased_node_classification/LICENSE` for convenience when browsing only the implementation folder.

## Repository layout

| Path | Purpose |
|------|---------|
| `code/bfsbased_node_classification/` | Experiment runners, models, and the dynamically loaded core study script |
| `data/splits/` | Pre-bundled dataset splits (`.npz`) |
| `logs/` | Canonical JSONL/CSV run logs for tables and reproducibility |
| `figures/`, `tables/` | Generated figures and merged tables (including `prl_final_additions/`) |
| `reports/` | Audits, method notes, and PRL-facing writeups |
| `scripts/` | Build scripts for PRL tables, figures, and inventories |
| `slurm/` | Batch job templates |
| `prl_final_additions/` | Working bundle for manuscript tables/figures (mirrored under `tables/` and `figures/` by build scripts) |
| `README_PRL_MANUSCRIPT.md` | Index of manuscript-ready artifacts and regeneration commands |
| `AGENTS.md` | Tooling and validation notes for automation (e.g. Cursor Cloud) |

Archival experiment snapshots previously under `code/bfsbased_node_classification/diagnosis/` are **not** tracked in this branch; cite **`logs/`** for paper-ready outputs. Historical copies may exist on other git branches if needed.

## Dependencies

- Python 3 with **PyTorch** (CPU is enough for development)
- **PyTorch Geometric**, NumPy, NetworkX, sortedcontainers, Matplotlib

Install example (CPU wheels):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric numpy networkx sortedcontainers matplotlib
```

## Quick start

Run from the **repository root**. Example smoke run on Cora (single split, one repeat):

```bash
python3 code/bfsbased_node_classification/manuscript_runner.py \
  --split-dir data/splits --datasets cora --splits 0 --repeats 1 \
  --methods mlp_only selective_graph_correction --output-tag dev_test
```

PRL resubmission runner (adds GCN, APPNP baselines and ablations):

```bash
python3 code/bfsbased_node_classification/prl_resubmission_runner.py \
  --split-dir data/splits --datasets cora --splits 0 --repeats 1 \
  --methods mlp_only gcn appnp selective_graph_correction --output-tag dev_test
```

Check the emitted JSONL under `logs/` for `"success": true` on each record.

## Related writing and evidence

- **PRL evidence package and artifact index:** [README_PRL_MANUSCRIPT.md](README_PRL_MANUSCRIPT.md)
- **Earlier arXiv line** (heterophilic interpretability): [arxiv.org/abs/2512.22221](https://arxiv.org/abs/2512.22221) — see `code/bfsbased_node_classification/README.md` for the historical one-line summary.

## Citation

If you use this code, please cite the work you are building on (e.g. the PRL submission and/or the arXiv preprint above) using the citation your venue requires once DOI/page numbers are fixed.

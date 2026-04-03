## Cursor Cloud specific instructions

### Overview

This is a Python research codebase for the paper "Uncertainty-Gated Selective Graph Correction for Node Classification." The **submission method** is **FINAL_V3** (reliability-gated selective graph correction on top of an MLP). The repo also retains the original **selective graph correction (SGC v1)** path via `manuscript_runner.py` for baselines and historical logs. There are no web servers, databases, or Docker services — all work is Python script execution.

### Key dependencies

- PyTorch (CPU-only is sufficient for dev/CI)
- PyTorch Geometric (`torch_geometric`)
- NumPy, NetworkX, sortedcontainers, Matplotlib

Install with:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric numpy networkx sortedcontainers matplotlib
```

### FINAL_V3 entry point

- Implementation: `code/bfsbased_node_classification/final_method_v3.py`
- Stable import path: `code/final_method_v3.py` (requires PyTorch when imported)
- Regenerate main tables/figures from CSVs: `bash scripts/run_all_selective_correction_results.sh` (FINAL_V3, **10-split** canonical log `reports/final_method_v3_results.csv`)

### Running experiments

All experiment scripts live in `code/bfsbased_node_classification/`. They must be run from the repo root (`/workspace`).

**Manuscript runner** (MLP vs propagation vs SGC):
```bash
python3 code/bfsbased_node_classification/manuscript_runner.py \
  --split-dir data/splits --datasets cora --splits 0 --repeats 1 \
  --methods mlp_only selective_graph_correction --output-tag dev_test
```

**Resubmission runner** (adds GCN, APPNP baselines and ablations):
```bash
python3 code/bfsbased_node_classification/resubmission_runner.py \
  --split-dir data/splits --datasets cora --splits 0 --repeats 1 \
  --methods mlp_only gcn appnp selective_graph_correction --output-tag dev_test
```

### Gotchas

- The core module `bfsbased-full-investigate-homophil.py` is dynamically loaded by `manuscript_runner.py` (not directly importable due to the hyphenated filename and notebook-style top-level code). Do not attempt to `import` it directly.
- All dataset split files (`.npz`) are pre-bundled in `data/splits/`. Always pass `--split-dir data/splits` when running locally.
- The `resubmission_runner.py` imports `manuscript_runner` and `standard_node_baselines` as sibling modules. It must be invoked from the repo root or with `code/bfsbased_node_classification` on `sys.path`.
- GCN/APPNP baseline training (via `resubmission_runner.py`) takes ~50s per dataset-split on CPU due to grid search. MLP + SGC alone is much faster (~2s).
- No GPU/CUDA is required; all code runs on CPU.

### Linting

No project-level linter config exists. Use `flake8 --max-line-length 120` for basic checks:
```bash
python3 -m flake8 code/bfsbased_node_classification/ --max-line-length 120
```

### Testing

There is no formal test suite. Validate by running the experiment runners with `--datasets cora --splits 0 --repeats 1` and checking that output JSONL files contain `"success": true` records.

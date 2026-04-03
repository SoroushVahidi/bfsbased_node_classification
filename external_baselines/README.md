# External Baselines

This directory is reserved for vendored third-party baseline code for node classification experiments.

## Purpose

When a baseline cannot be implemented inline (due to complexity, specialized C++ extensions, or the need for exact reproducibility with the original paper), clone the official repository here and add a wrapper in `benchmark/runners/`.

## Planned Baselines

| Method | Status | Source Repository |
|--------|--------|-------------------|
| OrderedGNN | Planned (Phase C) | https://github.com/LUMIA-Group/OrderedGNN |
| DJ-GNN | Planned (Phase C) | https://github.com/BUPT-GAMMA/DJ-GNN |

## How to Clone

```bash
cd external_baselines/
git clone https://github.com/LUMIA-Group/OrderedGNN orderedgnn
git clone https://github.com/BUPT-GAMMA/DJ-GNN djgnn
```

After cloning, install any additional dependencies listed in each repo's `requirements.txt`, then update the corresponding runner in `benchmark/runners/` to call the cloned code using `subprocess` or by adding the path to `sys.path`.

## Already Integrated (Inline)

The following baselines are implemented inline in `benchmark/runners/` and do **not** require cloning:

- **GPRGNN** (`run_gprgnn.py`) - Generalized Page Rank GNN
- **GCNII** (`run_gcnii.py`) - Deep GCN with Initial Residual + Identity Mapping
- **H2GCN** (`run_h2gcn.py`) - Beyond Homophily
- **LINKX** (`run_linkx.py`) - Large Scale Non-Homophilous Graph Learning
- **FSGNN** (`run_fsgnn.py`) - Feature Selection GNN
- **ACM-GNN** (`run_acmgnn.py`) - Adaptive Channel Mixing GNN

## Notes

- Never commit large model checkpoints or dataset files to this directory.
- Add each cloned repo's name to `.gitignore` to avoid accidentally committing upstream code.
- Always record the commit hash of the cloned repo when reporting results.

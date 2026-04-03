# External Baselines

This directory is reserved for external baseline integrations and notes.

## Purpose

When a baseline cannot be implemented inline (due to complexity, specialized
extensions, or strict reproduction needs), clone the official repository here
and add a thin wrapper under `code/bfsbased_node_classification/`.

## Planned Baselines

| Method | Status | Source Repository |
|--------|--------|-------------------|
| GPRGNN | Integrated inline (external baseline) | https://github.com/jianhao2016/GPRGNN |
| FSGNN | Integrated inline (external baseline) | https://github.com/sunilkmaurya/FSGNN |
| PairNorm (DeepGCN+PairNorm) | Integrated inline (external baseline) | https://github.com/LingxiaoShawn/PairNorm |
| OrderedGNN | Planned | https://github.com/LUMIA-Group/OrderedGNN |
| DJ-GNN | Planned | https://github.com/BUPT-GAMMA/DJ-GNN |

## How to Clone

```bash
cd archive/exploratory/external_baselines/
git clone https://github.com/LUMIA-Group/OrderedGNN orderedgnn
git clone https://github.com/BUPT-GAMMA/DJ-GNN djgnn
```

After cloning, install any additional dependencies listed in each repo's
`requirements.txt`, then add/update the corresponding wrapper in
`code/bfsbased_node_classification/` to call the cloned code using `subprocess`
or by adding the path to `sys.path`.

## Already Integrated (Inline)

The following baselines are implemented inline and do **not** require cloning:

- **GPRGNN**:
  - model helper: `code/bfsbased_node_classification/standard_node_baselines.py`
  - standalone runner: `code/bfsbased_node_classification/gprgnn_baseline_runner.py`
  - integration note: `archive/exploratory/external_baselines/GPRGNN_EXTERNAL_BASELINE.md`
- **FSGNN**:
  - model helper: `code/bfsbased_node_classification/standard_node_baselines.py`
  - standalone runner: `code/bfsbased_node_classification/fsgnn_baseline_runner.py`
  - integration note: `archive/exploratory/external_baselines/FSGNN_EXTERNAL_BASELINE.md`
- **PairNorm (DeepGCN + PairNorm)**:
  - model helper: `code/bfsbased_node_classification/standard_node_baselines.py`
  - standalone runner: `code/bfsbased_node_classification/pairnorm_baseline_runner.py`
  - integration note: `archive/exploratory/external_baselines/PAIRNORM_EXTERNAL_BASELINE.md`

## Notes

- Never commit large model checkpoints or dataset files to this directory.
- Add each cloned repo's name to `.gitignore` to avoid accidentally committing upstream code.
- Always record the commit hash of the cloned repo when reporting results.
- External baselines are comparison tools and are **not** part of the canonical
  `FINAL_V3` method.

# Reproducing FINAL_V3 outputs

Repository navigation: [`README.md`](README.md) (scripts index), [`../AGENTS.md`](../AGENTS.md), [`../slurm/README.md`](../slurm/README.md).

This repository is the **evidence package** for a research paper submission. The canonical numerical results are **frozen** in `reports/final_method_v3_results.csv` (**10 GEO-GCN splits** per dataset, indices 0–9).
This file documents reproducibility for the **canonical** method line (`FINAL_V3`).

## Quick refresh (no training)

```bash
bash scripts/run_all_selective_correction_results.sh
```

Regenerates canonical tables/figures from frozen CSV/log inputs.

## Full rerun (training)

```bash
python3 code/bfsbased_node_classification/run_final_evaluation.py \
  --split-dir data/splits \
  --datasets cora citeseer pubmed chameleon texas wisconsin \
  --splits 0 1 2 3 4 5 6 7 8 9 \
  --output-tag rerun_$(date +%Y%m%d)
```

### Safety

- Always pass `--output-tag` for non-canonical reruns.
- Keep `reports/final_method_v3_results.csv` immutable as canonical evidence.

## Dependency setup (CPU)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric numpy networkx sortedcontainers matplotlib
```

## Additional references

- `README.md`
- `docs/OVERVIEW.md`
- `docs/REPRODUCIBILITY.md`
- `docs/RESULTS_GUIDE.md`

# Reproducibility

This guide separates quick artifact refresh from full experiment reruns.

## Environment

Minimum:
```bash
pip install -r requirements.txt
```

CPU-only PyTorch + PyG stack (for training reruns):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric numpy networkx sortedcontainers matplotlib
```

## Canonical quick path (recommended)

From repo root:
```bash
bash scripts/run_all_selective_correction_results.sh
```

This regenerates canonical tables and figures from frozen logs/CSVs.

## Full rerun path (training)

```bash
python3 code/bfsbased_node_classification/run_final_evaluation.py \
  --split-dir data/splits \
  --datasets cora citeseer pubmed chameleon texas wisconsin \
  --splits 0 1 2 3 4 5 6 7 8 9 \
  --output-tag rerun_$(date +%Y%m%d)
```

### Safety rules

- Use `--split-dir data/splits`.
- Use `--output-tag` for reruns to avoid modifying frozen canonical files.
- Treat `reports/final_method_v3_results.csv` as immutable canonical evidence.

## Canonical outputs to inspect

- `tables/main_results_selective_correction.md`
- `tables/ablation_selective_correction.md`
- `tables/sensitivity_selective_correction.md`
- `reports/final_method_v3_analysis.md`
- `reports/safety_analysis.md`

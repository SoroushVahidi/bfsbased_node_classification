# How to Run Baselines

This guide explains how to run the GNN baselines integrated in this benchmark.

## Prerequisites

### Python Environment

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric numpy networkx pyyaml
```

All scripts should be run from the **repository root**:
```
/home/runner/work/bfsbased_node_classification/bfsbased_node_classification
```

### Data

Split files must exist in `data/splits/`. They are pre-bundled in this repo:
```
data/splits/cora_split_0.6_0.2_0.npz
data/splits/cora_split_0.6_0.2_1.npz
...
```

PyG datasets are auto-downloaded to `data/pyg_cache/` on first run.

---

## Running Individual Baselines

All runner scripts accept the same CLI interface:

```bash
python benchmark/runners/run_{METHOD}.py \
  --dataset {DATASET} \
  --split-id {SPLIT_ID} \
  --seed {SEED} \
  --output-tag {TAG} \
  [--config path/to/config.yaml] \
  [--split-dir path/to/splits/] \
  [--data-root path/to/pyg/cache/] \
  [--output-dir path/to/output/] \
  [--force-rerun]
```

### Examples

```bash
# Run GPRGNN on Cora, split 0, seed 42
python benchmark/runners/run_gprgnn.py --dataset cora --split-id 0 --seed 42 --output-tag test

# Run GCNII on Texas, split 3, with custom split directory
python benchmark/runners/run_gcnii.py \
  --dataset texas --split-id 3 --seed 0 \
  --split-dir data/splits --output-tag paper

# Run H2GCN on Chameleon
python benchmark/runners/run_h2gcn.py --dataset chameleon --split-id 0 --seed 42

# Run LINKX on Wisconsin
python benchmark/runners/run_linkx.py --dataset wisconsin --split-id 0 --seed 42

# Run FSGNN on Actor
python benchmark/runners/run_fsgnn.py --dataset actor --split-id 0 --seed 42

# Run ACM-GNN on Squirrel
python benchmark/runners/run_acmgnn.py --dataset squirrel --split-id 0 --seed 42
```

### Available Datasets

`cora`, `citeseer`, `pubmed`, `chameleon`, `squirrel`, `texas`, `cornell`, `wisconsin`, `actor`

### Available Methods

| Script | Method | Status |
|--------|--------|--------|
| `run_gprgnn.py` | GPRGNN | ✅ functional |
| `run_gcnii.py` | GCNII | ✅ functional |
| `run_h2gcn.py` | H2GCN | ✅ functional |
| `run_linkx.py` | LINKX | ✅ functional |
| `run_fsgnn.py` | FSGNN | ✅ functional |
| `run_acmgnn.py` | ACM-GNN | ✅ functional |
| `run_orderedgnn.py` | OrderedGNN | 🔲 planned stub |
| `run_djgnn.py` | DJ-GNN | 🔲 planned stub |

---

## Running the Full Benchmark Suite

Use `benchmark/run_benchmark_suite.py` to run multiple methods, datasets, and splits in sequence:

```bash
# Run all wrapped methods on Cora and Texas, splits 0-4
python benchmark/run_benchmark_suite.py \
  --methods gprgnn gcnii h2gcn linkx fsgnn acmgnn \
  --datasets cora texas \
  --splits 0 1 2 3 4 \
  --seed 42 \
  --output-tag paper_run

# Run everything on a single split for quick testing
python benchmark/run_benchmark_suite.py \
  --datasets cora \
  --splits 0 \
  --output-tag dev_test

# Force re-run even if results exist
python benchmark/run_benchmark_suite.py \
  --datasets cora --splits 0 --force-rerun
```

---

## Where Outputs Go

### Per-run JSON files

Each run writes an atomic JSON result:
```
benchmark/results/per_run/{method}_{dataset}_split{split_id}_seed{seed}_{tag}.json
```

Example:
```
benchmark/results/per_run/gprgnn_cora_split0_seed42_paper_run.json
```

### Summary files

The suite runner writes merged summaries:
```
benchmark/results/summaries/benchmark_summary_{tag}_{timestamp}.json
benchmark/results/summaries/benchmark_summary_{tag}_{timestamp}.csv
```

### Logs

Individual run stdout/stderr is printed to the terminal. Redirect as needed:
```bash
python benchmark/run_benchmark_suite.py ... 2>&1 | tee benchmark/logs/run_$(date +%Y%m%d).log
```

---

## How to Add a New Baseline

1. **Create a config** in `benchmark/configs/{method}.yaml` with hyperparameters.

2. **Add an entry** to `benchmark/baseline_registry.yaml` with status `planned`.

3. **Create the runner** `benchmark/runners/run_{method}.py`:
   - Copy an existing runner (e.g., `run_gprgnn.py`) as a template.
   - Replace the model implementation.
   - Keep the CLI args, JSON output format, and early stopping logic identical.

4. **Register the method** in `benchmark/run_benchmark_suite.py`:
   - Add to `ALL_METHODS` list.
   - The `METHOD_SCRIPTS` dict auto-generates the script path.

5. **Update** `benchmark/baseline_registry.yaml` status to `wrapped`.

6. **Test** with:
   ```bash
   python benchmark/runners/run_{method}.py --dataset cora --split-id 0 --seed 42 --output-tag test
   ```

7. **Update** `benchmark/docs/BASELINE_INTEGRATION_STATUS.md`.

---

## Troubleshooting

**`ModuleNotFoundError: benchmark`**: Run from repo root, not from inside `benchmark/`.

**`FileNotFoundError: Split file not found`**: Ensure `data/splits/` exists and contains `.npz` files. Pass `--split-dir data/splits` explicitly.

**PyG dataset download fails**: Check internet connectivity. PyG caches datasets in `data/pyg_cache/` after first download.

**OOM on large datasets**: GCNII with 64 layers can be memory-intensive on squirrel/chameleon. Reduce `num_layers` in `benchmark/configs/gcnii.yaml`.

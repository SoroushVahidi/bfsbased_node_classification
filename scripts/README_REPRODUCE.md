# Reproducing PRL evidence (FINAL_V3)

This repository is the **evidence package** for a Pattern Recognition Letters submission. The canonical numerical results are **frozen** in `reports/final_method_v3_results.csv`.

## One-command refresh (recommended)

From the repository root:

```bash
bash scripts/run_all_prl_results.sh
```

This regenerates:

- `tables/main_results_prl.md` and `tables/main_results_prl.csv`
- `figures/prl_graphical_abstract_v3.png`
- `figures/correction_rate_vs_homophily.png`
- `figures/safety_comparison.png`
- `figures/reliability_vs_accuracy.png`

**Runtime:** a few seconds (Matplotlib + CSV I/O only; **no** PyTorch training).

**Dependencies:** Python 3, NumPy, Matplotlib. Homophily values for the correction-rate figure are read from existing files under `logs/` (`manuscript_regime_analysis_final_validation_main.csv` and/or `regime_analysis_manuscript_final_validation_job2.csv`).

## If you must re-evaluate from scratch (heavy)

Re-running experiments retrains MLPs and repeats grid search — **minutes to tens of minutes on CPU** depending on datasets and methods:

```bash
python3 code/bfsbased_node_classification/run_final_evaluation.py \
  --split-dir data/splits \
  --datasets cora citeseer pubmed chameleon texas wisconsin \
  --splits 0 1 2 3 4
```

That overwrites or appends according to the runner’s logging options; the submission package assumes you keep the shipped `reports/final_method_v3_results.csv` unless you intentionally refresh it.

## Optional: broader PRL artifact bundle

Additional tables/figures derived from `logs/` and `prl_final_additions/`:

```bash
python3 scripts/prl_final_additions/build_all_prl_artifacts.py
```

This runs several plotting/table scripts and mirrors `prl_final_additions/` into `tables/prl_final_additions/` and `figures/prl_final_additions/`. Expect **tens of seconds** if Matplotlib is warm.

## Full Python stack

For training runs (not required for `run_all_prl_results.sh`):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric numpy networkx sortedcontainers matplotlib
```

See also the root [README.md](../README.md) and [README_PRL_MANUSCRIPT.md](../README_PRL_MANUSCRIPT.md).

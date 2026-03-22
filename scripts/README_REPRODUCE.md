# Reproducing PRL evidence (FINAL_V3)

This repository is the **evidence package** for a Pattern Recognition Letters submission. The canonical numerical results are **frozen** in `reports/final_method_v3_results.csv` (**10 GEO-GCN splits** per dataset, indices 0–9).

## One-command refresh (recommended)

From the repository root:

```bash
bash scripts/run_all_prl_results.sh
```

This regenerates:

- `tables/main_results_prl.md` and `tables/main_results_prl.csv`
- `tables/experimental_setup_prl.md`, `tables/ablation_prl.md`, `tables/sensitivity_prl.md` (and `.csv` partners)
- `figures/prl_graphical_abstract_v3.png`
- `figures/correction_rate_vs_homophily.png`
- `figures/safety_comparison.png`
- `figures/reliability_vs_accuracy.png`

**Runtime:** a few seconds (Matplotlib + CSV I/O only; **no** PyTorch training).

**Dependencies:** Python 3, NumPy, Matplotlib. Homophily values for the correction-rate figure are read from existing files under `logs/` (`manuscript_regime_analysis_final_validation_main.csv` and/or `regime_analysis_manuscript_final_validation_job2.csv`).

## Rebuild the canonical CSV from the frozen 10-split export (no training)

If you edit the archived per-split export, regenerate the canonical log before refreshing tables:

```bash
python3 scripts/prl_final_additions/rebuild_final_v3_results_10split.py
bash scripts/run_all_prl_results.sh
```

## If you must re-evaluate from scratch (heavy)

Re-running experiments retrains MLPs and repeats grid search — **tens of minutes on CPU** for all six datasets × **10 splits** × four methods:

```bash
python3 code/bfsbased_node_classification/run_final_evaluation.py \
  --split-dir data/splits \
  --datasets cora citeseer pubmed chameleon texas wisconsin \
  --splits 0 1 2 3 4 5 6 7 8 9
```

That **overwrites** `reports/final_method_v3_results.csv` in the default configuration; keep backups if you are comparing against the shipped evidence package.

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

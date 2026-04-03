# Reproducing PRL evidence (FINAL_V3)

This repository is the **evidence package** for a Pattern Recognition Letters submission. The canonical numerical results are **frozen** in `reports/final_method_v3_results.csv` (**10 GEO-GCN splits** per dataset, indices 0–9).

> ⚠️ **Safety rule:** Always pass `--output-tag <name>` when running any
> experiment script. Without a tag, some scripts write to the default canonical
> output path and may overwrite frozen evidence. See the "If you must re-evaluate
> from scratch" section below for the safe pattern.

## One-command refresh (recommended)

From the repository root:

```bash
bash scripts/run_all_prl_results.sh
```

This regenerates:

- `tables/main_results_prl.md` and `tables/main_results_prl.csv`
- `tables/experimental_setup_prl.md`, `tables/ablation_prl.md`, `tables/sensitivity_prl.md` (and `.csv` partners)
- `figures/prl_graphical_abstract_v3.png`
- `figures/prl_graphical_abstract_v3.pdf`
- `figures/prl_graphical_abstract_v3.svg`
- `figures/correction_rate_vs_homophily.png`
- `figures/safety_comparison.png`
- `figures/reliability_vs_accuracy.png`

**Runtime:** a few seconds (Matplotlib + CSV I/O only; **no** PyTorch training).

**Dependencies:** Python 3, NumPy, Matplotlib. Homophily values for the correction-rate figure are read from existing files under `logs/` (`manuscript_regime_analysis_final_validation_main.csv` and/or `regime_analysis_manuscript_final_validation_job2.csv`).

**Important caveat about auxiliary FINAL_V3 diagnostics:** the shipped
`reports/final_method_v3_results.csv` is standardized on the full **10-split**
benchmark for the main accuracy table, but some split-level diagnostic columns
(`frac_confident`, `frac_unreliable`, `tau`, `rho`, `runtime_sec`) are present
only for splits `0–4` in the frozen rebuild path. That is sufficient for the
current manuscript-facing behavior report because `reports/final_method_v3_analysis.md`
already restricts those specific aggregates to splits `0–4` and uses all `10`
splits for the main benchmark and harmful-split counts.

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

That **overwrites** `reports/final_method_v3_results.csv` in the default
configuration; keep backups if you are comparing against the shipped evidence
package.

> ⚠️ **Always** use `--output-tag <name>` for any re-run you do not intend to
> make canonical. This writes results to `reports/final_method_v3_results_<name>.csv`
> instead of the frozen canonical path.

Use `--output-tag <name>` if you want a side-by-side rerun without touching the
canonical CSV:

```bash
python3 code/bfsbased_node_classification/run_final_evaluation.py \
  --split-dir data/splits \
  --datasets cora citeseer pubmed chameleon texas wisconsin \
  --splits 0 1 2 3 4 5 6 7 8 9 \
  --output-tag scratch
```

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

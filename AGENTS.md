# Agent onboarding (AI assistants and new contributors)

This file is the **primary map** for navigating the repository. For a link-only overview, see [`docs/INDEX.md`](docs/INDEX.md).

---

## 1. Read this first (30 seconds)

1. **Canonical method:** `FINAL_V3` — reliability-gated selective graph correction on top of an MLP. Frozen evidence must not be overwritten (see §6).
2. **Main Python package:** `code/bfsbased_node_classification/` — almost all training runners and baselines live here.
3. **Legacy core:** `bfsbased-full-investigate-homophil.py` is loaded dynamically by `manuscript_runner.py`; do not `import` it like a normal module (hyphenated name + notebook-style globals).

---

## 2. Task → entry point

| Goal | Start here |
|------|------------|
| Understand what is canonical vs experimental | [`REPO_STATUS.md`](REPO_STATUS.md), [`CANONICAL_CLAIMS.md`](CANONICAL_CLAIMS.md) |
| Re-run the main 10-split benchmark (tagged) | `code/bfsbased_node_classification/run_final_evaluation.py` + [`scripts/README_REPRODUCE.md`](scripts/README_REPRODUCE.md) |
| Regenerate tables/figures from existing CSVs (no training) | `bash scripts/run_all_selective_correction_results.sh` |
| Run MLP + selective correction (UG-SGC line) | `code/bfsbased_node_classification/manuscript_runner.py` |
| Run GCN/APPNP/H2GCN + SGC variants + paper baselines (CLP, C&S, GraphSAGE, …) | `code/bfsbased_node_classification/resubmission_runner.py` + [`docs/BASELINES_SUITE.md`](docs/BASELINES_SUITE.md) |
| Call graph baselines from code | `standard_node_baselines.run_baseline(...)` in `code/bfsbased_node_classification/standard_node_baselines.py` |
| Large-scale / OGB / TabGraphs datasets | [`docs/DATASETS_EXTENDED.md`](docs/DATASETS_EXTENDED.md) |
| Heterophily-heavy baselines (GCNII, LINKX, …) | [`docs/BEGGA_HETEROPHILY_BASELINES.md`](docs/BEGGA_HETEROPHILY_BASELINES.md) |
| Slurm / cluster jobs | `slurm/` |
| Verify repo integrity | `python3 scripts/check_canonical.py` |

---

## 3. Method lines (do not conflate)

| Line | Status | Typical driver |
|------|--------|----------------|
| **FINAL_V3** | Canonical / frozen claims | `run_final_evaluation.py`, `final_method_v3.py` |
| **UG-SGC** | Legacy / supplementary | `manuscript_runner.py` + dynamic legacy module |
| **UG-SGC-S** | Exploratory structural extension | `resubmission_runner.py` |

Only **FINAL_V3** artifacts in `reports/` and `tables/` listed in `REPO_STATUS.md` support the main paper claims. UG-SGC / UG-SGC-S outputs belong under tagged logs or `archive/`.

---

## 4. Important files (by role)

### Canonical implementation

- `code/bfsbased_node_classification/final_method_v3.py` — FINAL_V3 logic
- `code/final_method_v3.py` — stable import shim (re-exports from above)

### Training and evaluation

- `run_final_evaluation.py` — MLP, baseline SGC v1, V2 multibranch, FINAL_V3; writes CSV when tagged
- `manuscript_runner.py` — legacy full manuscript protocol (loads `bfsbased-full-investigate-homophil.py`)
- `resubmission_runner.py` — extended methods + `run_baseline(...)` for standard GNNs and paper baselines
- `baseline_comparison_suite.py` — Wulver-scale JSONL comparison (MLP, FINAL_V3, optional external baselines)

### Baselines and shared models

- `standard_node_baselines.py` — GCN, APPNP, H2GCN, SGC, GraphSAGE, Correct & Smooth, CLP, Begga-style models; single dispatch: `run_baseline(model_name, data, train_idx, val_idx, test_idx, seed=...)`
- `gcn_baseline_runner.py` — dedicated GCN Table 1 protocol
- `pairnorm_baseline_runner.py`, `fsgnn_baseline_runner.py`, `gprgnn_baseline_runner.py` — standalone runners

### Data

- `split_paths.py`, `make_splits.py` — split naming and generation
- `extended_graph_datasets.py` — OGB / extended loaders (with `load_dataset` in legacy module)

### Dynamic legacy module (special)

- `bfsbased-full-investigate-homophil.py` — contains `load_dataset`, `train_mlp_and_predict`, `selective_graph_correction_predictclass`, etc. Loaded by slicing source in `manuscript_runner` / `run_final_evaluation`. **Treat as opaque unless you are editing the legacy line.**

---

## 5. How to run Python here

- **Working directory:** repo root is safest. Many scripts assume paths relative to root (`data/splits`, `reports/`).
- **Path:** scripts under `code/bfsbased_node_classification/` import each other as siblings. Run either:
  - `python3 code/bfsbased_node_classification/<script>.py` from repo root, or
  - `cd code/bfsbased_node_classification && python3 <script>.py` (ensure relative paths to `data/` still resolve if the script uses repo-root paths).
- **Splits:** pass `--split-dir data/splits` when applicable. Extended benchmarks may need prepared `.npz` per `docs/DATASETS_EXTENDED.md`.

---

## 6. Frozen and protected outputs

**Do not edit or overwrite** without explicit project approval:

- `reports/final_method_v3_results.csv` (canonical per-split log)
- Canonical tables under `tables/` listed in `REPO_STATUS.md` / `CONTRIBUTING.md`
- Designated figures in `figures/` that feed the paper

**Always** use a unique `--output-tag` (or separate output paths) when re-running benchmarks so canonical files are untouched. See `CONTRIBUTING.md`.

---

## 7. Dependencies and checks

- Install: `pip install -r requirements.txt` (see `README.md` for PyTorch CPU wheel hint).
- Lint (no dedicated config file): `python3 -m flake8 code/bfsbased_node_classification/ --max-line-length 120`
- Sanity: `python3 scripts/check_canonical.py`
- There is **no** pytest suite; smoke-test with `resubmission_runner.py` or `run_final_evaluation.py` on `--datasets cora --splits 0` when splits exist.

---

## 8. Gotchas

- **Hyphenated legacy file** cannot be imported normally; runners use `exec`/`importlib` patterns.
- **FULL_V3 vs FINAL_V3:** the repo standard name is **FINAL_V3**.
- **GraphSAGE / CLP / Correct & Smooth** refuse full-batch runs above a **node-count safety threshold** (see `standard_node_baselines.py` and `docs/BASELINES_SUITE.md`) — not all OGB graphs are supported at full scale in-repo.
- **GPU:** not required; CPU is supported.

---

## 9. Suggested change workflow for agents

1. Read `REPO_STATUS.md` + this file to classify your task (canonical vs exploratory).
2. Locate the right script from §2; read only the modules you need (avoid loading the entire legacy notebook module into context unless necessary).
3. Make minimal, scoped code changes; do not refactor unrelated baselines.
4. Run `flake8` and `scripts/check_canonical.py` where relevant.
5. Document new baselines or datasets in `docs/` and link from `docs/INDEX.md` if you add a new doc.

---

## 10. Paper links

- [arXiv:2512.22221](https://arxiv.org/abs/2512.22221)
- [SSRN abstract](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6462436)

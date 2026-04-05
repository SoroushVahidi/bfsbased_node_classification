# Uncertainty-Gated Selective Graph Correction for Node Classification

**Canonical method:** `FINAL_V3` — reliability-gated selective graph correction on top of a strong MLP  
**Canonical claims and result files:** [`CANONICAL_CLAIMS.md`](CANONICAL_CLAIMS.md)  
**Full reproduction guide:** [`scripts/README_REPRODUCE.md`](scripts/README_REPRODUCE.md)  
**Repository status and classification:** [`REPO_STATUS.md`](REPO_STATUS.md)  
**Scripts index:** [`scripts/README.md`](scripts/README.md) · **Slurm templates:** [`slurm/README.md`](slurm/README.md)  
**AI / agent onboarding:** [`AGENTS.md`](AGENTS.md) · [Documentation index](docs/INDEX.md)

This repository contains code and reproducible artifacts for a node-classification method that combines a strong feature-first MLP with **selective graph correction**.

---

## Paper evidence map

| Layer | Files | Status |
|-------|-------|--------|
| **Frozen canonical evidence** | `reports/final_method_v3_results.csv` | Do not overwrite |
| **Canonical main tables** | `tables/main_results_selective_correction.*`, `tables/experimental_setup_selective_correction.*`, `tables/ablation_selective_correction.*`, `tables/sensitivity_selective_correction.*` | Regenerated from frozen CSV |
| **Canonical analyses** | `reports/final_method_v3_analysis.md`, `reports/safety_analysis.md`, `reports/final_v3_regime_analysis.md` | Paper narrative support |
| **Canonical figures** | `figures/graphical_abstract_selective_correction_v3.*`, `figures/correction_rate_vs_homophily.png`, `figures/safety_comparison.png`, `figures/reliability_vs_accuracy.png` | Regenerated from frozen CSV |
| **Appendix / supplementary support** | Fresh ablation rerun (`tables/ablation_selective_correction_rerun.*`), full-scope diagnostics (`tables/final_v3_correction_diagnostics_20260404_fullscope.*`), gate sensitivity (`tables/gate_sensitivity_fullscope_20260404.*`), bounded intervention (`tables/bounded_intervention_fullscope_20260404.*`) | Non-canonical; current-code |
| **Manuscript-supporting bundles** | Bucket safety, correction explanation audit, harmful-correction audit, safeguard simulation, regime mapping, targeted-benefit decomposition | Non-canonical; see [`reports/SUPPORTING_BUNDLES_INDEX.md`](reports/SUPPORTING_BUNDLES_INDEX.md) |
| **Exploratory experiments** | MS-HSGC, CA-MRC, CA-MRD, recon-behavior | Non-canonical; informative context |
| **Archived / provenance** | `archive/`, `reports/archive/`, `reports/parallel_tracks_analysis_*.md` | Not for paper claims |

For a full artifact inventory, see [`docs/RESULTS_GUIDE.md`](docs/RESULTS_GUIDE.md) and [`reports/SUPPORTING_BUNDLES_INDEX.md`](reports/SUPPORTING_BUNDLES_INDEX.md).

---

## Start here

1. Read this README (especially the evidence map above).
2. Read [`docs/ANALYSIS_GUIDE.md`](docs/ANALYSIS_GUIDE.md) for a reviewer-oriented reading path.
3. Run the canonical reproduction command:
   ```bash
   bash scripts/run_all_selective_correction_results.sh
   ```
4. Inspect the canonical main table:
   - `tables/main_results_selective_correction.md`
   - `tables/main_results_selective_correction.csv`

---

## Canonical method (single source of truth)

**Canonical method name:** `FINAL_V3`  
**Canonical implementation entry point:** `code/final_method_v3.py`

`FINAL_V3` is the only method line treated as canonical for main claims in this repository.

High-level behavior:
- train a feature-only MLP,
- estimate per-node reliability from model confidence,
- apply graph correction only to low-reliability nodes,
- leave high-reliability nodes unchanged.

This is a conservative correction strategy, not a claim of universal superiority across all datasets and splits.

---

## Reproducibility at a glance

### Fast refresh (no training)

```bash
bash scripts/run_all_selective_correction_results.sh
```

Regenerates tables/figures from frozen CSV/log inputs.

### Full rerun (training)

```bash
python3 code/bfsbased_node_classification/run_final_evaluation.py \
  --split-dir data/splits \
  --datasets cora citeseer pubmed chameleon texas wisconsin \
  --splits 0 1 2 3 4 5 6 7 8 9 \
  --output-tag rerun_$(date +%Y%m%d)
```

Always use `--output-tag` for non-canonical reruns.

**Extended datasets (feature-rich nodes + official benchmark splits):** `ogbn-arxiv`, `ogbn-products`, and TabGraphs `hm-categories` are supported via `load_dataset` and the usual runners when split `.npz` files are prepared. See [`docs/DATASETS_EXTENDED.md`](docs/DATASETS_EXTENDED.md).

**Standard baselines (GCN, APPNP, SGC, GraphSAGE, Correct & Smooth, CLP, …):** method names, paper links, and implementation fidelity notes are in [`docs/BASELINES_SUITE.md`](docs/BASELINES_SUITE.md).

---
For details, see [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

---

## Repository layout

```text
.
├── code/                    # Method and runner implementations
├── data/                    # Datasets and pre-generated split files
├── reports/                 # Result CSVs and narrative analyses
├── tables/                  # Rendered benchmark/ablation/sensitivity tables
├── figures/                 # Rendered figures
├── logs/                    # Run logs used by report/table builders
├── scripts/                 # Reproduction and artifact build scripts
├── docs/                    # Reader-focused documentation
└── archive/                 # Legacy venue-specific and exploratory material
```

### Status model

The repository uses exactly three status labels:
- **Canonical** — active, claim-bearing path (`FINAL_V3`).
- **Archived** — historical/venue-specific/provenance material.
- **Exploratory** — non-canonical research branches and diagnostics.

See [`docs/METHOD_STATUS.md`](docs/METHOD_STATUS.md) for explicit classification.

---

## Canonical artifacts

- Frozen per-split results: `reports/final_method_v3_results.csv`
- Main table: `tables/main_results_selective_correction.md` and `.csv`
- Setup table: `tables/experimental_setup_selective_correction.md` and `.csv`
- Ablation table: `tables/ablation_selective_correction.md` and `.csv`
- Sensitivity table: `tables/sensitivity_selective_correction.md` and `.csv`
- Main analyses: `reports/final_method_v3_analysis.md`, `reports/safety_analysis.md`, `reports/final_v3_regime_analysis.md`

See [`docs/RESULTS_GUIDE.md`](docs/RESULTS_GUIDE.md) for a reviewer-oriented map.

---

## Archived and exploratory material

Older UG-SGC / structural-extension / venue-specific bundles are preserved under `archive/` with provenance intact and explicit non-canonical labeling.

Beyond the high-level layout above, these paths matter for baselines and cluster work:

- `code/baselines/djgnn/` — DJGNN helpers ([`docs/DJGNN_INTEGRATION.md`](docs/DJGNN_INTEGRATION.md))
- `code/bfsbased_node_classification/baseline_comparison_suite.py` — JSONL comparison driver for large sweeps
- `scripts/baseline_comparison_wulver/` — Slurm launch scripts and aggregation (see README inside)
- `slurm/` — batch templates ([`slurm/README.md`](slurm/README.md))
- `outputs/` — regenerated sweep JSONL (gitignored under `outputs/*/`; not canonical paper artifacts)

If you need those materials, start with:
- `archive/README.md`
- `archive/legacy_venue_specific/README.md`

---

## Dependencies

```bash
pip install -r requirements.txt
```

For CPU-only PyTorch + PyG setup details, see `docs/REPRODUCIBILITY.md`.

## License

[MIT](LICENSE)

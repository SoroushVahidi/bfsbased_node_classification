# Uncertainty-Gated Selective Graph Correction for Node Classification

This repository contains code and reproducible artifacts for a node-classification method that combines a strong feature-first MLP with **selective graph correction**.

## Start here

1. Read this README.
2. Read [`docs/OVERVIEW.md`](docs/OVERVIEW.md).
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

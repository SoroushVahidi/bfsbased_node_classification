# Scripts index

Short map of `scripts/` — for full reproduction steps see [`README_REPRODUCE.md`](README_REPRODUCE.md).

### Canonical artifact scripts

| Path | Role |
|------|------|
| [`README_REPRODUCE.md`](README_REPRODUCE.md) | End-to-end benchmark re-run, tags, frozen outputs |
| [`run_all_selective_correction_results.sh`](run_all_selective_correction_results.sh) | **Canonical** — regenerate canonical tables/figures from existing CSVs (no training) |
| [`check_canonical.py`](check_canonical.py) | **Canonical** — verify frozen canonical files are present and consistent |
| [`build_artifacts/`](build_artifacts/) | **Canonical** — Python builders for paper tables and figures (called from `run_all_selective_correction_results.sh`) |

### Supporting / appendix scripts

| Path | Role |
|------|------|
| [`final_v3_regime_mapping.py`](final_v3_regime_mapping.py) | Manuscript-supporting — descriptive regime map reusing full-scope outputs; writes to `reports/` and `tables/` |
| [`posthoc_safeguard_simulation.py`](posthoc_safeguard_simulation.py) | Manuscript-supporting — post hoc safeguard simulation on changed-node explanation exports |

### Infrastructure / data-prep scripts

| Path | Role |
|------|------|
| [`prepare_extended_dataset_splits.py`](prepare_extended_dataset_splits.py) | Prepare extended / OGB-style split `.npz` files |
| [`baseline_comparison_wulver/`](baseline_comparison_wulver/) | Slurm launchers, aggregation, and docs for large comparison / robustness sweeps |

### Exploratory / diagnostic scripts

| Path | Role |
|------|------|
| [`smoke_begga_baselines.py`](smoke_begga_baselines.py) | Quick smoke test for heterophily-style baselines |
| [`test_ms_hsgc.py`](test_ms_hsgc.py) | Exploratory — lightweight MS-HSGC sanity test; non-canonical |

Cluster job definitions live in [`../slurm/README.md`](../slurm/README.md).

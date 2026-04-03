# Repository status

This file describes the current state of the repository: which packages are
canonical, which are legacy, and which are still exploratory. See
[`REPO_STATUS.md`](../REPO_STATUS.md) for a shorter version at the root level.

---

## Canonical package (FINAL_V3)

**Status: ✅ Frozen — do not modify result files**

The canonical package for this research paper submission is **FINAL_V3**
(reliability-gated selective graph correction on top of a strong MLP).

### Frozen evidence files

| File | Description |
|------|-------------|
| `reports/final_method_v3_results.csv` | Per-split log, 10 splits × 6 datasets |
| `tables/main_results_selective_correction.{md,csv}` | Main benchmark table |
| `tables/experimental_setup_selective_correction.{md,csv}` | Experimental setup table |
| `tables/ablation_selective_correction.{md,csv}` | Ablation table |
| `tables/sensitivity_selective_correction.{md,csv}` | Sensitivity table |
| `figures/graphical_abstract_selective_correction_v3.*` | Graphical abstract |
| `figures/correction_rate_vs_homophily.png` | Correction-rate vs homophily |
| `figures/safety_comparison.png` | Safety analysis figure |
| `figures/reliability_vs_accuracy.png` | Reliability vs accuracy figure |
| `reports/safety_analysis.md` | Harmful-split analysis |
| `reports/final_method_v3_analysis.md` | Method analysis narrative |

### Canonical scripts

| Script | Role |
|--------|------|
| `code/final_method_v3.py` | Stable import path |
| `code/bfsbased_node_classification/final_method_v3.py` | Core implementation |
| `code/bfsbased_node_classification/run_final_evaluation.py` | 10-split benchmark driver |
| `bash scripts/run_all_selective_correction_results.sh` | One-command table/figure refresh |

---

## Legacy package (UG-SGC)

**Status: 📁 Preserved for provenance — supplementary use only**

The original uncertainty-gated selective correction line. Useful as background
and supplementary context but **not** the paper's main evidence.

### Key locations (archived)

- Runner: `code/bfsbased_node_classification/manuscript_runner.py`
- Core module: `code/bfsbased_node_classification/bfsbased-full-investigate-homophil.py`
- Tables: `archive/legacy_venue_specific/tables_prl_final_additions/`
- Figures: `archive/legacy_venue_specific/figures_prl_final_additions/`
- Reports: `archive/legacy_venue_specific/reports_prl_final_additions/`
- Logs: `logs/*manuscript_final_validation*`
- Threshold sensitivity: `archive/legacy_venue_specific/results_prl/`

---

## Exploratory package (UG-SGC-S)

**Status: 🔬 Partial / exploratory — some grouped jobs may be incomplete**

The structural extension adds a far-structural support term to the selective
correction. Evidence is promising but mixed; not a replacement for FINAL_V3.

### Key locations (archived)

- Runner: `code/bfsbased_node_classification/resubmission_runner.py`
- Tables: `archive/legacy_venue_specific/tables_prl_resubmission/`
- Logs: `archive/legacy_venue_specific/logs_prl_resubmission/`
- Reports: `archive/legacy_venue_specific/reports_resubmission_protocols/`

---

## GCN / APPNP baselines

**Status: ✅ Complete (6 datasets × 10 splits)**

Standalone GCN baseline for Table 1 comparison.

- Runner: `code/bfsbased_node_classification/gcn_baseline_runner.py`
- Log: `logs/gcn_runs_gcn_table1.jsonl`
- Summary: `reports/gcn_results_gcn_table1.csv`

---

## Benchmark suite

**Status: 🔧 Integration-ready — runners exist, full sweep not yet complete**

External graph learning baselines integrated via `benchmark/`.

- Registry: `benchmark/baseline_registry.yaml`
- Runner: `benchmark/run_benchmark_suite.py`
- Docs: `benchmark/docs/`

---

## Archived materials

**Status: 🗄️ Kept for provenance — do not use for paper claims**

| Location | Description |
|----------|-------------|
| `reports/archive/` | Superseded exports and older result files |
| `archive/legacy_venue_specific/` | Materials from earlier venue-specific submission cycle |
| `code/bfsbased_node_classification/experimental_archived/` | Legacy diagnostic scripts |

---

## Summary

| Package | Label | Evidence state |
|---------|-------|---------------|
| Reliability-gated SGC | **FINAL_V3** | ✅ Frozen canonical |
| Uncertainty-gated SGC | **UG-SGC** | 📁 Legacy / supplementary |
| Structural extension | **UG-SGC-S** | 🔬 Exploratory / partial |
| GCN/APPNP baselines | **baselines** | ✅ Complete |
| Benchmark suite | **benchmark** | 🔧 Integration-ready |
| Superseded exports | — | 🗄️ Archived |

# Uncertainty-Gated Selective Graph Correction for Node Classification

**Canonical method:** `FINAL_V3` — reliability-gated selective graph correction on top of a strong MLP  
**Canonical claims and result files:** [`CANONICAL_CLAIMS.md`](CANONICAL_CLAIMS.md)  
**Full reproduction guide:** [`scripts/README_REPRODUCE.md`](scripts/README_REPRODUCE.md)  
**Repository status and classification:** [`REPO_STATUS.md`](REPO_STATUS.md)

> **What this is.** A frozen evidence package for a research paper proposing a
> conservative, MLP-first graph correction strategy. The method identifies
> low-reliability nodes via an MLP confidence score and applies selective
> feature/graph correction only to those nodes. It does **not** globally replace
> the MLP with graph propagation and does not claim universal heterophily
> robustness.

---

## Quick start

**Regenerate canonical tables and figures (no model training, ~5 s):**

```bash
bash scripts/run_all_selective_correction_results.sh
```

**Full benchmark re-run (tagged — does NOT overwrite canonical frozen files):**

```bash
python3 code/bfsbased_node_classification/run_final_evaluation.py \
  --split-dir data/splits \
  --datasets cora citeseer pubmed chameleon texas wisconsin \
  --output-tag rerun_$(date +%Y%m%d)
```

> ⚠️ Always pass `--output-tag` when re-running experiments to avoid
> overwriting frozen evidence. See [`scripts/README_REPRODUCE.md`](scripts/README_REPRODUCE.md).

---

## Canonical artifacts

| Purpose | Path |
|---------|------|
| Frozen per-split log (10 splits × 6 datasets) | `reports/final_method_v3_results.csv` |
| **Main results table** | `tables/main_results_selective_correction.md` · `.csv` |
| Experimental setup table | `tables/experimental_setup_selective_correction.md` |
| Ablation table | `tables/ablation_selective_correction.md` |
| Sensitivity table | `tables/sensitivity_selective_correction.md` |
| Graphical abstract | `figures/graphical_abstract_selective_correction_v3.{png,pdf,svg}` |
| Safety / harmful-split summary | `reports/safety_analysis.md` |
| Method analysis | `reports/final_method_v3_analysis.md` |
| Canonical implementation | `code/final_method_v3.py` (stable import) |

---

## Repository map

```
bfsbased_node_classification/
│
├── code/
│   ├── final_method_v3.py                  ← stable canonical import
│   └── bfsbased_node_classification/
│       ├── final_method_v3.py              ← FINAL_V3 implementation  [CANONICAL]
│       ├── run_final_evaluation.py         ← 10-split benchmark driver [CANONICAL]
│       ├── gcn_baseline_runner.py          ← GCN Table 1 baseline      [CANONICAL]
│       ├── analyze_final_v3_regimes.py     ← regime analysis           [CANONICAL]
│       ├── build_final_v3_regime_analysis.py                           [CANONICAL]
│       ├── manuscript_runner.py            ← UG-SGC runner             [LEGACY]
│       ├── resubmission_runner.py          ← UG-SGC-S + baselines      [EXPLORATORY]
│       ├── pairnorm_baseline_runner.py     ← standalone DeepGCN+PairNorm baseline
│       ├── fsgnn_baseline_runner.py        ← standalone FSGNN baseline
│       ├── gprgnn_baseline_runner.py       ← standalone GPRGNN baseline
│       ├── standard_node_baselines.py      ← GCN/APPNP + external baseline helpers
│       ├── triple_trust_sgc.py             ← experimental triple-trust (opt-in)
│       ├── run_triple_trust_experiments.py ← triple-trust runner
│       ├── bfsbased-full-investigate-homophil.py  ← core module (all variants)
│       ├── make_splits.py                  ← split generation utility
│       └── experimental_archived/          ← legacy diagnostic scripts
│
├── data/splits/                    ← pre-bundled 60/20/20 splits (.npz)
│
├── reports/
│   ├── final_method_v3_results.csv ← CANONICAL FROZEN LOG (do not edit)
│   ├── final_method_v3_analysis.md
│   ├── final_v3_regime_analysis.md
│   ├── safety_analysis.md
│   └── archive/                    ← superseded exports
│
├── tables/                         ← all canonical tables
├── figures/                        ← all canonical figures
├── logs/                           ← canonical run logs (FINAL_V3 + GCN baseline)
├── scripts/
│   ├── run_all_selective_correction_results.sh  ← one-command canonical refresh
│   ├── README_REPRODUCE.md
│   └── build_artifacts/
│
├── archive/                        ← all legacy / exploratory / historical material
│   ├── legacy_venue_specific/      ← UG-SGC and UG-SGC-S evidence from earlier submission
│   ├── legacy_code/                ← superseded code (BFS runner, comparison runner, etc.)
│   ├── exploratory_code/           ← experimental scripts not part of main claim
│   ├── legacy_reports/             ← diagnostic and exploratory reports
│   ├── legacy_logs/                ← UG-SGC per-run logs and old analysis files
│   ├── exploratory/                ← benchmark baseline suite and external baselines
│   └── legacy_material/            ← run metadata, HPC output, broken job backups
│
├── slurm/                          ← HPC job templates (Wulver)
├── CANONICAL_CLAIMS.md             ← what the paper claims and what files support it
├── REPO_STATUS.md                  ← canonical / legacy / exploratory inventory
├── ANALYSIS_GUIDE.md               ← reviewer reading guide
└── CONTRIBUTING.md
```

---

## Method lines in this repository

| Line | Label | Status | Main entry |
|------|-------|--------|-----------|
| Reliability-gated selective correction | **FINAL_V3** | ✅ Canonical / frozen | `code/final_method_v3.py` |
| Uncertainty-gated selective correction | **UG-SGC** | 📁 Legacy | `code/bfsbased_node_classification/manuscript_runner.py` |
| Structural extension | **UG-SGC-S** | 🔬 Exploratory | `code/bfsbased_node_classification/resubmission_runner.py` |

External baselines (e.g., PairNorm, FSGNN, GPRGNN) are included for comparison only and are not
part of canonical `FINAL_V3`.

See [`REPO_STATUS.md`](REPO_STATUS.md) for a complete inventory.

---

## What not to confuse

- **`FINAL_V3` ≠ `UG-SGC`** — FINAL_V3 uses a reliability gate; UG-SGC uses
  uncertainty/confidence thresholding. Only FINAL_V3 is the canonical
  submission-facing package.
- **`tables/main_results_selective_correction.*`** is the canonical main table.
  Older UG-SGC tables live in `archive/legacy_venue_specific/`.
- **`archive/legacy_venue_specific/results_prl/`** is a threshold-sensitivity
  cross-run package from the UG-SGC line, not FINAL_V3.
- **`archive/legacy_venue_specific/reports_resubmission_protocols/`** and
  **`archive/legacy_venue_specific/tables_prl_resubmission/`** are the structural
  extension (UG-SGC-S) exploratory package.

---

## Dependencies

```bash
pip install -r requirements.txt
```

Table/figure regeneration (`bash scripts/run_all_selective_correction_results.sh`)
only needs NumPy and Matplotlib — no PyTorch required.

---

## License

[MIT License](LICENSE)

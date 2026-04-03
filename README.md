# Uncertainty-Gated Selective Graph Correction for Node Classification

**Paper target:** Pattern Recognition Letters (short paper)  
**Canonical method:** `FINAL_V3` — reliability-gated selective graph correction on top of a strong MLP  
**Best entry point for external readers:** [`ANALYSIS_GUIDE.md`](ANALYSIS_GUIDE.md)  
**Full reproduction guide:** [`scripts/README_REPRODUCE.md`](scripts/README_REPRODUCE.md)

> **What this is.** An evidence package for a short PRL submission proposing a
> conservative, MLP-first graph correction strategy. The method identifies
> low-reliability nodes via an MLP confidence score and applies selective
> feature/graph correction only to those nodes. It is **not** a claim of
> universal heterophily robustness, and it does not globally replace the MLP
> with graph propagation.

---

## Quick start

**Regenerate canonical tables and figures (no model training, ~5 s):**

```bash
bash scripts/run_all_prl_results.sh
```

**Smoke test (Cora, split 0 only, ~2 s):**

```bash
python3 code/bfsbased_node_classification/manuscript_runner.py \
  --split-dir data/splits --datasets cora --splits 0 --repeats 1 \
  --methods mlp_only selective_graph_correction --output-tag smoke_test
```

**Full benchmark re-run (writes to a tagged output — does NOT overwrite
canonical frozen files):**

```bash
python3 code/bfsbased_node_classification/run_final_evaluation.py \
  --split-dir data/splits \
  --datasets cora citeseer pubmed chameleon texas wisconsin \
  --output-tag rerun_$(date +%Y%m%d)
```

> ⚠️ Always pass a unique `--output-tag` when re-running experiments to avoid
> overwriting canonical evidence files. See
> [`scripts/README_REPRODUCE.md`](scripts/README_REPRODUCE.md).

---

## Canonical artifacts

| Purpose | Path |
|---------|------|
| Frozen per-split log (10 splits × 6 datasets) | `reports/final_method_v3_results.csv` |
| **Main results table** | `tables/main_results_prl.md` · `tables/main_results_prl.csv` |
| Experimental setup table | `tables/experimental_setup_prl.md` |
| Ablation table | `tables/ablation_prl.md` |
| Sensitivity table | `tables/sensitivity_prl.md` |
| Graphical abstract | `figures/prl_graphical_abstract_v3.{png,pdf,svg}` |
| Safety / harmful-split summary | `reports/safety_analysis.md` |
| Method analysis write-up | `reports/final_method_v3_analysis.md` |
| Canonical method implementation | `code/final_method_v3.py` (stable import) |

---

## Repository map

```
bfsbased_node_classification/
│
├── code/
│   ├── final_method_v3.py              ← stable canonical import
│   └── bfsbased_node_classification/
│       ├── final_method_v3.py          ← FINAL_V3 implementation
│       ├── run_final_evaluation.py     ← 10-split benchmark driver
│       ├── manuscript_runner.py        ← MLP vs SGC runner (UG-SGC line)
│       ├── prl_resubmission_runner.py  ← GCN/APPNP baselines + ablations
│       ├── gcn_baseline_runner.py      ← standalone GCN Table 1 baseline
│       ├── standard_node_baselines.py  ← shared GCN/APPNP training helpers
│       ├── bfsbased-full-investigate-homophil.py  ← core module (all variants)
│       └── experimental_archived/     ← legacy diagnostic scripts
│
├── data/splits/                        ← pre-bundled 60/20/20 splits (.npz)
│
├── reports/
│   ├── final_method_v3_results.csv     ← CANONICAL FROZEN LOG (do not edit)
│   ├── final_method_v3_analysis.md     ← main method analysis
│   ├── safety_analysis.md
│   ├── prl_resubmission/               ← legacy resubmission/structural docs
│   ├── prl_final_additions/            ← UG-SGC line audit and analysis
│   └── archive/                        ← superseded tables and exports
│
├── tables/
│   ├── main_results_prl.{md,csv}       ← CANONICAL TABLE (regenerated from CSV)
│   ├── ablation_prl.{md,csv}
│   ├── sensitivity_prl.{md,csv}
│   ├── experimental_setup_prl.{md,csv}
│   ├── prl_final_additions/            ← UG-SGC line supplementary tables
│   └── prl_resubmission/               ← structural-upgrade line tables
│
├── figures/
│   ├── prl_graphical_abstract_v3.*     ← CANONICAL FIGURE
│   ├── correction_rate_vs_homophily.png
│   ├── safety_comparison.png
│   ├── reliability_vs_accuracy.png
│   └── prl_final_additions/            ← supplementary UG-SGC figures
│
├── logs/                               ← run logs (JSONL + CSV)
├── results_prl/                        ← threshold-sensitivity package (UG-SGC)
├── scripts/
│   ├── run_all_prl_results.sh          ← one-command canonical refresh
│   ├── README_REPRODUCE.md             ← full reproduction guide
│   └── prl_final_additions/            ← individual build scripts
│
├── slurm/                              ← HPC job templates (Wulver)
├── ANALYSIS_GUIDE.md                   ← best single entry point for reviewers
├── REPO_STATUS.md                      ← what is canonical / legacy / exploratory
├── README_PRL_MANUSCRIPT.md            ← manuscript-oriented artifact index
└── CONTRIBUTING.md                     ← contribution guidelines
```

---

## Method lines in this repository

This repository contains **three** distinct but related lines of work. Do not
mix them:

| Line | Label | Status | Main entry |
|------|-------|--------|-----------|
| Reliability-gated selective correction | **FINAL_V3** | ✅ Canonical / frozen | `code/final_method_v3.py` |
| Uncertainty-gated selective correction | **UG-SGC** | 📁 Legacy / supplementary | `code/bfsbased_node_classification/manuscript_runner.py` |
| Structural extension | **UG-SGC-S** | 🔬 Exploratory / partial | `code/bfsbased_node_classification/prl_resubmission_runner.py` |

See [`REPO_STATUS.md`](REPO_STATUS.md) for a complete inventory.

---

## What not to confuse

- **`FINAL_V3` ≠ `UG-SGC`** — FINAL_V3 uses a reliability gate; UG-SGC uses
  uncertainty/confidence thresholding. Only FINAL_V3 is the canonical
  submission-facing package.
- **`tables/main_results_prl.*` ≠ `tables/prl_final_additions/prl_benchmark_summary.csv`**
  — the latter is the older UG-SGC line; use only `main_results_prl.*` for the
  main manuscript table.
- **`results_prl/`** is a threshold-sensitivity cross-run package from the
  UG-SGC line, not FINAL_V3.
- **`reports/prl_resubmission/`** and **`tables/prl_resubmission/`** are the
  structural extension (UG-SGC-S) exploratory package.

---

## Dependencies

Install once:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric numpy networkx sortedcontainers matplotlib
```

Or use the bundled requirements file:

```bash
pip install -r requirements.txt
```

Table/figure regeneration (`bash scripts/run_all_prl_results.sh`) only needs
NumPy and Matplotlib — no PyTorch required.

---

## License

[MIT License](LICENSE)

# Node-classification code directory

This directory contains the implementation code for the research paper.

## Canonical FINAL_V3 scripts

| File | Role |
|------|------|
| `final_method_v3.py` | Core FINAL_V3 implementation |
| `../final_method_v3.py` | Stable import path (one level up) |
| `run_final_evaluation.py` | 10-split benchmark driver |
| `gcn_baseline_runner.py` | Standalone GCN baseline (Table 1) |
| `analyze_final_v3_regimes.py` | Regime analysis helper |
| `build_final_v3_regime_analysis.py` | Regime analysis builder |
| `bfsbased-full-investigate-homophil.py` | Core module (all method variants) |
| `standard_node_baselines.py` | GCN/APPNP + external baselines (incl. Begga-style heterophily set; see below) |
| `make_splits.py` | Split generation utility |

## Legacy / exploratory scripts (kept for provenance)

| File | Status | Notes |
|------|--------|------|
| `manuscript_runner.py` | 📁 **Legacy** (UG-SGC) | Original uncertainty-gated runner; not canonical FINAL_V3 |
| `resubmission_runner.py` | 🔬 **Exploratory** (UG-SGC-S) | Structural extension + GCN/APPNP baselines |
| `experimental_archived/` | 🗄️ **Archived** | Older diagnostic scripts |

Additional legacy and exploratory scripts have been moved to
`archive/legacy_code/` and `archive/exploratory_code/`.

### Original uncertainty-gated selective correction (UG-SGC)

| Artifact | Path |
|----------|------|
| Core implementation | `bfsbased-full-investigate-homophil.py` |
| Original manuscript driver | `manuscript_runner.py` |

### Structural extension / resubmission package

| Artifact | Path |
|----------|------|
| Focused resubmission runner | `resubmission_runner.py` |
| Compact GCN/APPNP baselines | `standard_node_baselines.py` |
| External H2GCN baseline (compact in-repo adaptation) | `standard_node_baselines.py` |
| External DeepGCN+PairNorm baseline (inline) | `standard_node_baselines.py` + `pairnorm_baseline_runner.py` |
| External GPRGNN baseline (inline) | `standard_node_baselines.py` + `gprgnn_baseline_runner.py` |
| External FSGNN baseline (inline) | `standard_node_baselines.py` + `fsgnn_baseline_runner.py` |
| Triple-trust experimental variant | `triple_trust_sgc.py` |
| Triple-trust dedicated runner | `run_triple_trust_experiments.py` |
| Bucket-safety diagnostics (exploratory) | `archive/exploratory_code/run_margin_bucket_safety_experiment.py` |
| Resubmission analysis | `archive/exploratory_code/analyze_resubmission_results.py` |
| Grouped-run merge helper | `archive/exploratory_code/merge_resubmission_runs.py` |
| Begga heterophily baselines (H2GCN, GCNII, Geom-GCN, LINKX, ACM-style `acmii_gcn_plus_plus`) | `standard_node_baselines.py` · [`docs/BEGGA_HETEROPHILY_BASELINES.md`](../../docs/BEGGA_HETEROPHILY_BASELINES.md) |

## Notes

- Do not mix FINAL_V3 result files with UG-SGC or UG-SGC-S artifacts.
- `bfsbased-full-investigate-homophil.py` is dynamically loaded by
  `manuscript_runner.py` — it is not directly importable due to the hyphenated
  filename and notebook-style top-level code.
- Always pass `--output-tag` when re-running experiments to avoid overwriting
  canonical evidence files.
- `bfsbased-full-investigate-homophil.py` remains the main location for the
  original `UG-SGC` logic and the structural `UG-SGC-S` extension.
- `experimental_archived/` contains older diagnostic and pre-final exploration
  scripts.
- Some older files reflect earlier heterophily-oriented wording and should not
  be treated as the current paper framing.
- `FINAL_V3` remains the canonical frozen method; experimental variants
  (`TRIPLE_TRUST_SGC`, optional low-confidence structural term) are
  supplementary and opt-in.
- `FINAL_V3` includes an optional local graph-reliability gate in
  `final_method_v3.py`. The legacy soft probability-based local agreement
  modes (`linear`, `hard_threshold`, `shifted_linear`) are preserved, and an
  explicit propagated-label local homophily family
  (`exact_local_linear`, `exact_local_hard_threshold`,
  `exact_local_shifted_linear`) is available as an opt-in upgrade path.
  Canonical defaults are unchanged unless these options are enabled.
- `GPRGNN` is included as an **external baseline** for comparison only. It is
  not part of the canonical `FINAL_V3` package.
- `H2GCN` is included as an **external baseline** for comparison only. Official
  reference: https://github.com/GemsLab/H2GCN. The in-repo code is a compact
  PyTorch adaptation (see also https://github.com/GitEventhandler/H2GCN-PyTorch)
  using this repository's split/logging protocol. **GCNII**, **Geom-GCN**
  (compact), **LINKX**, and **`acmii_gcn_plus_plus`** (ACM-GNN-style mixer) are
  integrated the same way; see [`docs/BEGGA_HETEROPHILY_BASELINES.md`](../../docs/BEGGA_HETEROPHILY_BASELINES.md).
  None of these are part of the canonical `FINAL_V3` package.
- `FSGNN` is included as an **external baseline** for comparison only. It is
  not part of the canonical `FINAL_V3` package.
- `PairNorm` is included here as **DeepGCN+PairNorm** external baseline for
  comparison only. It is not part of the canonical `FINAL_V3` package.

For the full repository map see the root `README.md` and `REPO_STATUS.md`.

For manuscript-oriented artifact notes (historical / superseded top-level doc),
see [`archive/legacy_docs/README_MANUSCRIPT_ARTIFACTS.md`](../../archive/legacy_docs/README_MANUSCRIPT_ARTIFACTS.md) and
`archive/legacy_venue_specific/reports_resubmission_protocols/`.

License: [MIT](../../LICENSE)

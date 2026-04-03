# Node-classification code directory

This directory contains the main implementation code used by the current research paper
repository workflow. It now includes both the frozen submission package and the
newer resubmission / structural-extension line.

## Main method lines

### 1. Frozen submission method: `FINAL_V3`

| Artifact | Path |
|----------|------|
| Implementation | `final_method_v3.py` |
| Stable import from `code/` | `../final_method_v3.py` |
| Benchmark runner | `run_final_evaluation.py` |

`FINAL_V3` is the cleanest frozen submission-facing package currently available
on `main`.

### 2. Original uncertainty-gated selective correction

| Artifact | Path |
|----------|------|
| Core implementation | `bfsbased-full-investigate-homophil.py` |
| Original manuscript driver | `manuscript_runner.py` |

### 3. Structural extension / resubmission package

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
| Canonical bucket-safety runner | `run_margin_bucket_safety_experiment.py` |
| Resubmission analysis | `analyze_prl_resubmission.py` |
| Resubmission analysis (alt) | `analyze_resubmission_results.py` |
| Grouped-run merge helper | `merge_prl_resubmission_runs.py` |
| Grouped-run merge helper (alt) | `merge_resubmission_runs.py` |

## Notes

- `bfsbased-full-investigate-homophil.py` remains the main location for the
  original `UG-SGC` logic and the structural `UG-SGC-S` extension.
- `experimental_archived/` contains older diagnostic and pre-final exploration
  scripts.
- Some older files reflect earlier heterophily-oriented wording and should not
  be treated as the current paper framing.
- `FINAL_V3` remains the canonical frozen PRL method; experimental variants
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
- `H2GCN` is included as an **external baseline** for comparison only. Its
  in-repo implementation is a compact PyTorch adaptation aligned with
  https://github.com/GitEventhandler/H2GCN-PyTorch (and conceptually with the
  original H2GCN work), while reusing this repository's split/logging protocol.
  It is not part of the canonical `FINAL_V3` package.
- `FSGNN` is included as an **external baseline** for comparison only. It is
  not part of the canonical `FINAL_V3` package.
- `PairNorm` is included here as **DeepGCN+PairNorm** external baseline for
  comparison only. It is not part of the canonical `FINAL_V3` package.

For the manuscript-facing map, prefer:

- repository root `README.md`
- `README_MANUSCRIPT_ARTIFACTS.md`
- `archive/legacy_venue_specific/reports_resubmission_protocols/`

License: [MIT](../../LICENSE)

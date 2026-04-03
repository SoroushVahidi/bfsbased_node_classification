# Node-classification code directory

This directory contains the main implementation code used by the current PRL
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

### 3. Structural extension / PRL resubmission package

| Artifact | Path |
|----------|------|
| Focused resubmission runner | `prl_resubmission_runner.py` |
| Compact GCN/APPNP baselines | `standard_node_baselines.py` |
| Triple-trust experimental variant | `triple_trust_sgc.py` |
| Triple-trust dedicated runner | `run_triple_trust_experiments.py` |
| Resubmission analysis | `analyze_prl_resubmission.py` |
| Grouped-run merge helper | `merge_prl_resubmission_runs.py` |

## Notes

- `bfsbased-full-investigate-homophil.py` remains the main location for the
  original `UG-SGC` logic and the structural `UG-SGC-S` extension.
- `experimental_archived/` contains older diagnostic and pre-final exploration
  scripts.
- Some older files reflect earlier heterophily-oriented wording and should not
  be treated as the current paper framing.

For the manuscript-facing map, prefer:

- repository root `README.md`
- `README_PRL_MANUSCRIPT.md`
- `reports/prl_resubmission/`

License: [MIT](../../LICENSE)

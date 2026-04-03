# TRIPLE_TRUST_SGC (experimental)

## What it is

`TRIPLE_TRUST_SGC` is an experimental feature-first selective correction variant that adds three explicit trust components:

1. **Class trust** (`tau_c`) from trusted mass per class.
2. **Source-node trust** (`s_u`) to weight neighbor contributions.
3. **Target-node labelability** (`ell_v`) to gate risky corrections.

It keeps the base MLP and selective correction structure and does **not** change canonical `FINAL_V3` defaults.

## Core score

For class `c` at node `v`:

`S(v,c) = b1*log p_mlp(v,c) + b2*D(v,c) + b4*N_s(v,c) + b5*C_s(v,c)`

where:
- `D`: trust-aware prototype similarity,
- `N_s`: trust-weighted neighbor support,
- `C_s`: trust-aware compatibility support.

## Added parameters

- `alpha_class`
- `gamma_source`
- `lambda_deg`
- `lambda_proto`
- `tau_uncertain`
- `rho_target`
- `kappa_update`
- `q_weights` (sum normalized to 1)
- `refresh_fraction`
- `compatibility_update_eta`
- `enable_target_labelability_gate`

Conservative defaults are used in `triple_trust_sgc.py`.

## Method names / where integrated

- Method key: `triple_trust_sgc`
- Integrated into: `code/bfsbased_node_classification/prl_resubmission_runner.py`
- Included in analysis display map: `code/bfsbased_node_classification/analyze_prl_resubmission.py`

## Outputs

Runs are logged in the same resubmission JSONL stream under the selected `--output-tag`, with additional trust diagnostics fields:

- `tt_mean_q_corrected`
- `tt_mean_source_trust_pseudo`
- `tt_average_target_labelability`
- `tt_pseudo_update_eligible_count`
- `tt_class_trust_vector`
- `tt_per_class_trusted_mass`
- `tt_trust_neighbor_concentration_mean`
- `tt_compatibility_update_stats`
- `tt_n_helped`, `tt_n_hurt`

## How to run

### Smoke test (single dataset/split)

```bash
python3 code/bfsbased_node_classification/run_triple_trust_experiments.py \
  --split-dir data/splits \
  --datasets cora \
  --splits 0 \
  --repeats 1 \
  --output-tag triple_trust_sgc_smoke
```

### Multi-dataset benchmark

```bash
python3 code/bfsbased_node_classification/prl_resubmission_runner.py \
  --split-dir data/splits \
  --datasets cora citeseer pubmed chameleon texas wisconsin \
  --splits 0 1 2 3 4 5 6 7 8 9 \
  --repeats 1 \
  --methods mlp_only selective_graph_correction triple_trust_sgc \
  --output-tag triple_trust_sgc_core_r1
```

# Protocol summary

- output_tag: `prl_structural_smoke_cora_s0_v1`
- runs_file: `logs/prl_resubmission/comparison_runs_prl_structural_smoke_cora_s0_v1.jsonl`
- datasets: `cora`
- methods: `mlp_only, selective_graph_correction, selective_graph_correction_structural, sgcs_far_only, sgcs_margin_gate_only, sgcs_no_far, sgcs_no_gate`
- successful records: `7` / `7`
- splits: `0`
- repeats: `0`
- selective correction and all ablations use validation-only threshold/weight selection.
- GCN and APPNP use compact validation-based config grids with early stopping.

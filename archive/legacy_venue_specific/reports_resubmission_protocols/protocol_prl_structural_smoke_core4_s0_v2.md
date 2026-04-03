# Protocol summary

- output_tag: `prl_structural_smoke_core4_s0_v2`
- runs_file: `logs/prl_resubmission/comparison_runs_prl_structural_smoke_core4_s0_v2.jsonl`
- datasets: `chameleon, citeseer, cora, pubmed`
- methods: `mlp_only, selective_graph_correction, selective_graph_correction_structural, sgcs_far_only, sgcs_margin_gate_only, sgcs_no_far`
- successful records: `24` / `24`
- splits: `0`
- repeats: `0`
- selective correction and all ablations use validation-only threshold/weight selection.
- GCN and APPNP use compact validation-based config grids with early stopping.

# Protocol summary

- output_tag: `prl_resubmission_smoke_cora_s0_v3`
- runs_file: `logs/prl_resubmission/comparison_runs_prl_resubmission_smoke_cora_s0_v3.jsonl`
- datasets: `cora`
- methods: `appnp, gcn, mlp_only, selective_graph_correction, sgc_no_gate`
- successful records: `5` / `5`
- splits: `0`
- repeats: `0`
- selective correction and all ablations use validation-only threshold/weight selection.
- GCN and APPNP use compact validation-based config grids with early stopping.

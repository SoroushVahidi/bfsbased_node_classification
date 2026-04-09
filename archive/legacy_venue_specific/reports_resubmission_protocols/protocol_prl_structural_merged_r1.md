# Protocol summary

- output_tag: `prl_structural_merged_r1`
- runs_file: `/mmfs1/home/sv96/node_classification_manuscript/logs/prl_resubmission/comparison_runs_prl_structural_merged_r1.jsonl`
- datasets: `chameleon, citeseer, cora, pubmed, wisconsin`
- methods: `appnp, gcn, mlp_only, prop_only, selective_graph_correction, selective_graph_correction_structural, sgcs_far_only, sgcs_margin_gate_only, sgcs_no_far, sgcs_no_gate`
- successful records: `462` / `462`
- splits: `0, 1, 2, 3, 4, 5, 6, 7, 8, 9`
- repeats: `0`
- selective correction and all ablations use validation-only threshold/weight selection.
- GCN and APPNP use compact validation-based config grids with early stopping.

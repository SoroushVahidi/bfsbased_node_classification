# Protocol summary

- output_tag: `prl_resubmission_core_merged_r1`
- runs_file: `/mmfs1/home/sv96/node_classification_manuscript/logs/prl_resubmission/comparison_runs_prl_resubmission_core_merged_r1.jsonl`
- datasets: `chameleon, citeseer, cora, pubmed, wisconsin`
- methods: `appnp, gated_mlp_prop, gcn, mlp_only, prop_only, selective_graph_correction, sgc_knn_enabled, sgc_mlp_plus_graph, sgc_no_compatibility, sgc_no_feature_similarity, sgc_no_gate, sgc_no_graph_neighbor`
- successful records: `600` / `600`
- splits: `0, 1, 2, 3, 4, 5, 6, 7, 8, 9`
- repeats: `0`
- selective correction and all ablations use validation-only threshold/weight selection.
- GCN and APPNP use compact validation-based config grids with early stopping.

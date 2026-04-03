# PRL insertable text

These paragraphs are written to be copied into a manuscript draft outside this
repository. They are grounded in the repository evidence only.

## Related work paragraph

Our method is most closely related to graph-based post-processing and correction methods rather than end-to-end message-passing architectures. In particular, methods such as Correct and Smooth show that graph structure can be highly effective when used as a refinement stage on top of strong initial predictions. At the same time, recent benchmark studies on heterophilic or structurally noisy graphs have emphasized that strong MLP-style baselines and careful evaluation protocols can narrow many apparent gains from graph propagation. We therefore position `FINAL_V3` as a conservative, feature-first correction layer: it begins with a strong MLP, applies graph evidence only selectively, and is designed to reduce harmful graph influence rather than to claim universal superiority over graph-learning methods.

## Experimental setup clarification paragraph

We evaluate the method on six benchmark datasets using the standard fixed GEO-GCN split protocol with 60/20/20 train/validation/test partitions and 10 splits per dataset. The canonical comparison package in this repository uses `MLP_ONLY`, baseline `SGC v1`, `V2_MULTIBRANCH`, and the proposed `FINAL_V3` method, with split-level results frozen in `reports/final_method_v3_results.csv`. All manuscript-facing tables and figures can be regenerated from the frozen CSVs without retraining via `bash scripts/run_all_prl_results.sh`, which rebuilds `tables/main_results_prl.*`, the setup/ablation/sensitivity tables, and the core figure bundle.

## Limitations / discussion paragraph

The repository evidence supports a narrow interpretation of the method. `FINAL_V3` improves consistently on the citation-style datasets and reduces harmful graph influence relative to ungated correction, but it is not a universal heterophily solution. On Chameleon the method remains close to the MLP on average, and difficult realizations such as Texas split 5 show that reliability-aware gating does not remove every failure mode. The most defensible claim is therefore that graph structure is useful as a conservative correction signal on top of a strong feature-only baseline, not that the method should replace broader graph-learning pipelines across all regimes.

## Statistical support paragraph

The frozen 10-split benchmark already supports paired split-level summaries rather than single aggregated numbers only. In particular, `FINAL_V3` is positive against the MLP on all 10 splits for Cora, Citeseer, and Pubmed, while the bootstrap confidence intervals for the mean paired delta remain clearly above zero on those datasets. By contrast, Texas and Wisconsin remain high-variance small-data regimes, and Chameleon stays effectively near-neutral, which reinforces the paper's conservative regime-aware framing.

## Ablation / gate findings paragraph

Pending completion of the current `FINAL_V3` ablation package, insert the concise
dataset-level findings from:

- `reports/prl_ablation_interpretation.md`
- `reports/prl_gate_analysis.md`
- `logs/prl_ablation_summary.csv`
- `logs/prl_gate_analysis.csv`

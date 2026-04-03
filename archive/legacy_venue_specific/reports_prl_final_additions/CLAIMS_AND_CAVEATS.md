# Safe claims and caveats (PRL)

> **Scope:** These bullets were written around **aggregated logs** from the **manuscript_runner / selective_graph_correction** evaluation (`logs/manuscript_gain_over_mlp_final_validation.csv`, etc.). They remain useful for **descriptive** statements about gating and correction fractions in that pipeline.
>
> **For the submitted method (FINAL_V3, reliability gate R(v), 10-split six-dataset benchmark),** anchor quantitative claims in `tables/main_results_prl.md`, `reports/final_method_v3_results.csv`, `reports/final_method_v3_analysis.md`, and `reports/safety_analysis.md`. Do not mix the two packages in one table without relabeling.

## Safe to claim

- Strong **MLP** baseline; in this older `UG-SGC` pipeline, graph corrections are
  applied only to **uncertain** nodes.
- On **Cora, CiteSeer, PubMed**, mean test accuracy **improves** vs MLP under the reported protocol (see `logs/manuscript_gain_over_mlp_final_validation.csv`).
- **Descriptive** alignment: higher validation-selected **τ** and larger **uncertain** fractions on positive datasets vs some near-neutral sets — from aggregated correction CSV (not a causal ablation).
- **feature-kNN** inactive (`use_feature_knn=no`) in final **aggregated** correction table.
- **Win/tie/loss** over 30 split-level comparisons per dataset — describe as **empirical stability**, not a formal significance test unless you add one.
- **Gate / threshold:** cross-run associations between selected **τ** and test uncertain/changed fractions (`results_prl/`).

## Do not claim

- Universal **heterophily** robustness or “works on all graphs.”
- **Formal component ablations** from the correction table alone.
- **Causal** effect of τ on accuracy from tertiles or correlations.
- **Threshold sensitivity** in the sense of a **complete τ sweep** on a **fixed** split (not in repo).

## Metric hygiene

- `avg_changed_fraction` (gain CSV) = fraction of **all** test nodes with changed prediction.
- `avg_changed_of_uncertain_frac` (correction CSV) = among **uncertain** nodes only — do not mix.

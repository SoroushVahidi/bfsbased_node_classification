# Safe claims and caveats (PRL)

## Safe to claim

- Strong **MLP** baseline; proposed method applies **graph corrections** only to **uncertain** nodes (gating).
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

# Numbers for manuscript text (from canonical CSVs)

> **Scope warning (read before citing):** These tables summarize the **legacy manuscript_runner / selective_graph_correction (“UG-SGC”)** pipeline over **nine datasets** and **30 split-level comparisons per dataset** from `logs/manuscript_gain_over_mlp_final_validation.csv`. They are **not** the **FINAL_V3** submission package.
>
> **For the PRL paper’s main benchmark (FINAL_V3, 10 splits, six datasets), use:** `tables/main_results_prl.md` / `tables/main_results_prl.csv` and `reports/final_method_v3_results.csv`. Narrative: `reports/final_method_story.txt`, `reports/final_method_v3_analysis.md`, `reports/safety_analysis.md`.

Source gain: `logs/manuscript_gain_over_mlp_final_validation.csv`  
Source correction: `logs/manuscript_correction_behavior_final_validation.csv`

## Benchmark (MLP → SGC, test accuracy means)

| dataset   | MLP    | SGC    | Δ (SGC−MLP) |
|-----------|--------|--------|-------------|
| actor     | 0.3439 | 0.3430 | −0.0009     |
| chameleon | 0.4620 | 0.4620 | 0.0000      |
| citeseer  | 0.6797 | 0.7049 | +0.0251     |
| cora      | 0.7074 | 0.7425 | +0.0351     |
| cornell   | 0.7189 | 0.7189 | 0.0000      |
| pubmed    | 0.8723 | 0.8841 | +0.0117     |
| squirrel  | 0.3218 | 0.3220 | +0.0002     |
| texas     | 0.7874 | 0.7883 | +0.0009     |
| wisconsin | 0.8471 | 0.8451 | −0.0020     |

## Win / tie / loss (30 split-level comparisons per dataset)

| dataset   | SGC wins | MLP wins | ties |
|-----------|----------|----------|------|
| actor     | 10       | 16       | 4    |
| chameleon | 13       | 13       | 4    |
| citeseer  | 30       | 0        | 0    |
| cora      | 30       | 0        | 0    |
| cornell   | 5        | 5        | 20   |
| pubmed    | 30       | 0        | 0    |
| squirrel  | 14       | 14       | 2    |
| texas     | 5        | 4        | 21   |
| wisconsin | 1        | 3        | 26   |

## Correction behavior (aggregated over 30 runs)

Descriptive only — not a causal ablation.

| dataset   | avg_b4 | avg_threshold_high | avg_uncertain_frac | use_feature_knn |
|-----------|--------|--------------------|--------------------|-----------------|
| cora      | 0.900  | 0.592              | 0.207              | no              |
| citeseer  | 0.900  | 0.569              | 0.212              | no              |
| pubmed    | 0.900  | 0.857              | 0.198              | no              |
| chameleon | 0.663  | 0.125              | 0.068              | no              |

## Global vs uncertain-conditioned “changed” fraction

- `avg_changed_fraction` in gain CSV = fraction of **all** test nodes predicted to change after gating/correction.
- `avg_changed_of_uncertain_frac` in correction CSV = fraction **among uncertain** only. Do not conflate.

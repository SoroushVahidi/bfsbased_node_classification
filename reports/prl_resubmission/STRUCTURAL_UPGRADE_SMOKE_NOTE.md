# Structural upgrade smoke note

## Candidate method

Working name:
- `UG-SGC-S`

Operational change relative to original UG-SGC:
- keep the same MLP-first and uncertainty-gated selective-correction design,
- retain feature similarity and compatibility evidence,
- replace the purely local graph emphasis with a **near/far structural split**:
  - `s_near(v,c)`: local one-hop class support
  - `s_far(v,c)`: confidence-weighted multi-hop diffusion support from 2+ hops

Current code path:
- `code/bfsbased_node_classification/bfsbased-full-investigate-homophil.py`
- wrapper:
  `selective_graph_correction_structural_predictclass(...)`

## Formula

For the structural variant, the class score is:

`b1 log p_mlp(v,c) + b2 s_proto(v,c) + b3 s_knn(v,c) + b4 s_near(v,c) + b5 s_compat(v,c) + b6 s_far(v,c)`

The current implementation keeps the base uncertainty gate and also supports an
optional extra structure-quality gate, but the smoke results below do **not**
currently justify keeping the extra gate as a main contribution.

## Revised smoke results

Artifact paths:
- `tables/prl_resubmission/baseline_comparison_prl_structural_smoke_core4_s0_v2.csv`
- `tables/prl_resubmission/ablation_comparison_prl_structural_smoke_core4_s0_v2.csv`
- `logs/prl_resubmission/comparison_runs_prl_structural_smoke_core4_s0_v2.jsonl`

Split-0 smoke comparison on core datasets:

| Dataset | MLP | UG-SGC | UG-SGC-S | Delta vs UG-SGC |
|---------|-----|--------|----------|-----------------|
| Cora | 0.6861 | 0.7284 | 0.7304 | +0.0020 |
| CiteSeer | 0.6667 | 0.6877 | 0.6907 | +0.0030 |
| PubMed | 0.8666 | 0.8803 | 0.8780 | -0.0023 |
| Chameleon | 0.4627 | 0.4627 | 0.4671 | +0.0044 |

## Ablation reading from the revised smoke run

- `s_far` appears useful on `Cora` and `Chameleon`.
- `s_near` alone underperforms the combined structural variant on all four checked datasets.
- `s_far` alone is not consistently enough; the full structural variant is stronger than pure near-only or pure far-only on the positive smoke cases.
- The **extra structure-aware gate** did not help in the smoke runs:
  - `sgcs_margin_gate_only` matched or slightly beat the version with the extra quality gate.

## Current assessment

The structural upgrade looks **plausible but not yet decisively stronger**:
- positive smoke signal on `Cora`, `CiteSeer`, and `Chameleon`
- negative smoke signal on `PubMed`

That means the Wulver grouped runs are necessary before deciding whether this is:
- a real upgraded method worth promoting in the paper, or
- an interesting but mixed extension that should remain an ablation/optional note.

## Safe current takeaway

- The most defensible added contribution so far is the **interpretable far-structural support term**.
- The extra **structure-aware gate** is currently not supported as an important improvement.

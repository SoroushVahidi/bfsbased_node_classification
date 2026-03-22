# Propagation-only baseline definition

This note defines the `prop_only` baseline that appears in the resubmission
package.

## Exact code path

- Tuning / evaluation wrapper:
  - `code/bfsbased_node_classification/prl_resubmission_runner.py`
- Core algorithm:
  - `code/bfsbased_node_classification/bfsbased-full-investigate-homophil.py`
  - function: `predictclass(...)`

## Exact algorithm

The propagation-only baseline is obtained by calling:

- `predictclass(..., mlp_probs=None)`

with eight propagation coefficients `a1..a8` selected by:

- `robust_random_search(mod.predictclass, ..., mlp_probs=None)`

This means:
- the baseline does **not** use MLP soft predictions as a seed
- it relies on the propagation procedure defined inside `predictclass`
- unlabeled nodes are not initialized from the learned MLP output

## What inputs it uses

It uses:
- graph structure
- train labels / label information available through the propagation routine
- tuned propagation parameters `a1..a8`

It does **not** use:
- the `FINAL_V3` reliability score `R(v)`
- the `FINAL_V3` uncertainty threshold `τ`
- the `FINAL_V3` reliability threshold `ρ`
- the `FINAL_V3` score-combination rule over `b1, b2, b4, b5`

## How it differs from FINAL_V3

`prop_only` is a graph-propagation baseline.

`FINAL_V3` is:
- feature-first
- anchored on a trained MLP
- selective rather than global
- gated by uncertainty and reliability

So the key distinction is:
- `prop_only` tests what graph propagation can do on its own
- `FINAL_V3` tests whether graph information helps as a conservative correction layer on top of a strong MLP

## Manuscript-facing clarity note

The baseline is clearly implemented in code, but it is **not** part of the
canonical frozen `FINAL_V3` main benchmark table in:

- `tables/main_results_prl.*`

Instead, it appears in the supplementary resubmission package:

- `tables/prl_resubmission/baseline_comparison_prl_resubmission_core_merged_r1.csv`

If the manuscript mentions propagation-only, it should define it explicitly as
the tuned `predictclass(..., mlp_probs=None)` baseline rather than assuming that
readers will infer its behavior from the name alone.

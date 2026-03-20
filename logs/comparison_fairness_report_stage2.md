# Fairness / Configuration Report – Stage 2 Internal Pilot (texas, chameleon)

## Datasets

- texas
- chameleon

## Splits and repeats

- Splits: GEO-GCN canonical splits available for all selected datasets (10 total): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- Repeats per split: 1

## Methods included

- mlp_only
- prop_only
- mlp_refined
- prop_mlp_select
- priority_bfs
- ensemble_multi_source_bfs

## Parameter sharing

- `params_prop` tuned once per dataset/split/repeat via `predictclass` (robust_random_search) and reused by:
  - prop_only
  - priority_bfs
  - ensemble_multi_source_bfs
- `params_hyb` tuned separately for `mlp_refined`.
- MLP hyperparameters are specific to `mlp_only`.

## Runtime accounting policy

- `tuning_runtime_sec` measures the time spent in hyperparameter search or training (for MLP).
- `method_runtime_sec` measures the time for the final evaluation calls (val/test) using tuned parameters.
- `total_runtime_sec = tuning_runtime_sec + method_runtime_sec`.
- For methods that reuse parameters (e.g., priority_bfs, ensemble_multi_source_bfs), the original tuning cost is reported again so that
  comparisons can be made either including or excluding amortized tuning cost.

## Possible unfairness / caveats

- Parameter reuse: `params_prop` tuned on `predictclass` and reused by `prop_only`, `priority_bfs`, and `ensemble_multi_source_bfs`.
- Experimental methods: `priority_bfs` and `ensemble_multi_source_bfs` are more recent, complex propagation variants.
- Hybrid method: `mlp_refined` uses MLP pseudo-labels to augment the train set before propagation.
- Runtime: ensemble multi-source BFS incurs extra runtime by design due to multiple restarts.
- Order sensitivity: propagation methods remain order-sensitive due to priority-queue traversal and sequential commits.

## Methods excluded or merged

- `multi_source_priority_bfs_predictclass` currently uses the same seed set (`seed_mode="all_train"`) as `priority_bfs`.
  It is therefore **not reported as a separate row**; it is treated as a configuration wrapper around priority-BFS.

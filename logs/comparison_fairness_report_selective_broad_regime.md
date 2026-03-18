# Fairness / Configuration Report – selective_broad_regime

## Datasets

- texas
- wisconsin
- cornell
- actor
- chameleon
- cora
- citeseer

## Candidate dataset coverage and exclusions

- Candidate datasets requested: ['texas', 'wisconsin', 'cornell', 'actor', 'squirrel', 'chameleon', 'cora', 'citeseer', 'pubmed']
- Included datasets executed: ['texas', 'wisconsin', 'cornell', 'actor', 'chameleon', 'cora', 'citeseer']
- Excluded datasets and reasons:
  - squirrel: excluded for runtime feasibility in this environment (large graph; projected long propagation-search runtime)
  - pubmed: excluded for runtime feasibility in this environment (large graph; projected long propagation-search runtime)

- Split coverage status:
  - texas: has_complete_0_9=True, available_split_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  - wisconsin: has_complete_0_9=True, available_split_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  - cornell: has_complete_0_9=True, available_split_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  - actor: has_complete_0_9=True, available_split_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  - squirrel: has_complete_0_9=True, available_split_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  - chameleon: has_complete_0_9=True, available_split_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  - cora: has_complete_0_9=True, available_split_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  - citeseer: has_complete_0_9=True, available_split_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  - pubmed: has_complete_0_9=True, available_split_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

## Splits and repeats

- Splits: GEO-GCN canonical splits available for all selected datasets (10 total): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- Repeats per split: 1

## Split policy

- Random split fallback is disabled for this benchmark setup.
- Datasets without complete canonical split coverage are excluded.

## Methods included

- mlp_only
- prop_only
- selective_graph_correction

## Parameter sharing

- `params_prop` tuned once per dataset/split/repeat via `predictclass` (robust_random_search) and reused by:
  - prop_only
- `selective_graph_correction` selects uncertainty threshold and score weights on validation nodes only.
- MLP hyperparameters are specific to `mlp_only`.

## Runtime accounting policy

- `tuning_runtime_sec` measures the time spent in hyperparameter search or training (for MLP).
- `method_runtime_sec` measures the time for the final evaluation calls (val/test) using tuned parameters.
- `total_runtime_sec = tuning_runtime_sec + method_runtime_sec`.
- For methods that reuse parameters (e.g., priority_bfs, ensemble_multi_source_bfs), the original tuning cost is reported again so that
  comparisons can be made either including or excluding amortized tuning cost.

## Possible unfairness / caveats

- Parameter reuse: `params_prop` tuned on `predictclass` and reused by propagation-family variants.
- Order sensitivity: propagation methods remain order-sensitive due to priority-queue traversal and sequential commits.

- Selective correction risk: validation-tuned threshold/weights may overfit small validation sets on some splits.

## Methods excluded or merged

- `multi_source_priority_bfs_predictclass` currently uses the same seed set (`seed_mode="all_train"`) as `priority_bfs`.
  It is therefore **not reported as a separate row**; it is treated as a configuration wrapper around priority-BFS.

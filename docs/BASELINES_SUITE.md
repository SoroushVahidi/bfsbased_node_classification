# Baseline suite (methods, references, fidelity)

Part of the repo doc set: see [`INDEX.md`](INDEX.md) for all documentation links and [`../AGENTS.md`](../AGENTS.md) for how runners fit together.

This document summarizes graph and post-hoc baselines exposed through `standard_node_baselines.run_baseline` and `resubmission_runner.py`. Training uses the same split files as the rest of the repository (`data/splits/…`); seeds are set via `_set_seed` / runner seeds.

| Method name (`run_baseline` / CLI) | Paper | Official code | Implementation here | Typical datasets |
|-----------------------------------|-------|---------------|---------------------|------------------|
| `mlp_only` | — | — | Manuscript module (`train_mlp_and_predict`) | All with prepared splits |
| `gcn`, `appnp`, … | Various | Various | In `standard_node_baselines.py` | All |
| `sgc` / `sgc_wu2019` | Wu et al., *Simplifying Graph Convolutional Networks*, ICML 2019 · [arXiv:1902.07153](https://arxiv.org/abs/1902.07153) | [Tiiiger/SGC](https://github.com/Tiiiger/SGC) | **Lightweight in-repo:** precompute \(S^K X\) with symmetric normalized \(S\), linear classifier; lr/wd grid as in repo | All; very large graphs are fine (linear + sparse mm) |
| `correct_and_smooth` / `cs` | Huang et al., *Combining Label Propagation and Simple Models…*, 2020 · [arXiv:2010.13993](https://arxiv.org/abs/2010.13993) | [CUAI/CorrectAndSmooth](https://github.com/CUAI/CorrectAndSmooth) | **Adapted:** 2-layer MLP base (same style as other baselines), then autoscale residual correction + outcome smoothing; recurrence matches `outcome_correlation.py` (personalized propagation + clamps); adjacency uses `_normalize_adj` (symmetric normalized + self-loops). Deviations: no `torch_sparse`/`scipy` path—uses `torch.sparse.mm` only | Through `num_nodes` ≤ 500k safety threshold (see code); fails gracefully with `RuntimeError` above that |
| `clp` | Zhong et al., *Graph Neural Networks Meet Personalized PageRank*, TMLR 2022 · [OpenReview](https://openreview.net/forum?id=JBuCfkmKYu) | No stable official release referenced in-repo | **Faithful reimplementation (recipe-level):** MLP base → class compatibility matrix \(H\) from **training** edge label co-occurrences (row-normalized) → iterations \(Z \leftarrow (1-\alpha)Z_{\text{base}} + \alpha \,\mathrm{row\text{-}norm}(S Z H)\) with row-normalized \(S\) | Same node budget note as C&S |
| `graphsage` | Hamilton et al., *Inductive Representation Learning on Large Graphs*, NeurIPS 2017 · [arXiv:1706.02216](https://arxiv.org/abs/1706.02216) | [GraphSAGE](https://snap.stanford.edu/graphsage/) | **In-repo via PyG:** two `SAGEConv(..., aggr="mean")` layers; undirected `edge_index`; hyperparameter grid aligned with `_gcn_grid` | Full-batch only: `num_nodes` ≤ 500k; not intended for `ogbn-products`-scale full-batch training |

## Aliases

- `sgc` → same as `sgc_wu2019`.
- `cs` → same as `correct_and_smooth`.

## CLI smoke examples

From the repo root (adjust `--split-dir` if needed):

```bash
cd code/bfsbased_node_classification
python3 resubmission_runner.py --datasets cora --splits 0 --methods sgc --output-tag smoke_sgc
python3 resubmission_runner.py --datasets cora --splits 0 --methods clp --output-tag smoke_clp
python3 resubmission_runner.py --datasets cora --splits 0 --methods correct_and_smooth --output-tag smoke_cs
python3 resubmission_runner.py --datasets cora --splits 0 --methods graphsage --output-tag smoke_sage
```

Extended benchmarks (`ogbn-arxiv`, …): use split_id `0` and prepared `.npz` per `docs/DATASETS_EXTENDED.md`. GraphSAGE and the post-hoc methods may refuse very large `num_nodes` (see table).

## Wulver / `baseline_comparison_suite.py`

Pass additional baseline keys via `--external-baselines`, e.g. `clp`, `correct_and_smooth`, `graphsage`, `sgc`. Display names for some methods are mapped in the suite script.

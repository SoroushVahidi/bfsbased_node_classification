# Node classification experiments (BFS-based / selective graph correction)

Implementation of **feature-first, reliability-aware selective graph correction** for node classification. The repository root [README.md](../../README.md) describes layout, licensing, and reproduction.

## FINAL_V3 (PRL submission method)

| Artifact | Path |
|----------|------|
| Implementation | `final_method_v3.py` |
| Stable import from `code/` | `../final_method_v3.py` |
| Benchmark runner (MLP, SGC v1, V2 ablation, FINAL_V3) | `run_final_evaluation.py` |

**Archived exploration** (diagnostics, early gated variants, improvement sweeps): `experimental_archived/`

**Baselines / ablations:** `improved_sgc_variants.py` (SGC v1, V2_MULTIBRANCH and other experimental tags — only v1/v2 are used in the main FINAL_V3 table).

**Core study code** is still loaded dynamically from `bfsbased-full-investigate-homophil.py` (see root `AGENTS.md`).

**Earlier publication line:** [arxiv.org/abs/2512.22221](https://arxiv.org/abs/2512.22221) — interpretable node classification on heterophilic benchmarks.

**License:** [MIT](../../LICENSE) (copy also in this directory as `LICENSE`).

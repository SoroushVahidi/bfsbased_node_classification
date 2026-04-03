# Documentation index

Use this page to find the right doc quickly. **Start here if you are an AI agent:** read [`../AGENTS.md`](../AGENTS.md) first, then return here for deep links.

| Document | Purpose |
|----------|---------|
| [`../AGENTS.md`](../AGENTS.md) | **Onboarding for coding agents:** entry points, task routing, frozen files, gotchas |
| [`../README.md`](../README.md) | Project summary, quick start, repository map |
| [`../REPO_STATUS.md`](../REPO_STATUS.md) | Canonical vs legacy vs exploratory; what not to edit |
| [`../CANONICAL_CLAIMS.md`](../CANONICAL_CLAIMS.md) | What the paper claims and which artifacts support each claim |
| [`../ANALYSIS_GUIDE.md`](../ANALYSIS_GUIDE.md) | Reviewer-oriented reading order for results |
| [`../CONTRIBUTING.md`](../CONTRIBUTING.md) | Ground rules for changes (frozen outputs, output tags) |
| [`scripts/README_REPRODUCE.md`](../scripts/README_REPRODUCE.md) | Full reproduction and benchmark re-run instructions |
| [`BASELINES_SUITE.md`](BASELINES_SUITE.md) | Baseline method names, papers, CLI keys, fidelity vs official code |
| [`DATASETS_EXTENDED.md`](DATASETS_EXTENDED.md) | `ogbn-arxiv`, `ogbn-products`, `hm-categories`, split conventions |
| [`BEGGA_HETEROPHILY_BASELINES.md`](BEGGA_HETEROPHILY_BASELINES.md) | GCNII, Geom-GCN, LINKX, ACM-style baselines |
| [`DJGNN_INTEGRATION.md`](DJGNN_INTEGRATION.md) | Notes on DJGNN-related integration (if applicable) |
| [`../archive/README.md`](../archive/README.md) | Inventory of archived legacy and exploratory material |

## Code directories (short)

| Path | Role |
|------|------|
| `code/bfsbased_node_classification/` | All runnable experiment and baseline code (add new scripts here) |
| `code/final_method_v3.py` | Thin re-export; stable import path for FINAL_V3 |
| `scripts/` | Shell drivers, `check_canonical.py`, table builders |
| `data/splits/` | Pre-bundled train/val/test `.npz` splits (when present in checkout) |
| `archive/` | Historical code, logs, venue-specific tables — not for new canonical claims |

# Phase 2 — Gap analysis (labels)

| ID | Topic | Label | Notes |
|----|-------|-------|-------|
| A | Main benchmark table | **EXISTS AND SUFFICIENT** | `logs/manuscript_gain_over_mlp_final_validation.csv` + `tables/prl_final_additions/prl_benchmark_summary.csv` |
| B | Win/tie/loss stability | **EXISTS AND SUFFICIENT** | 30 comparisons/dataset in gain CSV |
| C | Score weighting / correction behavior | **EXISTS AND SUFFICIENT** | `manuscript_correction_behavior_final_validation.csv`; feature-kNN **no** in final aggregates |
| D | Results §3 (gate / threshold) | **EXISTS BUT NEEDS CLEANUP** | Cross-run evidence is strong; **do not** title “threshold sensitivity” without caveat — see `PRL_RESULTS_SUBSECTION_3.md` |
| E | Qualitative examples | **EXISTS BUT NEEDS CLEANUP** | One split per dataset in case-study MD — **illustration only** |
| F | Final figure set | **EXISTS** | 1–2 main figs + optional `results_prl/` supplement |
| G | Final table set | **EXISTS** | 1 benchmark + optional small correction/gate block |
| H | Reproducibility | **EXISTS BUT NEEDS CLEANUP** | JSONL + `manuscript_runner.py`; pin env at submission if required |

## Missing (expensive or not worth PRL)

| Item | Label |
|------|--------|
| True per-τ sweep on fixed split with logged val/test curves | **MISSING AND WOULD REQUIRE EXPENSIVE RUNS** + instrumentation |
| Formal component ablations | **NOT WORTH INCLUDING FOR PRL** unless precomputed — do not fake |
| Broad new benchmarks | **NOT WORTH INCLUDING FOR PRL** |

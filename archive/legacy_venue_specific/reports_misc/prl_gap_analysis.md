# PRL gap analysis

This gap analysis is based on the repository contents only.

## Executive judgment

The repository already supports a **narrow PRL claim** if the manuscript stays
centered on:

- a strong MLP baseline
- reliability-aware selective graph correction
- reduced harmful graph influence
- conservative, regime-aware behavior rather than universal superiority

The biggest risks were not broad benchmark scale, but missing
`FINAL_V3`-specific manuscript-support artifacts:

- direct gate-usefulness evidence
- current-code ablation support
- compact statistical support
- clearer propagation-baseline definition
- related-work guidance in the absence of a tracked manuscript/bibliography

## 1. Closest missing comparison papers in related work

Because there is no tracked manuscript or `.bib` file in this repo, the main
related-work risk is **missing positioning rather than missing experiments**.

Closest paper families that should be acknowledged explicitly:

- `Correct and Smooth`
- strong-MLP / "when graph learning is actually needed" style papers
- one uncertainty-aware graph-learning paper if the manuscript keeps the
  uncertainty framing

Repo-based conclusion:
- these are **not verifiably covered** inside this repository
- this is a writing/positioning gap, not an experimental gap

## 2. Are current experiments enough for a narrow PRL claim?

### Yes, for the core claim

The frozen `FINAL_V3` package already supports:
- 6 datasets
- 10 fixed splits
- MLP baseline
- baseline `SGC v1`
- predecessor `V2_MULTIBRANCH`
- safety analysis against harmful graph influence
- behavior analysis showing conservative correction rates on difficult datasets

Supporting files:
- `reports/final_method_v3_results.csv`
- `tables/main_results_prl.*`
- `reports/final_method_v3_analysis.md`
- `reports/safety_analysis.md`

### Not enough for stronger claims

Current repo evidence is **not** enough for:
- universal or broad heterophily claims
- a strong claim that every score component has been rigorously re-ablated under the final code path
- a claim that propagation-only has been fully integrated into the canonical `FINAL_V3` main benchmark

## 3. Does controlled ablation already exist?

### Partially

What existed already:
- `tables/ablation_prl.*`
- archived `run_prl_validation.py` logic for `A1_NO_RELIABILITY`, profile-only variants, and `ρ` sensitivity
- resubmission-package ablations for the older `UG-SGC` / `selective_graph_correction` line

Why this was still weak:
- the manuscript-facing ablation table mixed one live `FINAL_V3` row with archived historical rows
- there was no dedicated `FINAL_V3` ablation CSV/report under `logs/` / `reports/`
- there was no direct manuscript-facing gate comparison file

## 4. Is “propagation-only” defined clearly enough?

### In code: yes

The propagation-only baseline is implemented via:
- `predictclass(..., mlp_probs=None)` in `bfsbased-full-investigate-homophil.py`
- tuning in `prl_resubmission_runner.py`

### In manuscript-facing package: no

It was not clearly defined in a standalone manuscript-support note, and it is
not part of the canonical frozen `FINAL_V3` benchmark table.

## 5. Does the evidence support the claim:
##    “start from a strong MLP and use graph information only as selective correction for uncertain nodes”?

### Yes, with careful wording

Strong support:
- `FINAL_V3` keeps the MLP as the base classifier
- the correction module is selective
- safety analysis shows fewer harmful splits than ungated baseline SGC
- behavior analysis shows conservative correction rates on difficult datasets

Needed caution:
- the current repo title still references the older uncertainty-gated wording
- `FINAL_V3` actually uses both uncertainty and reliability
- some threshold/gate analysis files belong to the older `UG-SGC` package, not the frozen `FINAL_V3` package

## Priority ranking

### Priority 1 — highest-value, cheapest additions

1. Add a dedicated `FINAL_V3` ablation package.
- Why: strongest scientific support gap inside the canonical paper line
- Cost: moderate, but tightly scoped and manuscript-justified

2. Add a direct gate-usefulness analysis.
- Why: central to the paper’s identity
- Cost: can be derived from the same ablation run

3. Add a compact paired statistical summary from the frozen split-level results.
- Why: cheap and strengthens quantitative reporting without changing the paper’s scope
- Cost: very low

4. Add a propagation-baseline definition note.
- Why: clarifies an already-run baseline without new computation
- Cost: very low

5. Add a related-work patch note.
- Why: repo lacks manuscript/bib source, so the writer needs explicit guidance
- Cost: very low

### Priority 2 — useful but optional

1. Regenerate missing threshold cross-run figures in `results_prl/`.
- Useful if the writer wants the older gate-behavior subsection package
- Not canonical for `FINAL_V3`

2. Add a compact writer-facing readiness note and insertable text snippets.
- High usability value
- No scientific downside

### Priority 3 — probably not worth doing before submission

1. Broad new benchmark campaigns.
- The narrow PRL claim does not require them

2. Full current-code `FINAL_V3` sensitivity sweeps over many thresholds/ρ values.
- Scientifically nice, but not necessary for the narrow paper if claims stay conservative

3. Folding GCN/APPNP/structural-upgrade lines into the main short-paper table.
- Risks widening the paper unnecessarily

## Bottom line

The biggest PRL risks are now mostly **manuscript-support clarity** and a
small amount of **focused final-method evidence**, not missing large-scale
experiments.

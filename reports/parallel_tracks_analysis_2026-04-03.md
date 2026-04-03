# Parallel analysis: model improvement + measurable strengths (2026-04-03)

## A. Canonical method/path in the repo

- **Canonical PRL method** is `FINAL_V3` (reliability-gated selective graph correction), implemented in `code/bfsbased_node_classification/final_method_v3.py` and evaluated by `code/bfsbased_node_classification/run_final_evaluation.py` on 10 GEO-GCN splits.  
- The `FINAL_V3` core loop is: MLP -> evidence build -> reliability score `R(v)` -> validation search over `(profile, tau, rho)` -> selective correction only for uncertain+reliable nodes.  
- Frozen canonical results are in `reports/final_method_v3_results.csv`; manuscript-facing table uses `tables/main_results_prl.md`.
- Baseline/benchmark paths:
  - Canonical short-paper comparison: `MLP_ONLY`, `BASELINE_SGC_V1`, `V2_MULTIBRANCH`, `FINAL_V3` via `run_final_evaluation.py`.
  - Extended (supplementary) benchmark path: `prl_resubmission_runner.py` + `standard_node_baselines.py` for `GCN`, `APPNP`, `SGC (Wu)`, and selective-correction ablations.

### Baseline family status vs requested list

- **Already runnable in this repo now**: `MLP`, `GCN`, `APPNP`, `SGC`, plus selective-correction families (`FINAL_V3`, v1/v2, ablations).
- **Partially present / external hook exists**: `DJ-GNN` (comparison helper exists but expects external Diffusion-Jump-GNNs repo).
- **Missing / hard (not integrated here)**: `H2GCN`, `LINKX` (as baseline model), `GCNII`, `GPRGNN`, `FSGNN`, `Ordered GNN`, `ACM-GNN / ACMII-GCN++`.

---

## B. Track A: ranked model-improvement opportunities

Ranking criterion used: expected benefit -> implementation effort -> scientific value -> fit with PRL feature-first selective-correction framing.

### 1) Reliability score upgrade with **continuous agreement** + calibrated degree prior

- **Why it makes sense here**
  - Current `R(v)` uses a hard binary MLP-graph agreement term; this is brittle around near-ties.
  - Repo history shows richer reliability signals existed in earlier variants (entropy / feature-graph agreement), so this is a low-risk, in-family upgrade.
- **Likely files to change**
  - `code/bfsbased_node_classification/final_method_v3.py` (`compute_graph_reliability`, `rho_candidates`).
  - Optionally expose diagnostics in `run_final_evaluation.py`.
- **Concrete change**
  - Replace `agreement in {0,1}` with `agreement_soft = dot(mlp_probs, graph_neighbor_support)`.
  - Replace fixed `1-exp(-deg/5)` with dataset-adaptive degree scaling from train/val median degree.
  - Keep 3-signal formula and equal-weight default to preserve narrative simplicity.
- **Exact experiment**
  - `FINAL_V3_RELIAB_SOFTAGREE` vs `FINAL_V3` on 6 datasets x 10 splits using existing evaluation runner.
  - Add per-split harmful flag and corrected-node precision (already logged fields can be reused).
- **Metric confirming help**
  - Primary: mean `delta_vs_mlp` non-decrease and higher average on Texas/Wisconsin/Chameleon.
  - Safety: harmful split count <= current `FINAL_V3` (2/60).
  - Mechanistic: corrected-node precision not reduced by >0.02 absolute.

### 2) Two-stage gate: uncertainty gate + **structure-quality gate** in FINAL_V3 (already available in repo primitives)

- **Why it makes sense here**
  - Evidence builder already computes `structural_far_support` and `structural_quality`; structural gate logic already exists in legacy/selective paths.
  - This directly targets selective 1-hop vs >1-hop reliability without reframing as full propagation.
- **Likely files to change**
  - `code/bfsbased_node_classification/final_method_v3.py` (add optional quality threshold `q`).
  - Reuse helper logic already in `bfsbased-full-investigate-homophil.py`.
- **Concrete change**
  - Add optional third gate `structural_quality >= q` during correction application.
  - Validation search over `(profile, tau, rho, q)` with tiny `q` grid (e.g., `[-1.0, 0.0, 0.05, 0.1]`).
- **Exact experiment**
  - `FINAL_V3_PLUS_QGATE` vs `FINAL_V3` with same splits and seeds.
  - Stratify gain by dataset homophily bucket (high vs low).
- **Metric confirming help**
  - Fewer harmful corrections: total `n_hurt` decrease, harmful splits <= 2/60.
  - Non-inferior mean test acc on citation datasets (Cora/Citeseer/Pubmed).

### 3) Reliability-aware **score mixing** (keep gate, but down-weight graph terms near threshold)

- **Why it makes sense here**
  - Current FINAL_V3 applies full selected score once node passes gate; this creates a discontinuity at `R(v)=rho`.
  - Smooth mixing can reduce regret on borderline nodes while preserving selective design.
- **Likely files to change**
  - `code/bfsbased_node_classification/final_method_v3.py` (after `combined_scores` selection, before argmax).
- **Concrete change**
  - Define `alpha(v)=clip((R(v)-rho)/(1-rho),0,1)` for uncertain nodes.
  - Final logits for uncertain nodes: `(1-alpha)*log(mlp_probs)+alpha*combined_scores`.
- **Exact experiment**
  - `FINAL_V3_SOFTMIX` vs `FINAL_V3`.
  - Evaluate per-dataset corrected fraction bins (`0-5%`, `5-15%`, `>15%`).
- **Metric confirming help**
  - Lower `n_hurt` at similar `n_helped`.
  - Better correction precision on Chameleon/Pubmed where hurt counts are nontrivial.

### 4) Compatibility modeling refinement: **symmetric and shrinkage compatibility prior**

- **Why it makes sense here**
  - Current compatibility is train-train directed counts with global smoothing; small datasets (Texas/Wisconsin) can be noisy.
  - A shrinkage prior can stabilize class-pair estimates when train edges are sparse.
- **Likely files to change**
  - `code/bfsbased_node_classification/bfsbased-full-investigate-homophil.py` inside `_build_selective_correction_evidence` where `compat` is computed.
- **Concrete change**
  - Compute both directed and symmetrized compatibility; blend by data size.
  - Add shrinkage toward uniform or toward empirical class prior.
- **Exact experiment**
  - Ablation within selective correction: default compatibility vs symmetric vs shrinkage vs none.
- **Metric confirming help**
  - Improvement on low-node datasets without degrading large datasets.
  - In ablation table, compatibility removal gap should narrow less often (more robust compatibility term).

### 5) Prototype similarity robustness: **class covariance-normalized prototype similarity**

- **Why it makes sense here**
  - Current feature prototype term is cosine to class centroids; class spread differences are ignored.
  - Mahalanobis-like normalization can reduce overconfident prototype pull for broad classes.
- **Likely files to change**
  - `_update_D_weighted` path in `bfsbased-full-investigate-homophil.py` (called by evidence builder).
- **Concrete change**
  - Replace/augment cosine prototype score with diagonal-covariance normalized distance, then row-normalize.
- **Exact experiment**
  - `FINAL_V3_PROTO_DIAGCOV` vs `FINAL_V3` on full 10 splits.
- **Metric confirming help**
  - Higher corrected-node precision on Citeseer/Pubmed while keeping net gain positive.

### 6) Threshold/profile search upgrade: **dataset-specific candidate policy** and small nested-CV guard

- **Why it makes sense here**
  - Tau candidates now mix fixed + quantile values uniformly across datasets; heterophilic sets already choose very low tau.
  - Search policy aware of val-margin distribution could reduce overfitting to one split realization.
- **Likely files to change**
  - `final_method_v3.py` candidate generation section.
- **Concrete change**
  - Generate tau grid from val-margin quantiles only, clipped by dataset-adaptive floor/ceiling.
  - Add optional inner-split bootstrap tie-breaker for `(tau,rho,profile)` in tied cases.
- **Exact experiment**
  - `FINAL_V3_ADAPTIVE_TAU` vs `FINAL_V3` with same compute budget.
- **Metric confirming help**
  - Lower split-to-split variance on Texas/Wisconsin; unchanged means on citation datasets.

### 7) Controlled use of 2-hop evidence: **conditional b6 activation only when quality high**

- **Why it makes sense here**
  - The codebase already supports `b6` (structural_far_support) in score builder, but FINAL_V3 fixes `b6=0`.
  - A guarded activation preserves feature-first story while testing limited higher-order evidence.
- **Likely files to change**
  - `final_method_v3.py` weight profiles and gating logic.
- **Concrete change**
  - Add third profile with small `b6` (e.g., 0.2-0.4) only when `structural_quality >= q_hi`.
- **Exact experiment**
  - `FINAL_V3_PLUS_B6_GUARDED` vs `FINAL_V3`, plus ablation `b6 always on` to show guard necessity.
- **Metric confirming help**
  - Gain on Cora/Citeseer without increasing harmful splits on Chameleon/Texas/Wisconsin.

---

## C. Track B: ranked “better than others” aspects

Status labels:
- **Supported now** = can already be claimed from repo evidence.
- **Plausible, not yet measured directly** = likely true but needs explicit analysis scripts/tables.
- **Weak / avoid** = likely not a strong claim given available evidence.

### 1) Better safety vs ungated selective correction (high priority)

- **Why plausible**
  - FINAL_V3 uses reliability gate explicitly designed to suppress risky corrections.
- **Current evidence status**: **Supported now**.
  - Harmful splits: FINAL_V3 = 2/60 vs SGC v1 = 4/60, V2 = 4/60.
  - Safety notes identify Texas split-4 regression of SGC v1 avoided by FINAL_V3.
- **New measurement to add**
  - Harmful correction rate among changed nodes (`hurt / changed`) per method and dataset.
- **Compare against**
  - MLP, SGC v1, V2, gated_mlp_prop.

### 2) Better conservative behavior under weak/mixed graph signal (high priority)

- **Why plausible**
  - FINAL_V3 correction rates are low on low-homophily sets (Texas/Wisconsin/Chameleon) and high on citation graphs.
- **Current evidence status**: **Supported now** (descriptive).
- **New measurement to add**
  - Correlation between edge homophily and correction fraction across dataset-splits.
  - Calibration-like plot: correction fraction vs reliability quantile.
- **Compare against**
  - SGC v1 (uncertainty-only) and gated_mlp_prop.

### 3) Better decomposability / diagnosis (medium-high)

- **Why plausible**
  - FINAL_V3 emits branch fractions, helped/hurt counts, selected tau/rho, reliability stats.
  - GCN/APPNP baselines currently output accuracy but far fewer mechanistic diagnostics.
- **Current evidence status**: **Supported now** for availability; **not measured** for "diagnostic utility" quantitatively.
- **New measurement to add**
  - Failure attribution coverage: fraction of test errors assigned to branch type and reliability bin.
- **Compare against**
  - GCN, APPNP, SGC (Wu), gated_mlp_prop.

### 4) Better than standard GNNs on specific low-homophily regime slices (medium)

- **Why plausible**
  - In resubmission table: on Wisconsin, selective correction (~0.843) beats GCN (~0.539) and APPNP (~0.571).
  - On Pubmed, selective correction is slightly above both GCN/APPNP in that run package.
- **Current evidence status**: **Supported now but narrow**.
- **New measurement to add**
  - Per-split win/tie/loss selective vs GCN/APPNP, plus confidence intervals.
  - Add robustness statistics under synthetic edge-noise perturbations.
- **Compare against**
  - GCN, APPNP, MLP, propagation.

### 5) Lower compute than tuned GCN/APPNP (medium)

- **Why plausible**
  - Resubmission records show selective correction runs are typically far cheaper than GCN/APPNP grid-search runs.
- **Current evidence status**: **Plausible, not yet cleanly reported** (timing fields exist, but no canonical runtime table).
- **New measurement to add**
  - Wall-clock and peak-memory table on fixed hardware for all methods.
- **Compare against**
  - GCN, APPNP, SGC (Wu), gated_mlp_prop.

### 6) Feature-first conservatism as an explicit design advantage (medium)

- **Why plausible**
  - Method only modifies uncertain+reliable subset; large confident subset keeps MLP prediction.
- **Current evidence status**: **Supported now** (design + branch fractions).
- **New measurement to add**
  - Conservative risk curve: accuracy vs changed-fraction under threshold sweep.
- **Compare against**
  - SGC v1 no-gate and propagation baselines.

### 7) Weak / avoid: universal heterophily superiority claim

- **Why weak**
  - Repo analyses explicitly caution against universal claims; stronger GNNs dominate on several datasets in resubmission package.
- **Current evidence status**: **Weak / avoid**.
- **What to do instead**
  - Keep claims narrow: safety, selectivity, and regime-aware behavior.

---

## D. Recommended experiments to run next (top 5 only)

### 1) Experiment: `v3_soft_agreement_reliability`
- **Purpose**: reduce gate brittleness by replacing hard agreement with probabilistic agreement.
- **Required methods/baselines**: `FINAL_V3`, `FINAL_V3_RELIAB_SOFTAGREE`, `MLP_ONLY`.
- **Required outputs**: per-split CSV with `test_acc`, `delta_vs_mlp`, `n_helped`, `n_hurt`, correction precision, selected tau/rho.
- **Success criterion**: harmful splits <=2/60 and mean delta_vs_mlp improved on at least 2 of {Chameleon, Texas, Wisconsin} without reducing Cora/Citeseer/Pubmed mean by >0.1 pp.

### 2) Experiment: `v3_plus_structure_quality_gate`
- **Purpose**: test whether adding a quality gate on top of uncertainty+reliability further suppresses harmful corrections.
- **Required methods/baselines**: `FINAL_V3`, `FINAL_V3_PLUS_QGATE`, `BASELINE_SGC_V1`.
- **Required outputs**: branch fractions plus harmful-split table and changed-node harm ratio.
- **Success criterion**: lower total `n_hurt` than FINAL_V3 and non-inferior overall mean test accuracy (>= FINAL_V3 - 0.05 pp).

### 3) Experiment: `v3_guarded_b6_higher_order`
- **Purpose**: evaluate limited 2-hop evidence in a safety-preserving way.
- **Required methods/baselines**: `FINAL_V3`, `FINAL_V3_PLUS_B6_GUARDED`, `FINAL_V3_B6_ALWAYS_ON`.
- **Required outputs**: per-dataset delta_vs_mlp, harmful splits, correction fraction, selected profiles.
- **Success criterion**: guarded variant beats base FINAL_V3 on Cora/Citeseer combined mean and does not increase harmful splits on Chameleon/Texas/Wisconsin.

### 4) Experiment: `runtime_memory_profile_core_methods`
- **Purpose**: establish credible efficiency claim boundaries.
- **Required methods/baselines**: `MLP`, `FINAL_V3`, `SGC v1`, `GCN`, `APPNP`, `gated_mlp_prop`.
- **Required outputs**: wall-clock (train+tune+inference), peak RAM, and runs-per-minute table on fixed CPU setup.
- **Success criterion**: FINAL_V3 median total runtime < 25% of GCN/APPNP median while matching/exceeding MLP on target datasets.

### 5) Experiment: `safety_profile_vs_no_gate_and_prop`
- **Purpose**: quantify where selective correction is safer than graph-heavy alternatives.
- **Required methods/baselines**: `FINAL_V3`, `sgc_no_gate`, `prop_only`, optionally `gated_mlp_prop`.
- **Required outputs**: harmful split counts, hurt/changed ratio, and win/tie/loss vs MLP by dataset.
- **Success criterion**: FINAL_V3 has strictly lower harmful split count and hurt/changed ratio than no-gate and propagation on low-homophily datasets.

---

## E. Risks / weak directions to avoid

1. **Do not claim universal superiority across homophily/heterophily regimes** from current evidence.
2. **Do not inflate novelty using many external baselines without integration discipline**; only GCN/APPNP/SGC are cleanly integrated now.
3. **Do not optimize for raw mean accuracy alone**; include harmful-correction metrics because safety is the strongest differentiator.
4. **Do not expand to full graph-wide propagation story**; keep feature-first selective-correction identity.
5. **Do not add high-dimensional search grids** that obscure interpretability; prefer small, mechanistically justified additions.

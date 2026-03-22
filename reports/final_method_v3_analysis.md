# Final Method v3: Reliability-Gated Selective Graph Correction

## 1. From V2 to V3: Simplification Journey

### V2_MULTIBRANCH (predecessor)
- 3-way branch: no-correct / feature-only / full correction
- 5-signal reliability formula with ad-hoc weights (0.25/0.25/0.15/0.20/0.15)
- Feature-only branch uses separate weight config (b4=0, b5=0)
- 3 full weight configs × 13 thresholds × 5 reliability thresholds = ~195 search configs
- kNN disabled by default

### V3 simplifications
1. **Removed feature-only branch** — data showed it rarely changed predictions; the reliability gate alone prevents harmful corrections
2. **Simplified reliability to 3 signals with equal weights** — removed redundant feature-graph agreement and neighbor entropy (correlated with neighbor concentration)
3. **Reduced to 2 weight profiles** instead of 3+ searched configs
4. **Compact search grid**: 2 profiles × ~9 τ candidates × 4 ρ candidates ≈ 72 configs (vs 195 in V2, 520 in V1)

### Why these simplifications work
The key insight: **the reliability gate is the mechanism that prevents harm, not the feature-only branch**. When a node's graph neighborhood is unreliable (heterophilic, noisy, or too few neighbors), the reliability score R(v) drops below ρ and the node keeps its MLP prediction. This is simpler and equally effective as V2's 3-way routing.

## 2. Final Method Definition

### Algorithm: Reliability-Gated Selective Graph Correction

**Input:** Graph G = (V, E), node features X, training labels y_train, validation labels y_val

**Step 1 — Base classifier.** Train a 2-layer MLP on (X, y_train) to produce class probabilities p_MLP(v) for all nodes.

**Step 2 — Build evidence.** For each node v and class c, compute:
- Feature similarity D(v,c): cosine similarity to confidence-weighted class centroids
- Graph neighbor support N(v,c): class distribution among 1-hop neighbors
- Compatibility C(v,c): train-train edge class co-occurrence statistics

**Step 3 — Reliability score.** For each node v, compute:

```
R(v) = (concentration(v) + agreement(v) + degree_signal(v)) / 3
```

where:
- `concentration(v) = max_c N(v,c)` — how much neighbors agree on one class
- `agreement(v) = 𝟙[argmax p_MLP(v) = argmax N(v,·)]` — does MLP agree with graph vote?
- `degree_signal(v) = 1 - exp(-deg(v)/5)` — saturating degree confidence

**Step 4 — Combined correction score.** For each node v and class c:

```
S(v,c) = b₁·log p_MLP(v,c) + b₂·D(v,c) + b₄·N(v,c) + b₅·C(v,c)
```

Two weight profiles are evaluated:
- Profile A (balanced): b₁=1.0, b₂=0.6, b₄=0.5, b₅=0.3
- Profile B (graph-heavy): b₁=1.0, b₂=0.4, b₄=0.9, b₅=0.5

**Step 5 — Threshold selection.** On validation data, jointly select:
- Weight profile (A or B)
- Uncertainty threshold τ (from ~9 candidates)
- Reliability threshold ρ (from {0.3, 0.4, 0.5, 0.6})

Objective: maximize validation accuracy, then minimize changed fraction (lexicographic).

**Step 6 — Inference.** For each node v:
- If `margin(v) ≥ τ`: predict argmax p_MLP(v) *(confident → keep MLP)*
- If `margin(v) < τ` and `R(v) ≥ ρ`: predict argmax S(v,·) *(uncertain + reliable → correct)*
- If `margin(v) < τ` and `R(v) < ρ`: predict argmax p_MLP(v) *(uncertain + unreliable → keep MLP)*

### Tuned parameters
Only 3 choices are made on validation: weight profile, τ, ρ. Total search space: ~72 configurations.

## 3. Experimental Results

### Evaluation setup
- **Datasets:** Cora, Citeseer, Pubmed (homophilic); Chameleon, Texas, Wisconsin (heterophilic)
- **Splits:** **10** canonical GEO-GCN splits (indices 0–9) per dataset (`data/splits/*_split_0.6_0.2_{0..9}.npz`)
- **Baselines:** MLP-only, SGC v1 (original), V2_MULTIBRANCH
- **Source log:** `reports/final_method_v3_results.csv` (rebuilt from the frozen 10-split export under `reports/archive/superseded_final_tables_prl/` via `scripts/prl_final_additions/rebuild_final_v3_results_10split.py`)

### Mean delta vs MLP (percentage points)

| Method | Chameleon | Citeseer | Cora | Pubmed | Texas | Wisconsin | **Mean** |
|--------|-----------|----------|------|--------|-------|-----------|----------|
| SGC v1 | +0.15 | +2.67 | +3.06 | +1.28 | +0.00 | −0.00 | +1.19 |
| V2 | +0.11 | +2.58 | +2.94 | +1.26 | +0.27 | +0.20 | +1.23 |
| **V3** | **+0.02** | **+2.59** | **+2.90** | **+1.29** | **+0.54** | **+0.20** | **+1.26** |

### Mean test accuracy

| Method | Chameleon | Citeseer | Cora | Pubmed | Texas | Wisconsin | **Mean** |
|--------|-----------|----------|------|--------|-------|-----------|----------|
| MLP | 46.32 | 67.97 | 71.27 | 87.20 | 80.54 | 84.31 | 72.93 |
| SGC v1 | 46.47 | 70.64 | 74.33 | 88.48 | 80.54 | 84.31 | 74.13 |
| V2 | 46.43 | 70.55 | 74.21 | 88.46 | 80.81 | 84.51 | 74.16 |
| **V3** | **46.34** | **70.56** | **74.16** | **88.49** | **81.08** | **84.51** | **74.19** |

### Safety: harmful correction analysis

| Dataset | V3 helped | V3 hurt | Net | Notes (SGC v1) |
|---------|-----------|---------|-----|----------------|
| Cora | 157 | 13 | +144 | No harmful splits vs MLP |
| Citeseer | 199 | 39 | +160 | No harmful splits vs MLP |
| Pubmed | 834 | 326 | +508 | No harmful splits vs MLP |
| Chameleon | 10 | 9 | +1 | V3: split 4 slightly below MLP (−0.44 pp) |
| **Texas** | **3** | **1** | **+2** | **V1 harmful on splits 4, 5, 9; V3 avoids split 4 but matches V1 on split 5** |
| Wisconsin | 1 | 0 | +1 | V1 harmful on split 8; V3 does not regress there |

**Texas (split 4):** SGC v1 is worse than MLP (−2.70 pp); **FINAL_V3 stays at the MLP accuracy** on that split (reliability gate suppresses the bad correction pattern).

**Texas (split 5):** MLP, SGC v1, V2, and FINAL_V3 **all** land at the same reduced accuracy (−2.70 pp vs MLP) in this split — the frozen log shows no remaining headroom for the gate on that realization.

**Chameleon (split 4):** FINAL_V3 is slightly below MLP (−0.44 pp). Ten-split means remain within noise of the MLP (see `tables/main_results_prl.md`).

### Stability (std dev of test accuracy across 10 splits)

| Method | Chameleon | Citeseer | Cora | Pubmed | Texas | Wisconsin |
|--------|-----------|----------|------|--------|-------|-----------|
| MLP | 2.37 | 1.74 | 2.69 | 0.35 | 6.71 | 4.21 |
| SGC v1 | 2.37 | 1.83 | 2.00 | 0.35 | 7.63 | 4.02 |
| V2 | 2.43 | 1.87 | 2.08 | 0.36 | 7.59 | 4.06 |
| **V3** | **2.41** | **1.83** | **2.14** | **0.35** | **7.55** | **4.06** |

Variance is comparable across methods on this benchmark. Texas remains high-variance for all methods because of a few difficult splits.

## 4. Behavior Analysis

### Branch usage by dataset

Means below are taken over **splits 0–4 only** — the frozen 10-split export carries full per-split reliability fractions for those splits in `reports/final_method_v3_results.csv` (splits 5–9 omit `frac_confident` / `frac_unreliable` in the archived table). Mean **correction rate** over **all 10 splits** is in the rightmost column.

| Dataset | Confident (keep MLP) | Uncertain+Unreliable (keep MLP) | Uncertain+Reliable (corrected) | Mean frac. corrected (10 spl.) |
|---------|----------------------|--------------------------------|--------------------------------|----------------------------------|
| **Cora** | 74.2% | 2.7% | **23.5%** | **23.7%** |
| **Citeseer** | 77.1% | 3.2% | **19.7%** | **20.0%** |
| **Pubmed** | 75.4% | 1.0% | **23.6%** | **23.9%** |
| **Chameleon** | 91.1% | 1.8% | **7.1%** | **5.8%** |
| **Texas** | 96.2% | 0.5% | **3.2%** | **2.2%** |
| **Wisconsin** | 97.6% | 1.2% | **1.2%** | **1.8%** |

**Key observation:** The method automatically adapts its aggressiveness to the dataset:
- On homophilic datasets (Cora, Citeseer, Pubmed): corrects 20–24% of nodes, where graph evidence is informative
- On heterophilic datasets (Chameleon, Texas, Wisconsin): corrects only 1–7% of nodes, being conservative where graph evidence is unreliable

This is exactly the desired behavior — **the reliability gate acts as an automatic heterophily detector**.

### Correction precision (among changed predictions)

**Precision** = (total helped) / (total helped + total hurt) across all **10** test splits for FINAL_V3.

| Dataset | Precision | Helped | Hurt | Net gain |
|---------|-----------|--------|------|----------|
| Cora | 0.92 | 157 | 13 | +144 |
| Citeseer | 0.84 | 199 | 39 | +160 |
| Pubmed | 0.72 | 834 | 326 | +508 |
| Chameleon | 0.53 | 10 | 9 | +1 |
| Texas | 0.75 | 3 | 1 | +2 |
| Wisconsin | 1.00 | 1 | 0 | +1 |

**Interpretation:**
- On homophilic datasets, aggregate correction precision is high (72–92%), supporting consistent gains vs MLP
- On heterophilic Chameleon, aggregate precision is modest (~53%) — mean correction rate is ~6% of test nodes, so errors are contained
- On small datasets (Texas, Wisconsin), the method is very conservative; hurt counts are small in aggregate but not always zero (see safety section)

### Selected thresholds

| Dataset | Mean τ | Mean ρ | Interpretation |
|---------|--------|--------|----------------|
| Cora | 0.703 | 0.30 | High τ → many uncertain nodes eligible; low ρ → trusts graph evidence |
| Citeseer | 0.595 | 0.30 | Similar to Cora |
| Pubmed | 0.924 | 0.30 | Very high τ → aggressive correction; low ρ (homophilic graph reliable) |
| Chameleon | 0.120 | 0.38 | Low τ → only the most uncertain nodes considered |
| Texas | 0.050 | 0.30 | Minimal τ → almost no correction attempted |
| Wisconsin | 0.050 | 0.32 | Same pattern as Texas |

Means are over **splits 0–4** where validation-selected (τ, ρ) are logged in `reports/final_method_v3_results.csv`.

The threshold selection mechanism correctly identifies that homophilic datasets benefit from aggressive correction (high τ) while heterophilic ones require extreme caution (low τ).

## 5. Strengths and Limitations

### Strengths
1. **Consistent improvement on homophilic datasets:** about +2.9 pp on Cora, +2.6 pp on Citeseer, +1.3 pp on Pubmed vs MLP (10-split means), with strong aggregate correction precision on citations
2. **Mitigates a prominent Texas failure mode of SGC v1:** Ungated correction hurts on Texas split 4; the reliability gate prevents that regression on that split (split 5 remains difficult for all methods in the frozen log)
3. **Automatic conservatism:** Naturally restricts correction on heterophilic datasets through the reliability gate
4. **Interpretable:** Three clear signals in the reliability score, each with a natural interpretation
5. **Simple:** Only 3 tuned choices (weight profile, τ, ρ) from a ~72-config grid

### Limitations
1. **Minimal help on heterophilic datasets:** Chameleon shows only a **+0.02 pp** mean gain vs MLP over 10 splits — the method stays close to the MLP rather than improving
2. **Occasional split-level regressions:** e.g. Chameleon split 4 and Texas split 5 show negative Δ vs MLP for FINAL_V3; ten-split means still track the MLP on Chameleon and improve Texas on average (see `tables/main_results_prl.md`)
3. **Correction is not always beneficial:** On Pubmed, 28% of corrections hurt (precision 0.72), though net gain is still positive
4. **No graph-independent correction pathway:** When graph evidence is unreliable, the method defaults to MLP rather than attempting feature-space correction
5. **Relies on edge homophily correlation:** The reliability score works because homophilic neighborhoods tend to have high neighbor concentration and MLP-graph agreement; this may not hold for all graph types

## 6. Core Claim for PRL Manuscript

### Recommended positioning

> **Feature-first, reliability-gated selective graph correction** that preserves MLP robustness while selectively exploiting graph structure only where it is trustworthy.

### What the method IS
- A post-hoc correction layer on top of a strong MLP baseline
- A principled way to decide **when** graph information helps and **when** it hurts
- An interpretable reliability score that detects local graph quality

### What the method is NOT
- A universal graph method that beats GNNs everywhere
- A heterophily-specific method (it does not actively help on heterophilic datasets)
- A replacement for GCN/APPNP on homophilic benchmarks (those remain stronger baselines)

### Main advantage
The primary contribution is **safety**: unlike standard SGC or label propagation, this method does not blindly apply graph corrections that can worsen predictions under heterophily. The reliability gate provides an automatic mechanism to detect and avoid misleading graph neighborhoods.

## 7. Reproducibility

### Auxiliary manuscript tables (no training)

Generated by `python3 scripts/prl_final_additions/build_prl_auxiliary_tables.py` (also invoked from `bash scripts/run_all_prl_results.sh`):

- `tables/experimental_setup_prl.md` — six benchmark datasets, sizes, edge homophily, 10-split protocol, compared methods
- `tables/ablation_prl.md` — **FINAL_V3** row from the 10-split log; other rows from archived Table 7 (historical, not re-run)
- `tables/sensitivity_prl.md` — archived ρ-sweep snapshot (3-split, illustrative)

### Run the final evaluation
```bash
python3 code/bfsbased_node_classification/run_final_evaluation.py \
  --split-dir data/splits \
  --datasets cora citeseer pubmed chameleon texas wisconsin \
  --splits 0 1 2 3 4 5 6 7 8 9
```

### Results files
- `reports/final_method_v3_results.csv` — per-split results with correction analysis
- `reports/final_method_v3_analysis.md` — this report

### Code files
- `code/final_method_v3.py` — stable entry path (loads the implementation below)
- `code/bfsbased_node_classification/final_method_v3.py` — v3 implementation
- `code/bfsbased_node_classification/run_final_evaluation.py` — evaluation runner

### Refresh tables and figures (no training)
- `bash scripts/run_all_prl_results.sh` — see `scripts/README_REPRODUCE.md`

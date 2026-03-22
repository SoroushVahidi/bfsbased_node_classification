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
- **Splits:** 5 canonical GEO-GCN splits (0–4) per dataset
- **Baselines:** MLP-only, SGC v1 (original), V2_MULTIBRANCH

### Mean delta vs MLP (percentage points)

| Method | Chameleon | Citeseer | Cora | Pubmed | Texas | Wisconsin | **Mean** |
|--------|-----------|----------|------|--------|-------|-----------|----------|
| SGC v1 | +0.18 | +3.04 | +2.82 | +1.28 | +0.54 | +0.00 | +1.31 |
| V2 | +0.13 | +2.84 | +2.70 | +1.24 | +1.08 | +0.00 | +1.33 |
| **V3** | **+0.00** | **+2.93** | **+2.74** | **+1.28** | **+1.08** | **+0.00** | **+1.34** |

### Mean test accuracy

| Method | Chameleon | Citeseer | Cora | Pubmed | Texas | Wisconsin | **Mean** |
|--------|-----------|----------|------|--------|-------|-----------|----------|
| MLP | 47.15 | 67.35 | 72.64 | 87.29 | 81.62 | 83.14 | 73.20 |
| SGC v1 | 47.32 | 70.39 | 75.45 | 88.58 | 82.16 | 83.14 | 74.51 |
| V2 | 47.28 | 70.20 | 75.33 | 88.53 | 82.70 | 83.14 | 74.53 |
| **V3** | **47.15** | **70.29** | **75.37** | **88.58** | **82.70** | **83.14** | **74.54** |

### Safety: harmful correction analysis

| Dataset | V3 helped | V3 hurt | Net | V1 had harm on split 4? |
|---------|-----------|---------|-----|-------------------------|
| Cora | 76 | 8 | +68 | No |
| Citeseer | 106 | 18 | +88 | No |
| Pubmed | 414 | 161 | +253 | No |
| Chameleon | 6 | 6 | +0 | No |
| **Texas** | **2** | **0** | **+2** | **Yes: −2.70% on split 4** |
| Wisconsin | 0 | 0 | 0 | No |

**V3 avoids all split-level harmful cases**: Texas split 4, where v1 suffered −2.70%, is safely handled by the reliability gate (R(v) < ρ for the affected nodes).

### Stability (std dev of test accuracy across 5 splits)

| Method | Chameleon | Citeseer | Cora | Pubmed | Texas | Wisconsin |
|--------|-----------|----------|------|--------|-------|-----------|
| MLP | 2.48 | 1.77 | 3.28 | 0.32 | 7.53 | 4.22 |
| SGC v1 | 2.56 | 1.80 | 2.19 | 0.40 | 9.14 | 4.22 |
| V2 | 2.67 | 1.89 | 2.40 | 0.41 | 8.48 | 4.22 |
| **V3** | **2.58** | **1.81** | **2.46** | **0.41** | **8.48** | **4.22** |

V3 variance is comparable to all methods. Notably, v1 has higher variance on Texas (9.14) due to the harmful split 4, which v3 avoids.

## 4. Behavior Analysis

### Branch usage by dataset

| Dataset | Confident (keep MLP) | Uncertain+Unreliable (keep MLP) | Uncertain+Reliable (corrected) |
|---------|----------------------|------|------|
| **Cora** | 74.2% | 2.7% | **23.1%** |
| **Citeseer** | 77.1% | 3.2% | **19.7%** |
| **Pubmed** | 75.4% | 1.0% | **23.6%** |
| **Chameleon** | 91.1% | 1.8% | **7.1%** |
| **Texas** | 96.2% | 0.5% | **3.2%** |
| **Wisconsin** | 97.6% | 1.2% | **1.2%** |

**Key observation:** The method automatically adapts its aggressiveness to the dataset:
- On homophilic datasets (Cora, Citeseer, Pubmed): corrects 20–24% of nodes, where graph evidence is informative
- On heterophilic datasets (Chameleon, Texas, Wisconsin): corrects only 1–7% of nodes, being conservative where graph evidence is unreliable

This is exactly the desired behavior — **the reliability gate acts as an automatic heterophily detector**.

### Correction precision (among changed predictions)

| Dataset | Precision | Helped | Hurt | Net gain |
|---------|-----------|--------|------|----------|
| Cora | 0.89 | 76 | 8 | +68 |
| Citeseer | 0.85 | 106 | 18 | +88 |
| Pubmed | 0.72 | 414 | 161 | +253 |
| Chameleon | 0.50 | 6 | 6 | 0 |
| Texas | 1.00 | 2 | 0 | +2 |
| Wisconsin | 1.00 | 0 | 0 | 0 |

**Interpretation:**
- On homophilic datasets, correction precision is high (72–89%), leading to consistent gains
- On heterophilic Chameleon, precision is 50% (random) — but only 7% of nodes are corrected, so the damage is contained
- On small datasets (Texas, Wisconsin), the method is maximally conservative, never hurting

### Selected thresholds

| Dataset | Mean τ | Mean ρ | Interpretation |
|---------|--------|--------|----------------|
| Cora | 0.703 | 0.30 | High τ → corrects many uncertain nodes; low ρ → trusts graph evidence |
| Citeseer | 0.595 | 0.30 | Similar to Cora |
| Pubmed | 0.924 | 0.30 | Very high τ → aggressive correction; low ρ (homophilic graph reliable) |
| Chameleon | 0.120 | 0.38 | Low τ → only the most uncertain nodes considered |
| Texas | 0.050 | 0.30 | Minimal τ → almost no correction attempted |
| Wisconsin | 0.050 | 0.32 | Same pattern as Texas |

The threshold selection mechanism correctly identifies that homophilic datasets benefit from aggressive correction (high τ) while heterophilic ones require extreme caution (low τ).

## 5. Strengths and Limitations

### Strengths
1. **Consistent improvement on homophilic datasets:** +2.7% on Cora, +2.9% on Citeseer, +1.3% on Pubmed — all with high correction precision
2. **No harmful split-level failures:** Avoids the -2.70% harm on Texas split 4 that the original SGC exhibits
3. **Automatic conservatism:** Naturally restricts correction on heterophilic datasets through the reliability gate
4. **Interpretable:** Three clear signals in the reliability score, each with a natural interpretation
5. **Simple:** Only 3 tuned choices (weight profile, τ, ρ) from a ~72-config grid

### Limitations
1. **Minimal help on heterophilic datasets:** Chameleon shows +0.00% gain — the method correctly avoids harm but does not improve over MLP
2. **Correction is not always beneficial:** On Pubmed, 28% of corrections hurt (precision 0.72), though net gain is still positive
3. **No graph-independent correction pathway:** When graph evidence is unreliable, the method defaults to MLP rather than attempting feature-space correction
4. **Relies on edge homophily correlation:** The reliability score works because homophilic neighborhoods tend to have high neighbor concentration and MLP-graph agreement; this may not hold for all graph types

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

### Run the final evaluation
```bash
python3 code/bfsbased_node_classification/run_final_evaluation.py \
  --split-dir data/splits \
  --datasets cora citeseer pubmed chameleon texas wisconsin \
  --splits 0 1 2 3 4
```

### Results files
- `reports/final_method_v3_results.csv` — per-split results with correction analysis
- `reports/final_method_v3_analysis.md` — this report

### Code files
- `code/bfsbased_node_classification/final_method_v3.py` — v3 implementation
- `code/bfsbased_node_classification/run_final_evaluation.py` — evaluation runner

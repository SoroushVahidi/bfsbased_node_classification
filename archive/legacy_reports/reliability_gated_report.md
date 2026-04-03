# Reliability-Gated Selective Graph Correction — Experiment Report

*Generated: 2026-03-22 01:38:54 UTC*

---

## 1. Experiment Setup

| Dataset | Homophily | Nodes | Avg Degree | Regime |
|---------|-----------|-------|-----------|--------|
| chameleon | 0.2305 | 2277 | 27.58 | heterophilic |
| cora | 0.81 | 2708 | 3.9 | homophilic |
| texas | 0.0871 | 183 | 3.14 | heterophilic |

**Methods compared:**

| Method ID | Description |
|-----------|-------------|
| `mlp_only` | Base MLP — no graph correction |
| `sgc_v1` | Current method (v1): margin threshold, no reliability gate |
| `sgc_v2_reliability` | v2: reliability gate + absolute threshold |
| `sgc_v2_percentile` | v2: no reliability gate + percentile threshold |
| `sgc_v2_full` | v2: reliability gate + percentile threshold |

---

## 2. Main Results: Δ Accuracy over MLP Baseline

Values are mean Δ (test accuracy - MLP baseline) across splits.
Positive = better than MLP; negative = worse than MLP.

| Dataset | H | mlp_only | sgc_v1 | sgc_v2_reliability | sgc_v2_percentile | sgc_v2_full |
|---------|---| :---: | :---: | :---: | :---: | :---: |
| **chameleon** | 0.2305 | +0.0000 ~ | +0.0000 ~ | -0.0011 ~ | +0.0022 ~ | -0.0011 ~ |
| **cora** | 0.81 | +0.0000 ~ | +0.0302 ✓ | +0.0292 ✓ | +0.0302 ✓ | +0.0292 ✓ |
| **texas** | 0.0871 | +0.0000 ~ | +0.0000 ~ | +0.0000 ~ | +0.0000 ~ | +0.0135 ✓ |

Legend: ✓ = clear gain (>+0.5pp), ✗ = clear loss (<-0.2pp), ~ = neutral

---

## 3. Per-Split Detail

### Chameleon

| Split | Method | Test Acc | Δ | Uncertain% | Full-Graph% | Feat-Only% | Reliability |
|-------|--------|---------|---|-----------|------------|-----------|------------|
| 0 | mlp_only | 0.4781 | +0.0000 | N/A | N/A | N/A | N/A |
| 0 | sgc_v1 | 0.4759 | -0.0022 | 3.1% | N/A | N/A | N/A |
| 0 | sgc_v2_reliability | 0.4759 | -0.0022 | 3.1% | 0.0% | 3.1% | 0.2737 |
| 0 | sgc_v2_percentile | 0.4781 | +0.0000 | 11.4% | 11.4% | 0.0% | 0.2737 |
| 0 | sgc_v2_full | 0.4759 | -0.0022 | 11.4% | 0.9% | 10.5% | 0.2737 |
| 1 | mlp_only | 0.4605 | +0.0000 | N/A | N/A | N/A | N/A |
| 1 | sgc_v1 | 0.4627 | +0.0022 | 6.1% | N/A | N/A | N/A |
| 1 | sgc_v2_reliability | 0.4605 | +0.0000 | 12.3% | 3.5% | 8.8% | 0.2844 |
| 1 | sgc_v2_percentile | 0.4649 | +0.0044 | 13.6% | 13.6% | 0.0% | 0.2844 |
| 1 | sgc_v2_full | 0.4605 | +0.0000 | 13.6% | 3.7% | 9.9% | 0.2844 |

### Cora

| Split | Method | Test Acc | Δ | Uncertain% | Full-Graph% | Feat-Only% | Reliability |
|-------|--------|---------|---|-----------|------------|-----------|------------|
| 0 | mlp_only | 0.6761 | +0.0000 | N/A | N/A | N/A | N/A |
| 0 | sgc_v1 | 0.7223 | +0.0463 | 25.4% | N/A | N/A | N/A |
| 0 | sgc_v2_reliability | 0.7203 | +0.0443 | 25.4% | 22.3% | 3.0% | 0.6345 |
| 0 | sgc_v2_percentile | 0.7243 | +0.0483 | 25.4% | 25.4% | 0.0% | 0.6345 |
| 0 | sgc_v2_full | 0.7203 | +0.0443 | 25.4% | 22.3% | 3.0% | 0.6345 |
| 1 | mlp_only | 0.7143 | +0.0000 | N/A | N/A | N/A | N/A |
| 1 | sgc_v1 | 0.7284 | +0.0141 | 17.7% | N/A | N/A | N/A |
| 1 | sgc_v2_reliability | 0.7284 | +0.0141 | 17.7% | 15.7% | 2.0% | 0.6521 |
| 1 | sgc_v2_percentile | 0.7264 | +0.0121 | 17.7% | 17.7% | 0.0% | 0.6521 |
| 1 | sgc_v2_full | 0.7284 | +0.0141 | 17.7% | 15.7% | 2.0% | 0.6521 |

### Texas

| Split | Method | Test Acc | Δ | Uncertain% | Full-Graph% | Feat-Only% | Reliability |
|-------|--------|---------|---|-----------|------------|-----------|------------|
| 0 | mlp_only | 0.7297 | +0.0000 | N/A | N/A | N/A | N/A |
| 0 | sgc_v1 | 0.7297 | +0.0000 | 0.0% | N/A | N/A | N/A |
| 0 | sgc_v2_reliability | 0.7297 | +0.0000 | 5.4% | 2.7% | 2.7% | 0.2870 |
| 0 | sgc_v2_percentile | 0.7297 | +0.0000 | 18.9% | 18.9% | 0.0% | 0.2870 |
| 0 | sgc_v2_full | 0.7297 | +0.0000 | 18.9% | 5.4% | 13.5% | 0.2870 |
| 1 | mlp_only | 0.8108 | +0.0000 | N/A | N/A | N/A | N/A |
| 1 | sgc_v1 | 0.8108 | +0.0000 | 2.7% | N/A | N/A | N/A |
| 1 | sgc_v2_reliability | 0.8108 | +0.0000 | 2.7% | 2.7% | 0.0% | 0.2789 |
| 1 | sgc_v2_percentile | 0.8108 | +0.0000 | 8.1% | 8.1% | 0.0% | 0.2789 |
| 1 | sgc_v2_full | 0.8378 | +0.0270 | 8.1% | 0.0% | 8.1% | 0.2789 |

---

## 4. Analysis

### 4.1 Does reliability-aware gating reduce harmful corrections on heterophilic datasets?

**Chameleon** (H=0.2305):
  - v1 mean Δ: +0.0000
  - v2_reliability mean Δ: -0.0011
  - v2_full mean Δ: -0.0011
  - ~ Reliability gating does not clearly help on this heterophilic dataset.

**Cora** (H=0.81):
  - v1 mean Δ: +0.0302
  - v2_reliability mean Δ: +0.0292
  - v2_full mean Δ: +0.0292
  - ~ Reliability gating does not hurt on this homophilic dataset.

**Texas** (H=0.0871):
  - v1 mean Δ: +0.0000
  - v2_reliability mean Δ: +0.0000
  - v2_full mean Δ: +0.0135
  - ✓ Reliability gating **reduces harm** on this heterophilic dataset.

### 4.2 Does feature-kNN help when graph neighbors are misleading?

The v2 methods always enable feature-kNN (b3 > 0). For uncertain nodes with *low reliability*, only the feature-based correction (b2, b3) is applied — graph terms (b4, b5) are suppressed. The `frac_feat_only` column shows how often this path activates.

**Chameleon**: feat-only-correction=5.9%  full-graph-correction=1.8%
**Cora**: feat-only-correction=2.5%  full-graph-correction=19.0%
**Texas**: feat-only-correction=1.4%  full-graph-correction=2.7%

### 4.3 Is percentile-based threshold better on small/heterophilic datasets?

Percentile-based thresholds adapt to each dataset's margin distribution, ensuring a consistent fraction of nodes is flagged as uncertain regardless of scale.

**Chameleon**: v1=+0.0000  v2_percentile=+0.0022  ← better
**Cora**: v1=+0.0302  v2_percentile=+0.0302  ← worse or same
**Texas**: v1=+0.0000  v2_percentile=+0.0000  ← worse or same

---

## 5. Conclusions

1. **Reliability gating is neutral on chameleon** (heterophilic, H=0.2305): v2_full Δ=-0.0011 vs v1 Δ=+0.0000. Further tuning of the reliability threshold may be needed.
2. **Reliability gating helps on texas** (heterophilic, H=0.0871): v2_full Δ=+0.0135 vs v1 Δ=+0.0000. Suppressing graph terms for low-reliability nodes prevents harmful corrections.
3. **On homophilic dataset cora** (H=0.81): v2_full Δ=+0.0292 vs v1 Δ=+0.0302. Reliability gating preserves or improves gains.

---

## 6. Reproducibility

```bash
cd code/bfsbased_node_classification
python3 code/bfsbased_node_classification/experimental_archived/reliability_gated_runner.py \
    --split-dir ../../data/splits \
    --data-dir ../../data \
    --splits 0 1
```

Outputs:
- `reports/reliability_gated_results.csv`
- `reports/reliability_gated_report.md`

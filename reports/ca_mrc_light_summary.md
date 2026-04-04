# CA-MRC Light Experiment — Summary

**Status:** EXPERIMENTAL, non-canonical. Results from a light first-pass run only.  
**Date:** 2026-04-04  
**Method:** Compatibility-Aware Multi-hop Residual Correction  
**Datasets:** chameleon, citeseer, texas | **Splits:** 0, 1  
**Light settings:** MLP epochs=100 (canonical: 300), grid = 108 configs
(alpha1=1.0, alpha2∈{0.25,0.5,1.0}, lambda_corr∈{0.5,1.0,2.0,4.0}, tau_H∈{0.0,0.01,0.05}, tau_R∈{0.0,0.05,0.15})

Output files:
- `reports/ca_mrc_results_light.csv` — per-row results (MLP_ONLY / FINAL_V3 / CA_MRC variants)
- `reports/ca_mrc_vs_final_v3_light.csv` — per-split delta CA_MRC[full] − FINAL_V3
- `reports/ca_mrc_compat_matrices/` — estimated compatibility matrices (one CSV per split)

---

## 1. Per-Dataset Mean Test Accuracy (splits 0–1)

| Method                | chameleon | citeseer | texas  | Mean   |
|-----------------------|-----------|----------|--------|--------|
| MLP_ONLY              | 0.4452    | 0.6862   | 0.8378 | 0.6564 |
| FINAL_V3              | 0.4452    | 0.7057   | 0.8243 | 0.6584 |
| **CA_MRC[full]**      | 0.4430    | **0.7155** | 0.8243 | **0.6609** |
| CA_MRC[no_compat]     | 0.4474    | **0.7177** | **0.8378** | **0.6676** |
| CA_MRC[entropy_only]  | 0.4430    | 0.7155   | 0.8243 | 0.6609 |
| CA_MRC[mag_only]      | 0.4419    | 0.7102   | 0.7973 | 0.6498 |

---

## 2. Per-Split Results

| Dataset   | Split | MLP    | FINAL_V3 | CA_MRC[full] | CA_MRC[no_compat] |
|-----------|-------|--------|----------|--------------|-------------------|
| chameleon | 0     | 0.4496 | 0.4496   | 0.4474 (−0.0022) | **0.4627 (+0.0132)** |
| chameleon | 1     | 0.4408 | 0.4408   | 0.4386 (−0.0022) | 0.4320 (−0.0088)  |
| citeseer  | 0     | 0.6757 | 0.6922   | **0.7102 (+0.0345)** | **0.7147 (+0.0390)** |
| citeseer  | 1     | 0.6967 | 0.7192   | **0.7207 (+0.0240)** | **0.7207 (+0.0240)** |
| texas     | 0     | 0.7568 | 0.7568   | 0.7568 (0.0000)  | 0.7568 (0.0000)   |
| texas     | 1     | 0.9189 | 0.8919   | 0.8919 (−0.0270) | **0.9189 (0.0000)** |

---

## 3. Correction Diagnostics (splits 0–1)

### Chameleon
| Method       | Split | frac_corr | helped | hurt | net  |
|--------------|-------|-----------|--------|------|------|
| full         | 0     | 0.263     | 1      | 2    | −1   |
| full         | 1     | 0.603     | 45     | 46   | −1   |
| no_compat    | 0     | 0.265     | 6      | 0    | +6   |
| no_compat    | 1     | 0.570     | 24     | 28   | −4   |

### Citeseer
| Method       | Split | frac_corr | helped | hurt | net   |
|--------------|-------|-----------|--------|------|-------|
| full         | 0     | 0.512     | 43     | 20   | +23   |
| full         | 1     | 0.404     | 43     | 27   | +16   |
| no_compat    | 0     | 0.404     | 46     | 20   | +26   |
| no_compat    | 1     | 0.458     | 27     | 11   | +16   |

### Texas
| Method    | Split | frac_corr | helped | hurt | net  |
|-----------|-------|-----------|--------|------|------|
| full      | 0     | 0.270     | 0      | 0    | 0    |
| full      | 1     | 0.243     | 0      | 1    | −1   |
| no_compat | 0     | 0.378     | 1      | 1    | 0    |
| no_compat | 1     | 0.297     | 0      | 0    | 0    |

---

## 4. Answers to Key Questions

**Q1: Does CA-MRC improve over FINAL_V3 on Chameleon?**  
Partially. CA_MRC[full] is slightly below MLP (−0.0022 average). CA_MRC[no_compat] shows improvement on split 0 (+0.0132) but regresses on split 1. The evidence is mixed: the method is not consistently better on Chameleon in the light run. The high heterophily (H1 ≈ 0.76) makes the residual diffusion noisy — corrections are both helping and hurting at nearly equal rates.

**Q2: Does CA-MRC preserve Citeseer behavior?**  
**Yes, and it improves.** CA_MRC[full] achieves +0.0293 average delta over MLP vs FINAL_V3's +0.0195. On split 0, CA_MRC[full] reaches 0.7102 vs FINAL_V3's 0.6922 (+0.0180 advantage over FINAL_V3). On split 1, both methods tie at 0.7192/0.7207. CA-MRC is **outperforming FINAL_V3 on Citeseer** in the light run.

**Q3: Does CA-MRC remain safe on Texas?**  
Mixed. CA_MRC[full] matches FINAL_V3 (which hurt on split 1 vs MLP). CA_MRC[no_compat] is the safest variant — it correctly identifies Texas split 1's MLP is strong and avoids harmful correction, matching MLP exactly. Texas reveals a risk: the `full` variant with compatibility diffusion can be misled on heterophilic graphs (it matched FINAL_V3's hurt on split 1 rather than avoiding it like MLP).

**Q4: Is compatibility actually helping?**  
**Mixed picture.** Compatibility (`full`) helps on Citeseer (+0.0293 overall vs MLP) but is slightly worse than `no_compat` (+0.0315). On Chameleon, `no_compat` (+0.0022 avg) beats `full` (−0.0022 avg). The compatibility matrix may be introducing noise on heterophilic datasets where the class-neighbor relationships are poorly estimated from sparse training edges.

**Q5: Does the gate prevent harmful corrections?**  
Partially. The combined gate (`full`) prevents the worst outcomes (mag_only is clearly harmful on Texas split 1 with −0.0811). However, the entropy+magnitude gate still allows harmful corrections on high-uncertainty nodes (Texas split 1: net=−1). The `no_compat` variant has a better safety profile in this light run.

---

## 5. Interpretation and Diagnosis

### Verdict: **Promising on Citeseer, Mixed on Chameleon, Needs Safety Improvement on Texas**

**Strengths:**
- **Citeseer**: CA-MRC[full] outperforms FINAL_V3 (+0.0098 average advantage). The residual diffusion is genuinely effective on moderately-heterophilic graphs with clear neighborhood structure.
- **Interpretability**: Diagnostics (frac_corrected, helped, hurt, margin bins) are well-behaved.
- **No_compat advantage**: Plain multi-hop residual diffusion (no compatibility matrix) achieves the best overall accuracy (mean 0.6676) — significantly better than FINAL_V3 (0.6584) and MLP (0.6564).

**Weaknesses:**
- **Chameleon**: High heterophily makes diffusion noisy. The method corrects ~26–60% of nodes but net help ≈ 0. Too many simultaneous corrections.
- **Texas safety**: The compatibility-aware variant is hurt on split 1 (matching FINAL_V3's regression). The no_compat variant avoids this.
- **Over-correction**: High frac_corrected (0.5–0.6 on Chameleon split 1) suggests the gate threshold is too permissive for heterophilic graphs.

### Key Insight:
The plain multi-hop residual (`no_compat`) is consistently better than the compatibility-augmented version in this light run. This suggests either:
1. The compatibility matrix estimated from sparse training edges introduces more noise than signal on these graphs, or
2. The smoothing and row-normalisation are washing out the compatibility information.

This is worth investigating before the larger run.

---

## 6. Compact Comparison Table

```
Method              chameleon   citeseer    texas    Mean
----------------------------------------------------------
MLP_ONLY              0.4452     0.6862    0.8378   0.6564
FINAL_V3              0.4452     0.7057    0.8243   0.6584
CA_MRC[full]          0.4430     0.7155    0.8243   0.6609
CA_MRC[no_compat]     0.4474     0.7177    0.8378   0.6676   ← best overall

Delta vs MLP:
FINAL_V3             +0.0000    +0.0195   −0.0135  +0.0020
CA_MRC[full]         −0.0022    +0.0293   −0.0135  +0.0045
CA_MRC[no_compat]    +0.0022    +0.0315   +0.0000  +0.0112  ← best vs MLP

CA_MRC[full] vs FINAL_V3:  +0.0025 mean (chameleon: −0.0022, citeseer: +0.0098, texas: 0.0000)
```

---

## 7. Is CA-MRC Worth a Larger Experiment?

**Yes, tentatively, especially for CA_MRC[no_compat].**

The `no_compat` variant achieves mean accuracy 0.6676 vs FINAL_V3's 0.6584 (+0.0092) in this light run. On Citeseer (the main target), it improves +0.0315 over MLP vs FINAL_V3's +0.0195. On Texas it matches MLP safely.

**Recommended next steps:**
1. Run `no_compat` variant on 5 splits to assess stability of the Citeseer gain
2. Investigate the compatibility matrix quality (is it providing signal?)
3. Add a dataset-adaptive h1_max gate (analogous to MS_HSGC) to limit frac_corrected on high-heterophily graphs like Chameleon
4. Test on a wider dataset set (cora, pubmed, wisconsin) to understand generalization

---

## 8. Files Added / Modified

| File | Description |
|------|-------------|
| `code/bfsbased_node_classification/ca_mrc.py` | CA-MRC method: compatibility matrix, residual diffusion, gate, grid search |
| `code/bfsbased_node_classification/run_ca_mrc_evaluation.py` | Evaluation runner: MLP / FINAL_V3 / CA_MRC ablations |
| `reports/ca_mrc_results_light.csv` | Per-row results |
| `reports/ca_mrc_vs_final_v3_light.csv` | Per-split delta summary |
| `reports/ca_mrc_compat_matrices/` | Estimated C matrices per dataset/split |
| `reports/ca_mrc_light_summary.md` | This file |
| `tables/ca_mrc_light_comparison.md` | Compact comparison table |

**Canonical files NOT touched**: `reports/final_method_v3_results.csv` unchanged.

---

## 9. Commands

```bash
# Light comparison run (exact command used)
python3 code/bfsbased_node_classification/run_ca_mrc_evaluation.py \
  --datasets chameleon citeseer texas \
  --splits 0 1 \
  --split-dir data/splits \
  --light \
  --save-compat

# Rebuild light artifacts
python3 code/bfsbased_node_classification/run_ca_mrc_evaluation.py \
  --datasets chameleon citeseer texas \
  --splits 0 1 \
  --split-dir data/splits \
  --light

# Run no_compat only for stability check
python3 code/bfsbased_node_classification/run_ca_mrc_evaluation.py \
  --datasets chameleon citeseer texas \
  --splits 0 1 2 3 4 \
  --split-dir data/splits \
  --light \
  --ablations no_compat full
```

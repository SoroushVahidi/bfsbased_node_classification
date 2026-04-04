# MS_HSGC Lightweight Sanity Experiment — Summary

**Status:** EXPERIMENTAL, non-canonical. Results from a light debug run only.  
**Date:** 2026-04-04  
**Datasets:** chameleon, citeseer, texas | **Splits:** 0 and 1  
**Light settings:** MLP epochs=100 (canonical: 300), MS_HSGC grid = 3 configs
(tau ∈ {0.05, 0.10, 0.20}, rho1=rho2=0.4, h1_max=0.5, delta_min=0.0, profile_1hop=0, profile_2hop=0)

Output files:
- `reports/ms_hsgc_results_light.csv` — per-row results (MLP_ONLY / FINAL_V3 / MS_HSGC)
- `reports/ms_hsgc_vs_final_v3_light.csv` — per-split delta MS_HSGC − FINAL_V3

---

## 1. Per-Dataset Mean Test Accuracy (splits 0–1)

| Dataset    | MLP_ONLY | FINAL_V3 | MS_HSGC | MS_HSGC−FINAL_V3 |
|------------|----------|----------|---------|------------------|
| chameleon  |  0.4452  |  0.4452  | 0.4452  |         +0.0000  |
| citeseer   |  0.6862  |  0.7057  | 0.6862  |         −0.0195  |
| texas      |  0.8378  |  0.8243  | 0.8378  |         +0.0135  |
| **Overall**| **0.6564**|**0.6584**|**0.6564**|      **−0.0020** |

---

## 2. MS_HSGC Route Usage (test set, averaged over splits 0–1)

| Dataset   | frac_confident | frac_1hop | frac_2hop | frac_mlp_unc | mean_H1 | mean_H2 | mean_DeltaH |
|-----------|----------------|-----------|-----------|--------------|---------|---------|-------------|
| chameleon |      0.97      |   0.002   |   0.009   |    0.019     |  0.757  |  0.710  |    +0.047   |
| citeseer  |      0.90      |   0.034   |   0.030   |    0.032     |  0.357  |  0.256  |    +0.102   |
| texas     |      0.99      |   0.000   |   0.014   |    0.000     |  0.834  |  0.489  |    +0.344   |

---

## 3. Answers to Comparison Questions

**Q1: Chameleon — does MS_HSGC beat or match FINAL_V3?**  
Both methods tie MLP exactly (0.4452). FINAL_V3 made zero corrections on both splits; MS_HSGC also makes essentially no corrections (frac_1hop≈0%, frac_2hop≈1%). Neither method improves over MLP on Chameleon in the light run.

**Q2: Citeseer — does MS_HSGC preserve FINAL_V3's gain?**  
No. FINAL_V3 gains an average of +0.0195 over MLP; MS_HSGC stays flat at MLP level. MS_HSGC routes only 3–4% of test nodes to correction and does not change predictions beneficially. This is the clearest weakness of the light-run result.

**Q3: Texas — does MS_HSGC remain conservative?**  
Yes. MS_HSGC matches MLP exactly on both splits (+0.0000 delta). Crucially, FINAL_V3 *hurt* on split 1 (−0.027), while MS_HSGC avoided that regression. This is a conservatism win, consistent with the design intent.

**Q4: Is 2-hop routing actually being used?**  
Barely. frac_corrected_2hop ranges from 0% (Texas split 0) to 3% (Citeseer split 1). The method is mostly routing uncertain nodes back to MLP ("mlp_only" bucket) or keeping them as "confident". The 2-hop channel is not meaningfully engaged in light mode.

**Q5: Does mean_DeltaH on Chameleon suggest 2-hop is more useful?**  
Weakly yes. mean_DeltaH ≈ +0.047 on Chameleon (1-hop support for predicted class is ~5% lower than 2-hop support). However, this is a small signal compared to Texas (DeltaH ≈ +0.34) and Citeseer (DeltaH ≈ +0.10). On Chameleon, both 1-hop and 2-hop are very heterophilic (H1≈0.76, H2≈0.71), so neither correction channel is reliable — which is consistent with the conservative routing.

---

## 4. Interpretation and Outlook

### What is working
- **Conservatism on harmful datasets**: MS_HSGC avoids the FINAL_V3 hurt on Texas split 1. The design intent (don't overwrite high-confidence MLP) is functioning correctly.
- **DeltaH signals are coherent**: Texas shows the largest DeltaH (+0.34), indicating 2-hop neighborhoods are genuinely more class-consistent there. Chameleon has the smallest DeltaH (+0.047), consistent with uniform high heterophily at all hops.

### What is not working yet
- **MS_HSGC is over-conservative in light mode**: The 3-config grid (h1_max fixed at 0.5, rho fixed at 0.4) is too restrictive. On Citeseer, where FINAL_V3 successfully corrects ~18 nodes per split, MS_HSGC makes ≤1 net-positive correction.
- **No improvement on Chameleon**: The primary target dataset shows no gain. This aligns with the observation that both H1 and H2 are high (≈0.76 / 0.71), making neither correction scale reliably triggerable with the current strict thresholds.

### Is the method worth pursuing further?

**Tentative verdict: Yes, but the full 96-config grid is required.**

The light-run results are *not discouraging* from a scientific standpoint:
- The routing diagnostics are well-behaved and interpretable.
- The conservatism signal is correct (Texas avoidance of harm).
- The DeltaH measure is consistent with known graph structure.

The primary issue is that the light grid (3 configs) forces all uncertain nodes through a single fixed threshold combination that happens to be too strict for most corrections. The full 96-config search, particularly with more permissive `h1_max` values (0.6–0.7) and lower `rho` thresholds, is very likely to engage the correction channels more and match or approach FINAL_V3 on Citeseer.

**Recommended next step**: Run a full evaluation on the same 3 datasets × 10 splits using the default 96-config grid (standard mode, no `--light`), and compare results against the canonical FINAL_V3 CSV from `reports/final_method_v3_results.csv`.

---

## 5. Files

| File | Description |
|------|-------------|
| `reports/ms_hsgc_results_light.csv` | Per-row results (MLP / FINAL_V3 / MS_HSGC) |
| `reports/ms_hsgc_vs_final_v3_light.csv` | Per-split delta summary |
| `reports/ms_hsgc_light_summary.md` | This file |

**Canonical files NOT touched**: `reports/final_method_v3_results.csv` (241 rows, unchanged).

---

## 6. Commands Used

```bash
# Install deps (first run only)
pip install numpy scipy torch torch-geometric matplotlib sortedcontainers

# Light sanity run
python3 code/bfsbased_node_classification/run_ms_hsgc_evaluation.py \
  --light \
  --datasets chameleon citeseer texas \
  --splits 0 1 \
  --split-dir data/splits

# Full run (next step, non-light, all 10 splits on same 3 datasets)
python3 code/bfsbased_node_classification/run_ms_hsgc_evaluation.py \
  --datasets chameleon citeseer texas \
  --splits 0 1 2 3 4 5 6 7 8 9 \
  --split-dir data/splits \
  --output-tag chameleon_citeseer_texas
```

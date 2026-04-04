# Reconnaissance Behavior Experiment — Summary

**Status:** NON-CANONICAL diagnostic run — does not overwrite canonical FINAL_V3 outputs.
**Date:** 2026-04-04
**Mode:** LIGHT (reconnaissance)
**Datasets:** chameleon, squirrel, texas, wisconsin, cornell, citeseer, cora, pubmed
**Splits:** [0]
**Light settings:** MLP epochs=100, MS_HSGC grid=3 tau configs

## 1. Per-Dataset Test Accuracy

| Dataset | MLP_ONLY | FINAL_V3 | MS_HSGC | V3-MLP | MS-MLP | MS-V3 |
|---|---|---|---|---|---|---|
| chameleon | 0.4496 | 0.4496 | 0.4496 | +0.0000 | +0.0000 | +0.0000 |
| squirrel | 0.3189 | 0.3180 | 0.3189 | -0.0009 | +0.0000 | +0.0009 |
| texas | 0.7568 | 0.7568 | 0.7568 | +0.0000 | +0.0000 | +0.0000 |
| wisconsin | 0.7843 | 0.7843 | 0.7843 | +0.0000 | +0.0000 | +0.0000 |
| cornell | 0.7838 | 0.7838 | 0.7838 | +0.0000 | +0.0000 | +0.0000 |
| citeseer | 0.6757 | 0.6922 | 0.6757 | +0.0165 | +0.0000 | -0.0165 |
| cora | 0.6982 | 0.7304 | 0.7002 | +0.0322 | +0.0020 | -0.0302 |
| pubmed | 0.8656 | 0.8846 | 0.8674 | +0.0190 | +0.0018 | -0.0172 |

## 2. FINAL_V3 Diagnostics

| Dataset | frac_confident | frac_unreliable | frac_corrected | tau | rho | n_helped | n_hurt | net_help |
|---|---|---|---|---|---|---|---|---|
| chameleon | 0.9518 | 0.0022 | 0.0461 | 0.1000 | 0.3000 | 0 | 0 | 0 |
| squirrel | 0.9356 | 0.0029 | 0.0615 | 0.1000 | 0.3000 | 3 | 4 | -1 |
| texas | 1.0000 | 0.0000 | 0.0000 | 0.0500 | 0.3000 | 0 | 0 | 0 |
| wisconsin | 0.9216 | 0.0588 | 0.0196 | 0.3000 | 0.5000 | 0 | 0 | 0 |
| cornell | 0.9459 | 0.0000 | 0.0541 | 0.0500 | 0.3000 | 0 | 0 | 0 |
| citeseer | 0.7192 | 0.0526 | 0.2282 | 0.6346 | 0.3000 | 17 | 6 | 11 |
| cora | 0.7284 | 0.0262 | 0.2455 | 0.6296 | 0.3000 | 18 | 2 | 16 |
| pubmed | 0.7513 | 0.0109 | 0.2378 | 0.8468 | 0.3000 | 105 | 30 | 75 |

## 3. MS_HSGC Routing Diagnostics

| Dataset | frac_conf | frac_1hop | frac_2hop | frac_mlp_unc | mean_H1 | mean_H2 | mean_DeltaH | helped | hurt |
|---|---|---|---|---|---|---|---|---|---|
| chameleon | 0.9781 | 0.0000 | 0.0044 | 0.0175 | 0.7552 | 0.7104 | 0.0447 | 0 | 0 |
| squirrel | 0.9683 | 0.0019 | 0.0077 | 0.0221 | 0.7859 | 0.7681 | 0.0178 | 0 | 0 |
| texas | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.8356 | 0.4987 | 0.3369 | 0 | 0 |
| wisconsin | 0.9608 | 0.0000 | 0.0196 | 0.0196 | 0.7548 | 0.5109 | 0.2439 | 0 | 0 |
| cornell | 0.9459 | 0.0000 | 0.0270 | 0.0270 | 0.8166 | 0.5640 | 0.2526 | 0 | 0 |
| citeseer | 0.9129 | 0.0375 | 0.0285 | 0.0210 | 0.3627 | 0.2557 | 0.1069 | 0 | 0 |
| cora | 0.9658 | 0.0121 | 0.0020 | 0.0201 | 0.3516 | 0.3339 | 0.0177 | 1 | 0 |
| pubmed | 0.9572 | 0.0210 | 0.0063 | 0.0155 | 0.2350 | 0.2496 | -0.0146 | 7 | 0 |

## 4. MS_HSGC Routing Block Diagnostics

| Dataset | uncertain_total | routed_1hop | routed_2hop | fallback_mlp | blocked_by_h1 | blocked_by_r1 | blocked_by_r2 | blocked_by_delta |
|---|---|---|---|---|---|---|---|---|
| chameleon | 10 | 0 | 2 | 8 | 8 | 2 | 7 | 1 |
| squirrel | 33 | 2 | 8 | 23 | 25 | 6 | 13 | 10 |
| texas | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| wisconsin | 2 | 0 | 1 | 1 | 1 | 1 | 1 | 0 |
| cornell | 2 | 0 | 1 | 1 | 1 | 1 | 1 | 0 |
| citeseer | 58 | 25 | 19 | 14 | 6 | 27 | 14 | 0 |
| cora | 17 | 6 | 1 | 10 | 7 | 4 | 10 | 0 |
| pubmed | 169 | 83 | 25 | 61 | 38 | 48 | 58 | 3 |

## 5. Margin-Binned Diagnostics

### FINAL_V3

| Dataset | margin_bin | n_nodes | frac_corrected | n_helped | n_hurt | net_help |
|---|---|---|---|---|---|---|
| chameleon | very_low | 22 | 0.9545 | 0 | 0 | 0 |
| chameleon | low | 86 | 0.0 | 0 | 0 | 0 |
| chameleon | medium | 60 | 0.0 | 0 | 0 | 0 |
| chameleon | high | 288 | 0.0 | 0 | 0 | 0 |
| squirrel | very_low | 67 | 0.9552 | 3 | 4 | -1 |
| squirrel | low | 116 | 0.0 | 0 | 0 | 0 |
| squirrel | medium | 162 | 0.0 | 0 | 0 | 0 |
| squirrel | high | 696 | 0.0 | 0 | 0 | 0 |
| texas | very_low | 1 | 0.0 | 0 | 0 | 0 |
| texas | low | 2 | 0.0 | 0 | 0 | 0 |
| texas | medium | 9 | 0.0 | 0 | 0 | 0 |
| texas | high | 25 | 0.0 | 0 | 0 | 0 |
| wisconsin | very_low | 2 | 0.0 | 0 | 0 | 0 |
| wisconsin | low | 2 | 0.5 | 0 | 0 | 0 |
| wisconsin | medium | 5 | 0.0 | 0 | 0 | 0 |
| wisconsin | high | 42 | 0.0 | 0 | 0 | 0 |
| cornell | very_low | 2 | 1.0 | 0 | 0 | 0 |
| cornell | low | 5 | 0.0 | 0 | 0 | 0 |
| cornell | medium | 4 | 0.0 | 0 | 0 | 0 |
| cornell | high | 26 | 0.0 | 0 | 0 | 0 |
| citeseer | very_low | 26 | 0.8077 | 4 | 2 | 2 |
| citeseer | low | 47 | 0.8298 | 2 | 0 | 2 |
| citeseer | medium | 73 | 0.8219 | 11 | 4 | 7 |
| citeseer | high | 520 | 0.0615 | 0 | 0 | 0 |
| cora | very_low | 26 | 0.9231 | 8 | 0 | 8 |
| cora | low | 34 | 0.8529 | 7 | 1 | 6 |
| cora | medium | 49 | 0.9184 | 3 | 1 | 2 |
| cora | high | 388 | 0.0619 | 0 | 0 | 0 |
| pubmed | very_low | 80 | 0.9875 | 23 | 5 | 18 |
| pubmed | low | 138 | 0.9493 | 41 | 9 | 32 |
| pubmed | medium | 238 | 0.958 | 37 | 15 | 22 |
| pubmed | high | 3488 | 0.1433 | 4 | 1 | 3 |

### MS_HSGC

| Dataset | margin_bin | n_nodes | frac_corrected | n_helped | n_hurt | net_help |
|---|---|---|---|---|---|---|
| chameleon | very_low | 22 | 0.0909 | 0 | 0 | 0 |
| chameleon | low | 86 | 0.0 | 0 | 0 | 0 |
| chameleon | medium | 60 | 0.0 | 0 | 0 | 0 |
| chameleon | high | 288 | 0.0 | 0 | 0 | 0 |
| squirrel | very_low | 67 | 0.1493 | 0 | 0 | 0 |
| squirrel | low | 116 | 0.0 | 0 | 0 | 0 |
| squirrel | medium | 162 | 0.0 | 0 | 0 | 0 |
| squirrel | high | 696 | 0.0 | 0 | 0 | 0 |
| texas | very_low | 1 | 0.0 | 0 | 0 | 0 |
| texas | low | 2 | 0.0 | 0 | 0 | 0 |
| texas | medium | 9 | 0.0 | 0 | 0 | 0 |
| texas | high | 25 | 0.0 | 0 | 0 | 0 |
| wisconsin | very_low | 2 | 0.5 | 0 | 0 | 0 |
| wisconsin | low | 2 | 0.0 | 0 | 0 | 0 |
| wisconsin | medium | 5 | 0.0 | 0 | 0 | 0 |
| wisconsin | high | 42 | 0.0 | 0 | 0 | 0 |
| cornell | very_low | 2 | 0.5 | 0 | 0 | 0 |
| cornell | low | 5 | 0.0 | 0 | 0 | 0 |
| cornell | medium | 4 | 0.0 | 0 | 0 | 0 |
| cornell | high | 26 | 0.0 | 0 | 0 | 0 |
| citeseer | very_low | 26 | 0.7692 | 0 | 0 | 0 |
| citeseer | low | 47 | 0.5106 | 0 | 0 | 0 |
| citeseer | medium | 73 | 0.0 | 0 | 0 | 0 |
| citeseer | high | 520 | 0.0 | 0 | 0 | 0 |
| cora | very_low | 26 | 0.2692 | 1 | 0 | 1 |
| cora | low | 34 | 0.0 | 0 | 0 | 0 |
| cora | medium | 49 | 0.0 | 0 | 0 | 0 |
| cora | high | 388 | 0.0 | 0 | 0 | 0 |
| pubmed | very_low | 80 | 0.7 | 5 | 0 | 5 |
| pubmed | low | 138 | 0.3768 | 2 | 0 | 2 |
| pubmed | medium | 238 | 0.0 | 0 | 0 | 0 |
| pubmed | high | 3488 | 0.0 | 0 | 0 | 0 |

## 6. Degree-Binned Diagnostics

### FINAL_V3

| Dataset | degree_bin | n_nodes | frac_corrected | n_helped | n_hurt | net_help |
|---|---|---|---|---|---|---|
| chameleon | 0-2 | 37 | 0.0541 | 0 | 0 | 0 |
| chameleon | 3-5 | 72 | 0.0417 | 0 | 0 | 0 |
| chameleon | 6-10 | 93 | 0.0538 | 0 | 0 | 0 |
| chameleon | >10 | 254 | 0.0433 | 0 | 0 | 0 |
| squirrel | 0-2 | 64 | 0.0469 | 0 | 0 | 0 |
| squirrel | 3-5 | 128 | 0.0312 | 1 | 0 | 1 |
| squirrel | 6-10 | 180 | 0.0889 | 1 | 3 | -2 |
| squirrel | >10 | 669 | 0.0613 | 1 | 1 | 0 |
| texas | 0-2 | 20 | 0.0 | 0 | 0 | 0 |
| texas | 3-5 | 13 | 0.0 | 0 | 0 | 0 |
| texas | 6-10 | 2 | 0.0 | 0 | 0 | 0 |
| texas | >10 | 2 | 0.0 | 0 | 0 | 0 |
| wisconsin | 0-2 | 23 | 0.0 | 0 | 0 | 0 |
| wisconsin | 3-5 | 20 | 0.05 | 0 | 0 | 0 |
| wisconsin | 6-10 | 6 | 0.0 | 0 | 0 | 0 |
| wisconsin | >10 | 2 | 0.0 | 0 | 0 | 0 |
| cornell | 0-2 | 28 | 0.0 | 0 | 0 | 0 |
| cornell | 3-5 | 7 | 0.1429 | 0 | 0 | 0 |
| cornell | 6-10 | 1 | 0.0 | 0 | 0 | 0 |
| cornell | >10 | 1 | 1.0 | 0 | 0 | 0 |
| citeseer | 0-2 | 433 | 0.2471 | 14 | 4 | 10 |
| citeseer | 3-5 | 166 | 0.2169 | 2 | 2 | 0 |
| citeseer | 6-10 | 53 | 0.1509 | 1 | 0 | 1 |
| citeseer | >10 | 14 | 0.0714 | 0 | 0 | 0 |
| cora | 0-2 | 183 | 0.2131 | 7 | 1 | 6 |
| cora | 3-5 | 235 | 0.2596 | 8 | 0 | 8 |
| cora | 6-10 | 57 | 0.2982 | 3 | 0 | 3 |
| cora | >10 | 22 | 0.2273 | 0 | 1 | -1 |
| pubmed | 0-2 | 2476 | 0.2342 | 71 | 28 | 43 |
| pubmed | 3-5 | 620 | 0.2242 | 14 | 1 | 13 |
| pubmed | 6-10 | 368 | 0.2473 | 4 | 0 | 4 |
| pubmed | >10 | 480 | 0.2667 | 16 | 1 | 15 |

### MS_HSGC

| Dataset | degree_bin | n_nodes | frac_corrected | n_helped | n_hurt | net_help |
|---|---|---|---|---|---|---|
| chameleon | 0-2 | 37 | 0.0 | 0 | 0 | 0 |
| chameleon | 3-5 | 72 | 0.0 | 0 | 0 | 0 |
| chameleon | 6-10 | 93 | 0.0108 | 0 | 0 | 0 |
| chameleon | >10 | 254 | 0.0039 | 0 | 0 | 0 |
| squirrel | 0-2 | 64 | 0.0156 | 0 | 0 | 0 |
| squirrel | 3-5 | 128 | 0.0078 | 0 | 0 | 0 |
| squirrel | 6-10 | 180 | 0.0111 | 0 | 0 | 0 |
| squirrel | >10 | 669 | 0.009 | 0 | 0 | 0 |
| texas | 0-2 | 20 | 0.0 | 0 | 0 | 0 |
| texas | 3-5 | 13 | 0.0 | 0 | 0 | 0 |
| texas | 6-10 | 2 | 0.0 | 0 | 0 | 0 |
| texas | >10 | 2 | 0.0 | 0 | 0 | 0 |
| wisconsin | 0-2 | 23 | 0.0435 | 0 | 0 | 0 |
| wisconsin | 3-5 | 20 | 0.0 | 0 | 0 | 0 |
| wisconsin | 6-10 | 6 | 0.0 | 0 | 0 | 0 |
| wisconsin | >10 | 2 | 0.0 | 0 | 0 | 0 |
| cornell | 0-2 | 28 | 0.0 | 0 | 0 | 0 |
| cornell | 3-5 | 7 | 0.0 | 0 | 0 | 0 |
| cornell | 6-10 | 1 | 0.0 | 0 | 0 | 0 |
| cornell | >10 | 1 | 1.0 | 0 | 0 | 0 |
| citeseer | 0-2 | 433 | 0.0716 | 0 | 0 | 0 |
| citeseer | 3-5 | 166 | 0.0723 | 0 | 0 | 0 |
| citeseer | 6-10 | 53 | 0.0189 | 0 | 0 | 0 |
| citeseer | >10 | 14 | 0.0 | 0 | 0 | 0 |
| cora | 0-2 | 183 | 0.0328 | 0 | 0 | 0 |
| cora | 3-5 | 235 | 0.0043 | 1 | 0 | 1 |
| cora | 6-10 | 57 | 0.0 | 0 | 0 | 0 |
| cora | >10 | 22 | 0.0 | 0 | 0 | 0 |
| pubmed | 0-2 | 2476 | 0.0214 | 0 | 0 | 0 |
| pubmed | 3-5 | 620 | 0.029 | 2 | 0 | 2 |
| pubmed | 6-10 | 368 | 0.0489 | 0 | 0 | 0 |
| pubmed | >10 | 480 | 0.0396 | 5 | 0 | 5 |

## 7. Summary Questions

**Q1: On which datasets does FINAL_V3 help most?**
Top datasets by delta vs MLP: cora (+0.0322), pubmed (+0.0190), citeseer (+0.0165)

**Q2: On which datasets does MS_HSGC engage routing meaningfully?**
Top datasets by frac_1hop: citeseer (0.038), pubmed (0.021), cora (0.012)

**Q3: On which datasets is MS_HSGC mostly collapsing back to MLP?**
Datasets with frac_1hop < 0.01 and frac_2hop < 0.01: chameleon, squirrel, texas

**Q4: Is 2-hop routing being used at all, and where?**
Datasets with frac_2hop > 0.5%: citeseer (0.029), cornell (0.027), wisconsin (0.020), squirrel (0.008), pubmed (0.006)

**Q5: In which margin bins do corrections help most?**
FINAL_V3: Best margin bin overall is `low` (highest net_help).
MS_HSGC: Best margin bin overall is `very_low` (highest net_help).

**Q6: In which degree bins do corrections help or hurt?**
  FINAL_V3: Best degree bin = `0-2` (net_help=59), worst = `6-10` (net_help=6).
  MS_HSGC: Best degree bin = `>10` (net_help=5), worst = `0-2` (net_help=0).

**Q7: Does DeltaH correlate with useful 2-hop routing?**
Yes: High-DeltaH nodes (Q4) have mean frac_2hop=0.017 vs low-DeltaH (Q1) frac_2hop=0.000. 2-hop routing activates more where DeltaH is positive.

**Q8: What appears to be the main failure mode of MS_HSGC?**
Routing is moderately active (routed_frac=59.5%). No single dominant failure mode — review net_help per dataset.

## 8. Key Insights

1. FINAL_V3 helps most on **cora** (delta=+0.0322) and hurts/is neutral on **squirrel** (delta=-0.0009).
2. MS_HSGC routing is active (>2% corrected) on: [cornell, citeseer, pubmed]. Mostly dormant (≤1% corrected) on: [chameleon, squirrel, texas].
3. 2-hop routing is non-trivially active (mean_frac_2hop ≈ 0.012). DeltaH signal may be informative — investigate hetero-bin diagnostics.

## 9. Next Hypotheses

1. Lower the routing thresholds (rho1, rho2, h1_max) in a dedicated light grid to see if increased engagement improves accuracy, or hurts it. This tests whether the bottleneck is gate conservatism.
2. Run a 3-split (splits 0, 1, 2) version of this sweep on the top-3 datasets where FINAL_V3 showed the largest deltas, to build statistical confidence before any full 10-split benchmark.

---
*Reconnaissance sweep — non-canonical diagnostic. Do not cite in the paper without re-running with canonical settings.*
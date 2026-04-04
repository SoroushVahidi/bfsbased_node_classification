# MS_HSGC Medium Run — Comparison Table

**Status:** EXPERIMENTAL, non-canonical.  
**Run:** 2026-04-04 | Datasets: chameleon, citeseer, texas | Splits: 0, 1, 2  
**Grid:** 24 configs (tau∈{0.15,0.25,0.35}, rho1∈{0.3,0.4}, rho2∈{0.3,0.35}, h1_max∈{0.55,0.70})

## Mean Test Accuracy (3 splits)

| Method          | chameleon | citeseer | texas  | Mean   |
|-----------------|-----------|----------|--------|--------|
| MLP_ONLY        | 0.4510    | 0.6812   | 0.8198 | 0.6507 |
| FINAL_V3        | 0.4532    | 0.6987   | 0.8198 | 0.6572 |
| MS_HSGC (medium)| **0.4547**| 0.6857   | 0.8198 | 0.6534 |

## Mean Delta vs MLP

| Method          | chameleon | citeseer  | texas  | Mean    |
|-----------------|-----------|-----------|--------|---------|
| FINAL_V3        | +0.0022   | +0.0175   | 0.0000 | +0.0066 |
| MS_HSGC (medium)| **+0.0037**| +0.0045  | 0.0000 | +0.0027 |

## MS_HSGC vs FINAL_V3

| Dataset   | Delta (MS_HSGC − FINAL_V3) | Verdict                |
|-----------|---------------------------|------------------------|
| chameleon | **+0.0015**               | MS_HSGC edges out V3   |
| citeseer  | −0.0130                   | V3 still leads         |
| texas     | 0.0000                    | Tied (both safe)       |
| **Overall** | **−0.0039**             | V3 still ahead overall |

## Routing Summary (avg over splits 0–2)

| Dataset   | frac_confident | frac_1hop | frac_2hop | frac_fallback | main_block   |
|-----------|----------------|-----------|-----------|---------------|--------------|
| chameleon | ~0.91          | ~1.0%     | ~4.3%     | ~4.7%         | H1 gate      |
| citeseer  | ~0.87          | ~2.6%     | ~2.8%     | ~4.7%         | H1 + R2 gate |
| texas     | ~0.99          | ~1.9%     | ~1.8%     | ~0.4%         | (tiny unc)   |

## Correction Quality (all splits combined)

| Hop   | Total helped | Total hurt | Net gain | Precision |
|-------|-------------|------------|----------|-----------|
| 1-hop | 5           | 1          | +4       | 0.83      |
| 2-hop | 12          | 2          | **+10**  | 0.86      |
| **Both** | **17**   | **3**      | **+14**  | **0.85**  |

_Files: `reports/ms_hsgc_results_medium.csv`, `reports/ms_hsgc_vs_final_v3_medium.csv`_

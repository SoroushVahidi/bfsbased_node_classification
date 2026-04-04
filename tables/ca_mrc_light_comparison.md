# CA-MRC Light Run — Comparison Table

**Status:** EXPERIMENTAL, non-canonical.  
**Run:** 2026-04-04 | Datasets: chameleon, citeseer, texas | Splits: 0, 1  
**Light settings:** MLP epochs=100, grid = 108 configs

## Mean Test Accuracy (2 splits)

| Method                | chameleon | citeseer  | texas  | Mean   |
|-----------------------|-----------|-----------|--------|--------|
| MLP_ONLY              | 0.4452    | 0.6862    | 0.8378 | 0.6564 |
| FINAL_V3              | 0.4452    | 0.7057    | 0.8243 | 0.6584 |
| CA_MRC[full]          | 0.4430    | **0.7155**| 0.8243 | 0.6609 |
| CA_MRC[no_compat]     | 0.4474    | **0.7177**| **0.8378** | **0.6676** |
| CA_MRC[entropy_only]  | 0.4430    | 0.7155    | 0.8243 | 0.6609 |
| CA_MRC[mag_only]      | 0.4419    | 0.7102    | 0.7973 | 0.6498 |

## Mean Delta vs MLP

| Method                | chameleon  | citeseer   | texas    | Mean    |
|-----------------------|------------|------------|----------|---------|
| FINAL_V3              | +0.0000    | +0.0195    | −0.0135  | +0.0020 |
| CA_MRC[full]          | −0.0022    | +0.0293    | −0.0135  | +0.0045 |
| CA_MRC[no_compat]     | **+0.0022**| **+0.0315**| **0.0000** | **+0.0112** |
| CA_MRC[entropy_only]  | −0.0022    | +0.0293    | −0.0135  | +0.0045 |
| CA_MRC[mag_only]      | −0.0033    | +0.0240    | −0.0405  | −0.0066 |

## CA_MRC[full] vs FINAL_V3

| Dataset   | Mean delta | Verdict                          |
|-----------|-----------|----------------------------------|
| chameleon | −0.0022   | CA_MRC slightly below V3         |
| citeseer  | **+0.0098** | **CA_MRC outperforms FINAL_V3** |
| texas     | 0.0000    | Tied                              |
| **Overall** | **+0.0025** | CA_MRC edges out FINAL_V3      |

## Ablation: Compatibility vs No-Compat

| Dataset   | CA_MRC[full] | CA_MRC[no_compat] | Winner     |
|-----------|-------------|-------------------|------------|
| chameleon | 0.4430      | 0.4474            | no_compat  |
| citeseer  | 0.7155      | 0.7177            | no_compat  |
| texas     | 0.8243      | **0.8378**        | no_compat  |

**Finding:** `no_compat` (plain multi-hop residual, no compatibility matrix) outperforms the compatibility-aware version across all three datasets. Compatibility estimation from sparse training edges appears to add noise rather than signal in this light run.

## Gate Ablation Summary

| Gate mode    | Mean accuracy | Mean delta vs MLP | Safety on Texas |
|-------------|--------------|-------------------|----------------|
| full (H+R)  | 0.6609       | +0.0045           | matches V3 (−0.0135 on split 1) |
| no_compat   | 0.6676       | +0.0112           | **safe (0.0000)** |
| entropy_only| 0.6609       | +0.0045           | same as full   |
| mag_only    | 0.6498       | −0.0066           | most harmful   |

Combined entropy+magnitude gate is correct direction. Magnitude-only gate is clearly harmful.

_Files: `reports/ca_mrc_results_light.csv`, `reports/ca_mrc_vs_final_v3_light.csv`_

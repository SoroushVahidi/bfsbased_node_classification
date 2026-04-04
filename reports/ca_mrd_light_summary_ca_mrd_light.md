# CA-MRD Light Experiment — Summary

**Status:** NON-CANONICAL experimental run. Do not cite in manuscript.
**Date:** 2026-04-04
**Mode:** LIGHT (reduced MLP epochs + LIGHT_GRID)
**Datasets:** chameleon, citeseer, texas   **Splits:** [0, 1]

## 1. Per-Dataset Test Accuracy

| Dataset | Split | MLP | FINAL_V3 | CA_MRD | V3−MLP | CA−MLP | CA−V3 | hop_profile | rho_H |
|---|---|---|---|---|---|---|---|---|---|
| chameleon | 0 | 0.4496 | 0.4496 | 0.4496 | +0.0000 | +0.0000 | +0.0000 | profile_C | 0.5 |
| chameleon | 1 | 0.4408 | 0.4408 | 0.4386 | +0.0000 | -0.0022 | -0.0022 | profile_C | 0.0 |
| citeseer | 0 | 0.6757 | 0.6922 | 0.6832 | +0.0165 | +0.0075 | -0.0090 | profile_A | 0.0 |
| citeseer | 1 | 0.6967 | 0.7192 | 0.7192 | +0.0225 | +0.0225 | +0.0000 | profile_C | 0.0 |
| texas | 0 | 0.7568 | 0.7568 | 0.7568 | +0.0000 | +0.0000 | +0.0000 | profile_A | 0.0 |
| texas | 1 | 0.9189 | 0.8919 | 0.9189 | -0.0270 | +0.0000 | +0.0270 | profile_A | 0.0 |

## 2. CA_MRD Gate Diagnostics

| Dataset | Split | frac_corrected | n_helped | n_hurt | net_help | corr_prec | rho_H | lambda | tau_u | tau_r |
|---|---|---|---|---|---|---|---|---|---|---|
| chameleon | 0 | 0.5461 | 0 | 0 | 0 | 0.0000 | 0.5000 | 1.0000 | 0.1000 | 0.0000 |
| chameleon | 1 | 0.5811 | 2 | 3 | -1 | 0.2222 | 0.0000 | 1.0000 | 0.1000 | 0.0000 |
| citeseer | 0 | 0.2342 | 8 | 3 | 5 | 0.5714 | 0.0000 | 1.0000 | 0.1000 | 0.0000 |
| citeseer | 1 | 0.3003 | 16 | 1 | 15 | 0.8000 | 0.0000 | 1.0000 | 0.1000 | 0.0000 |
| texas | 0 | 0.3514 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.5000 | 0.1000 | 0.0000 |
| texas | 1 | 0.1892 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.5000 | 0.1000 | 0.0000 |

## 3. Answers to Key Questions

**chameleon:** MLP→FINAL_V3 delta=+0.0000  |  MLP→CA_MRD delta=-0.0011  |  CA_MRD vs FINAL_V3=-0.0011
**citeseer:** MLP→FINAL_V3 delta=+0.0195  |  MLP→CA_MRD delta=+0.0150  |  CA_MRD vs FINAL_V3=-0.0045
**texas:** MLP→FINAL_V3 delta=-0.0135  |  MLP→CA_MRD delta=+0.0000  |  CA_MRD vs FINAL_V3=+0.0135

**Q4: Does H_reg help versus H = I (rho_H = 0.0)?**
Most-selected rho_H = 0.0  (counts: 0.0→5, 0.5→1).
The identity (rho_H=0) is preferred — compatibility modeling appears not helpful in this run. H_reg ≈ I.

**Q5: Do multi-hop profiles outperform 1-hop only?**
Most-selected profile: profile_C  (counts: profile_A→3, profile_C→3).
Multi-hop profile (profile_C) is preferred over 1-hop only.

**Q7: Is CA-MRD more promising than MS_HSGC?**
CA_MRD achieves positive delta vs MLP on at least one dataset, suggesting it engages more than MS_HSGC in light runs. Whether this generalises requires a full 10-split run.

**Overall verdict:**
CA_MRD mean delta vs FINAL_V3 = +0.0026. CA_MRD is competitive with FINAL_V3 on average in this light run. Promising enough to warrant a 10-split follow-up.

---
*Non-canonical diagnostic. All canonical FINAL_V3 outputs are untouched.*
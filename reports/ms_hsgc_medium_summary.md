# MS_HSGC Medium-Tuning Experiment — Summary

**Status:** EXPERIMENTAL, non-canonical. Results from a medium tuning run only.  
**Date:** 2026-04-04  
**Datasets:** chameleon, citeseer, texas | **Splits:** 0, 1, 2  
**Medium settings:** MLP epochs=200 (canonical: 300), MS_HSGC grid = 24 configs

Grid parameters:
- `tau` ∈ {0.15, 0.25, 0.35} (up from {0.05, 0.10, 0.20} in light mode)
- `rho1` ∈ {0.3, 0.4}
- `rho2` ∈ {0.3, 0.35}
- `h1_max` ∈ {0.55, 0.70} (up from {0.50} in light mode)
- `delta_min` = 0.0 (fixed)
- `pi1` = 0, `pi2` = 0 (balanced profiles)

Output files:
- `reports/ms_hsgc_results_medium.csv` — per-row results (MLP_ONLY / FINAL_V3 / MS_HSGC)
- `reports/ms_hsgc_vs_final_v3_medium.csv` — per-split delta MS_HSGC − FINAL_V3
- `tables/ms_hsgc_medium_comparison.md` — comparison table

---

## 1. Per-Dataset Mean Test Accuracy (splits 0–2)

| Dataset    | MLP_ONLY | FINAL_V3 | MS_HSGC | MS_HSGC−FINAL_V3 | MS_HSGC−MLP |
|------------|----------|----------|---------|------------------|-------------|
| chameleon  |  0.4510  |  0.4532  | 0.4547  |       **+0.0015**|    +0.0037  |
| citeseer   |  0.6812  |  0.6987  | 0.6857  |         −0.0130  |    +0.0045  |
| texas      |  0.8198  |  0.8198  | 0.8198  |          0.0000  |     0.0000  |
| **Overall**| **0.6507**|**0.6572**|**0.6534**|      **−0.0039** | **+0.0027** |

---

## 2. Per-Split Results

| Dataset   | Split | MLP    | FINAL_V3 | MS_HSGC | MS_HSGC−V3 |
|-----------|-------|--------|----------|---------|-----------|
| chameleon | 0     | 0.4364 | 0.4408   | 0.4386  | −0.0022   |
| chameleon | 1     | 0.4561 | 0.4561   | 0.4649  | **+0.0088** |
| chameleon | 2     | 0.4605 | 0.4627   | 0.4605  | −0.0022   |
| citeseer  | 0     | 0.6772 | 0.7012   | 0.6802  | −0.0210   |
| citeseer  | 1     | 0.6817 | 0.7027   | 0.6847  | −0.0180   |
| citeseer  | 2     | 0.6847 | 0.6922   | 0.6922  | **0.0000** |
| texas     | 0     | 0.7568 | 0.7568   | 0.7568  | 0.0000    |
| texas     | 1     | 0.8919 | 0.8919   | 0.8919  | 0.0000    |
| texas     | 2     | 0.8108 | 0.8108   | 0.8108  | 0.0000    |

---

## 3. Routing Diagnostics (test set, splits 0–2)

### Chameleon
| Split | unc_total | blk_h1 | blk_r1 | blk_r2 | blk_delta | routed_1h | routed_2h | fallback |
|-------|-----------|--------|--------|--------|-----------|-----------|-----------|---------|
| 0     | 37        | 31     | 0      | 8      | 8         | 1 (1.1%)  | 3 (3.3%)  | 24      |
| 1     | 38        | 32     | 0      | 3      | 4         | 1 (1.1%)  | 5 (5.3%)  | 25      |
| 2     | 38        | 34     | 0      | 5      | 9         | 1 (0.9%)  | 4 (4.4%)  | 24      |

**Main bottleneck:** `blocked_by_h1_only` — 31–34 of ~38 uncertain nodes are blocked because H1 > h1_max (even at h1_max=0.70). Chameleon's mean H1 ≈ 0.76, so most uncertain nodes have 1-hop neighborhoods too heterophilic to trust.

### Citeseer
| Split | unc_total | blk_h1 | blk_r1 | blk_r2 | blk_delta | routed_1h | routed_2h | fallback |
|-------|-----------|--------|--------|--------|-----------|-----------|-----------|---------|
| 0     | 54        | 26     | 2      | 13     | 1         | 3 (3.2%)  | 3 (2.7%)  | 20      |
| 1     | 67        | 40     | 3      | 25     | 2         | 2 (2.3%)  | 3 (3.2%)  | 37      |
| 2     | 41        | 23     | 1      | 8      | 1         | 2 (2.3%)  | 2 (2.4%)  | 15      |

**Main bottlenecks:** `blocked_by_h1_only` (largest) and `blocked_by_r2_only` (significant). Citeseer still routes only ~5% of uncertain nodes to correction vs ~18 corrected nodes in FINAL_V3.

### Texas
| Split | unc_total | routed_1h | routed_2h | Note |
|-------|-----------|-----------|-----------|------|
| 0     | 2         | 0         | 1         | Almost entirely confident |
| 1     | 1         | 1         | 0         | Almost entirely confident |
| 2     | 4         | 1         | 2         | Almost entirely confident |

Texas MLP is so high-confidence that only 1–4 test nodes are uncertain per split. Zero corrections in all splits — correctly conservative.

---

## 4. Per-Hop Route Quality

| Dataset   | Split | helped_1h | hurt_1h | net_1h | helped_2h | hurt_2h | net_2h |
|-----------|-------|-----------|---------|--------|-----------|---------|--------|
| chameleon | 0     | 1         | 0       | +1     | 0         | 0       | 0      |
| chameleon | 1     | 0         | 0       | 0      | 4         | 0       | +4     |
| chameleon | 2     | 0         | 0       | 0      | 0         | 0       | 0      |
| citeseer  | 0     | 1         | 1       | 0      | 3         | 1       | +2     |
| citeseer  | 1     | 1         | 0       | +1     | 2         | 1       | +1     |
| citeseer  | 2     | 2         | 0       | +2     | 3         | 0       | +3     |
| texas     | 0–2   | 0         | 0       | 0      | 0         | 0       | 0      |

**Key finding:** 2-hop corrections are helping more than hurting (net_2h ≥ 0 in all cases). 2-hop is generating meaningful signal on Chameleon (split 1: +4 from 2-hop alone).

---

## 5. Answers to Key Questions

**Q1: Does MS_HSGC now engage routing meaningfully?**  
Yes. The medium mode successfully activates routing: 1–5% of test nodes reach 1-hop or 2-hop correction per split (vs ~0.2–0.9% in light mode). The H1 gate remains the dominant bottleneck, but nodes ARE reaching correction channels.

**Q2: Does Chameleon improve vs FINAL_V3?**  
**Yes on average (+0.0015).** On split 1, MS_HSGC gains +0.0088 over FINAL_V3 through 5 2-hop corrections (4 helped, 0 hurt). On splits 0 and 2, MS_HSGC is within −0.0022 of FINAL_V3. This is a promising result for a heterophilic dataset.

**Q3: Can Citeseer recover FINAL_V3's gain?**  
Partially. MS_HSGC gains +0.0045 over MLP (vs FINAL_V3's +0.0175), reaching FINAL_V3 exactly on split 2 (both at 0.6922). However, the −0.0130 average gap shows that H1 filtering is still blocking most of FINAL_V3's corrected nodes on Citeseer. The H1 gate is the primary remaining bottleneck.

**Q4: Does Texas preserve safety?**  
**Yes, perfectly.** All 9 MS_HSGC results on Texas exactly match MLP. Zero harmful corrections. The method correctly identifies that Texas test nodes are almost entirely confident (only 1–4 uncertain per split).

**Q5: Are 2-hop corrections helping more than hurting?**  
**Yes.** Across all splits and datasets: total helped_2hop = 12, total hurt_2hop = 2. Net 2-hop gain = +10. 2-hop is the more reliable correction channel in this medium run. 1-hop: total helped = 5, hurt = 1, net = +4.

**Q6: Which gating condition is the main bottleneck now?**  
**`blocked_by_h1_only`** is the dominant bottleneck in both Chameleon and Citeseer. Even with h1_max raised to 0.70, most uncertain nodes have H1 > 0.70 (mean H1 ≈ 0.76 on Chameleon). Secondary bottleneck on Citeseer: `blocked_by_r2_only` (R2 < rho2 threshold).

---

## 6. Diagnosis

**Verdict: Balanced / Promising**

The method is no longer over-blocked. Routing is engaging meaningfully:
- Chameleon: **MS_HSGC now edges out FINAL_V3** on the 3-split average (+0.0015)
- Texas: **Perfect safety maintained** (zero harmful corrections)
- Citeseer: Still lagging FINAL_V3 (−0.0130) but making correct net-positive corrections

The H1 gate is the last major conservative force. This is appropriate: it prevents correction in contexts where the local neighborhood is highly heterophilic and therefore untrustworthy. The question for the 10-split run is whether the gains on Chameleon are consistent.

**Is the method ready for a 10-split run?**  
**Yes, tentatively.** The medium results show:
1. No harmful corrections on Texas (0/0 hurt across 3 splits)
2. Positive net corrections on Chameleon and Citeseer (net gain = +4 and +8 respectively)
3. 2-hop channel is active and beneficial

A 10-split run would provide statistical confidence to answer whether MS_HSGC genuinely improves on Chameleon vs FINAL_V3 or whether split 1's +0.0088 is a lucky outlier.

---

## 7. Compact Comparison Table

```
Method          chameleon   citeseer    texas    Mean
-------------------------------------------------------
MLP_ONLY          0.4510     0.6812    0.8198   0.6507
FINAL_V3          0.4532     0.6987    0.8198   0.6572
MS_HSGC(medium)   0.4547     0.6857    0.8198   0.6534

Delta vs MLP:
FINAL_V3         +0.0022    +0.0175    0.0000  +0.0066
MS_HSGC(medium)  +0.0037    +0.0045    0.0000  +0.0027

MS_HSGC vs FINAL_V3:
                 +0.0015    −0.0130    0.0000  −0.0039
```

---

## 8. Code and Files Modified

| File | Change |
|------|--------|
| `code/bfsbased_node_classification/ms_hsgc.py` | Added `_compute_routing_block_diagnostics()`, per-hop quality metrics (helped/hurt/net/avg_margin per hop), routing-block diagnostics in `info_dict` |
| `code/bfsbased_node_classification/run_ms_hsgc_evaluation.py` | Added `MEDIUM_GRID` (24 configs), `MEDIUM_MLP_KWARGS` (epochs=200), `--medium` CLI flag, updated CSV fields, detailed diagnostic printing |

**Canonical files NOT touched**: `reports/final_method_v3_results.csv` unchanged.

---

## 9. Commands Used

```bash
# Medium tuning run
python3 code/bfsbased_node_classification/run_ms_hsgc_evaluation.py \
  --medium \
  --datasets chameleon citeseer texas \
  --splits 0 1 2 \
  --split-dir data/splits

# Outputs written to:
#   reports/ms_hsgc_results_medium.csv
#   reports/ms_hsgc_vs_final_v3_medium.csv
#   reports/ms_hsgc_medium_summary.md (this file)
```

---

## 10. Recommended Next Step

Run 10-split evaluation on Chameleon, Citeseer, Texas in medium mode to obtain reliable mean ± std estimates. Based on 3-split results, expect:
- Chameleon: MS_HSGC ≈ FINAL_V3 or slightly above
- Citeseer: MS_HSGC gains vs MLP but remains below FINAL_V3 (unless H1 gate is further relaxed)
- Texas: All methods tied at MLP (expected)

If 10-split results confirm Chameleon improvement, the method merits a broader 6-dataset evaluation.

# Project Overview

This repository studies selective graph correction for node classification with an MLP-first design.

## Motivation

Graph propagation can help on difficult nodes but can also hurt confident predictions. The central question is whether we can use graph structure **selectively**, based on reliability, instead of applying it uniformly.

## Canonical contribution

The canonical method is **FINAL_V3**:
- feature-first MLP baseline,
- reliability gate from MLP confidence,
- correction only on low-reliability nodes.

Canonical implementation entry point:
- `code/final_method_v3.py`

## Main evidence

Primary evidence is maintained in:
- `reports/final_method_v3_results.csv`
- `tables/main_results_selective_correction.md`
- `reports/final_method_v3_analysis.md`
- `reports/safety_analysis.md`

## What is not canonical

- Original UG-SGC line (legacy)
- Structural extension line (exploratory)
- Venue-specific packaging artifacts

These are preserved in `archive/` for provenance and reproducibility.

## Recommended reading order

1. `README.md` (including the paper evidence map)
2. `docs/METHOD_STATUS.md`
3. `docs/REPRODUCIBILITY.md`
4. `docs/RESULTS_GUIDE.md`
5. `reports/SUPPORTING_BUNDLES_INDEX.md` (for a complete map of non-canonical bundles)

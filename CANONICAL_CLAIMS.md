# Canonical Claims (FINAL_V3)

## Canonical method

`FINAL_V3` (reliability-gated selective graph correction) is the only canonical method line in this repository.

Canonical method entry point:
- `code/final_method_v3.py`

## Scope of claims

Supported claims are limited to the canonical artifacts derived from:
- `reports/final_method_v3_results.csv`
- canonical tables in `tables/`
- canonical figures in `figures/`
- canonical analyses in `reports/`

The method is presented as a conservative MLP-first correction strategy. It does **not** claim universal superiority on every dataset/split.

## Exclusions

The following are preserved but not canonical:
- UG-SGC legacy path (`manuscript_runner.py` and archived UG-SGC assets)
- Structural extension / resubmission exploratory path (`resubmission_runner.py` and archived assets)
- Venue-specific packaging artifacts under `archive/legacy_venue_specific/`

## Reproducibility command

```bash
bash scripts/run_all_selective_correction_results.sh
```

See `docs/REPRODUCIBILITY.md` for full rerun instructions and safety notes.

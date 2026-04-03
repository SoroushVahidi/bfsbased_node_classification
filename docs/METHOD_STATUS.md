# Method Status

This repository uses three explicit labels only.

## Canonical

### FINAL_V3 (only canonical method)

- Status: **Canonical**
- Entry point: `code/final_method_v3.py`
- Core runner: `code/bfsbased_node_classification/run_final_evaluation.py`
- Main evidence: `reports/final_method_v3_results.csv` and derived tables/figures.

## Archived

### UG-SGC legacy line

- Status: **Archived**
- Main runner retained for provenance: `code/bfsbased_node_classification/manuscript_runner.py`
- Historical artifacts: `archive/legacy_venue_specific/`, `archive/legacy_logs/`
- Replaced by: `FINAL_V3`

### Venue-specific material

- Status: **Archived**
- Location: `archive/legacy_venue_specific/`
- Use for claims: **No** (provenance only)

## Exploratory

### Structural extension and side branches

- Status: **Exploratory**
- Example runner: `code/bfsbased_node_classification/resubmission_runner.py`
- Additional exploratory code: `archive/exploratory_code/`, `archive/exploratory/`
- Use for claims: not part of canonical main claims.

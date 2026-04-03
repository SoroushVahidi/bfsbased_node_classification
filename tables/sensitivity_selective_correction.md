# Reliability threshold ρ — sensitivity (supplementary)

Mean test Δ vs MLP (percentage points) on **Chameleon, Cora, Texas** when ρ is fixed in a grid and other hyperparameters follow the historical evaluation protocol.

**Source:** `reports/archive/superseded_final_tables_prl/final_tables_prl.md` (Table 8). This is a **coarse 3-split** sensitivity snapshot, **not** the full 10-split aggregate.

**Reading:** Very low ρ (e.g. 0.0) allows almost all uncertain nodes through the reliability check, increasing harmful corrections (`total_hurt_nodes`). Around **ρ ≈ 0.3–0.4** mean Δ stays strongest before collapsing; **ρ ≥ 0.5** suppresses graph correction heavily (means approach 0).

| ρ | Chameleon Δ (pp) | Cora Δ (pp) | Texas Δ (pp) | Mean | Total hurt |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | +0.15 | +2.21 | +0.90 | +1.09 | 11 |
| 0.2 | +0.15 | +2.21 | +0.90 | +1.09 | 11 |
| 0.3 | +0.15 | +2.08 | +0.90 | +1.04 | 10 |
| 0.4 | +0.15 | +1.48 | +0.00 | +0.54 | 5 |
| 0.5 | -0.07 | +0.47 | +0.00 | +0.13 | 1 |
| 0.6 | +0.00 | +0.00 | +0.00 | +0.00 | 0 |
| 0.7 | +0.00 | +0.00 | +0.00 | +0.00 | 0 |

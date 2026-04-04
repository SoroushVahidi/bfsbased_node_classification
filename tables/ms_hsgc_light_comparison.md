# MS_HSGC vs FINAL_V3 — Light Comparison Table

> **Non-canonical.** Lightweight debug run (100 MLP epochs, 3-config grid).
> Datasets: chameleon, citeseer, texas — splits 0 and 1 only.

## Accuracy (test set)

| Dataset    | split | MLP    | FINAL_V3 | MS_HSGC | Δ(MS−V3) |
|------------|-------|--------|----------|---------|----------|
| chameleon  |   0   | 0.4496 |  0.4496  | 0.4496  |  +0.000  |
| chameleon  |   1   | 0.4408 |  0.4408  | 0.4408  |  +0.000  |
| citeseer   |   0   | 0.6757 |  0.6922  | 0.6757  |  −0.017  |
| citeseer   |   1   | 0.6967 |  0.7192  | 0.6967  |  −0.023  |
| texas      |   0   | 0.7568 |  0.7568  | 0.7568  |  +0.000  |
| texas      |   1   | 0.9189 |  0.8919  | 0.9189  |  +0.027  |
| **mean**   |       |**0.6564**|**0.6584**|**0.6564**| **−0.002** |

## MS_HSGC Route Fractions (mean over 2 splits)

| Dataset   | confident | 1-hop | 2-hop | mlp_unc | H1    | H2    | ΔH    |
|-----------|-----------|-------|-------|---------|-------|-------|-------|
| chameleon |   0.97    | 0.002 | 0.009 | 0.019   | 0.757 | 0.710 | +0.047|
| citeseer  |   0.90    | 0.034 | 0.030 | 0.032   | 0.357 | 0.256 | +0.102|
| texas     |   0.99    | 0.000 | 0.014 | 0.000   | 0.834 | 0.489 | +0.344|

**Note:** MS_HSGC is over-conservative with the light 3-config grid.
A full 96-config run is required to evaluate the method fairly.

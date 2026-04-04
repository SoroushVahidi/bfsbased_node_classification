# FINAL_V3 correction case studies (20260404)

Small illustrative bundle: 3 beneficial + 2 harmful changed-node examples.

### Beneficial 1: pubmed split 0 node 17991
- MLP → FINAL_V3: 2 → 1 (true=1)
- Reliability=0.394, margin=0.016, score_gap=1.309
- Gaps: anchor=-0.031, prototype=0.053, neighbor=0.900, compatibility=0.387
- Dominant reason: **neighbor_support** (0.900)
- Interpretation: correction shifted to the final class because its combined structural/feature evidence outweighed the original class anchor.

### Beneficial 2: pubmed split 1 node 12429
- MLP → FINAL_V3: 2 → 1 (true=1)
- Reliability=0.394, margin=0.005, score_gap=1.284
- Gaps: anchor=-0.010, prototype=0.018, neighbor=0.900, compatibility=0.375
- Dominant reason: **neighbor_support** (0.900)
- Interpretation: correction shifted to the final class because its combined structural/feature evidence outweighed the original class anchor.

### Beneficial 3: citeseer split 0 node 226
- MLP → FINAL_V3: 2 → 4 (true=4)
- Reliability=0.443, margin=0.004, score_gap=1.283
- Gaps: anchor=-0.011, prototype=0.013, neighbor=0.900, compatibility=0.381
- Dominant reason: **neighbor_support** (0.900)
- Interpretation: correction shifted to the final class because its combined structural/feature evidence outweighed the original class anchor.

### Harmful 1: citeseer split 1 node 326
- MLP → FINAL_V3: 0 → 3 (true=0)
- Reliability=0.434, margin=0.289, score_gap=0.002
- Gaps: anchor=-1.181, prototype=-0.002, neighbor=0.876, compatibility=0.309
- Dominant reason: **neighbor_support** (0.876)
- Interpretation: correction shifted to the final class because its combined structural/feature evidence outweighed the original class anchor.

### Harmful 2: pubmed split 1 node 18823
- MLP → FINAL_V3: 0 → 1 (true=0)
- Reliability=0.505, margin=0.177, score_gap=0.004
- Gaps: anchor=-0.357, prototype=-0.017, neighbor=0.256, compatibility=0.122
- Dominant reason: **neighbor_support** (0.256)
- Interpretation: correction shifted to the final class because its combined structural/feature evidence outweighed the original class anchor.

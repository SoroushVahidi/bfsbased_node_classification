# FINAL_V3 harmful correction case studies (20260404)

Illustrative harmful changed nodes (lowest score-gap failures).

## citeseer split 1 node 326
- MLP→FINAL_V3: 0→3, true=0
- Reliability=0.434, margin=0.289, score_gap=0.002
- Contribution gaps: anchor=-1.181, prototype=-0.002, neighbor=0.876, compatibility=0.309
- Dominant driver: neighbor_support
- Failure reading: graph-driven shift overrode a stronger/safer MLP anchor with only marginal total-score advantage.
- Blocked by safeguard `gap_ge_0p05`: yes.

## pubmed split 1 node 18823
- MLP→FINAL_V3: 0→1, true=0
- Reliability=0.505, margin=0.177, score_gap=0.004
- Contribution gaps: anchor=-0.357, prototype=-0.017, neighbor=0.256, compatibility=0.122
- Dominant driver: neighbor_support
- Failure reading: graph-driven shift overrode a stronger/safer MLP anchor with only marginal total-score advantage.
- Blocked by safeguard `gap_ge_0p05`: yes.

## pubmed split 0 node 9217
- MLP→FINAL_V3: 2→1, true=2
- Reliability=0.394, margin=0.560, score_gap=0.014
- Contribution gaps: anchor=-1.288, prototype=0.014, neighbor=0.900, compatibility=0.387
- Dominant driver: neighbor_support
- Failure reading: graph-driven shift overrode a stronger/safer MLP anchor with only marginal total-score advantage.
- Blocked by safeguard `gap_ge_0p05`: yes.

## pubmed split 0 node 16167
- MLP→FINAL_V3: 0→1, true=0
- Reliability=0.500, margin=0.256, score_gap=0.026
- Contribution gaps: anchor=-0.526, prototype=-0.022, neighbor=0.392, compatibility=0.183
- Dominant driver: neighbor_support
- Failure reading: graph-driven shift overrode a stronger/safer MLP anchor with only marginal total-score advantage.
- Blocked by safeguard `gap_ge_0p05`: yes.

## pubmed split 1 node 9881
- MLP→FINAL_V3: 2→1, true=2
- Reliability=0.596, margin=0.389, score_gap=0.034
- Contribution gaps: anchor=-0.821, prototype=0.030, neighbor=0.579, compatibility=0.246
- Dominant driver: neighbor_support
- Failure reading: graph-driven shift overrode a stronger/safer MLP anchor with only marginal total-score advantage.
- Blocked by safeguard `gap_ge_0p05`: yes.

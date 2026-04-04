# Post hoc safeguard case studies (20260404)

Best safeguard selected: **R2_score_gap_lt_0p30** (Block correction if score_gap < 0.30.)

The following harmful changed-node cases would be blocked by the selected safeguard:

- Dataset `pubmed`, split `0`, node `9843`: reliability=0.361, score_gap=0.266, dominant_reason=neighbor_support, mlp_margin=0.013, prototype_gap=-0.025, neighbor_support_gap=0.238.
- Dataset `pubmed`, split `1`, node `7118`: reliability=0.367, score_gap=0.106, dominant_reason=neighbor_support, mlp_margin=0.180, prototype_gap=0.026, neighbor_support_gap=0.313.
- Dataset `pubmed`, split `1`, node `2474`: reliability=0.380, score_gap=0.100, dominant_reason=neighbor_support, mlp_margin=0.044, prototype_gap=-0.026, neighbor_support_gap=0.163.

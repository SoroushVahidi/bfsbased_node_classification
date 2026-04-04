# FINAL_V3 correction behavior narrative (lightweight)

FINAL_V3 acts conservatively: only uncertain nodes that also pass the reliability gate are routed.
The changed-fraction is smaller than the routed-fraction, showing bounded intervention even after routing.
Across datasets where MLP is already strong, correction remains mostly silent and mean deltas remain near zero.
On Cora/CiteSeer/PubMed-style settings, correction activates more often and yields positive average delta vs MLP.
Mean reliability on corrected nodes is consistently above mid-range, matching the intended gate semantics.

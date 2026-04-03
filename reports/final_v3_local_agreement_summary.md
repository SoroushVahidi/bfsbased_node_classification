# FINAL_V3 + local agreement gating summary

Tag: `targeted_core6`. Datasets: ['chameleon', 'texas', 'citeseer', 'cora', 'pubmed', 'wisconsin']. Splits: [0, 1, 2].

## Dataset-level outcomes

- **chameleon**: local-minus-canonical=-0.0007, low-bucket Δ vs MLP=+0.0022, high-conf untouched=1.000.
- **citeseer**: local-minus-canonical=+0.0000, low-bucket Δ vs MLP=+0.0721, high-conf untouched=1.000.
- **cora**: local-minus-canonical=+0.0000, low-bucket Δ vs MLP=+0.0944, high-conf untouched=1.000.
- **pubmed**: local-minus-canonical=+0.0000, low-bucket Δ vs MLP=+0.0360, high-conf untouched=1.000.
- **texas**: local-minus-canonical=+0.0000, low-bucket Δ vs MLP=+0.0256, high-conf untouched=1.000.
- **wisconsin**: local-minus-canonical=+0.0000, low-bucket Δ vs MLP=+0.0000, high-conf untouched=1.000.

## Notes

- Local gate attenuates graph-only terms; MLP term remains unchanged.
- No true test labels are used in gating; train labels + unlabeled soft MLP probs only.
- Search includes none/linear/hard-threshold/shifted-linear with eta grid {0.3..0.7}.

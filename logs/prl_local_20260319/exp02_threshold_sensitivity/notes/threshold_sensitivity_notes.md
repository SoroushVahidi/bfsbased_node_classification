# Exp02 threshold sensitivity notes

- Scope: datasets={cora,citeseer}, splits=0..9, repeats=1
- Fixed weights: full (b1=1.0,b2=0.4,b3=0.0,b4=0.9,b5=0.5)
- Threshold grid: [0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.65]

## Best thresholds by mean delta vs MLP
- cora: best threshold=0.65, mean_delta_vs_mlp=+0.0322, avg_uncertain=0.2310, avg_changed=0.0449
- citeseer: best threshold=0.65, mean_delta_vs_mlp=+0.0266, avg_uncertain=0.2461, avg_changed=0.0502

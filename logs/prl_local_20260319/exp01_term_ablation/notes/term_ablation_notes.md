# Exp01 term ablation notes

- Scope: datasets={cora,citeseer}, splits=0..9, repeats=1
- Configs: full, no_graph, no_feature_sim, no_compatibility
- Interpretation uses mean delta vs MLP and mean selective acc vs full.

## Quick summary
- cora: best config by selective acc = full (+0.0000 vs full), worst = no_graph (-0.0316 vs full)
- citeseer: best config by selective acc = full (+0.0000 vs full), worst = no_graph (-0.0263 vs full)

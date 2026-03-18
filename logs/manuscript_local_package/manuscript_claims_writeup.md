# Manuscript claim shaping and write-up support

## Candidate paper titles (5)
1. Feature-First Node Classification with Selective Graph Correction: A Regime-Aware Study
2. When Graph Correction Helps: A Feature-First, Uncertainty-Gated Approach to Node Classification
3. Selective Graph Correction for Node Classification: Strong on Some Regimes, Safe Fallback on Others
4. Beyond Propagation-First Design: Regime-Aware Selective Correction on Top of Feature Models
5. Interpretable Selective Graph Correction: Stable Gains in High-Confidence-Conflict Regimes

## Paper positioning statement
This paper studies node classification from a feature-first perspective: a strong MLP is the default predictor, and graph evidence is only applied as a selective correction step for uncertain nodes. Rather than arguing that graph propagation should dominate the pipeline, we ask when graph correction is beneficial and when it should be avoided. Across canonical split/repeat evaluations, selective correction shows stable gains on some datasets (notably Cora/Citeseer), near-neutral behavior on others (Actor/Chameleon/Texas/Cornell), and clear negative-control behavior (Wisconsin). The central contribution is therefore a regime-aware design and analysis: uncertainty-gated, interpretable correction terms that can improve accuracy in specific settings while remaining close to MLP behavior when graph evidence is weak.

## Contribution bullets (4-6)
- A feature-first selective correction framework where MLP predictions are only revised on uncertainty-gated nodes.
- An interpretable correction score combining MLP log-probability, feature similarity, neighbor support, and compatibility evidence.
- Validation-only selection of threshold and score weights, with no test-label leakage.
- Stable positive results on specific regimes (Cora/Citeseer) under canonical split/repeat evaluation.
- Regime diagnostics linking gains to uncertainty mass, threshold regimes, and weight-pattern stability.
- Case-level evidence showing nontrivial useful corrections on promising datasets and fallback-like behavior in weak regimes.

## Claims we can defend
- Selective graph correction is stably positive over MLP on Cora and Citeseer under canonical splits (0..9) and repeats=3.
- Actor and Cornell remain neutral in repeats=3 (no significant directional win/loss trend vs MLP).
- Wisconsin behaves as a negative control for this method (0/30 wins vs MLP in repeats=3).
- Threshold/weight behavior separates regimes: promising datasets concentrate in high-threshold, stable-weight settings; mixed/negative datasets concentrate in lower-threshold settings.
- The method is feature-first in practice: many regimes show low changed-from-MLP fractions and, in some datasets, frequent zero-changed runs.

## Claims we should avoid
- Universal improvement over MLP across graph datasets.
- Broad claim that selective correction outperforms propagation-first methods overall.
- Broad heterophily-first claim from current evidence alone.
- Claim that one fixed threshold/weight setting is globally optimal across regimes.

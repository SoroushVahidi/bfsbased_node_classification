# Manuscript decision memo

## Local-cycle decision summary
- Actor and Cornell repeats=3 were completed on canonical splits 0..9 (no random fallback).
- actor: delta(sel-mlp)=-0.0002, wins/losses/ties=16/14/0, p=0.856, class=neutral
- cornell: delta(sel-mlp)=-0.0027, wins/losses/ties=14/16/0, p=0.856, class=neutral

## Threshold/weight stability takeaways
- cora: threshold 0.592±0.071, dominant tuple frac 1.000, zero-changed 0/30, sel-mlp +0.0353
- citeseer: threshold 0.559±0.087, dominant tuple frac 1.000, zero-changed 0/30, sel-mlp +0.0244
- chameleon: threshold 0.115±0.067, dominant tuple frac 0.400, zero-changed 0/30, sel-mlp +0.0007
- texas: threshold 0.053±0.012, dominant tuple frac 0.900, zero-changed 17/30, sel-mlp -0.0018
- actor: threshold 0.122±0.098, dominant tuple frac 0.400, zero-changed 0/30, sel-mlp -0.0002
- cornell: threshold 0.050±0.000, dominant tuple frac 0.900, zero-changed 18/30, sel-mlp -0.0027
- wisconsin: threshold 0.060±0.037, dominant tuple frac 0.733, zero-changed 24/30, sel-mlp -0.0033

## Decision
- Continue as a **regime-aware method paper** candidate, with one final larger benchmark phase before drafting.
- Avoid claiming universal improvement over MLP or superiority over propagation-first methods.
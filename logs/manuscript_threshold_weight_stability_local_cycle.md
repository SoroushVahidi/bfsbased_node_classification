# Threshold/weight stability analysis (local manuscript cycle)

| dataset | thr mean ± std | thr cluster (low/mid/high) | dominant tuple | dom frac | avg uncertain | avg changed | zero-uncertain | zero-changed | sel-mlp | fallback indicator |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| cora | 0.5917 ± 0.0715 | 0/0/30 | (1.0, 0.4, 0.0, 0.9, 0.5) | 1.000 | 0.2142 | 0.0461 | 0/30 | 0/30 | +0.0353 | weak_or_not_fallback_dominant |
| citeseer | 0.5590 ± 0.0871 | 0/1/29 | (1.0, 0.4, 0.0, 0.9, 0.5) | 1.000 | 0.2138 | 0.0494 | 0/30 | 0/30 | +0.0244 | weak_or_not_fallback_dominant |
| chameleon | 0.1150 ± 0.0673 | 19/11/0 | (1.0, 0.4, 0.0, 0.9, 0.5) | 0.400 | 0.0692 | 0.0130 | 0/30 | 0/30 | +0.0007 | weak_or_not_fallback_dominant |
| texas | 0.0533 ± 0.0125 | 30/0/0 | (1.0, 0.6, 0.0, 0.5, 0.3) | 0.900 | 0.0360 | 0.0144 | 9/30 | 17/30 | -0.0018 | strong |
| actor | 0.1216 ± 0.0978 | 22/8/0 | (1.0, 0.9, 0.0, 0.3, 0.2) | 0.400 | 0.1616 | 0.0354 | 0/30 | 0/30 | -0.0002 | weak_or_not_fallback_dominant |
| cornell | 0.0500 ± 0.0000 | 30/0/0 | (1.0, 0.6, 0.0, 0.5, 0.3) | 0.900 | 0.0306 | 0.0135 | 8/30 | 18/30 | -0.0027 | strong |
| wisconsin | 0.0600 ± 0.0374 | 29/1/0 | (1.0, 0.6, 0.0, 0.5, 0.3) | 0.733 | 0.0144 | 0.0039 | 17/30 | 24/30 | -0.0033 | strong |

## Group summary
- promising_avg_threshold_mean: 0.5754
- controls_avg_threshold_mean: 0.0800
- promising_avg_dominant_weight_fraction: 1.0000
- controls_avg_dominant_weight_fraction: 0.6667
- promising_avg_zero_changed_fraction: 0.0000
- controls_avg_zero_changed_fraction: 0.3933
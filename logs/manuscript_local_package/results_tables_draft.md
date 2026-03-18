# Results presentation package (local)

## Clean main result table draft
| dataset | mlp_only (mean±std) | prop_only (mean±std) | selective (mean±std) | runtime selective(s) |
|---|---:|---:|---:|---:|
| cora | 0.7078±0.0208 | 0.8173±0.0230 | 0.7432±0.0208 | 0.1085 |
| citeseer | 0.6858±0.0204 | 0.6903±0.0279 | 0.7102±0.0223 | 0.2146 |
| chameleon | 0.4591±0.0185 | 0.3961±0.0379 | 0.4599±0.0195 | 0.2829 |
| texas | 0.7793±0.0554 | 0.5685±0.0991 | 0.7775±0.0566 | 0.0063 |
| actor | 0.3421±0.0093 | 0.3226±0.0192 | 0.3419±0.0088 | 0.3809 |
| cornell | 0.7342±0.0499 | 0.5045±0.0867 | 0.7315±0.0483 | 0.0064 |
| wisconsin | 0.8549±0.0474 | 0.5399±0.0866 | 0.8516±0.0466 | 0.0087 |

## Gain-over-MLP table draft
| dataset | selective-mlp | wins/losses/ties | sign-test p | avg uncertain | avg changed | bucket |
|---|---:|---:|---:|---:|---:|---|
| cora | +0.0353 | 30/0/0 | 1.86e-09 | 0.2142 | 0.0461 | promising |
| citeseer | +0.0244 | 30/0/0 | 1.86e-09 | 0.2138 | 0.0494 | promising |
| chameleon | +0.0007 | 16/14/0 | 0.856 | 0.0692 | 0.0130 | neutral |
| texas | -0.0018 | 11/19/0 | 0.2 | 0.0360 | 0.0144 | neutral |
| actor | -0.0002 | 16/14/0 | 0.856 | 0.1616 | 0.0354 | neutral |
| cornell | -0.0027 | 14/16/0 | 0.856 | 0.0306 | 0.0135 | neutral |
| wisconsin | -0.0033 | 0/30/0 | 1.86e-09 | 0.0144 | 0.0039 | not_worth_keeping |

## Regime-analysis table draft
| dataset | nodes | edges | classes | homophily | mlp | prop | selective | selective-mlp | uncertain | bucket |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| cora | 2708 | 10556 | 7 | 0.810 | 0.7078 | 0.8173 | 0.7432 | +0.0353 | 0.2142 | promising |
| citeseer | 3327 | 9104 | 6 | 0.736 | 0.6858 | 0.6903 | 0.7102 | +0.0244 | 0.2138 | promising |
| chameleon | 2277 | 62792 | 5 | 0.231 | 0.4591 | 0.3961 | 0.4599 | +0.0007 | 0.0692 | neutral |
| texas | 183 | 574 | 5 | 0.087 | 0.7793 | 0.5685 | 0.7775 | -0.0018 | 0.0360 | neutral |
| actor | 7600 | 53411 | 5 | 0.218 | 0.3421 | 0.3226 | 0.3419 | -0.0002 | 0.1616 | neutral |
| cornell | 183 | 557 | 5 | 0.127 | 0.7342 | 0.5045 | 0.7315 | -0.0027 | 0.0306 | neutral |
| wisconsin | 251 | 916 | 5 | 0.192 | 0.8549 | 0.5399 | 0.8516 | -0.0033 | 0.0144 | neutral |

## Case-study table draft
| dataset | node | mlp | selective | true | outcome | margin | dominant term | graph sup | feature sup |
|---|---:|---:|---:|---:|---|---:|---|---:|---:|
| citeseer | 1876 | 3 | 1 | 1 | helped | 0.0042 | mlp_term | 1.2218 | 0.2089 |
| citeseer | 2422 | 1 | 2 | 2 | helped | 0.0076 | mlp_term | 0.6634 | 0.2055 |
| citeseer | 2848 | 5 | 3 | 3 | helped | 0.0178 | mlp_term | 1.0060 | 0.2118 |
| citeseer | 369 | 1 | 2 | 2 | helped | 0.0479 | graph_neighbor_term | 1.2905 | 0.2008 |
| citeseer | 2013 | 0 | 2 | 2 | helped | 0.0496 | graph_neighbor_term | 1.2876 | 0.2018 |
| citeseer | 696 | 2 | 3 | 3 | helped | 0.0735 | mlp_term | 1.2525 | 0.2062 |
| citeseer | 2802 | 2 | 3 | 3 | helped | 0.0974 | mlp_term | 0.9001 | 0.2072 |
| citeseer | 785 | 1 | 3 | 3 | helped | 0.0991 | mlp_term | 1.2547 | 0.2038 |
| chameleon | 124 | 2 | 4 | 4 | helped | 0.0307 | mlp_term | 0.6925 | 0.1965 |
| chameleon | 1131 | 2 | 4 | 4 | helped | 0.0344 | mlp_term | 0.4163 | 0.2000 |
| chameleon | 189 | 4 | 1 | 1 | helped | 0.0486 | mlp_term | 0.4889 | 0.2431 |
| chameleon | 892 | 0 | 1 | 1 | helped | 0.0547 | mlp_term | 0.3492 | 0.2202 |
| chameleon | 782 | 1 | 4 | 4 | helped | 0.0764 | mlp_term | 0.4388 | 0.2040 |
| chameleon | 1398 | 4 | 2 | 0 | changed_wrong_to_wrong | 0.0074 | mlp_term | 0.2923 | 0.1993 |
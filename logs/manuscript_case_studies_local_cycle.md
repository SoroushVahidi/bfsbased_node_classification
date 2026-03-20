# Figure-ready corrected-node case studies (local manuscript cycle)

## citeseer (promising)
- diagnostics file: `/workspace/logs/citeseer_selective_graph_correction_split7_rep2.json`
- selected run: split=7, repeat=2
- changed/helped/hurt in selected run: 42/26/7
| node | mlp | final | true | outcome | margin | dominant term | graph support | feature support | interpretation |
|---:|---:|---:|---:|---|---:|---|---:|---:|---|
| 1876 | 3 | 1 | 1 | helped | 0.0042 | mlp_term | 1.2218 | 0.2089 | helped: graph evidence dominated (neighbor+compatibility); margin=0.004 |
| 2422 | 1 | 2 | 2 | helped | 0.0076 | mlp_term | 0.6634 | 0.2055 | helped: graph evidence dominated (neighbor+compatibility); margin=0.008 |
| 2848 | 5 | 3 | 3 | helped | 0.0178 | mlp_term | 1.0060 | 0.2118 | helped: graph evidence dominated (neighbor+compatibility); margin=0.018 |
| 369 | 1 | 2 | 2 | helped | 0.0479 | graph_neighbor_term | 1.2905 | 0.2008 | helped: graph evidence dominated (neighbor+compatibility); margin=0.048 |
| 2013 | 0 | 2 | 2 | helped | 0.0496 | graph_neighbor_term | 1.2876 | 0.2018 | helped: graph evidence dominated (neighbor+compatibility); margin=0.050 |
| 696 | 2 | 3 | 3 | helped | 0.0735 | mlp_term | 1.2525 | 0.2062 | helped: graph evidence dominated (neighbor+compatibility); margin=0.073 |
| 2802 | 2 | 3 | 3 | helped | 0.0974 | mlp_term | 0.9001 | 0.2072 | helped: graph evidence dominated (neighbor+compatibility); margin=0.097 |
| 785 | 1 | 3 | 3 | helped | 0.0991 | mlp_term | 1.2547 | 0.2038 | helped: graph evidence dominated (neighbor+compatibility); margin=0.099 |
| 956 | 0 | 1 | 1 | helped | 0.1102 | mlp_term | 0.7655 | 0.1978 | helped: graph evidence dominated (neighbor+compatibility); margin=0.110 |
| 989 | 1 | 4 | 4 | helped | 0.1110 | mlp_term | 1.2668 | 0.2040 | helped: graph evidence dominated (neighbor+compatibility); margin=0.111 |

## chameleon (mixed)
- diagnostics file: `/workspace/logs/chameleon_selective_graph_correction_split2_rep0.json`
- selected run: split=2, repeat=0
- changed/helped/hurt in selected run: 15/5/4
| node | mlp | final | true | outcome | margin | dominant term | graph support | feature support | interpretation |
|---:|---:|---:|---:|---|---:|---|---:|---:|---|
| 124 | 2 | 4 | 4 | helped | 0.0307 | mlp_term | 0.6925 | 0.1965 | helped: graph evidence dominated (neighbor+compatibility); margin=0.031 |
| 1131 | 2 | 4 | 4 | helped | 0.0344 | mlp_term | 0.4163 | 0.2000 | helped: graph evidence dominated (neighbor+compatibility); margin=0.034 |
| 189 | 4 | 1 | 1 | helped | 0.0486 | mlp_term | 0.4889 | 0.2431 | helped: graph evidence dominated (neighbor+compatibility); margin=0.049 |
| 892 | 0 | 1 | 1 | helped | 0.0547 | mlp_term | 0.3492 | 0.2202 | helped: graph evidence dominated (neighbor+compatibility); margin=0.055 |
| 782 | 1 | 4 | 4 | helped | 0.0764 | mlp_term | 0.4388 | 0.2040 | helped: graph evidence dominated (neighbor+compatibility); margin=0.076 |
| 1398 | 4 | 2 | 0 | changed_wrong_to_wrong | 0.0074 | mlp_term | 0.2923 | 0.1993 | changed_wrong_to_wrong: graph evidence dominated (neighbor+compatibility); margin=0.007 |

## wisconsin (negative_control)
- diagnostics file: `/workspace/logs/wisconsin_selective_graph_correction_split0_rep0.json`
- selected run: split=0, repeat=0
- changed/helped/hurt in selected run: 0/0/0
| node | mlp | final | true | outcome | margin | dominant term | graph support | feature support | interpretation |
|---:|---:|---:|---:|---|---:|---|---:|---:|---|

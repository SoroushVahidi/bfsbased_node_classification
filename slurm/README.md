# Slurm job templates

Batch scripts for clusters that use Slurm (e.g. Wulver). Logs default to `slurm/*.out` and `slurm/*.err` (gitignored); the directory is kept in git via `.gitkeep`. Use the **launch scripts** under `scripts/baseline_comparison_wulver/` when available — they `mkdir -p slurm` before `sbatch`.

| File | Job name (typical) | How to launch |
|------|-------------------|---------------|
| [`baseline_comparison_array.sbatch`](baseline_comparison_array.sbatch) | `bl_cmp` | [`scripts/baseline_comparison_wulver/launch_wulver.sh`](../scripts/baseline_comparison_wulver/launch_wulver.sh) |
| [`baseline_comparison_remaining.sbatch`](baseline_comparison_remaining.sbatch) | `bl_cmp_rem` | [`launch_remaining_datasets.sh`](../scripts/baseline_comparison_wulver/launch_remaining_datasets.sh) |
| [`baseline_comparison_feature_posthoc_array.sbatch`](baseline_comparison_feature_posthoc_array.sbatch) | `bl_featph` | [`launch_feature_posthoc_wulver.sh`](../scripts/baseline_comparison_wulver/launch_feature_posthoc_wulver.sh) |
| [`label_budget_sweep_array.sbatch`](label_budget_sweep_array.sbatch) | `lb_budget` | [`launch_label_budget_wulver.sh`](../scripts/baseline_comparison_wulver/launch_label_budget_wulver.sh) |
| [`graph_noise_robustness_array.sbatch`](graph_noise_robustness_array.sbatch) | `gn_robust` | [`launch_graph_noise_wulver.sh`](../scripts/baseline_comparison_wulver/launch_graph_noise_wulver.sh) |
| [`djgnn_benchmark_array.sbatch`](djgnn_benchmark_array.sbatch) | (see file) | See [`docs/DJGNN_INTEGRATION.md`](../docs/DJGNN_INTEGRATION.md) |
| [`djgnn_benchmark_full.sbatch`](djgnn_benchmark_full.sbatch) | — | Full DJGNN benchmark |
| [`djgnn_benchmark_smoke.sbatch`](djgnn_benchmark_smoke.sbatch) | — | Short DJGNN smoke |
| `run_selective_validation_analysis.sbatch`, `run_selective_validation_large_controls.sbatch`, `run_selective_validation_main_r3.sbatch` | — | Legacy validation (read `#SBATCH` headers before use) |

Details, tags, and aggregation commands: [`scripts/baseline_comparison_wulver/README.md`](../scripts/baseline_comparison_wulver/README.md).

Partition, QoS, and account are set inside each `.sbatch` file — adjust for your site.

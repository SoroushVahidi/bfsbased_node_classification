# Archived experimental runners (not part of the FINAL_V3 package)

These scripts supported **exploration, diagnostics, and earlier reliability-gated variants**
(v1/v2 development cycles). They are kept for historical traceability only.

**For the PRL evidence package, use:**

| Purpose | Script |
|--------|--------|
| FINAL_V3 + baselines / ablations (same splits as `reports/final_method_v3_results.csv`) | `../run_final_evaluation.py` |
| Broader PRL resubmission grid (GCN, APPNP, structural ablations) | `../prl_resubmission_runner.py` |
| Core manuscript driver (MLP vs selective graph correction) | `../manuscript_runner.py` |

| Archived file | Role |
|---------------|------|
| `diagnostic_runner.py` | Diagnostic sweeps (why baselines differ) |
| `reliability_gated_runner.py` | Early reliability-gated experiments |
| `v2_next_phase_runner.py` | V2 follow-up grid |
| `run_improvement_experiments.py` | Method improvement screening |
| `run_prl_validation.py` | Earlier PRL validation harness |

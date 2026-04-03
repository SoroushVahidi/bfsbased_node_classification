# Contributing

Thank you for your interest in this repository.

**New here or using an AI coding agent?** Read [`AGENTS.md`](AGENTS.md) and [`docs/INDEX.md`](docs/INDEX.md) before making changes.

This is primarily a **frozen evidence package** for a research paper submission, not an actively developed library. Contributions are
welcome, but please follow these guidelines to keep the repository organized
and reviewer-safe.

---

## Ground rules

1. **Do not modify canonical frozen files.**  
   The following files are frozen evidence for the research paper submission.  
   Do not edit their contents:
   - `reports/final_method_v3_results.csv`
   - `tables/main_results_selective_correction.{md,csv}`
   - All files in `reports/archive/`

2. **Never overwrite canonical outputs when re-running experiments.**  
   Always pass a unique `--output-tag` argument to any runner script:
   ```bash
   python3 code/bfsbased_node_classification/run_final_evaluation.py \
     --split-dir data/splits \
     --datasets cora --splits 0 \
     --output-tag my_experiment_$(date +%Y%m%d)
   ```

3. **Keep the three method lines separate.**  
   Do not mix FINAL_V3 result files with UG-SGC or UG-SGC-S artifacts.
   See [`REPO_STATUS.md`](REPO_STATUS.md) for the boundary between lines.

4. **New scripts go in `code/bfsbased_node_classification/`.**  
   Experimental or diagnostic scripts that are not yet stable should go in
   `code/bfsbased_node_classification/experimental_archived/` until they are
   ready to be promoted.

5. **New result files go in `logs/` (raw JSONL) and `reports/` (CSV summaries).**  
   Use a descriptive `--output-tag` so the file name is self-documenting.  
   Large cluster sweeps may write under `outputs/<suite>/` (gitignored); keep those separate from canonical frozen artifacts.

6. **Documentation changes are welcome.**  
   If you improve `README.md`, `ANALYSIS_GUIDE.md`, or `REPO_STATUS.md`, keep
   the tone conservative and reviewer-safe. Do not strengthen scientific claims
   beyond what the frozen evidence supports.

---

## Sanity check

Before opening a pull request, run the repository sanity check:

```bash
python3 scripts/check_canonical.py
```

This verifies that all canonical files exist and are non-empty, and that the
main entry-point scripts are syntactically valid.

---

## Style

- Python: follow PEP 8; max line length 120 characters (`flake8 --max-line-length 120`).
- Markdown: keep lines reasonably short; use ATX-style headings.
- Commit messages: use the imperative mood (`Add X`, `Fix Y`, `Update Z`).

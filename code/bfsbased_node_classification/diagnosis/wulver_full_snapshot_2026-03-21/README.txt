Wulver full snapshot — what is on GitHub (2026-03-21)
======================================================

A) Git branch (code + in-repo diagnosis trees)
   Branch: archive/wulver-manuscript-2026-03-21
   Remote: https://github.com/SoroushVahidi/bfsbased_node_classification
   Contains: Python sources, manuscript_runner fixes, all diagnosis/* bundles under the repo, etc.

B) Branch preserve/wulver-full-filesystem-2026-03-21 — tarballs for the *parent* tree
   ~/node_classification_manuscript/ is not inside the git repo; these archives recreate it:

   1) data.tar.gz
      Full data/ subtree (compressed ~13 MB).

   2) manuscript_core_without_data_and_large_diag_json.tar.gz
      Everything under node_classification_manuscript/ except data/ and except
      logs/*selective_graph_correction*.json (large per-run diagnostics).

   3) logs_selective_graph_correction_diagnostics.tar.gz
      Only logs/*selective_graph_correction*.json files.

   Reconstruct locally (example):
     mkdir -p node_classification_manuscript && cd node_classification_manuscript
     tar -xzf manuscript_core_without_data_and_large_diag_json.tar.gz
     tar -xzf data.tar.gz
     tar -xzf logs_selective_graph_correction_diagnostics.tar.gz -C .

C) Earlier preserve/* branches
   Smaller API uploads from prior steps (JOB1/JOB2/merged analysis) remain on their own branches.

D) Full no-data tree (includes large SGC JSON in logs/) — split upload
   On branch preserve/wulver-full-filesystem-2026-03-21:
     diagnosis/wulver_full_snapshot_2026-03-21/full_no_data/_push_all_no_data.tar.gz.part.00
     diagnosis/wulver_full_snapshot_2026-03-21/full_no_data/_push_all_no_data.tar.gz.part.01
   Rejoin:
     cat _push_all_no_data.tar.gz.part.00 _push_all_no_data.tar.gz.part.01 > _push_all_no_data.tar.gz
     # then extract as usual; should match the monolithic no-data snapshot

E) Not included
   - Secrets, tokens, and .git credentials.

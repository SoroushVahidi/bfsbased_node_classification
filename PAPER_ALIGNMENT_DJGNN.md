# Diffusion-Jump GNN Paper → Repository Alignment Notes

Source context provided by user: **Neural Networks 181 (2025) 106830**, “Node classification in the heterophilic regime via diffusion-jump GNNs” (Begga, Escolano, Lozano).

## 1) What the paper proposes (high-level)
- A structural heterophily measure
  - \(k = E_{label} / E_{ground}\), where:
    - \(E_{label} = y^T \Delta y\)
    - \(E_{ground}\) is the first non-zero Laplacian eigenvalue / Fiedler energy.
- A **Diffusion-Jump GNN** architecture with:
  - A “diffusion pump” that learns empirical eigenfunctions.
  - Learnable diffusion-distance matrix and jump hierarchy.
  - Structural filters derived from diffusion distances.
  - Parallel jump branches + homophilic branch.

## 2) What this repository currently contains
- The repo currently centers on notebook-export scripts and hybrid pipelines:
  - `bfsbased-gnn-predictclass.py`
  - `bfsbased-full-investigate-homophil.py`
  - `bfsbased_heterophilic_compare_with_djgnn (2).py`
- The scripts include:
  - dataset loading for WebKB/WikipediaNetwork/Planetoid/Actor,
  - an MLP + refinement (`predictclass`) pipeline,
  - random-search-style tuning,
  - and a very large comparison script with multiple model definitions.

## 3) Alignment gaps between paper and code (practical)
1. **No explicit structural heterophily implementation**
   - The paper’s \(k = E_{label}/E_{ground}\) formula is not exposed as a stable first-class metric module in this repo.
2. **No isolated diffusion-pump module**
   - There is no clear, reusable implementation of a learnable diffusion-distance matrix `Q` tied to a trace-ratio/Dirichlet objective as described.
3. **No clean jump-filter bank API**
   - The paper describes learnable jump-based structural filters; repository code is mostly monolithic and notebook-style, making this difficult to verify/reuse.
4. **Reproducibility mismatch risk**
   - The paper references fixed benchmark split protocol; scripts fall back to ad-hoc random split generation when expected files are absent.
5. **Large duplication burden in comparison file**
   - The large `...compare_with_djgnn (2).py` script contains repeated symbol definitions, which complicates mapping code behavior to claimed architecture.

## 4) Minimal roadmap to bring code closer to paper
1. Create `metrics/structural_heterophily.py`
   - `ground_energy(L)`
   - `label_energy(y, L)`
   - `structural_heterophily_k(y, L)`
2. Create `models/djgnn.py`
   - diffusion pump, jump projector, structural filter bank, branch combiner.
3. Add `configs/*.yaml` for dataset-specific hyperparameters (e.g., jump count `K`, hidden size, dropout).
4. Add deterministic split handling and explicit erroring when benchmark splits are missing (optional flag for random fallback).
5. Add smoke tests:
   - shape/finite checks,
   - one tiny training step,
   - metric consistency test for `k`.

## 5) Why this note was added
This file records the paper context provided by the user and converts it into concrete implementation-oriented gaps and next steps for this repository.

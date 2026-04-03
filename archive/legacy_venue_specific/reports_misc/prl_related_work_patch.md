# PRL related-work patch

This patch is repository-only. No `.tex` manuscript or `.bib` file is tracked in
this repository, so this note provides writing guidance rather than a direct
manuscript edit.

## 1. What is missing in the repo

- No manuscript source is available to inspect the current related-work section.
- No `.bib` file is available to verify whether near-neighbor citations are already present.
- Repo text searches did not find explicit manuscript-facing mentions of:
  - `Correct and Smooth`
  - strong-MLP / "when graph learning is necessary" style papers
  - a graph-uncertainty paper aligned with the paper’s framing

## 2. What likely needs to be covered

For the current narrow PRL framing, the related-work section should acknowledge
at least three nearby threads:

1. **Post-processing / correction methods**
- Especially `Correct and Smooth`, because it is a natural comparison point for
  graph-based prediction correction rather than end-to-end message passing.

2. **Strong non-graph or graph-light baselines**
- At minimum, cite work arguing that strong MLP-style baselines and careful
  benchmarking can explain or narrow many reported gains from graph methods,
  especially on heterophilic or structurally noisy benchmarks.

3. **Uncertainty or confidence-aware graph learning**
- If the manuscript continues to emphasize uncertainty in the title or framing,
  cite at least one paper that treats uncertainty/confidence as a first-class
  object in graph learning, then explain that `FINAL_V3` uses uncertainty in a
  narrower, selective-correction sense rather than a broad Bayesian treatment.

## 3. Sections that should be updated

Because the manuscript source is not in the repo, the exact section names may
vary. The following locations should be updated in the writing repo:

- Introduction / motivation paragraph:
  - explain that the goal is not to replace the MLP globally, but to correct it selectively
- Related work section:
  - separate message-passing methods, post-processing correction methods, and strong non-graph baselines
- Discussion or limitations:
  - state that the method is not a universal heterophily solution and should be read as a conservative correction layer

## 4. Suggested related-work replacement paragraph

Use or adapt the following paragraph:

> Our work is most closely related to graph-based post-processing and correction methods rather than end-to-end message-passing architectures. In particular, methods such as Correct and Smooth show that graph structure can substantially improve predictions when used as a refinement stage on top of strong initial classifiers. At the same time, recent studies on heterophilic and weakly homophilic benchmarks have emphasized that strong MLP-style baselines and careful evaluation protocols can narrow or even overturn apparent gains from graph propagation. We therefore position `FINAL_V3` as a conservative, feature-first correction layer: it starts from a strong MLP, applies graph evidence only selectively, and emphasizes reduced harmful graph influence rather than universal graph-learning superiority.

## 5. Suggested introduction / discussion positioning patch

Use or adapt the following paragraph:

> The contribution of this paper is not a broad claim that graph propagation should replace feature-only classification. Instead, the repository evidence supports a narrower view: a strong MLP remains the primary predictor, and graph structure is most useful as a selective correction mechanism when model confidence is low and local graph evidence appears reliable. This framing is especially important on heterophilic or noisy graphs, where indiscriminate propagation can be harmful. Our method is therefore best understood as a reliability-aware correction layer that aims to preserve the robustness of the MLP while exploiting graph information only where it appears safe to do so.

## 6. BibTeX status

- No `.bib` file exists in this repository.
- Therefore, no bibliography entries were appended automatically in this pass.
- When the manuscript repo is available, add the missing entries there rather
  than creating a new bibliography file in this code-and-artifact repository.

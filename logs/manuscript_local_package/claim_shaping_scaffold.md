# Manuscript claim shaping draft

## Claims we can defend
- Selective graph correction is stably positive on cora and citeseer across canonical splits and repeats.
- Actor and Cornell remain neutral vs MLP under repeats=3; Wisconsin remains negative.
- High-threshold/high-uncertainty regime aligns with positive gains; low-threshold regimes often collapse toward MLP-like behavior.
- The method is feature-first and often changes only a small subset of nodes; this enables a fallback-style safety story.

## Claims we should avoid
- Universal improvement over strong MLP across all datasets.
- Superiority to propagation-first methods in general.
- Broad heterophily-focused claim based only on current local evidence.
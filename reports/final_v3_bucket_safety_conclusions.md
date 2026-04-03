# FINAL_V3 bucketed safety conclusions (canonical package, current run)

## Scope and protocol used in this run

- Methods: `mlp_only`, `FINAL_V3`, `gcn`, `appnp`
- Datasets: `cora`, `citeseer`, `pubmed`, `chameleon`, `texas`, `wisconsin`
- Splits executed in this run: `0,1,2`
- Bucket scheme: split-specific terciles of **MLP test margin** (`low`, `medium`, `high`)
- Baseline budget used for practicality in this run:
  - `max_epochs=20`, `patience=8`

## Core findings (this run)

1. **Feature-first conservatism is strongly supported.**
   - High-confidence bucket: FINAL_V3 changed fraction is **0.0** mean, with identical MLP/FINAL_V3 accuracy.
   - Medium bucket: changed fraction is also effectively **0.0** in this run.

2. **Gains are concentrated in low-confidence nodes.**
   - Low bucket mean MLP acc: **0.5403**
   - Low bucket mean FINAL_V3 acc: **0.5777**
   - Low bucket mean delta: **+0.0374**

3. **Safety profile remains favorable vs graph-wide baselines.**
   - Overall mean test accuracy:
     - MLP: **0.7265**
     - FINAL_V3: **0.7390**
     - GCN: **0.6963**
     - APPNP: **0.6775**
   - On harder regimes (`chameleon`, `texas`, `wisconsin`) mean test accuracy:
     - MLP: **0.7004**
     - FINAL_V3: **0.7039**
     - GCN: **0.5713**
     - APPNP: **0.5381**

## Low-confidence structural extension (targeted comparison)

Targeted run (`chameleon`, `texas`, `citeseer`; splits `0,1,2`) with optional low-confidence structural term enabled (`b6` validation-selected from `[0.0, 0.2, 0.4]`):

- Mean overall accuracy delta (structural - base FINAL_V3): **~0.0000** (nearly neutral)
- Mean low-bucket delta vs MLP difference: **+0.0004** (very small positive)
- Harmful changes increased by **+4** total across targeted runs

### Interpretation

- The structural low-confidence term appears **mixed**:
  - slight low-bucket benefit on average,
  - but a small safety cost (more harmful overwrites),
  - and near-zero net overall-accuracy change.
- Current evidence does **not** support promoting the structural term into default FINAL_V3.
- Best next step is a stricter safety-gated version (e.g., higher reliability/quality requirement for non-zero `b6`) before broader adoption.

## Caveat

- This session completed a full 6-dataset method suite but for **3 splits (0–2)**, not all 10 splits.
- The code and output paths are now canonicalized for full 10-split reruns using the same pipeline.

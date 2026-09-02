# SPEC-DS-12: Model Registry — tracking experiments with MLflow

**Status:** approved
**Subject:** Data Science
**Section:** Worked Examples
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8
**Prerequisites:** SPEC-DS-5/DS-6 (models to track)

## Intent
A Java dev has artifact repositories and CI dashboards; the ML analogue is an experiment tracker +
model registry. Teach MLflow: logging params/metrics/artifacts, comparing runs, and registering/
promoting model versions — all locally.

## Learning objectives
- LO1 — Explain why ad-hoc notebooks lose track of what produced which score, and what a tracker fixes.
- LO2 — Use MLflow Tracking to log params, metrics, and artifacts across several runs and compare them in the UI.
- LO3 — Log a model, register it in the Model Registry, and move a version through stages (Staging→Production) or aliases.
- LO4 — Reload a registered model for inference and reason about reproducibility.

## Scope
In: MLflow Tracking (local file/sqlite backend), the UI, autolog, Model Registry, load-for-inference.
Out: remote/hosted MLflow, full CI/CD (cross-link SDLC subject), cloud registries (→ DS cloud chapter).

## Outline
1. What & why — experiments as first-class artifacts; the "which run made this model?" problem.
2. Track runs — start_run, log_param/metric/artifact; autolog for sklearn.
3. Compare runs in the MLflow UI (screenshots/exported artefacts).
4. Register a model; stage/alias it; reload it for prediction.
5. Pitfalls — untracked randomness, environment drift, logging huge artifacts.

## Claims to ground (Haiku, before writing)
- [ ] Verify the current MLflow version on PyPI and that it installs + runs a LOCAL tracking server / file store in a sandbox.
- [ ] Verify the current MLflow API: mlflow.start_run, log_param/log_metric/log_artifact, mlflow.sklearn.autolog, mlflow.register_model / MlflowClient, and the CURRENT stage-vs-alias story (stages are being deprecated in favour of aliases — confirm the current recommendation).

## Assets to produce
- Prose: "Data Science/Worked Examples/model-registry-mlflow.md"
- Code: "Data Science/Worked Examples/code/mlflow_tracking.py"
- Artefacts: an exported runs-comparison table/plot; the run/registry structure (a captured listing).

## Acceptance criteria
- [ ] AC1 — LOs delivered. AC2 — mlflow_tracking.py runs against a local store, creating real runs + a registered model; snippet-check passes. UI steps that need a browser are fenced as ```text with described output (no fabricated screenshots — describe or export a real artifact). AC3 — MLflow version + API (esp. stages vs aliases) grounded. AC4 — the artifact-repo/CI-dashboard analogy used.

## Gates
Entry: approved; notes landed. Exit: DoD checklist.

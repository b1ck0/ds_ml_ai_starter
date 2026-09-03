# Data Science

Part 1 of the crash course. See [`../docs/curriculum.md`](../docs/curriculum.md) for the full
backlog and [`../docs/architecture.md`](../docs/architecture.md) for how chapters get written.

Organised under five sections. Each chapter is written only after its `specs/SPEC-DS-*.md` is
approved and its grounding notes have landed.

## Theory
Hypothesis testing · imputation · feature engineering · regression · classification (binary /
multi-class / multi-label) · class imbalance · undersampling · ensembles · forecasting · AutoML ·
feature selection · overfitting · regularization.

## Local Environment Setup
PyCharm · Python · pip/venv · pandas · NumPy · Matplotlib · scikit-learn · Jupyter.

## Worked Examples
Hypothesis testing & EDA · imputation · collinearity · train/valid/holdout split · regression
(NYC taxi fare) · classification (Titanic; + multi-class & multi-label) · forecasting (composite
synthetic signals) · feature selection · AutoML · model registry (MLflow) · feature store (Feast) ·
Bayesian inference (Gaussian-noise regression + AR(1) with PyMC).

## Cloud Environment Setup
Vertex AI · Azure ML · SageMaker · batch vs online inference · production monitoring (concept/data/
model drift, retraining triggers) · in-database ML (BigQuery ML, Redshift ML).

## Production Considerations
MLOps pipelines, model promotion decisions, monitoring, and retraining — see the Cloud section and
`docs/definition-of-done.md`.

_All 20 Data Science chapters (DS-0 → DS-19) are complete — see the sections above._

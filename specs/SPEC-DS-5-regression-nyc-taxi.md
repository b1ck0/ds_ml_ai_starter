# SPEC-DS-5: Regression — predicting NYC taxi fares

**Status:** done (written by Sonnet, grounded by Haiku, independently reviewed + merged 2026-09-03)
**Subject:** Data Science
**Section:** Worked Examples
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8
**Prerequisites:** SPEC-DS-1, SPEC-DS-3, SPEC-DS-4

## Intent
The flagship regression chapter. Predict taxi fare from trip features; use it to teach regression
metrics, three model families, the bagging-vs-boosting distinction, coordinate feature engineering,
feature scaling, categorical encoding, and how to judge whether a model is "fair".

## Learning objectives
- LO1 — Compute and interpret R², RMSE, MSE, MAE and know when each matters.
- LO2 — Train and compare Linear Regression, Random Forest, Gradient Boosting; explain bagging vs boosting and WHY boosting usually wins (sequential residual-fitting vs variance-averaging).
- LO3 — Engineer features from raw pickup/dropoff coordinates: haversine distance and 2D map-sector bucketing.
- LO4 — Diagnose model fairness: residuals should be ~normal and homoscedastic; y-vs-ŷ should hug the 45° line; read linear coefficients as importances.
- LO5 — Explain feature scaling (MinMax vs StandardScaler) — what it affects and which models need it (linear/distance yes; trees no) — and categorical encoding (one-hot vs ordinal/categorical) — which models need which.

## Scope
In: the five metrics; LinearRegression, RandomForestRegressor, gradient boosting (sklearn HistGradientBoosting and/or XGBoost/LightGBM); coordinate FE; residual & parity plots; scaling; encoding; feature importance.
Out: hyperparameter search depth (one light `RandomizedSearchCV` mention), SHAP (mention + link), temporal split (n/a — trips are independent rows).

## Outline
1. What & why — a regression problem a backend dev can picture (a pricing function to learn).
2. The data — a MANAGEABLE NYC taxi fare sample (see grounding). EDA + clean obviously-bad rows (neg fares, null coords).
3. Metrics — MSE/RMSE/MAE/R² defined, with the Java-friendly "loss function" framing.
4. Baseline Linear Regression; then Random Forest; then Gradient Boosting. Compare on the holdout.
5. Bagging vs boosting — the mechanism, why boosting typically edges ahead, and its overfitting risk.
6. Coordinate feature engineering — haversine distance; bucket lat/long into grid sectors; show the lift.
7. Scaling & encoding — demonstrate that trees are scale-invariant while linear/KNN are not; one-hot vs ordinal.
8. Fairness diagnostics — residual histogram (normality), residual-vs-fitted (homoscedasticity), y-vs-ŷ parity plot, coefficient importances.
9. Pitfalls — leakage via location target encoding, extrapolation, RMSE dominated by outliers.

## Assets to produce
- Prose: "Data Science/Worked Examples/regression-nyc-taxi.md"
- Code: "Data Science/Worked Examples/code/regression_taxi.py" (+ a small data-prep helper if needed)
- Datasets: a documented download+sample step; or a committed small sample under datasets/ if licence permits.
- Artefacts: metrics comparison table; residual histogram; residual-vs-fitted plot; y-vs-ŷ parity plot; map-sector scatter; feature-importance bar chart.

## Claims to ground (Haiku, before writing) — IMPORTANT
- [ ] Find a MANAGEABLE, freely-licensed NYC taxi fare dataset small enough to run in a sandbox (e.g., a sampled CSV, a single NYC-TLC monthly parquet sub-sampled, or the Kaggle "New York City Taxi Fare Prediction" train sample). Give a reachable URL + licence + column schema. FALLBACK: if nothing suitably small/licensed exists, the writer SYNTHESISES a realistic dataset (random NYC-bounded pickup/dropoff coords + distance-based fare + noise) so the chapter is fully runnable — the grounding NOTE should state which path was taken.
- [ ] Verify gradient-boosting library choice + version: sklearn `HistGradientBoostingRegressor` (in-stdlib, no extra dep) is preferred for runnability; if XGBoost/LightGBM is used, pin the current version and confirm it installs on the target Python.
- [ ] Verify the haversine formula against an authoritative source.
- [ ] Verify sklearn APIs: metrics (`mean_squared_error` incl. the `squared`/`root_mean_squared_error` situation on current sklearn), `RandomForestRegressor`, `HistGradientBoostingRegressor`, `OneHotEncoder`, `StandardScaler`, `MinMaxScaler`, `ColumnTransformer`.

## Acceptance criteria
- [ ] AC1 — LO1–LO5 each delivered → section map.
- [ ] AC2 — regression_taxi.py runs end-to-end on the chosen/synthesised data, trains all three models, and produces every artefact → run log + snippet-check pass.
- [ ] AC3 — dataset source/licence (or synthesis choice), GBM library+version, haversine, and all sklearn APIs grounded in NOTEs.
- [ ] AC4 — bagging-vs-boosting mechanism explained (not just asserted); scaling/encoding claims demonstrated empirically (trees invariant, linear not); Java framing for "learning a pricing function".

## Gates
Entry: approved; grounding notes (esp. the dataset decision) landed. Exit: DoD — code runs and reproduces artefacts; links resolve; fresh-Sonnet review; architect merge.

# SPEC-DS-9: Forecasting — trend, seasonality, and why the split is different

**Status:** done (written by Sonnet, grounded by Haiku, independently reviewed + merged 2026-09-03)
**Subject:** Data Science
**Section:** Worked Examples
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8
**Prerequisites:** SPEC-DS-4 (splitting), SPEC-DS-5 (regression)

## Intent
Forecasting looks like regression but the data is ordered in time, so the split rules change: you must
never leak the future into the past. Teach the time-series split, autocorrelation, trend, seasonality,
and autoregressive models — using composite SYNTHETIC signals so the structure is known exactly.

## Learning objectives
- LO1 — Explain why an i.i.d. random split leaks the future, and use non-overlapping / expanding-window time splits instead (`TimeSeriesSplit`).
- LO2 — Decompose a signal into trend + seasonality + residual; read an autocorrelation (ACF/PACF) plot.
- LO3 — Fit autoregressive models (AR / ARIMA) and a regression-with-lag-features baseline; forecast and back-test.
- LO4 — Choose the right approach per signal shape and know how to scale/detrend before modelling.

## Scope
In: the four composite synthetic signals below; time-based splitting; ACF/PACF; AR/ARIMA + lag-feature regression; walk-forward back-testing; detrending/normalisation.
Out: Prophet, multivariate/exogenous forecasting (mention). (Deep forecasting was later brought IN
scope via a 2026-09-03 enrichment: §6 now trains a small runnable LSTM on one signal and compares it
honestly against the ARIMA/lag-regression baselines — on this clean synthetic signal the neural model
does not beat the simple lag regression, which is the teaching point.)

## The signals (build them exactly)
For amplitude A and length N, with noise/wiggle = 10% of A:
1. linear trend + sine wave (10% of A)  → trend dominates; deseasonalise then extrapolate trend; seasonal AR.
2. linear trend + random noise (10% of A) → detrend, model residual as near-white; linear/AR.
3. sine wave + random noise (10% of A)   → no trend; seasonal/AR captures the cycle.
4. quadratic trend + random noise (10% of A) → non-linear trend; polynomial detrend or differencing then AR.
For EACH: say which model fits best and why, and how to normalise/scale (e.g. difference to stationarity, standardise residuals).

## Outline
1. What & why — regression's cousin, but time ordering forbids shuffling; the leakage picture.
2. Build the four signals; plot each with its components.
3. Stationarity, ACF/PACF, differencing — the diagnostic toolkit.
4. Model each signal (AR/ARIMA + a lag-feature regression baseline); walk-forward back-test; compare.
5. Per-signal recommendation table (model + scaling choice + why).
6. Pitfalls — random split on time data, fitting the scaler on the whole series, forecasting far beyond the trend's validity.

## Assets to produce
- Prose: "Data Science/Worked Examples/forecasting-composite-signals.md"
- Code: "Data Science/Worked Examples/code/forecasting_signals.py"
- Artefacts: the four signal plots (with decomposition); ACF/PACF plots; forecast-vs-actual back-test plots; the recommendation table.

## Claims to ground (Haiku, before writing)
- [ ] Verify statsmodels time-series APIs on the current version: AutoReg / ARIMA (statsmodels.tsa), acf/pacf & plot_acf/plot_pacf, seasonal_decompose, adfuller (stationarity). Reuse NOTE-6 for the statsmodels version.
- [ ] Verify sklearn `TimeSeriesSplit` API. Confirm the correct definition of a walk-forward / expanding-window back-test against an authoritative source.
- [ ] Confirm the mathematical definitions used (autocorrelation, stationarity, AR(p)) against an authoritative source.

## Acceptance criteria
- [ ] AC1 — LO1–LO4 delivered, incl. the per-signal recommendation table. AC2 — code builds all four signals, fits models, back-tests, and produces every artefact; snippet-check passes. AC3 — statsmodels/sklearn TS APIs + definitions grounded. AC4 — the "no shuffling in time" rule made vivid vs the earlier i.i.d. split chapter.

## Gates
Entry: approved; notes landed. Exit: DoD checklist.

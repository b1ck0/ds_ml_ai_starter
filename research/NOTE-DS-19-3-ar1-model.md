# NOTE-DS-19-3: AR(1) autoregressive model definition and stationarity condition

**Answer:**
The AR(1) autoregressive model is **x_t = c + φ x_{t-1} + ε_t**, where c is a constant (intercept), φ (phi) is the autoregressive coefficient, and ε_t ~ N(0, σ²) is white noise. **Stationarity condition:** |φ| < 1 (absolute value of phi strictly less than 1). If |φ| ≥ 1, the process is non-stationary and has infinite or unit-root variance.

**Evidence:**

1. **Definition (x_t = c + φ x_{t-1} + ε_t):**
   - Gregory Gundersen's tutorial: https://gregorygundersen.com/blog/2022/01/06/autoregressive-model/ — "An AR(1) process is defined as X_t = φ X_{t-1} + ε_t, where ε_t ~ WN(0,σ²) and φ is a real-valued constant."
   - Adam Djellouli's time series notes: https://adamdjellouli.com/articles/statistics_notes/time_series_analysis/autoregressive_models — general AR model "specifies that the output variable depends linearly on its own lagged values and on a stochastic term."
   - UR Machine Learning Blog: https://usmanr149.github.io/urmlblog/time%20series/2021/04/30/2021-4-30-AR-model.html — AR(1) definition and properties.
   - Applied Time Series Analysis (ATSA) for Fisheries: https://atsa-es.github.io/atsa-labs/sec-tslab-autoregressive-ar-models.html — standard AR(p) and AR(1) formulations.

2. **Stationarity condition (|φ| < 1):**
   - Gregory Gundersen and other sources above: "The condition of stationarity of the process is |φ₁| < 1."
   - UR Machine Learning Blog: "For an AR(1) model, stationarity is ensured if |φ₁| < 1, meaning the absolute value of the autoregressive coefficient is less than one."
   - GeeksforGeeks (AR Model for Time Series): https://www.geeksforgeeks.org/data-analysis/autoregressive-ar-model-for-time-series-forecasting/ — stationarity discussed in context of AR processes.
   - DigitalOcean tutorial: https://www.digitalocean.com/community/tutorials/auto-regressive-models-time-series-forecasting — explains AR model and stationarity.

3. **Boundary behavior:**
   - If |φ| = 1, the process is a random walk (unit root), non-stationary.
   - If |φ| > 1, the process explodes (variance grows unbounded).
   - If |φ| → 0, the model approaches white noise (strong mean-reversion).

4. **Variance under stationarity:**
   - When |φ| < 1: Var(X_t) = σ² / (1 − φ²). (Derived from stationarity assumption.)

**Caveats / limits:**
- The definition includes an intercept c (some texts omit it, writing x_t = φ x_{t-1} + ε_t, which forces E[X_t] = 0).
- Stationarity is required for standard inference; non-stationary AR(1) processes require differencing or other preprocessing.
- The constraint |φ| < 1 is strict; edge cases (φ = ±0.99999) may exhibit transient behavior before settling.
- Empirical time series may violate stationarity; formal unit-root tests (Augmented Dickey–Fuller) exist but are beyond this scope.

**Recommendation:**
Use the formula **x_t = c + φ x_{t-1} + ε_t** with intercept c for pedagogical clarity (matches owner's notebooks). Emphasize that in Bayesian estimation, the prior on φ should reflect prior belief that |φ| < 1 (e.g., Uniform on [−1, 1) or Beta-transformed). When fitting the worked example, check that posterior median of φ satisfies |φ| < 1 and explain what it would mean if it didn't (model fit is suspect, data suggests non-stationarity). Reference standard time series texts (e.g., Box–Jenkins, Brockwell–Davis) if needed; for this pedagogical level, cite the sources above.

**Date checked:** 2026-09-03

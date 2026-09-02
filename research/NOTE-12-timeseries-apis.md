# NOTE-12: statsmodels time-series APIs and sklearn TimeSeriesSplit (2026-09-02)

**Answer:** statsmodels 0.15.0 provides ARIMA, AutoReg, acf/pacf, plot_acf/plot_pacf, seasonal_decompose, and adfuller; sklearn 1.9.0 provides TimeSeriesSplit with expanding-window cross-validation. All APIs confirmed on official docs 2026-09-02.

**Evidence:**

| Component | API Signature | Version | Source | Date |
|-----------|---------------|---------|--------|------|
| **ARIMA** | `ARIMA(endog, exog=None, order=(0,0,0), seasonal_order=(0,0,0,0), trend=None, enforce_stationarity=True, enforce_invertibility=True, ...)` | statsmodels 0.15.0 | https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html | 2026-09-02 |
| **AutoReg** | `AutoReg(endog, lags, trend='c', seasonal=False, exog=None, hold_back=None, period=None, missing='none', *, deterministic=None)` | statsmodels 0.15.0 | https://www.statsmodels.org/stable/generated/statsmodels.tsa.ar_model.AutoReg.html | 2026-09-02 |
| **acf** | `acf(x, adjusted=False, nlags=None, qstat=False, fft=True, alpha=None, bartlett_confint=True, missing='none', *, result_object=None)` → returns array of autocorrelation values (shape: nlags+1) | statsmodels 0.15.0 | https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html | 2026-09-02 |
| **pacf** | `pacf(x, nlags=None, method='ywadjusted', alpha=None, *, result_object=None)` → returns array of partial autocorrelations (shape: nlags+1) | statsmodels 0.15.0 | https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.pacf.html | 2026-09-02 |
| **plot_acf** | `plot_acf(x, ax=None, lags=None, *, alpha=0.05, use_vlines=True, adjusted=False, fft=False, missing='none', title='Autocorrelation', zero=True, auto_ylims=False, bartlett_confint=True, ...)` | statsmodels 0.15.0 | https://www.statsmodels.org/stable/generated/statsmodels.graphics.tsaplots.plot_acf.html | 2026-09-02 |
| **plot_pacf** | `plot_pacf(x, ax=None, lags=None, alpha=0.05, method='ywm', use_vlines=True, title='Partial Autocorrelation', zero=True, ...)` | statsmodels 0.15.0 | https://www.statsmodels.org/stable/generated/statsmodels.graphics.tsaplots.plot_pacf.html | 2026-09-02 |
| **seasonal_decompose** | `seasonal_decompose(x, model='additive', filt=None, period=None, two_sided=True, extrapolate_trend=0)` → returns DecomposeResult object | statsmodels 0.15.0 | https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.seasonal_decompose.html | 2026-09-02 |
| **adfuller** | `adfuller(x, maxlag=None, regression='c', autolag='AIC', store=False, regresults=False, *, result_object=None)` → returns test statistic, p-value, lag count, critical values | statsmodels 0.15.0 | https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html | 2026-09-02 |
| **TimeSeriesSplit** | `TimeSeriesSplit(n_splits=5, max_train_size=None, test_size=None, gap=0)` with expanding-window cross-validation | sklearn 1.9.0 | https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html | 2026-09-02 |

**Definitions (from authoritative sources):**

1. **Autocorrelation:** Measures the linear relationship between lagged values of a time series. The autocorrelation function (ACF) quantifies how a series correlates with its own past values at different time lags (r_k measures correlation between y_t and y_{t-k}).  
   *Source:* https://otexts.com/fpp2/autocorrelation.html

2. **Stationarity:** A time series is stationary if its mean, variance, and autocovariance remain constant over time (weak stationarity, second-order stationarity). Autoregressive models require stationarity; testing is done via Augmented Dickey-Fuller (ADF) test using adfuller().  
   *Source:* https://blog.quantinsti.com/stationarity/

3. **AR(p) Autoregressive Model:** A regression of the variable against itself, defined as: y_t = c + φ₁y_{t-1} + φ₂y_{t-2} + ... + φ_p y_{t-p} + ε_t, where p is the order (number of lagged values) and ε_t is white noise.  
   *Source:* https://otexts.com/fpp2/AR.html

4. **Walk-Forward Backtesting (Expanding-Window):** A validation methodology that simulates live forecasting by using all previous data (samples 1, 2, ..., t-1) for training at each step t. The training set expands with each fold, preventing future data leakage into the past. This is distinct from rolling windows and mimics operational deployment.  
   *Source:* https://blog.quantinsti.com/walk-forward-optimization-introduction/, https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/

**Detrending and Differencing for Stationarity:**

- **Differencing:** Subtract consecutive values (y_t - y_{t-1}) to remove trends; useful for AutoReg/ARIMA. The order of differencing (d parameter in ARIMA) removes polynomial trends.
- **Detrending:** Use seasonal_decompose() to extract trend, then subtract from the original series for residual analysis. Alternatively, fit a polynomial trend and remove.
- **Seasonal adjustment:** seasonal_decompose(..., model='additive' or 'mul') isolates seasonal and trend components; use residual for modeling.

**Caveats / limits:**

- **ARIMA parameter order (p,d,q):** Requires domain knowledge or systematic search (grid/auto selection). Automated methods (autolag='AIC' in adfuller) help but are not foolproof.
- **Stationarity testing:** adfuller() tests the null hypothesis of a unit root (non-stationarity). A small p-value (e.g., <0.05) rejects the null and suggests stationarity, but visual inspection is also recommended.
- **seasonal_decompose() limitations:** Requires at least 2 complete seasonal cycles and uses centered moving averages, which can create NaN at endpoints (controlled by extrapolate_trend parameter).
- **plot_acf/plot_pacf:** Default confidence intervals use Bartlett's formula for raw data; pre-filtering (e.g., differencing) changes the interpretable bands.
- **TimeSeriesSplit with gap parameter:** The gap parameter (new feature) excludes samples between train and test to avoid lookahead bias, critical for financial/real-time data.

**Recommendation:**

- **For DS-9 chapter:** Pin statsmodels==0.15.0; reuse sklearn 1.9.0 from NOTE-5.
- **Import statements:**
  ```python
  from statsmodels.tsa.arima.model import ARIMA
  from statsmodels.tsa.ar_model import AutoReg
  from statsmodels.tsa.stattools import acf, pacf, adfuller
  from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
  from statsmodels.tsa.seasonal import seasonal_decompose
  from sklearn.model_selection import TimeSeriesSplit
  ```
- **Workflow for each synthetic signal:**
  1. Visualize and decompose using seasonal_decompose().
  2. Check stationarity with adfuller(); if non-stationary, apply differencing or detrending.
  3. Plot ACF/PACF (plot_acf/plot_pacf) to choose AR/MA orders.
  4. Fit ARIMA or AutoReg; use TimeSeriesSplit for expanding-window cross-validation.
  5. Compare forecast quality vs. detrended lag-feature regression baseline.
- **Scaling:** Before modeling, normalize/standardize residuals after detrending to ensure ACF/PACF are interpretable and time-series split respects temporal order.

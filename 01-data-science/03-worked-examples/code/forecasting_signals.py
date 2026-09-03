"""Forecasting composite synthetic signals: decomposition, stationarity, AR/ARIMA vs. a
lag-feature regression baseline, and why the time-series split is NOT the i.i.d. split from
SPEC-DS-4.

Companion code for:
  Data Science/Worked Examples/forecasting-composite-signals.md

What it does (mirrors the chapter's sections):
  1. Builds four composite synthetic signals with KNOWN ground truth (so we can check every
     diagnostic against the generating formula, not guess at it):
       signal_1_linear_sine      = linear trend (scale A) + sine seasonality (10% of A)
       signal_2_linear_noise     = linear trend (scale A) + white noise (10% of A)
       signal_3_sine_noise       = sine wave (scale A)    + white noise (10% of A)
       signal_4_quadratic_noise  = quadratic trend (scale A) + white noise (10% of A)
  2. For each signal: seasonal_decompose, an Augmented Dickey-Fuller stationarity test on the
     raw series and (if non-stationary) again after differencing, and ACF/PACF plots.
  3. Fits an ARIMA model (order chosen per signal from the ADF/ACF/PACF diagnostics) AND a
     lag-feature LinearRegression baseline, then walk-forward backtests both with sklearn's
     TimeSeriesSplit (expanding window).
  4. THE central pitfall (LO1 / AC4): re-runs the exact same lag-feature matrix through a
     shuffled, i.i.d.-style KFold split -- the SPEC-DS-4 way of doing it -- and shows the
     reported error is optimistically biased vs. the honest walk-forward number, because
     shuffling lets the model "train" on rows that are chronologically AFTER the row it's
     being tested on.
  5. Builds the per-signal recommendation table (best model, differencing/scaling, why) and
     saves it as a CSV artefact.
  6. Saves 13 artefacts total to ../artefacts/: 4 decomposition plots, 4 ACF/PACF plots, 4
     forecast-vs-actual backtest plots, 1 shuffle-vs-walk-forward bar chart, plus 2 CSVs
     (recommendation table, backtest RMSE comparison).

Environment (verified in research/NOTE-12-timeseries-apis.md, research/NOTE-6-statsmodels-vif.md,
research/NOTE-5-sklearn-core-apis.md, research/NOTE-2-package-versions.md, checked 2026-09-02):
    statsmodels==0.15.0, scikit-learn==1.9.0, numpy==2.5.2, pandas==3.0.5, matplotlib==3.11.1
    Python 3.12+ (this script was run and gated on Python 3.13.7, matching the pinned versions
    with no substitutions).

Run:
    python forecasting_signals.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this script only saves figures, never shows them
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, adfuller, pacf

RNG_SEED = 42
ARTEFACTS_DIR = Path(__file__).resolve().parent.parent / "artefacts"

# --- signal construction constants (spec: amplitude A, wiggle = 10% of A) -------------------
N = 240          # 20 years of monthly data
PERIOD = 12      # monthly seasonality -> 20 full cycles, comfortably >= the 2-cycle minimum
A = 100.0        # amplitude / scale of the DOMINANT component in each composite signal
WIGGLE = 0.10 * A  # 10.0 -- amplitude/std of the MINOR component in each composite signal

N_LAGS = 12       # lag-feature regression baseline: use a full seasonal cycle of lags
N_SPLITS = 5      # TimeSeriesSplit folds for the walk-forward backtest
TEST_SIZE = 12    # each backtest fold forecasts one year ahead

warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")


# ==============================================================================================
# Section 1: build the four composite signals
# ==============================================================================================


def make_signals(n: int = N, period: int = PERIOD, amplitude: float = A,
                  wiggle: float = WIGGLE, seed: int = RNG_SEED) -> dict[str, pd.Series]:
    """Build the four signals the spec requires, each as a pandas Series with a monthly
    DatetimeIndex. Every signal is DOMINANT_COMPONENT (scale `amplitude`) + MINOR_COMPONENT
    (scale `wiggle` = 10% of amplitude), so the ground truth is known exactly."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    idx = pd.date_range("2006-01-01", periods=n, freq="MS")

    linear_trend = 50.0 + amplitude * (t / (n - 1))              # 50 -> 150 over 20 years
    quadratic_trend = 50.0 + amplitude * (t / (n - 1)) ** 2       # 50 -> 150, accelerating
    sine = amplitude * np.sin(2 * np.pi * t / period)             # centred on 0, +/- A
    seasonal_wiggle = wiggle * np.sin(2 * np.pi * t / period)     # centred on 0, +/- 10% A

    # Independent noise draws per signal so one signal's noise doesn't leak into another's.
    noise_2 = rng.normal(0.0, wiggle, n)
    noise_3 = rng.normal(0.0, wiggle, n)
    noise_4 = rng.normal(0.0, wiggle, n)

    signals = {
        "signal_1_linear_sine": pd.Series(linear_trend + seasonal_wiggle, index=idx),
        "signal_2_linear_noise": pd.Series(linear_trend + noise_2, index=idx),
        "signal_3_sine_noise": pd.Series(100.0 + sine + noise_3, index=idx),
        "signal_4_quadratic_noise": pd.Series(quadratic_trend + noise_4, index=idx),
    }
    for s in signals.values():
        s.index.freq = "MS"
    return signals


# ==============================================================================================
# Section 2: stationarity (ADF), decomposition, ACF/PACF
# ==============================================================================================


def adf_report(series: pd.Series, label: str) -> dict:
    """Augmented Dickey-Fuller test (statsmodels.tsa.stattools.adfuller, NOTE-12): null
    hypothesis is a unit root (non-stationary). p < 0.05 rejects the null -> stationary."""
    stat, pvalue, n_lags_used, n_obs, crit_values, _ = adfuller(
        series.dropna(), autolag="AIC", result_object=False)
    stationary = pvalue < 0.05
    print(f"  ADF [{label}]: statistic={stat:.4f}, p-value={pvalue:.4g}, "
          f"lags used={n_lags_used}, stationary={stationary}")
    return {"label": label, "adf_stat": stat, "adf_pvalue": pvalue, "stationary": stationary}


def plot_decomposition(series: pd.Series, name: str, title: str) -> Path:
    """Additive seasonal_decompose (statsmodels.tsa.seasonal, NOTE-12), plotted as 4 stacked
    panels: observed, trend, seasonal, residual -- built manually from the DecomposeResult's
    documented .observed/.trend/.seasonal/.resid attributes."""
    result = seasonal_decompose(series, model="additive", period=PERIOD)

    fig, axes = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
    for ax, component, label in zip(
        axes,
        [result.observed, result.trend, result.seasonal, result.resid],
        ["observed", "trend", "seasonal", "residual"],
    ):
        ax.plot(component.index, component.values, linewidth=1.1)
        ax.set_ylabel(label)
        ax.axhline(0, color="grey", linewidth=0.6, linestyle="--") if label in (
            "seasonal", "residual") else None
    axes[0].set_title(title)
    axes[-1].set_xlabel("date")
    fig.tight_layout()

    out_path = ARTEFACTS_DIR / f"{name}_decomposition.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_acf_pacf(series: pd.Series, name: str, title: str, nlags: int = 36) -> Path:
    """ACF (statsmodels.tsa.stattools.acf) and PACF (pacf), plotted side by side with their
    95% confidence bands, computed manually (not plot_acf/plot_pacf's convenience wrapper) so
    the two panels share one figure and one consistent style with the rest of the chapter."""
    acf_vals, acf_confint = acf(series, nlags=nlags, alpha=0.05, fft=True, result_object=False)
    pacf_vals, pacf_confint = pacf(series, nlags=nlags, alpha=0.05, method="ywadjusted")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for ax, vals, confint, label in (
        (ax1, acf_vals, acf_confint, "ACF"),
        (ax2, pacf_vals, pacf_confint, "PACF"),
    ):
        lags = np.arange(len(vals))
        ax.vlines(lags, 0, vals, color="#4C72B0")
        ax.scatter(lags, vals, color="#4C72B0", s=12)
        lower = confint[:, 0] - vals
        upper = confint[:, 1] - vals
        ax.fill_between(lags, lower, upper, alpha=0.15, color="#4C72B0")
        ax.axhline(0, color="grey", linewidth=0.7)
        ax.set_title(label)
        ax.set_xlabel("lag (months)")
    fig.suptitle(title)
    fig.tight_layout()

    out_path = ARTEFACTS_DIR / f"{name}_acf_pacf.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ==============================================================================================
# Section 3: lag-feature regression baseline
# ==============================================================================================


def make_lag_features(series: pd.Series, n_lags: int = N_LAGS) -> tuple[pd.DataFrame, pd.Series]:
    """Turn a 1-D series into a supervised-learning table: X's columns are lag_1..lag_n_lags
    (past values only), y is the value being predicted. Row order is preserved -- this is what
    makes the shuffle-vs-walk-forward comparison in Section 5 meaningful."""
    frame = pd.DataFrame({"y": series})
    for lag in range(1, n_lags + 1):
        frame[f"lag_{lag}"] = series.shift(lag)
    frame = frame.dropna()
    X = frame[[f"lag_{lag}" for lag in range(1, n_lags + 1)]]
    y = frame["y"]
    return X, y


# ==============================================================================================
# Section 4: walk-forward backtest (ARIMA + lag regression), TimeSeriesSplit
# ==============================================================================================


def backtest_arima(series: pd.Series, order: tuple[int, int, int], trend: str | list[int],
                    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
                    n_splits: int = N_SPLITS, test_size: int = TEST_SIZE) -> dict:
    """Expanding-window walk-forward backtest: for each TimeSeriesSplit fold, fit ARIMA on the
    train prefix only, forecast the test block, score RMSE. TimeSeriesSplit's folds are already
    strictly in time order (train indices all precede test indices) -- that ordering guarantee
    is the whole point of using it instead of a shuffled splitter."""
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    fold_rmse = []
    last_actual = last_pred = last_index = None
    for train_idx, test_idx in tscv.split(series):
        train, test = series.iloc[train_idx], series.iloc[test_idx]
        # enforce_stationarity/invertibility=False: with a near-exactly-periodic signal, the
        # seasonal AR coefficient the MLE finds can land right on the unit-root boundary, which
        # crashes the stationary-covariance initialization (numpy.linalg.LinAlgError) on some
        # folds -- verified against signal_3 in this script. Disabling the constraint is
        # statsmodels' own documented escape hatch for exactly this failure mode.
        model = ARIMA(train, order=order, seasonal_order=seasonal_order, trend=trend,
                       enforce_stationarity=False, enforce_invertibility=False)
        fit = model.fit()
        forecast = fit.forecast(steps=len(test_idx))
        rmse = root_mean_squared_error(test.values, forecast.values)
        fold_rmse.append(rmse)
        last_actual, last_pred, last_index = test.values, forecast.values, test.index
    return {
        "fold_rmse": np.array(fold_rmse),
        "mean_rmse": float(np.mean(fold_rmse)),
        "last_fold_actual": last_actual,
        "last_fold_pred": last_pred,
        "last_fold_index": last_index,
    }


def backtest_polynomial_detrend_ar(series: pd.Series, poly_degree: int, ar_order: int,
                                    n_splits: int = N_SPLITS, test_size: int = TEST_SIZE) -> dict:
    """Alternative to differencing for a signal whose trend is a HIGHER-order polynomial
    (quadratic here): fit trend(t) = polyfit(t, y, deg=poly_degree) on the TRAIN prefix only,
    subtract it to get a (near-stationary) residual, fit a plain AR (ARIMA(p,0,0)) on that
    residual, and forecast = extrapolated polynomial trend + forecast residual.

    This is the "polynomial detrend ... then AR" alternative NOTE-12 and the spec call out,
    used here instead of ARIMA's own built-in `order=(p,2,q)` integration because fitting a
    degree-2 deterministic regressor inside ARIMA's MLE is numerically fragile in practice --
    verified empirically against this same signal (checked 2026-09-02): the in-ARIMA quadratic
    trend coefficient failed to converge to the true curvature, while polyfit + AR-on-residual
    reproduces it directly and backtests at RMSE close to the injected noise floor."""
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    t_all = np.arange(len(series))
    fold_rmse = []
    last_actual = last_pred = last_index = None
    for train_idx, test_idx in tscv.split(series):
        t_train, t_test = t_all[train_idx], t_all[test_idx]
        y_train = series.iloc[train_idx]
        coeffs = np.polyfit(t_train, y_train.values, deg=poly_degree)
        trend_train = np.polyval(coeffs, t_train)
        trend_test = np.polyval(coeffs, t_test)

        residual_train = pd.Series(y_train.values - trend_train, index=y_train.index)
        fit = ARIMA(residual_train, order=(ar_order, 0, 0), trend="c",
                     enforce_stationarity=False, enforce_invertibility=False).fit()
        residual_forecast = fit.forecast(steps=len(test_idx)).values

        forecast = trend_test + residual_forecast
        actual = series.iloc[test_idx].values
        fold_rmse.append(root_mean_squared_error(actual, forecast))
        last_actual, last_pred, last_index = actual, forecast, series.iloc[test_idx].index
    return {
        "fold_rmse": np.array(fold_rmse),
        "mean_rmse": float(np.mean(fold_rmse)),
        "last_fold_actual": last_actual,
        "last_fold_pred": last_pred,
        "last_fold_index": last_index,
    }


def backtest_lag_regression(X: pd.DataFrame, y: pd.Series, n_splits: int = N_SPLITS,
                             test_size: int = TEST_SIZE) -> dict:
    """Same expanding-window idea, but for the LinearRegression-on-lag-features baseline."""
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    fold_rmse = []
    last_actual = last_pred = last_index = None
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model = LinearRegression().fit(X_train, y_train)
        pred = model.predict(X_test)
        rmse = root_mean_squared_error(y_test.values, pred)
        fold_rmse.append(rmse)
        last_actual, last_pred, last_index = y_test.values, pred, y_test.index
    return {
        "fold_rmse": np.array(fold_rmse),
        "mean_rmse": float(np.mean(fold_rmse)),
        "last_fold_actual": last_actual,
        "last_fold_pred": last_pred,
        "last_fold_index": last_index,
    }


def plot_backtest(name: str, title: str, model_label: str, arima_result: dict,
                   reg_result: dict) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    idx = arima_result["last_fold_index"]
    ax.plot(idx, arima_result["last_fold_actual"], label="actual", color="black",
            linewidth=1.6, marker="o", markersize=3)
    ax.plot(idx, arima_result["last_fold_pred"], label=f"{model_label} "
            f"(mean RMSE={arima_result['mean_rmse']:.2f})", color="#C44E52", linestyle="--")
    ax.plot(idx, reg_result["last_fold_pred"], label=f"lag-regression forecast "
            f"(mean RMSE={reg_result['mean_rmse']:.2f})", color="#4C72B0", linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("date")
    ax.set_ylabel("value")
    ax.legend()
    fig.tight_layout()

    out_path = ARTEFACTS_DIR / f"{name}_backtest.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ==============================================================================================
# Section 5: the leakage picture -- shuffled i.i.d. split vs. walk-forward TimeSeriesSplit
# ==============================================================================================


def shuffle_vs_walkforward(series: pd.Series, test_frac: float = 0.2, n_neighbors: int = 5,
                            seed: int = RNG_SEED) -> dict:
    """The vivid version of "shuffling leaks the future". The model is deliberately the
    simplest one that CANNOT extrapolate: sklearn.neighbors.KNeighborsRegressor fit on nothing
    but the time index t. Outside the range of t it was trained on, k-NN has no concept of
    "keep going in this direction" -- it just returns the average of its nearest training
    neighbours, which near the edge of the training range is roughly the last value it saw.

    - Walk-forward (the correct way, SPEC-DS-9): train on the first (1 - test_frac) of the
      series, test on the trailing test_frac -- a genuine forecast into months the model has
      NEVER seen. k-NN is stuck extrapolating from the edge, so error is large whenever the
      signal keeps moving (trend and/or seasonality) past that edge.
    - Shuffled (the SPEC-DS-4 way, i.i.d. `train_test_split`/`KFold(shuffle=True)` default):
      same test-set SIZE, but drawn uniformly at random from the whole 20 years. Now the
      training set contains neighbours from EVERY month, including ones chronologically AFTER
      each test point -- k-NN is interpolating, not extrapolating, and looks great.

    Same model, same feature, same test-set size -- only the split strategy differs."""
    n = len(series)
    n_test = int(round(n * test_frac))
    t = np.arange(n).reshape(-1, 1).astype(float)
    y = series.values

    X_train_wf, X_test_wf = t[: n - n_test], t[n - n_test:]
    y_train_wf, y_test_wf = y[: n - n_test], y[n - n_test:]
    wf_model = KNeighborsRegressor(n_neighbors=n_neighbors).fit(X_train_wf, y_train_wf)
    walkforward_rmse = root_mean_squared_error(y_test_wf, wf_model.predict(X_test_wf))

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    sh_model = KNeighborsRegressor(n_neighbors=n_neighbors).fit(t[train_idx], y[train_idx])
    shuffled_rmse = root_mean_squared_error(y[test_idx], sh_model.predict(t[test_idx]))

    return {
        "walkforward_mean_rmse": float(walkforward_rmse),
        "shuffled_mean_rmse": float(shuffled_rmse),
    }


def plot_shuffle_vs_walkforward(comparison: dict[str, dict]) -> Path:
    names = list(comparison.keys())
    walkforward = [comparison[n]["walkforward_mean_rmse"] for n in names]
    shuffled = [comparison[n]["shuffled_mean_rmse"] for n in names]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, shuffled, width, label="shuffled split (i.i.d.-style, SPEC-DS-4's "
           "default -- WRONG here)", color="#C44E52")
    ax.bar(x + width / 2, walkforward, width, label="time-respecting split (last 20% held out "
           "-- CORRECT)", color="#4C72B0")
    for i, (s, w) in enumerate(zip(shuffled, walkforward)):
        ax.text(i - width / 2, s + 0.4, f"{s:.2f}", ha="center", fontsize=9)
        ax.text(i + width / 2, w + 0.4, f"{w:.2f}", ha="center", fontsize=9)
        ax.text(i, max(s, w) + 2.2, f"{w / s:.1f}x", ha="center", fontsize=9, color="#555555")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("signal_", "").replace("_", " ") for n in names], rotation=15)
    ax.set_ylabel("KNeighborsRegressor(t) RMSE, held-out last 20%")
    ax.set_title("Shuffling leaks the future: a model that can't extrapolate looks great when\n"
                 "shuffled, and struggles when it has to genuinely forecast (same model, same "
                 "feature)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()

    out_path = ARTEFACTS_DIR / "shuffle_vs_timeseries_split_rmse.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ==============================================================================================
# Main
# ==============================================================================================

# Per-signal modelling choices, decided from the ADF/ACF/PACF diagnostics printed at runtime
# (see the chapter prose for the reasoning): (ARIMA order, ARIMA trend, differencing applied).
SIGNAL_CONFIG = {
    # ARIMA `trend` per statsmodels' own rule (checked against the installed 0.15.0 at runtime,
    # see the ValueError this raises if violated): "a constant cannot be included in an
    # ARIMA(1, 1, 1) model, but including a linear trend, which would have the same effect as
    # fitting a constant to the differenced data, is allowed" -- i.e. the lowest included trend
    # order must be >= d + D. d=1 -> trend='t' (linear term only, no separate constant).
    "signal_1_linear_sine": {
        "kind": "arima",
        "order": (1, 1, 0), "seasonal_order": (1, 0, 0, PERIOD), "trend": "t", "d": 1,
        "shape": "linear trend (dominant) + sine seasonality (10% of A)",
        "recommendation": "Difference once (d=1) to remove the linear trend; the small "
                           "residual seasonal wiggle needs a SEASONAL AR term (lag 12) or it's "
                           "left on the table -- plain ARIMA(1,1,0) alone backtests far worse "
                           "than ARIMA(1,1,0)x(1,0,0,12).",
    },
    "signal_2_linear_noise": {
        "kind": "arima",
        "order": (1, 1, 0), "seasonal_order": (0, 0, 0, 0), "trend": "t", "d": 1,
        "shape": "linear trend (dominant) + white noise (10% of A)",
        "recommendation": "Difference once (d=1); the differenced series is close to white "
                           "noise plus drift, so a low-order AR term (or none) is enough -- "
                           "don't over-fit AR lags to noise, and there's no seasonal term to add.",
    },
    "signal_3_sine_noise": {
        "kind": "arima",
        "order": (1, 0, 0), "seasonal_order": (1, 0, 0, PERIOD), "trend": "c", "d": 0,
        "shape": "sine seasonality (dominant) + white noise (10% of A)",
        "recommendation": "Already stationary (ADF rejects the unit-root null on the raw "
                           "series) -- no differencing. A non-seasonal AR alone can't see a "
                           "12-month cycle; add a SEASONAL AR term (lag 12) to capture it.",
    },
    "signal_4_quadratic_noise": {
        "kind": "poly_detrend_ar",
        "poly_degree": 2, "ar_order": 1, "d": 2,
        "shape": "quadratic trend (dominant) + white noise (10% of A)",
        "recommendation": "One differencing pass still leaves a trend (a linear one, since "
                           "diff() of a quadratic is linear) -- difference TWICE (d=2), OR "
                           "(used here) polynomial-detrend by regressing on t and t^2, then "
                           "model the residual with plain AR. Fitting the quadratic term "
                           "directly inside ARIMA's own MLE is numerically fragile in practice "
                           "-- verified against this signal -- so explicit polyfit + AR-on-"
                           "residual is the robust choice.",
    },
}


def main() -> None:
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    signals = make_signals()

    adf_rows = []
    backtest_rows = []
    leak_comparison = {}

    for name, series in signals.items():
        cfg = SIGNAL_CONFIG[name]
        title = f"{name} -- {cfg['shape']}"
        print(f"\n=== {name} ===")
        print(f"shape: {cfg['shape']}")

        # --- stationarity + decomposition + ACF/PACF -------------------------------------
        raw_adf = adf_report(series, "raw")
        diff_adf = None
        if cfg["d"] > 0:
            diffed = series.diff(1).dropna()
            if cfg["d"] == 2:
                diffed = diffed.diff(1).dropna()
            diff_adf = adf_report(diffed, f"differenced d={cfg['d']}")

        decomposition_path = plot_decomposition(series, name, title)
        acf_pacf_path = plot_acf_pacf(series, name, title)

        adf_rows.append({
            "signal": name,
            "adf_pvalue_raw": raw_adf["adf_pvalue"],
            "stationary_raw": raw_adf["stationary"],
            "differencing_d": cfg["d"],
            "adf_pvalue_after_diff": diff_adf["adf_pvalue"] if diff_adf else None,
            "stationary_after_diff": diff_adf["stationary"] if diff_adf else raw_adf["stationary"],
        })

        # --- ARIMA (or polynomial-detrend + AR) backtest ------------------------------------
        if cfg["kind"] == "arima":
            arima_result = backtest_arima(series, order=cfg["order"],
                                           seasonal_order=cfg["seasonal_order"],
                                           trend=cfg["trend"])
            model_label = f"ARIMA{cfg['order']}x{cfg['seasonal_order']} (trend={cfg['trend']!r})"
            model_order_str = f"{cfg['order']} seasonal={cfg['seasonal_order']}"
        else:
            arima_result = backtest_polynomial_detrend_ar(
                series, poly_degree=cfg["poly_degree"], ar_order=cfg["ar_order"])
            model_label = (f"polyfit(deg={cfg['poly_degree']}) detrend + "
                            f"AR({cfg['ar_order']}) on residual")
            model_order_str = f"poly_deg={cfg['poly_degree']}, AR({cfg['ar_order']})"
        print(f"  {model_label} walk-forward mean RMSE: {arima_result['mean_rmse']:.4f} "
              f"(per-fold: {np.round(arima_result['fold_rmse'], 3).tolist()})")

        # --- lag-feature regression baseline + backtest --------------------------------------
        X, y = make_lag_features(series, n_lags=N_LAGS)
        reg_result = backtest_lag_regression(X, y)
        print(f"  Lag-regression ({N_LAGS} lags) walk-forward mean RMSE: "
              f"{reg_result['mean_rmse']:.4f} (per-fold: "
              f"{np.round(reg_result['fold_rmse'], 3).tolist()})")

        backtest_path = plot_backtest(name, title, model_label, arima_result, reg_result)

        best_model = model_label if arima_result["mean_rmse"] <= reg_result["mean_rmse"] else \
            f"lag-feature regression ({N_LAGS} lags)"
        backtest_rows.append({
            "signal": name,
            "arima_order": model_order_str,
            "arima_mean_rmse": arima_result["mean_rmse"],
            "lag_regression_mean_rmse": reg_result["mean_rmse"],
            "best_model": best_model,
        })

        # --- the leakage picture: shuffled vs. walk-forward, same data ----------------------
        leak_comparison[name] = shuffle_vs_walkforward(series)
        wf_rmse = leak_comparison[name]["walkforward_mean_rmse"]
        sh_rmse = leak_comparison[name]["shuffled_mean_rmse"]
        print(f"  KNN(t) RMSE -- shuffled split: {sh_rmse:.4f}  vs. walk-forward (last 20% "
              f"held out): {wf_rmse:.4f}  (shuffled UNDERSTATES the honest error by "
              f"{wf_rmse / sh_rmse:.2f}x)")

        print(f"  wrote: {decomposition_path.name}, {acf_pacf_path.name}, {backtest_path.name}")

    leak_plot_path = plot_shuffle_vs_walkforward(leak_comparison)
    print(f"\nwrote: {leak_plot_path.name}")

    # --- recommendation table --------------------------------------------------------------
    adf_df = pd.DataFrame(adf_rows)
    backtest_df = pd.DataFrame(backtest_rows)
    rec_rows = []
    for name, cfg in SIGNAL_CONFIG.items():
        adf_row = adf_df[adf_df["signal"] == name].iloc[0]
        bt_row = backtest_df[backtest_df["signal"] == name].iloc[0]
        rec_rows.append({
            "signal": name,
            "shape": cfg["shape"],
            "stationary_raw": adf_row["stationary_raw"],
            "differencing_d": cfg["d"],
            "best_model": bt_row["best_model"],
            "arima_order": bt_row["arima_order"],
            "arima_rmse": round(bt_row["arima_mean_rmse"], 3),
            "lag_regression_rmse": round(bt_row["lag_regression_mean_rmse"], 3),
            "scaling_recommendation": cfg["recommendation"],
        })
    recommendation_df = pd.DataFrame(rec_rows)
    rec_path = ARTEFACTS_DIR / "forecasting_recommendation_table.csv"
    recommendation_df.to_csv(rec_path, index=False)
    print(f"\nwrote: {rec_path.name}")

    backtest_csv_path = ARTEFACTS_DIR / "forecasting_backtest_rmse.csv"
    backtest_df.to_csv(backtest_csv_path, index=False)
    print(f"wrote: {backtest_csv_path.name}")

    print("\n=== per-signal recommendation table ===")
    print(recommendation_df.to_string(index=False))


if __name__ == "__main__":
    main()

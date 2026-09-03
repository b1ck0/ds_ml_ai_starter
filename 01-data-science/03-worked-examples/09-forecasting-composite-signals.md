# Forecasting Composite Signals — Trend, Seasonality, and Why the Split Is Different

*Data Science · Worked Examples · SPEC-DS-9*

## The forecast is a confession

In 1970, the statisticians George Box and Gwilym Jenkins published *Time Series Analysis:
Forecasting and Control* — a book that gave the world a disciplined way to look at a wobbling
line (sales, temperature, a sensor reading) and say, out loud, "here is what I expect next"
([source: Wikipedia, "Box–Jenkins method"](https://en.wikipedia.org/wiki/Box%E2%80%93Jenkins_method),
checked 2026-09-03). Their initials survive in the model family this chapter fits: **ARIMA** —
AutoRegressive Integrated Moving Average — is the Box-Jenkins method, packaged into one function
call.

Strip away the statistics and the problem is one you already own: you're staffing a warehouse and
need next month's order volume, or you're on call for a sensor that logs a reading every hour and
want to know whether tomorrow's value is going to trip an alert. Both are the same question
regression already answered in the [taxi-fare chapter](05-regression-nyc-taxi.md) — a label, some
features, a fitted formula, a way to grade it — except now the rows arrive in a strict order, and
that one fact quietly breaks almost every rule that chapter relied on.

One sentence you could repeat at dinner: **forecasting is regression where the calendar is also a
feature, and shuffling the calendar breaks everything.**

```mermaid
flowchart LR
    A["Step 1<br/>build 4 signals<br/>with a KNOWN true shape"] --> B["Step 2<br/>shuffle vs. walk-forward<br/>(feel the lie)"]
    B --> C["Step 3<br/>decompose:<br/>trend + seasonal + residual"]
    C --> D["Step 4<br/>diagnose:<br/>stationarity, ACF/PACF"]
    D --> E["Step 5<br/>model + walk-forward backtest<br/>per signal"]
    E --> F["Step 6<br/>recommend per signal,<br/>know the pitfalls"]
    F -.->|"the discipline every real forecast needs"| A
```

The [train/validation/holdout chapter](04-train-valid-holdout-split.md) drew a boundary around
itself on purpose: "This chapter covers the simple, non-temporal case: every row is an independent
observation … Time-series forecasting needs a different splitting strategy entirely — rows are not
independent across time, and a random split would let the model train on the future and be
'validated' on the past." This chapter picks up exactly there, and Step 2 above is where it starts
— not with a definition, but with a number that should make you distrust every forecast metric
you've never checked this way.

To keep every diagnostic checkable against ground truth, everything here runs on four
**synthetic** composite signals — built from a known formula, not downloaded — the same reason
[the collinearity chapter](03-collinearity.md) used a synthetic house-price table: you can't
verify a decomposition or a stationarity test is reading a signal correctly unless you already
know what's really in it.

## 1. The lie a shuffled split tells you

### 1.1 Four signals, one true shape each

Every experiment in this chapter runs on four composite synthetic signals, each built from a
**dominant** component at scale `A` plus a **minor** component at 10% of `A` (the spec calls this the
"wiggle") — 20 years of monthly data, `A = 100`, wiggle `= 10`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd

RNG_SEED = 42
N = 240          # 20 years of monthly data
PERIOD = 12      # monthly seasonality -> 20 full cycles
A = 100.0        # amplitude/scale of the DOMINANT component
WIGGLE = 0.10 * A  # 10.0 -- amplitude/std of the MINOR component (10% of A)


def make_signals(n: int = N, period: int = PERIOD, amplitude: float = A,
                  wiggle: float = WIGGLE, seed: int = RNG_SEED) -> dict[str, pd.Series]:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    idx = pd.date_range("2006-01-01", periods=n, freq="MS")

    linear_trend = 50.0 + amplitude * (t / (n - 1))          # 50 -> 150 over 20 years
    quadratic_trend = 50.0 + amplitude * (t / (n - 1)) ** 2   # 50 -> 150, accelerating
    sine = amplitude * np.sin(2 * np.pi * t / period)          # centred on 0, +/- A
    seasonal_wiggle = wiggle * np.sin(2 * np.pi * t / period)  # centred on 0, +/- 10% A

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


signals = make_signals()
```

Four shapes, matching the spec exactly:

1. **`signal_1_linear_sine`** — linear trend (dominant) + sine seasonality (10% of A). Trend
   dominates; the seasonal wiggle is a small, exactly-periodic ripple on top.
2. **`signal_2_linear_noise`** — linear trend (dominant) + white noise (10% of A). Same trend, but
   the minor component is unpredictable Gaussian noise, not a repeating pattern.
3. **`signal_3_sine_noise`** — sine wave (dominant, full amplitude A) + white noise (10% of A). No
   trend at all — the whole signal oscillates around a fixed level.
4. **`signal_4_quadratic_noise`** — quadratic trend (dominant) + white noise (10% of A). A
   *non-linear*, accelerating trend, plus noise.

Because these formulas are known, every diagnostic in the rest of this chapter can be checked
against ground truth instead of taken on faith — including the very first one, right now.

### 1.2 Same model, two splits, two very different stories

Here's the leak as a number, not just an argument. Take the simplest model that structurally
*cannot* extrapolate —
[`KNeighborsRegressor(n_neighbors=5, *, weights='uniform', algorithm='auto', ...)`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)
(signature verified directly against the installed scikit-learn 1.9.0, checked 2026-09-02 — it isn't
one of NOTE-5's tabulated APIs, so this chapter confirmed it against the live environment the same way
[the split chapter](04-train-valid-holdout-split.md) verified `load_breast_cancer()`), predicting purely
from the time index `t` — and evaluate it two ways on each signal, holding out the
same 20% of rows (48 months) either way:

```python
from sklearn.metrics import root_mean_squared_error
from sklearn.neighbors import KNeighborsRegressor


def shuffle_vs_walkforward(series: pd.Series, test_frac: float = 0.2, n_neighbors: int = 5,
                            seed: int = RNG_SEED) -> dict:
    n = len(series)
    n_test = int(round(n * test_frac))
    t = np.arange(n).reshape(-1, 1).astype(float)
    y = series.values

    # walk-forward (CORRECT): train on the first 80%, test on the trailing 20% -- a genuine
    # forecast into months the model has never seen.
    X_train_wf, X_test_wf = t[: n - n_test], t[n - n_test:]
    y_train_wf, y_test_wf = y[: n - n_test], y[n - n_test:]
    wf_model = KNeighborsRegressor(n_neighbors=n_neighbors).fit(X_train_wf, y_train_wf)
    walkforward_rmse = root_mean_squared_error(y_test_wf, wf_model.predict(X_test_wf))

    # shuffled (SPEC-DS-4's default, WRONG here): same test-set SIZE, drawn uniformly at random.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    sh_model = KNeighborsRegressor(n_neighbors=n_neighbors).fit(t[train_idx], y[train_idx])
    shuffled_rmse = root_mean_squared_error(y[test_idx], sh_model.predict(t[test_idx]))

    return {"walkforward_mean_rmse": float(walkforward_rmse),
            "shuffled_mean_rmse": float(shuffled_rmse)}


leak_comparison = {name: shuffle_vs_walkforward(s) for name, s in signals.items()}
```

```text
signal_1_linear_sine    : shuffled=4.89   walk-forward=20.30   (walk-forward is 4.15x worse)
signal_2_linear_noise   : shuffled=9.19   walk-forward=13.78   (walk-forward is 1.50x worse)
signal_3_sine_noise     : shuffled=49.17  walk-forward=102.16  (walk-forward is 2.08x worse)
signal_4_quadratic_noise: shuffled=12.72  walk-forward=24.14   (walk-forward is 1.90x worse)
```

![Grouped bar chart, one pair of bars per signal, comparing K-Nearest-Neighbours-on-time-index RMSE under a shuffled i.i.d.-style split versus a walk-forward split that holds out the last 20% of the series; the walk-forward bar is 1.5x to 4.2x taller than the shuffled bar for every signal](artefacts/shuffle_vs_timeseries_split_rmse.png)

Read that bar chart slowly — it's the whole chapter's thesis in one picture. Same model, same
single feature, same test-set *size* — the only thing that changed is which rows landed in "train"
versus "test." Under the shuffled split, k-NN's training set contains neighbours from every month
across the whole 20 years, including months chronologically *after* every test point, so it's
always interpolating between two known values. Under the walk-forward split, it has to predict
months it has never seen anything past — genuine extrapolation — and a k-NN model has no way to do
that except repeat whatever its nearest training neighbour (the last month it saw) told it. The
shuffled number isn't just optimistic, it's answering a *different question*: "how well can I fill
in a gap in a sequence I've already seen the whole span of" instead of "how well can I predict what
hasn't happened yet." Only the second question is what a production forecaster actually has to
answer — and every signal above shows the same direction of lie, 1.5x to 4.2x, before you've fit a
single "real" forecasting model.

This is the same shape of bug as [Section 3 of the split chapter](04-train-valid-holdout-split.md#3-the-leakage-demo-fit-on-the-whole-dataset-get-an-optimistic-score)
— fitting on data the model shouldn't have seen yet, reported as if it were an honest number — but
there it took 200 repeated small-training-set splits and a paired t-test to make the gap
statistically undeniable (the single-split gap was identical to six decimal places). Here it doesn't:
every signal in this chapter shows a 1.5x–4.2x gap on a single, ordinary-sized split, because time
order isn't a subtle correlation between two feature columns — it's the entire structure of the
problem.

### 1.3 Deriving the fix: why the calendar breaks shuffling

So how does a shuffled split manage to lie *that* confidently? Because
[`sklearn.model_selection.train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
defaults to `shuffle=True` — exactly the right call for the breast-cancer rows in the split chapter,
because a tumour sample doesn't know or care what row number it landed on. A monthly revenue figure
is not like that: row `t` was measured *after* row `t-1` and *before* row `t+1`, and a model that gets
to see row `t+1` while it's being fit and is then asked to predict row `t` isn't forecasting — it's
interpolating with a sneak preview. Shuffle a time series before splitting and that's exactly what
happens: some training rows sit chronologically *after* some test rows, so the model gets to train on
information that, in the real deployment scenario this whole exercise is standing in for, doesn't
exist yet.

```mermaid
flowchart TD
    subgraph SHUFFLED["shuffled split (WRONG for time series)"]
        S1["random 80% of all 240 months"] --> S2["train"]
        S3["random 20% of all 240 months<br/>(scattered across the whole 20 years)"] --> S4["test"]
    end
    subgraph WALKFWD["walk-forward split (CORRECT)"]
        W1["months 1-192<br/>(the past)"] --> W2["train"]
        W3["months 193-240<br/>(never-seen future)"] --> W4["test"]
    end
    SHUFFLED --> LEAK["train contains months AFTER some test months<br/>-- model interpolates, RMSE looks great but lies"]
    WALKFWD --> HONEST["train always precedes test<br/>-- model extrapolates, RMSE is the honest number"]
```

scikit-learn 1.9.0 ships the fix as
[`TimeSeriesSplit(n_splits=5, max_train_size=None, test_size=None, gap=0)`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
— it performs **expanding-window cross-validation**: fold *k*'s training set is every row before fold
*k*'s test block, and the training set grows fold over fold, never shrinks, and never contains a row
that comes after its own test block
([source: NOTE-12-timeseries-apis](../../research/NOTE-12-timeseries-apis.md), checked 2026-09-02).
This particular way of evaluating a forecaster — refit (or re-use) the model at each step using only
data available up to that point, forecast forward, then advance — is called a **walk-forward** or
**expanding-window backtest**: "a validation methodology that simulates live forecasting by using all
previous data … for training at each step, [with] the training set expand[ing] with each fold,
preventing future data leakage into the past"
([source: NOTE-12-timeseries-apis](../../research/NOTE-12-timeseries-apis.md), citing
[Machine Learning Mastery: How to Backtest Machine Learning Models for Time Series Forecasting](https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/)
and [QuantInsti: Walk-Forward Optimization](https://blog.quantinsti.com/walk-forward-optimization-introduction/),
checked 2026-09-02).

Picture the rotation as five folds sliding forward through the same 240 months, the training window
always growing and the test window always sitting strictly after it:

```mermaid
flowchart TD
    F1["fold 1<br/>train: earliest rows   test: next 12 months"]
    F2["fold 2<br/>train: fold 1's rows + fold 1's test   test: next 12 months"]
    F3["fold 3<br/>train: everything before   test: next 12 months"]
    F4["fold 4<br/>train: everything before   test: next 12 months"]
    F5["fold 5<br/>train: everything before   test: next 12 months"]
    F1 --> F2 --> F3 --> F4 --> F5
    F5 --> NOTE["training set only ever GROWS (n_splits=5, test_size=12)<br/>-- every fold's test block comes strictly after its own train block"]
```

This is `TimeSeriesSplit(n_splits=5, test_size=12)`, and it's the tool Section 4 uses to backtest
every model in this chapter — never a plain `train_test_split` or `KFold`, whose shuffling would
reintroduce exactly the lie Section 1.2 just measured.

## 2. Decomposing each signal

Each signal above breaks visually into trend, seasonality, and leftover residual — that decomposition
is a diagnostic tool in its own right, not just a picture, and it's the next stop on the roadmap: raw
signal in, three interpretable pieces out, feeding straight into the stationarity and ACF/PACF
diagnostics of Section 3.

```mermaid
flowchart LR
    RAW["raw signal"] --> DECOMP["decompose<br/>(seasonal_decompose)<br/>trend + seasonal + residual"]
    DECOMP --> DIAG["diagnose<br/>ADF test (stationary?)<br/>+ ACF/PACF (which lags matter?)"]
    DIAG -->|"non-stationary"| DIFF["difference<br/>(d=1 linear, d=2 quadratic)<br/>or polynomial-detrend"]
    DIFF --> DIAG
    DIAG -->|"stationary"| MODEL["model<br/>AR/ARIMA or lag-feature regression"]
    MODEL --> BACKTEST["walk-forward backtest<br/>(TimeSeriesSplit, Section 1.3)"]
```

**Seasonality**, in plain language, is a pattern that repeats on a fixed, known clock — every 12
months, every 7 days — as opposed to a **trend** (a slow drift with no fixed period) or noise (no
repeating pattern at all). statsmodels 0.15.0's
[`seasonal_decompose(x, model='additive', filt=None, period=None, two_sided=True, extrapolate_trend=0)`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.seasonal_decompose.html)
splits a series into an additive `observed = trend + seasonal + residual` (or multiplicative)
decomposition, using a centred moving average for the trend component
([source: NOTE-12-timeseries-apis](../../research/NOTE-12-timeseries-apis.md), checked 2026-09-02).
It needs at least two full seasonal cycles to work — this chapter's 240 months at `period=12` gives it
20:

```python
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

PERIOD = 12


def plot_decomposition(series: pd.Series, name: str, title: str) -> None:
    result = seasonal_decompose(series, model="additive", period=PERIOD)

    fig, axes = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
    for ax, component, label in zip(
        axes,
        [result.observed, result.trend, result.seasonal, result.resid],
        ["observed", "trend", "seasonal", "residual"],
    ):
        ax.plot(component.index, component.values, linewidth=1.1)
        ax.set_ylabel(label)
    axes[0].set_title(title)
    axes[-1].set_xlabel("date")
    fig.tight_layout()
    fig.savefig(f"{name}_decomposition.png", dpi=150)
    plt.close(fig)
```

Reading the four decompositions side by side:

**`signal_1_linear_sine`** (linear trend + sine seasonality, no noise):

![Four-panel decomposition of signal_1_linear_sine: observed shows a smooth linear rise with a regular small ripple; trend is a clean straight line; seasonal is a perfectly regular sine wave repeating every 12 months; residual is flat at zero everywhere](artefacts/signal_1_linear_sine_decomposition.png)

The residual panel sits essentially at zero the entire way — because signal 1 has **no random
component at all**. Trend and seasonal account for 100% of the signal by construction, which is the
whole point of using a synthetic signal here: you get to *confirm* the decomposition found exactly
what you built in, rather than trusting it on faith.

**`signal_2_linear_noise`** (linear trend + white noise):

![Four-panel decomposition of signal_2_linear_noise: observed is a noisy upward-trending line; trend is a smooth rising curve close to the underlying linear trend; seasonal shows a small but non-zero repeating pattern the decomposition mistakenly extracted from noise; residual looks like unstructured noise centred near zero](artefacts/signal_2_linear_noise_decomposition.png)

The trend panel tracks the true linear trend closely. But look at the "seasonal" panel: signal 2 was
built with **no seasonality whatsoever**, yet `seasonal_decompose` still reports a small,
repeating pattern there — because the function doesn't test *whether* a series is seasonal, it
*assumes* `period=12` and averages accordingly, extracting whatever 12-month-periodic structure
random noise happens to produce over 20 years. This is a real pitfall of the function, not a bug in
this chapter's code: **`seasonal_decompose` will hand you a "seasonal" component whether or not the
data actually has one — pair it with the ACF/PACF diagnostics in Section 3, which distinguish real
periodicity from noise, rather than trusting the seasonal panel alone.**

**`signal_3_sine_noise`** (sine, no trend, + white noise) and **`signal_4_quadratic_noise`**
(quadratic trend + white noise) follow the same reading:

![Four-panel decomposition of signal_3_sine_noise: observed oscillates around a fixed level; trend is flat/near-zero; seasonal is a clean, large-amplitude sine wave; residual is noisy](artefacts/signal_3_sine_noise_decomposition.png)

![Four-panel decomposition of signal_4_quadratic_noise: observed accelerates upward increasingly steeply; trend is a smooth curve bending upward, steeper at the end than the start; seasonal is small and noise-like (no true seasonality); residual is noisy](artefacts/signal_4_quadratic_noise_decomposition.png)

Signal 3's trend panel is flat — correctly, there is none — and its seasonal panel recovers the true
sine wave cleanly, since seasonality really is what's there. Signal 4's trend panel visibly *bends*
rather than staying straight, the tell for a non-linear trend a plain linear detrend would miss.

## 3. Stationarity, ACF/PACF, and differencing

### 3.1 Stationarity and the Augmented Dickey-Fuller test

In plain language: a series is **stationary** if it looks statistically the same in any window you
crop out of it — same average level, same spread, same tendency to move together with its own past
— no matter which stretch of the 20 years you happen to be looking at. Formally, "its mean,
variance, and autocovariance remain constant over time"
([source: NOTE-12-timeseries-apis](../../research/NOTE-12-timeseries-apis.md), citing
[QuantInsti: Stationarity](https://blog.quantinsti.com/stationarity/), checked 2026-09-02):

$$E[y_t] = \mu, \qquad \mathrm{Var}(y_t) = \sigma^2, \qquad \mathrm{Cov}(y_t, y_{t-k}) = \gamma_k$$

— none of the three depends on $t$ itself, only $\gamma_k$ depends on the lag $k$. ($\mu$ = "the
series' average level," $\sigma^2$ = "how spread out it is," $\gamma_k$ = "how strongly a value and
the value $k$ steps earlier move together" — none of them are allowed to drift as time passes.)

AR/ARIMA models need this, because an AR model's coefficients describe one fixed relationship
between a value and its own past; if the mean is drifting (a trend) that "one fixed relationship"
doesn't exist — you'd be asking one number to describe a relationship that's different in year 1
than in year 20. statsmodels 0.15.0's
[`adfuller(x, maxlag=None, regression='c', autolag='AIC', ...)`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html)
runs the **Augmented Dickey-Fuller test**: its null hypothesis is that the series has a unit root
(i.e. is *non*-stationary); a p-value below 0.05 rejects that null, meaning the test found the series
stationary ([source: NOTE-12-timeseries-apis](../../research/NOTE-12-timeseries-apis.md)).

```mermaid
flowchart LR
    ADF["adfuller(series)"] --> P{"p-value &lt; 0.05?"}
    P -->|"yes"| STAT["stationary --<br/>model directly (signal 3)"]
    P -->|"no"| NONSTAT["non-stationary --<br/>difference (d=1 or d=2)<br/>then re-run adfuller"]
    NONSTAT -.->|"re-test"| ADF
```

```python
from statsmodels.tsa.stattools import adfuller


def adf_report(series: pd.Series, label: str) -> dict:
    stat, pvalue, n_lags_used, n_obs, crit_values, _ = adfuller(
        series.dropna(), autolag="AIC", result_object=False)
    stationary = pvalue < 0.05
    print(f"ADF [{label}]: statistic={stat:.4f}, p-value={pvalue:.4g}, stationary={stationary}")
    return {"adf_stat": stat, "adf_pvalue": pvalue, "stationary": stationary}
```

Run on the raw signals:

| signal | ADF statistic | p-value | stationary? |
|---|---|---|---|
| `signal_1_linear_sine` | 0.0005 | 0.9586 | **No** — a trend is present |
| `signal_2_linear_noise` | -0.6183 | 0.8668 | **No** — a trend is present |
| `signal_3_sine_noise` | -4.3210 | 0.0004 | **Yes** — no trend, oscillates around a fixed level |
| `signal_4_quadratic_noise` | 0.7120 | 0.9901 | **No** — a (non-linear) trend is present |

Exactly as the decompositions suggested: the three trending signals fail the stationarity test, and
the pure oscillator (signal 3) passes it. **Differencing** — subtracting each value from the one
before it, $y_t - y_{t-1}$ — removes a linear trend; a second differencing pass removes a quadratic
one, because differencing a degree-`k` polynomial in `t` produces a degree-`(k-1)` polynomial
([source: NOTE-12-timeseries-apis](../../research/NOTE-12-timeseries-apis.md)):

| signal | differencing applied | ADF p-value after | stationary after? |
|---|---|---|---|
| `signal_1_linear_sine` | `d=1` | ~0 (see note below) | **Yes** |
| `signal_2_linear_noise` | `d=1` | 3.36e-12 | **Yes** |
| `signal_4_quadratic_noise` | `d=2` (twice) | 2.97e-13 | **Yes** |

One number needs a caveat: signal 1's differenced ADF statistic printed as roughly -1.25e14, an
absurd-looking magnitude. This is a genuine artefact of signal 1 having **zero random noise** — its
first difference is an *exactly deterministic* sequence (a constant step plus a perfectly repeating
differenced sine), leaving the ADF test's residual variance essentially at floating-point zero, which
blows up the ratio the test statistic is built from. The p-value is still meaningfully ~0 (reject
non-stationarity), but treat a test statistic of that magnitude as "this series has no meaningful
noise left to test," not as "extremely, unusually stationary" — a real dataset will never produce a
number like this, only a synthetic one built with no noise term can.

### 3.2 ACF and PACF

In plain language, autocorrelation asks one question, repeated at every lag: *if I already know
last month's value, how much does that tell me about this month's?* A high autocorrelation at lag 1
means consecutive months move together; a spike at lag 12 means "a value 12 months ago predicts
this one" — the fingerprint of a yearly cycle. Formally, **autocorrelation** measures "the linear
relationship between lagged values of a time series … how a series correlates with its own past
values at different time lags"
([source: NOTE-12-timeseries-apis](../../research/NOTE-12-timeseries-apis.md), citing
[Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* — Autocorrelation](https://otexts.com/fpp2/autocorrelation.html),
checked 2026-09-02):

$$r_k = \mathrm{Corr}(y_t,\, y_{t-k})$$

— "the correlation between the series and a copy of itself shifted back by $k$ steps." The **ACF**
(autocorrelation function) plots $r_k$ at every lag directly; the **PACF** (partial autocorrelation
function) plots the correlation at each lag *after* removing the effect already explained by
shorter lags — which is what makes PACF the tool for reading off an AR model's order: an AR(p)
process's PACF cuts off sharply after lag `p`. statsmodels 0.15.0 provides both as
[`acf(x, nlags=None, alpha=None, fft=True, ...)`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html)
and
[`pacf(x, nlags=None, method='ywadjusted', alpha=None, ...)`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.pacf.html)
([source: NOTE-12-timeseries-apis](../../research/NOTE-12-timeseries-apis.md)):

```python
from statsmodels.tsa.stattools import acf, pacf


def plot_acf_pacf(series: pd.Series, name: str, title: str, nlags: int = 36) -> None:
    acf_vals, acf_confint = acf(series, nlags=nlags, alpha=0.05, fft=True, result_object=False)
    pacf_vals, pacf_confint = pacf(series, nlags=nlags, alpha=0.05, method="ywadjusted")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for ax, vals, confint, label in ((ax1, acf_vals, acf_confint, "ACF"),
                                      (ax2, pacf_vals, pacf_confint, "PACF")):
        lags = np.arange(len(vals))
        ax.vlines(lags, 0, vals, color="#4C72B0")
        ax.fill_between(lags, confint[:, 0] - vals, confint[:, 1] - vals, alpha=0.15)
        ax.axhline(0, color="grey", linewidth=0.7)
        ax.set_title(label)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(f"{name}_acf_pacf.png", dpi=150)
    plt.close(fig)
```

![ACF and PACF side by side for signal_1_linear_sine: ACF decays very slowly and stays high across all 36 lags shown -- the classic signature of a trending, non-stationary series; PACF spikes at lag 1, dips sharply negative at lag 2, and shows small ripples near lags 6-8](artefacts/signal_1_linear_sine_acf_pacf.png)

Signal 1's ACF barely decays across 36 lags (three years) — a slowly-decaying ACF that never crosses
the confidence band is the classic visual signature of a *non-stationary* (trending) series, matching
the ADF result above exactly: every lag is correlated with every other lag mainly because they're all
riding the same upward trend, not because of any short-range dependency. This is precisely why you
read ACF/PACF *after* checking stationarity (Section 3.1), not instead of it — on a trending series,
ACF's shape tells you "there's a trend," not "here's the AR order."

![ACF and PACF side by side for signal_3_sine_noise: ACF oscillates in a decaying wave pattern, peaking near lags 12, 24, 36; PACF spikes sharply at lag 1 and lag 12 with everything else inside the confidence band](artefacts/signal_3_sine_noise_acf_pacf.png)

Signal 3 (already stationary — no differencing needed) tells a completely different, much more
useful story: ACF oscillates with peaks recurring every 12 lags — read directly off the plot, that's
the 12-month seasonal period, found from the data with no formula, only the plot. PACF spikes at lag 1
and lag 12 and stays inside the confidence band everywhere else — the textbook signature that both a
short-range AR(1) term and a **seasonal** AR term at lag 12 belong in the model. Section 4 uses exactly
that reading to choose signal 3's model order.

## 4. Modelling each signal: AR/ARIMA vs. a lag-feature regression baseline

### 4.1 The two model families

**AR(p)** — an autoregressive model of order `p` — is "a regression of the variable against itself"
([source: NOTE-12-timeseries-apis](../../research/NOTE-12-timeseries-apis.md), citing
[Hyndman & Athanasopoulos — Autoregressive models](https://otexts.com/fpp2/AR.html), checked
2026-09-02):

$$y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + \dots + \phi_p y_{t-p} + \varepsilon_t$$

where $y_t$ is "this month's value," each $\phi_i$ is "how much weight the value $i$ months back
gets" (a coefficient the model fits, exactly like a regression weight), $c$ is a constant offset,
and $\varepsilon_t$ is white noise — "the part no past value can explain." **ARIMA(p, d, q)**
generalises it with differencing (`d`, Section 3.1) and a moving-average term (`q`); statsmodels
0.15.0 exposes it as
[`ARIMA(endog, order=(p,d,q), seasonal_order=(P,D,Q,s), trend=None, ...)`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html),
where `seasonal_order` adds a *second*, seasonal AR/MA structure at period `s`
([source: NOTE-12-timeseries-apis](../../research/NOTE-12-timeseries-apis.md)). Alongside it, this
chapter fits a **lag-feature regression baseline** — an ordinary `LinearRegression` whose features are
simply the previous 12 months (`lag_1` … `lag_12`):

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit

N_LAGS = 12
N_SPLITS = 5
TEST_SIZE = 12  # each backtest fold forecasts one year ahead


def make_lag_features(series: pd.Series, n_lags: int = N_LAGS) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.DataFrame({"y": series})
    for lag in range(1, n_lags + 1):
        frame[f"lag_{lag}"] = series.shift(lag)
    frame = frame.dropna()
    X = frame[[f"lag_{lag}" for lag in range(1, n_lags + 1)]]
    return X, frame["y"]


def backtest_lag_regression(X: pd.DataFrame, y: pd.Series) -> dict:
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)
    fold_rmse = []
    for train_idx, test_idx in tscv.split(X):
        model = LinearRegression().fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        fold_rmse.append(root_mean_squared_error(y.iloc[test_idx].values, pred))
    return {"fold_rmse": np.array(fold_rmse), "mean_rmse": float(np.mean(fold_rmse))}
```

Both models are backtested with the *same* `TimeSeriesSplit(n_splits=5, test_size=12)` from Section
1.3 — five expanding-window folds, each forecasting one year ahead from everything seen so far. No
shuffling anywhere in this chapter, on principle.

### 4.2 A gotcha worth knowing about before you hit it: ARIMA's `trend` parameter

The first attempt at signal 2 — `ARIMA(train, order=(1, 1, 0), trend="c")` — raises, on the installed
statsmodels 0.15.0:

```text
ValueError: In models with integration (`d > 0`) or seasonal integration (`D > 0`), trend terms
of lower order than `d + D` cannot be (as they would be eliminated due to the differencing
operation). For example, a constant cannot be included in an ARIMA(1, 1, 1) model, but including
a linear trend, which would have the same effect as fitting a constant to the differenced data,
is allowed.
```

This is statsmodels enforcing real algebra, not being fussy: differencing an equation `d` times
algebraically eliminates any deterministic polynomial term of degree lower than `d`, so **the `trend`
parameter's lowest included degree must be `>= d + D`**. Concretely (verified against this installed
version, 2026-09-02): with `d=1`, use `trend="t"` (a pure linear-in-time term — no separate constant,
since a constant would be canceled by the first difference and is what raises the error above); with
`d=2` and a genuinely quadratic trend, the degree-0 *and* degree-1 terms both get cancelled by the
second difference, so neither a constant (`'c'`) nor `'ct'`/`'ctt'` work either — you need a trend
specified as `[0, 0, 1]` (statsmodels' polynomial-array convention, index = degree,
1 = "include") to keep **only** the degree-2 term.

Trying exactly that for signal 4's `ARIMA(order=(1, 2, 0), trend=[0, 0, 1])`, however, converges to a
quadratic coefficient near zero (`-0.0234`, versus the true curvature of roughly `+0.0017`) and
forecasts that diverge sharply from the actual data — a numerical-conditioning problem: `t²` for `t`
up to 240 spans values in the tens of thousands, which the maximum-likelihood optimiser handles badly
as an in-model regressor. **Rather than fight ARIMA's own trend-fitting for a quadratic, this chapter
uses the alternative NOTE-12 and the spec both flag: polynomial-detrend explicitly, then fit a plain
AR on the residual** (Section 4.3 shows the code). This is a real trade-off worth remembering: ARIMA's
built-in `trend` handling is reliable for `d <= 1` (a constant or linear drift) and gets fragile fast
for higher-degree deterministic trends — reach for explicit polynomial detrending instead once you're
past a linear trend.

A second, unrelated fragility showed up backtesting the two *seasonal* models (signals 1 and 3,
Section 4.3): on some folds, fitting `ARIMA(..., seasonal_order=(1, 0, 0, 12))` raised
`numpy.linalg.LinAlgError: LU decomposition error` — the seasonal AR coefficient the optimiser found
landed essentially on the unit-root boundary (unsurprising for signal 1, which has *zero* noise and is
therefore closer to a perfectly repeating sequence than to a statistical process), which crashes the
stationary-covariance initialisation used to start the Kalman filter. Passing
`enforce_stationarity=False, enforce_invertibility=False` — statsmodels' own documented escape hatch
— fixes it in every fold without changing the forecasts in any fold that had converged fine anyway.

### 4.3 Model choice and backtest results, per signal

**Signal 1 — linear trend (dominant) + sine seasonality.** Difference once (`d=1`, per Section 3.1)
to handle the trend; the small residual seasonal wiggle needs a **seasonal AR term** or it's left on
the table entirely — a plain (non-seasonal) `ARIMA(1,1,0)` backtests at a mean RMSE of **14.55**,
while adding `seasonal_order=(1,0,0,12)` drops it to **0.0001** — because signal 1 has zero noise,
capturing the seasonal structure captures essentially the whole signal:

```python
from statsmodels.tsa.arima.model import ARIMA


def backtest_arima(series: pd.Series, order, seasonal_order, trend) -> dict:
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)
    fold_rmse = []
    for train_idx, test_idx in tscv.split(series):
        train, test = series.iloc[train_idx], series.iloc[test_idx]
        model = ARIMA(train, order=order, seasonal_order=seasonal_order, trend=trend,
                       enforce_stationarity=False, enforce_invertibility=False)
        forecast = model.fit().forecast(steps=len(test_idx))
        fold_rmse.append(root_mean_squared_error(test.values, forecast.values))
    return {"fold_rmse": np.array(fold_rmse), "mean_rmse": float(np.mean(fold_rmse))}


arima_1 = backtest_arima(signals["signal_1_linear_sine"],
                          order=(1, 1, 0), seasonal_order=(1, 0, 0, 12), trend="t")
```

The lag-feature regression baseline does even better — mean RMSE **0.0000** (machine precision) —
because with a perfectly deterministic, exactly-12-month-periodic signal, `y_t - y_{t-12}` is a
*constant* (the trend's 12-month step), so a linear regression on `lag_12` alone reconstructs the
signal exactly, no statistics required:

![Forecast-vs-actual for the last backtest fold of signal_1_linear_sine: actual (black), seasonal ARIMA forecast (red dashed), and lag-regression forecast (blue dashed) are visually indistinguishable, all three tracking the same smooth rising-and-rippling line](artefacts/signal_1_linear_sine_backtest.png)

**Signal 2 — linear trend (dominant) + white noise.** Difference once (`d=1`); no seasonal
structure to add. `ARIMA(1,1,0)` with `trend="t"` backtests at mean RMSE **14.12**
(per-fold: `[10.61, 8.56, 12.80, 10.10, 28.56]` — note the last fold's spike, a reminder that
backtest folds have real variance of their own); the 12-lag regression baseline does modestly better
at **10.40**:

![Forecast-vs-actual for the last backtest fold of signal_2_linear_noise: actual values (black) scatter noisily around a rising trend; both ARIMA (red dashed) and lag-regression (blue dashed) forecasts track the general upward direction but miss the noise's specific ups and downs](artefacts/signal_2_linear_noise_backtest.png)

**Signal 3 — sine seasonality (dominant) + white noise, already stationary.** No differencing
needed (Section 3.1's ADF already rejects the unit-root null). `ARIMA(1,0,0)` with
`seasonal_order=(1,0,0,12)` — chosen directly from the PACF's lag-1-and-lag-12 spikes in Section 3.2
— backtests at mean RMSE **12.44**; the lag-regression baseline again edges it out at **10.29**:

![Forecast-vs-actual for the last backtest fold of signal_3_sine_noise: actual values (black) oscillate up and down following the seasonal cycle; both forecasts (red and blue dashed) track the oscillation's shape reasonably but lag the exact turning points](artefacts/signal_3_sine_noise_backtest.png)

**Signal 4 — quadratic trend (dominant) + white noise.** As Section 4.2 explained, this signal uses
explicit polynomial detrending instead of ARIMA's own `d=2` integration:

```python
def backtest_polynomial_detrend_ar(series: pd.Series, poly_degree: int, ar_order: int) -> dict:
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)
    t_all = np.arange(len(series))
    fold_rmse = []
    for train_idx, test_idx in tscv.split(series):
        t_train, t_test = t_all[train_idx], t_all[test_idx]
        y_train = series.iloc[train_idx]

        coeffs = np.polyfit(t_train, y_train.values, deg=poly_degree)  # fit on TRAIN prefix only
        trend_train = np.polyval(coeffs, t_train)
        trend_test = np.polyval(coeffs, t_test)

        residual_train = pd.Series(y_train.values - trend_train, index=y_train.index)
        fit = ARIMA(residual_train, order=(ar_order, 0, 0), trend="c",
                     enforce_stationarity=False, enforce_invertibility=False).fit()
        forecast = trend_test + fit.forecast(steps=len(test_idx)).values

        actual = series.iloc[test_idx].values
        fold_rmse.append(root_mean_squared_error(actual, forecast))
    return {"fold_rmse": np.array(fold_rmse), "mean_rmse": float(np.mean(fold_rmse))}
```

`polyfit(deg=2)` detrend + `AR(1)` on the residual backtests at mean RMSE **10.30** — close to the
injected noise's own standard deviation (10), essentially the best achievable — beating the lag-12
regression's **11.13**:

![Forecast-vs-actual for the last backtest fold of signal_4_quadratic_noise: actual values (black) swing noisily but trend upward with visible acceleration; the polynomial-detrend-plus-AR forecast (red dashed) tracks the accelerating curve smoothly; lag-regression (blue dashed) tracks less smoothly](artefacts/signal_4_quadratic_noise_backtest.png)

## 5. Per-signal recommendation table

Four signals, four diagnostic readings from Sections 2–3, four different right answers. Before the
table, here's the same logic as a chooser — walk any new signal through it and land on the same
family of model this chapter picked:

```mermaid
flowchart TD
    START["which signal am I looking at?"] --> Q1{"stationary already?<br/>(ADF p &lt; 0.05, Section 3.1)"}
    Q1 -->|"yes -- signal 3"| Q2{"repeating cycle in<br/>ACF/PACF? (e.g. spike at lag 12)"}
    Q1 -->|"no"| Q3{"trend shape?"}
    Q2 -->|"yes"| SEASONAL_AR["seasonal AR/ARIMA<br/>(seasonal_order=(...,12))<br/>or lag regression incl. lag_12"]
    Q3 -->|"linear"| Q4{"seasonal wiggle too?"}
    Q3 -->|"quadratic (accelerating)"| POLY["polynomial-detrend (deg=2)<br/>+ AR on the residual"]
    Q4 -->|"yes -- signal 1"| SEASONAL_D["difference d=1<br/>+ seasonal AR term"]
    Q4 -->|"no -- signal 2"| PLAIN_D["difference d=1<br/>+ plain low-order AR"]
```

| Signal | Shape | Stationary (raw)? | Differencing | Best model (backtest RMSE) | Scaling / detrending recommendation |
|---|---|---|---|---|---|
| `signal_1_linear_sine` | linear trend (dominant) + sine (10% A) | No | `d=1` | lag-feature regression (12 lags), RMSE ≈ 0.000 (seasonal ARIMA ties it at 0.0001) | Difference once to remove the trend; the residual seasonal wiggle needs a **seasonal AR term** (lag 12) or a lag-feature regression that includes `lag_12` — plain non-seasonal ARIMA leaves ~14.5 RMSE on the table. |
| `signal_2_linear_noise` | linear trend (dominant) + white noise (10% A) | No | `d=1` | lag-feature regression (12 lags), RMSE = 10.40 | Difference once; the differenced series is close to white noise plus drift — a low AR order is enough, don't over-fit AR lags to pure noise, and there's no seasonal term to add. |
| `signal_3_sine_noise` | sine (dominant) + white noise (10% A) | **Yes** | none | lag-feature regression (12 lags), RMSE = 10.29 | Already stationary — skip differencing. A non-seasonal AR alone can't see a 12-month cycle (backtests at 34.9 RMSE, see run log); add a **seasonal AR term** at lag 12, found directly from the PACF spike (Section 3.2). |
| `signal_4_quadratic_noise` | quadratic trend (dominant) + white noise (10% A) | No | `d=2` (or polynomial detrend) | `polyfit(deg=2)` detrend + AR(1) on residual, RMSE = 10.30 | One difference still leaves a (linear) trend — either difference **twice**, or polynomial-detrend by regressing on `t` and `t²` and model the residual with AR. Fitting the quadratic term directly inside ARIMA's own MLE is numerically fragile in practice (Section 4.2) — explicit `polyfit` + AR-on-residual is the robust choice. |

(Full precision in
[`artefacts/forecasting_recommendation_table.csv`](artefacts/forecasting_recommendation_table.csv)
and [`artefacts/forecasting_backtest_rmse.csv`](artefacts/forecasting_backtest_rmse.csv).)

The pattern across all four: the **lag-feature regression baseline is competitive with, and often
beats, ARIMA** on these signals, and it's simpler to reason about — but it wins specifically because
each signal's structure repeats with the same 12-month period the lags are built from. Ask it to
forecast a signal whose cycle length isn't a clean multiple of the lags you engineered, and ARIMA's
explicit `seasonal_order=(...,s)` parameter — which fits the period, rather than assuming you
guessed the right lags — becomes the more robust choice.

## 6. A neural take — an LSTM on the same signal

Every model in Sections 4–5 came from statistics, not deep learning: ARIMA, a lag-feature
`LinearRegression`, a `polyfit` detrend. That's deliberate, not an oversight — but it leaves an
obvious question sitting on the table for a reader who already knows PyTorch or TensorFlow exist:
where does a *neural* sequence model fit into any of this? This section answers it directly, on
the same data, so the comparison to Sections 4–5's numbers is apples-to-apples rather than a
separate demo on a separate dataset.

One sentence for dinner: **an LSTM turns forecasting into "read the window, remember what
mattered, predict the next value" — and on a signal this small and this clean, that extra
machinery mostly buys you a longer training run for a number the 12-lag regression already had.**

```mermaid
flowchart LR
    S["raw series<br/>(240 months)"] --> W["sliding window<br/>(12 months: t-12 .. t-1)"]
    W --> L["LSTM cell<br/>(hidden state carried<br/>across all 12 steps)"]
    L --> P["prediction<br/>(month t)"]
    P -.->|"slide the window<br/>forward one month"| W
```

### 6.1 Forecasting as supervised learning on a sliding window

Section 4.1's lag-feature regression already did the hard conceptual work here: `lag_1` …
`lag_12` are just "the 12 months before this one," laid out as 12 separate columns so
`LinearRegression` can take a flat weighted sum of them. An LSTM wants the *same* 12 numbers, but
kept in order as a **sequence** instead of flattened into columns — because unlike linear
regression, an LSTM's whole point is to notice that month `t-1` came right after month `t-2`, not
just that both happen to be inputs.

If you've ever hand-rolled a streaming aggregator over a bounded window — a ring buffer, or an
`ArrayDeque<Double>` you push the newest reading onto and evict the oldest from — this is the same
shape: push the newest month in, evict the oldest, and whatever's left in the buffer is one
training example's input. `make_windows` below does exactly that with NumPy slicing instead of a
manual deque, and it's built from the same trend-plus-noise signal Sections 2–4 already analysed —
`signal_2_linear_noise` — reproduced bit-for-bit from the same seed so nothing new is being
compared against:

```python
import numpy as np

RNG_SEED = 42
N = 240
A = 100.0
WIGGLE = 0.10 * A
WINDOW = 12  # 12 past months predict the 13th


def make_signal_2_linear_noise(n: int = N, amplitude: float = A, wiggle: float = WIGGLE,
                                seed: int = RNG_SEED) -> np.ndarray:
    """Reproduces forecasting_signals.py's signal_2_linear_noise. noise_2 is the FIRST array
    drawn from a freshly-seeded rng in both scripts, so the two arrays are bit-for-bit
    identical -- verified by running both and diffing the output."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    linear_trend = 50.0 + amplitude * (t / (n - 1))  # 50 -> 150 over 20 years
    noise_2 = rng.normal(0.0, wiggle, n)
    return linear_trend + noise_2


def make_windows(series: np.ndarray, window: int = WINDOW) -> tuple[np.ndarray, np.ndarray]:
    """Slide a length-`window` buffer over the series: X[i] = the `window` months strictly
    before target i, y[i] = the value AT target i. Reshaped to (n_windows, window, 1) -- the
    (batch, seq_len, input_size) shape torch.nn.LSTM's batch_first=True expects."""
    X, y = [], []
    for i in range(window, len(series)):
        X.append(series[i - window:i])
        y.append(series[i])
    X = np.asarray(X, dtype=np.float32).reshape(-1, window, 1)
    y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
    return X, y
```

The split stays exactly as disciplined as every other split in this chapter: windows are cut by
time, not shuffled, and the last 48 months — the same 20% Section 1.2 held out — become the test
set, with everything before it as train. The scaler (mean/std used to normalise the LSTM's inputs)
is fit on the train prefix only, per Section 7.2's rule below (this chapter's own pitfall,
applied here too): a real forecaster standing at month
192 could not have known the held-out months' mean or spread either, so neither can this model at
training time.

### 6.2 The LSTM cell: a tiny state machine

A plain RNN cell has one problem that gets worse the longer the sequence runs: everything it
"remembers" has to be squeezed through the same repeated multiplication, and gradients flowing
back through many repeated multiplications by small numbers shrink toward zero — the **vanishing
gradient problem**, the same failure mode NOTE-ML-2 documents for sigmoid/tanh stacked across many
layers ([source: NOTE-ML-2-nn-theory](../../research/NOTE-ML-2-nn-theory.md), checked 2026-09-02).
An **LSTM** (Long Short-Term Memory) fixes this the way a Java engineer would recognise
immediately if the problem were phrased as "how do I carry state across many stream elements
without it decaying": give the cell a *second*, mostly-additive channel — the **cell state** — and
three learned **gates** that decide, at every step, how much of the old cell state to keep, how
much of the new input to write in, and how much of the (possibly long-ago) memory to expose as
output right now
([source: NOTE-ML-3-architectures](../../research/NOTE-ML-3-architectures.md), checked
2026-09-02).

Think of the cell as a small hand-rolled state machine: a class with two private fields —
`hiddenState` and `cellState` — and one method, `step(input)`, called once per element of the
stream. Each call reads the current input plus both fields, and three learned gates (each just a
small neural layer, sigmoid-activated so their output is a "how much" between 0 and 1) decide how
much of the old cell state survives, how much of the new input gets written in, and how much of
the result gets exposed as this step's output — then it returns the updated `(hiddenState,
cellState)` pair. Run `step` 12 times, once per month in the window, and the final `hiddenState` is
the network's entire summary of everything it saw across all 12 months — that single vector is
what gets handed to the prediction head.

Written out as equations — verified directly against the installed `torch.nn.LSTM`'s own
docstring (torch 2.14.0, checked 2026-09-03; this exact gate formulation is not one of NOTE-ML-3's
prose-level facts, so, per this chapter's own convention for `KNeighborsRegressor` in Section 1.2,
it was confirmed against the live installed library rather than assumed):

$$
\begin{aligned}
i_t &= \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
f_t &= \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
g_t &= \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
o_t &= \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
c_t &= f_t \odot c_{t-1} + i_t \odot g_t \\
h_t &= o_t \odot \tanh(c_t)
\end{aligned}
$$

In plain language: $i_t$ (the **input gate**) decides how much of this month's value to write into
memory; $f_t$ (the **forget gate**) decides how much of last month's memory to keep; $o_t$ (the
**output gate**) decides how much of memory to expose right now; $g_t$ is "the candidate new
information," and $\odot$ is elementwise multiplication — "how much," not "which direction." The
cell state update $c_t = f_t \odot c_{t-1} + i_t \odot g_t$ is *additive* in $c_{t-1}$ rather than
another repeated matrix multiplication, which is precisely what keeps gradients from vanishing
across a 12-step window the way a plain RNN's would.

`torch.nn.LSTM`'s constructor signature — `LSTM(input_size, hidden_size, num_layers=1, bias=True,
batch_first=False, dropout=0.0, bidirectional=False, proj_size=0, ...)`, also read directly from
the installed 2.14.0 docstring — implements exactly this per layer, per time step, so building the
model is three lines: one `nn.LSTM` layer, and a `nn.Linear` head on the final hidden state,
`h_n[-1]`, to turn the 32-number summary back into one predicted value:

```python
import torch
from torch import nn

HIDDEN_SIZE = 32


class TinyLSTM(nn.Module):
    """One LSTM layer over the 12-month window, then a linear head on the final hidden state.
    `forward` returns `(output, (h_n, c_n))`; `h_n[-1]` is the last layer's hidden state after
    the last time step -- everything the network "remembers" about the whole 12-month window,
    compressed into one 32-number vector."""

    def __init__(self, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=1,
                             batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1])
```

For a Java engineer, `nn.Module` is the closest thing PyTorch has to an interface contract: extend
it, declare the layers you own as fields in `__init__` (PyTorch finds their trainable parameters by
walking those fields automatically — no manual registration, similar to how a DI container
discovers `@Autowired` fields by reflection), and implement `forward` the way you'd implement a
single abstract method.

### 6.3 Training it, and reading the result honestly

Training is a plain gradient-descent loop — no framework magic beyond `loss.backward()` computing
every gradient via autograd (PyTorch's own reverse-mode autodiff, the same backpropagation-by-chain-rule
NOTE-ML-2 documents, just applied automatically instead of by hand):

```python
EPOCHS = 300
LEARNING_RATE = 0.01

torch.manual_seed(42)
model = TinyLSTM()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

loss_history = []
for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    pred = model(X_train_t)
    loss = loss_fn(pred, y_train_t)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
```

Run against the 180 training windows (months 12–191), the loss drops fast for the first ~30 epochs
and then creeps down slowly for the rest of the run:

```text
epoch    1  train MSE (standardised units) = 0.98133
epoch   50  train MSE (standardised units) = 0.16057
epoch  100  train MSE (standardised units) = 0.15357
epoch  150  train MSE (standardised units) = 0.14830
epoch  200  train MSE (standardised units) = 0.14217
epoch  250  train MSE (standardised units) = 0.13309
epoch  300  train MSE (standardised units) = 0.11568

LSTM one-step-ahead RMSE (last 48 months, single time-respecting split): 13.1803
```

![Line chart of TinyLSTM training loss (mean squared error, standardised units, log-scaled y-axis) against epoch number, for signal_2_linear_noise with a 12-month window: loss falls sharply from about 1.0 to about 0.2 in the first 20 epochs, oscillates briefly, then decreases slowly and steadily from about 0.16 to about 0.12 over the remaining 280 epochs](artefacts/lstm_forecast_loss_curve.png)

That log-scaled curve is the "show the artefact" half of this section: real numbers, falling, on
real (if standardised) units — not a hand-wave that "the network learns." The forecast itself, read
against the 48 held-out months, tells the more useful story:

![Line chart comparing actual values (black, solid, with markers) to the LSTM's one-step-ahead forecast (green, dashed) for the last 48 months of signal_2_linear_noise: the actual series swings noisily between roughly 115 and 170, while the LSTM forecast stays in a much narrower band around 128-135, tracking the general level but missing essentially all of the month-to-month swings](artefacts/lstm_forecast_vs_actual.png)

The forecast line is almost flat next to the actual data's swings — the LSTM learned the *level*
of the series (roughly where the trend sits by month 200) but did not, and could not, predict the
noise riding on top of it, because that noise is independently drawn white noise by construction
(Section 1.1) — nothing in the training data carries any information about it. That's not a bug in
the model; it is, in fact, the textbook-correct behaviour, and it's exactly what Sections 4.2–4.3's
ARIMA and lag-regression forecasts do too, for the same reason.

| Model | Mean RMSE on `signal_2_linear_noise` | Evaluation | Cost to get this number |
|---|---|---|---|
| ARIMA(1,1,0), `trend='t'` (Section 4.3) | 14.12 | 5-fold walk-forward backtest | statsmodels MLE fit per fold, well under a second total |
| 12-lag `LinearRegression` (Section 4.3) | 10.40 | 5-fold walk-forward backtest | closed-form fit, milliseconds |
| TinyLSTM, 12-month window, 32 hidden units (this section) | 13.18 | single time-respecting split | 300 full-batch epochs, a few seconds on CPU, no GPU needed |

Read that table carefully rather than just reading off "13.18 beats 14.12": the LSTM's number comes
from *one* time-respecting split, not the 5-fold walk-forward backtest the other two rows report —
the honest comparison is "roughly the same ballpark," not "beats ARIMA." The injected noise's own
standard deviation is `WIGGLE = 10.0` — the practical floor no model can beat on this signal, since
that's the size of the part nothing can predict. The 12-lag regression already sits close to that
floor. The LSTM landing a few points above it, after two hundred times the code and a training loop
Sections 4–5 didn't need, is the honest headline of this section.

### 6.4 When the extra machinery earns its place

None of this means neural sequence models are a bad idea — it means this particular signal doesn't
need one. A clean, low-dimensional, single-frequency composite signal like the four this chapter
builds is exactly the case classical time-series statistics was designed for: ARIMA and a
hand-built lag-feature table both already encode the right inductive bias (linear trend, one
12-month cycle) directly into the model, so there's very little left for a neural network to
discover on its own — and discovering it costs a training loop, a random seed, a chosen
architecture, and a GPU-or-patience decision that a `LinearRegression.fit()` call never asks for.

An LSTM (or, for longer sequences, a Transformer — Section 4 of
[Network Architectures](../../02-machine-learning/01-theory/02-architectures.md) covers both, plus
why transformers have displaced RNNs for most new sequence work) earns its complexity when the
signal stops looking like this chapter's synthetic examples: many correlated input series instead
of one, patterns that shift across regimes instead of a fixed trend and a fixed period, non-linear
interactions between lags that no `polyfit` or fixed-order AR term can express, or enough training
data that the network can actually learn structure statistics has to be told by hand. If you're
staring at one wobbling line with an obvious trend and an obvious cycle, reach for Section 4's
tools first and only bring in a neural model once a classical baseline has demonstrably run out of
room to improve — the same "back-test and compare, every time" discipline Section 5's recommendation
table already argued for two model families; it applies just as much to a third.

For the neural-network mechanics this section leaned on without re-deriving — what a gate actually
is, backpropagation, why vanishing gradients happen at all, the general architecture-selection
question of "which network shape fits which data shape" — see
[Neural Network Fundamentals](../../02-machine-learning/01-theory/01-neural-network-fundamentals.md)
and [Network Architectures](../../02-machine-learning/01-theory/02-architectures.md) in the Machine
Learning subject; this section only needed enough of that theory to explain one worked example.

---

**Environment for this section (separate from the rest of this chapter):** the code above runs in
a different virtual environment (`.venv-ml`) from `forecasting_signals.py`, because it needs
`torch` and does not need `statsmodels`:

```text
torch==2.14.0 (CPU build)
numpy==2.5.2
scikit-learn==1.9.0
matplotlib==3.11.1
Python 3.12+
```

Pinned and verified against PyPI on 2026-09-02
([source: NOTE-ML-1-torch-install](../../research/NOTE-ML-1-torch-install.md)); numpy/scikit-learn/matplotlib
versions reused from [NOTE-2-package-versions](../../research/NOTE-2-package-versions.md) and
[NOTE-5-sklearn-core-apis](../../research/NOTE-5-sklearn-core-apis.md), confirmed installed at
exactly these versions in `.venv-ml` by running `pip show` there, 2026-09-03. `signal_2_linear_noise`
is reproduced from the same `RNG_SEED=42` as `forecasting_signals.py` and verified bit-for-bit
identical across the two separate environments. Full source:
[`code/lstm_forecast.py`](code/lstm_forecast.py); run it with `.venv-ml`'s Python, not the main
chapter environment's.

## 7. Pitfalls

### 7.1 A random split on time-ordered data

Covered in full in Section 1: `train_test_split`'s default `shuffle=True`, or a plain (non-time)
`KFold`, lets the model train on rows that come after the row it's being tested on. Section 1.2's bar
chart put a number on it — 1.5x to 4.2x worse honest error than the shuffled split reported — but the
size of the gap isn't the point; the *direction* always is. **For time-ordered data, always use
`TimeSeriesSplit` (or an explicit chronological holdout) — never `train_test_split`'s default or a
shuffled `KFold`.**

### 7.2 Fitting the scaler (or imputer) on the whole series

The temporal twin of [the split chapter's Section 3 leak](04-train-valid-holdout-split.md#3-the-leakage-demo-fit-on-the-whole-dataset-get-an-optimistic-score):
fitting a `StandardScaler` on the *entire* series before splitting means its mean and standard
deviation are computed partly from months that, at forecast time, haven't happened yet.

```mermaid
flowchart TD
    subgraph WRONG["leaky (wrong)"]
        FULLSERIES["fit StandardScaler on<br/>all 240 months"] --> FULLSTATS["mean/std pulled toward<br/>months the forecaster<br/>couldn't have seen yet"]
    end
    subgraph RIGHT["proper (correct)"]
        TRAINPREFIX["fit StandardScaler on<br/>the training prefix only<br/>(e.g. months 1-228)"] --> TRAINSTATS["mean/std reflect only<br/>what was known<br/>at forecast time"]
    end
```

On `signal_2_linear_noise`, fitting on the full 240 months versus fitting on only the first 228 (this
chapter's last backtest fold's training prefix) gives visibly different numbers:

```python
from sklearn.preprocessing import StandardScaler

full_scaler = StandardScaler().fit(signals["signal_2_linear_noise"].values.reshape(-1, 1))
train_scaler = StandardScaler().fit(
    signals["signal_2_linear_noise"].values[:228].reshape(-1, 1))
print(f"fit on FULL series  -> mean={full_scaler.mean_[0]:.4f} std={full_scaler.scale_[0]:.4f}")
print(f"fit on TRAIN prefix -> mean={train_scaler.mean_[0]:.4f} std={train_scaler.scale_[0]:.4f}")
```

```text
fit on FULL series  -> mean=99.5299 std=30.1301
fit on TRAIN prefix -> mean=97.2999 std=29.2015
```

The full-series mean is pulled about 2.2 units higher, because it includes the last 12 months' higher
trend level — information a real forecaster, standing at month 228, could not have had. It's a small
gap here (the same "invisible at this scale" story the split chapter told about its own leak) but the
mechanism and the fix are identical: **fit every preprocessing step — scaler, imputer, anything with
`.fit()` — on the training prefix only, inside a `Pipeline`, exactly as the split chapter recommends.
A `Pipeline` doesn't automatically know about time order, so this discipline still has to be applied
by hand: fit on `series[:train_end]`, never on `series` as a whole.**

### 7.3 Forecasting far beyond the trend's validity

A model trained on a trend is only trustworthy for extrapolating a *short* distance past its training
data — how short depends on the trend shape. `signal_4_quadratic_noise`'s polynomial fit (Section 4.3)
tracks the next 12 months well (mean absolute error 6.58, close to the noise floor), but extend that
same fitted quadratic much further:

```python
coeffs = np.polyfit(np.arange(200), signals["signal_4_quadratic_noise"].values[:200], deg=2)
for months_out, t_future in [(0, 200), (120, 320), (240, 440)]:
    print(f"{months_out:>3} months past training end -> predicted value "
          f"{np.polyval(coeffs, t_future):.2f}")
```

```text
  0 months past training end -> predicted value 119.61
120 months past training end -> predicted value 233.30
240 months past training end -> predicted value 401.09
```

The true signal never exceeds roughly 150–160 anywhere in its 20-year history — the quadratic
extrapolated 20 years further predicts **401**, nearly 3x the highest value the model was ever
trained on. A quadratic (or any polynomial of degree ≥ 2) grows without bound, and nothing in the
fitting procedure knows or cares that the real-world process it's modelling almost certainly doesn't.
**Treat a trend model's validity as bounded to roughly the same horizon it was validated on in the
backtest (Section 4) — a model that backtests well one year out is not evidence it's safe to forecast
ten years out, especially for a non-linear trend.**

## 8. Recap & what's next

- Time-ordered rows are not independent, so [the split chapter's](04-train-valid-holdout-split.md)
  `shuffle=True` default leaks the future into training. Section 1.2 showed the honest error running
  1.5x–4.2x higher than a shuffled split reports, on every one of the four signals — before you'd
  even fit a "real" forecasting model. Use
  [`TimeSeriesSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
  for an **expanding-window walk-forward backtest** instead — every fold's training set precedes its
  test set.
- [`seasonal_decompose`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.seasonal_decompose.html)
  splits a series into trend + seasonal + residual — useful for a first look, but it will report a
  "seasonal" component even on data with none (Section 2's signal 2), so pair it with ACF/PACF before
  trusting what it found.
- **[`adfuller`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html)**
  tests stationarity (needed for AR/ARIMA); **differencing** ($d=1$ for a linear trend, $d=2$ for a
  quadratic one) restores it; **ACF/PACF** read off the right AR order and reveal a seasonal period
  directly from the data, once the series is stationary.
- **ARIMA vs. a lag-feature regression baseline**: both were backtested honestly with
  `TimeSeriesSplit`, and the recommendation table (Section 5) shows the lag-feature baseline winning
  on three of four signals here — a reminder that "the more sophisticated model" isn't automatically
  the better one; back-test and compare, every time.
- Watch for three forecasting-specific pitfalls beyond the split itself: fitting a scaler/imputer on
  the whole series (Section 7.2), and extrapolating a trend model far past where it was validated
  (Section 7.3) — a quadratic trend in particular grows without bound.
- **A neural sequence model (Section 6)** frames forecasting as supervised learning on a sliding
  window and trains a small `torch.nn.LSTM` on `signal_2_linear_noise`, landing at RMSE 13.18 — in
  the same neighbourhood as, not clearly better than, Section 4.3's ARIMA (14.12) and lag-regression
  (10.40) baselines on this signal. The honest takeaway: a clean, low-dimensional synthetic signal
  like this chapter's four is exactly the case classical time-series statistics was built for: earn
  the neural model's extra complexity with messier, higher-dimensional, or regime-shifting data
  first, not by default.

This chapter closes out the "trend, split, and validate honestly" thread that started with
[train/validation/holdout](04-train-valid-holdout-split.md) and continued through
[collinearity](03-collinearity.md) and [regression on NYC taxi fares](05-regression-nyc-taxi.md).
**Feature selection** is next in the curriculum — choosing *which* features earn a place in a model
from a large candidate set, the harder version of the pruning [the collinearity chapter](03-collinearity.md)
started.

Section 4.1's `AR(p)` equation reappears in [Bayesian Inference](14-bayesian-inference.md) (DS-19),
fit the same way in spirit but with a posterior *distribution* over $\phi$ instead of one
maximum-likelihood point — the same "how sure are we" question this chapter's point-estimate
`ARIMA` fits never asked.

---

### Environment

```text
statsmodels==0.15.0
scikit-learn==1.9.0
numpy==2.5.2
pandas==3.0.5
matplotlib==3.11.1
Python 3.12+
```

Pinned and verified against PyPI/official docs on 2026-09-02
([source: NOTE-12-timeseries-apis](../../research/NOTE-12-timeseries-apis.md) for the statsmodels
time-series APIs and sklearn `TimeSeriesSplit`; [source: NOTE-6-statsmodels-vif](../../research/NOTE-6-statsmodels-vif.md)
for the statsmodels package version, reused per this chapter's spec;
[source: NOTE-5-sklearn-core-apis](../../research/NOTE-5-sklearn-core-apis.md) for the scikit-learn
version and `root_mean_squared_error`; [source: NOTE-2-package-versions](../../research/NOTE-2-package-versions.md)
for numpy/pandas/matplotlib). This chapter's code and every artefact were generated and gated on
**Python 3.13.7**, with all five packages installed at exactly the pinned versions above — no
substitutions. Full source: [`code/forecasting_signals.py`](code/forecasting_signals.py).

### Environment note (for the architect)

Three judgment calls made while getting the code to run cleanly, beyond what NOTE-12 specified:

1. **ARIMA `trend` parameter for `d >= 1`.** NOTE-12's grounded signature lists `trend=None` as the
   default but doesn't cover the `d`-dependent constraint on which trend strings/arrays are valid.
   Section 4.2 documents the exact rule and the two failure modes hit while writing this chapter
   (verified empirically against the installed statsmodels 0.15.0, not assumed): the `ValueError` for
   an under-order trend, and the numerical-conditioning failure fitting a quadratic trend directly
   inside ARIMA's MLE (worked around with explicit polynomial detrending instead, per NOTE-12's own
   "polynomial detrend" alternative).
2. **`enforce_stationarity=False, enforce_invertibility=False`** was added to every `ARIMA(...)` call
   in the backtest loop after a `LinAlgError` surfaced on specific folds of the two seasonal models
   (signals 1 and 3) — a known statsmodels failure mode when an MLE coefficient estimate lands on the
   unit-root boundary, most likely here because signal 1 is fully deterministic. This is statsmodels'
   own documented escape hatch, not a suppression of a real problem; forecasts on folds that had
   converged fine before this change did not move.
3. **The leakage demo (Section 1.2) uses `KNeighborsRegressor` on the time index, not the lag-feature
   regression.** An earlier version compared shuffled vs. walk-forward `TimeSeriesSplit` backtests
   using the same `LinearRegression`-on-lags baseline from Section 4, and the gap was real but small
   and occasionally reversed in direction (a linear model extrapolates a linear-in-lag relationship
   about as well whether or not it's seen "future" rows, since the recurrence relation itself doesn't
   change with time — the leak's advantage there comes from interpolation vs. extrapolation
   specifically, which a non-extrapolating model like k-NN exposes far more directly). k-NN on the raw
   time index was chosen deliberately to make Section 1.2's point unambiguous across all four signals;
   Section 4's later, more nuanced ARIMA/lag-regression comparison stands on its own as the
   "which model fits best" analysis and does not depend on Section 1's leak numbers.

**Restyle note (2026-09-03):** this pass reweaves the chapter into the book's storytelling/heavy-visual
house style (cold open citing Box & Jenkins 1970, problem-first ordering that shows the shuffle-vs-
walk-forward numbers before defining `TimeSeriesSplit`, plain-language glosses before notation, LaTeX
for autocorrelation/AR(p)/stationarity, and seven Mermaid diagrams). Every `python` code block, every
artefact reference, every number, and every grounding citation from the prior version is preserved
verbatim; only prose structure, headings, and section order changed. New claim added: the Box-Jenkins
1970 publication date and book title, grounded live against
[Wikipedia, "Box–Jenkins method"](https://en.wikipedia.org/wiki/Box%E2%80%93Jenkins_method) (checked
2026-09-03) — not one of NOTE-12's tabulated facts, since it's historical framing rather than an API
or metric definition, but cited inline per the style guide's "inline authoritative citation" allowance.

**Enrichment note (2026-09-03): Section 6, "A neural take — an LSTM on the same signal," added.**
This chapter's original scope (SPEC-DS-9) explicitly listed "Prophet/deep forecasting (mention +
link)" under Out of scope. This addition was requested directly by the owner as a deliberate,
separately-scoped enrichment — the chapter had no deep-learning angle at all, and the owner wanted
one added on the same data rather than as a new chapter. Flagging here because it goes beyond
SPEC-DS-9's written scope; **the architect should update SPEC-DS-9's Out of scope line** (or record
this as an approved scope amendment) so the spec and the shipped chapter stay consistent per the
Definition of Done's "anything cut from the spec is recorded in the spec's Out of scope, not
silently dropped" rule — the same rule in reverse, since this is something *added* past what was
scoped out.

Three judgment calls made writing Section 6, beyond what NOTE-ML-1/-2/-3 covered directly:

1. **Separate virtual environment.** `.venv-ml` (which has `torch==2.14.0` but not `statsmodels`)
   is not the same environment `forecasting_signals.py` runs in. Rather than add a ~250 MB CPU
   torch wheel to the rest of this chapter's dependency set, Section 6's code lives in its own file
   (`code/lstm_forecast.py`) with its own Environment block, and reproduces `signal_2_linear_noise`
   from scratch (same seed, verified bit-for-bit identical) instead of importing
   `forecasting_signals.py`, since that module's own imports (`statsmodels`) are not installed in
   `.venv-ml`.
2. **`torch.nn.LSTM`'s gate equations are not one of NOTE-ML-3's tabulated facts** (NOTE-ML-3 covers
   LSTM/GRU gating at the block-diagram level, explicitly flagging "full gate equations … technical
   detail; block-level intuition sufficient" as out of its own scope). The exact equations in
   Section 6.2 were instead read directly from the installed `torch==2.14.0` docstring
   (`nn.LSTM.__doc__`) and verified to match — the same "confirm against the live environment"
   convention this chapter already used for `KNeighborsRegressor`'s signature in Section 1.2.
3. **Signal choice.** `signal_2_linear_noise` (linear trend + real white noise) was chosen over
   `signal_1_linear_sine` (which Section 4.3 already fits to near-zero RMSE, because it has zero
   injected noise) specifically so the LSTM-vs-classical comparison has a real noise floor to reason
   about, rather than comparing against an edge case where classical models happen to reconstruct
   the signal exactly.

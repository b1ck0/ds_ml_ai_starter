"""A small LSTM forecaster on ONE of the chapter's synthetic signals (signal_2_linear_noise:
linear trend + white noise). Companion code for the "A neural take -- an LSTM on the same
signal" section of forecasting-composite-signals.md.

This script is DELIBERATELY self-contained and runs in a SEPARATE virtual environment
(.venv-ml, which has torch but not statsmodels) from forecasting_signals.py. The signal-
generation code below is copied from forecasting_signals.py's make_signals(): same formula,
same RNG_SEED=42, and noise_2 is drawn as the FIRST call from a freshly-seeded rng in both
scripts, so signal_2_linear_noise here is numerically IDENTICAL (bit-for-bit, verified) to the
one the rest of the chapter already decomposed, stationarity-tested, and backtested with
ARIMA and a lag-feature regression baseline (Sections 2-4). This script fits a THIRD model on
the exact same numbers -- it does not introduce a new dataset.

What it does:
  1. Rebuilds signal_2_linear_noise (240 months).
  2. Frames forecasting as supervised learning on sliding windows: X = the previous WINDOW=12
     months, y = the next month -- the same shape as forecasting_signals.py's make_lag_features
     (lag_1..lag_12 columns), just kept as a (window, 1) sequence instead of being flattened
     into separate columns, because that is the input shape torch.nn.LSTM expects.
  3. Splits by TIME, not randomly: the last 48 months (the same 20% held out in Section 1.2's
     shuffle-vs-walk-forward demo) become the test set; everything before that is train. No
     window's target in train comes after any window's target in test.
  4. Fits a StandardScaler-style (mean/std) normalisation on the TRAIN prefix only -- Section
     6.2's discipline: a real forecaster standing at month 192 could not have known the test
     months' mean or spread either.
  5. Trains a 1-layer, 32-hidden-unit LSTM (torch.nn.LSTM) for 300 full-batch epochs with Adam,
     records the loss curve, then forecasts the 48 held-out months one-step-ahead (each
     prediction uses the TRUE previous 12 months as input, the same convention
     backtest_lag_regression uses -- not a recursive multi-step forecast).
  6. Saves 2 artefacts to ../artefacts/: the training loss curve and a forecast-vs-actual plot,
     and prints the LSTM's RMSE next to Section 4.3's already-computed ARIMA and lag-regression
     numbers for the same signal.

Environment (verified against the installed .venv-ml environment, checked 2026-09-03):
    torch==2.14.0 (CPU build; research/NOTE-ML-1-torch-install.md pins this exact version)
    numpy==2.5.2, scikit-learn==1.9.0, matplotlib==3.11.1 (research/NOTE-2-package-versions.md,
    research/NOTE-5-sklearn-core-apis.md -- reused; these match forecasting_signals.py's own
    pinned versions, confirmed by running `pip show` in .venv-ml)
    Python 3.12+ (this script was run and gated on Python 3.13.7, matching the pinned versions).
    torch.nn.LSTM's constructor signature (input_size, hidden_size, num_layers=1, bias=True,
    batch_first=False, dropout=0.0, bidirectional=False, proj_size=0, ...) and its gate
    equations were read directly from the installed torch 2.14.0 docstring (`nn.LSTM.__doc__`),
    the same "verify against the live environment" convention forecasting_signals.py used for
    KNeighborsRegressor's signature.

Run (from .venv-ml, which is NOT the same venv as forecasting_signals.py):
    python lstm_forecast.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this script only saves figures, never shows them
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import root_mean_squared_error
from torch import nn

RNG_SEED = 42          # same seed forecasting_signals.py uses to build all four signals
TORCH_SEED = 42        # separate seed for the model's own random weight init
N = 240                # 20 years of monthly data -- must match forecasting_signals.py
A = 100.0               # amplitude/scale of the dominant (linear trend) component
WIGGLE = 0.10 * A        # 10.0 -- std of the injected white noise (the practical noise floor)
WINDOW = 12              # sliding window: 12 past months predict the 13th
TEST_FRAC = 0.2          # same held-out fraction as Section 1.2's shuffle-vs-walk-forward demo
HIDDEN_SIZE = 32         # LSTM hidden units -- deliberately tiny
EPOCHS = 300
LEARNING_RATE = 0.01

ARTEFACTS_DIR = Path(__file__).resolve().parent.parent / "artefacts"

# Section 4.3's already-computed, already-grounded backtest numbers for THIS SAME signal, for
# the honest side-by-side comparison this script prints (not recomputed here -- ARIMA needs
# statsmodels, which .venv-ml does not have; these are the chapter's own numbers, unchanged).
ARIMA_MEAN_RMSE = 14.12
LAG_REGRESSION_MEAN_RMSE = 10.40


def make_signal_2_linear_noise(n: int = N, amplitude: float = A, wiggle: float = WIGGLE,
                                seed: int = RNG_SEED) -> np.ndarray:
    """Reproduces forecasting_signals.py's signal_2_linear_noise. noise_2 is the FIRST array
    drawn from a freshly-seeded rng in both scripts (the deterministic trend/index computation
    before it consumes no randomness), so the two arrays are bit-for-bit identical -- verified
    by running both and diffing the output."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    linear_trend = 50.0 + amplitude * (t / (n - 1))  # 50 -> 150 over 20 years
    noise_2 = rng.normal(0.0, wiggle, n)
    return linear_trend + noise_2


def make_windows(series: np.ndarray, window: int = WINDOW) -> tuple[np.ndarray, np.ndarray]:
    """Slide a length-`window` buffer over the series: X[i] = the `window` months strictly
    before target i, y[i] = the value AT target i. Same information content as
    forecasting_signals.py's make_lag_features (lag_1..lag_12), reshaped to (n_windows, window,
    1) -- the (batch, seq_len, input_size) shape torch.nn.LSTM's batch_first=True expects."""
    X, y = [], []
    for i in range(window, len(series)):
        X.append(series[i - window:i])
        y.append(series[i])
    X = np.asarray(X, dtype=np.float32).reshape(-1, window, 1)
    y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
    return X, y


class TinyLSTM(nn.Module):
    """One LSTM layer over the 12-month window, then a linear head on the final hidden state.
    torch.nn.LSTM(input_size=1, hidden_size=32, num_layers=1, batch_first=True) -- signature
    and gate equations verified against the installed torch 2.14.0 docstring, checked
    2026-09-03. `forward` returns `(output, (h_n, c_n))`; `h_n[-1]` is the last layer's hidden
    state after the last time step -- everything the network "remembers" about the whole
    12-month window, compressed into one 32-number vector."""

    def __init__(self, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=1,
                             batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1])


def main() -> None:
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(TORCH_SEED)

    series = make_signal_2_linear_noise()
    X, y = make_windows(series)
    n_windows = len(X)
    n_test = int(round(N * TEST_FRAC))          # 48 months, same as Section 1.2
    split = n_windows - n_test

    X_train_raw, X_test_raw = X[:split], X[split:]
    y_train_raw, y_test_raw = y[:split], y[split:]

    # Fit the scaler on the TRAIN prefix only (Section 6.2's discipline, applied here too).
    mu, sigma = float(y_train_raw.mean()), float(y_train_raw.std())
    X_train = (X_train_raw - mu) / sigma
    X_test = (X_test_raw - mu) / sigma
    y_train = (y_train_raw - mu) / sigma

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_test_t = torch.from_numpy(X_test)

    model = TinyLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    loss_history: list[float] = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = loss_fn(pred, y_train_t)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if epoch == 1 or epoch % 50 == 0:
            print(f"epoch {epoch:4d}  train MSE (standardised units) = {loss.item():.5f}")

    model.eval()
    with torch.no_grad():
        pred_test_std = model(X_test_t).numpy()
    pred_test = pred_test_std * sigma + mu  # undo standardisation -> back to real units

    lstm_rmse = root_mean_squared_error(y_test_raw.ravel(), pred_test.ravel())
    print(f"\nLSTM one-step-ahead RMSE (last {n_test} months, single time-respecting split): "
          f"{lstm_rmse:.4f}")
    print(f"Section 4.3's backtested numbers on this SAME signal (5-fold walk-forward mean): "
          f"ARIMA(1,1,0) trend='t' = {ARIMA_MEAN_RMSE:.2f}, "
          f"12-lag regression = {LAG_REGRESSION_MEAN_RMSE:.2f} "
          f"(injected noise std = {WIGGLE:.1f} -- the practical noise floor).")

    # --- artefact 1: training loss curve ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, EPOCHS + 1), loss_history, color="#4C72B0")
    ax.set_xlabel("epoch")
    ax.set_ylabel("train MSE (standardised units, log scale)")
    ax.set_yscale("log")
    ax.set_title("TinyLSTM training loss -- signal_2_linear_noise, 12-month window")
    fig.tight_layout()
    loss_path = ARTEFACTS_DIR / "lstm_forecast_loss_curve.png"
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)

    # --- artefact 2: forecast vs actual, held-out last 48 months ---------------------------
    test_month_idx = np.arange(N - n_test, N)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(test_month_idx, y_test_raw.ravel(), label="actual", color="black", marker="o",
            markersize=3, linewidth=1.4)
    ax.plot(test_month_idx, pred_test.ravel(), label=f"LSTM forecast (RMSE={lstm_rmse:.2f})",
            color="#55A868", linestyle="--")
    ax.set_xlabel("month index (0-239)")
    ax.set_ylabel("value")
    ax.set_title("LSTM one-step-ahead forecast vs. actual -- signal_2_linear_noise, "
                 "held-out last 48 months")
    ax.legend()
    fig.tight_layout()
    forecast_path = ARTEFACTS_DIR / "lstm_forecast_vs_actual.png"
    fig.savefig(forecast_path, dpi=150)
    plt.close(fig)

    print(f"\nwrote: {loss_path.name}, {forecast_path.name}")


if __name__ == "__main__":
    main()

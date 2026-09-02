"""Bias-variance and overfitting illustrations -- companion code for
Data Science/Theory/theory-overview.md (SPEC-DS-14).

What it does:
  1. Builds a small synthetic 1-D regression problem with a KNOWN true function (a sine
     wave), so "bias" and "variance" can be measured against ground truth instead of
     eyeballed.
  2. Bias-variance illustration: draws many bootstrap-resampled training sets from the
     same population, fits a LOW-complexity model (a degree-1 line, via
     PolynomialFeatures + LinearRegression) and a HIGH-complexity model (a degree-15
     polynomial) to each resample, and plots every fitted curve next to the true
     function. The low-complexity curves cluster tightly but consistently miss the true
     shape (high bias, low variance); the high-complexity curves each hug their own
     training sample but disagree wildly with each other (low bias on average, high
     variance). Saves bias_variance_illustration.png.
  3. Overfitting curve: sweeps polynomial degree 1..15 on ONE train/validation split and
     plots train RMSE vs validation RMSE against degree -- training error falls
     monotonically while validation error falls then rises, the classic U-shaped
     generalisation curve that visually diagnoses under- vs over-fitting
     (research/NOTE-14-ds-theory-definitions.md: "visual inspection... learning curves
     plotting train/test error vs. model complexity help diagnose whether a model is
     under- or over-fitting"). Saves overfitting_curve.png.

Grounded APIs:
  - Pipeline, train_test_split, root_mean_squared_error -- signatures tabulated in
    research/NOTE-5-sklearn-core-apis.md (checked 2026-09-02).
  - PolynomialFeatures(degree=2, *, interaction_only=False, include_bias=True, order='C')
    is NOT one of NOTE-5's tabulated APIs. Its signature above was verified directly
    against this project's installed scikit-learn 1.9.0 via
    `inspect.signature(PolynomialFeatures.__init__)`, and cross-checked against the
    official docs:
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
    (checked 2026-09-02) -- same "verify directly against the installed environment"
    fallback used for un-tabulated APIs elsewhere in this repo (see the environment note
    in Data Science/Worked Examples/train-valid-holdout-split.md).

Environment (research/NOTE-2-package-versions.md, research/NOTE-5-sklearn-core-apis.md,
checked 2026-09-02):
    numpy==2.5.2, matplotlib==3.11.1, scikit-learn==1.9.0, Python 3.11+.

Run:
    python bias_variance_overfitting.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this script only saves figures, never shows them
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

ARTEFACTS_DIR = Path(__file__).resolve().parent.parent / "artefacts"
RNG_SEED = 42


def true_function(x: np.ndarray) -> np.ndarray:
    """The ground-truth signal the models are trying to recover -- unknown to the
    models, known to us so we can measure bias and variance directly."""
    return np.sin(1.5 * np.pi * x)


def make_dataset(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Draw n points x ~ Uniform(0, 1), y = true_function(x) + Gaussian noise."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 1, size=n))
    y = true_function(x) + rng.normal(scale=0.3, size=n)
    return x, y


def make_poly_pipeline(degree: int) -> Pipeline:
    """A Pipeline chaining PolynomialFeatures -> LinearRegression: one fit/predict
    contract, same pattern as every other worked-example chapter's Pipeline usage."""
    return Pipeline(
        [
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("linreg", LinearRegression()),
        ]
    )


def plot_bias_variance(path: Path) -> None:
    """Bootstrap-resampling illustration of bias vs variance."""
    rng = np.random.default_rng(RNG_SEED)
    x_base, y_base = make_dataset(n=30, seed=RNG_SEED)
    x_grid = np.linspace(0, 1, 200).reshape(-1, 1)
    y_true_grid = true_function(x_grid.ravel())

    n_bootstraps = 25
    settings = [("low complexity (degree=1)", 1), ("high complexity (degree=15)", 15)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, (title, degree) in zip(axes, settings):
        preds = np.zeros((n_bootstraps, x_grid.shape[0]))
        for b in range(n_bootstraps):
            idx = rng.integers(0, len(x_base), size=len(x_base))  # bootstrap resample
            x_b, y_b = x_base[idx], y_base[idx]
            model = make_poly_pipeline(degree)
            model.fit(x_b.reshape(-1, 1), y_b)
            preds[b] = model.predict(x_grid)
            ax.plot(x_grid, preds[b], color="tab:blue", alpha=0.15, linewidth=1)

        mean_pred = preds.mean(axis=0)
        bias_sq = float(np.mean((mean_pred - y_true_grid) ** 2))
        variance = float(np.mean(preds.var(axis=0)))

        ax.plot(x_grid, y_true_grid, color="black", linewidth=2, label="true function")
        ax.plot(
            x_grid, mean_pred, color="tab:red", linewidth=2, linestyle="--",
            label="average fit (across resamples)",
        )
        ax.scatter(x_base, y_base, color="dimgray", s=14, zorder=3, label="one training sample")
        ax.set_title(f"{title}\nbias²={bias_sq:.3f}   variance={variance:.3f}")
        ax.set_xlabel("x")
        ax.legend(loc="upper right", fontsize=7.5)
    axes[0].set_ylabel("y")
    fig.suptitle(f"Bias vs variance: {n_bootstraps} bootstrap-resampled fits per model")
    fig.tight_layout()
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def plot_overfitting_curve(path: Path) -> None:
    """Train vs validation RMSE across polynomial degree -- the classic overfitting
    curve."""
    x, y = make_dataset(n=120, seed=RNG_SEED)
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.3, random_state=RNG_SEED
    )
    x_train_col, x_val_col = x_train.reshape(-1, 1), x_val.reshape(-1, 1)

    degrees = list(range(1, 16))
    train_rmse: list[float] = []
    val_rmse: list[float] = []
    for degree in degrees:
        model = make_poly_pipeline(degree)
        model.fit(x_train_col, y_train)
        train_rmse.append(root_mean_squared_error(y_train, model.predict(x_train_col)))
        val_rmse.append(root_mean_squared_error(y_val, model.predict(x_val_col)))

    best_degree = degrees[int(np.argmin(val_rmse))]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(degrees, train_rmse, marker="o", label="train RMSE")
    ax.plot(degrees, val_rmse, marker="o", label="validation RMSE")
    ax.axvline(
        best_degree, color="gray", linestyle=":",
        label=f"best on validation (degree={best_degree})",
    )
    ax.set_xlabel("polynomial degree (model complexity)")
    ax.set_ylabel("RMSE")
    ax.set_title("Overfitting curve: train vs validation error vs complexity")
    ax.legend()
    fig.tight_layout()
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")
    print(f"train RMSE by degree: {[round(v, 4) for v in train_rmse]}")
    print(f"val   RMSE by degree: {[round(v, 4) for v in val_rmse]}")
    print(f"best degree on validation: {best_degree}")


def main() -> None:
    plot_bias_variance(ARTEFACTS_DIR / "bias_variance_illustration.png")
    plot_overfitting_curve(ARTEFACTS_DIR / "overfitting_curve.png")


if __name__ == "__main__":
    main()

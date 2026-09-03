"""R^2 intuition: "beat the dumbest model" -- companion code for
Data Science/Worked Examples/regression-nyc-taxi.md (SPEC-DS-5), section 3.

What it does:
  1. Builds a tiny synthetic 1-D dataset (10 points scattered near a line, seed=42) -- small
     enough that every residual segment in the plot below is individually visible.
  2. Fits a LinearRegression through it (the "fitted line" -- the model actually doing work).
  3. Also computes the flat mean line y_hat = ybar (the "dumbest possible model": ignore every
     feature, always predict the average of y).
  4. Draws two side-by-side panels showing the SAME points against BOTH baselines, with a
     vertical segment at every point marking that point's residual:
       left  panel: residuals to the FITTED LINE  -> SSE = sum((y_i - yhat_i)^2)
       right panel: residuals to the FLAT MEAN    -> SST = sum((y_i - ybar)^2)
     SSE is "how wrong the model is"; SST is "how wrong the dumbest model is." R^2 = 1 -
     SSE/SST is the fraction of that dumbest-model error the fitted line manages to remove.
  5. Prints SSE, SST, and R^2 = 1 - SSE/SST, and cross-checks that number against
     sklearn.metrics.r2_score on the same data (they must agree to floating-point precision).

Formula grounding (R^2 = 1 - SS_res/SS_tot, and SS_tot/n = population variance of y, i.e.
SST = Var(Y, ddof=0) * n): https://en.wikipedia.org/wiki/Coefficient_of_determination
(checked 2026-09-03) and https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
(checked 2026-09-03) -- both confirm SS_tot uses the POPULATION variance (denominator n, i.e.
numpy's default ddof=0), not the (n-1) sample variance.

Environment (research/NOTE-2-package-versions.md, research/NOTE-5-sklearn-core-apis.md,
checked 2026-09-02): numpy==2.5.2, matplotlib==3.11.1, scikit-learn==1.9.0, Python 3.12+.

Run:
    python r2_intuition.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this script only saves a figure, never shows one
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

RNG_SEED = 42
CODE_DIR = Path(__file__).resolve().parent
ARTEFACTS_DIR = CODE_DIR.parent / "artefacts"


def make_tiny_dataset(n: int = 10, seed: int = RNG_SEED) -> tuple[np.ndarray, np.ndarray]:
    """10 points scattered around y = 2x + 1, small enough to draw every residual by hand."""
    rng = np.random.default_rng(seed)
    x = np.linspace(1.0, 10.0, n)
    noise = rng.normal(0, 1.5, n)
    y = 2.0 * x + 1.0 + noise
    return x, y


def sse_sst_r2(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float, float, float]:
    """Fit a line, then return (y_hat, ybar, SSE, SST, r2) computed by hand."""
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    y_hat = model.predict(x.reshape(-1, 1))

    ybar = float(np.mean(y))
    sse = float(np.sum((y - y_hat) ** 2))  # residuals to the FITTED LINE: "how wrong the model is"
    sst = float(np.sum((y - ybar) ** 2))  # residuals to the FLAT MEAN: "how wrong the dumbest model is"
    r2 = 1.0 - sse / sst
    return y_hat, ybar, sse, sst, r2


def plot_sse_vs_sst(x: np.ndarray, y: np.ndarray, y_hat: np.ndarray, ybar: float, sse: float, sst: float, r2: float) -> Path:
    """Two panels, same points: residuals to the fitted line (left) vs. to the flat mean (right)."""
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    # Left: residuals to the FITTED LINE -> SSE.
    order = np.argsort(x)
    ax_left.plot(x[order], y_hat[order], color="tab:blue", linewidth=2, label="fitted line $\\hat{y}$")
    ax_left.scatter(x, y, color="black", zorder=3, label="data $y_i$")
    for xi, yi, yhat_i in zip(x, y, y_hat):
        ax_left.plot([xi, xi], [yi, yhat_i], color="tab:red", linewidth=1.5, zorder=2)
    ax_left.set_title(f"Residuals to the fitted line\nSSE = {sse:.2f} (\"how wrong the model is\")")
    ax_left.set_xlabel("x")
    ax_left.set_ylabel("y")
    ax_left.legend(loc="upper left", fontsize=8)

    # Right: residuals to the FLAT MEAN -> SST.
    ax_right.axhline(ybar, color="tab:orange", linewidth=2, label=f"flat mean $\\bar{{y}}$ = {ybar:.2f}")
    ax_right.scatter(x, y, color="black", zorder=3, label="data $y_i$")
    for xi, yi in zip(x, y):
        ax_right.plot([xi, xi], [yi, ybar], color="tab:red", linewidth=1.5, zorder=2)
    ax_right.set_title(f"Residuals to the dumbest model (flat mean)\nSST = {sst:.2f} (\"how wrong guessing the average is\")")
    ax_right.set_xlabel("x")
    ax_right.legend(loc="upper left", fontsize=8)

    fig.suptitle(f"R² = 1 − SSE/SST = 1 − {sse:.2f}/{sst:.2f} = {r2:.4f}", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTEFACTS_DIR / "r2_sse_vs_sst.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    x, y = make_tiny_dataset()
    y_hat, ybar, sse, sst, r2 = sse_sst_r2(x, y)

    # Cross-check the hand-computed R^2 against sklearn's own r2_score on the same data --
    # they must match to floating-point precision (research/NOTE-5-sklearn-core-apis.md).
    r2_sklearn = r2_score(y, y_hat)
    assert abs(r2 - r2_sklearn) < 1e-10, f"hand-computed R2 ({r2}) disagrees with sklearn r2_score ({r2_sklearn})"

    # Cross-check SST against Var(Y) * n using numpy's POPULATION variance (ddof=0), the
    # relationship this section's prose asserts (grounded, see module docstring).
    sst_from_var = float(np.var(y, ddof=0) * len(y))
    assert abs(sst - sst_from_var) < 1e-8, f"SST ({sst}) disagrees with Var(Y, ddof=0) * n ({sst_from_var})"

    out_path = plot_sse_vs_sst(x, y, y_hat, ybar, sse, sst, r2)

    print("=== R2 intuition: beat the dumbest model ===")
    print(f"n = {len(x)} points, ybar (the dumbest model's one prediction) = {ybar:.4f}")
    print(f"SSE (residuals to the fitted line)   = {sse:.4f}")
    print(f"SST (residuals to the flat mean)     = {sst:.4f}")
    print(f"SST via Var(Y, ddof=0) * n            = {sst_from_var:.4f}  (matches SST, as expected)")
    print(f"R2 = 1 - SSE/SST                      = {r2:.4f}")
    print(f"sklearn.metrics.r2_score cross-check  = {r2_sklearn:.4f}  (matches, as expected)")
    print(f"saved: {out_path.relative_to(CODE_DIR.parent.parent.parent)}")


if __name__ == "__main__":
    main()

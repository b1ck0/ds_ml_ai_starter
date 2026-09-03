"""Collinearity via "many simple regressions vs one multiple regression" on a wine-like dataset.

Companion code for:
  Data Science/Worked Examples/03-collinearity.md, section "Three features secretly
  measuring time"

The story: Orley Ashenfelter's Bordeaux wine-price dataset (see the regression chapter,
05-regression-nyc-taxi.md, "The wine that predicted its own price") has six candidate
features. Three of them -- Year, Age, and FrancePop -- are, structurally, all just
disguised measurements of *time*: Age is defined as (a reference year minus Year), and
France's population only ever goes up, so it tracks Year almost one-for-one. This script
builds a small, seeded, synthetic dataset with that exact property, then runs the
"weights ordering" test the owner's regression deck uses to explain collinearity:

  1. Rank every feature by its OWN single-feature R^2 (many simple regressions) -- this
     is the reference importance ordering, because a lone feature can't "steal" credit
     from a correlated twin; there's nothing else in the model to steal it from.
  2. Fit ONE multiple regression on ALL features, standardized first so that coefficient
     MAGNITUDE is comparable across features on wildly different scales. Compare the
     coefficient ordering to the single-feature ranking above.
  3. Show that comparison BREAKS when Year, Age, and FrancePop are all present: a
     redundant time-twin's weight collapses toward 0 or flips sign, and the design
     matrix's condition number (numpy.linalg.cond and statsmodels' OLS "Cond. No.")
     goes through the roof.
  4. Drop the redundant time-twins (Year, FrancePop), refit, and show the coefficient
     ordering snaps back to match the single-feature R^2 ranking, with a condition
     number back in a normal range.

Environment (installed versions in the project .venv; NOTE-2 / NOTE-5 / NOTE-6 checked
2026-09-02; numpy.linalg.cond and statsmodels OLS/RegressionResults.condition_number /
add_constant verified directly against their stable docs, checked 2026-09-03 -- see the
chapter's citations):
    pandas==3.0.5, numpy==2.5.2, matplotlib==3.11.1, seaborn==0.13.2,
    scikit-learn==1.9.0, statsmodels==0.15.0
    Python 3.12+ (this script was run and gated on Python 3.13.7 in the project .venv).

Run:
    .venv/Scripts/python.exe collinearity_wine.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this script only saves figures, never shows them
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

RNG_SEED = 7
ARTEFACTS_DIR = Path(__file__).resolve().parent.parent / "artefacts"

REFERENCE_YEAR = 1990  # Age = REFERENCE_YEAR - Year, exactly
START_YEAR, END_YEAR = 1952, 1978  # 27 vintages, matching the real dataset's span

ALL_FEATURES = ["Year", "WinterRain", "AGST", "HarvestRain", "Age", "FrancePop"]
NO_TWINS_FEATURES = ["WinterRain", "AGST", "HarvestRain", "Age"]
TARGET = "Price"


def make_wine_data(seed: int = RNG_SEED) -> pd.DataFrame:
    """Small, seeded, wine-like dataset where Year, Age, and FrancePop are all time.

    Age is *defined* as REFERENCE_YEAR - Year, so Age and Year are perfectly collinear
    (correlation exactly -1.0) by construction, not by coincidence -- exactly like a
    dummy-variable trap, but continuous instead of one-hot. FrancePop is built from
    always-positive year-over-year growth steps, so it is monotonically increasing with
    Year by construction too, without being an exact linear function of it (a population
    curve, not a ruler) -- a *near*-duplicate of Year/Age rather than a perfect one.

    Price is generated causally from Age (aging genuinely raises price), AGST (a warmer
    growing season makes better wine), HarvestRain (harvest-time rain dilutes the grapes)
    and WinterRain (a small, real, positive effect) -- Year and FrancePop are NOT used to
    generate Price at all. Any apparent relationship between Price and Year/FrancePop is
    entirely a side effect of their collinearity with Age.
    """
    rng = np.random.default_rng(seed)
    years = np.arange(START_YEAR, END_YEAR + 1)
    n = len(years)

    winter_rain = rng.uniform(400, 800, n)  # mm, dormant-season rainfall
    agst = rng.uniform(14.5, 17.5, n)  # deg C, Average Growing Season Temperature
    harvest_rain = rng.uniform(40, 300, n)  # mm, rain during the harvest window

    age = REFERENCE_YEAR - years  # exact: years the wine had aged in cask/bottle by 1990

    # France's population: a slow demand proxy that can only go up, year over year --
    # built from always-positive growth steps so it is monotonic by construction, the
    # same shape as the real population series it stands in for.
    pop_growth_steps = rng.uniform(300, 900, n - 1)
    france_pop = np.concatenate([[42_000.0], 42_000.0 + np.cumsum(pop_growth_steps)])

    noise = rng.normal(0, 0.15, n)
    price = (
        0.030 * age  # aging genuinely raises price
        + 0.650 * (agst - 16.0)  # warmer growing season -> better wine
        - 0.0010 * harvest_rain  # harvest rain dilutes/damages the grapes
        + 0.0012 * winter_rain  # small, real, positive effect
        + 7.500
        + noise
    )

    return pd.DataFrame(
        {
            "Year": years,
            "WinterRain": winter_rain,
            "AGST": agst,
            "HarvestRain": harvest_rain,
            "Age": age,
            "FrancePop": france_pop,
            "Price": price,
        }
    )


def plot_correlation_heatmap(df: pd.DataFrame) -> Path:
    corr = df[ALL_FEATURES].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True, ax=ax)
    ax.set_title("Wine feature correlation matrix -- Year/Age/FrancePop all light up")
    fig.tight_layout()

    out_path = ARTEFACTS_DIR / "collinearity_wine_correlation_heatmap.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_time_twins_pairplot(df: pd.DataFrame) -> Path:
    """Pairplot of the three time-twins plus Price -- Year/Age is a perfect line."""
    g = sns.pairplot(df[["Year", "Age", "FrancePop", TARGET]], diag_kind="hist", height=2.1)
    g.figure.suptitle(
        "Year, Age, FrancePop, Price -- three features secretly measuring time", y=1.02
    )

    out_path = ARTEFACTS_DIR / "collinearity_wine_pairplot.png"
    g.figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(g.figure)
    return out_path


def single_feature_r2(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Many simple regressions: rank each feature by its OWN R^2, one at a time.

    This is the reference importance ordering. A single feature fit alone has nothing
    to "steal" credit from -- its R^2 reflects only its own real relationship with
    Price, whether or not it's collinear with anything else.
    """
    y = df[TARGET].values
    rows = []
    for feature in features:
        X = df[[feature]].values
        model = LinearRegression().fit(X, y)
        rows.append({"feature": feature, "single_r2": r2_score(y, model.predict(X))})
    return pd.DataFrame(rows).sort_values("single_r2", ascending=False).reset_index(drop=True)


def multiple_regression_coefficients(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """One multiple regression on ALL given features, standardized first.

    Standardizing (StandardScaler) puts every coefficient on the same "per one
    standard deviation of that feature" scale, so raw coefficient MAGNITUDE is
    comparable across features that live on very different units (a year count vs a
    population count vs a temperature in degrees C) -- exactly the pattern used for
    the bootstrap demo earlier in this chapter.
    """
    X = df[features]
    y = df[TARGET].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegression().fit(X_scaled, y)

    coefs = pd.DataFrame({"feature": features, "coef": model.coef_})
    coefs["abs_coef"] = coefs["coef"].abs()
    return coefs.sort_values("abs_coef", ascending=False).reset_index(drop=True)


def condition_numbers(df: pd.DataFrame, features: list[str]) -> tuple[float, float]:
    """The numeric signature of multicollinearity: how ill-conditioned is the design matrix.

    numpy.linalg.cond(x, p=None) returns the ratio of the largest to the smallest
    singular value of x (the 2-norm condition number), computed via SVD -- a small
    value (single digits to low tens) means the matrix is well-behaved; a huge value
    means it is close to singular, i.e. close to having a redundant column
    ([source: numpy.linalg.cond](https://numpy.org/doc/stable/reference/generated/numpy.linalg.cond.html),
    checked 2026-09-03).

    statsmodels' OLS results carry the same idea as the "Cond. No." row printed by
    `.summary()`, exposed programmatically as `RegressionResults.condition_number`
    ([source: statsmodels RegressionResults](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.html),
    checked 2026-09-03) -- computed on the design matrix statsmodels actually fit,
    i.e. the standardized features plus the intercept column added by
    `sm.add_constant(data, prepend=True, has_constant='skip')`
    ([source: statsmodels add_constant](https://www.statsmodels.org/stable/generated/statsmodels.tools.tools.add_constant.html),
    checked 2026-09-03).
    """
    X = df[features]
    y = df[TARGET].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cond_numpy = float(np.linalg.cond(X_scaled))

    X_with_const = sm.add_constant(X_scaled, prepend=True, has_constant="add")
    ols_results = sm.OLS(y, X_with_const).fit()
    cond_statsmodels = float(ols_results.condition_number)

    return cond_numpy, cond_statsmodels


def plot_coefficient_comparison(
    single_r2_all: pd.DataFrame,
    coefs_all: pd.DataFrame,
    single_r2_no_twins: pd.DataFrame,
    coefs_no_twins: pd.DataFrame,
) -> Path:
    """Bar charts: standardized coefficient magnitude next to single-feature R^2 rank,
    with all features vs with the time-twins dropped."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)

    def _panel(ax, single_r2: pd.DataFrame, coefs: pd.DataFrame, title: str) -> None:
        order = single_r2["feature"].tolist()
        coefs_ordered = coefs.set_index("feature").loc[order].reset_index()
        x = np.arange(len(order))
        width = 0.38

        ax2 = ax.twinx()
        ax.bar(x - width / 2, single_r2["single_r2"], width, color="tab:blue", label="single-feature R2")
        ax2.bar(x + width / 2, coefs_ordered["coef"], width, color="tab:orange", label="standardized coef")
        ax2.axhline(0, color="grey", linewidth=0.8, linestyle="--")

        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=30, ha="right")
        ax.set_ylabel("single-feature R2", color="tab:blue")
        ax2.set_ylabel("standardized multiple-regression coefficient", color="tab:orange")
        ax.set_title(title)

    _panel(axes[0], single_r2_all, coefs_all, "All 6 features (time-twins present)")
    _panel(axes[1], single_r2_no_twins, coefs_no_twins, "Time-twins dropped (Year, FrancePop)")

    fig.suptitle("Single-feature R2 rank (blue) vs standardized coefficient (orange), sorted by R2")
    fig.tight_layout()

    out_path = ARTEFACTS_DIR / "collinearity_wine_coef_vs_r2.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    df = make_wine_data()

    print("=== wine dataset shape ===")
    print(df.shape)
    print("\n=== the time-twins: Year vs Age vs FrancePop ===")
    print(f"corr(Year, Age)       = {df['Year'].corr(df['Age']):.10f}")
    print(f"corr(Year, FrancePop) = {df['Year'].corr(df['FrancePop']):.6f}")
    print(f"corr(Age, FrancePop)  = {df['Age'].corr(df['FrancePop']):.6f}")

    heatmap_path = plot_correlation_heatmap(df)
    pairplot_path = plot_time_twins_pairplot(df)

    single_r2_all = single_feature_r2(df, ALL_FEATURES)
    print("\n=== step 2: many simple regressions -- single-feature R^2 ranking ===")
    print(single_r2_all.to_string(index=False))

    coefs_all = multiple_regression_coefficients(df, ALL_FEATURES)
    print("\n=== step 3: one multiple regression (all 6 features, standardized) ===")
    print(coefs_all.to_string(index=False))

    cond_np_all, cond_sm_all = condition_numbers(df, ALL_FEATURES)
    print("\n=== step 4: condition number, all 6 features ===")
    print(f"numpy.linalg.cond      = {cond_np_all:,.1f}")
    print(f"statsmodels Cond. No.  = {cond_sm_all:,.1f}")

    single_r2_no_twins = single_feature_r2(df, NO_TWINS_FEATURES)
    coefs_no_twins = multiple_regression_coefficients(df, NO_TWINS_FEATURES)
    cond_np_no_twins, cond_sm_no_twins = condition_numbers(df, NO_TWINS_FEATURES)

    print("\n=== step 5: dropped Year and FrancePop -- refit ===")
    print("single-feature R^2 (unchanged, feature-by-feature fits don't see other columns):")
    print(single_r2_no_twins.to_string(index=False))
    print("\nstandardized multiple-regression coefficients:")
    print(coefs_no_twins.to_string(index=False))
    print(f"\nnumpy.linalg.cond      = {cond_np_no_twins:,.2f}")
    print(f"statsmodels Cond. No.  = {cond_sm_no_twins:,.2f}")

    comparison_path = plot_coefficient_comparison(
        single_r2_all, coefs_all, single_r2_no_twins, coefs_no_twins
    )

    # Persist the numbers so the chapter's tables reproduce byte-for-byte from this script.
    single_r2_all.to_csv(ARTEFACTS_DIR / "collinearity_wine_single_r2_all.csv", index=False)
    coefs_all.to_csv(ARTEFACTS_DIR / "collinearity_wine_coefs_all.csv", index=False)
    single_r2_no_twins.to_csv(ARTEFACTS_DIR / "collinearity_wine_single_r2_no_twins.csv", index=False)
    coefs_no_twins.to_csv(ARTEFACTS_DIR / "collinearity_wine_coefs_no_twins.csv", index=False)

    condition_summary = pd.DataFrame(
        {
            "design_matrix": ["all 6 features", "time-twins dropped"],
            "numpy_cond": [cond_np_all, cond_np_no_twins],
            "statsmodels_cond_no": [cond_sm_all, cond_sm_no_twins],
        }
    )
    condition_summary.to_csv(ARTEFACTS_DIR / "collinearity_wine_condition_numbers.csv", index=False)

    print(f"\nWrote: {heatmap_path}")
    print(f"Wrote: {pairplot_path}")
    print(f"Wrote: {comparison_path}")


if __name__ == "__main__":
    main()

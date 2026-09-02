"""Collinearity detection & coefficient instability on a synthetic house-price dataset.

Companion code for:
  Data Science/Worked Examples/collinearity.md

What it does:
  1. Builds a synthetic house-price dataset with a deliberately redundant feature pair:
     sqft and sqm measure the same physical floor area in two different units (+ small
     measurement noise), so they're near-duplicates of each other.
  2. Plots a correlation heatmap of all numeric features (saved to ../artefacts/).
  3. Computes VIF (variance inflation factor) for every feature via statsmodels (NOTE-6).
  4. Bootstrap-refits sklearn LinearRegression B times on rows resampled with replacement,
     with vs without the redundant sqm column, and plots the coefficient spread.
  5. Compares held-out R^2 with vs without the redundant column -- near-identical
     predictive power, which is the empirical backbone of the "minimum viable feature
     set" principle: dropping the redundant column costs nothing and buys stability.
  6. Demonstrates the one-hot dummy-variable trap: encoding a category without dropping
     a reference level makes the dummy columns sum to the intercept column exactly,
     which is a perfect (rank-deficient) collinearity invisible to a pairwise heatmap.
  7. Writes the VIF table to ../artefacts/collinearity_vif_table.csv.

Environment (installed versions in the project .venv; NOTE-2 / NOTE-5 / NOTE-6 checked
2026-09-02):
    pandas==3.0.5, numpy==2.5.2, matplotlib==3.11.1, seaborn==0.13.2,
    scikit-learn==1.9.0, statsmodels==0.15.0
    Python 3.11+ (this script was run and gated on Python 3.13.7 in the project .venv).

Run:
    .venv/Scripts/python.exe collinearity.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this script only saves figures, never shows them
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

RNG_SEED = 42
ARTEFACTS_DIR = Path(__file__).resolve().parent.parent / "artefacts"

SQFT_TO_SQM = 0.092903  # exact unit conversion: 1 square foot = 0.092903 square metres

FEATURES_WITH_DUP = ["sqft", "sqm", "bedrooms", "age_years", "distance_to_city_km"]
FEATURES_NO_DUP = ["sqft", "bedrooms", "age_years", "distance_to_city_km"]


def make_house_price_data(n: int = 500, seed: int = RNG_SEED) -> pd.DataFrame:
    """Synthetic house-price dataset with one deliberately redundant feature pair.

    sqft and sqm measure the *same physical quantity* (floor area) in two different
    units, plus small independent measurement noise -- the data equivalent of logging
    both bytes and kilobytes for the same payload size and feeding both columns to a
    model. `price` is generated from sqft (not sqm), bedrooms, age_years and
    distance_to_city_km -- sqm carries no information about price that the model
    doesn't already get from sqft.
    """
    rng = np.random.default_rng(seed)
    sqft = rng.uniform(600, 3600, n)
    sqm = sqft * SQFT_TO_SQM + rng.normal(0, 2.5, n)  # near-duplicate of sqft
    bedrooms = rng.integers(1, 6, n).astype(float)
    age_years = rng.uniform(0, 60, n)
    distance_to_city_km = rng.uniform(0.5, 40, n)
    noise = rng.normal(0, 15_000, n)

    price = (
        120 * sqft
        + 8_000 * bedrooms
        - 900 * age_years
        - 1_500 * distance_to_city_km
        + 50_000
        + noise
    )

    return pd.DataFrame(
        {
            "sqft": sqft,
            "sqm": sqm,
            "bedrooms": bedrooms,
            "age_years": age_years,
            "distance_to_city_km": distance_to_city_km,
            "price": price,
        }
    )


def plot_correlation_heatmap(df: pd.DataFrame) -> Path:
    corr = df[FEATURES_WITH_DUP].corr()

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True, ax=ax
    )
    ax.set_title("Feature correlation matrix (house-price dataset)")
    fig.tight_layout()

    out_path = ARTEFACTS_DIR / "collinearity_correlation_heatmap.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def compute_vif(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """VIF per column via statsmodels.stats.outliers_influence (NOTE-6).

    statsmodels 0.15.0's variance_inflation_factor(exog, exog_idx, *, standardize=True)
    standardizes columns internally, so no manual mean-centering or added constant
    column is needed -- pass the raw numeric feature matrix directly.
    """
    X = df[columns]
    vif = pd.DataFrame(
        {
            "feature": columns,
            "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
        }
    )
    return vif.sort_values("VIF", ascending=False).reset_index(drop=True)


def bootstrap_coefficients(
    X: pd.DataFrame, y: pd.Series, n_boot: int = 300, seed: int = RNG_SEED
) -> pd.DataFrame:
    """Refit a standardized LinearRegression n_boot times on rows resampled with replacement.

    Standardizing first (StandardScaler, fit once on the full sample) puts every
    feature's coefficient on the same "per one standard deviation of that feature"
    scale, so spreads are visually comparable across features with very different raw
    units (square feet vs a bedroom count).
    """
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    rng = np.random.default_rng(seed)
    n = len(X_scaled)
    rows = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)  # bootstrap resample: same n, with replacement
        model = LinearRegression().fit(X_scaled.iloc[idx], y.iloc[idx])
        rows.append(dict(zip(X.columns, model.coef_)))
    return pd.DataFrame(rows)


def plot_coefficient_spread(with_dup: pd.DataFrame, no_dup: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

    axes[0].boxplot(
        [with_dup[c] for c in FEATURES_WITH_DUP],
        tick_labels=FEATURES_WITH_DUP,
        showfliers=False,
    )
    axes[0].axhline(0, color="grey", linewidth=0.8, linestyle="--")
    axes[0].set_title("With redundant 'sqm' column")
    axes[0].set_ylabel("Bootstrap coefficient (standardized features)")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].boxplot(
        [no_dup[c] for c in FEATURES_NO_DUP],
        tick_labels=FEATURES_NO_DUP,
        showfliers=False,
    )
    axes[1].axhline(0, color="grey", linewidth=0.8, linestyle="--")
    axes[1].set_title("Without 'sqm' (sqft only)")
    axes[1].tick_params(axis="x", rotation=30)

    fig.suptitle(f"Bootstrap coefficient spread ({len(with_dup)} refits per panel)")
    fig.tight_layout()

    out_path = ARTEFACTS_DIR / "collinearity_coefficient_spread.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def compare_holdout_r2(df: pd.DataFrame, seed: int = RNG_SEED) -> tuple[float, float]:
    """Held-out R^2 with vs without the redundant column -- should be ~equal.

    This is the evidence for the "minimum viable feature set" principle: if dropping a
    column doesn't cost predictive power on unseen data, keeping it only adds
    instability and noise to interpret.
    """
    train, test = train_test_split(df, test_size=0.25, random_state=seed)

    model_with = LinearRegression().fit(train[FEATURES_WITH_DUP], train["price"])
    r2_with = r2_score(test["price"], model_with.predict(test[FEATURES_WITH_DUP]))

    model_no = LinearRegression().fit(train[FEATURES_NO_DUP], train["price"])
    r2_no = r2_score(test["price"], model_no.predict(test[FEATURES_NO_DUP]))

    return float(r2_with), float(r2_no)


def one_hot_dummy_trap_demo(df: pd.DataFrame) -> tuple[int, int, pd.DataFrame]:
    """Collinearity can hide inside one-hot dummies, invisible to a pairwise heatmap.

    Buckets houses into 3 distance-based tiers, one-hot encodes them two ways, and
    checks the *rank* of [intercept + dummy columns]:
      - drop_first=False: intercept + 3 dummies -> the 3 dummy columns always sum to
        1 (every row belongs to exactly one tier), which equals the intercept column
        exactly. That's a perfect linear dependency -- rank stays 3, not 4. This is
        the classic "dummy variable trap." Per NOTE-6, VIF is undefined/->infinity in
        this exact-collinearity case, so we check rank directly instead of calling
        variance_inflation_factor on a singular design matrix.
      - drop_first=True: intercept + 2 dummies -> full rank, and VIF is finite and
        reasonable.
    """
    tier = pd.cut(
        df["distance_to_city_km"], bins=[0, 10, 25, 100], labels=["near", "mid", "far"]
    )

    trap_dummies = pd.get_dummies(tier, prefix="tier", drop_first=False).astype(float)
    trap_design = trap_dummies.copy()
    trap_design.insert(0, "intercept", 1.0)
    trap_rank = int(np.linalg.matrix_rank(trap_design.values))

    safe_dummies = pd.get_dummies(tier, prefix="tier", drop_first=True).astype(float)
    safe_design = safe_dummies.copy()
    safe_design.insert(0, "intercept", 1.0)
    safe_rank = int(np.linalg.matrix_rank(safe_design.values))

    vif_safe = compute_vif(safe_design, list(safe_design.columns))

    print("\n=== one-hot dummy trap: distance_to_city_km tier ===")
    print(f"drop_first=False: {trap_design.shape[1]} columns, rank={trap_rank} "
          f"({'RANK-DEFICIENT -- dummy trap' if trap_rank < trap_design.shape[1] else 'full rank'})")
    print(f"drop_first=True:  {safe_design.shape[1]} columns, rank={safe_rank} "
          f"({'full rank' if safe_rank == safe_design.shape[1] else 'RANK-DEFICIENT'})")
    print("\nVIF with drop_first=True (full-rank design):")
    print(vif_safe.to_string(index=False))

    return trap_rank, safe_rank, vif_safe


def main() -> None:
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    df = make_house_price_data()

    print("=== shape ===")
    print(df.shape)
    print("\n=== correlation (sqft vs sqm) ===")
    print(f"r = {df['sqft'].corr(df['sqm']):.4f}")

    heatmap_path = plot_correlation_heatmap(df)

    vif_with_dup = compute_vif(df, FEATURES_WITH_DUP)
    print("\n=== VIF: all features, including redundant sqm ===")
    print(vif_with_dup.to_string(index=False))

    vif_no_dup = compute_vif(df, FEATURES_NO_DUP)
    print("\n=== VIF: sqm dropped ===")
    print(vif_no_dup.to_string(index=False))

    vif_path = ARTEFACTS_DIR / "collinearity_vif_table.csv"
    vif_with_dup.to_csv(vif_path, index=False)

    boot_with_dup = bootstrap_coefficients(df[FEATURES_WITH_DUP], df["price"])
    boot_no_dup = bootstrap_coefficients(df[FEATURES_NO_DUP], df["price"])
    spread_path = plot_coefficient_spread(boot_with_dup, boot_no_dup)

    print("\n=== bootstrap coefficient std dev (standardized features) ===")
    print("with sqm:   ", boot_with_dup.std().round(1).to_dict())
    print("without sqm:", boot_no_dup.std().round(1).to_dict())

    r2_with, r2_no = compare_holdout_r2(df)
    print(f"\n=== held-out R^2 ===\nwith sqm: {r2_with:.4f}    without sqm: {r2_no:.4f}")

    one_hot_dummy_trap_demo(df)

    print(f"\nWrote: {heatmap_path}")
    print(f"Wrote: {spread_path}")
    print(f"Wrote: {vif_path}")


if __name__ == "__main__":
    main()

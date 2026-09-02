"""Hypothesis testing & EDA on the Palmer Penguins dataset.

Companion code for:
  Data Science/Worked Examples/hypothesis-testing-and-eda.md

What it does:
  1. Loads the Palmer Penguins dataset via seaborn's bundled loader (NOTE-1).
  2. Runs a first EDA pass: shape, dtypes, missingness, describe().
  3. Plots a histogram and a boxplot-by-species (saved to ../artefacts/).
  4. Answers two concrete questions with the right test:
       - "Does flipper length differ between Adelie and Chinstrap?" -> Welch's t-test + Cohen's d
       - "Is species associated with island?"                       -> chi-square + Cramer's V
  5. Demonstrates the "huge N makes anything significant" pitfall on bill_depth_mm.
  6. Writes a small results table to ../artefacts/hypothesis_test_results.csv.

Environment (verified in research/NOTE-2-package-versions.md, checked 2026-09-02):
    pandas==3.0.5, numpy==2.5.2, matplotlib==3.11.1, scipy==1.18.1, seaborn==0.13.2
    Python 3.11+ (this script was run and gated on Python 3.13.7).

Run:
    python hypothesis_testing_eda.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this script only saves figures, never shows them
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, ttest_ind

RNG_SEED = 42
ARTEFACTS_DIR = Path(__file__).resolve().parent.parent / "artefacts"


def cohens_d(x: pd.Series, y: pd.Series) -> float:
    """Cohen's d for two independent samples, using the pooled standard deviation.

    d = (mean(x) - mean(y)) / pooled_std
    pooled_std = sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))

    Not provided by scipy.stats -- formula verified in research/NOTE-4-effect-sizes.md.
    Thresholds (Cohen, 1988): 0.2 small, 0.5 medium, 0.8 large.
    """
    n1, n2 = len(x), len(y)
    var1, var2 = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return float((np.mean(x) - np.mean(y)) / pooled_std)


def cramers_v(chi2_stat: float, n: int, rows: int, cols: int) -> float:
    """Cramer's V for an r x c contingency table.

    V = sqrt(chi2 / (n * min(rows-1, cols-1)))

    Not provided by scipy.stats -- formula verified in research/NOTE-4-effect-sizes.md.
    Thresholds (conventional): <0.1 negligible, 0.1-0.3 weak, 0.3-0.5 moderate, >0.5 strong.
    """
    min_dim = min(rows - 1, cols - 1)
    return float(np.sqrt(chi2_stat / (n * min_dim))) if min_dim > 0 else float("nan")


def load_data() -> pd.DataFrame:
    """Palmer Penguins via seaborn.load_dataset -- CC0, see research/NOTE-1-eda-dataset.md."""
    return sns.load_dataset("penguins")


def eda_pass(df: pd.DataFrame) -> None:
    print("=== shape ===")
    print(df.shape)

    print("\n=== dtypes ===")
    print(df.dtypes)

    print("\n=== missingness (NaN count per column) ===")
    print(df.isna().sum())

    print("\n=== describe() (numeric columns) ===")
    print(df.describe())


def plot_histogram(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(df["flipper_length_mm"].dropna(), bins=20, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Flipper length (mm)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of flipper length -- all species")
    fig.tight_layout()

    out_path = ARTEFACTS_DIR / "flipper_length_histogram.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_boxplot_by_species(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    order = ["Adelie", "Chinstrap", "Gentoo"]
    sns.boxplot(data=df, x="species", y="flipper_length_mm", order=order, ax=ax)
    ax.set_xlabel("Species")
    ax.set_ylabel("Flipper length (mm)")
    ax.set_title("Flipper length by species")
    fig.tight_layout()

    out_path = ARTEFACTS_DIR / "flipper_length_boxplot_by_species.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def run_t_test(df: pd.DataFrame) -> dict:
    """Question: does flipper_length_mm differ between Adelie and Chinstrap?"""
    clean = df.dropna(subset=["flipper_length_mm", "species"])
    adelie = clean.loc[clean["species"] == "Adelie", "flipper_length_mm"]
    chinstrap = clean.loc[clean["species"] == "Chinstrap", "flipper_length_mm"]

    # equal_var=False -> Welch's t-test (does not assume equal variances). See NOTE-3.
    result = ttest_ind(adelie, chinstrap, equal_var=False)
    d = cohens_d(adelie, chinstrap)

    print("\n=== t-test: flipper_length_mm, Adelie vs Chinstrap ===")
    print(f"n_adelie={len(adelie)}, n_chinstrap={len(chinstrap)}")
    print(f"mean_adelie={adelie.mean():.2f}mm, mean_chinstrap={chinstrap.mean():.2f}mm")
    print(f"t={result.statistic:.3f}, p={result.pvalue:.3e}, df={result.df:.1f}")
    print(f"Cohen's d={d:.3f}")

    return {
        "test": "Welch t-test: flipper_length_mm, Adelie vs Chinstrap",
        "statistic": result.statistic,
        "p_value": result.pvalue,
        "effect_size_name": "Cohen's d",
        "effect_size": d,
    }


def run_chi_square(df: pd.DataFrame) -> dict:
    """Question: is species associated with island?"""
    clean = df.dropna(subset=["species", "island"])
    contingency = pd.crosstab(clean["species"], clean["island"])

    chi2, p, dof, expected = chi2_contingency(contingency)  # see NOTE-3
    n = int(contingency.to_numpy().sum())
    v = cramers_v(chi2, n, *contingency.shape)

    print("\n=== chi-square: species vs island ===")
    print(contingency)
    print(f"chi2={chi2:.3f}, p={p:.3e}, dof={dof}")
    print(f"Cramer's V={v:.3f}")

    return {
        "test": "Chi-square: species vs island",
        "statistic": chi2,
        "p_value": p,
        "effect_size_name": "Cramer's V",
        "effect_size": v,
    }


def pitfall_huge_n_demo(df: pd.DataFrame) -> None:
    """Same (trivial) effect size, wildly different p-value once N is inflated.

    Illustrates: statistical significance is not practical significance. We resample
    (with replacement) from the real bill_depth_mm distributions to simulate a much
    bigger dataset, and rerun the same Welch's t-test.
    """
    clean = df.dropna(subset=["bill_depth_mm", "species"])
    adelie = clean.loc[clean["species"] == "Adelie", "bill_depth_mm"].to_numpy()
    chinstrap = clean.loc[clean["species"] == "Chinstrap", "bill_depth_mm"].to_numpy()

    small_result = ttest_ind(adelie, chinstrap, equal_var=False)
    small_d = cohens_d(adelie, chinstrap)

    rng = np.random.default_rng(RNG_SEED)
    adelie_big = rng.choice(adelie, size=50_000, replace=True)
    chinstrap_big = rng.choice(chinstrap, size=50_000, replace=True)
    big_result = ttest_ind(adelie_big, chinstrap_big, equal_var=False)
    big_d = cohens_d(adelie_big, chinstrap_big)

    print("\n=== pitfall demo: bill_depth_mm, Adelie vs Chinstrap ===")
    print(f"actual sample   (n={len(adelie)} vs {len(chinstrap)}): "
          f"p={small_result.pvalue:.3f}, d={small_d:.3f}")
    print(f"resampled to n=50,000 each:            "
          f"p={big_result.pvalue:.3e}, d={big_d:.3f}")


def main() -> None:
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    eda_pass(df)
    hist_path = plot_histogram(df)
    box_path = plot_boxplot_by_species(df)

    t_test_row = run_t_test(df)
    chi_sq_row = run_chi_square(df)
    pitfall_huge_n_demo(df)

    results = pd.DataFrame([t_test_row, chi_sq_row])
    results_path = ARTEFACTS_DIR / "hypothesis_test_results.csv"
    results.to_csv(results_path, index=False)

    print(f"\nWrote: {hist_path}")
    print(f"Wrote: {box_path}")
    print(f"Wrote: {results_path}")


if __name__ == "__main__":
    main()

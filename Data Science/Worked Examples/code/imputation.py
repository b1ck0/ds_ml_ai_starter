"""Imputation strategies on the Palmer Penguins dataset's real missing values.

Companion code for:
  Data Science/Worked Examples/imputation.md

What it does:
  1. Loads Palmer Penguins (seaborn's bundled loader, NOTE-1 / NOTE-8) and prints a
     missingness table.
  2. Plots a missingness heatmap and shows that the 4 numeric columns lose exactly the
     same 2 rows (an MCAR-shaped pattern), while `sex` is missing on 11 scattered rows.
  3. Runs a controlled experiment: take the 342 penguins with a complete
     `body_mass_g` + `flipper_length_mm` pair, synthetically blank out 30% of
     `body_mass_g` (missing completely at random, seeded), then compare mean
     imputation, median imputation, and KNNImputer against the ground truth we just
     hid -- on standard deviation and correlation with `flipper_length_mm`.
  4. Plots the before/after distribution (true vs mean-imputed vs KNN-imputed).
  5. Imputes the categorical `sex` column with `strategy="most_frequent"` and shows
     why that's a near coin-flip here.
  6. Demonstrates `add_indicator=True` -- keeping a "was this missing?" signal instead
     of silently erasing it.
  7. Shows what `dropna()` would have cost (LO4: drop vs impute).
  8. Demonstrates the leakage pitfall: fitting an imputer on the whole dataset before
     splitting vs fitting it on the training split only (inside a Pipeline).
  9. Writes a comparison table to ../artefacts/imputation_strategy_comparison.csv.

Environment (verified in research/NOTE-2-package-versions.md and
research/NOTE-5-sklearn-core-apis.md, checked 2026-09-02):
    pandas==3.0.5, numpy==2.5.2, matplotlib==3.11.1, scipy==1.18.1, seaborn==0.13.2,
    scikit-learn==1.9.0
    Python 3.11+ (this script was run and gated on Python 3.13.7).

Run:
    python imputation.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this script only saves figures, never shows them
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

RNG_SEED = 42
AMPLIFIED_MISSING_FRAC = 0.30
ARTEFACTS_DIR = Path(__file__).resolve().parent.parent / "artefacts"

NUMERIC_COLS = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]


def load_data() -> pd.DataFrame:
    """Palmer Penguins via seaborn.load_dataset -- CC0, see research/NOTE-8-imputation-dataset.md."""
    return sns.load_dataset("penguins")


def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column NaN count and percentage -- the DataFrame equivalent of grepping a log for null."""
    counts = df.isna().sum()
    pct = (counts / len(df) * 100).round(2)
    table = pd.DataFrame({"missing_count": counts, "missing_pct": pct})
    print("=== missingness by column ===")
    print(table)
    return table


def plot_missingness_heatmap(df: pd.DataFrame) -> Path:
    """One row per penguin, one column per feature -- yellow where a value is missing.

    This is the plain seaborn/matplotlib equivalent of the missingno package's
    matrix plot: sns.heatmap on the boolean isna() frame.
    """
    fig, ax = plt.subplots(figsize=(6, 7))
    sns.heatmap(df.isna(), cbar=False, yticklabels=False, cmap=["#EAEAF2", "#C44E52"], ax=ax)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row (344 penguins)")
    ax.set_title("Missingness map -- red = NaN")
    fig.tight_layout()

    out_path = ARTEFACTS_DIR / "missingness_heatmap.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def describe_missingness_pattern(df: pd.DataFrame) -> None:
    """Show WHICH rows are missing and whether they overlap -- the MCAR/MAR intuition check."""
    numeric_missing_idx = set(df.index[df[NUMERIC_COLS].isna().any(axis=1)])
    sex_missing_idx = set(df.index[df["sex"].isna()])

    print("\n=== missingness pattern ===")
    print(f"Rows missing >=1 numeric measurement: {sorted(numeric_missing_idx)}")
    print(df.loc[sorted(numeric_missing_idx), ["species", "island"] + NUMERIC_COLS])
    print(f"\nAll 4 numeric columns share the SAME missing rows: "
          f"{all(set(df.index[df[c].isna()]) == numeric_missing_idx for c in NUMERIC_COLS)}")
    print("-> shape of a measurement failure for one penguin, not a per-column problem "
          "(MCAR-consistent).")

    print(f"\n`sex` is missing on {len(sex_missing_idx)} rows, scattered across species/island:")
    print(df.loc[sorted(sex_missing_idx), ["species", "island"]].value_counts())
    print(f"Overlap with the numeric-missing rows: {sorted(numeric_missing_idx & sex_missing_idx)}")
    print("-> the 2 measurement-failure penguins ALSO have no recorded sex; the other 9 are "
          "missing sex only -- consistent with sex being harder to determine in the field for "
          "some birds (MAR/MNAR-shaped, not pure noise).")


def build_amplified_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Synthetic experiment: start from the 342 penguins with a real body_mass_g +
    flipper_length_mm pair (both fully observed there), then hide 30% of body_mass_g
    ourselves (MCAR by construction, seeded). This is NOT a claim about the real
    dataset -- the real numeric missingness is only 2 rows (0.6%), too small to see a
    variance/correlation effect by eye. Blanking out 30% ourselves, with a fixed seed,
    lets us compare each imputation strategy's output against ground truth we still have.
    """
    base = df.dropna(subset=["body_mass_g", "flipper_length_mm"]).reset_index(drop=True)
    rng = np.random.default_rng(RNG_SEED)
    n_missing = int(round(AMPLIFIED_MISSING_FRAC * len(base)))
    missing_idx = rng.choice(len(base), size=n_missing, replace=False)

    amplified = base.copy()
    amplified.loc[missing_idx, "body_mass_g"] = np.nan
    print(f"\n=== amplified-missingness demo ===")
    print(f"Base rows (real body_mass_g + flipper_length_mm, no NaN): {len(base)}")
    print(f"Synthetically blanked {n_missing} body_mass_g values "
          f"({AMPLIFIED_MISSING_FRAC:.0%}, seed={RNG_SEED})")
    return amplified


def compare_imputation_strategies(base: pd.DataFrame, amplified: pd.DataFrame) -> pd.DataFrame:
    """Mean vs median vs KNN imputation of body_mass_g, scored against the ground truth
    we hid in build_amplified_missing(). Reports std and correlation with
    flipper_length_mm -- the two things mean/median imputation is known to distort.
    """
    true_std = base["body_mass_g"].std()
    true_corr = base["body_mass_g"].corr(base["flipper_length_mm"])
    flipper = amplified["flipper_length_mm"].to_numpy()

    rows = [{
        "strategy": "true (no missing)",
        "fill_value": np.nan,
        "std": true_std,
        "std_change_pct": 0.0,
        "corr_with_flipper": true_corr,
        "corr_change_pct": 0.0,
    }]

    mean_imp = SimpleImputer(strategy="mean")
    mean_filled = mean_imp.fit_transform(amplified[["body_mass_g"]]).ravel()
    rows.append(_score_strategy("mean imputation", mean_imp.statistics_[0], mean_filled,
                                 flipper, true_std, true_corr))

    median_imp = SimpleImputer(strategy="median")
    median_filled = median_imp.fit_transform(amplified[["body_mass_g"]]).ravel()
    rows.append(_score_strategy("median imputation", median_imp.statistics_[0], median_filled,
                                 flipper, true_std, true_corr))

    # KNNImputer needs a feature to measure "similarity" on -- flipper_length_mm is always
    # observed here, so distance is computed on it (nan_euclidean handles the missing
    # body_mass_g column automatically). This mirrors "find the 5 most similar penguins by
    # flipper length, then average their body mass."
    knn_imp = KNNImputer(n_neighbors=5)
    knn_input = amplified[["flipper_length_mm", "body_mass_g"]].to_numpy()
    knn_filled = knn_imp.fit_transform(knn_input)[:, 1]
    rows.append(_score_strategy("KNN imputation (k=5)", np.nan, knn_filled,
                                 flipper, true_std, true_corr))

    table = pd.DataFrame(rows)
    print("\n=== imputation strategy comparison (body_mass_g, 30% synthetically missing) ===")
    print(table.to_string(index=False))
    return table, mean_filled, knn_filled


def _score_strategy(name: str, fill_value: float, filled: np.ndarray, flipper: np.ndarray,
                     true_std: float, true_corr: float) -> dict:
    std = filled.std(ddof=1)
    corr = float(np.corrcoef(filled, flipper)[0, 1])
    return {
        "strategy": name,
        "fill_value": fill_value,
        "std": std,
        "std_change_pct": (std - true_std) / true_std * 100,
        "corr_with_flipper": corr,
        "corr_change_pct": (corr - true_corr) / true_corr * 100,
    }


def plot_before_after_distribution(base: pd.DataFrame, mean_filled: np.ndarray,
                                    knn_filled: np.ndarray) -> Path:
    """Overlaid histograms: the true (hidden) distribution vs what mean imputation and
    KNN imputation actually produced. Mean imputation shows a visible spike at the mean.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(2700, 6300, 30)
    ax.hist(base["body_mass_g"], bins=bins, alpha=0.5, label="true (before hiding 30%)",
            color="#4C72B0", edgecolor="white")
    ax.hist(mean_filled, bins=bins, alpha=0.5, label="mean-imputed", color="#C44E52",
            edgecolor="white")
    ax.hist(knn_filled, bins=bins, alpha=0.5, label="KNN-imputed (k=5)", color="#55A868",
            edgecolor="white")
    ax.axvline(base["body_mass_g"].mean(), color="#C44E52", linestyle="--", linewidth=1)
    ax.set_xlabel("Body mass (g)")
    ax.set_ylabel("Count")
    ax.set_title("body_mass_g: true vs mean- vs KNN-imputed (30% synthetic missing)")
    ax.legend()
    fig.tight_layout()

    out_path = ARTEFACTS_DIR / "body_mass_g_before_after_imputation.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def impute_categorical_sex(df: pd.DataFrame) -> None:
    """Categorical missingness needs a different strategy: most_frequent, not mean/median."""
    before = df["sex"].value_counts(dropna=False)
    imp = SimpleImputer(strategy="most_frequent")
    filled = imp.fit_transform(df[["sex"]]).ravel()
    after = pd.Series(filled).value_counts()

    print("\n=== categorical imputation: sex (strategy='most_frequent') ===")
    print(f"Before:\n{before}")
    print(f"\nFill value chosen: {imp.statistics_[0]!r}")
    print(f"After:\n{after}")
    print("Male led Female by only 168 vs 165 (3 penguins) before imputation -- "
          "most_frequent breaks that near-tie by filling ALL 11 unknowns with the same "
          "label, which is closer to a coin flip than a real signal.")


def demonstrate_missing_indicator(df: pd.DataFrame) -> None:
    """add_indicator=True keeps a 'was this missing?' column instead of erasing the fact."""
    X = df[NUMERIC_COLS].to_numpy()
    imp = SimpleImputer(strategy="mean", add_indicator=True)
    out = imp.fit_transform(X)

    print("\n=== missingness indicator (SimpleImputer(add_indicator=True)) ===")
    print(f"Input shape: {X.shape} -> output shape: {out.shape}")
    print(f"Indicator added for column indices: {list(imp.indicator_.features_)} "
          f"({[NUMERIC_COLS[i] for i in imp.indicator_.features_]})")
    print("Every numeric column gets its own indicator here because every one of them has "
          "at least one NaN (both from the same 2 rows). The model now sees the imputed "
          "value AND a flag saying 'this was estimated, not measured.'")


def demonstrate_drop_vs_impute(df: pd.DataFrame) -> None:
    """What dropna() actually costs on this dataset -- LO4."""
    dropped = df.dropna()
    n_lost = len(df) - len(dropped)
    print("\n=== drop vs impute ===")
    print(f"df.dropna() (any column): {len(df)} -> {len(dropped)} rows "
          f"({n_lost} lost, {n_lost / len(df) * 100:.1f}%)")
    print("Here that's tolerable: 3.2% of rows, and the missingness looks MCAR/MAR, not "
          "correlated with the target. Dropping stops being acceptable when the missing "
          "rows are a meaningful fraction of the data, OR when missingness is informative "
          "(MNAR) -- e.g. if heavier penguins were systematically harder to weigh in the "
          "field, dropping them would bias body_mass_g's distribution downward, silently.")


def demonstrate_leakage(amplified: pd.DataFrame) -> None:
    """The core discipline: fit the imputer on TRAIN only, never on train+test combined.

    Wrong: fit SimpleImputer on the full dataset, THEN split -- the statistic used to
    fill missing values already "saw" the test rows.
    Right: split first, fit only on train, wrap it in a Pipeline so the same discipline
    applies automatically to .transform(test) / .predict(test) later.
    """
    train, test = train_test_split(amplified, test_size=0.25, random_state=RNG_SEED)

    leaky = SimpleImputer(strategy="mean").fit(amplified[["body_mass_g"]])  # fit on ALL rows
    correct = SimpleImputer(strategy="mean").fit(train[["body_mass_g"]])    # fit on TRAIN only

    print("\n=== leakage demo: fit-before-split vs fit-after-split ===")
    print(f"train={len(train)} rows ({train['body_mass_g'].isna().sum()} missing), "
          f"test={len(test)} rows ({test['body_mass_g'].isna().sum()} missing)")
    print(f"Leaky mean (fit on train+test combined): {leaky.statistics_[0]:.2f} g")
    print(f"Correct mean (fit on train only):        {correct.statistics_[0]:.2f} g")
    print(f"Difference: {leaky.statistics_[0] - correct.statistics_[0]:.2f} g")
    print("The gap is small here because the missingness is random (MCAR) and train/test "
          "come from the same distribution -- the leak barely moves the number, which is "
          "exactly why it's easy to miss. It doesn't stay small: with informative "
          "missingness, many imputed columns, or small folds, the same mistake compounds.")

    # The idiomatic fix: encapsulate imputer + model in one Pipeline. .fit() only ever
    # touches whatever X you pass it (X_train); the SAME fitted imputer is reused for
    # .transform(X_test) -- there is no code path left where test data can leak into a
    # training statistic.
    pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean"))])
    pipeline.fit(train[["body_mass_g"]])
    test_transformed = pipeline.transform(test[["body_mass_g"]])
    print(f"\nPipeline.fit(train) then .transform(test) reuses the train-only statistic: "
          f"{pipeline.named_steps['imputer'].statistics_[0]:.2f} g "
          f"(matches 'correct' above: {correct.statistics_[0]:.2f} g)")


def main() -> None:
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    missingness_table(df)
    heatmap_path = plot_missingness_heatmap(df)
    describe_missingness_pattern(df)

    amplified = build_amplified_missing(df)
    base = df.dropna(subset=["body_mass_g", "flipper_length_mm"]).reset_index(drop=True)
    comparison_table, mean_filled, knn_filled = compare_imputation_strategies(base, amplified)
    dist_path = plot_before_after_distribution(base, mean_filled, knn_filled)

    impute_categorical_sex(df)
    demonstrate_missing_indicator(df)
    demonstrate_drop_vs_impute(df)
    demonstrate_leakage(amplified)

    table_path = ARTEFACTS_DIR / "imputation_strategy_comparison.csv"
    comparison_table.to_csv(table_path, index=False)

    print(f"\nWrote: {heatmap_path}")
    print(f"Wrote: {dist_path}")
    print(f"Wrote: {table_path}")


if __name__ == "__main__":
    main()

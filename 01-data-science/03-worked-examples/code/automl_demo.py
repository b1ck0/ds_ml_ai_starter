"""AutoML on Titanic: let FLAML search preprocessing + model + hyperparameters, and
compare the result to the hand-built LogisticRegression from classification_titanic.py.

Companion code for:
  Data Science/Worked Examples/automl.md

What it does:
  1. Loads Titanic via seaborn's bundled loader and engineers the SAME features as
     SPEC-DS-6's classification_titanic.py (family_size, is_alone, fare_bin) so the
     comparison in this script is apples-to-apples with that chapter's numbers
     (NOTE-10-classification-datasets.md).
  2. Splits train/test with the identical stratified 75/25 split (random_state=42) used
     in classification_titanic.py.
  3. Hand-built baseline: the exact preprocessing ColumnTransformer + LogisticRegression
     Pipeline from classification_titanic.py -- ONE model family, hyperparameters left
     at scikit-learn's defaults (max_iter=1000 aside), fit once. Timed with
     time.perf_counter().
  4. FLAML AutoML: flaml.AutoML().fit() on the RAW engineered DataFrame (untransformed --
     NaNs in age, string/categorical columns for fare_bin/sex/embarked) with a small
     30-second time budget, searching six estimator families (lgbm, xgboost, rf,
     extra_tree, lrl1, lrl2) using FLAML's BlendSearch algorithm (NOTE-15). FLAML does
     its OWN imputation/encoding internally -- no ColumnTransformer supplied -- which is
     itself part of what "AutoML" automates versus the hand-built pipeline in step 3.
     Also timed with time.perf_counter().
  5. Reads FLAML's leaderboard (automl.best_loss_per_estimator -- one CV loss per
     estimator family it tried) and its single best config (automl.best_config,
     automl.best_estimator) -- confirmed against the installed flaml==2.6.0 API
     (NOTE-15-automl-framework.md).
  6. Scores both the hand-built pipeline and the FLAML model on the SAME held-out test
     set: accuracy, precision, recall, F1, ROC-AUC, PR-AUC.
  7. Writes a leaderboard CSV, a metrics-comparison CSV, and a two-panel comparison PNG
     (metrics grouped bars + fit-time bars on a log scale) to ../artefacts/.

Environment (verified in research/NOTE-15-automl-framework.md and re-verified by direct
installation into a DEDICATED virtualenv on 2026-09-02, Python 3.13.7):
    flaml[automl]==2.6.0, scikit-learn==1.9.0, pandas==3.0.5, numpy==2.5.2,
    matplotlib==3.11.1, seaborn==0.13.2

    NOTE-15 recommended `pip install flaml==2.6.0` as a ~349 KB wheel with numpy as its
    only dependency. That is true for the bare `flaml` package, but running
    AutoML().fit(task="classification") with FLAML's default/named estimators (lgbm,
    xgboost, ...) additionally requires the `automl` extra
    (`pip install "flaml[automl]==2.6.0"`), which pulls in lightgbm==4.7.0 and
    xgboost==2.1.4 (xgboost's wheel alone is ~125 MB). This gap in NOTE-15's install-size
    claim was found empirically while gating this chapter and is reported to the
    architect rather than silently corrected -- see the chapter's environment note.

Run (from a DEDICATED venv -- see the chapter's "Local setup" section, NOT the shared
project .venv, to avoid pulling lightgbm/xgboost into every other chapter's environment):
    automl-venv/Scripts/python automl_demo.py     (Windows)
    automl-venv/bin/python automl_demo.py         (macOS/Linux)
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this script only saves figures, never shows them
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

RNG_SEED = 42
ARTEFACTS_DIR = Path(__file__).resolve().parent.parent / "artefacts"

FARE_BIN_LABELS = ["low", "mid", "high", "very_high"]
NUMERIC_FEATURES = ["age", "family_size", "pclass", "is_alone"]
ORDINAL_FEATURES = ["fare_bin"]
NOMINAL_FEATURES = ["sex", "embarked"]
FEATURE_COLUMNS = NUMERIC_FEATURES + ORDINAL_FEATURES + NOMINAL_FEATURES
TARGET = "survived"

# Small on purpose -- this is a classroom sandbox demo, not a production search.
# FLAML's own docs frame the time budget as the main dial the user turns; 30s is enough
# for BlendSearch to complete dozens of trials across six estimator families on a
# 668-row training set (NOTE-15-automl-framework.md).
TIME_BUDGET_SECONDS = 30
ESTIMATOR_LIST = ["lgbm", "xgboost", "rf", "extra_tree", "lrl1", "lrl2"]


def load_and_engineer_features() -> pd.DataFrame:
    """Titanic via seaborn, with the SAME feature engineering as classification_titanic.py.

    Reproduced here (not imported) so this script stays runnable standalone, exactly the
    way a reader would copy-paste it. See classification-titanic.md Section 2 for why
    each engineered column exists and research/NOTE-10-classification-datasets.md for the
    dataset's shape/NaN counts this recipe assumes.
    """
    df = sns.load_dataset("titanic")
    df["family_size"] = df["sibsp"] + df["parch"] + 1
    df["is_alone"] = (df["family_size"] == 1).astype(int)
    df["fare_bin"] = pd.qcut(df["fare"], q=4, labels=FARE_BIN_LABELS)
    return df


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Identical stratified 75/25 split (random_state=42) to classification_titanic.py."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.25, random_state=RNG_SEED, stratify=y)


def build_handbuilt_pipeline() -> Pipeline:
    """The exact preprocessing + LogisticRegression pipeline from classification_titanic.py.

    ONE model family, default hyperparameters (aside from max_iter), no search --
    this is the "senior engineer spent an afternoon on it" baseline AutoML gets
    compared against. Signatures verified against scikit-learn 1.9.0
    (research/NOTE-5-sklearn-core-apis.md).
    """
    numeric_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    ordinal_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(categories=[FARE_BIN_LABELS])),
    ])
    nominal_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(drop="if_binary", handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("ord", ordinal_pipe, ORDINAL_FEATURES),
        ("nom", nominal_pipe, NOMINAL_FEATURES),
    ])
    return Pipeline(steps=[
        ("prep", preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=RNG_SEED)),
    ])


def score_predictions(y_test: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """Same six metrics classification_titanic.py reports -- for a like-for-like table."""
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
    }


def run_handbuilt_baseline(X_train, y_train, X_test, y_test) -> tuple[dict, float]:
    """Fit the hand-built pipeline once, time it, score it on the held-out test set."""
    pipeline = build_handbuilt_pipeline()
    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t0

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = score_predictions(y_test, y_pred, y_proba)
    metrics["model"] = "hand_built_logistic_regression"
    metrics["fit_seconds"] = fit_seconds
    return metrics, fit_seconds


def run_flaml_search(X_train, y_train, X_test, y_test):
    """AutoML().fit() on the RAW (untransformed) features -- FLAML does its own
    imputation/encoding internally, unlike the hand-built pipeline above.

    API confirmed against the installed flaml==2.6.0
    (research/NOTE-15-automl-framework.md): AutoML().fit(X, y, task=, time_budget=,
    metric=, seed=, estimator_list=); best_estimator / best_config / best_loss /
    best_loss_per_estimator / predict / predict_proba are all real attributes/methods on
    the fitted AutoML object, verified directly against this version, not assumed.

    FLAML's estimator_list includes named scikit-learn-style hyperparameter search
    spaces for classic model families (lrl1/lrl2 = L1/L2-penalised LogisticRegression,
    rf/extra_tree = scikit-learn ensembles, lgbm/xgboost = gradient-boosted trees) --
    this IS the "grid search on steroids" search space LO1 asks this chapter to name.
    """
    from flaml import AutoML  # imported here so the module still imports without flaml
                               # installed, for the snippet-compile gate

    automl = AutoML()
    # FLAML's own estimators (in particular lrl1/lrl2, which wrap
    # sklearn.linear_model.LogisticRegression) still pass the now-deprecated `penalty=`
    # keyword under scikit-learn 1.9.0 (research/NOTE-5-sklearn-core-apis.md notes
    # `penalty` deprecated as of 1.8) -- confirmed empirically while gating this chapter,
    # not documented in NOTE-15. The warnings are cosmetic (the fit still completes and
    # scores normally) but would otherwise flood the search log for 30 seconds straight.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        t0 = time.perf_counter()
        automl.fit(
            X_train,
            y_train,
            task="classification",
            time_budget=TIME_BUDGET_SECONDS,
            metric="accuracy",
            seed=RNG_SEED,
            estimator_list=ESTIMATOR_LIST,
            verbose=0,
        )
        fit_seconds = time.perf_counter() - t0

        y_pred = automl.predict(X_test)
        y_proba = automl.predict_proba(X_test)[:, 1]

    metrics = score_predictions(y_test, y_pred, y_proba)
    metrics["model"] = f"flaml_automl ({automl.best_estimator})"
    metrics["fit_seconds"] = fit_seconds
    return metrics, fit_seconds, automl


def build_leaderboard(automl) -> pd.DataFrame:
    """One row per estimator family FLAML tried, ranked by best cross-validated loss.

    automl.best_loss_per_estimator is a real attribute on the fitted AutoML object
    (verified directly against flaml==2.6.0 -- research/NOTE-15-automl-framework.md).
    Because metric="accuracy" was passed to fit(), FLAML's internal "loss" for each
    estimator is 1 - (best cross-validated accuracy found for that estimator within the
    time budget), so cv_accuracy_estimate below is the natural inverse.
    """
    rows = []
    for estimator, loss in automl.best_loss_per_estimator.items():
        rows.append({
            "estimator": estimator,
            "cv_best_loss": loss,
            "cv_accuracy_estimate": 1.0 - loss,
            "is_overall_best": estimator == automl.best_estimator,
        })
    table = pd.DataFrame(rows).sort_values("cv_best_loss").reset_index(drop=True)
    table.insert(0, "rank", range(1, len(table) + 1))
    return table


def plot_comparison(handbuilt: dict, flaml: dict) -> Path:
    """Two-panel comparison: metric bars (left) + fit-time bars, log scale (right).

    The time panel is why this needs its own log-scale axis: the hand-built fit takes a
    fraction of a second, FLAML's search runs for the full 30-second budget by design --
    plotting both on one linear axis would flatten the hand-built bar to invisible.
    """
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    x = np.arange(len(metric_names))
    width = 0.35

    fig, (ax_metrics, ax_time) = plt.subplots(1, 2, figsize=(13, 5.5),
                                               gridspec_kw={"width_ratios": [2.2, 1]})

    handbuilt_vals = [handbuilt[m] for m in metric_names]
    flaml_vals = [flaml[m] for m in metric_names]
    ax_metrics.bar(x - width / 2, handbuilt_vals, width, label=handbuilt["model"],
                    color="#4C72B0")
    ax_metrics.bar(x + width / 2, flaml_vals, width, label=flaml["model"],
                    color="#DD8452")
    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels(metric_names, rotation=20)
    ax_metrics.set_ylim(0, 1.0)
    ax_metrics.set_ylabel("Score (test set)")
    ax_metrics.set_title("AutoML vs hand-built -- test-set metrics")
    ax_metrics.legend(loc="lower right", fontsize=9)
    ax_metrics.axhline(0, color="black", linewidth=0.5)

    labels = ["hand-built\n(1 model, defaults)", f"FLAML AutoML\n({TIME_BUDGET_SECONDS}s budget)"]
    times = [handbuilt["fit_seconds"], flaml["fit_seconds"]]
    bars = ax_time.bar(labels, times, color=["#4C72B0", "#DD8452"])
    ax_time.set_yscale("log")
    ax_time.set_ylabel("Fit wall-clock time (seconds, log scale)")
    ax_time.set_title("Cost to obtain the model")
    for bar, t in zip(bars, times):
        ax_time.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.15,
                     f"{t:.3f}s" if t < 1 else f"{t:.1f}s",
                     ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    out_path = ARTEFACTS_DIR / "automl_vs_handbuilt_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_engineer_features()
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"=== split === train={len(X_train)} test={len(X_test)} (stratified on survived)")

    print("\n=== hand-built baseline: LogisticRegression (single model, one fit) ===")
    handbuilt_metrics, handbuilt_seconds = run_handbuilt_baseline(X_train, y_train, X_test, y_test)
    for k, v in handbuilt_metrics.items():
        if k not in ("model",):
            print(f"  {k}: {v}")

    print(f"\n=== FLAML AutoML search: time_budget={TIME_BUDGET_SECONDS}s, "
          f"estimator_list={ESTIMATOR_LIST} ===")
    flaml_metrics, flaml_seconds, automl = run_flaml_search(X_train, y_train, X_test, y_test)
    print(f"  best_estimator: {automl.best_estimator}")
    print(f"  best_config: {automl.best_config}")
    print(f"  time_to_find_best_model: {automl.time_to_find_best_model:.2f}s "
          f"(within the {TIME_BUDGET_SECONDS}s budget)")
    for k, v in flaml_metrics.items():
        if k not in ("model",):
            print(f"  {k}: {v}")

    leaderboard = build_leaderboard(automl)
    print("\n=== leaderboard (one row per estimator family FLAML searched) ===")
    print(leaderboard.to_string(index=False))
    leaderboard_path = ARTEFACTS_DIR / "automl_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)

    comparison_table = pd.DataFrame([handbuilt_metrics, flaml_metrics])
    print("\n=== hand-built vs FLAML AutoML -- test-set metrics + fit time ===")
    print(comparison_table.to_string(index=False))
    comparison_csv_path = ARTEFACTS_DIR / "automl_vs_handbuilt_metrics.csv"
    comparison_table.to_csv(comparison_csv_path, index=False)

    chart_path = plot_comparison(handbuilt_metrics, flaml_metrics)

    print(f"\nWrote: {leaderboard_path}")
    print(f"Wrote: {comparison_csv_path}")
    print(f"Wrote: {chart_path}")


if __name__ == "__main__":
    main()

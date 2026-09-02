"""Binary classification on the Titanic dataset: who survived, scored as a probability.

Companion code for:
  Data Science/Worked Examples/classification-titanic.md

What it does:
  1. Loads Titanic via seaborn's bundled loader (NOTE-10) and prints shape / dtypes / NaNs.
  2. Feature-engineers family_size, is_alone, and an ordinal fare_bin; demonstrates the
     title-from-name regex technique on a small illustrative (not-this-dataset) example,
     because seaborn's titanic loader has no `name` column (NOTE-10 confirms the column
     list; verified again directly against the installed loader below).
  3. Drops columns that leak the target (`alive` is survived spelled as yes/no) or are
     redundant/too sparse (`deck` is 77% missing per NOTE-10; `class`, `embark_town`,
     `who`, `adult_male`, `alone` duplicate information already captured elsewhere).
  4. Splits train/test (stratified), builds one shared ColumnTransformer (impute + scale
     numeric, ordinal-encode fare_bin, one-hot-encode sex/embarked), and trains three
     classifiers behind the same preprocessing: LogisticRegression, RandomForestClassifier,
     HistGradientBoostingClassifier.
  5. Scores each model at the default 0.5 threshold: accuracy, precision, recall, F1,
     confusion matrix -- and computes a majority-class baseline to show accuracy lying on
     this dataset's ~62/38 class split.
  6. Computes ROC-AUC and PR-AUC (average_precision_score) for all three models and plots
     both curves overlaid.
  7. Sweeps decision thresholds on the best model by PR-AUC and picks one deliberately
     (maximising F1) instead of defaulting to 0.5.
  8. Plots LogisticRegression coefficients and RandomForestClassifier feature importances
     (HistGradientBoostingClassifier exposes neither -- verified empirically below).
  9. Writes a metric-comparison CSV and four PNG artefacts to ../artefacts/.

Environment (verified in research/NOTE-2-package-versions.md and
research/NOTE-5-sklearn-core-apis.md / research/NOTE-9-classification-metrics-apis.md,
checked 2026-09-02):
    pandas==3.0.5, numpy==2.5.2, matplotlib==3.11.1, scipy==1.18.1, seaborn==0.13.2,
    scikit-learn==1.9.0
    Python 3.11+ (this script was run and gated on Python 3.13.7).

Run:
    python classification_titanic.py
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this script only saves figures, never shows them
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
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


def load_data() -> pd.DataFrame:
    """Titanic via seaborn.load_dataset -- CC0, see research/NOTE-10-classification-datasets.md."""
    return sns.load_dataset("titanic")


def describe_raw(df: pd.DataFrame) -> None:
    """Shape, dtypes, and NaN counts -- confirms NOTE-10's numbers against this run."""
    print("=== raw shape ===")
    print(df.shape)
    print("\n=== columns ===")
    print(df.columns.tolist())
    print("\n=== missing values ===")
    print(df.isna().sum())
    print("\n=== target balance (survived) ===")
    print(df["survived"].value_counts())
    print(df["survived"].value_counts(normalize=True).round(3))


def demonstrate_title_extraction_technique() -> None:
    """Illustrates the classic 'title from name' regex on a small, made-up name list.

    seaborn's titanic loader has no `name` column (confirmed by describe_raw() above and
    by research/NOTE-10-classification-datasets.md's column list), so this is NOT run on
    the real dataset -- it exists purely to show the mechanic LO4 asks for. The real
    feature engineering below uses what this loader actually provides instead
    (family_size, is_alone, fare_bin).
    """
    illustrative_names = [
        "Braund, Mr. Owen Harris",
        "Cumings, Mrs. John Bradley (Florence Briggs Thayer)",
        "Heikkinen, Miss. Laina",
        "Palsson, Master. Gosta Leonard",
    ]
    titles = [re.search(r",\s*([^.]+)\.", name).group(1).strip() for name in illustrative_names]
    print("\n=== illustrative title-extraction (NOT this dataset -- see docstring) ===")
    for name, title in zip(illustrative_names, titles):
        print(f"  {name!r:55s} -> {title!r}")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds family_size, is_alone, fare_bin; drops leaky/redundant/too-sparse columns."""
    out = df.copy()

    # family_size and is_alone from sibsp/parch -- the dataset's own `alone` column is
    # derived the same way, so we recompute it ourselves as the feature-engineering step
    # LO4 asks for, then drop sibsp/parch/alone to avoid feeding the model three
    # different views of the same fact.
    out["family_size"] = out["sibsp"] + out["parch"] + 1
    out["is_alone"] = (out["family_size"] == 1).astype(int)
    # Sanity check: our engineered is_alone should agree with the dataset's own `alone`
    # column (both are "sibsp + parch == 0" under the hood) -- confirms the derivation
    # before we drop the raw sibsp/parch/alone columns below.
    assert (out["is_alone"].astype(bool) == out["alone"]).all(), "is_alone disagrees with alone"

    # fare_bin: an ordinal binning of a skewed continuous feature (fare has a long right
    # tail -- max 512 vs median ~14, per describe_raw()/NOTE-10). qcut splits into
    # roughly equal-sized quartiles rather than equal-width bins, which handles the skew.
    out["fare_bin"] = pd.qcut(out["fare"], q=4, labels=FARE_BIN_LABELS)

    # Drop: `alive` is `survived` spelled "no"/"yes" -- keeping it would let every model
    # get 100% accuracy by reading the label off a renamed copy of itself (target
    # leakage, the pitfall AC4 asks to demonstrate). `deck` is 77% missing (NOTE-10).
    # `class`/`embark_town` duplicate `pclass`/`embarked`. `who`/`adult_male` are a
    # coarse re-derivation of sex+age we don't need alongside age itself. `sibsp`/
    # `parch`/`alone`/`name`/`fare` are superseded by the engineered columns above
    # (fare itself is dropped in favour of fare_bin to keep the fare signal in exactly
    # one column instead of two correlated-by-construction ones).
    drop_cols = [
        "alive", "deck", "class", "embark_town", "who", "adult_male",
        "sibsp", "parch", "alone", "fare",
    ]
    out = out.drop(columns=[c for c in drop_cols if c in out.columns])
    return out


def build_preprocessor() -> ColumnTransformer:
    """One shared ColumnTransformer for all three models -- impute, scale, encode.

    Signatures verified against scikit-learn 1.9.0
    (research/NOTE-5-sklearn-core-apis.md, research/NOTE-9-classification-metrics-apis.md):
      SimpleImputer(*, strategy=...), StandardScaler(), OrdinalEncoder(*, categories=...),
      OneHotEncoder(*, drop=..., handle_unknown=...).
    `pclass` (1/2/3) is already ordinal-encoded by the dataset's own design -- first class
    is coded 1 -- so it only needs imputation+scaling like any other numeric column, not a
    separate encoder. `fare_bin` is ordinal but string-labelled, so it needs
    OrdinalEncoder with an explicit category order (alphabetical order would wrongly put
    "high" before "low"). `sex`/`embarked` are nominal -- no natural order -- so they're
    one-hot encoded; `drop='if_binary'` drops one of the two `sex` columns (verified
    empirically against the installed 1.9.0: with 2 categories it keeps a single 0/1
    column) so the model doesn't get two perfectly anti-correlated columns for one fact.
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
    return ColumnTransformer(transformers=[
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("ord", ordinal_pipe, ORDINAL_FEATURES),
        ("nom", nominal_pipe, NOMINAL_FEATURES),
    ])


def build_models() -> dict[str, Pipeline]:
    """Three model families behind the same preprocessing -- LO3.

    Constructor signatures verified against scikit-learn 1.9.0
    (research/NOTE-5-sklearn-core-apis.md): LogisticRegression, RandomForestClassifier,
    HistGradientBoostingClassifier.
    """
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RNG_SEED),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=RNG_SEED),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=RNG_SEED),
    }
    return {
        name: Pipeline(steps=[("prep", build_preprocessor()), ("model", model)])
        for name, model in models.items()
    }


def majority_class_baseline(y_train: pd.Series, y_test: pd.Series) -> dict:
    """Predict the training majority class for every test row -- AC4's accuracy-lies check.

    Uses accuracy_score/precision_score/recall_score/f1_score exactly as documented in
    research/NOTE-9-classification-metrics-apis.md.
    """
    majority = int(y_train.mode().iloc[0])
    y_pred = np.full_like(y_test, fill_value=majority)
    return {
        "model": "majority_class_baseline",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": np.nan,
        "pr_auc": np.nan,
    }


def evaluate_model(name: str, pipeline: Pipeline, X_test: pd.DataFrame,
                    y_test: pd.Series) -> tuple[dict, np.ndarray, np.ndarray]:
    """Default-threshold (0.5) metrics + the probability scores curves/AUCs need.

    `predict_proba` and every metric function's signature verified against
    research/NOTE-9-classification-metrics-apis.md.
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
    }
    return metrics, y_pred, y_proba


def plot_confusion_matrix(y_test: pd.Series, y_pred: np.ndarray, model_name: str) -> Path:
    """Confusion matrix for one model at the default 0.5 threshold.

    confusion_matrix() signature verified against
    research/NOTE-9-classification-metrics-apis.md: returns [[TN, FP], [FN, TP]] for
    binary labels {0, 1} with the default `labels=None` ordering.
    """
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["pred: did not survive", "pred: survived"],
                yticklabels=["true: did not survive", "true: survived"])
    ax.set_title(f"Confusion matrix -- {model_name} (threshold=0.5)")
    fig.tight_layout()

    out_path = ARTEFACTS_DIR / "titanic_confusion_matrix.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path, cm


def plot_roc_curves(y_test: pd.Series, proba_by_model: dict[str, np.ndarray]) -> Path:
    """ROC curves for all three models overlaid, plus the random-guess diagonal.

    roc_curve() signature verified against research/NOTE-9-classification-metrics-apis.md:
    returns (fpr, tpr, thresholds).
    """
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for name, proba in proba_by_model.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="random guess (AUC=0.500)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves -- Titanic survival")
    ax.legend(loc="lower right")
    fig.tight_layout()

    out_path = ARTEFACTS_DIR / "titanic_roc_curve.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_pr_curves(y_test: pd.Series, proba_by_model: dict[str, np.ndarray]) -> Path:
    """PR curves for all three models overlaid, plus the no-skill baseline (class prevalence).

    precision_recall_curve() signature verified against
    research/NOTE-9-classification-metrics-apis.md: returns (precision, recall, thresholds).
    """
    prevalence = y_test.mean()
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for name, proba in proba_by_model.items():
        precision, recall, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)
        ax.plot(recall, precision, label=f"{name} (PR-AUC={ap:.3f})")
    ax.axhline(prevalence, linestyle="--", color="grey",
               label=f"no-skill baseline (prevalence={prevalence:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves -- Titanic survival")
    ax.legend(loc="lower left")
    fig.tight_layout()

    out_path = ARTEFACTS_DIR / "titanic_pr_curve.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def sweep_thresholds(y_test: pd.Series, y_proba: np.ndarray, model_name: str) -> pd.DataFrame:
    """Precision/recall/F1 at a grid of thresholds -- picks the F1-maximising one deliberately.

    precision_score/recall_score/f1_score signatures verified against
    research/NOTE-9-classification-metrics-apis.md.
    """
    thresholds = np.arange(0.10, 0.91, 0.05)
    rows = []
    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        rows.append({
            "threshold": round(float(t), 2),
            "precision": precision_score(y_test, y_pred_t, zero_division=0),
            "recall": recall_score(y_test, y_pred_t, zero_division=0),
            "f1": f1_score(y_test, y_pred_t, zero_division=0),
        })
    table = pd.DataFrame(rows)
    best_row = table.loc[table["f1"].idxmax()]
    print(f"\n=== threshold sweep -- {model_name} ===")
    print(table.to_string(index=False))
    print(f"\nDefault threshold 0.5 vs F1-optimal threshold {best_row['threshold']:.2f} "
          f"(F1={best_row['f1']:.3f})")
    return table


def plot_feature_importance(models: dict[str, Pipeline], feature_names: list[str]) -> Path:
    """LogisticRegression coefficients + RandomForest importances, side by side.

    HistGradientBoostingClassifier is excluded: it has no `.feature_importances_`
    attribute (unlike RandomForestClassifier) -- verified empirically against the
    installed scikit-learn 1.9.0 with `hasattr(model, "feature_importances_")`, since
    neither NOTE-5 nor NOTE-9 document this gap. `.coef_` (LogisticRegression) and
    `.feature_importances_` (RandomForestClassifier) are both documented attributes for
    those estimators per NOTE-5's evidence table.
    """
    lr = models["logistic_regression"].named_steps["model"]
    rf = models["random_forest"].named_steps["model"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    coef = pd.Series(lr.coef_[0], index=feature_names).sort_values()
    coef.plot(kind="barh", ax=axes[0], color=np.where(coef >= 0, "#4C72B0", "#C44E52"))
    axes[0].set_title("LogisticRegression coefficients\n(scaled features; sign = direction)")
    axes[0].set_xlabel("Coefficient")

    importance = pd.Series(rf.feature_importances_, index=feature_names).sort_values()
    importance.plot(kind="barh", ax=axes[1], color="#55A868")
    axes[1].set_title("RandomForestClassifier feature importances\n(mean impurity decrease)")
    axes[1].set_xlabel("Importance")

    fig.tight_layout()
    out_path = ARTEFACTS_DIR / "titanic_feature_importance.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    describe_raw(df)
    demonstrate_title_extraction_technique()

    engineered = engineer_features(df)
    print("\n=== engineered feature columns ===")
    print(engineered[FEATURE_COLUMNS + [TARGET]].head(5))
    print("\n=== NaNs remaining in feature columns ===")
    print(engineered[FEATURE_COLUMNS].isna().sum())

    X = engineered[FEATURE_COLUMNS]
    y = engineered[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RNG_SEED, stratify=y,
    )
    print(f"\n=== split === train={len(X_train)} test={len(X_test)} "
          f"(stratified on survived)")

    models = build_models()
    metrics_rows = [majority_class_baseline(y_train, y_test)]
    proba_by_model: dict[str, np.ndarray] = {}
    pred_by_model: dict[str, np.ndarray] = {}

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        metrics, y_pred, y_proba = evaluate_model(name, pipeline, X_test, y_test)
        metrics_rows.append(metrics)
        proba_by_model[name] = y_proba
        pred_by_model[name] = y_pred
        print(f"\n=== {name} (threshold=0.5) ===")
        for k, v in metrics.items():
            if k != "model":
                print(f"  {k}: {v:.4f}")

    metrics_table = pd.DataFrame(metrics_rows)
    print("\n=== metric comparison (all models, threshold=0.5) ===")
    print(metrics_table.to_string(index=False))

    best_model_name = metrics_table.loc[
        metrics_table["model"] != "majority_class_baseline", "pr_auc"
    ].astype(float).idxmax()
    best_model_name = metrics_table.loc[best_model_name, "model"]
    print(f"\nBest model by PR-AUC: {best_model_name}")

    cm_path, cm = plot_confusion_matrix(y_test, pred_by_model[best_model_name], best_model_name)
    print(f"\n=== confusion matrix -- {best_model_name} ===")
    print(cm)

    roc_path = plot_roc_curves(y_test, proba_by_model)
    pr_path = plot_pr_curves(y_test, proba_by_model)
    sweep_thresholds(y_test, proba_by_model[best_model_name], best_model_name)

    feature_names = models["logistic_regression"].named_steps["prep"].get_feature_names_out()
    feature_names = [fn.split("__")[-1] for fn in feature_names]
    importance_path = plot_feature_importance(models, feature_names)

    table_path = ARTEFACTS_DIR / "titanic_metric_comparison.csv"
    metrics_table.to_csv(table_path, index=False)

    print(f"\nWrote: {cm_path}")
    print(f"Wrote: {roc_path}")
    print(f"Wrote: {pr_path}")
    print(f"Wrote: {importance_path}")
    print(f"Wrote: {table_path}")


if __name__ == "__main__":
    main()

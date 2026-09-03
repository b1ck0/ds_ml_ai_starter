"""Filter, wrapper, and embedded feature selection on the Breast Cancer Wisconsin
(diagnostic) dataset -- the knee/elbow method, and the selection-leakage pitfall.

Companion code for:
  Data Science/Worked Examples/feature-selection.md

What it does (mirrors the chapter's sections):
  1. Loads sklearn's bundled load_breast_cancer() dataset (569 rows, 30 features, binary
     target) -- same loader as train-valid-holdout-split.md (NOTE-5, NOTE-13).
  2. Baseline: 5-fold CV accuracy of a StandardScaler + LogisticRegression pipeline using
     all 30 features.
  3. Filter method: SelectKBest(f_classif) run for every k in 1..30, INSIDE a
     cross-validated pipeline, to build the performance-vs-#features "knee" curve (LO3).
     A small tolerance-based elbow finder locates the point of diminishing returns, and a
     sensitivity table shows how the tolerance choice moves that point (NOTE-13: the
     elbow heuristic is subjective).
  4. Wrapper method: RFECV traces its own performance-vs-#features curve (via a different
     selection criterion -- recursive coefficient elimination, not a univariate score) and
     is overlaid on the same knee plot. A fixed-size RFE(k=ELBOW_K) reports the specific
     features it keeps.
  5. Wrapper method: SequentialFeatureSelector, direction='forward' and 'backward', both
     targeting ELBOW_K features. Compares which features each direction picks, how long
     each took (wrapper methods are expensive -- NOTE-13), and each one's HONEST
     cross-validated accuracy (selection re-run inside every outer fold).
  6. Embedded methods: a short Lasso demo (grounds the literal Lasso API from NOTE-13 on a
     linear-probability-model version of the target), then the real classification-embedded
     method -- L1-penalised LogisticRegression + SelectFromModel -- and tree importance via
     RandomForestClassifier + SelectFromModel(threshold='median').
  7. THE CRITICAL PITFALL (LO4): selection leakage. Fits SelectKBest on 500 columns of
     PURE RANDOM NOISE (zero true relationship to the label) two ways -- once outside the
     CV loop (peeking at every row before scoring) and once inside a Pipeline (refit per
     training fold only) -- and shows the outside-CV version reports accuracy well above
     the majority-class baseline on data that is, by construction, unpredictable.
  8. Writes three artefacts to ../artefacts/:
       - feature_selection_knee_plot.png       (LO3: performance vs #features + elbow)
       - feature_selection_selected_features.csv (LO2: selection matrix, every method)
       - feature_selection_forward_vs_backward.png (LO2: SFS forward vs backward)

Environment (installed versions in the project .venv; NOTE-5 / NOTE-13 checked
2026-09-02):
    numpy==2.5.2, pandas==3.0.5, matplotlib==3.11.1, scikit-learn==1.9.0, scipy==1.18.1
    Python 3.12+ (this script was run and gated on Python 3.13.7 in the project .venv).

Run:
    .venv/Scripts/python.exe feature_selection.py
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
from scipy import stats
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    RFE,
    RFECV,
    SelectFromModel,
    SelectKBest,
    SequentialFeatureSelector,
    f_classif,
    mutual_info_classif,
)
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RNG_SEED = 42
ARTEFACTS_DIR = Path(__file__).resolve().parent.parent / "artefacts"

SCORING = "accuracy"
N_SPLITS = 5

# The single feature count reused across RFE / SequentialFeatureSelector so every wrapper
# method in the "selected feature table" is selecting the *same number* of features -- the
# comparison is then about *which* features and *how expensive*, not about who got to pick
# a bigger budget. Computed from the filter knee curve at ELBOW_TOL (see find_elbow()).
ELBOW_TOL = 0.01


def load_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Breast Cancer Wisconsin (diagnostic) -- bundled with scikit-learn, no download.

    569 rows, 30 numeric features, binary target (0=malignant, 1=benign), class counts
    [212, 357] (verified against the installed sklearn==1.9.0 in
    research/NOTE-13-feature-selection-apis.md and research/NOTE-5-sklearn-core-apis.md;
    loader confirmed by running it against this environment).
    [source: sklearn.datasets.load_breast_cancer docs](
    https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset)
    (checked 2026-09-02)
    """
    data = load_breast_cancer()
    return data.data.copy(), data.target.copy(), list(data.feature_names)


def make_cv(seed: int = RNG_SEED) -> StratifiedKFold:
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)


def base_classifier() -> LogisticRegression:
    return LogisticRegression(max_iter=2000, random_state=RNG_SEED)


# --------------------------------------------------------------------------------------
# Section 2: baseline -- all 30 features
# --------------------------------------------------------------------------------------


def baseline_score(X: np.ndarray, y: np.ndarray) -> float:
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", base_classifier())])
    scores = cross_val_score(pipe, X, y, cv=make_cv(), scoring=SCORING)
    print(f"=== baseline: all 30 features ===\nCV {SCORING}: {scores.mean():.4f} "
          f"(std {scores.std():.4f}) per-fold={np.round(scores, 4).tolist()}")
    return float(scores.mean())


# --------------------------------------------------------------------------------------
# Section 3: filter method + the knee/elbow curve (LO2, LO3)
# --------------------------------------------------------------------------------------


def filter_knee_curve(X: np.ndarray, y: np.ndarray, max_k: int) -> pd.DataFrame:
    """CV accuracy for every k in 1..max_k, selection done INSIDE the pipeline (CV-safe).

    This is simultaneously the LO3 knee-curve data *and* a worked demonstration of LO4:
    SelectKBest is a pipeline step, refit fresh on each training fold by cross_val_score --
    never fit once on the whole dataset before scoring.
    """
    rows = []
    for k in range(1, max_k + 1):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif, k=k)),
            ("clf", base_classifier()),
        ])
        scores = cross_val_score(pipe, X, y, cv=make_cv(), scoring=SCORING)
        rows.append({"k": k, "mean_score": scores.mean(), "std_score": scores.std()})
    return pd.DataFrame(rows)


def find_elbow(k: np.ndarray, scores: np.ndarray, tol: float) -> int:
    """Smallest k whose score is within `tol` of the best score achieved by any k.

    This is the practical form of the knee/elbow heuristic described in NOTE-13: find
    where marginal gains from adding more features plateau, and stop there. `tol` is a
    judgment call -- NOTE-13 flags the elbow heuristic itself as inherently subjective ("
    'maximum curvature' can be ambiguous on noisy curves ... visual inspection
    recommended"), so this function is deliberately parameterised by tol rather than
    hard-coding one "correct" answer.
    """
    best = scores.max()
    for ki, si in zip(k, scores):
        if si >= best - tol:
            return int(ki)
    return int(k[-1])


def elbow_sensitivity_table(df: pd.DataFrame) -> pd.DataFrame:
    k = df["k"].to_numpy()
    scores = df["mean_score"].to_numpy()
    rows = [
        {"tol": tol, "elbow_k": find_elbow(k, scores, tol),
         "score_at_elbow": scores[find_elbow(k, scores, tol) - 1]}
        for tol in (0.005, 0.01, 0.02, 0.03)
    ]
    return pd.DataFrame(rows)


def rfecv_curve(X: np.ndarray, y: np.ndarray) -> tuple[pd.DataFrame, RFECV]:
    """RFECV traces its own performance-vs-#features curve via a *different* selection
    criterion (recursive elimination using the estimator's coefficients) than the
    univariate SelectKBest filter curve -- worth overlaying on the same axes precisely
    because the two methods can (and here, do) land on different feature counts."""
    rfecv = RFECV(
        estimator=base_classifier(),
        step=1,
        min_features_to_select=1,
        cv=make_cv(),
        scoring=SCORING,
    )
    rfecv.fit(StandardScaler().fit_transform(X), y)
    df = pd.DataFrame({
        "n_features": rfecv.cv_results_["n_features"],
        "mean_score": rfecv.cv_results_["mean_test_score"],
    }).sort_values("n_features").reset_index(drop=True)
    return df, rfecv


def plot_knee_curve(
    filter_df: pd.DataFrame,
    rfecv_df: pd.DataFrame,
    elbow_k: int,
    rfecv_n: int,
    baseline: float,
) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    ax.plot(filter_df["k"], filter_df["mean_score"], marker="o", markersize=3.5,
             color="#4C72B0", label="Filter: SelectKBest(f_classif), selected inside CV")
    ax.fill_between(
        filter_df["k"],
        filter_df["mean_score"] - filter_df["std_score"],
        filter_df["mean_score"] + filter_df["std_score"],
        color="#4C72B0", alpha=0.15,
    )
    ax.plot(rfecv_df["n_features"], rfecv_df["mean_score"], marker="s", markersize=3.5,
             color="#DD8452", label="Wrapper: RFECV (recursive elimination)")

    ax.axvline(elbow_k, color="#4C72B0", linestyle="--", linewidth=1.2,
               label=f"Filter elbow (tol={ELBOW_TOL}): k={elbow_k}")
    ax.axvline(rfecv_n, color="#DD8452", linestyle="--", linewidth=1.2,
               label=f"RFECV's own optimum: n={rfecv_n}")
    ax.axhline(baseline, color="grey", linestyle=":", linewidth=1,
               label=f"All-30-feature baseline ({baseline:.4f})")

    ax.set_xlabel("Number of features (k)")
    ax.set_ylabel(f"Cross-validated {SCORING} (mean ± 1 std, {N_SPLITS}-fold)")
    ax.set_title("Performance vs. number of features -- the knee/elbow curve\n"
                  "(breast-cancer dataset, selection re-fit inside every CV fold)")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_xlim(0, 31)
    fig.tight_layout()

    out_path = ARTEFACTS_DIR / "feature_selection_knee_plot.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def filter_score_comparison(X: np.ndarray, y: np.ndarray, feature_names: list[str], k: int) -> pd.DataFrame:
    """f_classif vs. mutual_info_classif: same filter idea (score each feature alone,
    keep the top k), two different scoring functions -- f_classif assumes a linear
    relationship, mutual_info_classif can catch non-linear dependence but costs more to
    compute (NOTE-13)."""
    f_scores, _ = f_classif(X, y)
    mi_scores = mutual_info_classif(X, y, random_state=RNG_SEED)
    df = pd.DataFrame({"feature": feature_names, "f_classif": f_scores, "mutual_info": mi_scores})
    top_f = set(df.sort_values("f_classif", ascending=False).head(k)["feature"])
    top_mi = set(df.sort_values("mutual_info", ascending=False).head(k)["feature"])
    print(f"\n=== filter scoring functions: top-{k} agreement ===")
    print(f"f_classif top-{k}      : {sorted(top_f)}")
    print(f"mutual_info top-{k}    : {sorted(top_mi)}")
    print(f"agree on {len(top_f & top_mi)}/{k} features")
    return df


# --------------------------------------------------------------------------------------
# Section 4: the selection-leakage pitfall (LO4) -- pure noise, no real signal
# --------------------------------------------------------------------------------------


def selection_leakage_demo(
    y: np.ndarray, n_noise: int = 500, k: int = 10, n_repeats: int = 20
) -> dict:
    """Selection leakage, isolated from any real signal.

    Builds `n_noise` columns of pure Gaussian noise -- by construction, unrelated to y.
    Any "signal" SelectKBest finds in them is chance. Compares:
      WRONG: SelectKBest.fit(X_noise, y) on the WHOLE dataset once, THEN cross_val_score
             the classifier on those fixed columns. The selector has already looked at
             every row that will later play validation fold -- the classic leakage shape
             NOTE-13 warns about ("apply feature selection inside the outer CV loop").
      RIGHT: SelectKBest as a Pipeline step, refit on the training fold only, every fold.

    Repeated n_repeats times with independent noise draws so the result isn't one lucky
    (or unlucky) random matrix.
    """
    print(f"\n=== selection leakage demo: {n_noise} PURE NOISE features, "
          f"k={k}, {n_repeats} independent noise draws ===")
    n = len(y)
    cv = make_cv()
    majority_baseline = max(np.bincount(y)) / n

    wrong_scores, right_scores = [], []
    for noise_seed in range(n_repeats):
        rng = np.random.default_rng(noise_seed)
        X_noise = rng.normal(size=(n, n_noise))

        # WRONG: select using the whole dataset (including every future validation fold)
        selector = SelectKBest(score_func=f_classif, k=k).fit(X_noise, y)
        cols = selector.get_support(indices=True)
        wrong = cross_val_score(
            Pipeline([("scaler", StandardScaler()), ("clf", base_classifier())]),
            X_noise[:, cols], y, cv=cv, scoring=SCORING,
        )

        # RIGHT: selection is a pipeline step, refit on the training fold only, every fold
        right_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif, k=k)),
            ("clf", base_classifier()),
        ])
        right = cross_val_score(right_pipe, X_noise, y, cv=cv, scoring=SCORING)

        wrong_scores.append(wrong.mean())
        right_scores.append(right.mean())

    wrong_scores = np.array(wrong_scores)
    right_scores = np.array(right_scores)
    t_stat, p_value = stats.ttest_rel(wrong_scores, right_scores)

    print(f"majority-class baseline (true accuracy ceiling on pure noise): {majority_baseline:.4f}")
    print(f"WRONG (select outside CV) mean accuracy: {wrong_scores.mean():.4f} "
          f"(std {wrong_scores.std():.4f})")
    print(f"RIGHT (select inside CV)  mean accuracy: {right_scores.mean():.4f} "
          f"(std {right_scores.std():.4f})")
    print(f"paired t-test WRONG vs RIGHT: t={t_stat:.3f}, p={p_value:.3e}")
    print("The WRONG number looks like a working model on data that is, by construction, "
          "unpredictable -- pure phantom signal manufactured by letting the selector see "
          "rows it will later be scored on.")

    return {
        "majority_baseline": majority_baseline,
        "wrong_mean": float(wrong_scores.mean()),
        "right_mean": float(right_scores.mean()),
        "p_value": float(p_value),
    }


# --------------------------------------------------------------------------------------
# Section 5: wrapper methods -- RFE and Sequential Feature Selector, forward vs backward
# --------------------------------------------------------------------------------------


def rfe_selected_features(X: np.ndarray, y: np.ndarray, feature_names: list[str], k: int) -> np.ndarray:
    """Fit RFE on the FULL dataset -- this is the "what ships in the deployed model"
    use of selection (see the note in the chapter distinguishing this from the CV-safe
    *scoring* use above): once you've already validated the selection process (Section 3
    /4), fitting the final selector on all available data is the normal, correct way to
    decide the feature set you actually deploy."""
    Xs = StandardScaler().fit_transform(X)
    rfe = RFE(estimator=base_classifier(), n_features_to_select=k, step=1)
    rfe.fit(Xs, y)
    selected = np.array(feature_names)[rfe.support_]
    print(f"\n=== RFE (n_features_to_select={k}), fit on full data ===")
    print(f"selected: {sorted(selected.tolist())}")
    return rfe.support_


def sequential_selection_comparison(
    X: np.ndarray, y: np.ndarray, feature_names: list[str], k: int
) -> dict:
    """SequentialFeatureSelector forward vs. backward: which features each direction
    picks, how long each takes, and each one's HONEST cross-validated accuracy (selection
    re-run inside every outer fold, not reused from the full-data fit above)."""
    print(f"\n=== SequentialFeatureSelector: forward vs. backward (n_features_to_select={k}) ===")
    Xs = StandardScaler().fit_transform(X)
    results = {}
    outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RNG_SEED)

    for direction in ("forward", "backward"):
        t0 = time.perf_counter()
        sfs = SequentialFeatureSelector(
            estimator=base_classifier(),
            n_features_to_select=k,
            direction=direction,
            scoring=SCORING,
            cv=make_cv(),
        )
        sfs.fit(Xs, y)
        fit_seconds = time.perf_counter() - t0
        selected = np.array(feature_names)[sfs.support_]

        # Honest CV score: selection re-fit inside every outer fold (3-fold here, purely
        # to keep this nested-CV demo's wall-clock time reasonable for a chapter example --
        # a production run would use more folds/repeats).
        honest_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("select", SequentialFeatureSelector(
                estimator=base_classifier(), n_features_to_select=k,
                direction=direction, scoring=SCORING, cv=make_cv(),
            )),
            ("clf", base_classifier()),
        ])
        honest_scores = cross_val_score(honest_pipe, X, y, cv=outer_cv, scoring=SCORING)

        print(f"\n{direction.upper()}: fit_time={fit_seconds:.1f}s, "
              f"selected={sorted(selected.tolist())}")
        print(f"  honest nested-CV {SCORING} (selection redone per outer fold): "
              f"{honest_scores.mean():.4f} (std {honest_scores.std():.4f})")

        results[direction] = {
            "support": sfs.support_,
            "fit_seconds": fit_seconds,
            "honest_mean": float(honest_scores.mean()),
            "honest_std": float(honest_scores.std()),
        }

    overlap = np.logical_and(results["forward"]["support"], results["backward"]["support"]).sum()
    print(f"\nforward and backward agree on {overlap}/{k} of the selected features.")
    return results


def plot_forward_vs_backward(results: dict, baseline: float) -> Path:
    fig, (ax_score, ax_time) = plt.subplots(1, 2, figsize=(9.5, 4.5))

    directions = ["forward", "backward"]
    colors = ["#4C72B0", "#DD8452"]
    means = [results[d]["honest_mean"] for d in directions]
    stds = [results[d]["honest_std"] for d in directions]
    bars = ax_score.bar(directions, means, yerr=stds, capsize=8, color=colors, edgecolor="white")
    for bar, mean in zip(bars, means):
        ax_score.text(bar.get_x() + bar.get_width() / 2, mean + 0.005, f"{mean:.4f}",
                       ha="center", va="bottom", fontsize=9)
    ax_score.axhline(baseline, color="grey", linestyle=":", linewidth=1,
                      label=f"all-30-feature baseline ({baseline:.4f})")
    ax_score.set_ylabel(f"Honest nested-CV {SCORING}")
    ax_score.set_ylim(0.85, 1.0)
    ax_score.set_title("Selected-feature accuracy")
    ax_score.legend(fontsize=8, loc="lower right")

    times = [results[d]["fit_seconds"] for d in directions]
    ax_time.bar(directions, times, color=colors, edgecolor="white")
    for i, t in enumerate(times):
        ax_time.text(i, t + max(times) * 0.02, f"{t:.1f}s", ha="center", va="bottom", fontsize=9)
    ax_time.set_ylabel("Wall-clock fit time (seconds, single full-data fit)")
    ax_time.set_title("Selection cost")

    fig.suptitle("SequentialFeatureSelector: forward vs. backward")
    fig.tight_layout()

    out_path = ARTEFACTS_DIR / "feature_selection_forward_vs_backward.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------------------
# Section 6: embedded methods -- L1 (Lasso / L1-logistic) and tree importance
# --------------------------------------------------------------------------------------


def lasso_grounding_demo(X: np.ndarray, y: np.ndarray, feature_names: list[str], alpha: float = 0.05) -> None:
    """Grounds the literal `Lasso` API from NOTE-13. Lasso is a *regression* estimator, so
    this fits it to the 0/1 label treated as a numeric target (a "linear probability
    model") purely to show the mechanic -- L1 regularisation drives some coefficients to
    exactly zero. The chapter's real classification-embedded method (next function) uses
    LogisticRegression's L1 penalty instead, which optimises the actual classification
    loss."""
    Xs = StandardScaler().fit_transform(X)
    lasso = Lasso(alpha=alpha, random_state=RNG_SEED, max_iter=10_000)
    lasso.fit(Xs, y.astype(float))
    nonzero = np.array(feature_names)[lasso.coef_ != 0]
    print(f"\n=== Lasso(alpha={alpha}) on the 0/1 label as a numeric target (API grounding only) ===")
    print(f"{(lasso.coef_ != 0).sum()}/30 coefficients survive: {sorted(nonzero.tolist())}")


def embedded_l1_logistic(X: np.ndarray, y: np.ndarray, feature_names: list[str], C: float = 0.5) -> np.ndarray:
    """The real classification-embedded method: LogisticRegression with an L1 penalty.

    sklearn 1.9.0 DEPRECATED the `penalty=` parameter (removed in 1.10) -- passing
    `penalty='l1'` raises a FutureWarning and, worse, a second UserWarning about an
    inconsistent value with the default `l1_ratio=0.0`, and silently keeps behaving like
    the OLD default. The replacement is `l1_ratio=1.0` (matching the documented mapping:
    l1_ratio=0 <-> old penalty='l2', l1_ratio=1 <-> old penalty='l1') with a solver that
    supports it -- 'lbfgs' (the default) does not, so this needs solver='saga'
    (verified directly against the installed sklearn 1.9.0; NOTE-5's caveat flagged the
    deprecation, this is the concrete fix). See the chapter's Pitfalls section.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # prove this call raises no deprecation warning
        Xs = StandardScaler().fit_transform(X)
        clf = LogisticRegression(l1_ratio=1.0, C=C, solver="saga", max_iter=5000, random_state=RNG_SEED)
        clf.fit(Xs, y)

    sfm = SelectFromModel(clf, prefit=True)  # default threshold='mean' of |coef_|
    selected = np.array(feature_names)[sfm.get_support()]
    print(f"\n=== embedded: LogisticRegression(l1_ratio=1.0, C={C}, solver='saga') "
          f"+ SelectFromModel ===")
    print(f"{sfm.get_support().sum()}/30 features selected: {sorted(selected.tolist())}")
    return sfm.get_support()


def embedded_tree_importance(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> np.ndarray:
    rf = RandomForestClassifier(n_estimators=500, random_state=RNG_SEED, n_jobs=-1)
    rf.fit(X, y)
    sfm = SelectFromModel(rf, prefit=True, threshold="median")  # keeps the top half
    selected = np.array(feature_names)[sfm.get_support()]
    print(f"\n=== embedded: RandomForestClassifier.feature_importances_ "
          f"+ SelectFromModel(threshold='median') ===")
    print(f"{sfm.get_support().sum()}/30 features selected: {sorted(selected.tolist())}")
    return sfm.get_support()


# --------------------------------------------------------------------------------------
# Section 7: consolidated selected-feature table (LO2)
# --------------------------------------------------------------------------------------


def build_selected_feature_table(
    feature_names: list[str],
    filter_support: np.ndarray,
    rfe_support: np.ndarray,
    rfecv_support: np.ndarray,
    sfs_results: dict,
    l1_support: np.ndarray,
    rf_support: np.ndarray,
) -> pd.DataFrame:
    df = pd.DataFrame({
        "feature": feature_names,
        "filter_SelectKBest": filter_support,
        "wrapper_RFE": rfe_support,
        "wrapper_RFECV": rfecv_support,
        "wrapper_SFS_forward": sfs_results["forward"]["support"],
        "wrapper_SFS_backward": sfs_results["backward"]["support"],
        "embedded_L1_logistic": l1_support,
        "embedded_RF_importance": rf_support,
    })
    method_cols = [c for c in df.columns if c != "feature"]
    df["n_methods_selected"] = df[method_cols].sum(axis=1)
    df = df.sort_values("n_methods_selected", ascending=False).reset_index(drop=True)

    totals = {"feature": "TOTAL_SELECTED", **{c: int(df[c].sum()) for c in method_cols}}
    totals_row = pd.DataFrame([totals])
    return pd.concat([df, totals_row], ignore_index=True)


def filter_support_at_k(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """The filter method's final selected set, fit on the whole dataset -- the "what ships"
    use of selection, same distinction as rfe_selected_features()."""
    return SelectKBest(score_func=f_classif, k=k).fit(X, y).get_support()


def rfecv_support(X: np.ndarray, rfecv: RFECV) -> np.ndarray:
    return rfecv.support_


# --------------------------------------------------------------------------------------


def main() -> None:
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y, feature_names = load_data()
    print("=== dataset ===")
    print(f"X shape: {X.shape}, classes: {np.bincount(y).tolist()} (0=malignant, 1=benign)")

    baseline = baseline_score(X, y)

    # --- Section 3: filter knee curve + elbow ---
    filter_df = filter_knee_curve(X, y, max_k=X.shape[1])
    sensitivity = elbow_sensitivity_table(filter_df)
    print("\n=== elbow sensitivity: how the tolerance choice moves the elbow ===")
    print(sensitivity.to_string(index=False))
    elbow_k = find_elbow(filter_df["k"].to_numpy(), filter_df["mean_score"].to_numpy(), ELBOW_TOL)
    print(f"\nUsing tol={ELBOW_TOL} -> elbow_k={elbow_k} "
          f"(score {filter_df.loc[filter_df['k'] == elbow_k, 'mean_score'].iloc[0]:.4f}, "
          f"vs best {filter_df['mean_score'].max():.4f} at k={int(filter_df.loc[filter_df['mean_score'].idxmax(), 'k'])})")

    filter_score_comparison(X, y, feature_names, k=elbow_k)

    rfecv_df, rfecv_model = rfecv_curve(X, y)
    print(f"\n=== RFECV: its own optimum ===\nn_features_={rfecv_model.n_features_} "
          f"(best CV {SCORING}={rfecv_df['mean_score'].max():.4f})")

    knee_path = plot_knee_curve(filter_df, rfecv_df, elbow_k, rfecv_model.n_features_, baseline)

    # --- Section 4: selection leakage pitfall ---
    selection_leakage_demo(y)

    # --- Section 5: wrapper methods at elbow_k ---
    rfe_support = rfe_selected_features(X, y, feature_names, k=elbow_k)
    sfs_results = sequential_selection_comparison(X, y, feature_names, k=elbow_k)
    fwd_bwd_path = plot_forward_vs_backward(sfs_results, baseline)

    # --- Section 6: embedded methods ---
    lasso_grounding_demo(X, y, feature_names)
    l1_support = embedded_l1_logistic(X, y, feature_names)
    rf_support = embedded_tree_importance(X, y, feature_names)

    # --- Section 7: consolidated table ---
    filter_final_support = filter_support_at_k(X, y, elbow_k)
    table = build_selected_feature_table(
        feature_names,
        filter_final_support,
        rfe_support,
        rfecv_support(X, rfecv_model),
        sfs_results,
        l1_support,
        rf_support,
    )
    table_path = ARTEFACTS_DIR / "feature_selection_selected_features.csv"
    table.to_csv(table_path, index=False)
    print("\n=== consolidated selected-feature table (head) ===")
    print(table.head(10).to_string(index=False))

    print(f"\nWrote: {knee_path}")
    print(f"Wrote: {fwd_bwd_path}")
    print(f"Wrote: {table_path}")


if __name__ == "__main__":
    main()

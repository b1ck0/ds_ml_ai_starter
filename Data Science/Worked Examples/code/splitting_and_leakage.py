"""Train / validation / holdout splitting, and data leakage, on the Breast Cancer Wisconsin
(diagnostic) dataset.

Companion code for:
  Data Science/Worked Examples/train-valid-holdout-split.md

What it does (mirrors the chapter's sections):
  1. Loads sklearn's bundled load_breast_cancer() dataset (NOTE-5).
  2. A clean train/test split with train_test_split, and what stratify=y actually buys you.
  3. The leakage demo (LO3):
       a. Injects realistic missing values, then shows the *fitted preprocessing parameters*
          (SimpleImputer.statistics_, StandardScaler.mean_/scale_) really do differ depending
          on whether you fit on the whole dataset or on the training split alone -- the leak is
          not hypothetical, it's a different number.
       b. Shows that on one *normal-sized* split, that parameter drift is too small to move the
          reported accuracy at all (both pipelines score identically) -- the leak is invisible
          at this scale, which is exactly why it survives code review.
       c. Shrinks the training set to a small-pilot-study size (5% of the data) and repeats the
          split 200 times, revealing a small but statistically real, systematically optimistic
          bias from fitting on the whole dataset -- proven with a paired t-test, not eyeballing.
  4. k-fold cross-validation with cross_val_score + StratifiedKFold: what one holdout split
     can't tell you that k splits can.
  5. Three pitfall demos: leaking through duplicate rows, leaking through a feature that encodes
     the target, and (discussed in the chapter's prose only, no code needed) peeking at the
     holdout.
  6. Saves two artefacts to ../artefacts/: a bar chart of leaky vs. correct validation scores,
     and a schematic k-fold cross-validation diagram.

Environment (verified in research/NOTE-5-sklearn-core-apis.md and research/NOTE-2-package-versions.md,
checked 2026-09-02):
    scikit-learn==1.9.0, numpy==2.5.2, matplotlib==3.11.1, scipy==1.18.1
    Python 3.11+ (this script was run and gated on Python 3.13.7, matching the pinned versions
    with no substitutions).

Run:
    python splitting_and_leakage.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this script only saves figures, never shows them
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RNG_SEED = 42
ARTEFACTS_DIR = Path(__file__).resolve().parent.parent / "artefacts"

# Columns to make artificially missing, and the per-cell probability of being missing.
# Chosen to give every repeated-split experiment below a genuinely mixed numeric+imputation
# problem to solve, not a trivially clean matrix.
MISSING_COLUMNS = list(range(15))
MISSING_RATE = 0.5


def load_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Breast Cancer Wisconsin (diagnostic) -- bundled with scikit-learn, no download.

    569 rows, 30 numeric features, binary target (0=malignant, 1=benign), class counts
    [212, 357] (verified against the installed sklearn==1.9.0 in research/NOTE-5-sklearn-core-apis.md;
    loader confirmed by running it against this environment -- see the chapter's Environment section).
    [source: sklearn.datasets.load_breast_cancer docs](
    https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset)
    (checked 2026-09-02)
    """
    data = load_breast_cancer()
    return data.data.copy(), data.target.copy(), list(data.feature_names)


def inject_missing(X: np.ndarray, seed: int = RNG_SEED) -> np.ndarray:
    """MCAR missingness in MISSING_COLUMNS at MISSING_RATE, independent per cell."""
    rng = np.random.default_rng(seed)
    Xm = X.copy()
    mask = rng.random((Xm.shape[0], len(MISSING_COLUMNS))) < MISSING_RATE
    for i, col in enumerate(MISSING_COLUMNS):
        Xm[mask[:, i], col] = np.nan
    return Xm


# --------------------------------------------------------------------------------------
# Section 2: a clean split, and why stratify matters
# --------------------------------------------------------------------------------------


def stratify_demo(X: np.ndarray, y: np.ndarray) -> None:
    overall_counts = np.bincount(y)
    print("=== stratify demo (test_size=0.1, random_state=4) ===")
    print(f"overall class counts (malignant, benign): {overall_counts.tolist()} "
          f"({overall_counts / overall_counts.sum() * 100})")

    _, _, y_tr_ns, y_te_ns = train_test_split(X, y, test_size=0.1, random_state=4)
    _, _, y_tr_s, y_te_s = train_test_split(X, y, test_size=0.1, random_state=4, stratify=y)

    ns_counts = np.bincount(y_te_ns)
    s_counts = np.bincount(y_te_s)
    print(f"NO stratify  -> test class counts: {ns_counts.tolist()} "
          f"({np.round(ns_counts / ns_counts.sum() * 100, 1)}%)")
    print(f"stratify=y   -> test class counts: {s_counts.tolist()} "
          f"({np.round(s_counts / s_counts.sum() * 100, 1)}%)")


# --------------------------------------------------------------------------------------
# Section 3: the leakage demo
# --------------------------------------------------------------------------------------


def parameter_drift_demo(Xm: np.ndarray, y: np.ndarray) -> None:
    """Show the fitted imputer/scaler parameters really differ, whole-data vs train-only,
    and that -- on one normal-sized 75/25 split -- the resulting accuracy doesn't move."""
    print("\n=== leakage demo, part 1: do the fitted parameters actually differ? ===")

    Xtr, Xte, ytr, yte = train_test_split(Xm, y, test_size=0.25, random_state=42, stratify=y)

    imputer_train_only = SimpleImputer(strategy="mean").fit(Xtr)
    imputer_whole_data = SimpleImputer(strategy="mean").fit(Xm)  # LEAK: saw Xte too
    print("SimpleImputer.statistics_ (mean), first 3 columns with missing values:")
    print(f"  fit on TRAIN only : {np.round(imputer_train_only.statistics_[:3], 4)}")
    print(f"  fit on WHOLE data : {np.round(imputer_whole_data.statistics_[:3], 4)}  <- saw the "
          "test rows")

    scaler_train_only = StandardScaler().fit(imputer_train_only.transform(Xtr))
    scaler_whole_data = StandardScaler().fit(imputer_whole_data.transform(Xm))  # LEAK
    print("StandardScaler.scale_ (std-dev used to divide), same 3 columns:")
    print(f"  fit on TRAIN only : {np.round(scaler_train_only.scale_[:3], 4)}")
    print(f"  fit on WHOLE data : {np.round(scaler_whole_data.scale_[:3], 4)}  <- saw the test "
          "rows")

    # Correct pipeline: every transform is fit on Xtr only, then applied to Xte.
    correct_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000)),
    ])
    correct_pipe.fit(Xtr, ytr)
    correct_acc = correct_pipe.score(Xte, yte)

    # Leaky pipeline: impute + scale on the WHOLE dataset first, THEN split.
    Xm_imputed_leaky = SimpleImputer(strategy="mean").fit_transform(Xm)
    Xm_scaled_leaky = StandardScaler().fit_transform(Xm_imputed_leaky)
    Xtr_leaky, Xte_leaky, ytr_leaky, yte_leaky = train_test_split(
        Xm_scaled_leaky, y, test_size=0.25, random_state=42, stratify=y
    )
    leaky_clf = LogisticRegression(max_iter=2000).fit(Xtr_leaky, ytr_leaky)
    leaky_acc = leaky_clf.score(Xte_leaky, yte_leaky)

    print(f"\nResulting held-out accuracy on THIS ONE split:")
    print(f"  correct (fit-on-train-only) : {correct_acc:.6f}")
    print(f"  leaky   (fit-on-whole-data) : {leaky_acc:.6f}")
    print("  -> identical to 6 decimal places: the leak is real (the numbers above prove it)"
          " but invisible at this dataset size on this split.")


def repeated_small_train_leakage_demo(
    Xm: np.ndarray, y: np.ndarray, n_repeats: int = 200, test_size: float = 0.95
) -> dict:
    """Shrink the training set to ~5% of the data (a stand-in for an early pilot study with
    ~28 labelled patients) and repeat the split 200 times. This is where the same leak that
    was invisible above becomes a real, directionally consistent, statistically significant
    optimistic bias -- because a 28-row training sample's mean/std genuinely diverge from the
    full 569-row population's mean/std, and the leaky pipeline gets to use the population's."""
    print(f"\n=== leakage demo, part 2: same leak, {n_repeats} repeated small-train splits "
          f"(train={1 - test_size:.0%} of the data) ===")

    def leaky_once(seed: int) -> float:
        Xi = SimpleImputer(strategy="mean").fit_transform(Xm)  # LEAK: whole Xm
        Xs = StandardScaler().fit_transform(Xi)  # LEAK: whole Xm
        Xtr, Xte, ytr, yte = train_test_split(
            Xs, y, test_size=test_size, random_state=seed, stratify=y
        )
        clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
        return clf.score(Xte, yte)

    def correct_once(seed: int) -> float:
        Xtr, Xte, ytr, yte = train_test_split(
            Xm, y, test_size=test_size, random_state=seed, stratify=y
        )
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000)),
        ])
        pipe.fit(Xtr, ytr)
        return pipe.score(Xte, yte)

    leaky_scores = np.array([leaky_once(s) for s in range(n_repeats)])
    correct_scores = np.array([correct_once(s) for s in range(n_repeats)])

    gap = leaky_scores.mean() - correct_scores.mean()
    t_stat, p_value = stats.ttest_rel(leaky_scores, correct_scores)
    wins = int(np.sum(leaky_scores > correct_scores))
    ties = int(np.sum(leaky_scores == correct_scores))

    print(f"leaky   (fit-on-whole-data) mean accuracy: {leaky_scores.mean():.4f} "
          f"(std {leaky_scores.std():.4f})")
    print(f"correct (fit-on-train-only) mean accuracy: {correct_scores.mean():.4f} "
          f"(std {correct_scores.std():.4f})")
    print(f"mean gap (leaky - correct): {gap:+.4f}")
    print(f"paired t-test: t={t_stat:.3f}, p={p_value:.3e}")
    print(f"leaky scored strictly higher in {wins}/{n_repeats} repeats "
          f"({ties} ties, correct never won by more repeats than leaky)")

    return {
        "leaky_scores": leaky_scores,
        "correct_scores": correct_scores,
        "gap": gap,
        "p_value": p_value,
        "wins": wins,
        "ties": ties,
        "n_repeats": n_repeats,
    }


def plot_leaky_vs_correct(summary: dict) -> Path:
    leaky = summary["leaky_scores"]
    correct = summary["correct_scores"]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    means = [leaky.mean(), correct.mean()]
    stds = [leaky.std(), correct.std()]
    labels = ["Leaky\n(fit on whole dataset)", "Correct\n(Pipeline, fit on train only)"]
    colors = ["#C44E52", "#4C72B0"]

    bars = ax.bar(labels, means, yerr=stds, capsize=8, color=colors, edgecolor="white")
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.006, f"{mean:.4f}",
                ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Held-out accuracy")
    ax.set_ylim(0.85, 1.0)
    ax.set_title(
        f"Leaky vs. correct preprocessing\n"
        f"(mean +/- std over {summary['n_repeats']} small-train splits, paired p={summary['p_value']:.1e})"
    )
    fig.tight_layout()

    out_path = ARTEFACTS_DIR / "leaky_vs_correct_scores.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------------------
# Section 4: k-fold cross-validation
# --------------------------------------------------------------------------------------


def kfold_cv_demo(Xm: np.ndarray, y: np.ndarray, n_splits: int = 5) -> np.ndarray:
    print(f"\n=== k-fold cross-validation ({n_splits}-fold, StratifiedKFold) ===")
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000)),
    ])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RNG_SEED)
    scores = cross_val_score(pipe, Xm, y, cv=cv, scoring="accuracy")

    print(f"per-fold accuracy: {np.round(scores, 4).tolist()}")
    print(f"mean={scores.mean():.4f}, std={scores.std():.4f}, "
          f"range=[{scores.min():.4f}, {scores.max():.4f}]")
    print("Compare: a single 80/20 holdout only ever shows you ONE of these numbers -- you "
          "wouldn't know from one run whether you landed near 0.947 or 0.982.")
    return scores


def plot_cv_fold_diagram(n_splits: int = 5) -> Path:
    """Schematic (not data-driven) diagram of k-fold cross-validation: the holdout is set
    aside once, up front, and never touched by the CV loop; the remaining pool is split into
    n_splits folds, rotating which fold plays validation."""
    fig, ax = plt.subplots(figsize=(8, 4.8))

    train_color = "#4C72B0"
    val_color = "#DD8452"
    holdout_color = "#8C8C8C"

    total_width = 10.0
    holdout_width = total_width * 0.15
    pool_width = total_width - holdout_width
    fold_width = pool_width / n_splits

    top_y = n_splits + 1.3
    # "All data" bar at the top: pool vs. holdout, set aside before any CV happens.
    ax.add_patch(plt.Rectangle((0, top_y), pool_width, 0.8, facecolor="#DDDDDD",
                                edgecolor="white"))
    ax.text(pool_width / 2, top_y + 0.4, "Train + validation pool (used by CV)",
            ha="center", va="center", fontsize=9)
    ax.add_patch(plt.Rectangle((pool_width, top_y), holdout_width, 0.8, facecolor=holdout_color,
                                edgecolor="white"))
    ax.text(pool_width + holdout_width / 2, top_y + 0.4, "Holdout\n(untouched)",
            ha="center", va="center", fontsize=8, color="white")

    for fold in range(n_splits):
        row_y = n_splits - fold - 1
        for block in range(n_splits):
            x = block * fold_width
            color = val_color if block == fold else train_color
            ax.add_patch(plt.Rectangle((x, row_y), fold_width, 0.8, facecolor=color,
                                        edgecolor="white"))
        ax.text(-0.4, row_y + 0.4, f"Fold {fold + 1}", ha="right", va="center", fontsize=9)

    train_patch = plt.Rectangle((0, 0), 1, 1, facecolor=train_color)
    val_patch = plt.Rectangle((0, 0), 1, 1, facecolor=val_color)
    holdout_patch = plt.Rectangle((0, 0), 1, 1, facecolor=holdout_color)
    ax.legend([train_patch, val_patch, holdout_patch],
              ["train", "validation", "holdout (never touched during CV)"],
              loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=False, fontsize=9)

    ax.set_xlim(-1.6, total_width + 0.3)
    ax.set_ylim(-0.3, top_y + 1.3)
    ax.axis("off")
    ax.set_title(f"{n_splits}-fold cross-validation over the train+validation pool")
    fig.tight_layout()

    out_path = ARTEFACTS_DIR / "cv_fold_diagram.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------------------
# Section 5: pitfalls
# --------------------------------------------------------------------------------------


def duplicate_row_leakage_demo(X: np.ndarray, y: np.ndarray) -> None:
    """Duplicate ~15% of rows (simulating an ETL bug that double-exports some records), add
    noise features so the task isn't trivially easy, then show how many duplicate pairs land
    split across train/test by chance -- and the resulting accuracy bump from that leak."""
    print("\n=== pitfall: duplicate rows split across train and test ===")

    rng = np.random.default_rng(7)
    noise = rng.normal(size=(X.shape[0], 40))
    X_hard = np.hstack([X, noise])  # harder task: 1-NN no longer aces it trivially

    n = X_hard.shape[0]
    dup_idx = rng.choice(n, size=int(0.15 * n), replace=False)
    X_dup = np.vstack([X_hard, X_hard[dup_idx]])
    y_dup = np.concatenate([y, y[dup_idx]])
    original_row_id = np.concatenate([np.arange(n), dup_idx])

    Xtr, Xte, ytr, yte, id_tr, id_te = train_test_split(
        X_dup, y_dup, original_row_id, test_size=0.25, random_state=0, stratify=y_dup
    )
    split_across = set(id_tr.tolist()) & set(id_te.tolist())
    print(f"duplicated {len(dup_idx)} rows; {len(split_across)} of them ended up with one copy "
          f"in train and the other in test, purely from a random shuffle.")

    leaky_pipe = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(1))])
    leaky_pipe.fit(Xtr, ytr)
    leaky_acc = leaky_pipe.score(Xte, yte)

    X_unique, first_seen = np.unique(X_hard, axis=0, return_index=True)
    order = np.sort(first_seen)
    Xu, yu = X_hard[order], y[order]
    Xtr2, Xte2, ytr2, yte2 = train_test_split(Xu, yu, test_size=0.25, random_state=0, stratify=yu)
    clean_pipe = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(1))])
    clean_pipe.fit(Xtr2, ytr2)
    clean_acc = clean_pipe.score(Xte2, yte2)

    print(f"1-NN accuracy WITH duplicate-row leak   : {leaky_acc:.4f}")
    print(f"1-NN accuracy deduplicated BEFORE split : {clean_acc:.4f}")


def target_leakage_demo(X: np.ndarray, y: np.ndarray) -> None:
    """A feature that is really the target in disguise (e.g. a field only populated after
    the diagnosis was confirmed) makes cross-validated accuracy shoot toward 100% -- the
    'too good to be true' smell test."""
    print("\n=== pitfall: a feature that leaks the target ===")

    leaked_column = y.astype(float).reshape(-1, 1)
    X_with_leak = np.hstack([X, leaked_column])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG_SEED)
    honest_scores = cross_val_score(
        Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))]),
        X, y, cv=cv, scoring="accuracy",
    )
    leaked_scores = cross_val_score(
        Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))]),
        X_with_leak, y, cv=cv, scoring="accuracy",
    )

    print(f"honest (30 real features)       CV accuracy: {np.round(honest_scores, 4).tolist()}, "
          f"mean={honest_scores.mean():.4f}")
    print(f"with target-leaking column       CV accuracy: {np.round(leaked_scores, 4).tolist()}, "
          f"mean={leaked_scores.mean():.4f}")
    print("Near-perfect CV accuracy on a messy real-world problem is a red flag, not a win -- "
          "go find the leaked column before you celebrate.")


# --------------------------------------------------------------------------------------


def main() -> None:
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y, feature_names = load_data()
    print("=== dataset ===")
    print(f"X shape: {X.shape}, classes: {np.bincount(y).tolist()} "
          f"(0=malignant, 1=benign), first 3 features: {feature_names[:3]}")

    stratify_demo(X, y)

    Xm = inject_missing(X)
    print(f"\ninjected missingness: {int(np.isnan(Xm).sum())} / {Xm.size} cells "
          f"({np.isnan(Xm).sum() / Xm.size * 100:.1f}%) across columns {MISSING_COLUMNS}")

    parameter_drift_demo(Xm, y)
    leak_summary = repeated_small_train_leakage_demo(Xm, y)
    bar_path = plot_leaky_vs_correct(leak_summary)

    kfold_cv_demo(Xm, y)
    diagram_path = plot_cv_fold_diagram()

    duplicate_row_leakage_demo(X, y)
    target_leakage_demo(X, y)

    print(f"\nWrote: {bar_path}")
    print(f"Wrote: {diagram_path}")


if __name__ == "__main__":
    main()

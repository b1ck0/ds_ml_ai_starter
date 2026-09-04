"""Trustworthy probabilities on imbalanced data: OOT validation, Brier, precision@top-N,
and isotonic calibration.

Companion code for:
  Data Science/Worked Examples/15-calibration-ranking-imbalanced.md

What it does (mirrors the chapter's section order):
  1. Builds a synthetic ~2%-positive dataset with sklearn.datasets.make_classification, a
     synthetic timestamp (0-100 "days"), and mild engineered temporal drift on one feature
     (research/NOTE-DS-20-7-dataset-choice.md).
  2. Section 3 -- out-of-time (OOT) validation: trains the SAME plain LogisticRegression on
     (a) a random 80/20 split and (b) an out-of-time split (train day<=80, test day>80), and
     compares their metrics on their own held-out fold -- the OOT numbers are honestly lower
     because of the drift.
  3. Section 4 -- Brier score (by hand on a tiny example, then sklearn.metrics.brier_score_loss;
     the Brier skill score) and precision@top-N / lift, computed on the OOT model's honest
     test-window predictions (research/NOTE-DS-20-2, NOTE-DS-20-4).
  4. Section 5 -- trains a SECOND model with undersampling (day<=60 only, reusing DS-8's
     random-undersampling technique) and shows it is badly miscalibrated on the true-prevalence
     test window: reliability diagram far off the diagonal, mean predicted probability far above
     the true rate. Illustrates the King & Zeng (2001) analytic prior-correction intercept shift
     (research/NOTE-DS-20-6-king-zeng-prior-correction.md).
  5. Section 6 -- fixes it empirically: fits IsotonicRegression and a manual Platt/sigmoid fit
     on a TRUE-PREVALENCE, out-of-time calibration window (day 60-80) that the classifier never
     trained on; also demonstrates "the trap" (CalibratedClassifierCV fit on the resampled
     training data instead) still being miscalibrated. Shows Brier improving sharply while
     precision@top-N is preserved by Platt exactly (a strictly monotonic transform is injective)
     and by isotonic up to small wobbles from its step function's ties at sparse extremes
     (research/NOTE-DS-20-3, NOTE-DS-20-5-sklearn-api.md).
  6. Writes all artefacts (reliability diagrams, OOT-vs-random bar chart, precision@N/lift
     curve, one combined metrics table) to ../artefacts/, namespaced calib_*.

Environment (verified in research/NOTE-DS-20-1-package-versions.md, checked 2026-09-04):
    numpy==2.5.2, pandas==3.0.5, matplotlib==3.11.1, scikit-learn==1.9.0
    Python 3.12+ (this script was run and gated on Python 3.13.7).

Run:
    python calibration_ranking.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this script only saves figures, never shows them
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.datasets import make_classification
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RNG_SEED = 42
ARTEFACTS_DIR = Path(__file__).resolve().parent.parent / "artefacts"

# --- Dataset shape (research/NOTE-DS-20-7-dataset-choice.md) ---------------------------------
N_SAMPLES = 100_000
N_FEATURES = 20
N_INFORMATIVE = 8
N_REDUNDANT = 4
POSITIVE_RATE = 0.02          # ~2% positive -- credit-default / fraud / churn territory
CLASS_SEP = 1.3
FLIP_Y = 0.01
TIME_RANGE_DAYS = 100.0
DRIFT_FEATURE_IDX = list(range(N_INFORMATIVE))  # all 8 informative features (shuffle=False
                                                 # keeps informative columns first) get noisier
DRIFT_NOISE_MAX_STD = 3.0      # by day 100, each drifting feature has this much extra
                                # zero-mean Gaussian noise added on top of its signal

# --- Chronological windows ---------------------------------------------------------------------
# Section 3-4 baseline model: trained on everything up to day 80 (no calibration reserve needed).
FULL_TRAIN_CUTOFF_DAY = 80.0
# Section 5-6 undersampled model: trained on a SMALLER window (day<=60) so that 60-80 can be
# reserved, untouched, as a true-prevalence calibration hold-out.
CLASSIFIER_TRAIN_CUTOFF_DAY = 60.0
# Every model's FINAL, honest evaluation happens on day>80 -- never touched until the very end.


def make_dataset() -> pd.DataFrame:
    """~2% positive, fully synthetic, reproducible, with a synthetic timestamp and drift.

    make_classification gives full control over the true prevalence (needed later to show
    exactly how undersampling distorts it). timestamp_day ~ Uniform(0, 100) stands in for
    100 days of production traffic.

    The drift itself is CONCEPT drift, not a cosmetic rescale: each feature in
    DRIFT_FEATURE_IDX picks up independent zero-mean Gaussian noise whose standard
    deviation grows linearly with time, reaching DRIFT_NOISE_MAX_STD by day 100. Adding
    noise strictly increases within-class variance without touching the between-class
    means, which can only shrink the achievable separation between classes -- unlike a
    multiplicative rescale (which StandardScaler mostly undoes) or a mean shift shared by
    both classes (which a linear boundary can absorb), added noise is a textbook, provably
    one-directional way to make the classes harder to tell apart. Late rows are noisier;
    early rows are close to their original, cleanly-separated values -- a classifier fit on
    the (mostly clean) past cannot see this coming, and is honestly worse on the (noisier)
    future.
    """
    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=N_INFORMATIVE,
        n_redundant=N_REDUNDANT,
        n_clusters_per_class=1,
        weights=[1 - POSITIVE_RATE, POSITIVE_RATE],
        flip_y=FLIP_Y,
        class_sep=CLASS_SEP,
        shuffle=False,  # keep the first N_INFORMATIVE columns as the informative ones,
                         # so DRIFT_FEATURE_IDX reliably targets real signal, not noise
        random_state=RNG_SEED,
    )

    rng = np.random.default_rng(RNG_SEED)
    timestamp_day = rng.uniform(0.0, TIME_RANGE_DAYS, size=N_SAMPLES)

    noise_std = DRIFT_NOISE_MAX_STD * (timestamp_day / TIME_RANGE_DAYS)
    for idx in DRIFT_FEATURE_IDX:
        X[:, idx] = X[:, idx] + rng.normal(0.0, 1.0, size=N_SAMPLES) * noise_std

    feature_cols = [f"feat_{i}" for i in range(N_FEATURES)]
    df = pd.DataFrame(X, columns=feature_cols)
    df["timestamp_day"] = timestamp_day
    df["y"] = y
    return df.sort_values("timestamp_day").reset_index(drop=True)


FEATURE_COLS = [f"feat_{i}" for i in range(N_FEATURES)]


def report_class_balance(df: pd.DataFrame) -> None:
    n_pos = int(df["y"].sum())
    print("=== dataset ===")
    print(f"{len(df)} rows, {n_pos} positive ({n_pos / len(df):.3%}), "
          f"timestamp_day range [{df['timestamp_day'].min():.1f}, {df['timestamp_day'].max():.1f}]")


# ==================================================================================================
# Section 3 -- out-of-time validation vs a random split
# ==================================================================================================

def fit_plain_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """StandardScaler + LogisticRegression, no resampling -- the baseline used for the
    OOT-vs-random comparison (Section 3) and reused as the honest model in Section 4."""
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=RNG_SEED)),
    ])
    model.fit(X_train, y_train)
    return model


def evaluate_split(name: str, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    base_rate = float(y_test.mean())
    bs = brier_score_loss(y_test, y_proba)
    bs_ref = base_rate * (1 - base_rate)
    bss = 1 - bs / bs_ref if bs_ref > 0 else float("nan")
    metrics = {
        "split": name,
        "n_test": len(y_test),
        "base_rate": base_rate,
        "recall@0.5": recall_score(y_test, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_test, y_proba),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "brier": bs,
        "bss": bss,
    }
    print(f"{name:22s} n={metrics['n_test']:6d} base_rate={base_rate:.4f} "
          f"recall@0.5={metrics['recall@0.5']:.4f} PR-AUC={metrics['pr_auc']:.4f} "
          f"ROC-AUC={metrics['roc_auc']:.4f} Brier={bs:.5f} BSS={bss:.4f}")
    return metrics


def oot_vs_random(df: pd.DataFrame) -> tuple[dict, dict, Pipeline, pd.DataFrame, pd.Series]:
    """Trains ONE model and evaluates it TWO ways: a naive random holdout carved from the
    same training-era pool it was fit on, vs a genuine out-of-time holdout from the future.

    Both holdouts are the same size class and drawn to the same 80/20 proportion, and neither
    was seen during training -- the only thing that differs is WHEN the holdout rows are from.
    "Random split" here means: take the day<=80 pool (the only data available to train on),
    carve out a random 20% of IT as a validation set (what a team that never looked at the
    timestamp column would naturally do), fit on the rest, and validate on that random slice.
    "OOT split" means: validate the SAME fitted model on day>80 -- rows that did not exist yet
    when the model was trained, exactly like a deployed model scores tomorrow's rows.
    """
    print("\n=== Section 3: same trained model, naive random holdout vs genuine OOT holdout ===")

    pool = df[df["timestamp_day"] <= FULL_TRAIN_CUTOFF_DAY]
    oot_test = df[df["timestamp_day"] > FULL_TRAIN_CUTOFF_DAY]

    X_pool, y_pool = pool[FEATURE_COLS], pool["y"]
    X_train, X_random_holdout, y_train, y_random_holdout = train_test_split(
        X_pool, y_pool, test_size=0.2, stratify=y_pool, random_state=RNG_SEED
    )
    model = fit_plain_model(X_train, y_train)

    random_metrics = evaluate_split("random holdout (day<=80)", model,
                                     X_random_holdout, y_random_holdout)
    oot_metrics = evaluate_split(f"OOT holdout (day>{FULL_TRAIN_CUTOFF_DAY:.0f})", model,
                                  oot_test[FEATURE_COLS], oot_test["y"])

    pr_auc_drop = (random_metrics["pr_auc"] - oot_metrics["pr_auc"]) / random_metrics["pr_auc"]
    bss_drop = (random_metrics["bss"] - oot_metrics["bss"]) / random_metrics["bss"]
    print(f"Same fitted model. OOT PR-AUC is {pr_auc_drop:.1%} lower than the random-holdout "
          f"PR-AUC; OOT BSS is {bss_drop:.1%} lower than the random-holdout BSS -- the honest "
          f"cost of validating on genuinely unseen future rows instead of a random slice of "
          f"the training-era pool. (Raw Brier is NOT a fair side-by-side number here: the two "
          f"holdouts have slightly different base rates, and Brier is sensitive to base rate; "
          f"BSS corrects for that by dividing out each set's own BS_ref, which is exactly why "
          f"NOTE-DS-20-2 insists on using BSS, not raw Brier, whenever the sets being compared "
          f"don't share a base rate.)")

    return random_metrics, oot_metrics, model, oot_test[FEATURE_COLS], oot_test["y"]


def plot_oot_vs_random(random_metrics: dict, oot_metrics: dict, out_name: str) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    # Brier itself is left out here on purpose: the two test sets have slightly different
    # base rates, and raw Brier is not comparable across differing base rates (NOTE-DS-20-2)
    # -- BSS is the fair, base-rate-normalised stand-in.
    metric_keys = [("pr_auc", "PR-AUC"), ("bss", "Brier Skill Score (higher is better)"),
                   ("recall@0.5", "Recall @ 0.5")]
    for ax, (key, label) in zip(axes, metric_keys):
        vals = [random_metrics[key], oot_metrics[key]]
        bars = ax.bar(["random holdout", "OOT holdout"], vals,
                       color=["#4C72B0", "#C44E52"])
        ax.set_title(label, fontsize=10)
        ax.bar_label(bars, fmt="%.3f")
        ax.set_ylim(0, max(vals) * 1.25)
    fig.suptitle("Same trained model: naive random holdout vs genuine out-of-time holdout")
    fig.tight_layout()
    out_path = ARTEFACTS_DIR / out_name
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ==================================================================================================
# Section 4 -- Brier score and precision@top-N
# ==================================================================================================

def brier_by_hand() -> None:
    """A tiny 5-prediction example: compute Brier by hand, then verify against
    sklearn.metrics.brier_score_loss (signature: research/NOTE-DS-20-5-sklearn-api.md)."""
    print("\n=== Section 4a: Brier score by hand ===")
    y_true = np.array([0, 1, 1, 0, 1])
    y_proba = np.array([0.1, 0.8, 0.9, 0.2, 0.7])
    sq_errors = (y_proba - y_true) ** 2
    by_hand = sq_errors.mean()
    from_sklearn = brier_score_loss(y_true, y_proba)
    print(f"y_true  = {y_true.tolist()}")
    print(f"y_proba = {y_proba.tolist()}")
    print(f"squared errors = {np.round(sq_errors, 4).tolist()}")
    print(f"by hand: mean = {by_hand:.5f}   brier_score_loss() = {from_sklearn:.5f}")
    assert abs(by_hand - from_sklearn) < 1e-9, "hand computation does not match sklearn"


def brier_skill_score(y_true: pd.Series, y_proba: np.ndarray) -> tuple[float, float, float]:
    """BSS = 1 - BS / BS_ref, BS_ref = base_rate * (1 - base_rate) -- ALWAYS using the true
    prevalence of the set actually being scored (research/NOTE-DS-20-2-brier-score-definitions.md)."""
    base_rate = float(y_true.mean())
    bs = brier_score_loss(y_true, y_proba)
    bs_ref = base_rate * (1 - base_rate)
    bss = 1 - bs / bs_ref if bs_ref > 0 else float("nan")
    return bs, bs_ref, bss


def precision_at_n(y_true: np.ndarray, y_score: np.ndarray, n: int) -> tuple[float, int]:
    """precision@N = TP in the top N ranked rows / N (research/NOTE-DS-20-4-precision-lift.md)."""
    order = np.argsort(-y_score)
    top_idx = order[:n]
    tp = int(y_true[top_idx].sum())
    return tp / n, tp


def precision_and_lift_table(y_true: pd.Series, y_proba: np.ndarray,
                              ns: list[int]) -> pd.DataFrame:
    y_arr = y_true.to_numpy()
    base_rate = float(y_true.mean())
    rows = []
    for n in ns:
        precision, tp = precision_at_n(y_arr, y_proba, n)
        lift = precision / base_rate if base_rate > 0 else float("nan")
        rows.append({"N": n, "precision@N": precision, "TP": tp, "lift@N": lift})
    return pd.DataFrame(rows)


def measuring_what_matters(oot_test_X: pd.DataFrame, oot_test_y: pd.Series,
                            model_oot: Pipeline) -> dict:
    print("\n=== Section 4: precision@top-N and lift (the honest OOT test window) ===")
    y_proba = model_oot.predict_proba(oot_test_X)[:, 1]
    base_rate = float(oot_test_y.mean())

    bs, bs_ref, bss = brier_skill_score(oot_test_y, y_proba)
    print(f"OOT test base rate = {base_rate:.4f}")
    print(f"Brier = {bs:.5f}   BS_ref (always-predict-base-rate) = {bs_ref:.5f}   BSS = {bss:.4f}")

    ns = [50, 100, 200, 500, 1000]
    table = precision_and_lift_table(oot_test_y, y_proba, ns)
    print(table.to_string(index=False))

    return {"y_proba": y_proba, "base_rate": base_rate, "brier": bs, "bss": bss,
            "precision_table": table}


def plot_precision_at_n_curve(y_true: pd.Series, y_proba: np.ndarray, out_name: str) -> Path:
    y_arr = y_true.to_numpy()
    base_rate = float(y_true.mean())
    ns = np.arange(10, 3001, 10)
    precisions = np.array([precision_at_n(y_arr, y_proba, int(n))[0] for n in ns])
    lifts = precisions / base_rate

    fig, ax1 = plt.subplots(figsize=(7.5, 5))
    ax1.plot(ns, precisions, color="#4C72B0", linewidth=2, label="precision@N")
    ax1.axhline(base_rate, color="grey", linestyle="--", linewidth=1,
                label=f"base rate ({base_rate:.3f})")
    ax1.set_xlabel("N (action budget -- top N ranked rows)")
    ax1.set_ylabel("precision@N", color="#4C72B0")
    ax1.tick_params(axis="y", labelcolor="#4C72B0")

    ax2 = ax1.twinx()
    ax2.plot(ns, lifts, color="#55A868", linewidth=1.5, linestyle=":", label="lift@N")
    ax2.set_ylabel("lift@N (precision@N / base rate)", color="#55A868")
    ax2.tick_params(axis="y", labelcolor="#55A868")

    fig.suptitle("precision@N and lift@N vs action budget N (OOT test window)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
    fig.tight_layout()

    out_path = ARTEFACTS_DIR / out_name
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ==================================================================================================
# Section 5 -- why the probabilities lie (undersampling breaks calibration)
# ==================================================================================================

def undersample(X: pd.DataFrame, y: pd.Series, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.Series]:
    """Random undersampling of the majority class down to a 1:1 ratio -- the same technique
    DS-8 uses via imbalanced-learn's RandomUnderSampler, implemented here with a plain pandas
    sample so this chapter does not introduce a new pinned dependency."""
    pos_idx = y[y == 1].index
    neg_idx = y[y == 0].index
    neg_sample_idx = rng.choice(neg_idx, size=len(pos_idx), replace=False)
    keep_idx = np.concatenate([pos_idx.to_numpy(), neg_sample_idx])
    rng.shuffle(keep_idx)
    return X.loc[keep_idx], y.loc[keep_idx]


def fit_undersampled_model(classifier_train: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame, pd.Series]:
    print(f"\n=== Section 5: undersampled classifier (trained on day<={CLASSIFIER_TRAIN_CUTOFF_DAY:.0f}) ===")
    rng = np.random.default_rng(RNG_SEED)
    X_train_full = classifier_train[FEATURE_COLS]
    y_train_full = classifier_train["y"]
    print(f"before undersampling: {len(y_train_full)} rows, {int(y_train_full.sum())} positive "
          f"({y_train_full.mean():.4%})")

    X_train_us, y_train_us = undersample(X_train_full, y_train_full, rng)
    print(f"after undersampling:  {len(y_train_us)} rows, {int(y_train_us.sum())} positive "
          f"({y_train_us.mean():.4%})")

    model_us = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=RNG_SEED)),
    ])
    model_us.fit(X_train_us, y_train_us)
    return model_us, X_train_us, y_train_us


def show_miscalibration(model_us: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_proba = model_us.predict_proba(X_test)[:, 1]
    base_rate = float(y_test.mean())
    mean_pred = float(y_proba.mean())
    bs, bs_ref, bss = brier_skill_score(y_test, y_proba)
    print(f"\nOn the TRUE-PREVALENCE test window (day>{FULL_TRAIN_CUTOFF_DAY:.0f}, "
          f"base rate={base_rate:.4f}):")
    print(f"  mean predicted probability = {mean_pred:.4f}  "
          f"(should be close to {base_rate:.4f} if calibrated -- it is not)")
    print(f"  Brier = {bs:.5f}   BSS = {bss:.4f}")
    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10, strategy="quantile")
    print(f"  reliability curve (predicted, observed) pairs, equal-count bins: "
          f"{list(zip(np.round(prob_pred, 3), np.round(prob_true, 3)))}")
    return {"y_proba": y_proba, "mean_pred": mean_pred, "base_rate": base_rate,
            "brier": bs, "bss": bss, "prob_true": prob_true, "prob_pred": prob_pred}


def king_zeng_illustration(model_us: Pipeline, y_train_us: pd.Series, true_prevalence: float) -> None:
    """Illustrates the King & Zeng (2001) analytic prior-correction intercept shift on the
    fitted logistic regression -- intuition only, not the chapter's primary fix
    (research/NOTE-DS-20-6-king-zeng-prior-correction.md)."""
    print("\n--- King & Zeng (2001) prior-correction, illustrative ---")
    clf = model_us.named_steps["clf"]
    raw_intercept = float(clf.intercept_[0])
    sample_prevalence = float(y_train_us.mean())  # ~0.5 after undersampling

    def logit(p: float) -> float:
        return np.log(p / (1 - p))

    corrected_intercept = raw_intercept + logit(true_prevalence) - logit(sample_prevalence)
    print(f"raw (uncorrected) intercept beta_0        = {raw_intercept:.4f}  "
          f"(learned at sample prevalence {sample_prevalence:.4f})")
    print(f"corrected intercept beta_0_corrected       = {corrected_intercept:.4f}  "
          f"(shifted to true prevalence {true_prevalence:.4f})")
    print(f"intercept shift = {corrected_intercept - raw_intercept:.4f} log-odds "
          "(this only corrects the intercept -- it assumes resampling changed nothing else "
          "about the decision boundary, which is why this chapter's fix is the empirical "
          "calibrator instead, applicable to any model family).")


# ==================================================================================================
# Section 6 -- isotonic calibration on true-prevalence, out-of-time data
# ==================================================================================================

def the_trap(X_train_us: pd.DataFrame, y_train_us: pd.Series,
             X_test: pd.DataFrame, y_test: pd.Series) -> np.ndarray:
    """CalibratedClassifierCV fit on the RESAMPLED training data -- still wrong, because the
    calibrator only ever sees the 50/50 prevalence, never the true one
    (research/NOTE-DS-20-3-calibration-isotonic-platt.md, research/NOTE-DS-20-5-sklearn-api.md)."""
    print("\n=== Section 6a: THE TRAP -- calibrating on the resampled training data ===")
    trap_model = CalibratedClassifierCV(
        Pipeline([("scaler", StandardScaler()),
                  ("clf", LogisticRegression(max_iter=2000, random_state=RNG_SEED))]),
        method="isotonic", cv=5,
    )
    trap_model.fit(X_train_us, y_train_us)
    y_proba_trap = trap_model.predict_proba(X_test)[:, 1]
    bs, _, bss = brier_skill_score(y_test, y_proba_trap)
    print(f"CalibratedClassifierCV(method='isotonic', cv=5).fit(X_train_UNDERSAMPLED, ...): "
          f"mean predicted proba={y_proba_trap.mean():.4f} (true base rate={y_test.mean():.4f}), "
          f"Brier={bs:.5f}, BSS={bss:.4f} -- still badly miscalibrated. Calling '.fit' with "
          "method='isotonic' didn't fix anything, because the calibrator was still shown only "
          "the resampled 50/50 data.")
    return y_proba_trap


def calibrate_on_true_prevalence(raw_scores_calibrate: np.ndarray, y_calibrate: pd.Series,
                                  raw_scores_test: np.ndarray) -> tuple[np.ndarray, IsotonicRegression]:
    """Isotonic regression fit on 1-D raw scores vs binary outcomes -- exactly the
    IsotonicRegression.fit(scores, y) contract (research/NOTE-DS-20-5-sklearn-api.md)."""
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(raw_scores_calibrate, y_calibrate.to_numpy())
    y_proba_iso = iso.predict(raw_scores_test)
    return y_proba_iso, iso


def platt_on_true_prevalence(raw_scores_calibrate: np.ndarray, y_calibrate: pd.Series,
                              raw_scores_test: np.ndarray) -> np.ndarray:
    """Platt/sigmoid scaling: fit a 1-feature logistic regression of the outcome on the raw
    score -- this is the exact mechanism CalibratedClassifierCV(method='sigmoid') automates
    (research/NOTE-DS-20-3-calibration-isotonic-platt.md)."""
    platt = LogisticRegression(max_iter=2000, random_state=RNG_SEED)
    platt.fit(raw_scores_calibrate.reshape(-1, 1), y_calibrate.to_numpy())
    y_proba_platt = platt.predict_proba(raw_scores_test.reshape(-1, 1))[:, 1]
    return y_proba_platt


def fix_calibration(model_us: Pipeline, calibrate_df: pd.DataFrame,
                     X_test: pd.DataFrame, y_test: pd.Series, raw_result: dict) -> dict:
    print(f"\n=== Section 6b: fitting isotonic + Platt on the TRUE-PREVALENCE calibration "
          f"window (day {CLASSIFIER_TRAIN_CUTOFF_DAY:.0f}-{FULL_TRAIN_CUTOFF_DAY:.0f}) ===")
    X_calibrate = calibrate_df[FEATURE_COLS]
    y_calibrate = calibrate_df["y"]
    print(f"calibration window: {len(y_calibrate)} rows, {int(y_calibrate.sum())} positive "
          f"({y_calibrate.mean():.4%}) -- true prevalence, never touched during classifier training")

    raw_scores_calibrate = model_us.predict_proba(X_calibrate)[:, 1]
    raw_scores_test = model_us.predict_proba(X_test)[:, 1]

    y_proba_iso, iso = calibrate_on_true_prevalence(raw_scores_calibrate, y_calibrate, raw_scores_test)
    bs_iso, _, bss_iso = brier_skill_score(y_test, y_proba_iso)
    print(f"isotonic:  mean predicted proba={y_proba_iso.mean():.4f}  "
          f"Brier={bs_iso:.5f}  BSS={bss_iso:.4f}")

    y_proba_platt = platt_on_true_prevalence(raw_scores_calibrate, y_calibrate, raw_scores_test)
    bs_platt, _, bss_platt = brier_skill_score(y_test, y_proba_platt)
    print(f"platt:     mean predicted proba={y_proba_platt.mean():.4f}  "
          f"Brier={bs_platt:.5f}  BSS={bss_platt:.4f}")

    print(f"\nBrier improvement vs raw/uncalibrated ({raw_result['brier']:.5f}):")
    print(f"  isotonic: {raw_result['brier']:.5f} -> {bs_iso:.5f} "
          f"({(1 - bs_iso / raw_result['brier']):.1%} lower)")
    print(f"  platt:    {raw_result['brier']:.5f} -> {bs_platt:.5f} "
          f"({(1 - bs_platt / raw_result['brier']):.1%} lower)")

    # Ranking check: a STRICTLY monotonic transform (Platt's sigmoid) preserves precision@N
    # exactly -- it is injective, so it can never reorder two distinct raw scores. Isotonic
    # regression is only WEAKLY monotonic (non-decreasing): its step function can map several
    # distinct raw scores to the identical calibrated probability, and precision@N can wobble
    # if the cut at N falls inside one of those tied plateaus. NOTE-DS-20-3 documents exactly
    # this failure mode ("jagged plateaus at the extremes... where data is sparse") -- watch
    # for it especially at small N, where a handful of tied top scores matter a lot.
    ns = [50, 100, 200, 500, 1000]
    y_arr = y_test.to_numpy()
    table = pd.DataFrame({
        "N": ns,
        "precision@N raw": [precision_at_n(y_arr, raw_result["y_proba"], n)[0] for n in ns],
        "precision@N isotonic": [precision_at_n(y_arr, y_proba_iso, n)[0] for n in ns],
        "precision@N platt": [precision_at_n(y_arr, y_proba_platt, n)[0] for n in ns],
    })
    print("\nprecision@N: raw vs isotonic vs Platt -- Platt preserves ranking exactly; "
          "isotonic can wobble at small N where its step function ties raw scores together:")
    print(table.to_string(index=False))

    return {
        "y_proba_iso": y_proba_iso, "y_proba_platt": y_proba_platt,
        "brier_iso": bs_iso, "bss_iso": bss_iso,
        "brier_platt": bs_platt, "bss_platt": bss_platt,
        "precision_table": table,
    }


def plot_reliability_diagrams(y_test: pd.Series, raw_proba: np.ndarray, trap_proba: np.ndarray,
                               iso_proba: np.ndarray, platt_proba: np.ndarray,
                               out_name: str) -> Path:
    panels = [
        ("(a) before -- raw / uncalibrated", raw_proba, "#C44E52"),
        ("(b) the trap -- calibrated on resampled data", trap_proba, "#DD8452"),
        ("(c) after -- isotonic on true-prevalence hold-out", iso_proba, "#55A868"),
        ("(d) after -- Platt/sigmoid on true-prevalence hold-out", platt_proba, "#4C72B0"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, (title, y_proba, color) in zip(axes.flat, panels):
        prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10, strategy="quantile")
        bs = brier_score_loss(y_test, y_proba)
        ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, label="perfectly calibrated")
        ax.plot(prob_pred, prob_true, marker="o", color=color, linewidth=2, label="this model")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("mean predicted probability (per bin)")
        ax.set_ylabel("observed fraction positive (per bin)")
        ax.set_title(f"{title}\nBrier={bs:.5f}", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Reliability diagrams -- true-prevalence OOT test window (day>80)", fontsize=12)
    fig.tight_layout()
    out_path = ARTEFACTS_DIR / out_name
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def build_metrics_table(random_metrics: dict, oot_metrics: dict, baseline_oot_proba: np.ndarray,
                         raw_result: dict, trap_proba: np.ndarray, fix_result: dict,
                         y_test: pd.Series, out_name: str) -> Path:
    bs_trap, _, bss_trap = brier_skill_score(y_test, trap_proba)
    baseline_p100 = precision_at_n(y_test.to_numpy(), baseline_oot_proba, 100)[0]
    rows = [
        {"model": "baseline, naive random holdout", "test_window": "random 20% of day<=80",
         "brier": random_metrics["brier"], "bss": random_metrics["bss"],
         "pr_auc": random_metrics["pr_auc"], "roc_auc": random_metrics["roc_auc"],
         "precision@100": np.nan, "lift@100": np.nan},
        {"model": "baseline, genuine OOT holdout", "test_window": "day>80",
         "brier": oot_metrics["brier"], "bss": oot_metrics["bss"],
         "pr_auc": oot_metrics["pr_auc"], "roc_auc": oot_metrics["roc_auc"],
         "precision@100": baseline_p100,
         "lift@100": baseline_p100 / oot_metrics["base_rate"]},
        {"model": "undersampled, raw/uncalibrated", "test_window": "day>80",
         "brier": raw_result["brier"], "bss": raw_result["bss"], "pr_auc": np.nan, "roc_auc": np.nan,
         "precision@100": precision_at_n(y_test.to_numpy(), raw_result["y_proba"], 100)[0],
         "lift@100": precision_at_n(y_test.to_numpy(), raw_result["y_proba"], 100)[0] / raw_result["base_rate"]},
        {"model": "undersampled, THE TRAP (calibrated on resampled data)", "test_window": "day>80",
         "brier": bs_trap, "bss": bss_trap, "pr_auc": np.nan, "roc_auc": np.nan,
         "precision@100": precision_at_n(y_test.to_numpy(), trap_proba, 100)[0],
         "lift@100": precision_at_n(y_test.to_numpy(), trap_proba, 100)[0] / raw_result["base_rate"]},
        {"model": "undersampled + isotonic (true-prevalence hold-out)", "test_window": "day>80",
         "brier": fix_result["brier_iso"], "bss": fix_result["bss_iso"], "pr_auc": np.nan, "roc_auc": np.nan,
         "precision@100": precision_at_n(y_test.to_numpy(), fix_result["y_proba_iso"], 100)[0],
         "lift@100": precision_at_n(y_test.to_numpy(), fix_result["y_proba_iso"], 100)[0] / raw_result["base_rate"]},
        {"model": "undersampled + Platt (true-prevalence hold-out)", "test_window": "day>80",
         "brier": fix_result["brier_platt"], "bss": fix_result["bss_platt"], "pr_auc": np.nan, "roc_auc": np.nan,
         "precision@100": precision_at_n(y_test.to_numpy(), fix_result["y_proba_platt"], 100)[0],
         "lift@100": precision_at_n(y_test.to_numpy(), fix_result["y_proba_platt"], 100)[0] / raw_result["base_rate"]},
    ]
    table = pd.DataFrame(rows)
    out_path = ARTEFACTS_DIR / out_name
    table.to_csv(out_path, index=False)
    print(f"\n=== combined metrics table ===\n{table.to_string(index=False)}")
    return out_path


def main() -> None:
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = make_dataset()
    report_class_balance(df)

    # Section 3
    random_metrics, oot_metrics, model_oot, oot_test_X, oot_test_y = oot_vs_random(df)
    oot_bar_path = plot_oot_vs_random(random_metrics, oot_metrics, "calib_oot_vs_random.png")
    print(f"Wrote: {oot_bar_path}")

    # Section 4
    brier_by_hand()
    section4 = measuring_what_matters(oot_test_X, oot_test_y, model_oot)
    prec_curve_path = plot_precision_at_n_curve(oot_test_y, section4["y_proba"],
                                                 "calib_precision_at_n_curve.png")
    print(f"Wrote: {prec_curve_path}")

    # Section 5 -- undersampled classifier, true-prevalence windows
    classifier_train = df[df["timestamp_day"] <= CLASSIFIER_TRAIN_CUTOFF_DAY]
    calibrate_df = df[(df["timestamp_day"] > CLASSIFIER_TRAIN_CUTOFF_DAY)
                       & (df["timestamp_day"] <= FULL_TRAIN_CUTOFF_DAY)]
    test_df = df[df["timestamp_day"] > FULL_TRAIN_CUTOFF_DAY]
    X_test, y_test = test_df[FEATURE_COLS], test_df["y"]

    model_us, X_train_us, y_train_us = fit_undersampled_model(classifier_train)
    raw_result = show_miscalibration(model_us, X_test, y_test)
    king_zeng_illustration(model_us, y_train_us, true_prevalence=float(y_test.mean()))

    # Section 6
    trap_proba = the_trap(X_train_us, y_train_us, X_test, y_test)
    fix_result = fix_calibration(model_us, calibrate_df, X_test, y_test, raw_result)

    reliability_path = plot_reliability_diagrams(
        y_test, raw_result["y_proba"], trap_proba,
        fix_result["y_proba_iso"], fix_result["y_proba_platt"],
        "calib_reliability_diagrams.png",
    )
    print(f"\nWrote: {reliability_path}")

    table_path = build_metrics_table(random_metrics, oot_metrics, section4["y_proba"], raw_result,
                                      trap_proba, fix_result, y_test, "calib_metrics_table.csv")
    print(f"Wrote: {table_path}")

    # Sanity assertions -- fail loudly if the story this chapter tells stops being true.
    assert oot_metrics["pr_auc"] < random_metrics["pr_auc"], "OOT should look honestly worse"
    assert raw_result["mean_pred"] > 3 * raw_result["base_rate"], \
        "undersampled model should badly over-predict on true-prevalence data"
    assert fix_result["brier_iso"] < raw_result["brier"], "isotonic should improve Brier"
    assert fix_result["brier_platt"] < raw_result["brier"], "Platt should improve Brier"
    raw_p100 = precision_at_n(y_test.to_numpy(), raw_result["y_proba"], 100)[0]
    platt_p100 = precision_at_n(y_test.to_numpy(), fix_result["y_proba_platt"], 100)[0]
    assert abs(raw_p100 - platt_p100) < 1e-9, \
        "Platt is a strictly monotonic (injective) transform -- it must preserve ranking exactly"
    iso_p1000 = precision_at_n(y_test.to_numpy(), fix_result["y_proba_iso"], 1000)[0]
    raw_p1000 = precision_at_n(y_test.to_numpy(), raw_result["y_proba"], 1000)[0]
    assert abs(raw_p1000 - iso_p1000) < 0.02, \
        "isotonic is only weakly monotonic, but should still track the raw ranking closely at N=1000"
    print("\nAll sanity assertions passed.")


if __name__ == "__main__":
    main()

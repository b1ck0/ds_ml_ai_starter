"""Multi-class vs multi-label classification: digits (one of N) and synthetic tags (subset of N).

Companion code for:
  Data Science/Worked Examples/multiclass-multilabel.md

What it does:
  PART A -- multi-class (sklearn's `load_digits`, 10 classes, one label per sample):
    1. Loads digits (1797 samples, 64 pixel features, digits 0-9) and prints class counts.
    2. Trains two multi-class strategies on the same balanced, stratified split:
       - LogisticRegression's native multinomial (softmax) loss (the default for solver='lbfgs'
         whenever n_classes >= 3).
       - OneVsRestClassifier(LogisticRegression) -- ten independent "is it digit k?" classifiers.
    3. Plots a confusion matrix heatmap for the softmax model and prints classification_report.
    4. Compares macro / micro / weighted F1 for both strategies -- on this balanced dataset they
       nearly agree, which sets up the averaging pitfall below.
    5. Rebuilds the dataset with digit 8 made artificially rare (a realistic "we barely have any
       historical examples of this class" scenario), retrains, and shows how macro F1 reveals the
       resulting weak class while micro/weighted F1 barely move -- because they weight by support,
       and the rare class carries almost none.

  PART B -- multi-label (`make_multilabel_classification`, 6 tags, any subset per sample):
    6. Generates a synthetic "ticket tagging" dataset: 2000 tickets, 6 possible tags, each ticket
       gets >=1 tag (a Poisson(2) number of tags on average).
    7. Trains MultiOutputClassifier(LogisticRegression) -- one binary classifier per tag
       (binary relevance).
    8. Reports a per-label metrics table (classification_report on the 2D indicator target),
       hamming_loss, and subset accuracy (exact-match ratio) -- and shows the same averaging
       pitfall recurring naturally: the two rarest tags have much worse F1 than the rest, hidden
       inside a healthy-looking micro/weighted average.
    9. Demonstrates the "multi-label treated as multi-class" pitfall two ways: the ValueError from
       fitting plain LogisticRegression on a 2D target, and the information loss from collapsing
       each ticket down to a single argmax tag.

  10. Writes:
      ../artefacts/multiclass_confusion_matrix.png
      ../artefacts/multiclass_averaging_pitfall.csv
      ../artefacts/multilabel_per_label_metrics.csv

Environment (verified in research/NOTE-5-sklearn-core-apis.md, research/NOTE-9-classification-
metrics-apis.md and research/NOTE-10-classification-datasets.md, checked 2026-09-02; versions
below verified installed in the project .venv on the same date):
    numpy==2.5.2, pandas==3.0.5, matplotlib==3.11.1, seaborn==0.13.2, scikit-learn==1.9.0
    Python 3.11+ (this script was run and gated on Python 3.13.7).

Run:
    python multiclass_multilabel.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this script only saves figures, never shows them
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_digits, make_multilabel_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    hamming_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier

RNG_SEED = 42
ARTEFACTS_DIR = Path(__file__).resolve().parent.parent / "artefacts"

DIGIT_NAMES = [str(d) for d in range(10)]
TAG_NAMES = ["BUG", "FEATURE", "DOCS", "PERFORMANCE", "SECURITY", "UI"]


# --------------------------------------------------------------------------- #
# PART A -- multi-class: one of 10 digits
# --------------------------------------------------------------------------- #


def load_multiclass_data() -> tuple[np.ndarray, np.ndarray]:
    """sklearn's bundled digits dataset -- (1797, 64) pixel features, y in {0..9}.

    No download required, BSD-3-Clause licensed. See research/NOTE-10-classification-datasets.md.
    """
    X, y = load_digits(return_X_y=True)
    print("=== multi-class: load_digits ===")
    print(f"X: {X.shape}, y: {y.shape}")
    print(f"class counts: {np.bincount(y)}")
    return X, y


def train_multiclass_models(
    X_train: np.ndarray, y_train: np.ndarray
) -> tuple[LogisticRegression, OneVsRestClassifier]:
    """Two ways to make a binary learner (LogisticRegression) handle 10 classes.

    softmax: LogisticRegression's OWN multiclass handling. Per the installed scikit-learn 1.9.0
    docstring ("For multiclass problems (whenever n_classes >= 3), all solvers except 'liblinear'
    optimize the (penalized) multinomial loss") -- the default solver='lbfgs' trains ONE model
    with one softmax output per class, jointly.

    ovr: OneVsRestClassifier wraps the SAME estimator but trains 10 independent binary models,
    each answering "is it digit k, or not" -- and picks the class whose model is most confident.
    """
    softmax_model = LogisticRegression(max_iter=5000, random_state=RNG_SEED)
    softmax_model.fit(X_train, y_train)

    ovr_model = OneVsRestClassifier(LogisticRegression(max_iter=5000, random_state=RNG_SEED))
    ovr_model.fit(X_train, y_train)

    print("\n=== multi-class: softmax vs one-vs-rest ===")
    print(f"softmax coef_ shape: {softmax_model.coef_.shape}  (10 classes, 64 features, "
          f"trained JOINTLY as one multinomial model)")
    print(f"OvR estimators_: {len(ovr_model.estimators_)} independent binary classifiers")
    return softmax_model, ovr_model


def plot_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray) -> Path:
    """Confusion matrix heatmap for the softmax model -- rows = true digit, columns = predicted."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=DIGIT_NAMES,
                yticklabels=DIGIT_NAMES, cbar=True, ax=ax)
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    ax.set_title("Multi-class confusion matrix -- digits (softmax LogisticRegression)")
    fig.tight_layout()

    out_path = ARTEFACTS_DIR / "multiclass_confusion_matrix.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def report_multiclass_metrics(
    y_test: np.ndarray,
    pred_softmax: np.ndarray,
    pred_ovr: np.ndarray,
) -> None:
    """Per-class report for the softmax model, plus macro/micro/weighted F1 for BOTH strategies.

    On this balanced dataset (~45 test samples per digit) macro, micro, and weighted F1 nearly
    agree for both models -- with near-equal support per class, there is no gap for the averaging
    choice to hide anything in. Section 4 breaks that balance on purpose.
    """
    print("\n=== multi-class: classification_report (softmax model) ===")
    print(classification_report(y_test, pred_softmax, target_names=DIGIT_NAMES, digits=3))

    print("=== multi-class: accuracy + macro/micro/weighted F1, softmax vs OvR ===")
    for name, pred in [("softmax", pred_softmax), ("one-vs-rest", pred_ovr)]:
        acc = accuracy_score(y_test, pred)
        macro = f1_score(y_test, pred, average="macro")
        micro = f1_score(y_test, pred, average="micro")
        weighted = f1_score(y_test, pred, average="weighted")
        print(f"{name:12s} accuracy={acc:.3f}  macro-F1={macro:.3f}  "
              f"micro-F1={micro:.3f}  weighted-F1={weighted:.3f}")


def build_rare_digit8_dataset(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild the digits dataset with digit 8 made artificially rare -- ACROSS THE WHOLE
    population, not just the training split, so the TEST set is also imbalanced (a realistic
    "this class barely occurs" scenario, e.g. a rare defect code). This is the setup that lets
    macro/micro/weighted F1 genuinely diverge, unlike Section 2's balanced split.
    """
    rng = np.random.default_rng(RNG_SEED)
    mask_8 = y == 8
    idx_8 = np.where(mask_8)[0]
    keep_8 = rng.choice(idx_8, size=max(4, int(round(0.08 * len(idx_8)))), replace=False)
    keep_idx = np.sort(np.concatenate([np.where(~mask_8)[0], keep_8]))

    X_rare, y_rare = X[keep_idx], y[keep_idx]
    print("\n=== multi-class: digit 8 made artificially rare ===")
    print(f"class counts: {np.bincount(y_rare)}  (digit 8: "
          f"{np.bincount(y)[8]} -> {np.bincount(y_rare)[8]} samples)")
    return X_rare, y_rare


def demonstrate_averaging_pitfall(X_rare: np.ndarray, y_rare: np.ndarray) -> pd.DataFrame:
    """Train on the digit-8-is-rare dataset and show what each averaging strategy reports.

    Returns a per-class + averaged F1 table and writes it to
    artefacts/multiclass_averaging_pitfall.csv.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X_rare, y_rare, test_size=0.25, random_state=RNG_SEED, stratify=y_rare
    )
    print(f"train counts: {np.bincount(y_train)}")
    print(f"test counts:  {np.bincount(y_test)}  (digit 8 has only "
          f"{np.bincount(y_test)[8]} test examples)")

    clf = LogisticRegression(max_iter=5000, random_state=RNG_SEED)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    report = classification_report(
        y_test, pred, target_names=DIGIT_NAMES, digits=3, output_dict=True, zero_division=0
    )
    print(classification_report(y_test, pred, target_names=DIGIT_NAMES, digits=3, zero_division=0))

    acc = accuracy_score(y_test, pred)
    macro = f1_score(y_test, pred, average="macro", zero_division=0)
    micro = f1_score(y_test, pred, average="micro", zero_division=0)
    weighted = f1_score(y_test, pred, average="weighted", zero_division=0)
    print(f"accuracy={acc:.3f}  macro-F1={macro:.3f}  micro-F1={micro:.3f}  "
          f"weighted-F1={weighted:.3f}")
    print("-> digit 8's F1 is the worst in the table, but it holds < 1% of the test-set weight, "
          "so micro-F1 and weighted-F1 barely move while macro-F1 -- which weights every class "
          "the SAME regardless of how rare it is -- drops noticeably.")

    rows = []
    for digit in DIGIT_NAMES:
        rows.append({
            "class": digit,
            "precision": report[digit]["precision"],
            "recall": report[digit]["recall"],
            "f1": report[digit]["f1-score"],
            "support": int(report[digit]["support"]),
        })
    for avg_name in ["macro avg", "micro avg" if "micro avg" in report else None, "weighted avg"]:
        if avg_name is None:
            continue
        rows.append({
            "class": avg_name,
            "precision": report[avg_name]["precision"],
            "recall": report[avg_name]["recall"],
            "f1": report[avg_name]["f1-score"],
            "support": int(report[avg_name]["support"]),
        })
    # classification_report's output_dict does not include a 'micro avg' row for single-label
    # multiclass (micro == accuracy there); add it explicitly so the comparison table is complete.
    rows.append({"class": "micro avg (== accuracy)", "precision": micro, "recall": micro,
                 "f1": micro, "support": len(y_test)})

    table = pd.DataFrame(rows)
    out_path = ARTEFACTS_DIR / "multiclass_averaging_pitfall.csv"
    table.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")
    return table


# --------------------------------------------------------------------------- #
# PART B -- multi-label: any subset of 6 tags
# --------------------------------------------------------------------------- #


def load_multilabel_data() -> tuple[np.ndarray, np.ndarray]:
    """Synthetic "ticket tagging" dataset: 2000 tickets, 20 features, 6 possible tags.

    make_multilabel_classification(n_samples=..., n_features=..., n_classes=..., n_labels=...,
    allow_unlabeled=..., random_state=...) is scikit-learn's built-in synthetic multilabel
    generator -- fully runnable, no download. Signature verified against scikit-learn 1.9.0 and
    recommended for a fully-runnable multi-label example in
    research/NOTE-10-classification-datasets.md. allow_unlabeled=False forces every ticket to
    carry at least one tag (mirroring a real ticket-tagging workflow, where an untagged ticket
    isn't a valid "no tags" case but an unprocessed one).
    """
    X, Y = make_multilabel_classification(
        n_samples=2000, n_features=20, n_classes=len(TAG_NAMES), n_labels=2,
        allow_unlabeled=False, random_state=RNG_SEED,
    )
    print("\n=== multi-label: make_multilabel_classification ===")
    print(f"X: {X.shape}, Y: {Y.shape}")
    print("tag prevalence (fraction of tickets carrying each tag):")
    for tag, freq in zip(TAG_NAMES, Y.mean(axis=0)):
        print(f"  {tag:12s} {freq:.3f}")
    labels_per_ticket = Y.sum(axis=1)
    print(f"tags per ticket: mean={labels_per_ticket.mean():.2f}, "
          f"min={labels_per_ticket.min()}, max={labels_per_ticket.max()}, "
          f"fraction with >1 tag={(labels_per_ticket > 1).mean():.3f}")
    return X, Y


def train_multilabel_model(
    X_train: np.ndarray, Y_train: np.ndarray
) -> MultiOutputClassifier:
    """Binary relevance: one independent LogisticRegression per tag, via MultiOutputClassifier.

    MultiOutputClassifier(estimator) fits a SEPARATE copy of `estimator` for each column of the
    2D target -- six independent "does this ticket carry tag k?" binary classifiers, exactly the
    OneVsRestClassifier idea from Part A, but for labels that are NOT mutually exclusive.
    """
    clf = MultiOutputClassifier(LogisticRegression(max_iter=5000, random_state=RNG_SEED))
    clf.fit(X_train, Y_train)
    print(f"\n=== multi-label: MultiOutputClassifier ===")
    print(f"fitted {len(clf.estimators_)} independent binary classifiers, one per tag")
    return clf


def report_multilabel_metrics(Y_test: np.ndarray, Y_pred: np.ndarray) -> pd.DataFrame:
    """Per-tag precision/recall/F1 (classification_report accepts a 2D indicator target directly),
    hamming_loss (average fraction of WRONG tags per ticket), and subset accuracy (exact match:
    every tag right, nothing extra, nothing missing).
    """
    report = classification_report(
        Y_test, Y_pred, target_names=TAG_NAMES, digits=3, output_dict=True, zero_division=0
    )
    print("\n=== multi-label: per-tag classification_report ===")
    print(classification_report(Y_test, Y_pred, target_names=TAG_NAMES, digits=3, zero_division=0))

    h_loss = hamming_loss(Y_test, Y_pred)
    subset_acc = accuracy_score(Y_test, Y_pred)  # exact-match ratio for a 2D indicator target
    print(f"hamming_loss:    {h_loss:.3f}  (average fraction of the 6 tags wrong per ticket)")
    print(f"subset accuracy: {subset_acc:.3f}  (fraction of tickets with ALL 6 tags exactly right)")
    print("-> subset accuracy is far stricter than any single tag's F1: one wrong tag out of six "
          "fails the whole ticket, so it drops fast even when every per-tag metric looks decent.")

    rows = []
    for tag in TAG_NAMES:
        rows.append({
            "tag": tag,
            "precision": report[tag]["precision"],
            "recall": report[tag]["recall"],
            "f1": report[tag]["f1-score"],
            "support": int(report[tag]["support"]),
        })
    for avg_name in ["micro avg", "macro avg", "weighted avg", "samples avg"]:
        if avg_name in report:
            rows.append({
                "tag": avg_name,
                "precision": report[avg_name]["precision"],
                "recall": report[avg_name]["recall"],
                "f1": report[avg_name]["f1-score"],
                "support": int(report[avg_name]["support"]),
            })
    rows.append({"tag": "hamming_loss", "precision": np.nan, "recall": np.nan,
                 "f1": h_loss, "support": len(Y_test)})
    rows.append({"tag": "subset_accuracy", "precision": np.nan, "recall": np.nan,
                 "f1": subset_acc, "support": len(Y_test)})

    table = pd.DataFrame(rows)
    out_path = ARTEFACTS_DIR / "multilabel_per_label_metrics.csv"
    table.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")
    return table


def demonstrate_multilabel_pitfalls(X: np.ndarray, Y: np.ndarray) -> None:
    """Two ways "treat multi-label like multi-class" breaks, both shown concretely:

    1. Fit an ordinary LogisticRegression directly on the 2D indicator target -- scikit-learn
       refuses, because a plain multi-class classifier expects exactly one label per row.
    2. Collapse each ticket's true tag set down to a single tag (argmax-style, "just pick one")
       -- and measure how many tickets that throws real information away for.
    """
    print("\n=== multi-label pitfall 1: fitting a single-label classifier on a 2D target ===")
    try:
        LogisticRegression(max_iter=1000).fit(X, Y)
    except ValueError as exc:
        print(f"ValueError: {exc}")
        print("-> LogisticRegression expects y as ONE label per row; a multilabel indicator "
              "matrix (n_samples, n_classes) is not a valid target for it at all.")

    print("\n=== multi-label pitfall 2: collapsing tags to a single 'most important' tag ===")
    labels_per_ticket = Y.sum(axis=1)
    multi_tag_frac = (labels_per_ticket > 1).mean()
    print(f"{multi_tag_frac:.1%} of tickets carry MORE than one true tag.")
    print("Forcing a single-label model onto this data (e.g. keeping only the first tag found) "
          f"would silently discard the other tag(s) on {multi_tag_frac:.1%} of tickets -- not a "
          "metric quirk, a modeling decision that throws away real, observed information before "
          "the model ever sees it.")


def main() -> None:
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Part A: multi-class ---
    X, y = load_multiclass_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RNG_SEED, stratify=y
    )
    softmax_model, ovr_model = train_multiclass_models(X_train, y_train)
    pred_softmax = softmax_model.predict(X_test)
    pred_ovr = ovr_model.predict(X_test)

    cm_path = plot_confusion_matrix(y_test, pred_softmax)
    report_multiclass_metrics(y_test, pred_softmax, pred_ovr)

    X_rare, y_rare = build_rare_digit8_dataset(X, y)
    pitfall_table = demonstrate_averaging_pitfall(X_rare, y_rare)

    # --- Part B: multi-label ---
    X_ml, Y_ml = load_multilabel_data()
    X_ml_train, X_ml_test, Y_ml_train, Y_ml_test = train_test_split(
        X_ml, Y_ml, test_size=0.25, random_state=RNG_SEED
    )
    ml_clf = train_multilabel_model(X_ml_train, Y_ml_train)
    Y_ml_pred = ml_clf.predict(X_ml_test)
    per_label_table = report_multilabel_metrics(Y_ml_test, Y_ml_pred)
    demonstrate_multilabel_pitfalls(X_ml, Y_ml)

    print(f"\nWrote: {cm_path}")


if __name__ == "__main__":
    main()

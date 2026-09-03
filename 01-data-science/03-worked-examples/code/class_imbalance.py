"""Class imbalance: why accuracy lies, and what actually catches the minority class.

Companion code for:
  Data Science/Worked Examples/class-imbalance.md

What it does:
  1. Builds a synthetic ~3%-positive dataset with sklearn.datasets.make_classification
     (NOTE-10-classification-datasets) and splits it train/test (stratified, so both
     splits keep the same ~3% minority rate).
  2. Shows the "always predict majority" trap: a DummyClassifier that never predicts
     the positive class scores ~97% accuracy and 0% recall -- accuracy alone cannot
     tell you this model is useless.
  3. Trains a plain LogisticRegression baseline (no resampling) and reports recall +
     PR-AUC (average_precision_score) alongside accuracy, per NOTE-9-classification-
     metrics-apis's guidance that PR-AUC/recall are what matter on imbalanced data.
  4. Compares four remedies, each as an imblearn.pipeline.Pipeline fit ONLY on the
     training split (NOTE-11-imblearn-apis): class_weight='balanced' DecisionTree,
     RandomUnderSampler, RandomOverSampler, SMOTE (base learner: DecisionTreeClassifier
     throughout, so every comparison isolates the resampling strategy, not the model).
  5. Trains two ensembles-of-undersampled-learners (BalancedBaggingClassifier,
     EasyEnsembleClassifier, NOTE-11) and shows their lift over the single-model
     baseline on recall and PR-AUC.
  6. Tunes the decision threshold on the best ensemble's PR curve against an explicit
     false-negative:false-positive cost ratio, instead of the default 0.5 cutoff.
  7. Demonstrates the resample-BEFORE-split leakage pitfall concretely: oversampling
     the whole dataset before splitting duplicates exact feature rows across train
     and test; resampling after splitting does not.
  8. Writes all artefacts (PR-curve comparison, confusion matrices, threshold-tuning
     plot, comparison table) to ../artefacts/.

Environment (verified in research/NOTE-2-package-versions.md, research/NOTE-5-sklearn-
core-apis.md, and research/NOTE-11-imblearn-apis.md, checked 2026-09-02):
    numpy==2.5.2, pandas==3.0.5, matplotlib==3.11.1, scikit-learn==1.9.0,
    imbalanced-learn==0.14.2
    Python 3.12+ (this script was run and gated on Python 3.13.7).

Run:
    python class_imbalance.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this script only saves figures, never shows them
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedBaggingClassifier, EasyEnsembleClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

RNG_SEED = 42
ARTEFACTS_DIR = Path(__file__).resolve().parent.parent / "artefacts"

# Cost of missing a real positive vs. raising a false alarm -- used in the threshold
# tuning section. 10:1 mirrors a fraud/defect-style setting: a missed positive is far
# more expensive than an analyst reviewing one extra false alarm.
FALSE_NEGATIVE_COST = 10.0
FALSE_POSITIVE_COST = 1.0


def make_data() -> tuple[np.ndarray, np.ndarray]:
    """~3% positive class, fully synthetic and reproducible (NOTE-10-classification-datasets).

    weights=[0.97, 0.03] -> ~97% majority / ~3% minority, inside the 1-5% band NOTE-10
    recommends for this chapter. n_informative=4 keeps a real (if noisy) signal so the
    remedies below have something to find; flip_y=0.01 adds label noise so no model
    gets a trivially perfect score.
    """
    X, y = make_classification(
        n_samples=8000,
        n_features=20,
        n_informative=4,
        n_redundant=2,
        n_clusters_per_class=1,
        weights=[0.97, 0.03],
        flip_y=0.01,
        class_sep=1.0,
        random_state=RNG_SEED,
    )
    return X, y


def split_data(X: np.ndarray, y: np.ndarray):
    """Stratified split so both train and test keep the ~3% minority rate."""
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=RNG_SEED)


def report_class_balance(y: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> None:
    def counts(arr: np.ndarray) -> str:
        n_pos = int(arr.sum())
        return f"{len(arr)} rows, {n_pos} positive ({n_pos / len(arr):.2%})"

    print("=== class balance ===")
    print(f"full dataset: {counts(y)}")
    print(f"train split:  {counts(y_train)}")
    print(f"test split:   {counts(y_test)}")


def show_majority_trap(y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """DummyClassifier(strategy='most_frequent') never predicts the positive class.

    It still scores ~97% accuracy, because accuracy only counts how often the
    prediction matches the label -- and on this dataset, predicting "negative" is
    right 97% of the time by construction. Recall (of the positive class) is exactly
    0: it is the metric that exposes the trap; accuracy cannot.
    """
    dummy = DummyClassifier(strategy="most_frequent", random_state=RNG_SEED)
    dummy.fit(np.zeros((len(y_train), 1)), y_train)
    y_pred = dummy.predict(np.zeros((len(y_test), 1)))

    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== the 'always predict majority' trap ===")
    print(f"DummyClassifier(strategy='most_frequent'): accuracy={acc:.4f}, recall={recall:.4f}")
    print(f"confusion matrix [[TN FP] [FN TP]]:\n{cm}")
    print(f"It caught {cm[1, 1]} of {cm[1].sum()} real positives. Accuracy alone would "
          f"call this model {acc:.1%} correct.")
    return {"model": "always-predict-majority", "accuracy": acc, "recall@0.5": recall,
            "pr_auc": np.nan}


def evaluate_model(name: str, model, X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Fit on train only, score on test: accuracy, recall@0.5, PR-AUC, confusion matrix.

    PR-AUC (average_precision_score) is threshold-independent -- it scores the
    probability ranking, not one fixed cutoff -- which is why NOTE-9-classification-
    metrics-apis recommends it over ROC-AUC for imbalanced data: ROC-AUC can stay high
    even when a model misses most positives, because the majority class dominates the
    false-positive-rate denominator.
    """
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"{name:32s} accuracy={acc:.4f}  recall@0.5={recall:.4f}  PR-AUC={pr_auc:.4f}  "
          f"caught={cm[1, 1]}/{cm[1].sum()} positives")

    return {"model": name, "accuracy": acc, "recall@0.5": recall, "pr_auc": pr_auc,
            "_y_proba": y_proba, "_confusion_matrix": cm}


def build_candidates(base_estimator_kwargs: dict) -> dict:
    """The single-model baseline plus four remedies, all as fit-on-train-only pipelines.

    The base learner is a single DecisionTreeClassifier throughout -- deliberately a
    high-variance model (a shallow tree's boundary shifts a lot depending on exactly
    which rows it sees). That variance is what Section 4's ensemble exploits: bagging
    many independently-undersampled trees averages that variance away. Every resampler
    here is wrapped in imblearn.pipeline.Pipeline so fit_resample() only ever touches
    whatever X/y the pipeline is fit on -- never the test split. Signatures verified
    against imbalanced-learn 0.14.2 (research/NOTE-11-imblearn-apis).
    """
    return {
        "baseline (no resampling)": DecisionTreeClassifier(**base_estimator_kwargs),
        "class_weight='balanced'": DecisionTreeClassifier(class_weight="balanced",
                                                            **base_estimator_kwargs),
        "RandomUnderSampler": ImbPipeline(steps=[
            ("resample", RandomUnderSampler(random_state=RNG_SEED)),
            ("clf", DecisionTreeClassifier(**base_estimator_kwargs)),
        ]),
        "RandomOverSampler": ImbPipeline(steps=[
            ("resample", RandomOverSampler(random_state=RNG_SEED)),
            ("clf", DecisionTreeClassifier(**base_estimator_kwargs)),
        ]),
        "SMOTE": ImbPipeline(steps=[
            ("resample", SMOTE(random_state=RNG_SEED)),
            ("clf", DecisionTreeClassifier(**base_estimator_kwargs)),
        ]),
    }


def build_ensembles(base_estimator_kwargs: dict) -> dict:
    """BalancedBaggingClassifier and EasyEnsembleClassifier: each trains many learners,
    every one on its own randomly-undersampled balanced subset of the training data,
    then averages their votes. BalancedBaggingClassifier is given the same
    DecisionTreeClassifier as its base learner, so the comparison against the single-
    tree baseline/remedies above isolates exactly one variable: bagging over many
    balanced subsets instead of fitting once. Signatures verified against
    imbalanced-learn 0.14.2 (research/NOTE-11-imblearn-apis).
    """
    return {
        "BalancedBaggingClassifier": BalancedBaggingClassifier(
            estimator=DecisionTreeClassifier(**base_estimator_kwargs),
            n_estimators=25,
            sampling_strategy="auto",
            random_state=RNG_SEED,
            n_jobs=-1,
        ),
        "EasyEnsembleClassifier": EasyEnsembleClassifier(
            n_estimators=25,
            sampling_strategy="auto",
            random_state=RNG_SEED,
            n_jobs=-1,
        ),
    }


def plot_pr_curves(y_test: np.ndarray, curves: dict[str, np.ndarray], out_name: str) -> Path:
    """One PR curve per model, overlaid, plus the no-skill baseline (the positive
    class's prevalence in the test set -- what a random-score model would achieve).
    """
    fig, ax = plt.subplots(figsize=(7.5, 6))
    colors = plt.get_cmap("tab10").colors
    for i, (name, y_proba) in enumerate(curves.items()):
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        ax.plot(recall, precision, label=f"{name} (PR-AUC={ap:.3f})",
                 color=colors[i % len(colors)], linewidth=2)

    no_skill = y_test.sum() / len(y_test)
    ax.axhline(no_skill, color="grey", linestyle="--", linewidth=1,
               label=f"no-skill baseline (prevalence={no_skill:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves: baseline vs remedies vs ensembles")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()

    out_path = ARTEFACTS_DIR / out_name
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_confusion_matrices(cms: dict[str, np.ndarray], out_name: str) -> Path:
    """Side-by-side confusion matrices, one panel per model, annotated with counts."""
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, cm) in zip(axes, cms.items()):
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["pred neg", "pred pos"])
        ax.set_yticklabels(["true neg", "true pos"])
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color,
                        fontsize=13)
        ax.set_title(name, fontsize=10)
    fig.tight_layout()

    out_path = ARTEFACTS_DIR / out_name
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def tune_threshold(y_test: np.ndarray, y_proba: np.ndarray, model_name: str) -> Path:
    """Sweep the decision threshold from the PR curve and pick the one that minimises
    a business cost: FALSE_NEGATIVE_COST per missed positive + FALSE_POSITIVE_COST per
    false alarm. The default 0.5 cutoff optimises nothing in particular -- it is just
    the midpoint of predict_proba's [0, 1] range.
    """
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    # precision_recall_curve returns len(thresholds) == len(precision) - 1
    # (the last precision/recall point has no corresponding threshold: it is the
    # "predict everyone positive" limit). Evaluate cost only at the points with a
    # real threshold.
    n_pos = int(y_test.sum())
    n_neg = len(y_test) - n_pos

    costs = []
    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        tp = r * n_pos
        fn = n_pos - tp
        fp = (tp / p - tp) if p > 0 else n_neg  # fp = tp * (1 - p) / p
        cost = FALSE_NEGATIVE_COST * fn + FALSE_POSITIVE_COST * fp
        costs.append(cost)
    costs = np.array(costs)

    best_idx = int(np.argmin(costs))
    best_threshold = thresholds[best_idx]
    best_precision = precision[best_idx]
    best_recall = recall[best_idx]
    default_idx = int(np.argmin(np.abs(thresholds - 0.5)))

    print(f"\n=== threshold tuning ({model_name}) ===")
    print(f"cost model: {FALSE_NEGATIVE_COST:.0f} per missed positive, "
          f"{FALSE_POSITIVE_COST:.0f} per false alarm")
    print(f"default threshold 0.5:  precision={precision[default_idx]:.3f}, "
          f"recall={recall[default_idx]:.3f}, cost={costs[default_idx]:.1f}")
    print(f"cost-optimal threshold {best_threshold:.3f}: precision={best_precision:.3f}, "
          f"recall={best_recall:.3f}, cost={costs[best_idx]:.1f}")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(recall, precision, color="#4C72B0", linewidth=2, label="PR curve")
    ax.scatter([recall[default_idx]], [precision[default_idx]], color="#C44E52", zorder=5,
               s=70, label=f"threshold=0.500 (cost={costs[default_idx]:.0f})")
    ax.scatter([best_recall], [best_precision], color="#55A868", zorder=5, s=90, marker="*",
               label=f"threshold={best_threshold:.3f} (cost={costs[best_idx]:.0f}, cost-optimal)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Threshold tuning: {model_name}\n"
                 f"cost = {FALSE_NEGATIVE_COST:.0f}x FN + {FALSE_POSITIVE_COST:.0f}x FP")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()

    out_path = ARTEFACTS_DIR / "threshold_tuning.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def demonstrate_leakage(X: np.ndarray, y: np.ndarray) -> None:
    """The core discipline: resample AFTER the split, never before.

    RandomOverSampler duplicates minority rows to reach the target ratio. If you
    resample the WHOLE dataset first and split afterwards, some of those exact
    duplicate rows land in train and their identical twin lands in test -- the model
    is then partly "tested" on rows it memorised during training. If you split first
    and resample only the training fold, no such duplicate can ever cross the
    train/test boundary, because the test fold was never touched by the resampler.
    """
    print("\n=== leakage demo: resample-before-split vs resample-after-split ===")

    # WRONG: resample the entire dataset, then split.
    ros_leak = RandomOverSampler(random_state=RNG_SEED)
    X_res, y_res = ros_leak.fit_resample(X, y)
    X_train_leak, X_test_leak, y_train_leak, y_test_leak = train_test_split(
        X_res, y_res, test_size=0.25, stratify=y_res, random_state=RNG_SEED
    )
    train_rows_leak = {tuple(row) for row in X_train_leak}
    test_rows_leak = {tuple(row) for row in X_test_leak}
    overlap_leak = train_rows_leak & test_rows_leak
    print(f"WRONG (resample whole dataset, then split): "
          f"{len(overlap_leak)} identical feature rows appear in BOTH train and test.")

    # RIGHT: split first, resample only the training fold.
    X_train, X_test, y_train, y_test = split_data(X, y)
    ros_ok = RandomOverSampler(random_state=RNG_SEED)
    X_train_res, y_train_res = ros_ok.fit_resample(X_train, y_train)
    train_rows_ok = {tuple(row) for row in X_train_res}
    test_rows_ok = {tuple(row) for row in X_test}
    overlap_ok = train_rows_ok & test_rows_ok
    print(f"RIGHT (split first, resample train only):   "
          f"{len(overlap_ok)} identical feature rows appear in BOTH train and test.")
    print("The wrong order leaks exact copies of test rows into the training set -- any "
          "metric computed on that test split is now partly measuring memorisation, not "
          "generalisation. Sklearn/imblearn will not warn you; the leaked model will "
          "just look better than it is.")


def main() -> None:
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y = make_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    report_class_balance(y, y_train, y_test)

    trap_result = show_majority_trap(y_train, X_test, y_test)

    tree_kwargs = {"random_state": RNG_SEED, "max_depth": 6}
    results = [trap_result]
    proba_by_model: dict[str, np.ndarray] = {}
    cm_by_model: dict[str, np.ndarray] = {}

    print("\n=== baseline vs remedies (single DecisionTreeClassifier, max_depth=6) ===")
    candidates = build_candidates(tree_kwargs)
    for name, model in candidates.items():
        r = evaluate_model(name, model, X_train, y_train, X_test, y_test)
        proba_by_model[name] = r.pop("_y_proba")
        cm_by_model[name] = r.pop("_confusion_matrix")
        results.append(r)

    print("\n=== ensembles of undersampled learners ===")
    ensembles = build_ensembles(tree_kwargs)
    for name, model in ensembles.items():
        r = evaluate_model(name, model, X_train, y_train, X_test, y_test)
        proba_by_model[name] = r.pop("_y_proba")
        cm_by_model[name] = r.pop("_confusion_matrix")
        results.append(r)

    baseline_recall = next(r["recall@0.5"] for r in results if r["model"] == "baseline (no resampling)")
    baseline_pr_auc = next(r["pr_auc"] for r in results if r["model"] == "baseline (no resampling)")
    best_ensemble_name = max(
        ("BalancedBaggingClassifier", "EasyEnsembleClassifier"),
        key=lambda n: next(r["pr_auc"] for r in results if r["model"] == n),
    )
    best_ensemble_recall = next(r["recall@0.5"] for r in results if r["model"] == best_ensemble_name)
    best_ensemble_pr_auc = next(r["pr_auc"] for r in results if r["model"] == best_ensemble_name)
    print(f"\nBest ensemble: {best_ensemble_name} "
          f"(recall@0.5 {baseline_recall:.4f} -> {best_ensemble_recall:.4f}, "
          f"PR-AUC {baseline_pr_auc:.4f} -> {best_ensemble_pr_auc:.4f})")
    assert best_ensemble_recall > baseline_recall, "ensemble did not beat baseline on recall"
    assert best_ensemble_pr_auc > baseline_pr_auc, "ensemble did not beat baseline on PR-AUC"

    # Comparison table -> CSV artefact.
    table = pd.DataFrame(results)[["model", "accuracy", "recall@0.5", "pr_auc"]]
    table_path = ARTEFACTS_DIR / "comparison_table.csv"
    table.to_csv(table_path, index=False)
    print(f"\n=== full comparison table ===\n{table.to_string(index=False)}")

    # PR curves: baseline, each remedy, both ensembles.
    pr_curve_path = plot_pr_curves(y_test, proba_by_model, "pr_curves_comparison.png")

    # Confusion matrices: baseline vs best ensemble.
    cm_path = plot_confusion_matrices(
        {"baseline (no resampling)": cm_by_model["baseline (no resampling)"],
         best_ensemble_name: cm_by_model[best_ensemble_name]},
        "confusion_matrices.png",
    )

    # Threshold tuning on the best ensemble's probabilities.
    threshold_path = tune_threshold(y_test, proba_by_model[best_ensemble_name], best_ensemble_name)

    # Leakage pitfall demo.
    demonstrate_leakage(X, y)

    print(f"\nWrote: {table_path}")
    print(f"Wrote: {pr_curve_path}")
    print(f"Wrote: {cm_path}")
    print(f"Wrote: {threshold_path}")


if __name__ == "__main__":
    main()

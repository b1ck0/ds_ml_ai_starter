# NOTE-9: Classification metrics APIs and probability predictions (2026-09-02)

**Answer:** scikit-learn 1.9.0 provides all required classification metrics with stable, documented signatures; `predict_proba()` (probability estimates) and `decision_function()` (confidence scores) are the methods to obtain predicted probabilities. PR-AUC is preferred over ROC-AUC for imbalanced datasets because it focuses on the minority class; ROC-AUC can report overly optimistic results when the negative class dominates.

**Evidence:**

| Metric/Function | Signature | Notes | Source |
|-----------------|-----------|-------|--------|
| **confusion_matrix** | `confusion_matrix(y_true, y_pred, *, labels=None, sample_weight=None, normalize=None)` | Returns (n_classes, n_classes) matrix; normalize={'true', 'pred', 'all', None} | https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html |
| **classification_report** | `classification_report(y_true, y_pred, *, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division='warn')` | Text/dict report of precision, recall, F1 per class | https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html |
| **roc_auc_score** | `roc_auc_score(y_true, y_score, *, average='macro', sample_weight=None, max_fpr=None, multi_class='raise', labels=None)` | Computes area under ROC curve; average={'micro','macro','samples','weighted',None}; requires y_score (probabilities/decision values) | https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html |
| **roc_curve** | `roc_curve(y_true, y_score, *, pos_label=None, sample_weight=None, drop_intermediate=True)` | Returns (fpr, tpr, thresholds) — three 1D arrays for plotting | https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html |
| **precision_recall_curve** | `precision_recall_curve(y_true, y_score, *, pos_label=None, sample_weight=None, drop_intermediate=False)` | Returns (precision, recall, thresholds); binary classification only | https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html |
| **average_precision_score** | `average_precision_score(y_true, y_score, *, average='macro', pos_label=1, sample_weight=None)` | Computes area under precision-recall curve; average={'micro','macro','samples','weighted',None} | https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html |
| **f1_score** | `f1_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')` | Harmonic mean of precision and recall; average={'binary','micro','macro','weighted','samples',None} | https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html |
| **precision_score** | `precision_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')` | TP / (TP + FP); average={'binary','micro','macro','weighted','samples',None} | https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html |
| **recall_score** | `recall_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')` | TP / (TP + FN); average={'binary','micro','macro','weighted','samples',None} | https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html |
| **accuracy_score** | `accuracy_score(y_true, y_pred, *, normalize=True, sample_weight=None)` | Fraction of correct predictions (normalize=True) or count (normalize=False) | https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html |
| **hamming_loss** | `hamming_loss(y_true, y_pred, *, sample_weight=None)` | Average Hamming distance (avg # of incorrect labels per sample); suitable for multilabel | https://scikit-learn.org/stable/modules/generated/sklearn.metrics.hamming_loss.html |
| **Probability predictions** | `estimator.predict_proba(X)` → (n_samples, n_classes); `estimator.decision_function(X)` → (n_samples,) or (n_samples, n_classes) | Use predict_proba for probability estimates; decision_function for confidence scores on linear models | https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html |
| **OneVsRestClassifier** | `from sklearn.multiclass import OneVsRestClassifier(estimator, *, n_jobs=None, verbose=0)` | One-vs-rest strategy for multiclass; wraps any binary classifier | https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html |
| **MultiOutputClassifier** | `from sklearn.multioutput import MultiOutputClassifier(estimator, *, n_jobs=None)` | Wraps any estimator to handle multilabel classification; trains separate models per target | https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html |

### Average Parameter Values (for f1_score, precision_score, recall_score, roc_auc_score, average_precision_score)

| Value | Description | Use Case |
|-------|-------------|----------|
| `'binary'` (default for f1/precision/recall) | Only report results for positive class (pos_label) | Binary classification only |
| `'micro'` | Compute metrics globally (total TP, FP, FN) | Useful for multilabel |
| `'macro'` | Unweighted mean across classes (treats each class equally) | Multi-class when all classes matter equally |
| `'weighted'` | Weighted mean by support (# samples per class) | Multi-class when class imbalance is expected |
| `'samples'` | Average per sample (multilabel only) | Multi-label classification |
| `None` | Return metric for each class separately | When you need per-class breakdown |

### When to Use PR-AUC vs ROC-AUC

**PR-AUC is preferred for imbalanced datasets** because:
- ROC-AUC can give overly optimistic estimates when the negative class dominates.
- PR-AUC focuses specifically on the positive (minority) class performance.
- For severely imbalanced data (e.g., 1–5% positives), ROC-AUC may remain high even when the model misses most positives.
- PR-AUC has a data-dependent baseline (not fixed at 0.5 like ROC-AUC), reflecting the minority class prevalence.

**ROC-AUC** is still useful for balanced or general-purpose multi-class problems.

**Recommendation:** Always use `predict_proba()` or `decision_function()` to obtain probability/confidence scores for ROC/PR curves and AUC metrics. For binary classification, both methods work; for multiclass, `predict_proba()` is standard. For imbalanced tasks (DS-8), compute both ROC-AUC and PR-AUC, but prioritize PR-AUC for reporting. Use `average='macro'`, `'weighted'`, or `'micro'` to handle multiclass/multilabel correctly.

**Caveats / limits:**
- `precision_recall_curve()` is **binary classification only**; for multilabel, use per-label loop or multilabel_confusion_matrix.
- `roc_auc_score()` with `multi_class='raise'` (default) will error on multiclass; use `multi_class='ovr'` or `'ovo'`.
- The `average` parameter is ignored for binary classification in some metrics (e.g., `roc_auc_score`).
- `zero_division='warn'` (default) raises a warning when precision/recall/F1 denominators are zero; set to 0.0 or 1.0 to suppress.


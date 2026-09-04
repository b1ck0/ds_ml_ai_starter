# NOTE-DS-20-5: Current scikit-learn 1.9.0 calibration API signatures

**Answer:** All functions/classes exist with documented APIs (below); scikit-learn 1.9.0 docs explicitly warn against fitting calibrators on training data and document isotonic overfitting on small samples (<~1000).

**Evidence:**

### 1. `sklearn.metrics.brier_score_loss`

From https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html (sklearn 1.9.0):

**Signature:**
```python
sklearn.metrics.brier_score_loss(
    y_true,
    y_proba,
    *,
    sample_weight=None,
    pos_label=None,
    labels=None,
    scale_by_half='auto'
)
```

**Parameters:**
- `y_true` : array-like of shape (n_samples,). True targets (0 or 1 for binary).
- `y_proba` : array-like of shape (n_samples,) or (n_samples, n_classes). Predicted probabilities. If 1-D, assumed to be proba for positive class.
- `sample_weight` : optional weights per sample.
- `pos_label` : positive label identifier (rarely used for binary).
- `labels` : class labels (for multiclass).
- `scale_by_half` : {'auto', True, False} (default 'auto'). If True, divides the Brier score by 2 (historical convention in some fields). Default 'auto' uses scikit-learn's convention (no scaling).

**Returns:** float, the Brier score.

**Example:**
```python
from sklearn.metrics import brier_score_loss
y_true = [0, 1, 1, 0, 1]
y_proba = [0.1, 0.8, 0.9, 0.2, 0.7]
bs = brier_score_loss(y_true, y_proba)  # ≈ 0.068
```

### 2. `sklearn.calibration.calibration_curve`

From https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html (sklearn 1.9.0):

**Signature:**
```python
sklearn.calibration.calibration_curve(
    y_true,
    y_proba,
    *,
    pos_label=None,
    n_bins=5,
    strategy='uniform',
    sample_weight=None
)
```

**Parameters:**
- `y_true` : array-like. True targets.
- `y_proba` : array-like. Predicted probabilities (1-D or 2-D; if 2-D, uses second column for binary positive class).
- `pos_label` : positive label (for binary).
- `n_bins` : int (default 5). Number of bins for binning probabilities.
- `strategy` : {'uniform', 'quantile'} (default 'uniform'). If 'uniform', bins are fixed width [0, 0.2, 0.4, ..., 1.0]. If 'quantile', bins are chosen so each has equal sample count.
- `sample_weight` : optional.

**Returns:** (prob_true, prob_pred)
- `prob_true` : array, fraction of positives in each bin.
- `prob_pred` : array, mean predicted probability in each bin.

**Note:** Used internally by `CalibrationDisplay` to generate reliability diagram data.

### 3. `sklearn.calibration.CalibrationDisplay`

From https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibrationDisplay.html (sklearn 1.9.0):

**Main method:**
```python
sklearn.calibration.CalibrationDisplay.from_estimator(
    estimator,
    X,
    y,
    *,
    pos_label=None,
    n_bins=5,
    strategy='uniform',
    sample_weight=None,
    ax=None,
    name=None
)
```

**Purpose:** Fits the estimator (if needed), calls `calibration_curve`, and plots the result (reliability diagram).

**Key attributes of the returned CalibrationDisplay object:**
- `ax_` : matplotlib axes where the plot was drawn.
- `figure_` : matplotlib figure.
- `line_` : calibration line (points and connected line).
- `histogram_` : histogram of predicted probabilities (if drawn).

**Example:**
```python
from sklearn.calibration import CalibrationDisplay
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

clf = LogisticRegression()
clf.fit(X_train, y_train)

disp = CalibrationDisplay.from_estimator(clf, X_test, y_test, n_bins=10)
plt.show()
```

### 4. `sklearn.isotonic.IsotonicRegression`

From https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html (sklearn 1.9.0):

**Signature:**
```python
sklearn.isotonic.IsotonicRegression(
    y_min=None,
    y_max=None,
    increasing=True,
    out_of_bounds='nan'
)
```

**Parameters:**
- `y_min`, `y_max` : optional min/max for output (often 0, 1 for probabilities).
- `increasing` : bool (default True). If True, fit monotonically increasing function; if False, decreasing.
- `out_of_bounds` : {'nan', 'clip', 'raise'} (default 'nan'). How to handle predictions outside the training range.

**Fit and predict:**
```python
iso = IsotonicRegression(y_min=0, y_max=1)
iso.fit(scores_train, proba_train)  # scores_train = raw model scores, proba_train = true outcomes (0/1)
proba_calibrated = iso.predict(scores_test)  # transform raw scores to probabilities
```

**Note:** Input to `fit` is not (X, y) but (X_scores, y_binary), where X_scores are 1-D uncalibrated scores and y_binary are the actual outcomes. This is unusual compared to typical sklearn estimators.

### 5. `sklearn.calibration.CalibratedClassifierCV`

From https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html (sklearn 1.9.0):

**Signature:**
```python
sklearn.calibration.CalibratedClassifierCV(
    estimator=None,
    *,
    method='sigmoid',
    cv=5,
    n_jobs=None,
    ensemble=True
)
```

**Parameters:**
- `estimator` : sklearn classifier with `fit` and `predict_proba`.
- `method` : {'sigmoid', 'isotonic'} (default 'sigmoid'). Calibration method.
- `cv` : int (default 5) or sklearn cross-validator. Number of folds or cv splitter.
- `n_jobs` : parallel jobs (None = serial).
- `ensemble` : bool (default True). If True, trains k classifier-calibrator pairs (one per cv fold) and averages probabilities. If False, fits calibrator on cross-validated predictions from a single base classifier.

**Fit and predict:**
```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

base_clf = LogisticRegression()
calibrated_clf = CalibratedClassifierCV(base_clf, method='isotonic', cv=5)
calibrated_clf.fit(X_train, y_train)
y_proba_calibrated = calibrated_clf.predict_proba(X_test)[:, 1]  # proba for positive class
```

**Default behavior (ensemble=True):** Automatically prevents training data leakage by training k pairs on cv folds.

**Alternative (ensemble=False):**
```python
calibrated_clf = CalibratedClassifierCV(base_clf, method='isotonic', cv=5, ensemble=False)
```
Uses `cross_val_predict` to generate unbiased predictions on training data, fits calibrator on those, then retrains base classifier on all data.

### 6. Documented Guidance: No Training Data, Isotonic Overfitting

From https://scikit-learn.org/stable/modules/calibration.html (sklearn 1.9.0):

**On not calibrating on training data:**
> "Ideally, the calibrator is fit on a dataset independent of the training data used to fit the classifier in the first place. This is because performance of the classifier on its training data would be better than for novel data. Using the classifier output of training data to fit the calibrator would thus result in a biased calibrator that maps to probabilities closer to 0 and 1 than it should."

**On isotonic overfitting:**
> "However, it is not advised to use isotonic calibration with too few calibration samples (<<1000) since it tends to overfit."

Also in the method description:
> "Sigmoid Method: Can be 'sigmoid' which corresponds to Platt's method (i.e. a logistic regression model)."
> "Isotonic Method: Can be 'isotonic' which is a non-parametric approach."

**Date verified:** 2026-09-04 (all signatures and documentation from official sklearn 1.9.0 docs)

**Caveats / limits:**
- `CalibrationDisplay` only exists in sklearn >= 0.24; older code uses manual `calibration_curve` + matplotlib.
- `IsotonicRegression.fit()` takes (X_scores, y_binary), not (X, y_proba); this is non-standard and can confuse users. X_scores are 1-D raw classifier outputs.
- `CalibratedClassifierCV` with `ensemble=True` returns probability averages; for binary classification, this is equivalent to averaging the calibrated scores, which is correct but not the same as fitting a single calibrator on all data.
- `method='isotonic'` will internally use `IsotonicRegression` with `y_min=0, y_max=1` to ensure valid probabilities.

**Recommendation:**
- Show both methods side-by-side: isotonic on a true-prevalence OOT hold-out (large enough, >~100–200 samples for a synthetic dataset) and Platt for comparison.
- Use `CalibratedClassifierCV` with `ensemble=False` if you want to fit the calibrator on a manually selected hold-out (not cv folds), to have full control over which data is used. Otherwise, use `ensemble=True` (default) for automatic leakage prevention.
- Show the reliability diagram improvement (CalibrationDisplay before and after) to visualize how the curve snaps toward the diagonal.

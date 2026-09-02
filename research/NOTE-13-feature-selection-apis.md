# NOTE-13: sklearn 1.9.0 feature selection APIs and knee/elbow method (2026-09-02)

**Answer:** sklearn 1.9.0 provides SelectKBest (filter), RFE/RFECV (wrapper), SequentialFeatureSelector (forward/backward, direction='forward'|'backward'), SelectFromModel (embedded), and mutual_info_* scoring functions. Knee/elbow method is a heuristic to find inflection point in performance-vs-#features curve. load_breast_cancer has 569 samples × 30 features.

**Evidence:**

| Component | API Signature | Version | Source | Date |
|-----------|---------------|---------|--------|------|
| **SelectKBest** | `SelectKBest(score_func=f_classif, *, k=10)` → selects k top features by score | sklearn 1.9.0 | https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html | 2026-09-02 |
| **f_classif** | `f_classif(X, y)` → returns (f_statistic, p_values) for classification; ANOVA F-value per feature | sklearn 1.9.0 | https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html | 2026-09-02 |
| **f_regression** | `f_regression(X, y, *, center=True, force_finite=True)` → returns (f_statistic, p_values) for regression; univariate linear regression F-scores | sklearn 1.9.0 | https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html | 2026-09-02 |
| **mutual_info_classif** | `mutual_info_classif(X, y, *, discrete_features='auto', n_neighbors=3, copy=True, random_state=None, n_jobs=None)` → returns MI scores (nat units) for each feature | sklearn 1.9.0 | https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html | 2026-09-02 |
| **mutual_info_regression** | `mutual_info_regression(X, y, *, discrete_features='auto', n_neighbors=3, copy=True, random_state=None, n_jobs=None)` → returns MI scores for regression targets | sklearn 1.9.0 | https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html | 2026-09-02 |
| **RFE** | `RFE(estimator, n_features_to_select=None, *, step=1, verbose=0, importance_getter='auto')` → recursive elimination until k features remain; attributes: support_, ranking_, n_features_ | sklearn 1.9.0 | https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html | 2026-09-02 |
| **RFECV** | `RFECV(estimator, *, step=1, min_features_to_select=1, cv=None, scoring=None, verbose=0, n_jobs=None, importance_getter='auto')` → auto-tunes n_features via cross-validation; returns n_features_, support_, cv_results_ | sklearn 1.9.0 | https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html | 2026-09-02 |
| **SequentialFeatureSelector** | `SequentialFeatureSelector(estimator, n_features_to_select='auto', *, direction='forward', scoring=None, cv=5, tol=1e-3, n_jobs=None)` with direction ∈ {'forward', 'backward'} | sklearn 1.9.0 | https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html | 2026-09-02 |
| **SelectFromModel** | `SelectFromModel(estimator, *, threshold='mean', prefit=False, max_features=None, importance_getter='auto')` → selects features with importance > threshold (e.g., Lasso coef_, forest feature_importances_) | sklearn 1.9.0 | https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html | 2026-09-02 |
| **Lasso** | `Lasso(alpha=1.0, *, fit_intercept=True, precompute=False, copy_X=True, max_iter=1000, tol=1e-4, warm_start=False, positive=False, random_state=None, selection='cyclic')` → L1-regularized linear regression; produces sparse coef_ (many zeros) | sklearn 1.9.0 | https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html | 2026-09-02 |
| **load_breast_cancer** | `load_breast_cancer()` → (569 samples, 30 features, binary classification: malignant/benign); X.shape=(569, 30), y.shape=(569,) | sklearn 1.9.0 | https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html | 2026-09-02 |

**Knee/Elbow Method Definition:**

The knee (or elbow) method is a heuristic approach for parameter optimization that identifies the inflection point in a performance curve. The algorithm finds the point of maximum curvature (the "knee" or "elbow"), representing an optimal balance between system performance and operational cost. In feature selection, it is applied by:
1. Computing model performance (e.g., cross-validation score) for increasing numbers of selected features.
2. Plotting #features (x-axis) vs. performance score (y-axis).
3. Identifying the elbow point where marginal performance gains plateau, beyond which adding features provides little benefit.
4. Selecting the feature count at (or just after) the elbow to avoid overfitting and unnecessary complexity.

*Sources:* https://www.scikit-yb.org/en/latest/api/cluster/elbow.html; "Elbow Method" is widely documented in clustering literature; application to feature selection follows the same principle.

**Caveats / limits:**

- **SelectKBest:** Univariate filter; ignores feature interactions and multicollinearity. Must set k explicitly or use "all" as a placeholder in hyperparameter search.
- **f_classif/f_regression:** Assumes linear relationships; may miss non-linear dependencies captured by mutual information or tree-based methods.
- **mutual_info_* methods:** Computationally more expensive than f-scores (uses k-NN entropy estimation); suitable for non-linear feature discovery.
- **RFE:** Greedy algorithm; computationally expensive for large feature sets (requires refitting model O(n_features) times). Does not guarantee globally optimal selection. Performance depends on the base estimator's importance weights.
- **RFECV:** Adds cross-validation overhead; results can vary with CV fold count and random seeds. Best used when the optimal feature count is unknown.
- **SequentialFeatureSelector:** Forward selection starts with empty set (may miss feature interactions at the beginning); backward selection may suffer from collinearity interactions. Both use CV internally (expensive).
- **SelectFromModel with Lasso:** Only applicable to regression/linear models or models with coef_ attribute. L1 penalty strength (alpha in Lasso) must be tuned; too high → all zeros, too low → no selection.
- **Knee/Elbow heuristic:** Subjective; "maximum curvature" can be ambiguous on noisy curves. Visual inspection recommended; algorithmic knee-detection (e.g., kneedle library) helps but is not in sklearn core.
- **load_breast_cancer dataset:** Only 30 features; suitable for demonstration but may not show dramatic feature-selection effects on performance. Consider using synthetic datasets or larger feature sets (e.g., from DS-5 regression dataset) for comparing selection methods.

**Recommendation:**

- **For DS-10 chapter:** Pin sklearn==1.9.0 (reuse from NOTE-5).
- **Import statements:**
  ```python
  from sklearn.feature_selection import (
      SelectKBest, f_classif, f_regression, 
      mutual_info_classif, mutual_info_regression,
      RFE, RFECV, SequentialFeatureSelector, SelectFromModel
  )
  from sklearn.linear_model import Lasso
  from sklearn.datasets import load_breast_cancer
  ```
- **Dataset:** Use load_breast_cancer() for binary classification demo (569×30, balanced-ish: 357 benign, 212 malignant). Suitable for filter/wrapper/embedded comparison.
- **Knee/elbow detection workflow:**
  ```python
  import numpy as np
  import matplotlib.pyplot as plt
  
  # For each k in range, compute CV score
  ks = range(1, max_features + 1)
  scores = []  # e.g., from SelectKBest with different k values
  # Plot and identify elbow visually or via kneedle algorithm
  plt.plot(ks, scores)
  plt.xlabel("Number of Features")
  plt.ylabel("Cross-Validation Score")
  plt.title("Knee/Elbow Plot")
  plt.show()
  # Manual or algorithmic elbow detection; recommend 5-10 features as starting point
  ```
- **Method comparison (for DS-10 section):**
  - **Filter (SelectKBest + f_classif/f_regression):** Fastest, ignores interactions, baseline.
  - **Wrapper (SequentialFeatureSelector forward/backward, RFE, RFECV):** Slower, accounts for estimator performance, can overfit to CV splits.
  - **Embedded (Lasso, tree importance via SelectFromModel):** Medium cost, directly tied to model objective, prone to bias toward high-variance features.
- **Cross-validation safety:** Always apply feature selection *inside* the outer CV loop to avoid selection leakage (e.g., pipeline with SequentialFeatureSelector as a preprocessing step).

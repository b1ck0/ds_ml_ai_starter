# NOTE-11: imbalanced-learn APIs and resampling pipeline safety (2026-09-02)

**Answer:** imbalanced-learn 0.14.2 provides stable resampling and ensemble APIs compatible with scikit-learn; `fit_resample()` must be called INSIDE a pipeline or AFTER train/test split to avoid leakage. Key classes: `RandomUnderSampler`, `RandomOverSampler`, `SMOTE`, `BalancedBaggingClassifier`, `EasyEnsembleClassifier`, and `imblearn.pipeline.Pipeline` (extends sklearn.pipeline.Pipeline to support resampling).

**Evidence:**

| API | Import Path | Signature | Key Parameters | Source |
|-----|-------------|-----------|-----------------|--------|
| **RandomUnderSampler** | `from imblearn.under_sampling import RandomUnderSampler` | `RandomUnderSampler(*, sampling_strategy='auto', random_state=None, replacement=False)` | `sampling_strategy`: ratio control (default 'auto'=balance); `random_state`: seed; `replacement`: with/without replacement | https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.RandomUnderSampler.html |
| **RandomOverSampler** | `from imblearn.over_sampling import RandomOverSampler` | `RandomOverSampler(*, sampling_strategy='auto', random_state=None, shrinkage=None)` | `sampling_strategy`: ratio control; `random_state`: seed; `shrinkage`: optional bootstrap smoothing | https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html |
| **SMOTE** | `from imblearn.over_sampling import SMOTE` | `SMOTE(*, sampling_strategy='auto', random_state=None, k_neighbors=5)` | `sampling_strategy`: ratio control; `k_neighbors`: # nearest neighbors for synthetic generation (default 5) | https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html |
| **BalancedBaggingClassifier** | `from imblearn.ensemble import BalancedBaggingClassifier` | `BalancedBaggingClassifier(n_estimators=10, estimator=None, *, warm_start=False, sampling_strategy='auto', replacement=False, n_jobs=None, random_state=None, verbose=0)` | `n_estimators`: # of learners; `estimator`: base classifier (default DecisionTreeClassifier); `sampling_strategy`: resampling ratio; `n_jobs`: parallelization | https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedBaggingClassifier.html |
| **EasyEnsembleClassifier** | `from imblearn.ensemble import EasyEnsembleClassifier` | `EasyEnsembleClassifier(n_estimators=10, estimator=None, *, warm_start=False, sampling_strategy='auto', replacement=False, n_jobs=None, random_state=None, verbose=0)` | `n_estimators`: # of AdaBoost learners (default 10); `estimator`: base AdaBoost classifier; `sampling_strategy`: undersampling ratio; `random_state`: seed | https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.EasyEnsembleClassifier.html |
| **imblearn.pipeline.Pipeline** | `from imblearn.pipeline import Pipeline` | `Pipeline(steps, *, transform_input=None, memory=None, verbose=False)` | `steps`: list of (name, transformer/sampler/estimator) tuples; supports `fit_resample()` on intermediate samplers | https://imbalanced-learn.org/stable/references/generated/imblearn.pipeline.Pipeline.html |

### Package Version
- **Current:** imbalanced-learn 0.14.2 (verified 2026-09-02)
- **Source:** https://imbalanced-learn.org/stable/install.html
- **sklearn compatibility:** Works with scikit-learn 1.9.0 (requires sklearn >= 1.0.0)

### Resampler Common Parameters

All resamplers (`RandomUnderSampler`, `RandomOverSampler`, `SMOTE`) have the same interface as sklearn transformers with one key difference: they provide **`fit_resample(X, y)`** instead of just `fit()` and `transform()`.

| Method | Behavior | When to Use |
|--------|----------|------------|
| `fit(X, y)` | Analyzes data to determine resampling strategy | Rare; not needed for most workflows |
| `fit_resample(X, y)` | Analyzes data AND returns resampled (X_resampled, y_resampled) | Direct resampling outside a pipeline |
| `fit_transform(X, y)` | In pipeline: calls `fit_resample()`; outside pipeline: error | Only inside `imblearn.pipeline.Pipeline` |

### CRITICAL: Resampling Inside Pipeline vs. Outside (Leakage Prevention)

**SAFE PATTERN (inside pipeline for cross-validation):**
```python
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

pipe = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('clf', LogisticRegression(random_state=42))
])
# Resampling happens INSIDE each CV fold, preventing data leakage
scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='roc_auc')
```

**SAFE PATTERN (outside pipeline after train/test split):**
```python
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Resample ONLY training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

clf = LogisticRegression(random_state=42)
clf.fit(X_train_resampled, y_train_resampled)
y_pred = clf.predict(X_test)  # Evaluate on ORIGINAL test set
```

**DANGEROUS (leakage):**
```python
# WRONG: Resampling entire dataset before split
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)  # Leaks test data info into training
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2)
```

### Pipeline Differences from sklearn.pipeline.Pipeline

| Aspect | sklearn.Pipeline | imblearn.Pipeline |
|--------|------------------|-------------------|
| **Intermediate transforms** | Only `fit` / `transform` | Supports `fit_resample` in addition |
| **fit_transform behavior** | Chains `fit(X, y).transform(X)` | For samplers: calls `fit_resample(X, y)` |
| **Applicability** | Standard ML pipelines | Imbalanced classification pipelines with resampling |
| **Inherited from** | sklearn | sklearn with imbalanced-learn extension |

**Note:** Inside an imblearn.Pipeline, `fit_transform(X, y)` behaves DIFFERENTLY than `fit(X, y).transform(X)` for samplers—this is intentional to support resampling workflows.

**Recommendation:**
1. Always use `imblearn.pipeline.Pipeline` when combining resamplers with classifiers in cross-validation.
2. For manual resampling: call `fit_resample()` AFTER `train_test_split()` to prevent leakage.
3. For BalancedBaggingClassifier and EasyEnsembleClassifier: these handle resampling internally; no need to call fit_resample separately.
4. Set `random_state=42` on all resamplers and models for reproducibility.
5. Use `sampling_strategy='auto'` (default) to let the algorithm decide the target ratio, or specify manually (e.g., `sampling_strategy=0.5` for 1:1 ratio after resampling).

**Caveats / limits:**
- SMOTE requires at least `k_neighbors + 1` samples in the minority class; may fail on very rare classes.
- SMOTE generates synthetic samples; it does NOT work well on categorical features without preprocessing (encode first).
- RandomUnderSampler discards majority samples; may lose important information. RandomOverSampler duplicates minority samples; may cause overfitting.
- BalancedBaggingClassifier and EasyEnsembleClassifier use AdaBoost internally; different from sklearn's standard Bagging.
- The `transform_input` parameter in Pipeline is only active when metadata routing is enabled (sklearn >= 1.6).
- Imbalanced-learn version 0.14.2 requires scikit-learn >= 1.0.0 but is tested and compatible with sklearn 1.9.0.


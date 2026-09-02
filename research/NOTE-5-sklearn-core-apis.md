# NOTE-5: scikit-learn core APIs and metrics (2026-09-02)

**Answer:** scikit-learn 1.9.0 (released 2026-06-02, Python >=3.11) provides all required imputation, preprocessing, model, CV, and metrics APIs; notably, `root_mean_squared_error()` is the canonical function (added v1.4), while `mean_squared_error()` no longer accepts a `squared` parameter.

**Evidence:**

| Component | API | Version | Source |
|-----------|-----|---------|--------|
| **Package** | scikit-learn | 1.9.0 (2026-06-02) | https://pypi.org/project/scikit-learn/ |
| **Python requirement** | Python >=3.11 | — | PyPI, official docs |
| **Imputation** | `sklearn.impute.SimpleImputer(missing_values=nan, strategy='mean'\|'median'\|'most_frequent'\|'constant', fill_value=None, copy=True, add_indicator=False, keep_empty_features=False)` | 1.9.0 | https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html |
| **Imputation** | `sklearn.impute.KNNImputer(missing_values=nan, n_neighbors=5, weights='uniform', metric='nan_euclidean', copy=True, add_indicator=False, keep_empty_features=False)` | 1.9.0 | https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html |
| **Imputation** | `sklearn.impute.MissingIndicator(missing_values=nan, features='auto', sparse='auto', error_on_new=True)` | 1.9.0 | https://scikit-learn.org/stable/modules/generated/sklearn.impute.MissingIndicator.html |
| **Splitting** | `train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None)` | 1.9.0 | https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html |
| **Pipeline** | `Pipeline(steps, *, transform_input=None, memory=None, verbose=False)` | 1.9.0 | https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html |
| **ColumnTransformer** | `ColumnTransformer(transformers, *, remainder='drop', sparse_threshold=0.3, n_jobs=None, transformer_weights=None, verbose=False, verbose_feature_names_out=True)` | 1.9.0 | https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html |
| **CV** | `cross_val_score(estimator, X, y=None, *, groups=None, scoring=None, cv=None, n_jobs=None, verbose=0, params=None, pre_dispatch='2*n_jobs', error_score=nan)` | 1.9.0 | https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html |
| **CV Splitter** | `StratifiedKFold(n_splits=5, *, shuffle=False, random_state=None)` | 1.9.0 | https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html |
| **Preprocessing** | `StandardScaler(*, copy=True, with_mean=True, with_std=True)` | 1.9.0 | https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html |
| **Preprocessing** | `MinMaxScaler(feature_range=(0,1), *, copy=True, clip=False)` | 1.9.0 | https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html |
| **Encoding** | `OneHotEncoder(*, categories='auto', drop=None, sparse_output=True, dtype=np.float64, handle_unknown='error', min_frequency=None, max_categories=None, feature_name_combiner='concat')` | 1.9.0 | https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html |
| **Encoding** | `OrdinalEncoder(*, categories='auto', dtype=np.float64, handle_unknown='error', unknown_value=None, encoded_missing_value=nan, min_frequency=None, max_categories=None)` | 1.9.0 | https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html |
| **Regression** | `LinearRegression(*, fit_intercept=True, copy_X=True, tol=1e-06, n_jobs=None, positive=False)` | 1.9.0 | https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html |
| **Regression** | `RandomForestRegressor(n_estimators=100, *, criterion='squared_error', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=1.0, bootstrap=True, n_jobs=None, random_state=None, ...)` | 1.9.0 | https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html |
| **Regression** | `HistGradientBoostingRegressor(loss='squared_error', *, learning_rate=0.1, max_iter=100, max_leaf_nodes=31, max_depth=None, min_samples_leaf=20, l2_regularization=0.0, max_features=1.0, random_state=None, ...)` | 1.9.0 | https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html |
| **Classification** | `RandomForestClassifier(n_estimators=100, *, criterion='gini', max_depth=None, max_features='sqrt', bootstrap=True, n_jobs=None, random_state=None, ...)` | 1.9.0 | https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html |
| **Classification** | `HistGradientBoostingClassifier(loss='log_loss', *, learning_rate=0.1, max_iter=100, max_leaf_nodes=31, max_depth=None, min_samples_leaf=20, l2_regularization=0.0, random_state=None, ...)` | 1.9.0 | https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html |
| **Classification** | `LogisticRegression(*, C=1.0, l1_ratio=0.0, dual=False, tol=0.0001, fit_intercept=True, solver='lbfgs', max_iter=100, random_state=None, ...)` | 1.9.0 | https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html |
| **Metrics (RMSE)** | `root_mean_squared_error(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average')` | 1.4+ (canonical in 1.9.0) | https://scikit-learn.org/stable/modules/generated/sklearn.metrics.root_mean_squared_error.html |
| **Metrics (MSE)** | `mean_squared_error(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average')` | 1.9.0 (no `squared=` param) | https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html |
| **Metrics (MAE)** | `mean_absolute_error(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average')` | 1.9.0 | https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html |
| **Metrics (R²)** | `r2_score(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average', force_finite=True)` | 1.9.0 | https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html |

**Caveats / limits:**
- Python 3.11 is the minimum supported version; NumPy and SciPy both require Python >=3.12 (per NOTE-2, may need numpy 1.26.x and scipy 1.17.x for 3.11 compatibility).
- **CRITICAL METRIC CHANGE:** The `squared=` parameter was removed from `mean_squared_error()` in v1.9.0. To compute RMSE, use `root_mean_squared_error()` directly (available since v1.4). The old pattern `mean_squared_error(..., squared=False)` will raise a TypeError.
- `LogisticRegression.penalty` parameter is deprecated as of v1.8; use `l1_ratio` and `C` together instead.
- All preprocessing classes support `set_output(transform="pandas")` for pandas DataFrame outputs (metadata routing in v1.6+).
- `StratifiedKFold` requires numeric class labels; convert categorical targets first.

**Recommendation:**
- **Pin sklearn==1.9.0** in requirements.txt. All 30+ core APIs documented above are stable and available.
- **For RMSE:** Always use `root_mean_squared_error(y_true, y_pred)`, not `mean_squared_error(..., squared=False)`.
- **For MSE, MAE, R²:** Use `mean_squared_error()`, `mean_absolute_error()`, `r2_score()` directly; all are consistent with v1.8 and v1.9.
- **Python 3.11:** If required, pin numpy==1.26.x and scipy==1.17.x per NOTE-2 guidance; sklearn itself remains compatible.
- **Example validation snippet:**
  ```python
  import sklearn
  from sklearn.metrics import root_mean_squared_error, mean_squared_error, mean_absolute_error, r2_score
  print(f"sklearn={sklearn.__version__}")  # Should print 1.9.0
  # Test RMSE on sample data
  y_true = [3, -0.5, 2, 7]
  y_pred = [2.5, 0.0, 2, 8]
  rmse = root_mean_squared_error(y_true, y_pred)
  mse = mean_squared_error(y_true, y_pred)
  mae = mean_absolute_error(y_true, y_pred)
  r2 = r2_score(y_true, y_pred)
  assert rmse == np.sqrt(mse), "RMSE should equal sqrt(MSE)"
  ```

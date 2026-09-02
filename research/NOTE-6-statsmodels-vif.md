# NOTE-6: statsmodels VIF API and multicollinearity thresholds (2026-09-02)

**Answer:** statsmodels 0.15.0 (released 2026-08-27, Python >=3.10) provides `variance_inflation_factor(exog, exog_idx, *, standardize=True)` from `statsmodels.stats.outliers_influence`; standard interpretation: VIF >5 indicates concerning collinearity, VIF >10 indicates severe multicollinearity requiring remediation.

**Evidence:**

| Item | Details | Source |
|------|---------|--------|
| **Package version** | statsmodels 0.15.0 (released 2026-08-27) | https://pypi.org/project/statsmodels/ |
| **Python requirement** | Python >=3.10 | PyPI |
| **VIF function** | `variance_inflation_factor(exog, exog_idx, *, standardize=True)` | https://www.statsmodels.org/stable/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html |
| **Module path** | `statsmodels.stats.outliers_influence.variance_inflation_factor` | Official docs |
| **Parameters** | `exog` (array-like design matrix), `exog_idx` (int column index to analyze), `standardize=True` (bool, default True for numerical stability) | Official docs |
| **Returns** | float: VIF value for the specified column | Official docs |
| **How it works** | Measures "the increase of the variance of the parameter estimates if an additional variable, given by exog_idx is added to the linear regression." Performs auxiliary regression but does not save it. | Official docs |

**Interpretation Thresholds (with rationale):**

| VIF Value | Interpretation | Action |
|-----------|----------------|--------|
| **VIF = 1** | No correlation with other predictors (orthogonal) | ✓ Acceptable |
| **1 < VIF < 5** | Low to moderate collinearity; generally acceptable | ✓ Acceptable |
| **VIF ≥ 5** | Concerning collinearity; parameter estimates have large standard errors. The quoted statsmodels documentation states: "the explanatory variable given by exog_idx is highly collinear with the other explanatory variables, and the parameter estimates will have large standard errors because of this." | ⚠ Investigate; consider removal or combination |
| **VIF ≥ 10** | Severe multicollinearity; unstable coefficients; strong need for remediation (dropping, combining, or regularization) | ✗ Must address |

**Caveats / limits:**
- VIF assumes linear relationships; highly non-linear feature interactions may not be captured.
- VIF is scale-invariant (standardize=True by default); non-numeric columns must be pre-encoded before passing to VIF.
- VIF>5 as a threshold is widely used but not universal; some practitioners use 2.5 (very conservative) or even 10 as the sole cutoff. The SPEC-DS-3 chapter should cite this as a "rule of thumb" rather than a hard rule.
- VIF is undefined for categorical variables; must use one-hot or ordinal encoding first.
- Auxiliary regression inside VIF can fail or produce unstable results if the design matrix has perfect collinearity (VIF → ∞).
- **statsmodels 0.15.0 is slightly newer than the 2026-09-02 verification date; if an older version is in use, check the version history for any changes to VIF's standardization parameter (added in recent versions for numerical stability).**

**Recommendation:**
- **Pin statsmodels==0.15.0** in requirements.txt.
- **Import statement for the chapter:**
  ```python
  from statsmodels.stats.outliers_influence import variance_inflation_factor
  ```
- **Standard usage pattern (to compute VIF for all columns):**
  ```python
  import pandas as pd
  from statsmodels.stats.outliers_influence import variance_inflation_factor
  
  # Assume X is a pandas DataFrame of numeric features (scaled/encoded already)
  vif_data = pd.DataFrame({
      'Feature': X.columns,
      'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
  })
  # Filter for VIF > 5 to identify problematic features
  problematic = vif_data[vif_data['VIF'] > 5]
  ```
- **Interpretation guidance for the chapter:**
  - Cite the >5 threshold as a warning sign; >10 as a red flag.
  - Emphasize that collinearity doesn't violate linear regression assumptions, but inflates standard errors and makes coefficients unstable (demo this with bootstrap resampling in SPEC-DS-3).
  - Recommend iterative removal of the highest-VIF feature, recomputing VIF, until all are <5 (or <10 if a looser tolerance is acceptable).
  - Mention alternatives: combining features (e.g., ratios or PCA), or using regularization (Ridge/Lasso in a forward section).

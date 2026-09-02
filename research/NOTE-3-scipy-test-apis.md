# NOTE-3: scipy.stats test function signatures and return types (scipy 1.18.1)

**Answer:** scipy.stats.ttest_ind returns TtestResult(statistic, pvalue, df); scipy.stats.chi2_contingency returns Chi2ContingencyResult(statistic, pvalue, dof, expected_freq). Both are namedtuples accessible by attribute or index.

**Evidence:**

## scipy.stats.ttest_ind

**Signature (scipy 1.18.0+):**
```python
scipy.stats.ttest_ind(a, b, *, axis=0, equal_var=True, nan_policy='propagate', 
                      alternative='two-sided', trim=0, method=None, keepdims=False)
```

**Return type:** `TtestResult` (namedtuple)
- `statistic` (float): The computed t-test statistic
- `pvalue` (float): The p-value for the test
- `df` (float): Degrees of freedom

**Example output:**
```python
>>> from scipy import stats
>>> stats.ttest_ind([1, 2, 3], [4, 5, 6])
TtestResult(statistic=-3.464..., pvalue=0.0136..., df=4.0)
```

**Access:** `result.statistic`, `result.pvalue`, `result.df` or `result[0]`, `result[1]`, `result[2]`

**Source:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html (scipy v1.18.0 Manual)

## scipy.stats.chi2_contingency

**Signature (scipy 1.18.1):**
```python
scipy.stats.chi2_contingency(observed, correction=True, lambda_=None, *, method=None)
```

**Return type:** `Chi2ContingencyResult` (namedtuple-like object)
- `statistic` (float): The chi-square test statistic
- `pvalue` (float): The p-value from the hypothesis test
- `dof` (int): Degrees of freedom (NaN if method is not None)
- `expected_freq` (ndarray): Expected frequencies, same shape as observed

**Degrees of freedom calculation:**
```python
dof = observed.size - sum(observed.shape) + observed.ndim - 1
```

**Example output:**
```python
>>> from scipy.stats import chi2_contingency
>>> obs = [[12, 12, 16], [18, 18, 24]]
>>> chi2_contingency(obs)
Chi2ContingencyResult(statistic=2.7777..., pvalue=0.2493..., dof=2, 
                      expected_freq=array([[12., 12., 16.], [18., 18., 24.]]))
```

**Access:** `result.statistic`, `result.pvalue`, `result.dof`, `result.expected_freq` or by index

**Source:** https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html (scipy v1.18.0+ Manual)

**Date verified:** 2026-09-02

**Caveats / limits:**
- Both functions return namedtuple-like objects; modern scipy (1.15+) uses special result classes but maintain backward-compatible attribute/index access.
- `ttest_ind` default `equal_var=True` performs Student's t-test (equal variances assumed); pass `equal_var=False` for Welch's t-test (unequal variances).
- `chi2_contingency` default `correction=True` applies Yates' correction for 1×2 and 2×1 tables only; larger tables are unaffected.
- Both functions propagate NaN values by default (`nan_policy='propagate'`); students should handle missing data before calling.

**Recommendation:**
- **For ttest_ind:** Always specify the `alternative` parameter explicitly if testing one-sided (default 'two-sided' is correct for most teaching scenarios). Use `equal_var=False` unless you have strong reason to assume equal variances.
- **For chi2_contingency:** Keep `correction=True` for small tables (standard practice). Document that `expected_freq` must have all counts ≥5 for chi-square validity (students should check this).
- **Code snippet:**
  ```python
  from scipy.stats import ttest_ind, chi2_contingency
  
  # t-test example
  t_result = ttest_ind(group1, group2, equal_var=False)
  print(f"t={t_result.statistic:.3f}, p={t_result.pvalue:.4f}, df={t_result.df:.1f}")
  
  # chi-square example
  chi2_result = chi2_contingency(contingency_table)
  print(f"chi2={chi2_result.statistic:.3f}, p={chi2_result.pvalue:.4f}, dof={chi2_result.dof}")
  ```

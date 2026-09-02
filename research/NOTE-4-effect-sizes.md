# NOTE-4: Effect size formulas — Cohen's d and Cramér's V

**Answer:** 
- **Cohen's d** = (mean₁ − mean₂) / s_pooled, interpreted as 0.2 (small), 0.5 (medium), 0.8 (large); not in scipy.stats, must compute by hand.
- **Cramér's V** = √(χ²/(n·min(k−1, r−1))), ranges 0–1, interpreted as <0.1 (negligible), 0.1–0.3 (weak), 0.3–0.5 (moderate), >0.5 (strong); not in scipy.stats, must compute by hand.

**Evidence:**

## Cohen's d (Two-Group Mean Difference)

**Definition:** Standardized mean difference, quantifying effect size for two-sample comparison.

**Formula:**
```
d = (M₁ − M₂) / s_pooled

where s_pooled = √[((n₁−1)·s₁² + (n₂−1)·s₂²) / (n₁ + n₂ − 2)]
```

- M₁, M₂ = group means
- n₁, n₂ = sample sizes
- s₁, s₂ = group standard deviations

**Interpretation (Cohen, 1988):**
- d = 0.2 → Small effect (noticeable difference, but modest)
- d = 0.5 → Medium effect (clear, practical difference)
- d = 0.8 → Large effect (dominant difference)
- d > 1.0 → Very large effect

**Caveat:** Cohen (1988) emphasized these thresholds are context-dependent; "small," "medium," and "large" are relative to discipline and research goals.

**Source:** Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.). Erlbaum. Referenced in https://stats.libretexts.org/Bookshelves/Applied_Statistics/Business_Statistics_(OpenStax)/10:_Hypothesis_Testing_with_Two_Samples/10.02:_Cohen's_Standards_for_Small_Medium_and_Large_Effect_Sizes

## Cramér's V (Categorical Association Strength)

**Definition:** Standardized measure of association between two categorical variables, derived from chi-square statistic, ranges 0 (no association) to 1 (perfect association).

**Formula:**
```
V = √( χ² / (n·min(k−1, r−1)) )
```

- χ² = chi-square test statistic (from scipy.stats.chi2_contingency)
- n = total sample size
- k = number of columns
- r = number of rows
- min(k−1, r−1) = smaller of (# columns − 1) or (# rows − 1)

**Interpretation (conventional thresholds):**
- V < 0.10 → Negligible association
- V = 0.10–0.30 → Weak association
- V = 0.30–0.50 → Moderate association
- V > 0.50 → Strong association

**Key advantage:** V is normalized for table size, making it comparable across different contingency table dimensions (unlike raw χ²).

**Source:** Cramér, Harald. (1946). Mathematical Methods of Statistics. Princeton University Press. Referenced in https://en.wikipedia.org/wiki/Cram%C3%A9r's_V and https://www.spss-tutorials.com/cramers-v-what-and-why/

**Date verified:** 2026-09-02

**Caveats / limits:**
- **Neither function is in scipy.stats.** A proposal to add Cohen's d to scipy.stats exists (https://discuss.scientific-python.org/t/rfc-adding-cohen-s-d-effect-size-function-to-scipy-stats/2130) but has not been implemented; must compute manually.
- **Cohen's d assumptions:** Assumes approximately normal distributions and homogeneous variances (if using pooled SD); for unequal variances, some authors prefer Hedges' g (uses bias-corrected pooled SD) or Glass's delta (uses control group SD only).
- **Cramér's V assumptions:** Assumes expected frequencies ≥ 5 in at least 80% of contingency table cells (same as chi-square test); violated, results unreliable.
- **Alternative packages:** `pingouin.cohens_d()` and `pingouin.cramers_v()` provide implementations; `effectsize` (R ecosystem) has extensive effect size tools.

**Recommendation:**
- **Compute by hand in the chapter** to teach the concept. Provide reusable helper functions:
  ```python
  import numpy as np
  from scipy.stats import ttest_ind, chi2_contingency
  
  def cohens_d(x, y):
      """Cohen's d for two independent samples."""
      n1, n2 = len(x), len(y)
      var1, var2 = np.var(x, ddof=1), np.var(y, ddof=1)
      pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2))
      return (np.mean(x) - np.mean(y)) / pooled_std
  
  def cramers_v(chi2_stat, n, rows, cols):
      """Cramér's V for chi-square contingency table."""
      min_dim = min(rows - 1, cols - 1)
      return np.sqrt(chi2_stat / (n * min_dim)) if min_dim > 0 else np.nan
  
  # Usage:
  chi2, p, dof, expected = chi2_contingency(contingency_table)
  v = cramers_v(chi2, contingency_table.sum(), *contingency_table.shape)
  ```
- **Link to hypothesis testing:** Emphasize that p-values measure statistical significance (evidence against null), while effect sizes measure practical significance (size of the real difference). A tiny p-value on a large sample can have a trivial effect size—always report both.
- **Bridge to next chapter:** In DS-2 (imputation), revisiting effect sizes shows how missing-data handling affects both significance and practical impact.

# NOTE-DS-20-6: Resampling → miscalibration and King & Zeng prior-correction

**Answer:** A model trained on resampled (under/over-sampled) data learns the resampled prevalence, not the true one; all predicted probabilities are calibrated to the training prevalence, making them systematically biased when applied to data with the true prevalence. King & Zeng (2001) **"Logistic Regression in Rare Events Data"** provides analytic prior-correction: adjust the intercept β₀ ← β₀ + log(y₁/(1−y₁)) − log(ȳ₁/(1−ȳ₁)), where y₁ = true population positive rate, ȳ₁ = sample (training) positive rate. Empirically, fit a calibrator (isotonic or Platt) on a hold-out that reflects the **true prevalence**.

**Evidence:**

### 1. The Resampling → Miscalibration Problem

**Mechanism:** Logistic regression (and most classifiers) learn a decision boundary and output probabilities via P(y=1|X) = sigmoid(β₀ + β₁X₁ + ...). The intercept β₀ encodes the *prior* log-odds of the positive class based on the training data.

When training on resampled data (e.g., undersampled 50/50 instead of true 1% positive):
- The learned β₀ corresponds to a 50% prior.
- At test time on real data (1% prevalence), the learned β₀ is too high.
- All predicted probabilities are **inflated** (shifted toward higher values).

**Example:** A model trained 50/50 learns that the base odds are 1:1 (log-odds = 0). On true 1% data, the model's predicted 0.5 should really be ~0.01. Probabilities are systematically wrong.

**Why this breaks calibration:** A reliability diagram shows the classifier far above the diagonal (over-confident), and Brier score is misleadingly low because the majority class (0s at 1% prevalence) gets high confidence (~0.5) instead of low.

### 2. King & Zeng (2001): Analytic Prior-Correction

From https://gking.harvard.edu/files/0s.pdf (**Political Analysis**, Vol. 9, 2001, pp. 137–163):

**Title:** "Logistic Regression in Rare Events Data"

**Key result:** For logistic regression trained on a case-control / resampled sample with prevalence ȳ₁ (observed in sample), the maximum likelihood estimate of the intercept β̂₀ is biased when the true population prevalence is y₁ ≠ ȳ₁.

**Prior-correction formula for logistic regression:**

```
β₀_corrected = β̂₀ + log(y₁ / (1 − y₁)) − log(ȳ₁ / (1 − ȳ₁))
```

where:
- β̂₀ = learned (uncorrected) intercept from the resampled training data.
- y₁ = true population positive rate (e.g., 0.01 for 1% fraud).
- ȳ₁ = observed positive rate in the training sample (e.g., 0.5 for 50/50 undersampling).
- log(p/(1−p)) = log-odds (logit of prevalence).

**Intuition:** The correction adds back the difference between the true and training log-odds, shifting the intercept to what it should be on the true prevalence.

**Implementation for predict_proba:**
After fitting logistic regression on resampled data:
1. Compute β₀_corrected using the formula above.
2. When predicting, use:
   ```
   z = β₀_corrected + β₁X₁ + β₂X₂ + ...
   p = 1 / (1 + exp(−z))
   ```
   Or more simply, adjust all predicted probabilities using the corrected intercept instead of the training one.

**Limitations:** 
- Only applies analytically to logistic regression; other models (trees, SVMs) do not have a simple intercept-correction formula.
- Assumes the only bias is in the intercept; if resampling also changes the decision boundary (e.g., imbalanced class weights), correction is incomplete.

### 3. Empirical Alternative: Calibrate on True-Prevalence Hold-Out

This is the approach the chapter uses (per spec):

1. Train a classifier on resampled (50/50) data.
2. Reserve a **true-prevalence, out-of-time hold-out** that reflects the real ~1–2% positive rate.
3. Fit a calibrator (isotonic or Platt) on this hold-out.
4. The calibrator learns to map the resampled-trained model's inflated scores back to true probabilities.

**Advantages over analytic prior-correction:**
- Works for any classifier (trees, neural nets, etc.), not just logistic regression.
- Automatically handles any bias in the model, not just intercept shift.
- No need to derive formulas per classifier.

**Trade-off:** Requires reserving labeled data for calibration; the analytic formula requires no extra data.

**Date verified:** 2026-09-04

**Caveats / limits:**
- **King & Zeng applies only to logistic regression:** Other classifiers (especially tree-based) do not have a simple intercept to adjust. The formula is exact for logistic but approximate/inapplicable to others.
- **Resampling-only bias assumption:** The formula assumes the only distortion is due to resampling the prior. If resampling also changed feature distributions (via stratified sampling on a confounding variable), the formula is incomplete.
- **True prevalence must be known:** To apply the formula, you need accurate y₁. If you don't know the true rate, the calibrator approach (learning from a true-prevalence hold-out) is safer.
- **The spec focuses on the empirical route:** The chapter centers on fitting the calibrator on a true-prevalence hold-out, not on deriving / implementing the King & Zeng intercept correction analytically. Keep the prior-correction to the intuition + formula + citation (LO4), then move quickly to the isotonic calibrator (LO5) as the practical fix.

**Recommendation:**
- **Mention King & Zeng early** (section 5, "Why the probabilities lie") to ground the *why* resampling breaks calibration.
- **State the intercept-correction formula exactly** (with the log-odds terms clearly labeled) so readers can follow it and implement it for logistic regression if they choose.
- **Emphasize the empirical calibrator approach** as the practical, generalizable fix in the chapter (section 6, "Isotonic calibration on true-prevalence data").
- **Show the contrast:** Calibrator fit on rebalanced training data (wrong) vs. true-prevalence OOT hold-out (correct), with reliability diagrams before/after.

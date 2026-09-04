# NOTE-DS-20-3: Calibration curves, isotonic regression, and Platt/sigmoid scaling

**Answer:** A calibrated classifier is one where for cases assigned probability p, a fraction ~p are actually positive (alignment of predicted and observed frequencies). A calibration/reliability diagram plots (average predicted probability per bin on x-axis, observed positive fraction on y-axis); perfect calibration is the y=x diagonal. Isotonic regression fits a monotonically non-decreasing step function via the Pool-Adjacent-Violators (PAV) algorithm; it overfits on small samples (<~1000) but is more flexible than Platt/sigmoid scaling (parametric logistic fit). Both methods must be fit on hold-out data separate from the training set to avoid leakage.

**Evidence:**

### 1. What "Calibrated" Means

From https://codefinity.com/courses/v2/75aa05fa-b08d-4685-a9a7-7df97ee1910b/1461dac6-8ac2-409a-a75c-94def2e47347/6c26d1f5-8711-4490-a804-78baf957f389 and https://arize.com/blog-course/what-is-calibration-reliability-curve/:

**Definition:** A probabilistic classifier is **reliable** or **calibrated** if, when looking at the set of cases where it predicted probability p, approximately a fraction p of those cases are actually positive.

More formally: For predicted probabilities grouped into bins, if a bin has average predicted probability p, then the observed frequency of positive outcomes in that bin should be approximately p.

### 2. Calibration Curve (Reliability Diagram) Construction & Interpretation

**How to build it:**
1. Binning: Divide predicted probabilities into bins (e.g., [0–0.1), [0.1–0.2), ..., [0.9–1.0]).
2. Per bin, compute: (a) average predicted probability, (b) observed fraction of positive cases.
3. Plot (a) on x-axis, (b) on y-axis.
4. The diagonal y=x represents perfect calibration.

**Interpretation:**
- Points **above** the diagonal: model is **under-confident** (says p but more than p are positive).
- Points **below** the diagonal: model is **over-confident** (says p but fewer than p are positive).
- Points **on** the diagonal: well-calibrated for that bin.

### 3. Isotonic Regression for Calibration

From https://www.zeroentropy.dev/concepts/isotonic-regression/ and https://scikit-learn.org/stable/modules/calibration.html (sklearn 1.9.0):

**Method:** Fits a monotonically non-decreasing step function to map raw classifier scores to probabilities. The fit is via the **Pool-Adjacent-Violators (PAV) algorithm**, which iteratively merges adjacent bins whose order violates monotonicity.

**Properties:**
- Non-parametric (no assumptions about functional form).
- Flexible: can correct any monotonic distortion, not just sigmoid-shaped biases.
- Creates a step-like function, not smooth.

**Data requirements & failure modes:**
- **Sample size:** Performs well with ≥1000 calibration samples. Below ~500–1000, it tends to **overfit**, creating jagged plateaus at the extremes (top and bottom of score range where data is sparse).
- **Why overfitting occurs:** Most labels cluster in the middle of the score range; the top/bottom few percentiles have few samples. PAV merges sparse bins according to their few data points, which have high variance across resamples.
- **Recommendation (sklearn docs):** "It is not advised to use isotonic calibration with too few calibration samples (<<1000) since it tends to overfit."

### 4. Platt / Sigmoid Scaling

From https://zeroentropy.dev/concepts/platt-scaling/ and https://medium.com/@amehta1_be20/platt-scaling-calibration-0121d4761297:

**Method:** Fits a logistic sigmoid on top of raw classifier scores:
```
p_calibrated = 1 / (1 + exp(A*score + B))
```
where A and B are fit on a hold-out calibration set via maximum likelihood (logistic regression).

**Properties:**
- Parametric (exactly 2 parameters).
- Assumes the calibration error is **sigmoid-shaped** (symmetric around the decision boundary).
- Cheap and simple.

**Data requirements & when to use:**
- Works well with small calibration sets (even <100 samples).
- Best for symmetric miscalibration errors.
- Less flexible than isotonic for arbitrary (e.g., curved) distortions.

**Comparison:**

| Aspect | Isotonic | Platt/Sigmoid |
|--------|----------|---------------|
| **Flexibility** | High (any monotonic distortion) | Medium (sigmoid only) |
| **Parametric** | No (step function) | Yes (2 params: A, B) |
| **Sample requirement** | ≥1000 (overfits <~500) | ≥~50 (robust to small sets) |
| **Overfitting risk** | High on small data | Low |
| **Ranking preservation** | Good | Better (fewer steps = fewer ties) |
| **When to use** | Large calibration sets | Small sets or symmetric bias |

### 5. Preventing Data Leakage: No Training Data, No Resampled Data

From https://scikit-learn.org/stable/modules/calibration.html (sklearn 1.9.0):

**Critical requirement:** "The calibrator should be fit on data independent of the training data used for the classifier to avoid bias."

**Why:** If the calibrator is fit on the classifier's training data, the calibrator will overfit to the training distribution, producing probabilities closer to 0 and 1 than warranted on novel data.

**How sklearn.calibration.CalibratedClassifierCV enforces it:**
- When `ensemble=False`, it uses `cross_val_predict` to generate unbiased predictions on the training set, then fits the calibrator on those out-of-fold predictions.
- When `ensemble=True` (default), it trains k classifier-calibrator pairs on cv folds, ensuring calibration data is never used to train the base classifier.

**For this chapter's context (resampling → miscalibration):**
- A model trained on undersampled (rebalanced) data is biased toward the resampled prevalence.
- The calibrator must be fit on a hold-out set that reflects the **true (population) prevalence**, not the resampled one.
- This hold-out should be an out-of-time hold-out if possible (to avoid temporal leakage too).

**Date verified:** 2026-09-04

**Caveats / limits:**
- Sklearn's documentation does not distinguish between "resampled data" and "wrong prevalence" explicitly; the writer must make the connection that a model trained 50/50 (undersampled) has learned a 50/50 prior, and must be recalibrated to the true rate.
- The isotonic overfitting warning uses "~1000" as a guideline; exact threshold depends on score distribution (sparser tail → lower threshold for overfitting).
- Both methods assume the raw classifier scores are monotonically related to true probability; if not (rare), neither method will help.

**Recommendation:**
- **For the chapter:** Show isotonic on the true-prevalence OOT hold-out with adequate sample size (100–200 is likely sufficient for a synthetic dataset). Mention the >~1000 guideline as a caveat.
- **For Platt:** Show as a comparison on the same hold-out; emphasize it would also work well with a smaller calibration set.
- **For leakage:** Explicitly show what happens if you try to calibrate on the *resampled* training set (wrong prevalence → worse calibration) vs the true-prevalence hold-out (correct).

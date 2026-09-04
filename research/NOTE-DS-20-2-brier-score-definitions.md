# NOTE-DS-20-2: Brier score, proper scoring rules, Murphy decomposition, and Brier skill score

**Answer:** Brier score is BS = mean over samples of (predicted_probability − outcome)², is a strictly proper scoring rule (rewards calibration and discrimination equally), decomposes into reliability/calibration + resolution + uncertainty (Murphy decomposition), and the Brier skill score normalizes it against the base-rate reference: BSS = 1 − BS / BS_ref, where BS_ref = base_rate × (1 − base_rate).

**Evidence:**

### 1. Brier Score Definition & Formula

From https://en.wikipedia.org/wiki/Brier_score and https://scores.readthedocs.io/en/1.1.0/tutorials/Brier_Score.html:

**Binary classification formula:**
```
BS = (1/N) * Σ(p_i - y_i)²
```
where p_i is the predicted probability and y_i ∈ {0, 1} is the true outcome. Lower is better (range 0–1, perfect score is 0).

**Strictly proper scoring rule property:** A scoring rule is strictly proper if the expected score is **uniquely minimized when reporting the true probability**, not when biasing forecasts. Brier score is strictly proper, meaning it incentivizes honest probability estimates over both over- and under-confidence.

### 2. Murphy Decomposition

From https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.2985 and https://arxiv.org/pdf/2005.01835:

**Three components:**

The Brier score decomposes as:
```
BS = Reliability + (max(BS) - Resolution) - Uncertainty
```

or equivalently (cleaner form):
```
BS = Calibration - Resolution + Uncertainty
```

**Component definitions:**
- **Reliability (Calibration):** Measures systematic bias; how often predicted probability p occurs, the actual fraction of positives should be ~p. Quantifies deviation from perfect calibration (the diagonal in a reliability diagram).
- **Resolution (Refinement):** Measures the informational content of forecasts; ability to distinguish between different outcomes (higher is better). Variance of the forecast probabilities across samples.
- **Uncertainty:** Baseline variance of the outcome; q(1−q) where q is the base rate. This is the irreducible randomness.

Lower reliability + higher resolution = better Brier score.

### 3. Brier Skill Score (BSS)

From https://www.emergentmind.com/topics/brier-skill-score and https://metricgate.com/blogs/brier-score-explained/:

**Formula:**
```
BSS = 1 - (BS / BS_ref)
```

where BS_ref is a reference baseline, typically the "climatological" (base-rate) Brier score:
```
BS_ref = base_rate × (1 - base_rate)
```

**Interpretation:**
- BSS = 0: Model performs no better than random/base-rate baseline.
- BSS > 0: Model outperforms baseline.
- BSS = 1: Perfect predictions.
- BSS < 0: Model worse than baseline.

**Critical for imbalanced data:** On imbalanced datasets (e.g., 1% positives), the majority class dominates raw Brier score. A naive "always predict 0" model gets BS ≈ 0.01 × 0.99 = 0.0099, which looks excellent. BSS normalizes by base rate, exposing this; BSS of a trivial predictor is exactly 0. **This is the reason the spec emphasizes "imbalanced-data caveat"** — report both Brier and BSS to avoid being misled.

**Date verified:** 2026-09-04

**Caveats / limits:**
- Brier score treats calibration and discrimination symmetrically; a model with poor calibration but high AUC (sharp but miscalibrated) still has high Brier score — this is intentional and proper.
- The base rate choice for BS_ref is crucial; using the wrong prevalence gives misleading BSS. On resampled data (e.g., 50/50 undersampled), BS_ref must use the *true* population base rate, not the resampled one — this is the central tension the chapter addresses.
- Murphy decomposition interpretation requires care; "resolution" is often called "refinement" or "discrimination"; terminology varies by field.

**Recommendation:**
- Compute Brier by hand on a toy example (3–5 predictions) to cement the formula.
- Always report BSS alongside raw Brier on imbalanced data; BSS exposes models that fool raw Brier.
- When comparing Brier scores across train/validation/OOT splits, ensure BS_ref uses the *true* prevalence for each set, not the training/resampled prevalence.

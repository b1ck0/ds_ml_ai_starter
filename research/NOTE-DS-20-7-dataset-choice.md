# NOTE-DS-20-7: Dataset for calibration chapter — synthetic with drift vs. real credit-card fraud

**Answer:** A **reproducible synthetic dataset** via `sklearn.datasets.make_classification` with ~1–2% positive rate + synthetic timestamps and mild feature drift is the recommended choice. A real dataset (Kaggle Credit Card Fraud Detection) exists but requires download, authentication, and license confirmation; the synthetic approach gives full control, reproducibility, and visible effects.

**Evidence:**

### 1. Real Dataset Option: Kaggle Credit Card Fraud Detection

From https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud (ULB - Université Libre de Bruxelles):

**Dataset characteristics:**
- **Size:** 284,807 transactions with 31 features.
- **Imbalance:** 492 frauds (0.172% positive rate); class distribution is 99.83% non-fraud, 0.17% fraud.
- **Features:** 28 PCA-transformed features (V1–V28, anonymized for privacy), plus Time (seconds since first transaction), Amount.
- **Time span:** Two days in September 2013; transactions by European cardholders.
- **License:** CC-0 (public domain) via Kaggle.
- **Size on disk:** ~150 MB compressed.
- **Availability:** Requires Kaggle account and API authentication (or manual download).

**Characteristics relevant to the chapter:**
- ✓ Real, imbalanced rare-event data.
- ✓ Has a time column (Transaction Time in seconds).
- ✓ Runs quickly on CPU.
- ✗ Does not have date/datetime column; time is in seconds since the start of the dataset (not calendar dates).
- ✗ Requires download + API setup.
- ✗ Anonymized features; no interpretability.

### 2. Recommended Option: Synthetic Dataset with Drift

**Approach:** Use `sklearn.datasets.make_classification()` with strategic parameters:

```python
from sklearn.datasets import make_classification
import numpy as np

n_samples = 10000
# ~1% positive rate (240 positives, 9760 negatives)
X, y = make_classification(
    n_samples=n_samples,
    n_features=20,
    n_informative=10,
    n_redundant=3,
    weights=[0.99, 0.01],  # 1% positive
    random_state=42,
    class_sep=0.8
)

# Add synthetic timestamp column (0 to 100 days)
timestamps = np.random.uniform(0, 100, size=n_samples)
X = np.column_stack([X, timestamps])

# Optional: Add mild temporal drift (feature relationship changes slightly over time)
# e.g., feature importance of X[:, 0] decreases linearly with time
drift_slope = -0.5
X[:, 0] = X[:, 0] * (1 + drift_slope * timestamps / 100)
```

**Advantages:**
- ✓ **Reproducible:** Seeded, runs identically every time.
- ✓ **Full control:** Set base rate, feature count, class separation, drift magnitude explicitly.
- ✓ **Fast:** Generates instantly, no download.
- ✓ **Clear effects:** Undersampling breaks calibration exactly because you know the true 1% rate. Drift is visible because you created it.
- ✓ **Self-contained:** No external dependencies beyond sklearn.
- ✓ **Pedagogical:** Reader can tweak parameters to see effects.

**Disadvantages:**
- ✗ **Not real:** Features are synthetic; no real-world domain insight.
- ✗ **Too clean:** Real data has missing values, outliers, correlated features (all absent here).

### 3. Comparison for the Chapter

| Criterion | Real (Kaggle) | Synthetic |
|-----------|---------------|-----------|
| **Authenticity** | Real fraud data | Toy data |
| **Imbalance** | 0.17% (ultra-rare) | 1% (configurable) |
| **OOT split clarity** | Two-day span (limited drift) | Controlled drift visible |
| **Reproducibility** | Requires download; results depend on Kaggle API | Instant, seeded, 100% reproducible |
| **Runtime** | ~1–2 min on CPU | <1 sec |
| **Pedagogical control** | Fixed; can't change imbalance | Full control; students can tweak |
| **Setup complexity** | High (auth, download) | Low (one function call) |
| **Licensing** | CC-0; clear | N/A (generated) |

### 4. Recommendation

**Use synthetic with drift.** Reasons:
1. **Spec intent:** The chapter teaches the *mechanics* of OOT validation, calibration, and resampling → miscalibration. A synthetic dataset where every effect is controlled and visible achieves this better than real data.
2. **Reproducibility & speed:** Every reader gets identical results; no wait for downloads.
3. **Clarity:** When the OOT split shows lower metrics than random split, the reader understands why: temporal drift is baked in and is the only difference between the two splits.
4. **Teaching:** Readers can easily modify base rate, drift, and sample size to experiment.

**If the writer prefers real data (e.g., for authenticity):**
- Kaggle Credit Card Fraud Detection is acceptable.
- **Caveat:** The 0.17% base rate is ultra-rare and makes the synthetic "1–2%" representation unrealistic. The 492 frauds might not be enough for a true-prevalence OOT hold-out (you'd split 492 into calibration + test, leaving very few per set).
- Must document the download process and ensure the link/API stays valid (Kaggle can remove datasets).
- The two-day span is small for drift; OOT split is less dramatic than with synthetic.

**Date verified:** 2026-09-04

**Caveats / limits:**
- The Kaggle dataset's ultra-low 0.17% rate (vs. spec's 1–2% target) means undersampling to 50/50 gives a much more extreme prior shift than the synthetic example would. This is pedagogically *harder* to understand at first.
- The synthetic dataset's "drift" (modified feature relationship) is simplistic; real temporal drift involves concept drift, distribution shift, and seasonal patterns. The chapter should mention real drift is more complex (see DS-17 for monitoring).
- If using Kaggle's dataset, verify the license and attribution; the spec says "clearly-licensed," which CC-0 is, but confirm it hasn't changed.

**Recommendation for the writer:**
- **Default:** Synthetic with ~1% positive rate, timestamps, and mild drift (e.g., one feature's relationship weakens over time).
- **Optional:** If authenticity is paramount, swap in Kaggle after confirming file exists, license is current, and that the ultra-rare 0.17% imbalance is acceptable for the pedagogy (may require adjusting the undersampling target from 50/50 to something higher, like 30/70, to preserve calibration-set size).

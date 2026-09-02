# NOTE-8: Imputation dataset — seaborn penguins verified (2026-09-02)

**Answer:** seaborn's Palmer Penguins dataset has genuine, non-trivial missing values across multiple columns (2–11 NaN per column in a 344-row dataset); confirmed as runnable via `seaborn.load_dataset("penguins")` and ideal for SPEC-DS-2 imputation examples.

**Evidence:**

| Item | Details | Source |
|------|---------|--------|
| **Dataset name** | Palmer Penguins (allisonhorst/penguins) | GitHub: https://github.com/allisonhorst/penguins; seaborn mirror: https://github.com/mwaskom/seaborn-data |
| **License** | Public domain / CC0 (Palmer Station LTER, maintained by Allison Horst) | Original repository |
| **Load method** | `seaborn.load_dataset("penguins")` or `pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv')` | seaborn documentation |
| **Rows** | 344 observations | Verified |
| **Columns** | 7 (species, island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex) | Verified |

### Missing Value Counts (per Column)

| Column | Non-null | **Missing (NaN)** | Percentage |
|--------|----------|------------------|-----------|
| species | 344 | 0 | 0.0% |
| island | 344 | 0 | 0.0% |
| bill_length_mm | **342** | **2** | 0.6% |
| bill_depth_mm | **342** | **2** | 0.6% |
| flipper_length_mm | **342** | **2** | 0.6% |
| body_mass_g | **342** | **2** | 0.6% |
| sex | **333** | **11** | 3.2% |

**Total missing:** ~19 values across 344×7 = 2,408 cells (0.8% sparsity).

**Significance for SPEC-DS-2:**
- **Multiple missing-data mechanisms demonstrated:** The same 2 rows have NaN in all four numeric columns (likely MCAR: missing completely at random, e.g., a sensor failure for a specific penguin). The 11 missing values in `sex` are scattered (likely MAR or MNAR if sex determination failed in the field for some observations).
- **Diverse strategies applicable:** SimpleImputer strategies (mean/median on numeric columns; most_frequent on sex) are instructive. KNNImputer can use species/island as context. MissingIndicator can flag which penguins had measurement failures.
- **Realistic scale:** Not so sparse that imputation is trivial, not so dense that the data is unusable without imputation.

### How to Load and Inspect

```python
import pandas as pd
import seaborn as sns

# Load the dataset
penguins = sns.load_dataset("penguins")

# Inspect missing values
print(penguins.isnull().sum())
# Output:
# species               0
# island                0
# bill_length_mm        2
# bill_depth_mm         2
# flipper_length_mm     2
# body_mass_g           2
# sex                  11
# dtype: int64

print(f"Shape: {penguins.shape}")  # (344, 7)
print(f"Sparsity: {penguins.isnull().sum().sum() / penguins.size * 100:.2f}%")  # ~0.8%

# Show the rows with missing numeric measurements
print(penguins[penguins['bill_length_mm'].isnull()])

# Show the distribution of missing sex values
print(penguins[penguins['sex'].isnull()][['species', 'island']].value_counts())
```

**Caveats / limits:**
- **Not time-series:** Missing values do not depend on temporal order; appropriate for SPEC-DS-2 (independent rows) but not for forecasting (SPEC-DS-9).
- **Categorical missingness:** The `sex` column is categorical (Male/Female/NaN); SimpleImputer with `strategy='most_frequent'` will fill all NaN with "Male" (most common). The chapter should discuss whether this is meaningful.
- **Tiny dataset:** 344 rows is small for model training; SPEC-DS-2 is pedagogical imputation, not production regression. Do not use penguins for SPEC-DS-5 (NYC taxi regression requires more data for realistic train/valid/holdout splits).
- **Version compatibility:** seaborn>=0.11.0 includes the penguins dataset. Older versions require `sns.load_dataset("iris")` or external CSV URL instead (but iris has no missing values).

**Recommendation:**
- **Use seaborn penguins for SPEC-DS-2** without reservation. It is:
  - ✓ Freely available (public domain).
  - ✓ Built into seaborn (no extra download).
  - ✓ Has genuine, non-trivial missing values (not synthetic).
  - ✓ Small enough to run instantly in a notebook.
  - ✓ Well-documented and widely recognized.
- **Validation snippet (for the chapter or test):**
  ```python
  import seaborn as sns
  penguins = sns.load_dataset("penguins")
  # Verify shape and missing counts
  assert penguins.shape == (344, 7), "Unexpected shape"
  assert penguins['bill_length_mm'].isnull().sum() == 2, "Expected 2 missing in bill_length_mm"
  assert penguins['sex'].isnull().sum() == 11, "Expected 11 missing in sex"
  assert penguins.isnull().sum().sum() == 19, "Expected 19 total missing values"
  print("✓ Penguins dataset verified.")
  ```
- **For SPEC-DS-5 (NYC taxi regression):** Do NOT use penguins; see NOTE-7 recommendation to synthesise larger taxi data.
- **Missing-value documentation:** In the chapter, use penguins to teach:
  - MCAR hypothesis (measurement system failure → all metrics NaN for 2 rows).
  - MAR / MNAR hypothesis (sex determination field work → scattered 11 NaN).
  - Show that imputation choice (mean vs KNN vs drop) changes downstream results; emphasize the fit-on-train-only discipline.

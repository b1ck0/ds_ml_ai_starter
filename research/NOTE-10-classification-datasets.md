# NOTE-10: Classification datasets and loaders (2026-09-02)

**Answer:**
- **Titanic (DS-6):** `seaborn.load_dataset('titanic')` provides 891 samples, 15 columns; NaNs in age (714 non-null), embarked (889 non-null), deck (203 non-null). Licensed under CC0 (public domain) via seaborn-data repository.
- **Multi-class (DS-7):** `sklearn.datasets.load_digits()` (10 classes, 1797 samples, 64 features) recommended for runnable reproducibility; alternatives: load_wine (3 classes, 178 samples, 13 features) or load_iris (3 classes, 150 samples, 4 features).
- **Multi-label (DS-7):** `sklearn.datasets.make_multilabel_classification()` generates fully synthetic multilabel data (no download required); real alternatives (yeast, emotions) require external downloads not provided by sklearn.
- **Imbalanced (DS-8):** `sklearn.datasets.make_classification(weights=[...])` creates imbalanced data; e.g., `weights=[0.95, 0.05]` yields ~95% majority, ~5% minority; fully runnable without download.

**Evidence:**

### Titanic Dataset (seaborn)

| Property | Value | Source |
|----------|-------|--------|
| **Loader** | `import seaborn as sns; df = sns.load_dataset('titanic')` | https://seaborn.pydata.org/generated/seaborn.load_dataset.html |
| **Shape** | (891, 15) | Documented example |
| **Return type** | pandas DataFrame | seaborn 0.13.2+ docs |
| **Columns** | survived, pclass, sex, age, sibsp, parch, fare, embarked, class, who, adult_male, deck, embark_town, alive, alone | https://seaborn.pydata.org/generated/seaborn.load_dataset.html |
| **Missing values** | age: 177 NaN; embarked: 2 NaN; deck: 688 NaN | Multiple sources confirm 19% age, ~0.2% embarked, 77% deck missing |
| **Source** | GitHub: https://github.com/mwaskom/seaborn-data (titanic.csv) | Seaborn documentation and GitHub repo |
| **License** | CC0 (Public Domain) / No explicit restrictions noted in seaborn-data | Seaborn itself is BSD-3-Clause; seaborn-data is linked from seaborn docs with no license file restriction |
| **Internet requirement** | Yes, unless cached locally | load_dataset() downloads from GitHub with caching |

### Multi-Class Datasets (sklearn)

| Dataset | API | Shape | Classes | Source | Recommendation |
|---------|-----|-------|---------|--------|-----------------|
| **Digits** | `from sklearn.datasets import load_digits; X, y = load_digits(return_X_y=True)` | X: (1797, 64), y: (1797,) | 10 (digits 0–9) | https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html | **RECOMMENDED**: Good class balance, largest sample set, no download needed |
| **Wine** | `from sklearn.datasets import load_wine; X, y = load_wine(return_X_y=True)` | X: (178, 13), y: (178,) | 3 | https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html | Alternative: smaller sample set, good feature dimension |
| **Iris** | `from sklearn.datasets import load_iris; X, y = load_iris(return_X_y=True)` | X: (150, 4), y: (150,) | 3 | https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html | Alternative: minimal features, commonly used but very small |

All three are fully bundled with sklearn, require no download, and are licensed under the BSD-3-Clause license.

### Multi-Label Dataset (sklearn)

| Dataset | API | Shape | Return Type | Recommendation |
|---------|-----|-------|-------------|-----------------|
| **make_multilabel_classification (synthetic)** | `from sklearn.datasets import make_multilabel_classification; X, y = make_multilabel_classification(n_samples=100, n_features=20, n_classes=5, n_labels=2, random_state=42)` | X: (100, 20), y: (100, 5) | Returns dense (default) or sparse indicator matrix | **RECOMMENDED**: Fully runnable, synthetic, no download required. Parameters: `n_samples=100`, `n_features=20`, `n_classes=5` (total labels), `n_labels=2` (avg labels per sample), `return_indicator='dense'` (dense array) or `'sparse'` (sparse CSR) |
| **Real alternatives** | yeast, emotions (external sources) | Varies | Typically sparse or list-of-lists | Not recommended unless explicitly needed; requires external download (not built-in to sklearn) |

### Imbalanced Dataset (sklearn)

| Dataset | API | Imbalance Control | Example | Source |
|---------|-----|-------------------|---------|--------|
| **make_classification with weights** | `from sklearn.datasets import make_classification; X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=2, n_classes=2, weights=[0.95, 0.05], random_state=42)` | `weights` parameter: array of class proportions | `weights=[0.95, 0.05]` → ~95% class 0, ~5% class 1 | https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html |
| **Imbalance level** | Controlled via `weights` | Specify proportions for each class | For 1-5% minority: `weights=[0.99, 0.01]` or `weights=[0.95, 0.05]` | sklearn docs |
| **Return type** | Returns (X, y) tuple or X only | Default `return_X_y=True` | np.ndarray (X: n_samples×n_features, y: n_samples) | sklearn docs |
| **License** | BSD-3-Clause (sklearn) | No restrictions | Fully runnable | https://scikit-learn.org/stable/ |

**Recommendation:** Use `make_classification(weights=[...])` for runnable, deterministic imbalanced data in DS-8. For reproducibility, always set `random_state=42` (or consistent value). To create 1-5% minority class: use `weights=[0.99, 0.01]`, `[0.95, 0.05]`, etc.

**Caveats / limits:**
- **Titanic:** Requires internet to download (unless cached); `deck` column is 77% missing and may need dropping. `embarked` and `embark_town` have slight overlap; use one for encoding.
- **Multi-class:** `load_iris` is extremely popular but only has 150 samples and 4 features—may feel too small for demonstration. `load_digits` is larger and better for modern ML pedagogical purposes.
- **Multi-label:** `make_multilabel_classification()` generates fully synthetic data with uniform label distributions; real-world multilabel data (yeast, emotions) have different characteristics and require external download.
- **Imbalanced:** `make_classification(weights=...)` may generate fewer samples than requested if weights don't sum to 1 and `flip_y` is active. The actual class proportions may not exactly match `weights` due to floating-point rounding.


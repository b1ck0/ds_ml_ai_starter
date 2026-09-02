# NOTE-2: Current stable package versions (2026-09-02)

**Answer:** Pin these versions for Python 3.11+: pandas==3.0.5, numpy==2.5.2, matplotlib==3.11.1, scipy==1.18.1, seaborn==0.13.2.

**Evidence:**

| Package | Latest Stable | Release Date | Python Requirement | Verified Source |
|---------|---------------|--------------|-------------------|-----------------|
| pandas | 3.0.5 | 2026-07-22 | >=3.11 | https://pypi.org/project/pandas/ |
| numpy | 2.5.2 | 2026-08-09 | >=3.12 | https://pypi.org/project/numpy/ |
| matplotlib | 3.11.1 | 2026-07-18 | >=3.11 | https://pypi.org/project/matplotlib/ |
| scipy | 1.18.1 | 2026-08-21 | >=3.12 | https://pypi.org/project/scipy/ |
| seaborn | 0.13.2 | 2024-01-25 | >=3.8 (active support 3.11+) | https://pypi.org/project/seaborn/ |

**Date verified:** 2026-09-02 (all versions checked against PyPI)

**Caveats / limits:**
- **Critical:** numpy (2.5.2) and scipy (1.18.1) both require Python >=3.12, while pandas and matplotlib support 3.11+. If targeting Python 3.11, must use older numpy/scipy versions.
- **Recommendation for Python 3.11 compatibility:** Use numpy 1.26.x (last 3.11-compatible major version) and scipy 1.17.x instead; see PyPI version history for exact stable versions in those series.
- **seaborn 0.13.2** is older (released 2024-01-25) but stable and widely used; newer development may occur but breaking changes are rare in minor releases.
- All five packages have active Python 3.12+ wheel builds and are tested on modern CPython versions (3.12, 3.13, 3.14, 3.15 where applicable).

**Recommendation:**
- **For Python 3.12+:** Use the pinned versions above directly in `requirements.txt`:
  ```
  pandas==3.0.5
  numpy==2.5.2
  matplotlib==3.11.1
  scipy==1.18.1
  seaborn==0.13.2
  ```
- **For Python 3.11 only:** Pin pandas==3.0.5, matplotlib==3.11.1, seaborn==0.13.2, but check PyPI for latest numpy 1.26.x and scipy 1.17.x versions.
- **Verification step:** In the chapter's environment setup, run `import pandas, numpy, matplotlib, scipy, seaborn; print(f"pandas={pandas.__version__}, numpy={numpy.__version__}, scipy={scipy.__version__}, seaborn={seaborn.__version__}")` to confirm versions in use.

## Correction (verified during authoring, 2026-09-03)
sklearn `MissingIndicator` default `features` is `'missing-only'` (not `'auto'`) on scikit-learn 1.9.0 — verified live during DS-2 authoring.

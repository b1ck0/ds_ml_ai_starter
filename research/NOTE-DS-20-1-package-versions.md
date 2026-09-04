# NOTE-DS-20-1: Current stable package versions for calibration chapter

**Answer:** Pin scikit-learn==1.9.0 (2026-06-02), numpy==2.5.2 (2026-08-09), pandas==3.0.5 (2026-07-22), matplotlib==3.11.1 (2026-07-17) for Python 3.12+.

**Evidence:**

| Package | Latest Stable | Release Date | Source |
|---------|---------------|--------------|--------|
| scikit-learn | 1.9.0 | 2026-06-02 | https://pypi.org/project/scikit-learn/ |
| numpy | 2.5.2 | 2026-08-09 | https://pypi.org/project/numpy/ |
| pandas | 3.0.5 | 2026-07-22 | https://pypi.org/project/pandas/ |
| matplotlib | 3.11.1 | 2026-07-17 | https://pypi.org/project/matplotlib/ |

**Date verified:** 2026-09-04 (checked against PyPI)

**Caveats / limits:**
- numpy (2.5.2) requires Python >=3.12; for Python 3.11, use numpy 1.26.x (last 3.11-compatible version).
- All versions are stable and actively maintained; no imminent breaking changes anticipated.
- scikit-learn 1.9.0 is the version used in the official documentation URLs cited for calibration, isotonic, and Brier score APIs.

**Recommendation:**
Pin exactly as shown above in `requirements.txt` for reproducibility. Verify in environment setup:
```python
import sklearn, numpy, pandas, matplotlib
print(f"sklearn={sklearn.__version__}, numpy={numpy.__version__}, pandas={pandas.__version__}, matplotlib={matplotlib.__version__}")
```

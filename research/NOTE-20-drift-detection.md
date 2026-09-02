# NOTE-20: Drift Definitions, PSI Formula, and Open-Source Detection Tools (2026)

**Answer:**
Data drift = P(X) shift (input distribution); Concept drift = change in X→y relationship (also "model drift"). PSI formula: Σ(Actual% – Expected%) × ln(Actual%/Expected%). Thresholds: <0.1 stable, 0.1–0.25 moderate, >0.25 significant. scipy.stats.ks_2samp (scipy 1.18.1 from NOTE-2) performs KS test. Evidently v0.7.21 (2026-03-10) is the maintained open-source drift tool.

**Evidence:**

### Drift Definitions (Authoritative)

**Data Drift:** Changes in the statistical properties of input data, i.e., a change in P(X) over time. Features or distributions of input differ from training. Source: Deepchecks and DataCamp tutorials; Evidently AI production documentation.

**Concept Drift:** Evolution of the underlying relationship between inputs and outputs; the function F changes over time. Also called "model drift". Changes in the X→y mapping render the model invalid/inaccurate. Source: Wikipedia (Concept drift), Evidently AI production docs, ArXiv papers on continual learning.

**Model Drift / Prediction Drift:** Synonymous with concept drift; the task the model was designed for changes, or the decision boundary shifts due to external factors (e.g., fraud patterns change).

Reference URLs:
- https://www.evidentlyai.com/ml-in-production/concept-drift
- https://deepchecks.com/data-drift-vs-concept-drift-what-are-the-main-differences/
- https://www.datacamp.com/tutorial/understanding-data-drift-model-drift
- https://en.wikipedia.org/wiki/Concept_drift

### PSI Formula and Thresholds

**Formula:**
```
PSI = Σ_i (Actual%_i – Expected%_i) × ln(Actual%_i / Expected%_i)
```
Where:
- Actual% = proportion in current period (binned for continuous variables)
- Expected% = proportion in reference period (training data)
- Σ sums over all bins/categories

**Thresholds (industry standard):**
- PSI < 0.1: No significant change; population is stable
- 0.1 ≤ PSI < 0.25: Moderate shift; investigate potential drift
- PSI ≥ 0.25: Significant drift; retrain model recommended

Sources:
- https://machinelearningplus.com/deployment/population-stability-index-psi/
- https://www.fiddler.ai/blog/measuring-data-drift-population-stability-index
- https://coralogix.com/ai-blog/a-practical-introduction-to-population-stability-index-psi/
- https://support.minitab.com/en-us/model-ops/monitor-deployed-models/details-about-the-population-stability-index-psi/

### Kolmogorov-Smirnov Test (scipy)

**API (scipy 1.18.1):**
```python
from scipy.stats import ks_2samp
statistic, pvalue = ks_2samp(data1, data2, alternative='two-sided', method='auto')
```

**Parameters:**
- `data1, data2`: Two samples of observations (arrays)
- `alternative`: {'two-sided', 'less', 'greater'}; default 'two-sided'
- `method`: {'auto', 'exact', 'asymp'}; auto is recommended
- Returns: statistic (float), pvalue (float)

**Interpretation:** If p-value < 0.05 (or chosen α), reject the null hypothesis that the distributions are equal; drift is detected.

Source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html (scipy v1.18.0+ Manual)

**Note on scipy version:** scipy==1.18.1 verified 2026-08-21 (see NOTE-2); ks_2samp API has remained stable across scipy versions ≥1.11.

### Open-Source Drift Tool: Evidently

**Package:** evidently  
**Latest Stable Version:** 0.7.21  
**Release Date:** 2026-03-10  
**PyPI:** https://pypi.org/project/evidently/  
**Python Support:** 3.8+  

**Features:**
- Data drift detection (supports multiple statistical tests including KS, chi-square, Jenson-Shannon divergence)
- Concept drift detection
- ML model quality monitoring
- Data quality checks
- Report generation (HTML dashboards)
- Integration with MLOps pipelines

**Install:**
```bash
pip install evidently==0.7.21
```

**Basic Usage (reference):**
```python
from evidently.report import Report
from evidently.metrics import DataDriftTable

report = Report(metrics=[DataDriftTable()])
report.run(reference_data=train_df, current_data=test_df)
```

Source: https://pypi.org/project/evidently/ (verified 2026-09-02)

**Caveats / limits:**
- **PSI thresholds are guidelines, not absolutes:** Business context and baseline stability matter. A PSI of 0.12 in a high-variance domain may not warrant retraining; conversely, 0.15 in a fraud-detection model demands investigation.
- **Continuous vs categorical:** PSI requires binning continuous variables; choice of bin edges affects the result. Evidently automates binning, but manual tuning may be needed for domain-specific requirements.
- **KS test assumptions:** The KS test assumes continuous distributions and independent samples. For categorical data or small samples, Fisher's exact or chi-square tests may be more appropriate.
- **Concept drift detection is harder:** Unlike data drift (input shift), concept drift requires labeled data or performance metrics from production to detect. Evidently can monitor model predictions and outcomes, but true concept drift diagnosis requires ground truth.
- **Evidently 0.7.21 age:** Released 2026-03-10; check for newer patches (0.7.22+) or major versions on PyPI if the chapter is written after March 2026. Major version updates (0.8.x+) may introduce API changes.

**Recommendation:**
- Use both **PSI and KS-test** in drift-detection code: PSI for percentile-based interpretation, KS test for statistical hypothesis testing.
- Pin scipy==1.18.1 (from NOTE-2) and evidently==0.7.21 for reproducibility.
- For the runnable drift_detection.py, simulate drift on synthetic data (e.g., shift the mean or variance of features), compute PSI and KS statistics, and plot them over time to show degradation.
- Document the PSI thresholds in comments; note they are industry heuristics, not universal cutoffs.
- Reference Evidently for monitoring in production; mention that open-source tools like NannyML and Alibi Detect exist but are less widely maintained.
- **Caveats to highlight in text:** Drift ≠ model degradation; a drifted input distribution might not hurt a robust model. Always validate with recent labeled data (champion/challenger testing) before retraining.

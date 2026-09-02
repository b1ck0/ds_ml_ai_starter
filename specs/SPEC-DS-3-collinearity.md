# SPEC-DS-3: Collinearity & keeping features minimal

**Status:** approved
**Subject:** Data Science
**Section:** Worked Examples
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8
**Prerequisites:** SPEC-DS-1

## Intent
Define the vocabulary (independent variable / feature vs label / target) and show why correlated
features are dangerous — unstable coefficients, misleading importance, wasted capacity. Establish the
recurring theme of the whole course: the fewest features that do the job.

## Learning objectives
- LO1 — Define feature (independent variable), label (target/dependent variable), and design matrix.
- LO2 — Detect collinearity via a correlation heatmap and VIF (variance inflation factor).
- LO3 — Show concretely how collinearity destabilises linear-model coefficients (fit twice on resampled data, watch coefficients swing).
- LO4 — State the "minimum viable feature set" principle and how it connects to overfitting and interpretability.

## Scope
In: correlation matrix, VIF, coefficient instability demo, dropping/combining collinear features.
Out: formal feature selection algorithms (→ SPEC-DS-10), regularization as a remedy (→ regression chapter / theory).

## Outline
1. Vocabulary — X vs y, one row = one independent observation.
2. Correlation heatmap on a dataset with correlated columns.
3. VIF: what it measures, how to read it (>5 / >10 rules of thumb — verify).
4. Instability demo — bootstrap-refit a linear model; plot coefficient spread with vs without a collinear pair.
5. Remedies — drop one, combine (e.g., ratio/PCA-lite), or regularize (forward-link).
6. Pitfalls — high importance ≠ causation; collinearity hides in one-hot dummies too.

## Assets to produce
- Prose: "Data Science/Worked Examples/collinearity.md"
- Code: "Data Science/Worked Examples/code/collinearity.py"
- Artefacts: correlation heatmap; coefficient-spread plot; VIF table.

## Claims to ground (Haiku, before writing)
- [ ] Verify the VIF computation approach and current API (statsmodels `variance_inflation_factor`) + statsmodels version on PyPI.
- [ ] Verify the standard VIF interpretation thresholds against an authoritative source.
- [ ] Reuse NOTE-2 for pandas/numpy/matplotlib; confirm scikit-learn LinearRegression API.

## Acceptance criteria
- [ ] AC1 — LOs delivered. AC2 — collinearity.py runs + artefacts + snippet-check. AC3 — VIF API/thresholds + statsmodels version grounded. AC4 — vocabulary bridged from a Java dev's model.

## Gates
Entry: approved; notes landed. Exit: DoD checklist.

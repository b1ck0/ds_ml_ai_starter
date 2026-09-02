# SPEC-DS-2: Imputation — filling missing values without lying to your model

**Status:** done (written by Sonnet, grounded by Haiku, independently reviewed + merged 2026-09-03)
**Subject:** Data Science
**Section:** Worked Examples
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8
**Prerequisites:** SPEC-DS-1 (EDA), SPEC-DS-0 (env)

## Intent
Real data has holes. A Java dev's instinct is a null check; in ML, dropping or naively filling rows
changes the distribution your model learns. Teach why we impute, the common strategies, and their
failure modes — on a dataset with genuine missingness.

## Learning objectives
- LO1 — Detect and quantify missingness; distinguish MCAR / MAR / MNAR at an intuition level.
- LO2 — Apply mean/median imputation with scikit-learn `SimpleImputer` and explain what it distorts (variance, correlations).
- LO3 — Apply better strategies: KNN imputation (`KNNImputer`), and the "add a missingness indicator column" trick.
- LO4 — Explain why "just drop the rows" is often the worst option, and when it's acceptable.

## Scope
In: SimpleImputer (mean/median/most_frequent), KNNImputer, indicator columns, drop-vs-impute trade-off, fit-on-train/apply-to-valid discipline.
Out: IterativeImputer/MICE (mention + link forward), time-series imputation (→ forecasting chapter).

## Outline
1. What & why — nulls in Java vs missing values in a model's input matrix.
2. See the holes — missingness table + heatmap on the dataset (penguins has real NaNs).
3. Mean/median imputation and what it does to the variance/correlation (show before/after).
4. KNNImputer and indicator columns; when each helps.
5. Leakage warning — impute using statistics learned on TRAIN only.
6. Pitfalls — imputing the target, inflating confidence, category vs numeric imputation.

## Assets to produce
- Prose: "Data Science/Worked Examples/imputation.md"
- Code: "Data Science/Worked Examples/code/imputation.py"
- Artefacts: missingness heatmap; a before/after distribution plot; a small comparison table.

## Claims to ground (Haiku, before writing)
- [ ] Reuse NOTE-2 versions; verify current scikit-learn version + the exact APIs of `SimpleImputer`, `KNNImputer`, `MissingIndicator` (module path, params) on that version.
- [ ] Confirm the dataset with real missingness (penguins from NOTE-1 has NaNs — verify counts) or pick another documented one.

## Acceptance criteria
- [ ] AC1 — LO1–LO4 delivered. AC2 — imputation.py runs, produces artefacts, snippet-check passes.
- [ ] AC3 — sklearn version + imputer APIs grounded in a NOTE. AC4 — leakage discipline shown, Java framing present.

## Gates
Entry: approved; notes landed. Exit: DoD checklist.

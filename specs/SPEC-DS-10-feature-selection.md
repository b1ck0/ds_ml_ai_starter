# SPEC-DS-10: Feature Selection — the fewest features that work

**Status:** approved
**Subject:** Data Science
**Section:** Worked Examples (advanced)
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8
**Prerequisites:** SPEC-DS-3 (collinearity), SPEC-DS-5/DS-6 (a model to select for)

## Intent
Make the course-long theme operational: find the smallest feature set that keeps performance. Teach
the problem it solves (cost, overfitting, interpretability) and the practical methods.

## Learning objectives
- LO1 — State why fewer features help (variance, cost, latency, interpretability) and the risk of over-selecting.
- LO2 — Apply filter (univariate/correlation), wrapper (forward & backward selection / RFE), and embedded (L1, tree importance) methods.
- LO3 — Use the "knee/elbow" method: plot performance vs number-of-features and pick the inflection.
- LO4 — Validate selection inside cross-validation to avoid selection leakage.

## Scope
In: filter/wrapper/embedded methods, forward & backward selection, RFE, knee method, CV-safe selection.
Out: full AutoML search (→ SPEC-DS-11), SHAP-based selection (mention + link).

## Outline
1. What & why — the minimal-feature-set principle, costs of extra features.
2. Filter methods — univariate scores, correlation pruning.
3. Wrapper methods — forward selection, backward elimination, RFE; the compute cost.
4. Embedded — L1/Lasso, tree importances.
5. The knee method — performance-vs-#features curve; choosing the elbow.
6. Doing it safely — selection inside CV; pitfalls (selecting on the whole dataset = leakage).

## Assets to produce
- Prose: "Data Science/Worked Examples/feature-selection.md"
- Code: "Data Science/Worked Examples/code/feature_selection.py"
- Artefacts: performance-vs-#features knee plot; selected-feature table; a forward-vs-backward comparison.

## Claims to ground (Haiku, before writing)
- [ ] Reuse NOTE-5; confirm sklearn APIs: SelectKBest, f_classif/f_regression, RFE / RFECV, SequentialFeatureSelector (forward/backward), SelectFromModel, Lasso.
- [ ] Confirm the definition/rationale of the knee/elbow method from an authoritative source.
- [ ] Pick a dataset with enough features to make selection meaningful (e.g. sklearn load_breast_cancer, or the regression dataset from DS-5) — verify.

## Acceptance criteria
- [ ] AC1 — LOs delivered. AC2 — code runs, produces the knee plot + selected sets, snippet-check passes. AC3 — sklearn selection APIs + knee method grounded. AC4 — CV-safe selection shown; the minimal-set theme tied back.

## Gates
Entry: approved; notes landed. Exit: DoD checklist.

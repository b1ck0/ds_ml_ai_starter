# SPEC-DS-4: Train / Validation / Holdout — why we split, and leakage

**Status:** done (written by Sonnet, grounded by Haiku, independently reviewed + merged 2026-09-03)
**Subject:** Data Science
**Section:** Worked Examples
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8
**Prerequisites:** SPEC-DS-1, SPEC-DS-3

## Intent
A Java dev knows unit vs integration tests; here the analogue is train vs validation vs holdout. Teach
why we hold data back, what data leakage is, and how to split correctly for the SIMPLE (non-temporal)
case where every row is independent and there is no future-into-past leakage. Forecasting's different
split is deferred to SPEC-DS-9.

## Learning objectives
- LO1 — Explain the purpose of each split (fit / tune / final unbiased estimate) using the testing analogy.
- LO2 — Perform a correct split with `train_test_split`, including stratification for classification.
- LO3 — Demonstrate a concrete leakage bug (scaler/imputer fit on the whole dataset) and its inflated score, then fix it with a Pipeline.
- LO4 — Explain k-fold cross-validation and when to prefer it over a single validation split.

## Scope
In: holdout split, stratification, the fit-on-train-only rule, Pipeline to prevent leakage, k-fold CV.
Out: temporal/rolling splits (→ SPEC-DS-9), nested CV (mention + link).

## Outline
1. Why hold data back — the testing-set analogy; what "generalisation" means.
2. A clean split; stratify for imbalanced classes.
3. Leakage demo — fit a StandardScaler on all data → optimistic score; fix with `Pipeline` inside CV.
4. k-fold cross-validation — the picture, the trade-off, `cross_val_score`.
5. Pitfalls — leaking via preprocessing, via duplicate rows, via the target; peeking at the holdout.

## Assets to produce
- Prose: "Data Science/Worked Examples/train-valid-holdout-split.md"
- Code: "Data Science/Worked Examples/code/splitting_and_leakage.py"
- Artefacts: a bar chart of "leaky vs correct" validation scores; a CV-fold diagram (matplotlib).

## Claims to ground (Haiku, before writing)
- [ ] Verify scikit-learn APIs: `train_test_split` (stratify param), `Pipeline`, `cross_val_score`, `StratifiedKFold` on the current sklearn version (reuse NOTE-2/DS-2 sklearn note).
- [ ] Confirm a small dataset to use (penguins or a sklearn built-in like `load_breast_cancer`) — verify the loader.

## Acceptance criteria
- [ ] AC1 — LOs delivered. AC2 — code runs, leakage gap is real and reproduced, artefacts produced, snippet-check passes. AC3 — sklearn APIs grounded. AC4 — testing analogy used.

## Gates
Entry: approved; notes landed. Exit: DoD checklist.

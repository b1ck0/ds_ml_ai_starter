# SPEC-DS-1: Hypothesis Testing & Exploratory Data Analysis

**Status:** approved
**Subject:** Data Science
**Section:** Worked Examples
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8
**Prerequisites:** DS Local Environment Setup (Python, pandas, Jupyter) — write that first or note it as assumed.

> This is the **first backlog item scoped as a demonstration** of the pipeline. Approved by the
> owner on 2026-09-02 to kick off the pipeline. It shows the shape every chapter spec should take.

## Intent
A senior Java dev is comfortable asserting behaviour with unit tests but has never asked "are these
two samples *really* different, or is it noise?". This chapter introduces statistical hypothesis
testing through hands-on EDA on a classic tabular dataset, so the reader leaves able to compare
distributions and reason about p-values and effect size the way they already reason about
confidence in a flaky test.

## Learning objectives
After this chapter the reader can:
- LO1 — Load a tabular dataset in pandas and run a first EDA pass (shape, dtypes, missingness, summary stats, a few plots).
- LO2 — State a null and alternative hypothesis for a concrete question about the data.
- LO3 — Pick and run the right test (t-test for means of two groups; chi-square for categorical association) and read the p-value correctly.
- LO4 — Distinguish statistical significance from effect size, and explain why a tiny p-value on a huge sample can still be uninteresting.

## Scope
In scope: EDA workflow; comparing two distributions; t-test, chi-square; p-value interpretation;
effect size (Cohen's d / Cramér's V).
Out of scope: imputation (→ SPEC-DS-2), multiple-testing correction beyond a one-line mention,
Bayesian testing, ANOVA (mention as "more than two groups → next tier").

## Outline (section-by-section)
1. **What & why** — "significant" as a precise claim, not a vibe; the courtroom analogy for null hypothesis.
2. **The dataset** — one classic EDA dataset (see research brief: confirm Titanic vs a better teaching set).
3. **EDA pass** — `df.info()`, missingness, `describe()`, a histogram + a boxplot by group.
4. **A concrete question → a test** — e.g. "did fare differ by survival?" → t-test; "was survival associated with sex?" → chi-square.
5. **p-value & effect size** — what the number means, what it does not; effect size as the "so what".
6. **Pitfalls** — p-hacking, huge-N significance, assuming normality; how to see each.
7. **Recap & next** — bridges to imputation (DS-2) and collinearity (DS-3).

## Assets to produce
- Prose: `Data Science/Worked Examples/hypothesis-testing-and-eda.md`
- Code: `Data Science/Worked Examples/code/hypothesis_testing_eda.py` (+ optional `.ipynb`)
- Dataset: documented load step (a built-in loader if available; otherwise a URL + licence in the chapter)
- Artefacts: `Data Science/Worked Examples/artefacts/` — fare histogram, boxplot-by-group, a small results table

## Claims to ground (Haiku research brief — do BEFORE writing)
- [ ] Confirm the best teaching dataset for distribution comparison + EDA (Titanic vs alternatives such as
      penguins/seaborn, Iris, tips) — pick one, with a reachable source URL and licence.
- [ ] Pin current versions of pandas, numpy, matplotlib, scipy, seaborn (verify on PyPI, note dates).
- [ ] Verify the exact scipy API for the tests used (`scipy.stats.ttest_ind`, `scipy.stats.chi2_contingency`)
      and their return shapes on the installed version.
- [ ] Verify the effect-size definitions used (Cohen's d, Cramér's V) against an authoritative source.

## Acceptance criteria
- [ ] AC1 — LO1–LO4 each delivered by a section → evidence: section map in PR body.
- [ ] AC2 — every snippet runs against the pinned env → evidence: verify.sh compile pass + a run log producing the artefacts.
- [ ] AC3 — dataset link, versions, test APIs, and effect-size formulas all grounded → evidence: NOTE ids.
- [ ] AC4 — audience-fit: at least the null-hypothesis and p-value ideas are bridged from a testing/CI mental model; artefacts shown.

## Gates
Entry: owner approves this spec; research NOTEs for all four grounding items have landed.
Exit: all ACs satisfied; snippets run and reproduce the artefacts; links resolve; fresh-Sonnet review
sign-off; architect merge. (See `docs/definition-of-done.md`.)

# SPEC-DS-20: Trustworthy probabilities on imbalanced data — out-of-time validation, Brier, precision@top-N, and isotonic calibration

**Status:** written by Sonnet, grounded by Haiku (NOTE-DS-20-1..7) — pending independent review + architect merge
**Subject:** Data Science
**Section:** Worked Examples
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8
**Prerequisites:** SPEC-DS-4 (train/validation/holdout & leakage), SPEC-DS-6 (classification metrics),
SPEC-DS-8 (class imbalance — undersampling & ensembles). Related: SPEC-DS-17 (production monitoring),
SPEC-DS-9 (the *other* kind of time split).

## Intent
The book teaches class imbalance (DS-8) and classification metrics (DS-6), but stops short of the
techniques a practitioner actually reaches for when a **rare-event model has to drive real decisions**
— the exact set the owner met on a recent project and wants to understand in depth. A model can have a
great AUC and still be **useless or misleading** in production for three separate reasons, and this
chapter is the fix for each:

1. **You validated the wrong way.** A random train/test split lets the model peek at the future — in
   production you always predict *forward in time*. An **out-of-time (OOT) split** validates the way
   the model will actually be used and exposes temporal drift.
2. **You measured the wrong thing.** Accuracy/AUC don't tell you whether the predicted *probabilities*
   are honest, and your team can only act on a handful of cases. **Brier score** measures probability
   quality directly; **precision@top-N** measures what a capacity-limited team actually gets when it
   works the N highest-scored cases.
3. **Your probabilities are silently wrong.** To learn from rare positives you probably **undersampled
   to ~50/50** — so the model now believes the event is far more common than it is, and a predicted
   "0.9" does **not** mean 90%. The fix is **calibration fit on a hold-out that reflects the TRUE
   prevalence** (not the rebalanced one) — and **isotonic regression** is the flexible, non-parametric
   way to do it.

The chapter walks all four on one imbalanced dataset, with reliability diagrams and real numbers, and
shows the honest (lower) OOT metrics next to the optimistic random-split ones. Java-dev framing: a
model's score is like a hash that ranks well but whose *magnitude* is meaningless until you calibrate
it against reality — and you must test it against tomorrow's data, not a shuffled copy of today's.

## Learning objectives
After this chapter the reader can:
- LO1 — Explain **out-of-time validation**: split by a timestamp (train ≤ cutoff, test > cutoff),
  why it beats a random split and k-fold for a model that predicts forward, how it differs from DS-9's
  single-series forecasting split, and how it surfaces drift. Build one and show its metrics are
  (honestly) worse than the random-split ones.
- LO2 — Define and compute the **Brier score** (mean squared error of predicted probability vs.
  outcome), explain why it is a *proper scoring rule* (rewards calibration **and** discrimination),
  its Murphy decomposition (calibration + refinement + uncertainty) at an intuitive level, and the
  imbalanced-data caveat (the majority class can dominate it → the Brier *skill score* / class-wise
  view).
- LO3 — Define and compute **precision@top-N** (rank by score, take the top N, precision = TP/N),
  explain why it is the decision-relevant metric under a fixed action budget (fraud analysts, a
  call-list, a review queue), and relate it to precision@k / lift.
- LO4 — Explain **why undersampling breaks calibration**: a model trained on a rebalanced set outputs
  probabilities calibrated to the *training* prevalence, not the real one, so scores are inflated;
  state the analytic **prior-correction** intuition (the intercept/log-odds shift; rare-events
  logistic regression, King & Zeng 2001) AND the empirical route this chapter uses.
- LO5 — **Calibrate** the model with **isotonic regression** fit on a **true-prevalence, out-of-time**
  hold-out; read a **reliability diagram** (calibration curve) before vs. after; show Brier improving
  and the curve moving onto the diagonal; and contrast isotonic (non-parametric, monotonic step
  function, data-hungry, can overfit) with Platt/sigmoid scaling (parametric) — when to use each.

## Scope
In scope: one imbalanced binary-classification dataset **with a time column**; a baseline model
trained with undersampling (reusing DS-8's technique); an OOT split; Brier, precision@top-N, and
reliability diagrams; isotonic calibration on a true-prevalence OOT hold-out, with a Platt-scaling
comparison. CPU-runnable in a minute, seeded, reproducible.
Out of scope (name + link): multiclass calibration, conformal prediction (mention as a pointer),
full drift monitoring (DS-17), cost-sensitive threshold optimisation beyond a brief tie-in to DS-8's
threshold tuning. Keep the analytic prior-correction to the intuition + formula + citation; the
worked fix is the empirical isotonic-on-true-prevalence route.

## The dataset (architect decision — confirm/allow the writer to swap after grounding)
Default to a **reproducible synthetic dataset** built so every effect is visible and no download is
needed: `make_classification` (or similar) at a realistic rare-event base rate (~1–2% positives),
plus a synthetic **timestamp** with **mild temporal drift** in one feature/relationship so the OOT
split behaves differently from a random split. This lets the chapter *control the true prevalence*,
demonstrate the undersampling-miscalibration precisely, and fix it — the mechanics are the point.
Mention the real-world analogues explicitly (credit-card fraud, credit default, churn) and, if the
grounding turns up a small, clearly-licensed, timestamped real imbalanced dataset that runs fast on
CPU, the writer may use it instead (with a documented licence).

## Outline (section-by-section)
1. **Cold open** — a rare-event model (1% positives) with a great AUC that is quietly useless: shipped
   on a random split, reporting inflated probabilities, evaluated on a metric the ops team can't use.
   Pose the three problems.
2. **What & why** — the "you are here" map; the three fixes; the Java framing (score ranks, but its
   magnitude is meaningless until calibrated; test against tomorrow, not a shuffle of today).
3. **Out-of-time validation** — build the timestamped split; compare random-split vs OOT metrics on
   the same model; show OOT is lower and *why that's the honest number*. Contrast with DS-9's split.
4. **Measuring what matters** — Brier (by hand on a tiny example, then `brier_score_loss`; the
   imbalanced caveat + skill score) and precision@top-N (compute it; vary N; the action-budget story).
5. **Why the probabilities lie** — show the model (trained with undersampling) is badly miscalibrated:
   reliability diagram far off the diagonal, over-predicting. Explain the base-rate shift + the
   prior-correction intuition/formula (grounded).
6. **Isotonic calibration on true-prevalence data** — fit isotonic on a true-prevalence OOT hold-out;
   reliability diagram + Brier before vs. after; the curve snaps to the diagonal, precision@top-N and
   ranking are preserved (calibration is monotonic). Compare to Platt/sigmoid. State the data-size and
   overfitting caveats for isotonic.
7. **Pitfalls & recap** — calibrating on rebalanced data (the trap), calibrating on the training fold
   (leakage), isotonic on too little data, reading Brier without the base-rate context, precision@top-N
   with the wrong N; recap table (problem → metric/technique → sklearn tool).

## Assets to produce
- Prose: `01-data-science/03-worked-examples/15-calibration-ranking-imbalanced.md`
- Code: `01-data-science/03-worked-examples/code/calibration_ranking.py` (self-contained, seeded,
  deps pinned)
- Artefacts: reliability diagrams **before vs. after** calibration, a Brier/precision@top-N comparison
  table (random-split vs OOT; raw vs isotonic vs Platt), under `.../artefacts/` (namespaced, e.g.
  `calib_*.png` — check existing filenames first so nothing is overwritten).
- Coherence: update `docs/curriculum.md` (architect will do this), the DS `README.md` worked-examples
  list, and cross-links to DS-4, DS-6, DS-8, DS-17.

## Claims to ground (Haiku research brief — do BEFORE writing)
- [ ] Package versions to pin: `scikit-learn`, `numpy`, `matplotlib`, `pandas` — current PyPI + dates.
- [ ] Exact definitions/formulas from authoritative sources (cite each): **Brier score** (and that it
      is a strictly proper scoring rule; the Murphy decomposition; the Brier *skill score*);
      **reliability diagram / calibration curve** and what "calibrated" means; **isotonic regression**
      for calibration (monotonic, PAV algorithm) vs **Platt/sigmoid** scaling — data requirements and
      failure modes; **precision@top-N / precision@k** and **lift**.
- [ ] The current scikit-learn API so snippets run: `sklearn.metrics.brier_score_loss`,
      `sklearn.calibration.calibration_curve`, `CalibrationDisplay`, `sklearn.isotonic.IsotonicRegression`,
      `sklearn.calibration.CalibratedClassifierCV(method='isotonic'|'sigmoid')` — signatures + the
      documented guidance on **not** calibrating on the training set and on isotonic overfitting on
      small samples.
- [ ] The **resampling → miscalibration** result and the **prior-correction / rare-events** fix:
      cite an authoritative source (e.g. King & Zeng 2001 "Logistic Regression in Rare Events Data",
      and/or the analytic intercept-adjustment for under/oversampling), stated precisely enough to
      write the intuition + formula correctly.
- [ ] Whether a small, clearly-licensed, timestamped, imbalanced real dataset exists that runs fast on
      CPU (fraud/credit/churn) — else confirm the synthetic-with-drift approach is the right call.

## Acceptance criteria (each maps to evidence)
- [ ] AC1 (LO1) — an OOT split is built and its metrics shown honestly below the random-split ones →
      evidence: runnable code + the comparison table.
- [ ] AC2 (LO2–LO3) — Brier and precision@top-N computed by hand AND via library, with the imbalanced
      caveats → evidence: run log + artefact.
- [ ] AC3 (LO4–LO5) — miscalibration from undersampling shown (reliability diagram), then fixed with
      isotonic on true-prevalence OOT data; Brier improves; before/after diagrams; Platt comparison →
      evidence: the two reliability-diagram artefacts + the metrics table.
- [ ] AC4 — every snippet runs (`check_snippets.py` + a real run log); every formula/version/claim
      grounded (NOTE ids); reference repos not imported.
- [ ] AC5 — audience-fit: the base-rate-shift and "score magnitude is meaningless until calibrated"
      ideas are made concrete; every artefact shown and interpreted.
- [ ] AC6 — renders on GitHub (`check_markdown_render.py` pass; watch `\text{}` escaping in the Brier /
      log-odds / prior-correction formulas — subscripts everywhere).
- [ ] AC7 — repository coherence: curriculum, DS README, and DS-4/6/8/17 cross-links updated.

## Gates
Entry: this spec approved; research NOTEs landed. Exit: all ACs satisfied with evidence; snippets run;
links resolve; fresh-Sonnet review sign-off; architect merge. (See `docs/definition-of-done.md`.)

# SPEC-DS-17: Production Monitoring — drift and when to retrain

**Status:** done (written by Sonnet, grounded by Haiku, independently reviewed + merged 2026-09-03)
**Subject:** Data Science
**Section:** Production Considerations
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8
**Prerequisites:** SPEC-DS-16, SPEC-DS-12
**Nature:** MIXED — drift DETECTION on synthetic data is runnable; the retraining/promotion workflow is
conceptual (grounded).

## Intent
A deployed model silently rots as the world changes. Teach the vocabulary of drift, how to detect it,
when to trigger retraining, and how to decide whether a new model should replace the incumbent — and
why an MLOps pipeline makes all of this a non-event.

## Learning objectives
- LO1 — Distinguish data drift (input distribution shifts), concept drift (X→y relationship shifts), and model/prediction drift; give examples of each.
- LO2 — Detect drift on a stream: compare distributions (PSI / KS-test) and track live metrics; runnable on synthetic drifting data.
- LO3 — Decide when to retrain (scheduled vs triggered by a drift/metric threshold) and why an MLOps pipeline makes retraining cheap and safe.
- LO4 — Decide whether to PROMOTE a candidate: compare new vs old on recent data AND a frozen golden dataset before switching.

## Scope
In: the three drifts; PSI + KS detection on synthetic drift (runnable); retrain triggers; champion/challenger promotion criteria; the MLOps-pipeline payoff.
Out: full monitoring stacks (Evidently/WhyLabs beyond a mention), automated rollback infra depth.

## Outline
1. What & why — models degrade; a passing "test" today fails next quarter as inputs shift.
2. The three drifts, with concrete examples.
3. Detect it (RUNNABLE) — simulate drift; compute PSI and a KS-test; plot the metric decaying over time.
4. When to retrain — schedule vs trigger; the pipeline that makes it a button-press.
5. Should the new model win? — recent-data + golden-dataset comparison; champion/challenger; guardrails.
6. Pitfalls — chasing noise, no golden set, silent label delay, retraining on drifted-but-wrong data.

## Claims to ground (Haiku, before writing)
- [ ] Verify the definitions of data vs concept vs model drift against an authoritative source.
- [ ] Verify the PSI (Population Stability Index) formula + common thresholds, and the KS-test API (scipy.stats.ks_2samp) — reuse NOTE-2/NOTE-3.
- [ ] Verify (reference) that open-source drift tools exist and name one current maintained one (e.g. Evidently) with its version, for the "in practice" aside.

## Assets to produce
- Prose: "Data Science/Production Considerations/monitoring-and-drift.md"
- Code: "Data Science/Production Considerations/code/drift_detection.py"
- Artefacts: distribution-shift plot; PSI/KS over time; a metric-decay chart; a promotion-decision diagram.

## Acceptance criteria
- [ ] AC1 — LOs delivered. AC2 — drift_detection.py runs on synthetic drift, computes PSI + KS, produces artefacts; snippet-check passes. AC3 — drift definitions + PSI/KS grounded; tool name+version grounded. AC4 — the champion/challenger + golden-dataset decision made concrete; MLOps payoff explicit.

## Gates
Entry: approved; notes landed. Exit: DoD checklist.

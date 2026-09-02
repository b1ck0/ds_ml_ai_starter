# SPEC-DS-11: AutoML — letting the machine search the pipeline

**Status:** done (written by Sonnet, grounded by Haiku, independently reviewed + merged 2026-09-03)
**Subject:** Data Science
**Section:** Worked Examples (advanced)
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8
**Prerequisites:** SPEC-DS-5/DS-6, SPEC-DS-10

## Intent
Introduce AutoML: what it automates (preprocessing, model choice, hyperparameters, sometimes
ensembling) and how, using a maintained open-source framework. Frame it honestly — a productivity
tool, not magic — and compare its result to the hand-built models from earlier chapters.

## Learning objectives
- LO1 — Explain what AutoML searches over and the techniques it uses (Bayesian/meta-learning/successive halving).
- LO2 — Run an open-source AutoML framework on a dataset already used in the course and read its leaderboard.
- LO3 — Compare AutoML's best pipeline against the earlier hand-built model (accuracy AND cost/time).
- LO4 — Know AutoML's limits: compute budget, leakage risks, interpretability, when NOT to use it.

## Scope
In: one maintained open-source AutoML framework, a runnable search, leaderboard, comparison, limits.
Out: commercial/cloud AutoML (→ DS cloud chapter), NAS for deep learning (mention).

## Outline
1. What & why — the search space AutoML explores; the "grid search on steroids" framing.
2. Set up + run the framework on a familiar dataset with a small time budget.
3. Read the leaderboard / best pipeline; inspect what it chose.
4. Compare to the earlier hand-built model; discuss the trade-off.
5. Limits & pitfalls — budget, leakage, reproducibility, over-trust.

## Claims to ground (Haiku, before writing) — IMPORTANT
- [ ] Choose a CURRENTLY-MAINTAINED, pip-installable open-source AutoML framework that runs in a CPU sandbox with a small time budget. Candidates to evaluate: auto-sklearn (check maintenance + install pain), FLAML (lightweight, Microsoft), AutoGluon (heavier), TPOT, H2O AutoML. Recommend ONE that installs cleanly on the target Python, with its current version and a minimal working API. Note install size/time.
- [ ] Verify the chosen framework's real API (fit/leaderboard/predict) on its current version from official docs.
- [ ] Confirm the technique claims (how it searches) against the framework's docs/paper.

## Assets to produce
- Prose: "Data Science/Worked Examples/automl.md"
- Code: "Data Science/Worked Examples/code/automl_demo.py"
- Artefacts: leaderboard table; AutoML-vs-handbuilt comparison chart.

## Acceptance criteria
- [ ] AC1 — LOs delivered. AC2 — the AutoML run actually executes in-sandbox (small budget) and the comparison is real; snippet-check passes. If install is infeasible in-sandbox, ESCALATE to the architect before writing rather than faking output. AC3 — framework choice/version/API grounded. AC4 — honest limits section; the grid-search framing used.

## Gates
Entry: approved; grounding (esp. framework installability) landed. Exit: DoD checklist.

# SPEC-DS-19: Bayesian inference — priors, likelihood, and posteriors you can sample

**Status:** approved
**Subject:** Data Science
**Section:** Worked Examples
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8
**Prerequisites:** SPEC-DS-1 (hypothesis testing & EDA — the frequentist view), SPEC-DS-5 (regression),
SPEC-DS-14 (DS theory). Related forward link: SPEC-DS-9 (forecasting — the AR model reappears).

## Intent
The book is entirely frequentist so far: point estimates, p-values, confidence intervals. Bayesian
inference is the other paradigm, and the owner has real notebooks on it. Instead of "the single best
line," a Bayesian fit returns a **distribution over lines** — every parameter comes with honest
uncertainty, and you can ask "what's the probability the slope is positive?" directly. For a senior
Java dev this reframes modelling from "solve for the answer" to "update a belief with evidence"
(prior → data → posterior), which is also the mental model behind A/B testing, Kalman filters, and
much of probabilistic ML. The chapter teaches the Bayesian machinery on the owner's own use-cases —
**linear regression with Gaussian noise** and an **AR(1) time-series model** — building each up with
real numbers and a modern sampler, and shows how the posterior *is* the uncertainty the frequentist
chapters could only approximate.

Owner's source material (study, then reimplement self-contained with a current library):
`b1ck0/Bayesian-Inference` (cloned locally for the writer) — notebooks "01 Linear Function with
Gaussian Noise", "02 Multi-variable Linear Function with Gaussian Noise", "03 Autoregressive model
AR(1)", originally written with **PyStan**. Keep the *use-cases and structure*; the book uses a
current, pip-installable sampler (see grounding) so a reader can run it today.

## Learning objectives
After this chapter the reader can:
- LO1 — State Bayes' theorem for parameters — posterior ∝ likelihood × prior — and explain prior,
  likelihood, posterior, and the role of the evidence, in plain language before notation.
- LO2 — Build a **Bayesian linear regression** with Gaussian noise: put priors on slope/intercept/
  noise, sample the posterior, and read parameter posteriors + **credible intervals** (and contrast a
  95% credible interval with a 95% confidence interval — they answer different questions).
- LO3 — Read sampler diagnostics honestly: what MCMC is at an intuitive level, trace plots,
  R-hat / effective sample size, and what a bad fit looks like.
- LO4 — Extend to an **AR(1)** time-series model — `x_t = c + φ x_{t-1} + ε_t` — estimate φ with
  uncertainty, and connect it back to the forecasting chapter (DS-9).
- LO5 — Produce a **posterior predictive** — not one prediction but a band — and say when the Bayesian
  approach earns its extra cost vs. the frequentist fit.

## Scope
In scope: conjugate intuition then MCMC sampling with a modern library; the two owner use-cases
(Gaussian-noise linear regression, AR(1)); credible intervals; posterior predictive; diagnostics;
a clear frequentist-vs-Bayesian contrast on the *same* small dataset. CPU-runnable in minutes.
Out of scope (name + link): variational inference, hierarchical/multilevel models beyond a one-line
mention, full Bayesian deep learning. Multi-variable regression from notebook 02 is optional depth if
it stays runnable and short.

## Outline (section-by-section)
1. **Cold open** — a grounded origin note (Bayes 1763 / Laplace; the modern MCMC revival with
   Metropolis-Hastings and Gibbs) and the problem: the DS-5 regression gave *one* slope — but how sure
   are we? Show the single-line fit, then ask the question it can't answer.
2. **What & why** — prior/likelihood/posterior with the "update a belief" framing; Bayesian vs.
   frequentist as two honest answers to different questions; the "you are here" map.
3. **Bayesian linear regression** — priors → likelihood (Gaussian noise) → sample → parameter
   posteriors + credible intervals, on a small real dataset, with the sampler diagnostics read out.
4. **AR(1) in the Bayesian frame** — the model, estimate φ with uncertainty, posterior predictive
   band; tie to DS-9.
5. **Credible vs confidence** — the same data, both intervals, what each actually claims.
6. **Pitfalls** — priors that dominate small data, non-convergence (bad R-hat), reading MCMC noise as
   signal, posterior predictive misused.
7. **Recap & next** — when Bayesian is worth it; pointers (PyMC docs, Statistical Rethinking).

## Assets to produce
- Prose: `01-data-science/03-worked-examples/14-bayesian-inference.md`
- Code: `01-data-science/03-worked-examples/code/bayesian_inference.py` (self-contained, seed set,
  deps pinned; small synthetic or documented dataset).
- Artefacts: a posterior/trace plot and a posterior-predictive band plot, reproduced by the code, in
  `01-data-science/03-worked-examples/artefacts/`.
- **Coherence:** add to `docs/curriculum.md`, the Data Science `README.md` worked-examples list, and a
  forward/back cross-link with DS-5 and DS-9.

## Claims to ground (Haiku research brief — do BEFORE writing)
- [ ] Package versions to pin and the recommended sampler: **PyMC** (current v5.x on PyPI) vs
      `cmdstanpy`/`pystan` — confirm the smallest, most reliable pip install for a CPU laptop today,
      with `arviz` for diagnostics; pin `numpy`, `matplotlib`. Note that the owner's originals used
      PyStan.
- [ ] Verify Bayes' theorem for parameters, the definition of a credible interval vs a confidence
      interval, and R-hat / effective sample size — from an authoritative source (Gelman *BDA3* or the
      PyMC/ArviZ docs) — cite + date.
- [ ] Verify the AR(1) model definition and stationarity condition (|φ| < 1) — cite.
- [ ] Confirm current PyMC API (`pm.Model()`, `pm.sample`, `pm.Normal`, posterior predictive via
      `pm.sample_posterior_predictive`) so the snippet runs as written.

## Acceptance criteria (each maps to evidence)
- [ ] AC1 (LO1–LO5) — Bayes framing, Bayesian linear regression + AR(1), credible-vs-confidence, and a
      posterior predictive all delivered → evidence: runnable script + trace/predictive artefacts.
- [ ] AC2 — snippets run (`check_snippets.py` + a real run log incl. sampler convergence).
- [ ] AC3 — every version/formula/claim grounded → NOTE ids.
- [ ] AC4 — audience-fit: "update a belief" framing, credible-vs-confidence made explicit, artefacts shown.
- [ ] AC5 — renders on GitHub (`check_markdown_render.py` pass; watch `\text{}`/subscripts in the AR
      and Bayes formulas).
- [ ] AC6 — repository coherence: curriculum, DS README, and DS-5/DS-9 cross-links updated.

## Gates
Entry: this spec approved; research NOTEs landed. Exit: all ACs satisfied; snippets run; links
resolve; fresh-Sonnet review sign-off; architect merge. (See `docs/definition-of-done.md`.)

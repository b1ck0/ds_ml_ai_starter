# SPEC-DS-21: The Central Limit Theorem, bootstrapping, and Monte Carlo simulation

**Status:** done (written + self-grounded with inline citations, code executed, gated, reviewed by architect 2026-09-13)
**Subject:** Data Science
**Section:** Worked Examples
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8
**Prerequisites:** SPEC-DS-1 (hypothesis testing & EDA), SPEC-DS-14 (theory — bootstrap in bagging), SPEC-DS-8 (class imbalance — BalancedBagging resamples)

## Intent
Two of the most useful "simulate instead of solve" ideas in applied statistics, taught with runnable
code and real numbers. The Central Limit Theorem explains *why* the average of a sample is so
well-behaved (approximately Normal, with a shrinking, predictable spread) even when the underlying
population is wildly non-normal — and bootstrapping is the practical trick that lets you estimate that
spread when all you have is one sample. Monte Carlo simulation is the flip side: when a probability is
hard to derive by hand, simulate it a few million times. The owner's use-case: a coin puzzle whose
exact answer is surprising, and a bootstrap-vs-population-mean demonstration.

## Learning objectives
After this chapter the reader can:
- LO1 — state the CLT operationally: the distribution of the sample mean is ≈ Normal(μ, σ²/n), with
  standard error σ/√n, regardless of the population's own shape; and *see* it on a skewed population.
- LO2 — explain what bootstrapping does (resample the sample, with replacement) and, crucially, what
  it does NOT do — it centers on the *sample* mean, not the population mean, so it inherits any bias
  in the one sample you drew.
- LO3 — write a Monte Carlo simulation, judge its convergence, and know when to trust it.
- LO4 — solve "P(more heads in 2026 tosses than in 2025 tosses)" both by simulation (≈0.5) and by the
  exact symmetry argument (exactly 1/2).

## Scope
In scope: CLT (intuition + the σ/√n law), sampling distribution of the mean on a non-normal
population, bootstrapping one sample (mini-resamples), the honest caveat that bootstrap reproduces a
biased sample's bias, Monte Carlo method + the coin problem with its closed form.
Out of scope: formal proofs of the CLT (Lindeberg/Lyapunov conditions — cited, not derived); bootstrap
confidence-interval methods beyond the percentile idea (BCa etc.); MCMC (that's SPEC-DS-19, Bayesian).

## Outline (section-by-section)
1. Cold open — averaging skewed latencies; why the average is predictable.
2. What & why — CLT, bootstrap, Monte Carlo defined; Java framing.
3. The population — 1000 right-skewed observations, clearly non-normal.
4. CLT by fresh sampling — many samples of n=10/30/100; means go Normal; SE=σ/√n table + plot.
5. Bootstrapping one sample — 10,000 batches of 30; averaging the batch means ≈ population mean; a
   single batch misses; the biased-sample caveat.
6. Monte Carlo — the 2026-vs-2025 coin problem; simulate → ≈0.5; the exact 1/2 by symmetry; convergence.
7. Pitfalls.
8. Recap & next.

## Assets to produce
- Prose: `01-data-science/03-worked-examples/16-clt-and-monte-carlo.md`
- Code: `01-data-science/03-worked-examples/code/clt_and_monte_carlo.py` (runnable, seeded)
- Artefacts: `artefacts/clt_population.png`, `clt_sampling_distributions.png`, `clt_bootstrap.png`,
  `monte_carlo_coin_convergence.png`

## Claims to ground
- [x] CLT statement + SE=σ/√n, and the "term coined by Pólya 1920 / de Moivre–Laplace origin" history —
      inline citation to Wikipedia "Central limit theorem" (checked 2026-09-13).
- [x] The exact answer to the n+1-vs-n fair-coins problem is 1/2 — proved inline by the head/tail
      symmetry argument (self-contained, verified by the 2,000,000-trial simulation → 0.50026).
- [x] Package versions: numpy==2.5.2, matplotlib==3.11.1 — the repo-wide DS pins (NOTE-2), executed on
      Python 3.13.7.

## Acceptance criteria
- [x] AC1 — CLT shown on a non-normal population; empirical SE matches σ/√n → evidence: the results table
      (4.26≈4.15, 2.35≈2.40, 1.34≈1.31) + `clt_sampling_distributions.png`.
- [x] AC2 — every snippet runs → evidence: `clt_and_monte_carlo.py` executed; outputs quoted verbatim.
- [x] AC3 — averaging the batch means ≈ population mean, a single batch misses, biased sample
      reproduces bias → evidence: bootstrap output block + `clt_bootstrap.png`.
- [x] AC4 — Monte Carlo → 0.50026, exact 1/2 proved → evidence: MC output + `monte_carlo_coin_convergence.png`.
- [x] AC5 — renders on GitHub (fenced `math`, simple inline) + audience-fit Java framing.

## Gates
Entry: owner requested the chapter and approved the CLT/bootstrap demo design (2026-09-13).
Exit: all ACs satisfied; code runs; render lint passes; wired into curriculum + DS README; architect merge.

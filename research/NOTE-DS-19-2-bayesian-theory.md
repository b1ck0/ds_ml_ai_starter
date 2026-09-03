# NOTE-DS-19-2: Bayes' theorem, credible intervals, R-hat, and effective sample size definitions

**Answer:**
(1) **Bayes' theorem for parameters:** Posterior ∝ likelihood × prior, i.e., P(θ|y) ∝ P(y|θ)P(θ), where P(θ|y) is posterior, P(y|θ) is likelihood, P(θ) is prior.
(2) **Credible interval (Bayesian):** A region of posterior probability mass where a parameter lies with stated probability; e.g., "given observed data, 95% probability θ ∈ [a, b]."
(3) **Confidence interval (Frequentist, for contrast):** Long-run interpretation: "if experiment repeated many times, 95% of computed intervals contain true θ."
(4) **R-hat:** Rank-normalized diagnostic comparing between-chain to within-chain variance in MCMC; R-hat < 1.05 indicates convergence.
(5) **Effective Sample Size (ESS):** Number of independent samples equivalent to the correlated MCMC draws in terms of standard error.

**Evidence:**

**1. Bayes' theorem (posterior ∝ likelihood × prior):**
- Gelman et al., *Bayesian Data Analysis, 3rd Edition*: https://www.routledge.com/Bayesian-Data-Analysis/Gelman-Carlin-Stern-Dunson-Vehtari-Rubin/p/book/9781439840955 (covers prior, likelihood, posterior, and evidence)
- Wikipedia on Bayes' theorem: https://en.wikipedia.org/wiki/Bayes'_theorem — states P(A|B) = (P(B|A) × P(A)) / P(B), with posterior ∝ likelihood × prior when proportionality is acceptable.
- Posterior probability definition: https://en.wikipedia.org/wiki/Posterior_probability — "conditional probability distribution representing what parameters are likely after observing the data object."

**2. Credible interval (Bayesian) vs. Confidence interval (Frequentist):**
- Towards Data Science: https://towardsdatascience.com/statistics-101-credible-vs-confidence-interval-af7b7e8fdd79 — Bayesian definition: "an interval, in the domain of the posterior distribution, within which an unobserved parameter value falls with a particular probability."
- Bayesian Reasoning and Methods textbook: https://bookdown.org/kevin_davisross/bayesian-reasoning-and-methods/comparing-bayesian-and-frequentist-interval-estimates.html — contrasts both.
- Statsig: https://www.statsig.com/perspectives/credible-vs-confidence-intervals — Bayesian: "Given our observed data, there is a 95% probability that the true value of θ lies within the credible region"; Frequentist: "If this experiment is repeated many times, in 95% of these cases the computed confidence interval will contain the true θ."
- Towards Data Science: https://towardsdatascience.com/bayesian-credible-intervals-simply-explained-24989c9259a3 — explicit probability statement on parameter given data (Bayesian) vs. on interval given fixed parameter (Frequentist).

**3. R-hat (Rhat) convergence diagnostic:**
- Stan documentation: https://mc-stan.org/learn-stan/diagnostics-warnings.html — "tests for lack of convergence by comparing the variance between multiple chains to the variance within each chain. If convergence has been achieved, the between-chain and within-chain variances should be identical. If chains have not mixed well, Rhat is larger than 1. It's recommended to run at least four chains by default and only use the sample if Rhat is less than 1.05."
- ArviZ v0.23.0 documentation: https://python.arviz.org/en/v0.23.0/api/generated/arviz.rhat.html — provides rank-normalized split-R-hat implementation.
- ArviZ v0.21.0 documentation: https://python.arviz.org/en/v0.21.0/api/generated/arviz.rhat.html — same definition.
- Improved R-hat (Vehtari et al., 2021): https://arxiv.org/pdf/1903.08008 — "An improved R-hat for assessing convergence of MCMC," using rank-normalization and folding for robustness to thick-tailed distributions.

**4. Effective Sample Size (ESS):**
- Stan documentation (same URL as R-hat): "The effective sample size (ESS) of a quantity of interest captures how many independent draws contain the same amount of information as the dependent sample obtained by the MCMC algorithm. ESS is the number of independent samples equivalent to a set of correlated Markov chain samples, having the same standard error."
- ArviZ documentation: https://python.arviz.org/en/v0.23.0/api/generated/arviz.rhat.html — same concept; ArviZ implements rank-normalized ESS.
- Convergence and efficiency diagnostics for Markov Chains (RStan): https://mc-stan.org/rstan/reference/Rhat.html — covers both Rhat and ESS.

**Caveats / limits:**
- Bayes' theorem in practice must include the "evidence" P(y) to be exact; the proportionality (∝) suffices for posterior computation when evidence is constant.
- R-hat < 1.05 is the recommended threshold; values 1.01–1.05 may indicate marginal convergence; > 1.10 suggests non-convergence.
- ESS can vary across parameters; report the minimum ESS for all parameters together to ensure all chains have mixed sufficiently.
- Credible and confidence intervals will often have similar numerical values on the same data, but their interpretations differ fundamentally.
- Gelman et al. BDA3 is the gold standard; PyMC and ArviZ docs cite it and provide practical implementations of these concepts.

**Recommendation:**
Cite Gelman et al. BDA3 (standard reference); use ArviZ for practical R-hat and ESS computation (both built into PyMC's sample output). In prose, emphasize the **interpretive difference**: a 95% credible interval says "given data, parameter is here with 95% probability"; a 95% confidence interval says "this procedure, repeated, produces intervals containing the true value 95% of the time." Use ArviZ's summarize output to show R-hat < 1.05 and ESS values in the worked example; explain that ESS < 400 per 1000 draws is a sign of poor mixing (strong autocorrelation).

**Date checked:** 2026-09-03

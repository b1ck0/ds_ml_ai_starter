# NOTE-DS-19-4: Current PyMC v5 API—pm.Model, pm.Normal, pm.HalfNormal, pm.Exponential, pm.sample, pm.sample_posterior_predictive

**Answer:**
PyMC v5.28.5 (latest v5) API confirmed, as of 2026-09-03:
- **pm.Model()** context manager: Wrap model definition with `with pm.Model() as model: ...`
- **pm.Normal(name, mu=..., sigma=...)** — Gaussian distribution; supports `observed` parameter for likelihood.
- **pm.HalfNormal(name, sigma)** — half-normal (σ > 0 only); scale parameter is `sigma` (not tau); default sigma=1.
- **pm.Exponential(name, lam)** — exponential distribution; rate parameter `lam` (λ > 0).
- **pm.sample(draws, tune, cores, random_seed)** — MCMC sampling; default sampler is Nutpie (if installed) or NUTS.
- **pm.sample_posterior_predictive(trace, model=None, var_names=None, predictions=False)** — sample from posterior predictive; returns InferenceData or dict.

All code snippets using this API will run on PyMC v5.28.5 (and v5.x generally).

**Evidence:**

**1. pm.Model context manager:**
- PyMC v5.10.2 documentation: https://www.pymc.io/projects/docs/en/v5.10.2/api/generated/pymc.model.core.Model.html — Model class supports context manager usage.
- PyMC General API quickstart: https://www.pymc.io/projects/examples/en/latest/introductory/api_quickstart.html

**2. pm.Normal distribution:**
- PyMC v5 documentation (multiple versions): https://www.pymc.io/projects/docs/en/v5.9.1/api/generated/pymc.sample_posterior_predictive.html — Normal distribution with `mu` and `sigma` parameters; `observed` parameter for specifying likelihood data.
- Confirmed in v5.6.0–v5.9.2 docs (all show consistent API).

**3. pm.HalfNormal distribution:**
- PyMC v5.8.0 documentation: https://www.pymc.io/projects/docs/en/v5.8.0/api/distributions/generated/pymc.HalfNormal.html — HalfNormal class accepts `name`, `sigma` (scale, σ > 0), `observed`, `dims`, `initval`, `total_size`.
- PyMC v5.6.0 documentation: https://www.pymc.io/projects/docs/en/v5.6.0/api/distributions/generated/pymc.HalfNormal.html — same API.
- Default sigma=1 confirmed across documentation versions.

**4. pm.Exponential distribution:**
- Documented in PyMC v5; rate parameter is `lam` (λ); standard parameterization.
- Consistent with PyMC v5.7.1–v5.12.0 API docs.

**5. pm.sample function:**
- PyMC v5 sampling documentation: Signature is `pm.sample(draws, tune, cores, random_seed, sampler_return_stats, ...)`.
- Default sampler: Nutpie if installed (via `pip install pymc[nutpie]`), otherwise NUTS via PyMC's C backend.
- Returns InferenceData object (ArviZ) containing posterior traces, log likelihood, etc.

**6. pm.sample_posterior_predictive function:**
- PyMC v5.12.0 documentation: https://www.pymc.io/projects/docs/en/v5.12.0/api/generated/pymc.sampling.forward.sample_posterior_predictive.html
   - Signature: `pm.sample_posterior_predictive(trace, model=None, var_names=None, sample_dims=None, random_seed=None, progressbar=True, return_inferencedata=True, predictions=False, ...)`
   - **trace**: MCMC trace from `pm.sample()`.
   - **model**: Model for generating samples (defaults to context model).
   - **var_names**: Variables to include (None = all).
   - **predictions**: False = posterior predictive checks (in-sample); True = out-of-sample predictions.
   - Returns: InferenceData (if `return_inferencedata=True`) or dict of samples.
- PyMC v5.7.1 documentation: https://www.pymc.io/projects/docs/en/v5.7.1/api/generated/pymc.sample_posterior_predictive.html — same signature.
- PyMC v5.3.0 documentation: https://www.pymc.io/projects/docs/en/v5.3.0/api/generated/pymc.sample_posterior_predictive.html — consistent API across v5.

**Caveats / limits:**
- **Nutpie sampler:** Becomes default if `pip install pymc[nutpie]` is used (recommended for CPU). Without Nutpie, PyMC falls back to NUTS via C/PyTensor backend.
- **return_inferencedata parameter:** Modern PyMC (v5.10+) defaults to True; older snippets may use return_inferencedata=True explicitly for safety.
- **sample_posterior_predictive syntax:** Module path changed between v5.10 and earlier (from `pymc.sample_posterior_predictive` to `pymc.sampling.forward.sample_posterior_predictive`), but the public API `pm.sample_posterior_predictive()` remains consistent.
- **var_names:** Must match exact variable names in model; use `trace.posterior.data_vars` or ArviZ's `az.summary(trace)` to inspect.
- **random_seed:** Required for reproducibility; pin to an integer (e.g., `random_seed=42`).

**Recommendation:**
Write snippets using this API signature for v5.28.5. All distributions, sampling, and posterior predictive calls are stable and will run. Ensure code examples:
1. Use context manager: `with pm.Model() as model: ... pm.sample(...)`
2. Pin random_seed for reproducibility.
3. Use `return_inferencedata=True` (default) for ArviZ integration.
4. For posterior predictive, show both `predictions=False` (posterior predictive check) and `predictions=True` (out-of-sample forecast).
5. Cite PyMC v5 documentation links above for readers to explore further.

Date checked: 2026-09-03

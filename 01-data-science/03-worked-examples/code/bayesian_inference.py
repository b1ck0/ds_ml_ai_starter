"""Bayesian inference with PyMC: Gaussian-noise linear regression and an AR(1) model --
companion code for Data Science/Worked Examples/bayesian-inference.md (SPEC-DS-19).

What it does:
  1. LINEAR REGRESSION WITH GAUSSIAN NOISE (owner's notebook 1, reimplemented in PyMC v5
     instead of the original PyStan): simulate y = alpha + beta*x + noise, fit it two ways on
     the SAME dataset -- frequentist OLS (scipy.stats.linregress) and Bayesian MCMC (PyMC) --
     and directly compare a 95% confidence interval to a 95% credible interval for beta.
  2. AR(1) TIME SERIES (owner's notebook 3): simulate a stationary AR(1) series
     x_t = c + phi*x_{t-1} + eps_t, fit it the same two ways, check the posterior respects the
     |phi| < 1 stationarity condition, and produce a posterior-predictive band.
  3. Prints MCMC convergence diagnostics (R-hat, effective sample size) for both models via
     arviz.summary, and saves four artefacts: a posterior/trace plot for each model, a
     posterior-predictive band plot, and a credible-vs-confidence interval comparison plot.
  4. Two PITFALL demos, run fast and on purpose so a bad fit is something you've actually seen:
     (a) a tight prior overwhelming a tiny (n=5) sample, versus a weak prior on the same data;
     (b) a deliberately under-tuned sampler (5 tuning steps) producing a visibly bad R-hat.

Grounding:
  - Bayes' theorem, credible vs confidence interval, R-hat < 1.05, ESS definitions:
    research/NOTE-DS-19-2-bayesian-theory.md (Gelman et al., BDA3; Stan/ArviZ docs; checked
    2026-09-03).
  - AR(1) model x_t = c + phi*x_{t-1} + eps_t and stationarity |phi| < 1:
    research/NOTE-DS-19-3-ar1-model.md (checked 2026-09-03).
  - PyMC v5 API (pm.Model, pm.Normal, pm.HalfNormal, pm.sample, pm.sample_posterior_predictive):
    research/NOTE-DS-19-4-pymc-api.md (checked 2026-09-03).
  - Package versions: research/NOTE-DS-19-1-pymc-versions.md pinned pymc==5.28.5 and
    arviz==1.3.0. Installing that EXACT pair on this Windows box fails: pymc==5.28.5's own
    published dependency metadata requires `arviz<1.0,>=0.13.0` (verified live via
    `pip install pymc==5.28.5 arviz==1.3.0`, checked 2026-09-03 -- pip reports
    "pymc 5.28.5 depends on arviz<1.0 and >=0.13.0"; arviz 1.3.0 itself does exist on PyPI, it
    is simply not installable alongside pymc 5.28.5). Resolved by installing the newest arviz
    release pymc 5.28.5 actually accepts: arviz==0.23.4. See the chapter's Environment note.

Environment (this project's .venv-bayes, Python 3.13.7, verified live 2026-09-03):
    pymc==5.28.5
    arviz==0.23.4      (see grounding note above -- NOT arviz==1.3.0)
    numpy==2.4.6
    matplotlib==3.11.1
    scipy==1.18.1       (already pinned elsewhere in this repo, research/NOTE-2-package-versions.md)

Run:
    python bayesian_inference.py
"""
from __future__ import annotations

from pathlib import Path

import arviz as az
import matplotlib

matplotlib.use("Agg")  # headless: this script only saves figures, never shows one
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
from scipy import stats

RNG_SEED = 42
CODE_DIR = Path(__file__).resolve().parent
ARTEFACTS_DIR = CODE_DIR.parent / "artefacts"

# MCMC sampling settings shared by both models -- 4 chains (the minimum Stan/PyMC docs
# recommend for a reliable R-hat, research/NOTE-DS-19-2-bayesian-theory.md), enough draws to
# get comfortably past ESS ~400, small enough to stay CPU-runnable in minutes without a C
# compiler on this box (see the module docstring's Environment note).
N_DRAWS = 1000
N_TUNE = 1000
N_CHAINS = 4


# ---------------------------------------------------------------------------
# Part 1: Bayesian linear regression with Gaussian noise
# ---------------------------------------------------------------------------


def make_linreg_data(
    n: int = 100, alpha_true: float = 4.0, beta_true: float = 0.5, sigma_true: float = 1.0, seed: int = RNG_SEED
) -> tuple[np.ndarray, np.ndarray]:
    """y = alpha + beta*x + Normal(0, sigma) noise -- same shape as the owner's notebook 1
    ("01. Linear Function with Gaussian Noise"), reimplemented with numpy's modern Generator
    API instead of the legacy np.random.seed/np.random.rand global state.
    """
    rng = np.random.default_rng(seed)
    x = 10.0 * rng.random(n)
    y = rng.normal(alpha_true + beta_true * x, scale=sigma_true)
    return x, y


def fit_frequentist_ols(x: np.ndarray, y: np.ndarray, conf_level: float = 0.95) -> dict:
    """Ordinary least squares via scipy.stats.linregress, plus a (1 - alpha) confidence
    interval for the slope built from its standard error and a t critical value -- the
    textbook OLS CI construction (Gelman BDA3, research/NOTE-DS-19-2-bayesian-theory.md).
    """
    result = stats.linregress(x, y)
    n = len(x)
    dof = n - 2
    t_crit = stats.t.ppf(1 - (1 - conf_level) / 2, df=dof)
    margin = t_crit * result.stderr
    return {
        "intercept": float(result.intercept),
        "slope": float(result.slope),
        "slope_stderr": float(result.stderr),
        "slope_ci_low": float(result.slope - margin),
        "slope_ci_high": float(result.slope + margin),
        "conf_level": conf_level,
    }


def fit_bayesian_linreg(x: np.ndarray, y: np.ndarray, seed: int = RNG_SEED) -> az.InferenceData:
    """Bayesian counterpart of fit_frequentist_ols on the SAME (x, y): weakly informative
    priors on the intercept/slope, a HalfNormal prior on the noise scale (both distributions
    and pm.Normal(..., observed=...) verified against the installed pymc==5.28.5,
    research/NOTE-DS-19-4-pymc-api.md), then MCMC-sample the posterior.
    """
    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0.0, sigma=10.0)
        beta = pm.Normal("beta", mu=0.0, sigma=10.0)
        sigma = pm.HalfNormal("sigma", sigma=5.0)
        pm.Normal("y_obs", mu=alpha + beta * x, sigma=sigma, observed=y)
        idata = pm.sample(
            draws=N_DRAWS, tune=N_TUNE, chains=N_CHAINS, random_seed=seed, progressbar=False
        )
    return idata


# ---------------------------------------------------------------------------
# Part 2: AR(1) time series
# ---------------------------------------------------------------------------


def make_ar1_data(
    n: int = 150, c_true: float = 1.0, phi_true: float = 0.7, sigma_true: float = 1.5, seed: int = RNG_SEED
) -> np.ndarray:
    """x_t = c + phi*x_{t-1} + eps_t, eps_t ~ Normal(0, sigma) -- the AR(1) definition from
    research/NOTE-DS-19-3-ar1-model.md. phi_true=0.7 satisfies the stationarity condition
    |phi| < 1 the same NOTE grounds. Same use-case as the owner's notebook 3
    ("03. Autoregressive model AR(1)"), reimplemented with numpy's Generator API.
    """
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = c_true / (1 - phi_true)  # start at the process's stationary mean
    for t in range(1, n):
        x[t] = rng.normal(c_true + phi_true * x[t - 1], scale=sigma_true)
    return x


def fit_frequentist_ar1(x: np.ndarray, conf_level: float = 0.95) -> dict:
    """Point estimate + confidence interval for phi via OLS: regress x_t on x_{t-1}. This is
    the frequentist counterpart to fit_bayesian_ar1 below, on the same data.
    """
    x_prev, x_curr = x[:-1], x[1:]
    result = stats.linregress(x_prev, x_curr)
    n = len(x_curr)
    dof = n - 2
    t_crit = stats.t.ppf(1 - (1 - conf_level) / 2, df=dof)
    margin = t_crit * result.stderr
    return {
        "c": float(result.intercept),
        "phi": float(result.slope),
        "phi_stderr": float(result.stderr),
        "phi_ci_low": float(result.slope - margin),
        "phi_ci_high": float(result.slope + margin),
        "conf_level": conf_level,
    }


def fit_bayesian_ar1(x: np.ndarray, seed: int = RNG_SEED) -> az.InferenceData:
    """Bayesian AR(1): a Uniform(-1, 1) prior on phi directly encodes the stationarity belief
    |phi| < 1 (research/NOTE-DS-19-3-ar1-model.md's recommendation), a weakly informative
    Normal prior on the intercept c, HalfNormal on the noise scale.
    """
    x_prev, x_curr = x[:-1], x[1:]
    with pm.Model() as model:
        c = pm.Normal("c", mu=0.0, sigma=10.0)
        phi = pm.Uniform("phi", lower=-1.0, upper=1.0)
        sigma = pm.HalfNormal("sigma", sigma=5.0)
        pm.Normal("x_obs", mu=c + phi * x_prev, sigma=sigma, observed=x_curr)
        idata = pm.sample(
            draws=N_DRAWS, tune=N_TUNE, chains=N_CHAINS, random_seed=seed, progressbar=False
        )
    return idata


def posterior_predictive_ar1(idata: az.InferenceData, x: np.ndarray, seed: int = RNG_SEED) -> az.InferenceData:
    """Posterior predictive check: for every posterior draw of (c, phi, sigma), redraw a
    prediction for each x_t from x_{t-1}. pm.sample_posterior_predictive signature verified
    against the installed pymc==5.28.5, research/NOTE-DS-19-4-pymc-api.md.
    """
    x_prev, x_curr = x[:-1], x[1:]
    with pm.Model() as model:
        c = pm.Normal("c", mu=0.0, sigma=10.0)
        phi = pm.Uniform("phi", lower=-1.0, upper=1.0)
        sigma = pm.HalfNormal("sigma", sigma=5.0)
        pm.Normal("x_obs", mu=c + phi * x_prev, sigma=sigma, observed=x_curr)
        post_pred = pm.sample_posterior_predictive(idata, random_seed=seed, progressbar=False)
    return post_pred


# ---------------------------------------------------------------------------
# Artefacts
# ---------------------------------------------------------------------------


def plot_posterior_trace(idata: az.InferenceData, var_names: list[str], title: str, filename: str) -> Path:
    """One combined figure: for each parameter, the marginal posterior density (left) next to
    its MCMC trace across all chains (right) -- az.plot_trace, verified against the installed
    arviz==0.23.4.
    """
    axes = az.plot_trace(idata, var_names=var_names, compact=True, figsize=(10, 2.4 * len(var_names)))
    fig = axes.ravel()[0].figure
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTEFACTS_DIR / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_ar1_posterior_predictive(x: np.ndarray, post_pred: az.InferenceData) -> Path:
    """Posterior predictive BAND (not one line): the observed AR(1) series against the
    posterior predictive median and 94% HDI at every time step -- az.hdi verified against the
    installed arviz==0.23.4.
    """
    x_curr = x[1:]
    t = np.arange(1, len(x))

    pred = post_pred.posterior_predictive["x_obs"]  # dims: (chain, draw, x_obs_dim_0)
    pred_median = pred.median(dim=("chain", "draw")).values
    hdi = az.hdi(post_pred, group="posterior_predictive", var_names=["x_obs"], hdi_prob=0.94)
    hdi_low = hdi["x_obs"].sel(hdi="lower").values
    hdi_high = hdi["x_obs"].sel(hdi="higher").values

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.fill_between(t, hdi_low, hdi_high, color="tab:blue", alpha=0.25, label="94% posterior predictive HDI")
    ax.plot(t, pred_median, color="tab:blue", linewidth=1.5, label="posterior predictive median")
    ax.plot(t, x_curr, color="black", linewidth=1.2, marker="o", markersize=2.5, label="observed $x_t$")
    ax.set_xlabel("time step $t$")
    ax.set_ylabel("$x_t$")
    ax.set_title("AR(1) posterior predictive: not one forecast, a band")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTEFACTS_DIR / "ar1_posterior_predictive.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_credible_vs_confidence(freq: dict, bayes_ci: dict, true_beta: float) -> Path:
    """Same data, same parameter (beta), two intervals: a 95% confidence interval (frequentist)
    and a 95% credible interval (Bayesian) drawn on the same axis so the reader can compare
    them directly, per SPEC-DS-19's credible-vs-confidence requirement.
    """
    fig, ax = plt.subplots(figsize=(7, 3))
    y_positions = [1, 0]
    labels = ["95% confidence interval\n(frequentist OLS)", "95% credible interval\n(Bayesian posterior)"]
    centers = [freq["slope"], bayes_ci["median"]]
    lows = [freq["slope"] - freq["slope_ci_low"], bayes_ci["median"] - bayes_ci["ci_low"]]
    highs = [freq["slope_ci_high"] - freq["slope"], bayes_ci["ci_high"] - bayes_ci["median"]]

    ax.errorbar(
        centers, y_positions, xerr=[lows, highs], fmt="o", capsize=6, color="tab:blue", ecolor="tab:blue", markersize=8
    )
    ax.axvline(true_beta, color="tab:red", linestyle="--", linewidth=1.2, label=f"true beta = {true_beta}")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("beta (slope)")
    ax.set_title("Same data, same parameter, two different intervals")
    ax.set_ylim(-0.7, 1.7)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTEFACTS_DIR / "credible_vs_confidence_interval.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Pitfall demos (Section 6 of the chapter)
# ---------------------------------------------------------------------------


def demo_prior_dominates_small_n(x_full: np.ndarray, y_full: np.ndarray, n_small: int = 5, seed: int = RNG_SEED) -> dict:
    """Fit the SAME n=5 subsample of the linear-regression data twice: once with a weakly
    informative prior on beta (Normal(0, 10), what Part 1 used), once with a tight prior that
    strongly believes beta is near 0 (Normal(0, 0.05)). With only 5 points the likelihood is
    weak, so whichever prior you picked does most of the talking -- exactly the "priors that
    dominate small data" pitfall this chapter's spec calls out.
    """
    x_small, y_small = x_full[:n_small], y_full[:n_small]
    ols = fit_frequentist_ols(x_small, y_small)

    with pm.Model():
        alpha = pm.Normal("alpha", mu=0.0, sigma=10.0)
        beta = pm.Normal("beta", mu=0.0, sigma=10.0)  # weak: matches Part 1's prior
        sigma = pm.HalfNormal("sigma", sigma=5.0)
        pm.Normal("y_obs", mu=alpha + beta * x_small, sigma=sigma, observed=y_small)
        idata_weak = pm.sample(draws=N_DRAWS, tune=N_TUNE, chains=N_CHAINS, random_seed=seed, progressbar=False)

    with pm.Model():
        alpha = pm.Normal("alpha", mu=0.0, sigma=10.0)
        beta = pm.Normal("beta", mu=0.0, sigma=0.05)  # tight: "I'm confident beta is near 0"
        sigma = pm.HalfNormal("sigma", sigma=5.0)
        pm.Normal("y_obs", mu=alpha + beta * x_small, sigma=sigma, observed=y_small)
        idata_strong = pm.sample(draws=N_DRAWS, tune=N_TUNE, chains=N_CHAINS, random_seed=seed, progressbar=False)

    return {
        "n": n_small,
        "ols_slope": ols["slope"],
        "ols_slope_stderr": ols["slope_stderr"],
        "weak_prior_beta_mean": float(idata_weak.posterior["beta"].mean()),
        "strong_prior_beta_mean": float(idata_strong.posterior["beta"].mean()),
    }


def demo_bad_convergence(x: np.ndarray, y: np.ndarray, seed: int = RNG_SEED) -> dict:
    """The SAME linear-regression model as Part 1, sampled deliberately badly: only 5 tuning
    steps instead of 1000, so the sampler never adapts its step size. This is "what a bad fit
    looks like" (LO3) -- produced on purpose, not hidden, so the reader has actually seen the
    R-hat/ESS numbers that should make them distrust a result.
    """
    with pm.Model():
        alpha = pm.Normal("alpha", mu=0.0, sigma=10.0)
        beta = pm.Normal("beta", mu=0.0, sigma=10.0)
        sigma = pm.HalfNormal("sigma", sigma=5.0)
        pm.Normal("y_obs", mu=alpha + beta * x, sigma=sigma, observed=y)
        idata_bad = pm.sample(draws=30, tune=5, chains=N_CHAINS, random_seed=seed, progressbar=False)

    summary_bad = az.summary(idata_bad, var_names=["alpha", "beta", "sigma"])
    return {
        "max_rhat": float(summary_bad["r_hat"].max()),
        "min_ess_bulk": float(summary_bad["ess_bulk"].min()),
        "summary": summary_bad,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=== Part 1: Bayesian linear regression with Gaussian noise ===")
    x, y = make_linreg_data()
    freq = fit_frequentist_ols(x, y)
    print(
        f"frequentist OLS: intercept={freq['intercept']:.3f}, slope={freq['slope']:.3f}, "
        f"95% CI for slope = [{freq['slope_ci_low']:.3f}, {freq['slope_ci_high']:.3f}]"
    )

    idata_linreg = fit_bayesian_linreg(x, y)
    summary_linreg = az.summary(idata_linreg, var_names=["alpha", "beta", "sigma"])
    print("\nBayesian posterior summary (linear regression):")
    print(summary_linreg)

    beta_samples = idata_linreg.posterior["beta"].values.flatten()
    beta_median = float(np.median(beta_samples))
    beta_ci_low, beta_ci_high = (float(v) for v in np.percentile(beta_samples, [2.5, 97.5]))
    print(
        f"\nBayesian: posterior median beta={beta_median:.3f}, "
        f"95% equal-tailed credible interval = [{beta_ci_low:.3f}, {beta_ci_high:.3f}]"
    )

    max_rhat_linreg = float(summary_linreg["r_hat"].max())
    min_ess_linreg = float(summary_linreg["ess_bulk"].min())
    print(f"convergence: max R-hat = {max_rhat_linreg:.4f} (want < 1.05), min ESS_bulk = {min_ess_linreg:.0f}")
    assert max_rhat_linreg < 1.05, f"linear regression model did not converge: max R-hat = {max_rhat_linreg}"

    trace_path = plot_posterior_trace(
        idata_linreg, ["alpha", "beta", "sigma"], "Linear regression: posterior + trace (4 chains)", "posterior_trace_linreg.png"
    )
    print(f"saved: {trace_path.relative_to(CODE_DIR.parent.parent.parent)}")

    ci_path = plot_credible_vs_confidence(
        freq, {"median": beta_median, "ci_low": beta_ci_low, "ci_high": beta_ci_high}, true_beta=0.5
    )
    print(f"saved: {ci_path.relative_to(CODE_DIR.parent.parent.parent)}")

    print("\n=== Part 2: AR(1) time series ===")
    ar1_series = make_ar1_data()
    freq_ar1 = fit_frequentist_ar1(ar1_series)
    print(
        f"frequentist OLS: c={freq_ar1['c']:.3f}, phi={freq_ar1['phi']:.3f}, "
        f"95% CI for phi = [{freq_ar1['phi_ci_low']:.3f}, {freq_ar1['phi_ci_high']:.3f}]"
    )

    idata_ar1 = fit_bayesian_ar1(ar1_series)
    summary_ar1 = az.summary(idata_ar1, var_names=["c", "phi", "sigma"])
    print("\nBayesian posterior summary (AR(1)):")
    print(summary_ar1)

    phi_samples = idata_ar1.posterior["phi"].values.flatten()
    phi_median = float(np.median(phi_samples))
    phi_ci_low, phi_ci_high = (float(v) for v in np.percentile(phi_samples, [2.5, 97.5]))
    print(
        f"\nBayesian: posterior median phi={phi_median:.3f}, "
        f"95% equal-tailed credible interval = [{phi_ci_low:.3f}, {phi_ci_high:.3f}]"
    )
    print(f"stationarity check: |posterior median phi| = {abs(phi_median):.3f} < 1 -> {abs(phi_median) < 1}")

    max_rhat_ar1 = float(summary_ar1["r_hat"].max())
    min_ess_ar1 = float(summary_ar1["ess_bulk"].min())
    print(f"convergence: max R-hat = {max_rhat_ar1:.4f} (want < 1.05), min ESS_bulk = {min_ess_ar1:.0f}")
    assert max_rhat_ar1 < 1.05, f"AR(1) model did not converge: max R-hat = {max_rhat_ar1}"

    trace_path_ar1 = plot_posterior_trace(
        idata_ar1, ["c", "phi", "sigma"], "AR(1): posterior + trace (4 chains)", "posterior_trace_ar1.png"
    )
    print(f"saved: {trace_path_ar1.relative_to(CODE_DIR.parent.parent.parent)}")

    post_pred = posterior_predictive_ar1(idata_ar1, ar1_series)
    pred_path = plot_ar1_posterior_predictive(ar1_series, post_pred)
    print(f"saved: {pred_path.relative_to(CODE_DIR.parent.parent.parent)}")

    print("\n=== Pitfall demo (a): priors dominate small data (n=5) ===")
    prior_demo = demo_prior_dominates_small_n(x, y)
    print(f"n={prior_demo['n']} points")
    print(f"OLS slope on these 5 points: {prior_demo['ols_slope']:.3f} (stderr={prior_demo['ols_slope_stderr']:.3f})")
    print(f"Bayesian, weak prior beta~N(0,10):   posterior mean beta = {prior_demo['weak_prior_beta_mean']:.3f}")
    print(f"Bayesian, tight prior beta~N(0,0.05): posterior mean beta = {prior_demo['strong_prior_beta_mean']:.3f}")

    print("\n=== Pitfall demo (b): deliberately under-tuned sampler (bad R-hat) ===")
    bad_demo = demo_bad_convergence(x, y)
    print(bad_demo["summary"])
    print(f"max R-hat = {bad_demo['max_rhat']:.3f} (>> 1.05 -- do not trust this posterior)")
    print(f"min ESS_bulk = {bad_demo['min_ess_bulk']:.0f} (out of 120 total draws -- badly under-mixed)")


if __name__ == "__main__":
    main()

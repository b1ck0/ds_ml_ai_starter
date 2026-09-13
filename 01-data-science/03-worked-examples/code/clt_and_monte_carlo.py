"""Central Limit Theorem + bootstrapping, and a Monte Carlo coin problem.

Runnable end to end against this project's pinned DS stack
(numpy==2.5.2, matplotlib==3.11.1, Python 3.12+). Everything is seeded, so the
numbers and plots reproduce exactly. Saves four figures into ../artefacts/ and
prints a labelled results block the chapter quotes verbatim.

    .venv/Scripts/python.exe 01-data-science/03-worked-examples/code/clt_and_monte_carlo.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: save figures, never call plt.show()
import matplotlib.pyplot as plt
import numpy as np

ART = Path(__file__).resolve().parent.parent / "artefacts"
ART.mkdir(exist_ok=True)

BLUE, RED, GREY = "#4C72B0", "#C44E52", "#8C8C8C"


# --------------------------------------------------------------------------
# Part 1 - the population (right-skewed, deliberately NOT normal)
# --------------------------------------------------------------------------
def make_population(seed: int = 7, size: int = 1000) -> np.ndarray:
    """1000 simulated API response latencies (ms), right-skewed like real ones:
    most requests are fast, a long tail is slow. Lognormal -> clearly non-normal."""
    rng = np.random.default_rng(seed)
    return rng.lognormal(mean=3.2, sigma=0.5, size=size)


def plot_population(pop: np.ndarray) -> None:
    mu = pop.mean()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(pop, bins=40, color=BLUE, edgecolor="white")
    ax.axvline(mu, color=RED, linestyle="--", linewidth=2,
               label=f"population mean mu = {mu:.2f} ms")
    ax.set_xlabel("Response latency (ms)")
    ax.set_ylabel("Count")
    ax.set_title("The population: 1000 latencies -- right-skewed, not a bell curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ART / "clt_population.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Part 2 - CLT via MANY FRESH samples drawn from the population
# --------------------------------------------------------------------------
def sampling_distributions(pop: np.ndarray, sizes=(10, 30, 100),
                           n_samples: int = 2000, seed: int = 21):
    """Draw n_samples fresh samples of each size from the population (with
    replacement = i.i.d. draws from the empirical distribution) and collect the
    sample means. Returns {n: array_of_means}."""
    rng = np.random.default_rng(seed)
    out = {}
    for n in sizes:
        idx = rng.integers(0, len(pop), size=(n_samples, n))
        out[n] = pop[idx].mean(axis=1)
    return out


def plot_sampling_distributions(pop, dists) -> None:
    mu, sigma = pop.mean(), pop.std(ddof=0)
    fig, axes = plt.subplots(1, len(dists), figsize=(13, 4.2), sharex=True)
    for ax, (n, means) in zip(axes, dists.items()):
        se = sigma / np.sqrt(n)
        ax.hist(means, bins=40, density=True, color=BLUE, edgecolor="white", alpha=0.8)
        grid = np.linspace(means.min(), means.max(), 200)
        normal = np.exp(-0.5 * ((grid - mu) / se) ** 2) / (se * np.sqrt(2 * np.pi))
        ax.plot(grid, normal, color=RED, linewidth=2, label=f"Normal(mu, sigma/sqrt{n})")
        ax.axvline(mu, color=GREY, linestyle="--", linewidth=1)
        ax.set_title(f"sample size n = {n}\nSE = sigma/sqrt(n) = {se:.2f}")
        ax.set_xlabel("sample mean (ms)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("density")
    fig.suptitle("CLT: means of fresh samples go Normal and tighten as n grows "
                 "(same population, non-normal)", y=1.02)
    fig.tight_layout()
    fig.savefig(ART / "clt_sampling_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Part 3 - bootstrapping ONE sample of 100 (mini-resamples of 30)
# --------------------------------------------------------------------------
def bootstrap_one_sample(pop, sample_seed=27, n_sample=100, n_resample=30,
                         n_boot=10_000, boot_seed=202):
    rng_s = np.random.default_rng(sample_seed)
    sample = pop[rng_s.integers(0, len(pop), size=n_sample)]  # one sample of 100

    rng_b = np.random.default_rng(boot_seed)
    idx = rng_b.integers(0, n_sample, size=(n_boot, n_resample))
    boot_means = sample[idx].mean(axis=1)  # each: mean of a resample of 30
    return sample, boot_means


def plot_bootstrap(pop, sample, boot_means) -> None:
    mu, xbar = pop.mean(), sample.mean()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(boot_means, bins=45, density=True, color=BLUE, edgecolor="white", alpha=0.8,
            label="10,000 bootstrap means (resamples of 30)")
    ax.axvline(mu, color=RED, linestyle="--", linewidth=2, label=f"population mean = {mu:.2f}")
    ax.axvline(xbar, color="black", linestyle=":", linewidth=2,
               label=f"the one sample's mean = {xbar:.2f}")
    ax.axvline(boot_means.mean(), color="#55A868", linestyle="-", linewidth=1.5,
               label=f"mean of bootstrap means = {boot_means.mean():.2f}")
    ax.set_xlabel("resample mean (ms)")
    ax.set_ylabel("density")
    ax.set_title("Bootstrap: resampling one sample of 100 recovers ~the population mean")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ART / "clt_bootstrap.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Part 4 - Monte Carlo: P(more heads in 2026 tosses than in 2025 tosses)
# --------------------------------------------------------------------------
def monte_carlo_coins(n_a=2026, n_b=2025, n_trials=2_000_000, seed=2026):
    rng = np.random.default_rng(seed)
    heads_a = rng.binomial(n_a, 0.5, size=n_trials)
    heads_b = rng.binomial(n_b, 0.5, size=n_trials)
    wins = heads_a > heads_b
    # running estimate for the convergence plot
    running = np.cumsum(wins) / np.arange(1, n_trials + 1)
    return wins.mean(), running


def plot_convergence(running) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(np.arange(1, len(running) + 1), running, color=BLUE, linewidth=1,
            label="Monte Carlo estimate")
    ax.axhline(0.5, color=RED, linestyle="--", linewidth=2, label="exact answer = 1/2")
    ax.set_xscale("log")
    ax.set_ylim(0.40, 0.60)
    ax.set_xlabel("number of simulated trials (log scale)")
    ax.set_ylabel("P(more heads in 2026 than 2025)")
    ax.set_title("Monte Carlo converges to the exact 1/2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ART / "monte_carlo_coin_convergence.png", dpi=150)
    plt.close(fig)


def main() -> None:
    pop = make_population()
    mu, sigma = pop.mean(), pop.std(ddof=0)
    plot_population(pop)

    print("=== POPULATION (1000 observations) ===")
    print(f"population mean  mu    = {mu:.4f} ms")
    print(f"population stdev sigma = {sigma:.4f} ms")

    dists = sampling_distributions(pop)
    plot_sampling_distributions(pop, dists)
    print("\n=== CLT: 2000 fresh samples at each size (drawn FROM the population) ===")
    print(f"{'n':>5} | {'mean of sample means':>21} | {'empirical SE':>12} | {'sigma/sqrt(n)':>13}")
    for n, means in dists.items():
        print(f"{n:>5} | {means.mean():>21.4f} | {means.std(ddof=1):>12.4f} | {sigma/np.sqrt(n):>13.4f}")
    # one single sample of 100 for the contrast line
    one = np.random.default_rng(999).choice(pop, size=100, replace=True)
    print(f"\nA SINGLE sample of 100 has mean {one.mean():.4f} ms -- close to mu={mu:.4f}, "
          f"but not equal (off by {one.mean()-mu:+.4f}).")

    sample, boot = bootstrap_one_sample(pop)
    plot_bootstrap(pop, sample, boot)
    print("\n=== BOOTSTRAP: one sample of 100, then 10,000 resamples of 30 ===")
    print(f"the one sample's mean  x_bar        = {sample.mean():.4f} ms")
    print(f"mean of 10,000 bootstrap means      = {boot.mean():.4f} ms")
    print(f"population mean        mu           = {mu:.4f} ms")
    print(f"a single resample of 30 mean        = {boot[0]:.4f} ms  (misses mu by {boot[0]-mu:+.4f})")
    print(f"mean of just the first 100 resamples= {boot[:100].mean():.4f} ms  "
          f"(off by {boot[:100].mean()-mu:+.4f})")
    # honest caveat: a biased sample -> bootstrap faithfully reproduces the bias
    low = np.sort(pop)[:100]  # deliberately unrepresentative (100 fastest)
    rng_c = np.random.default_rng(303)
    low_boot = low[rng_c.integers(0, 100, size=(10_000, 30))].mean(axis=1)
    print(f"\nCAVEAT -- a BIASED sample (the 100 fastest): its mean = {low.mean():.4f} ms; "
          f"bootstrap mean = {low_boot.mean():.4f} ms -> reproduces the bias, never reaches mu={mu:.4f}.")

    p_hat, running = monte_carlo_coins()
    plot_convergence(running)
    print("\n=== MONTE CARLO: P(more heads in 2026 tosses than in 2025 tosses) ===")
    print(f"2,000,000 simulated trials -> estimate = {p_hat:.5f}")
    print("exact answer (n+1 vs n fair coins)      = 0.50000")

    print(f"\nSaved 4 figures to {ART}")


if __name__ == "__main__":
    main()

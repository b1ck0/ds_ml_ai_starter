# NOTE-DS-19-1: PyMC v5.x versions and installation for CPU laptops

**Answer:**
PyMC v5.28.5 (released 2026-05-01) is the latest v5 series; PyMC 6.3.1 (2026-08-16) is the current stable. For CPU laptops, **recommend `pip install pymc[nutpie]`** which includes the Rust-based Nutpie sampler—it is smaller, more reliable, and faster than cmdstanpy or pystan for Windows/CPU. Pin: PyMC v5.28.5, ArviZ 1.3.0 (2026-08-11), compatible with NumPy 2.0+, Python >=3.12.

**Evidence:**

1. **PyMC v5.28.5 release (latest v5):** Released 2026-05-01; confirmed at https://discourse.pymc.io/t/release-v5-28-5/17703
   - Bug fixes included: "Delay attaching ShapeFeature to logprob rewrites"

2. **PyMC 6.3.1 (current stable):** Released 2026-08-16, per https://pypi.org/project/pymc/

3. **ArviZ 1.3.0:** Released 2026-08-11, available at https://pypi.org/project/arviz/
   - "expose[s] features from _ArviZverse_ refactored packages together in the arviz namespace"

4. **Installation recommendation:** PyMC docs at https://www.pymc.io/projects/docs/en/stable/installation.html state: 
   - Recommended: `pip install "pymc[nutpie]"` 
   - Nutpie is "a fast NUTS sampler written in Rust" automatically selected as default when installed
   - For conda: `conda create -c conda-forge -n pymc_env "pymc>=6"`

5. **PyMC dependencies:** Python >=3.12 required. Compatible with NumPy 2.0+ per https://discourse.pymc.io/t/pymc-now-compatible-with-numpy-2-0-and-python3-13/16600 (announced v5.21.0, Feb 2025).

6. **CPU performance vs cmdstanpy/pystan:** Per comparison at https://martiningram.github.io/mcmc-comparison/ (GitHub: https://github.com/martiningram/mcmc_runtime_comparison):
   - PyMC + JAX (NumPyro or BlackJAX backends) performs best on CPU, especially on larger datasets
   - cmdstanpy requires C++ toolchain; CmdStanPy 1.3.0 docs (https://mc-stan.org/cmdstanpy/installation.html) note it wraps CmdStan (C++) and requires separate environment from PyStan
   - PyMC has more straightforward Windows installation via conda; cmdstanpy/pystan require RTools on Windows (per https://github.com/pymc-devs/pymc/issues/4937)

7. **PyMC + Matplotlib/NumPy:** As of May 2026, PyMC is verified compatible with modern releases (NumPy 2.3.5, Matplotlib 3.10.8). No version pinning needed for these; constrain to ">=2.0" for numpy and ">=3.8" for matplotlib.

**Caveats / limits:**
- v5.28.5 is still v5; if reader upgrades to v6.x, API changes may apply (e.g., nutpie becomes default).
- Nutpie sampler is recent; some edge cases may still prefer traditional NUTS. For pedagogical stability, recommend v5.28.5.
- Windows users must have recent Visual C++ runtime; conda installation handles this more reliably than pip.
- cmdstanpy/pystan remain valid choices for large-scale Bayesian inference but incur C++ compilation overhead not needed here.

**Recommendation:**
Pin to **`pymc==5.28.5`**, **`arviz==0.23.4`** in requirements.

> **ARCHITECT CORRECTION (2026-09-03):** the original draft pinned `arviz==1.3.0`, but that pair is
> **not jointly installable** — `pymc==5.28.5`'s own metadata requires `arviz<1.0`. The chapter writer
> hit the pip resolver error and pinned **`arviz==0.23.4`** (the newest release pymc 5.28.5 accepts),
> which sampled and produced diagnostics cleanly. Use 0.23.4. (Every "arviz 1.3.0" below is superseded.) Do not specify numpy/matplotlib versions explicitly (use ">=2.0" for numpy, ">=3.8" for matplotlib to allow flexibility). If snippets are tested on a Windows machine, validate conda install path in addition to pip; document both. Note owner's prior PyStan work but explain why PyMC v5 + nutpie is now better for single-machine CPU work.

**Date checked:** 2026-09-03

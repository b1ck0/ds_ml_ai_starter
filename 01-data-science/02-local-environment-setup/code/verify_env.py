"""Verify the Data Science local environment.

Imports every library this course's Data Science chapters depend on and
prints its installed version, plus the Python interpreter version. A clean
run with no ImportError, and every line filled in, means the environment
this chapter just built is ready for every DS chapter after it.

`jupyter` is a metapackage (see the chapter, Section 5) -- it has no
`jupyter.__version__` attribute of its own, so its version is read via
`importlib.metadata`, the same way you'd read a declared version out of a
POM rather than off a compiled class. `jupyter_core` (one of the components
jupyter installs) IS a regular importable module with `__version__`, so
that is printed too, as a second confirmation that the notebook engine
itself is present.

Run this with the shared DS virtualenv's interpreter, NOT your system
Python -- see the chapter for why an isolated venv matters:

    .venv\\Scripts\\python.exe "Data Science/Local Environment Setup/code/verify_env.py"   # Windows
    .venv/bin/python "Data Science/Local Environment Setup/code/verify_env.py"              # macOS/Linux

Expects (pinned in this chapter, see local-environment-setup.md):
    pandas==3.0.5 numpy==2.5.2 matplotlib==3.11.1 scipy==1.18.1
    seaborn==0.13.2 scikit-learn==1.9.0 jupyter==1.1.1
"""
from __future__ import annotations

import sys
from importlib.metadata import version as pkg_version

import matplotlib
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import sklearn
import jupyter_core


def main() -> None:
    print(f"Python:        {sys.version.split()[0]}")
    print(f"pandas:        {pd.__version__}")
    print(f"numpy:         {np.__version__}")
    print(f"matplotlib:    {matplotlib.__version__}")
    print(f"scipy:         {scipy.__version__}")
    print(f"seaborn:       {sns.__version__}")
    print(f"scikit-learn:  {sklearn.__version__}")
    print(f"jupyter_core:  {jupyter_core.__version__}")
    print(f"jupyter (meta): {pkg_version('jupyter')}")


if __name__ == "__main__":
    main()

# SPEC-DS-0: Local Environment Setup (Data Science)

**Status:** done (written by Sonnet, grounded by Haiku, independently reviewed + merged 2026-09-03)
**Subject:** Data Science
**Section:** Local Environment Setup
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8
**Prerequisites:** none (this is the on-ramp)

## Intent
Get a senior Java dev from zero to a working Python DS environment, mapping every tool to its
Java/JVM analogue so the mental model transfers. This is the environment every other DS chapter
assumes.

## Learning objectives
- LO1 — Install Python 3.12+, create and activate a virtualenv, and understand it as an isolated classpath.
- LO2 — Use pip + requirements.txt the way they use Maven/Gradle; know what pinning buys you.
- LO3 — Set up PyCharm for a data project and run a Jupyter notebook; understand a notebook as a persisted REPL/worksheet.
- LO4 — Verify the stack (pandas, numpy, matplotlib, scikit-learn, scipy, seaborn, jupyter) imports and prints its versions.

## Scope
In: Python, venv, pip, requirements, PyCharm, Jupyter, the core DS libraries and what each is for.
Out: conda/poetry/uv (mention as alternatives, one line each), cloud notebooks (→ DS cloud chapters).

## Outline
1. What & why — the Python toolchain vs the Java one (venv≈isolated classpath, pip≈Maven, requirements≈pom).
2. Install Python + make a venv; activate on Windows/macOS/Linux.
3. Install the stack from requirements; verify with a version-printing script.
4. PyCharm: open the project, point it at the venv interpreter, run a file and a notebook cell.
5. Jupyter: what a notebook is, cell state, when to use it vs a script.
6. Pitfalls — global vs venv installs, the "works on my machine" trap, kernel≠interpreter.

## Assets to produce
- Prose: "Data Science/Local Environment Setup/local-environment-setup.md"
- Code: "Data Science/Local Environment Setup/code/verify_env.py" (imports each lib, prints versions)
- Artefacts: a captured console output block of verify_env.py (fenced as ```text)

## Claims to ground (Haiku, before writing)
- [ ] Reuse research/NOTE-2 for pandas/numpy/matplotlib/scipy/seaborn versions; additionally verify current stable scikit-learn and jupyter versions on PyPI (date-checked).
- [ ] Verify the current recommended PyCharm edition (Community is free) and that it supports Jupyter notebooks.
- [ ] Verify the venv activation commands per-OS from official Python docs.

## Acceptance criteria
- [ ] AC1 — LO1–LO4 delivered → section map.
- [ ] AC2 — verify_env.py runs and prints versions → run log; snippet-check passes.
- [ ] AC3 — every version/tool claim grounded → NOTE ids.
- [ ] AC4 — every setup step has a Java analogue where one clarifies.

## Gates
Entry: approved; grounding notes landed. Exit: DoD in docs/definition-of-done.md — snippets run, links resolve, review + merge.

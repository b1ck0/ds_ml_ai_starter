# NOTE-ML-15-1: PyPI Package Versions for Tabular Q-learning Chess Environment

**Date checked:** 2026-09-03

## Answer
For a minimal tabular Q-learning agent on a CPU-runnable chess-derived environment (laptop, ≤5 min training):
- **chess** (python-chess) **1.11.2** (2025-02-25): Provides legal move generation and terminal detection; overhead is minimal for small endgame/grid tasks.
- **gymnasium** **1.3.0** (2026-04-22): Optional; not required for tabular Q-learning. If used, provides standard RL interface but adds API surface. Lightweight alternative to OpenAI Gym (deprecated).
- **numpy** **2.5.2** (2026-08-09): Core dependency for arrays and numerical operations; no special version concerns.
- **matplotlib** **3.11.1** (2026-07-18): Standard for plotting learning curves (reward per episode, ε decay).

**Recommendation for smallest/most robust setup:** Use `python-chess` 1.11.2 + **hand-rolled grid environment** (avoid gymnasium overhead). Reason: tabular Q-learning on a small state space (dict-based Q-table) does not need gymnasium's environment interface; a simple Python class (state → tuple, action → int, reward → float) is clearer and less coupled. Gymnasium adds ~500 MB dependency weight for no benefit to a 50-state-space task. The grid approach scales training to <1 minute on CPU.

### State-space concern for dict Q-table
A minimal endgame (e.g., King+Rook vs King on 5×5 board region) or capture-grid (~100 reachable states) is feasible in a dict. Full chess has ~10^43–10^47 reachable positions—prohibitively large. The chapter's "tiny environment" must be genuinely small (~50–1000 states) for dict Q-table to be inspectable and training to complete in minutes. This is the **tractability hard constraint** stated in the spec.

## Evidence
- **chess 1.11.2**: https://pypi.org/project/chess/ (verified 2026-09-03; released 2025-02-25)
- **gymnasium 1.3.0**: https://pypi.org/project/gymnasium/ (verified 2026-09-03; released 2026-04-22)
- **numpy 2.5.2**: https://pypi.org/project/numpy/ (verified 2026-09-03; released 2026-08-09)
- **matplotlib 3.11.1**: https://pypi.org/project/matplotlib/ (verified 2026-09-03; released 2026-07-18)

## Caveats
- **python-chess** is the correct package name on PyPI (formerly `python-chess`, now also called `chess`).
- **gymnasium** is the active successor to OpenAI Gym (Gym was archived); both provide wrappers for chess via external envs (e.g., `chess-gym`), but requiring such a wrapper for a toy Q-learning agent is over-engineering.
- **numpy 2.5.2** requires Python ≥3.12 (per PyPI); ensure the chapter environment declares this minimum.
- **matplotlib 3.11.1** requires Python ≥3.11; compatible with numpy 2.5.2.

## Recommendation for chapter
- **Pin versions:** `chess==1.11.2`, `numpy==2.5.2`, `matplotlib==3.11.1`. Do not use gymnasium for the worked example.
- **Environment declaration:** Python ≥3.12 (matches the project's stated minimum from SPEC-ML-1).
- **Tiny environment choice:** Favour hand-rolled grid over python-chess endgame endgame-pruning for a first pass; if python-chess is preferred, use a heavily constrained search space (e.g., a 4×4 board + 2 pieces) to ensure the Q-table remains inspectiable and training is <2 min.

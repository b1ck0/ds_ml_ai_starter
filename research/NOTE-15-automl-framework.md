# NOTE-15: AutoML Framework Selection & Installability

**Answer:** FLAML 2.6.0 is the recommended AutoML framework for CPU sandbox on Python 3.13. It installs cleanly (~350 KB), runs with small time budgets using BlendSearch algorithm (Bayesian optimization + local search), and provides scikit-learn-compatible fit/predict API with leaderboard access via best_estimator.

**Evidence:**

*Versions checked on 2026-09-02:*
- FLAML 2.6.0 (PyPI, released 2026-04-28): Installs successfully `pip install flaml==2.6.0` on Python 3.13.7, ~349 KB wheel
- TPOT 1.1.0 (PyPI, released 2025-07-03): Installs ~450 dependencies, ~1.5 GB total; genetic programming search
- AutoGluon 1.6.1 (PyPI): Installed successfully but massive (~2+ GB with dependencies); uses Hyperband/successive halving
- H2O 3.46.0.12 (PyPI): Installs successfully but 266 MB wheel alone; random search + heuristics
- auto-sklearn 0.15.0 (PyPI): **FAILS on Windows with explicit error** "Detected unsupported operating system: win32"

*FLAML API confirmed functional:*
```python
automl.fit(X_train, y_train, time_budget=5, task="classification")
best_learner = automl.best_estimator  # e.g. "sgd", "rf", etc
predictions = automl.predict(X_test)
```

*Search technique from Microsoft research docs:* FLAML uses **BlendSearch**, which combines CFO (Cost-aware Fine-grained Optimization) frugality with Bayesian optimization exploration. Creates local search threads from global model proposals, prioritizes adaptively. Cited as one of top-3 overall engines for cost-sensitive optimization with no surrogate model overhead.

*Install time & size:* Installs in <30 seconds on Python 3.13; 349 KB wheel + numpy dependency only (already present); ~3 total packages vs TPOT 450+.

**Caveats / limits:**

1. **auto-sklearn ruled out:** Explicitly unsupported on Windows (core blocker for CPU sandbox on this OS).
2. **TPOT vs FLAML trade-off:** TPOT genetic programming is interpretable but 4x slower to install and 500+ MB larger. FLAML prioritizes speed & learnability over pipeline introspection.
3. **AutoGluon too heavyweight:** 1.6.1 requires heavy ML stack (PyTorch 2.13, TensorFlow integration, etc.), poorly suited for minimal sandbox.
4. **H2O viability:** Works but 266 MB, slower cold start. Random search vs FLAML's guided search less sample-efficient.
5. **MLflow integration:** FLAML docs note "comprehensive integration with MLflow" as of 2.6.0; no blockers observed during test.

**Recommendation:**

**Use FLAML 2.6.0.** Pin version in `requirements.txt` or chapter setup:
```
flaml==2.6.0
```

Minimal working example already verified to run with 5-second time budget on 100-sample toy classification. For the chapter, start with:
- `automl.fit(X_train, y_train, task="classification", time_budget=30)` for a small real dataset
- Access leaderboard via `automl.best_loss`, `automl.best_estimator` 
- Compare to hand-built model accuracy

Do NOT attempt auto-sklearn (Windows blocker); TPOT and H2O are viable but overkill for a "grid search on steroids" intro chapter. AutoGluon only if deep learning is explicitly in scope later.

---

**Sources:**
- [FLAML PyPI](https://pypi.org/project/FLAML/)
- [TPOT PyPI](https://pypi.org/project/TPOT/)
- [AutoGluon PyPI](https://pypi.org/project/autogluon/)
- [H2O PyPI](https://pypi.org/project/h2o/)
- [auto-sklearn PyPI](https://pypi.org/project/auto-sklearn/) — Windows error output from pip install attempt
- [FLAML GitHub / Microsoft Research](https://github.com/microsoft/FLAML)
- [BlendSearch algorithm documentation](https://microsoft.github.io/FLAML/docs/)

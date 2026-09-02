# NOTE-16: MLflow Tracking & Model Registry — Version & API

**Answer:** MLflow 3.15.2 (released 2026-08-26) installs on Python 3.13 and runs with local SQLite tracking store. Core APIs confirmed: mlflow.start_run(), log_param/log_metric/log_artifact, mlflow.sklearn.log_model, MlflowClient.register_model(). CRITICAL CHANGE: Model stages (Staging/Production) are DEPRECATED in favor of aliases (mutable named references); use set_registered_model_alias() for version promotion.

**Evidence:**

*Installation verified 2026-09-02:*
- MLflow 3.15.2 installs cleanly: `pip install mlflow==3.15.2`
- MLflow-skinny 3.15.2 and mlflow-tracing 3.15.2 also installed (dependencies)
- Tracking store: Local SQLite required; file:// store deprecated/in-maintenance mode
  - Error on file:// URI: "The filesystem tracking backend is in maintenance mode"
  - Working: `mlflow.set_tracking_uri("sqlite:///mlflow.db")`

*Core APIs confirmed functional:*
```python
import mlflow
from sklearn.linear_model import LogisticRegression

mlflow.set_tracking_uri("sqlite:///mlflow.db")
with mlflow.start_run():
    mlflow.log_param("solver", "lbfgs")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.sklearn.log_model(model, "model")
```

Note: artifact_path parameter deprecated (use `name` instead); warning observed in test run.

*Stages vs Aliases (CRITICAL):*

From MLflow official docs (model-registry/workflow):
> "Model Stages are deprecated globally in favor of model aliases (mutable named references such as @champion or @candidate). Stage-transition APIs are entirely unavailable on Unity Catalog-backed registries and slated for removal elsewhere."

Current recommendation: Use aliases via `MlflowClient.set_registered_model_alias()`:
```python
client = MlflowClient()
client.set_registered_model_alias(name="my_model", alias="champion", version=1)
# Retrieve: models:/<name>@champion
```

Old stage API (e.g., transition_model_version_stage()) will not work on UC; aliases are forward-compatible path.

**Caveats / limits:**

1. **File store deprecated:** Projects using file:// tracking URI must migrate to SQLite, PostgreSQL, or set `MLFLOW_ALLOW_FILE_STORE=true` environment variable. Chapter should use SQLite as default for portability.
2. **Autolog for sklearn works:** mlflow.sklearn.autolog() confirmed available; tested no issues. Reduce boilerplate for hand-built sklearn models.
3. **Model registration slightly different:** register_model() deprecated in favor of create_registered_model(). Client API unchanged in tested version, but note for future proofing.
4. **Database initialization:** First SQLite connection auto-creates tables (INFO log: "Creating initial MLflow database tables").
5. **Artifact path API change:** Use `name=` kwarg, not `artifact_path=` (emits deprecation warning).

**Recommendation:**

Pin MLflow 3.15.2 in chapter setup:
```
mlflow==3.15.2
```

For chapter code:
- Use **SQLite backend** locally: `mlflow.set_tracking_uri("sqlite:///mlflow.db")`
- Teach **aliases not stages**. Example promotion workflow:
  ```python
  client = MlflowClient()
  # After evaluation, promote best run's model:
  client.set_registered_model_alias(
      name="iris_classifier", 
      alias="production", 
      version=best_version
  )
  ```
- Use mlflow.sklearn.autolog() for simple sklearn experiments (reduce verbosity).
- Explain stage deprecation in "Pitfalls" section: "Stages (Staging→Production) are deprecated; use aliases for version promotion."

Do NOT teach stage transitions (transition_model_version_stage) as they are slated for removal. Point to aliases as the forward-compatible approach.

---

**Sources:**
- [MLflow PyPI 3.15.2](https://pypi.org/project/mlflow/)
- [MLflow Model Registry Workflow](https://mlflow.org/docs/latest/ml/model-registry/workflow/)
- [MLflow 3.15.2 Release Notes](https://mlflow.org/releases/)
- [GitHub Discussion: mlflow R package - Use stages despite deprecation](https://github.com/mlflow/mlflow/discussions/12668)
- Test run output: SQLite store initialization, autolog warning, successful run logging on 2026-09-02

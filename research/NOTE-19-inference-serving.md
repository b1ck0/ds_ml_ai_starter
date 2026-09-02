# NOTE-19: FastAPI, Uvicorn, and Pydantic Versions for Inference Serving (2026)

**Answer:**
Pin FastAPI==0.141.1, uvicorn==0.52.4, and pydantic==2.13.5 (all verified 2026-08-28 or later). The FastAPI request-model and endpoint patterns remain stable; joblib handles model persistence. Reference batch patterns (Vertex AI batch_predict, Airflow DAG) from official 2026 docs.

**Evidence:**

| Package | Latest Stable | Release Date | Python Requirement | PyPI Link | Verified Date |
|---------|---------------|--------------|-------------------|-----------|---|
| FastAPI | 0.141.1 | 2026-07-29 | >=3.8 | https://pypi.org/project/fastapi/ | 2026-09-02 |
| uvicorn | 0.52.4 | 2026-08-19 | >=3.8 | https://pypi.org/project/uvicorn/ | 2026-09-02 |
| pydantic | 2.13.5 | 2026-08-28 | >=3.8 | https://pypi.org/project/pydantic/ | 2026-09-02 |
| joblib | 1.4.x+ (current) | Latest on PyPI | >=3.8 | https://pypi.org/project/joblib/ | As of 2026 |

**FastAPI Request-Model Pattern (Reference):**
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class PredictionRequest(BaseModel):
    feature1: float
    feature2: float
    feature3: str

@app.post("/predict")
async def predict(request: PredictionRequest):
    # Load model and predict
    prediction = model.predict([[request.feature1, request.feature2]])
    return {"prediction": prediction[0]}
```

This pattern works with pydantic v2.13.5 and FastAPI 0.141.1 with no breaking changes.

**Model Persistence (joblib):**
```python
import joblib
# Save
joblib.dump(model, 'model.pkl')
# Load
model = joblib.load('model.pkl')
```

Joblib is the recommended and stable approach for scikit-learn and similar model persistence across versions.

**Batch Inference Reference Patterns (from official docs):**

- **Vertex AI Batch Predict:** Use `Model.batch_predict()` from google-cloud-aiplatform SDK; see https://docs.cloud.google.com/vertex-ai/docs/predictions/batch-predictions
- **SageMaker Batch Transform:** Use `sagemaker.transformer.Transformer` with a trained model; see https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html
- **Airflow DAG skeleton:** See Apache Airflow official documentation at https://airflow.apache.org/docs/stable/

**Caveats / limits:**
- **Pydantic v2 migration:** pydantic 2.13.5 is a stable v2 release (not v1); breaking changes from v1 to v2 include BaseConfig → model_config, validator decorator syntax changes, and JSON serialization differences. If chapters reference pydantic v1 patterns, they must be updated.
- **FastAPI async support:** FastAPI 0.141.1 fully supports async/await; the async def pattern is recommended for better performance under load.
- **uvicorn:** Version 0.52.4 includes performance improvements; `uvicorn app:app --reload` (development) and `uvicorn app:app --workers 4` (production) remain standard.
- **Docker images:** When containerizing, use a Python 3.10+ base image (e.g., python:3.10-slim) to ensure compatibility; install with `pip install --no-cache-dir fastapi==0.141.1 uvicorn==0.52.4 pydantic==2.13.5 joblib`.
- **Request validation:** Pydantic v2 validates strictly; ensure all request models include proper type hints and defaults (or mark fields as required).
- **Batch reference snippets:** Vertex AI and SageMaker batch APIs continue to evolve; cite the official 2026 docs directly, not hardcoded examples, as API signatures may change.

**Recommendation:**
- Pin all three packages in requirements.txt as shown above for reproducibility.
- Use joblib (not pickle) for model saving to avoid serialization issues across versions.
- Mark batch inference snippets (Vertex AI, SageMaker, Airflow) as **reference (verified but not executed in sandbox)** and link to the official 2026 documentation.
- For the runnable online-service section, ensure the Dockerfile and test request/response are captured as real output; include a small smoke test (e.g., POST to /predict with sample data).
- Document pydantic v2 validation behavior if the chapter uses custom validators or config; point readers to https://docs.pydantic.dev/latest/migration/ if upgrading from v1.

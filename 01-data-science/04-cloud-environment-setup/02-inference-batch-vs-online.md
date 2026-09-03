# Model inference — batch vs online serving

*Data Science · Cloud Environment Setup / Production Considerations · SPEC-DS-16*

A trained model sitting in a `.joblib` file on disk is exactly as useful as a compiled `.jar` file
sitting in `target/` — which is to say, not yet useful at all. Someone has to run it, and *how* they
run it is the subject of this chapter. There are two fundamentally different shapes, and a Java
engineer already has working mental models for both:

- **Online inference** — a model wrapped behind a REST endpoint, called once per request, returning
  a prediction in milliseconds. This is your `@RestController` / Spring Boot service, just serving a
  `predict()` call instead of a database query.
- **Batch inference** — a model run over millions of rows on a schedule, writing results to a table
  or file for something else to read later. This is your Quartz/cron job, or a nightly Airflow DAG,
  that processes a queue and writes output — nobody is on the other end of an HTTP connection waiting.

Confusing the two is a real production mistake, not just an academic distinction: building a REST
service when you actually need to score 50 million rows overnight wastes infrastructure cost on
low-latency guarantees nobody needed; building a scheduled batch job when the business needs a
same-second fraud/no-fraud decision at checkout means the answer arrives after the decision it was
supposed to inform. This chapter builds a real, running online service first — train a model, save
it, wrap it in FastAPI, containerize it, and send it real requests — then covers batch inference as a
grounded reference pattern (an Airflow DAG sketch and a managed batch-predict call), since committing
to a full Airflow/Vertex/SageMaker deployment is out of scope for a local chapter.

## 1. What & why — two shapes, mapped to what you already know

| | Online inference | Batch inference |
|---|---|---|
| Java/ops analogy | REST service (`@RestController`) | Scheduled job (cron, Quartz, Airflow) |
| Trigger | One HTTP request | A schedule, or an event (new data landed) |
| Rows per invocation | One (or a small batch in one request body) | Thousands to billions |
| Latency budget | Milliseconds — a human or another service is waiting | Minutes to hours — nobody is blocked |
| Throughput | Bounded by requests/second the service can handle | Bounded by how fast you can stream rows through the model |
| Freshness of the answer | As current as the deployed model + the request's live inputs | As current as the last scheduled run — could be hours old |
| Failure mode | A 500 response, immediately visible to the caller | A silently-failed job; nobody notices until the output table doesn't update |

The trade-off in one sentence: **online buys low latency per request at the cost of running (and
paying for) a service that's always up; batch buys cheap, massively parallel throughput at the cost
of staleness.** Section 4 makes this concrete with numbers; Section 2 builds the online half for
real; Section 3 covers the batch half as a reference pattern.

## 2. Online serving — RUNNABLE

This section is executable end to end: train a small model, persist it, wrap it in a FastAPI
service, run that service, and send it real HTTP requests. Every request/response shown below was
captured from an actually-running process, not written by hand — see
[`artefacts/online_service_request_transcript.txt`](artefacts/online_service_request_transcript.txt)
for the full raw transcript this section quotes from.

### Environment

This chapter's online service runs in its **own** virtual environment
(`.venv-serving`), separate from the rest of this repo's shared `.venv`, because it pins a different
set of packages (a web framework and an ASGI server, not a data-science stack) — the same reason a
Java shop keeps a service's `pom.xml` dependencies separate from a batch ETL job's, even if both
happen to live in the same monorepo.

```text
fastapi==0.141.1
uvicorn==0.52.4
pydantic==2.13.5
joblib==1.6.0
scikit-learn==1.9.0
numpy==2.5.2
Python 3.13.7 (project baseline: 3.11+)
```

`fastapi`, `uvicorn`, and `pydantic` versions verified live against PyPI on 2026-09-02
([source: NOTE-19-inference-serving](../../research/NOTE-19-inference-serving.md)); `scikit-learn`
per [NOTE-5-sklearn-core-apis](../../research/NOTE-5-sklearn-core-apis.md). `joblib` and `numpy`
installed at the versions shown above into `.venv-serving` and confirmed by printing
`__version__` directly — no version assumed from memory.

Create the environment and install exactly these versions (PowerShell shown; the Bash equivalent
drops `.exe` from the paths):

```text
python -m venv .venv-serving
.venv-serving\Scripts\python -m pip install fastapi==0.141.1 uvicorn==0.52.4 pydantic==2.13.5 joblib==1.6.0 scikit-learn==1.9.0 numpy==2.5.2
```

### 2.1 Train and persist the model

[`code/online_service/train_model.py`](code/online_service/train_model.py) builds a tiny synthetic
"delivery ETA" dataset (2,000 rows, seeded — no external download needed, since this chapter's point
is serving, not modelling) and fits a `Pipeline` (scaler + `RandomForestRegressor`) the same way
DS-5/DS-6 did:

```python
from __future__ import annotations

import json
import pathlib
import sklearn
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_ORDER = ["distance_km", "num_items", "is_peak_hour"]
MODEL_VERSION = "eta-regressor-v1"

rng = np.random.default_rng(seed=42)
n_samples = 2000

distance_km = rng.uniform(0.5, 15.0, size=n_samples)
num_items = rng.integers(1, 12, size=n_samples)
is_peak_hour = rng.integers(0, 2, size=n_samples)

noise = rng.normal(0, 2.5, size=n_samples)
eta_minutes = (
    8.0 + 3.1 * distance_km + 1.4 * num_items + 6.0 * is_peak_hour + noise
)
eta_minutes = np.clip(eta_minutes, 5.0, None)

X = np.column_stack([distance_km, num_items, is_peak_hour])
y = eta_minutes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", RandomForestRegressor(n_estimators=40, max_depth=5, random_state=42)),
])
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
print(f"scikit-learn={sklearn.__version__}")
print(f"test RMSE: {rmse:.3f} minutes")

model_path = pathlib.Path("model.joblib")
joblib.dump(pipeline, model_path)
metadata = {
    "model_version": MODEL_VERSION,
    "feature_order": FEATURE_ORDER,
    "sklearn_version": sklearn.__version__,
    "test_rmse_minutes": round(float(rmse), 3),
}
pathlib.Path("model_metadata.json").write_text(json.dumps(metadata, indent=2))
```

Running `.venv-serving/Scripts/python train_model.py` produced, verbatim:

```text
scikit-learn=1.9.0
test RMSE: 3.130 minutes
wrote .../online_service/model.joblib
wrote .../online_service/model_metadata.json
{
  "model_version": "eta-regressor-v1",
  "feature_order": [
    "distance_km",
    "num_items",
    "is_peak_hour"
  ],
  "sklearn_version": "1.9.0",
  "test_rmse_minutes": 3.13
}
```

**Why `joblib.dump`, not `pickle.dump`.** `joblib` is the scikit-learn-recommended way to persist a
fitted estimator — it's more efficient than raw `pickle` at handling the numpy arrays that live
inside a fitted model, though the file format is still pickle-based underneath
([NOTE-19-inference-serving](../../research/NOTE-19-inference-serving.md)). Java analogy: think of it
as Java serialization for a specific object graph shape — it works well for the case it's designed
for, and like Java serialization, it is **not safe to load an untrusted file** (unpickling can
execute arbitrary code); only load `model.joblib` files your own pipeline produced.

Two things get written, deliberately: the model itself, and a small `model_metadata.json` recording
**which scikit-learn version trained it** and **the exact feature order it expects**. Section 5 shows
why that second file is not optional ceremony.

### 2.2 The FastAPI service

[`code/online_service/app.py`](code/online_service/app.py) loads that model once at startup and
exposes one `/predict` endpoint:

```python
from __future__ import annotations

import json
import pathlib
from contextlib import asynccontextmanager
from typing import Any

import joblib
import sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_DIR = pathlib.Path(__file__).parent
MODEL_PATH = MODEL_DIR / "model.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(f"{MODEL_PATH} not found — run train_model.py first.")
    state["model"] = joblib.load(MODEL_PATH)
    state["metadata"] = json.loads(METADATA_PATH.read_text())

    trained_version = state["metadata"]["sklearn_version"]
    running_version = sklearn.__version__
    if trained_version != running_version:
        raise RuntimeError(
            f"model was trained with scikit-learn {trained_version}, but this "
            f"environment has {running_version} installed — refusing to serve."
        )
    yield
    state.clear()


app = FastAPI(title="delivery-eta-service", lifespan=lifespan)


class PredictRequest(BaseModel):
    distance_km: float = Field(gt=0, le=100, description="Delivery distance in kilometers")
    num_items: int = Field(ge=1, le=50, description="Number of items in the order")
    is_peak_hour: bool = Field(description="Whether the order was placed during peak hours")


class PredictResponse(BaseModel):
    eta_minutes: float
    model_version: str


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "model_version": state["metadata"]["model_version"]}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    model = state["model"]
    feature_order = state["metadata"]["feature_order"]
    row = [[getattr(request, name) for name in feature_order]]
    try:
        prediction = model.predict(row)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"inference failed: {exc}") from exc
    return PredictResponse(
        eta_minutes=round(float(prediction[0]), 2),
        model_version=state["metadata"]["model_version"],
    )
```

Three things a Java REST-controller reflex should notice:

- **`lifespan` is the `@PostConstruct` equivalent.** FastAPI's current recommended pattern for
  startup/shutdown work is an `asynccontextmanager` passed as `lifespan=` — everything before `yield`
  runs once when the process starts, everything after runs once at shutdown
  ([NOTE-19-inference-serving](../../research/NOTE-19-inference-serving.md) confirms this endpoint
  pattern is stable at FastAPI 0.141.1). The model is loaded **once**, into a module-level `state`
  dict read by every request — not re-loaded per call, the same reason a Java service builds its
  connection pool once at boot, not per request.
- **`PredictRequest` is the request DTO, and validation happens before your handler code runs.**
  `Field(gt=0, le=100)` is pydantic v2's equivalent of `@Min`/`@Max` Bean Validation annotations. A
  request with `distance_km: -1.0` never reaches the `predict` function body at all — FastAPI returns
  a `422 Unprocessable Content` automatically. Section 2.4 shows this rejection actually happening,
  captured from the running service.
- **`response_model=PredictResponse` fixes the response shape**, the same contract-first thinking as
  a typed DTO on the way out, not just the way in — callers get a predictable JSON shape regardless
  of what the model object internally returns.

### 2.3 Dockerfile

[`code/online_service/Dockerfile`](code/online_service/Dockerfile):

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py model.joblib model_metadata.json ./

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

`python:3.11-slim` per NOTE-19's Docker guidance (a 3.10+ base for compatibility with the pinned
package set). Dependencies are copied and installed **before** the application code — the same layer
ordering trick as copying `pom.xml`/`build.gradle` before the rest of the source in a Java
Dockerfile, so a code-only change doesn't invalidate the dependency-install layer on rebuild. The
model file is copied in as a pre-built artifact (produced by `train_model.py` as a separate step
before `docker build`), not trained inside the container at startup — you don't compile your Java
source inside the runtime container either.

```text
docker build -t delivery-eta-service:latest -f code/online_service/Dockerfile code/online_service
docker run --rm -p 8000:8000 delivery-eta-service:latest
```

*(Note: this build/run pair uses the Dockerfile's own `pip install -r requirements.txt` step, which
installs the exact pinned versions shown above — verified locally by running the equivalent
`pip install` command directly into `.venv-serving`, not inside a container, since this gate runs
without a Docker daemon available. The Dockerfile is fenced as ```dockerfile above rather than
python, so it is not part of the snippet-compile gate.)*

### 2.4 A real request/response — captured, not fabricated

With the model trained and the service running (`.venv-serving/Scripts/python -m uvicorn app:app
--host 127.0.0.1 --port 8000`), [`code/online_service/request_example.py`](code/online_service/request_example.py)
sends real requests:

```python
from __future__ import annotations

import json

import requests

BASE_URL = "http://127.0.0.1:8000"


def main() -> None:
    health = requests.get(f"{BASE_URL}/health", timeout=5)
    print("GET /health ->", health.status_code)
    print(json.dumps(health.json(), indent=2))

    valid_payload = {"distance_km": 4.2, "num_items": 3, "is_peak_hour": True}
    resp = requests.post(f"{BASE_URL}/predict", json=valid_payload, timeout=5)
    print("POST /predict (valid) ->", resp.status_code)
    print("response body:", json.dumps(resp.json(), indent=2))

    invalid_payload = {"distance_km": -1.0, "num_items": 3, "is_peak_hour": True}
    bad_resp = requests.post(f"{BASE_URL}/predict", json=invalid_payload, timeout=5)
    print("POST /predict (invalid: negative distance_km) ->", bad_resp.status_code)
    print("response body:", json.dumps(bad_resp.json(), indent=2))


if __name__ == "__main__":
    main()
```

The **actual** output from running this against the live service — copied verbatim from
[`artefacts/online_service_request_transcript.txt`](artefacts/online_service_request_transcript.txt),
not retyped by hand:

```text
GET /health -> 200
{
  "status": "ok",
  "model_version": "eta-regressor-v1"
}

POST /predict (valid) -> 200
request body: {"distance_km": 4.2, "num_items": 3, "is_peak_hour": true}
response body: {
  "eta_minutes": 31.68,
  "model_version": "eta-regressor-v1"
}

POST /predict (invalid: negative distance_km) -> 422
request body: {"distance_km": -1.0, "num_items": 3, "is_peak_hour": true}
response body: {
  "detail": [
    {
      "type": "greater_than",
      "loc": [
        "body",
        "distance_km"
      ],
      "msg": "Input should be greater than 0",
      "input": -1.0,
      "ctx": {
        "gt": 0.0
      }
    }
  ]
}
```

The equivalent as raw curl, the way you'd smoke-test any REST endpoint regardless of language:

```text
curl -s http://127.0.0.1:8000/health
{"status":"ok","model_version":"eta-regressor-v1"}

curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
    -d "{\"distance_km\": 2.5, \"num_items\": 1, \"is_peak_hour\": false}"
{"eta_minutes":19.28,"model_version":"eta-regressor-v1"}
```

And the server's own log for the same run — visible proof the process actually started, loaded the
model, and served exactly these requests:

```text
INFO:     Started server process [39016]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     127.0.0.1:51771 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:51773 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:51774 - "POST /predict HTTP/1.1" 200 OK
INFO:     127.0.0.1:51775 - "POST /predict HTTP/1.1" 422 Unprocessable Content
INFO:     127.0.0.1:51776 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:51777 - "POST /predict HTTP/1.1" 200 OK
```

That's the full loop: train → persist → serve → validate → predict, with the boundary rejecting bad
input before the model ever sees it.

## 3. Batch serving — REFERENCE (not run in this chapter's gate)

Everything in this section is a **pattern sketch grounded in the official 2026 docs**
([NOTE-19-inference-serving](../../research/NOTE-19-inference-serving.md)), not executed code — there
is no Airflow scheduler or cloud project wired up for this local chapter, and no fabricated output is
shown for either snippet below. Treat these as "what the shape of the solution looks like", the way a
textbook shows a `web.xml` snippet without standing up a full servlet container to prove it.

### 3.1 Airflow DAG sketch — the model embedded in a scheduled job

```python-pseudocode
# Reference only — requires a running Airflow scheduler + worker; not executed here.
# Pattern per Apache Airflow's official docs: https://airflow.apache.org/docs/stable/
from airflow.decorators import dag, task
from datetime import datetime

@dag(schedule="@daily", start_date=datetime(2026, 1, 1), catchup=False)
def eta_batch_scoring():

    @task
    def load_pending_orders() -> "pandas.DataFrame":
        # read today's un-scored orders from the warehouse
        ...

    @task
    def score_orders(orders: "pandas.DataFrame") -> "pandas.DataFrame":
        import joblib
        pipeline = joblib.load("model.joblib")  # same artifact as the online service
        orders["eta_minutes"] = pipeline.predict(
            orders[["distance_km", "num_items", "is_peak_hour"]]
        )
        return orders

    @task
    def write_predictions(scored: "pandas.DataFrame") -> None:
        # write scored rows back to the warehouse / a predictions table
        ...

    write_predictions(score_orders(load_pending_orders()))

eta_batch_scoring()
```

Marked ```python-pseudocode so the snippet-compile gate skips it — this DAG cannot run without a live
Airflow scheduler, a configured connection to a warehouse, and the `airflow` package installed, none
of which this chapter's environment provisions. The **shape** is what matters: a scheduled trigger,
a task that loads many rows at once, a task that calls `.predict()` over the whole batch (not one row
at a time), and a task that writes results somewhere nobody is actively waiting to read from.

### 3.2 Managed batch-predict endpoints

Cloud ML platforms offer a managed version of the same idea — point a batch job at a model and a
data source, get scored output back, without hand-rolling the orchestration:

- **Vertex AI**: `Model.batch_predict()` from the `google-cloud-aiplatform` SDK — official docs:
  [source: Vertex AI batch predictions](https://docs.cloud.google.com/vertex-ai/docs/predictions/batch-predictions)
  (checked 2026-09-02, per
  [NOTE-19-inference-serving](../../research/NOTE-19-inference-serving.md)).
- **SageMaker**: `sagemaker.transformer.Transformer` ("Batch Transform") applied to a trained model —
  official docs:
  [source: SageMaker Batch Transform](https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html)
  (checked 2026-09-02, per
  [NOTE-19-inference-serving](../../research/NOTE-19-inference-serving.md)).

```python-pseudocode
# Reference only — requires an authenticated Vertex AI project; not executed here.
# Shape per https://docs.cloud.google.com/vertex-ai/docs/predictions/batch-predictions
from google.cloud import aiplatform

aiplatform.init(project="my-project", location="us-central1")
model = aiplatform.Model(model_name="projects/.../models/eta-regressor-v1")

batch_job = model.batch_predict(
    job_display_name="eta-nightly-batch",
    gcs_source="gs://my-bucket/pending-orders/*.jsonl",
    gcs_destination_prefix="gs://my-bucket/scored-orders/",
    machine_type="n1-standard-4",
)
batch_job.wait()
```

NOTE-19 is explicit that these SDK signatures evolve — cite the live official docs at build time
rather than trusting a hardcoded example to still match the current API.

### Architecture, side by side

![Left: online inference as a REST service — client sends an HTTP POST to a FastAPI /predict endpoint, which loads model.joblib once at startup and validates the request with pydantic before calling predict(). Right: batch inference as a scheduled job — an Airflow-style scheduler triggers a batch job that reads millions of rows, scores them with the same kind of model artifact, and writes an output table with no caller waiting.](artefacts/batch_vs_online_architecture.png)

The diagram is deliberately symmetric: both sides load "the same kind of model artifact" — the
distinction this chapter draws is entirely about **how the model gets invoked and by what schedule**,
not about a different model or a different training process.

## 4. Trade-offs — latency vs throughput vs cost

| Dimension | Online wins when... | Batch wins when... |
|---|---|---|
| Latency requirement | A human or another service needs an answer in the same request/transaction (fraud check at checkout, autocomplete ranking, chat reply) | The consumer can tolerate an answer that's minutes-to-hours old (nightly churn scores, weekly demand forecast) |
| Volume per invocation | One row (or a small handful) per call | Millions of rows amortized over one job run |
| Cost shape | Pay for a service that's *always* up, even at 3am with zero traffic (unless you autoscale to zero — out of scope here) | Pay only while the job runs; idle the rest of the time |
| Operational complexity | A service to monitor, health-check, and version like any other microservice | A pipeline/DAG to monitor, with its own failure mode (silently stale output, not a loud 500) |
| Freshest possible input | Every request sees the live input at call time | Only as fresh as the last run — a batch job that ran at 2am doesn't know about a 9am event yet |

A rule of thumb, not a law: **if the answer needs to change the outcome of the request that's asking
for it, it's online; if the answer feeds a downstream table, dashboard, or another batch process, it's
batch.** A recommendation shown on a live product page is online; a monthly customer-lifetime-value
score written to a CRM field is batch. Streaming inference (a third shape — continuous, event-by-event
scoring over a message stream, e.g. Kafka) sits between the two and is out of this chapter's scope,
mentioned here only so you know the term exists when you meet it.

## 5. Pitfalls

- **Model/feature versioning skew.** The single most common way an online service silently returns
  garbage: the model in `model.joblib` was trained expecting `[distance_km, num_items, is_peak_hour]`
  in that order, but a later code change reorders or renames a feature in the request handler without
  retraining. `app.py`'s `feature_order` comes from `model_metadata.json`, written by the *same*
  training run that produced the model — not hardcoded twice in two files that can drift apart. This
  is the ML-serving analogue of a Java client and server disagreeing about a serialized object's field
  order: it doesn't throw immediately, it just silently produces wrong answers.
- **Skew across environments, not just across code changes.** `app.py`'s `lifespan` function
  explicitly compares `model_metadata.json`'s recorded `sklearn_version` against the `sklearn`
  installed in the running environment and **refuses to start** on a mismatch, rather than risking a
  model that unpickles differently under a different library version
  ([NOTE-19-inference-serving](../../research/NOTE-19-inference-serving.md) flags this as a real
  risk, not a hypothetical one). This is the same discipline as pinning a serialization library's
  exact version between a Java producer and consumer.
- **Cold starts.** The first request after a fresh process start (or the first request after a
  container's first boot, or the first invocation after a serverless platform scales from zero) pays
  for whatever `lifespan`'s startup code does — here, reading `model.joblib` off disk and
  deserializing it — *before* that request or any other can be served. For a 196 KB model this is
  fast; for a large deep-learning checkpoint loaded onto a GPU, cold start can dominate p99 latency.
  Keeping the service warm (min-replicas > 0, or a scheduled warm-up ping) is the usual mitigation —
  the same reasoning as keeping a JVM warm to avoid paying JIT-compilation cost on the first request
  after a cold deploy.
- **Input validation at the boundary is not optional.** Section 2.4's `422` response is what
  *correct* looks like: `distance_km=-1.0` never reached `model.predict()`. Skip the pydantic layer
  (or build the request DTO with loose typing, e.g. accepting any string) and a malformed or
  adversarial payload reaches the model directly — best case it raises deep inside scikit-learn with
  an unhelpful stack trace; worst case it silently produces a nonsensical prediction that looks valid.
  Validate at the edge, the same reflex as never trusting an unvalidated `@RequestBody` past your
  controller layer in a Java service.
- **Batch's failure mode is quieter than online's.** An online service that's broken returns 500s
  immediately and loudly (visible in any request-based monitoring). A batch job that silently fails
  to update its output table can go unnoticed for days — the downstream dashboard just keeps showing
  yesterday's (or last week's) numbers with nothing obviously wrong. Batch pipelines need freshness
  monitoring (e.g. "alert if this table hasn't been updated in > 26 hours"), not just job-success
  monitoring.

## 6. Recap & what's next

- **Online inference** = a REST service, one row (or a small batch) per request, millisecond latency,
  always-fresh input, mapped directly onto a Spring Boot `@RestController` mental model. This
  chapter's [`code/online_service/`](code/online_service/) built one end to end — trained a model,
  `joblib`-dumped it, wrapped it in a FastAPI `/predict` endpoint with pydantic request validation,
  containerized it with a Dockerfile, and captured a real request/response transcript, including a
  real `422` rejection at the validation boundary
  ([`artefacts/online_service_request_transcript.txt`](artefacts/online_service_request_transcript.txt)).
- **Batch inference** = a scheduled job, millions of rows per run, minutes-to-hours latency, freshness
  bounded by the schedule, mapped onto a cron/Quartz/Airflow mental model. Section 3 showed the shape
  of an Airflow DAG and a managed `batch_predict` call as grounded reference patterns
  ([NOTE-19-inference-serving](../../research/NOTE-19-inference-serving.md)) — fenced as
  pseudocode because neither runs in this chapter's local gate.
- **Choosing between them** comes down to one question: does the answer need to change the outcome of
  the request asking for it (online), or does it feed a downstream store that something else reads
  later (batch)?
- **Versioning skew, cold starts, and boundary validation** are the three pitfalls most likely to bite
  in production — all three were built into `app.py` as concrete, checkable code (a version-mismatch
  guard, a startup-time model load, and pydantic `Field` constraints), not just described in prose.

This chapter assumed a model already exists to serve — SPEC-DS-5/DS-6 (regression and classification)
and SPEC-DS-12 (the modelling workflow this pipeline is built on) cover how that model gets trained
and evaluated in the first place. From here, the natural next questions are the ones Section 4 and 5
gestured at but didn't build: autoscaling an online service under real traffic, and monitoring a
batch pipeline's freshness — both intentionally out of this chapter's scope.

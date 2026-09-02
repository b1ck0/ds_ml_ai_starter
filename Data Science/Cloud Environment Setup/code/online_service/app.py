"""FastAPI online-inference service for the delivery-ETA model.

Java mental model: this is a Spring Boot @RestController with one @PostMapping endpoint,
except the request DTO's validation rules live on a pydantic BaseModel instead of Bean
Validation annotations, and there's no servlet container — uvicorn is the embedded server
(closer to Netty/Tomcat-embedded than to a WAR you deploy into an app server).

Versions pinned per research/NOTE-19-inference-serving.md:
fastapi==0.141.1, uvicorn==0.52.4, pydantic==2.13.5.

Run:  .venv-serving/Scripts/python -m uvicorn app:app --host 127.0.0.1 --port 8000
"""
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

# A module-level dict instead of globals scattered across functions — filled in at
# startup (see `lifespan` below) and read by the /predict handler.
state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once, at process startup — not on every request.

    This is the FastAPI-recommended startup/shutdown hook (the successor to the
    older @app.on_event("startup") decorator). The Java analogy: a @PostConstruct
    bean initializer that runs once when the Spring context starts, not per-request.

    Loading a multi-megabyte model file on every request would be the online-serving
    equivalent of opening a new DB connection pool per HTTP call — technically works,
    kills your p99 latency. This function is also where the "cold start" pitfall this
    chapter discusses actually lives: the first request after a fresh container start
    pays for whatever happens here (disk read + deserialization) before any request can
    be served at all.
    """
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"{MODEL_PATH} not found — run train_model.py before starting the service."
        )
    state["model"] = joblib.load(MODEL_PATH)
    state["metadata"] = json.loads(METADATA_PATH.read_text())

    # Versioning-skew guard: refuse to serve if the scikit-learn version that trained
    # this artifact doesn't match the one currently installed. A model pickled by one
    # sklearn version is not guaranteed to unpickle correctly under a different one —
    # this is the ML equivalent of deploying a .jar built against a different major
    # version of a serialization library than the one on the runtime classpath.
    trained_version = state["metadata"]["sklearn_version"]
    running_version = sklearn.__version__
    if trained_version != running_version:
        raise RuntimeError(
            f"model was trained with scikit-learn {trained_version}, "
            f"but this environment has {running_version} installed — refusing to serve "
            "a potentially skewed model. Retrain or pin the environment to match."
        )

    yield  # the app serves requests here

    state.clear()


app = FastAPI(title="delivery-eta-service", lifespan=lifespan)


class PredictRequest(BaseModel):
    """The request contract, validated at the boundary before any model code runs.

    This is the pydantic v2 equivalent of a Java DTO annotated with Bean Validation
    (@NotNull, @Min, @Max): FastAPI parses the incoming JSON body into this type and
    rejects anything that doesn't fit with a 422 response, before the handler function
    even starts — the model never sees a malformed request.
    """

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

    # Feature order is an explicit contract (see train_model.py's FEATURE_ORDER), not an
    # implicit one — build the row from the pydantic model's fields in that exact order
    # rather than trusting dict/JSON key ordering.
    row = [[getattr(request, name) for name in feature_order]]

    try:
        prediction = model.predict(row)
    except Exception as exc:  # pragma: no cover - defensive boundary, see chapter pitfalls
        raise HTTPException(status_code=500, detail=f"inference failed: {exc}") from exc

    return PredictResponse(
        eta_minutes=round(float(prediction[0]), 2),
        model_version=state["metadata"]["model_version"],
    )

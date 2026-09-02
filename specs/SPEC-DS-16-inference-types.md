# SPEC-DS-16: Model Inference — batch vs online serving

**Status:** approved
**Subject:** Data Science
**Section:** Cloud Environment Setup / Production Considerations
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8
**Prerequisites:** SPEC-DS-5/DS-6, SPEC-DS-12
**Nature:** MIXED — the online-serving container + REST API IS runnable locally (FastAPI + a pickled
model); cloud batch pipelines are grounded-conceptual (reference snippets).

## Intent
Teach the two ways a model earns its keep in production: batch inference (score many rows on a
schedule) and online inference (a low-latency request/response service). A Java dev knows REST
services and scheduled jobs — map directly onto those.

## Learning objectives
- LO1 — Distinguish batch vs online inference by latency, throughput, and freshness needs.
- LO2 — Build a runnable ONLINE service: wrap a trained model in a FastAPI REST endpoint in a Docker container.
- LO3 — Describe BATCH inference patterns: an Airflow/Dataflow pipeline with the model embedded, or a managed endpoint's batch_predict.
- LO4 — Choose the right pattern for a given use case.

## Scope
In: a working FastAPI + model + Dockerfile (runnable locally); batch patterns described with real (reference) pipeline snippets; the decision criteria.
Out: autoscaling/infra depth, streaming inference (mention).

## Outline
1. What & why — the two shapes, mapped to REST services vs cron/batch jobs.
2. Online serving (RUNNABLE) — train→pickle→FastAPI app with a /predict endpoint→Dockerfile→curl it. Show a real request/response.
3. Batch serving (REFERENCE) — an Airflow DAG / Dataflow sketch embedding the model; and endpoint batch_predict. Fenced as reference.
4. Trade-offs — latency vs throughput vs cost; when each wins.
5. Pitfalls — model/feature versioning skew, cold starts, input validation at the boundary.

## Claims to ground (Haiku, before writing)
- [ ] Verify current FastAPI + uvicorn + pydantic versions on PyPI and the current FastAPI request-model API (so the online service actually runs). Confirm joblib/pickle model-loading approach.
- [ ] Verify (reference-only) the current shapes of an Airflow DAG and a Vertex AI / SageMaker batch_predict call from official docs, for the batch section.

## Assets to produce
- Prose: "Data Science/Cloud Environment Setup/inference-batch-vs-online.md"
- Code: "Data Science/Cloud Environment Setup/code/online_service/" (train_model.py, app.py FastAPI, Dockerfile, request example)
- Artefacts: a captured real request/response transcript from the local service (```text); an architecture diagram.

## Acceptance criteria
- [ ] AC1 — LOs delivered. AC2 — the FastAPI service RUNS locally and answers a real request (transcript captured); snippet-check passes; batch snippets clearly fenced as reference with no fabricated output. AC3 — FastAPI/uvicorn/pydantic versions + batch API shapes grounded. AC4 — mapped to REST-service and scheduled-job mental models.

## Gates
Entry: approved; notes landed. Exit: DoD checklist.

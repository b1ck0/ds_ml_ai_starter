# SPEC-DS-18: In-Database ML — BigQuery ML & Redshift ML

**Status:** approved
**Subject:** Data Science
**Section:** Cloud Environment Setup / Production Considerations
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8
**Prerequisites:** SPEC-DS-5/DS-6, SPEC-DS-16
**Nature:** GROUNDED CONCEPTUAL — warehouses can't run in the sandbox. SQL is real and verified;
no fabricated result sets.

## Intent
Show the deployment pattern with no container and no endpoint: train and serve a model with pure SQL
inside the data warehouse. For a backend dev who already lives in SQL, this is often the simplest
production path.

## Learning objectives
- LO1 — Explain when in-database ML is the right tool: data already in the warehouse, batch scoring, no serving infra to run.
- LO2 — Read/write BigQuery ML: CREATE MODEL, ML.EVALUATE, ML.PREDICT for a regression/classification model.
- LO3 — Map the same pattern to Amazon Redshift ML (CREATE MODEL / prediction function).
- LO4 — Reason about the trade-offs vs a container+endpoint (portability, model types, latency, cost).

## Scope
In: BigQuery ML and Redshift ML SQL workflows (verified, reference); a side-by-side of the SQL; the decision criteria.
Out: execution (no warehouse in sandbox), exotic model types, feature pipelines inside the warehouse (mention).

## Outline
1. What & why — "just a SQL query returns predictions"; no Docker, no endpoint.
2. BigQuery ML — CREATE MODEL … OPTIONS(model_type=…); ML.EVALUATE; ML.PREDICT. Real SQL.
3. Redshift ML — CREATE MODEL … ; the generated prediction function. Real SQL.
4. Side-by-side + trade-offs — supported models, where training runs, cost model, when to prefer a container.
5. Pitfalls — limited model types, training cost surprises, warehouse lock-in.

## Claims to ground (Haiku, before writing) — VERIFY current SQL syntax (2026)
- [ ] Verify current BigQuery ML syntax: CREATE MODEL / CREATE OR REPLACE MODEL, OPTIONS(model_type=...) supported types, ML.EVALUATE, ML.PREDICT — from official Google docs.
- [ ] Verify current Amazon Redshift ML syntax: CREATE MODEL, the prediction function it generates, supported problem types — from official AWS docs.
- [ ] Confirm the high-level trade-offs (which model types each supports, where compute happens).

## Assets to produce
- Prose: "Data Science/Cloud Environment Setup/in-database-ml.md"
- Artefacts: a side-by-side SQL comparison table (in-prose); a "warehouse-as-model-server" diagram.

## Acceptance criteria
- [ ] AC1 — LOs delivered. AC2 — all SQL is verified/reference (fenced ```sql), NO fabricated result rows; snippet-check passes (no python to compile, or only the diagram script). AC3 — BigQuery ML + Redshift ML syntax grounded with dated citations. AC4 — the "SQL-only deployment" appeal made clear for a SQL-fluent engineer.

## Gates
Entry: approved; grounding (current SQL syntax) landed. Exit: DoD checklist.

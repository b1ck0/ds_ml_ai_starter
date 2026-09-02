# SPEC-AGENT-0: Local Environment Setup (Agentic Engineering)

**Status:** done (written by Sonnet, grounded by Haiku, independently reviewed + merged 2026-09-03)
**Subject:** Agentic Engineering
**Section:** Local Environment Setup
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8
**Prerequisites:** SPEC-DS-0, SPEC-ML-3 (embeddings)
**Nature:** MIXED — Postgres+pgvector, FastAPI, FastMCP run locally; LLM calls need an API key (env).

## Intent
Stand up the agentic toolchain: Python env, a local Postgres with pgvector, FastAPI, FastMCP, and the
Google Agent Development Kit (ADK). Explain the LLM-provider key model and how secrets stay in env.

## Learning objectives
- LO1 — Install the stack (google-adk, fastapi, fastmcp, pgvector client) and verify imports/versions.
- LO2 — Run a local Postgres with the pgvector extension (Docker) and create a vector column.
- LO3 — Configure LLM provider credentials via .env (never committed) and verify a minimal call path (or mark it key-gated).
- LO4 — Understand how these pieces fit: agent runtime (ADK) + tools (MCP) + memory (pgvector) + API (FastAPI).

## Scope
In: installs + version checks; pgvector via Docker; .env secret handling; a component-map diagram.
Out: full app builds (→ AGENT-2..5), cloud deployment (→ AGENT-6).

## Outline
1. What & why — the agent stack vs a normal web backend a Java dev knows.
2. Python deps — google-adk, fastapi, fastmcp, psycopg/pgvector; verify.
3. Postgres + pgvector via Docker; create a table with a vector column; a sanity insert/query.
4. Secrets in env — .env / .env.example; the provider-key model; never commit keys.
5. The component map — how ADK/MCP/pgvector/FastAPI connect.

## Claims to ground (Haiku, before writing)
- [ ] Verify current versions + install of: google-adk (confirm the correct PyPI package name), fastapi, fastmcp, pgvector (Python) / psycopg. Confirm the pgvector Postgres extension Docker image + the CREATE EXTENSION vector step.
- [ ] Verify the minimal ADK "hello agent" API shape from official docs (mark LLM call as key-gated).

## Assets to produce
- Prose: "Agentic Engineering/Local Environment Setup/local-environment-setup.md"
- Code: "Agentic Engineering/Local Environment Setup/code/verify_agentic_env.py"; a docker-compose.yml for pgvector.
- Artefacts: version/import check output (```text); a component-map diagram.

## Acceptance criteria
- [ ] AC1 — LOs delivered. AC2 — the import/version check + the pgvector docker + a vector insert/query RUN locally (no LLM key needed for those); LLM-call steps clearly fenced as key-gated reference; snippet-check passes. AC3 — package names/versions + pgvector setup + ADK API grounded. AC4 — the component map orients a backend dev.

## Gates
Entry: approved; grounding (esp. correct package names) landed. Exit: DoD checklist.

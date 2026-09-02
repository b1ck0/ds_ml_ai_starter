# SPEC-AGENT-2: MCP — a database query layer as a tool server

**Status:** done (written by Sonnet, grounded by Haiku, independently reviewed + merged 2026-09-03)
**Subject:** Agentic Engineering
**Section:** Worked Examples
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8
**Prerequisites:** SPEC-AGENT-0, SPEC-AGENT-1
**Nature:** RUNNABLE — the MCP server + DB run locally; an LLM client call is key-gated (shown, plus a
scripted non-LLM client so the server is testable without a key).

## Intent
Build a real, simple MCP server that exposes a database as a tool: the agent asks for data in natural
terms, the server builds and runs a safe SQL query and returns rows. This is the canonical "give the
LLM a capability" pattern, and a Java dev will recognise it as a typed service boundary.

## Learning objectives
- LO1 — Implement an MCP server with FastMCP exposing tools (e.g. list_tables, query(entity, filters)).
- LO2 — Safely translate a structured tool call into parameterised SQL (no injection) against a local DB.
- LO3 — Test the server with a direct client (no LLM) and then wire an LLM client that calls the tool.
- LO4 — Reason about tool design: clear schemas, least privilege, read-only, validation at the boundary.

## Scope
In: a FastMCP server, 2–3 tools over a seeded SQLite/Postgres DB, parameterised SQL, a test client, an (optional, key-gated) LLM client.
Out: write/mutation tools (read-only here), auth/multi-tenant depth (mention).

## Outline
1. What & why — MCP as a standard tool boundary; the "typed service for the LLM" framing.
2. Seed a small DB; define the schema.
3. Build the FastMCP server + tools with explicit input schemas; parameterised SQL only.
4. Test with a direct client (assert rows) — runnable without any LLM.
5. Wire an LLM client that discovers + calls the tool (key-gated; show the transcript shape).
6. Pitfalls — SQL injection, over-broad tools, unvalidated inputs, leaking the whole DB.

## Assets to produce
- Prose: "Agentic Engineering/Worked Examples/mcp-database-query-layer.md"
- Code: "Agentic Engineering/Worked Examples/code/mcp_db/" (server.py, seed.py, test_client.py, llm_client.py)
- Artefacts: a request→SQL→rows sequence diagram; a captured test-client transcript (real).

## Claims to ground (Haiku, before writing)
- [ ] Verify current FastMCP version + the current server/tool decorator API and how to run + connect a client (from official FastMCP docs). Confirm the MCP client call shape.
- [ ] Confirm the safe parameterised-SQL approach for the chosen driver (sqlite3/psycopg).

## Acceptance criteria
- [ ] AC1 — LOs delivered. AC2 — the server + seed + test_client RUN locally and the test client gets real rows (no LLM key needed); the LLM client is clearly key-gated; snippet-check passes; no fabricated transcripts for the runnable parts. AC3 — FastMCP API + safe-SQL grounded. AC4 — injection prevention shown; tool-design principles stated.

## Gates
Entry: approved; grounding (FastMCP API) landed. Exit: DoD checklist.

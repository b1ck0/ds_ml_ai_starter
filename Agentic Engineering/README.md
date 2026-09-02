# Agentic Engineering

Part 3 of the crash course. See [`../docs/curriculum.md`](../docs/curriculum.md) for the full
backlog. Chapters are written from approved `specs/SPEC-AGENT-*.md`.

## Theory
Vector databases · RAG · MCP (Model Context Protocol) · context window.

## Local Environment Setup
Python · Google Agent Development Kit (ADK) · pgvector · FastAPI · FastMCP.

## Worked Examples
- **MCP** — a simple MCP server as a database query layer: the agent asks for data, the MCP builds
  the query and returns it.
- **RAG** — a simple retrieval-augmented app over PDFs.
- **Invoice Agent** — extracts fields from a PDF invoice and uses the MCP to write them to the DB.
- **Elders Tribunal App** — a multi-agent app where several LLMs debate a topic and report consensus.

## Cloud Environment Setup
GCP / AWS / Azure — the appropriate managed services for deploying agentic applications.

## Production Considerations
Cost/latency of LLM calls, guardrails, evaluation of agent outputs, and data privacy for RAG corpora.

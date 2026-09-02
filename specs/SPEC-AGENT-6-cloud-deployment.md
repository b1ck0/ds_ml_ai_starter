# SPEC-AGENT-6: Cloud — deploying agentic applications (GCP/AWS/Azure)

**Status:** done (written by Sonnet, grounded by Haiku, independently reviewed + merged 2026-09-03)
**Subject:** Agentic Engineering
**Section:** Cloud Environment Setup
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8
**Prerequisites:** SPEC-AGENT-2..5, SPEC-DS-16 (serving)
**Nature:** GROUNDED CONCEPTUAL — verified service mapping + reference IaC/CLI snippets; no cloud
execution, no fabricated output.

## Intent
Take the local agentic apps to production: where the LLM calls, vector store, MCP servers, and API
live in each cloud, and the cross-cutting concerns (secrets, cost, observability, guardrails). Maps
services so a Java dev recognises the equivalent stack per cloud.

## Learning objectives
- LO1 — Map the agentic stack to managed services per cloud: serverless/container runtime, managed vector store, managed Postgres+pgvector, secrets, and the hosted LLM options.
- LO2 — Read a reference deployment (container API + managed vector DB + secret-managed keys) for one cloud.
- LO3 — Reason about cost (token + infra), latency, and observability/tracing for agent apps.
- LO4 — Apply guardrails: secret management, rate limits, prompt-injection defence, human-in-the-loop for side effects.

## Scope
In: cross-cloud service mapping table; one reference deployment sketch; cost/latency/observability; guardrails.
Out: hands-on deploy; deep IaC (link Terraform/official docs).

## Outline
1. What & why — from localhost to production; what changes.
2. Service mapping TABLE — GCP (Cloud Run/Vertex, AlloyDB/Cloud SQL pgvector, Secret Manager, Vertex/hosted LLMs) vs AWS (ECS/Lambda/Bedrock, RDS/Aurora pgvector/OpenSearch, Secrets Manager) vs Azure (Container Apps/Azure OpenAI/Foundry, Postgres flexible pgvector, Key Vault).
3. A reference deployment for one cloud (container API + vector store + secrets).
4. Cost, latency, observability/tracing.
5. Guardrails — secrets, rate limiting, injection defence, human approval for side-effecting tools.
6. Pitfalls — leaking keys, unbounded token cost, no tracing, trusting agent side effects.

## Claims to ground (Haiku, before writing) — VERIFY 2026 service names
- [ ] Verify current service names + the mapping across GCP/AWS/Azure for: container/serverless runtime, hosted LLM (Vertex, Bedrock, Azure OpenAI/Foundry), managed vector store / Postgres+pgvector, secret manager. Confirm against current official docs.
- [ ] Verify the reference deployment snippet's CLI/IaC shape for ONE cloud from official docs (reference only).

## Assets to produce
- Prose: "Agentic Engineering/Cloud Environment Setup/deploying-agentic-apps.md"
- Artefacts: the cross-cloud mapping table (in-prose) + a production-architecture diagram.

## Acceptance criteria
- [ ] AC1 — LOs delivered incl. the mapping table + guardrails. AC2 — code blocks are runnable (diagram) or clearly fenced reference; snippet-check passes; NO fabricated output. AC3 — all service names + the reference snippet grounded with dated citations. AC4 — guardrails (esp. human-in-the-loop for side effects) emphasised for a security-minded engineer.

## Gates
Entry: approved; grounding (2026 service names) landed. Exit: DoD checklist.

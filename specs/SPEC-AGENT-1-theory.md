# SPEC-AGENT-1: Theory — Vector DBs, RAG, MCP, Context Windows

**Status:** done (written by Sonnet, grounded by Haiku, independently reviewed + merged 2026-09-03)
**Subject:** Agentic Engineering
**Section:** Theory
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8
**Prerequisites:** SPEC-ML-3 (embeddings/similarity), SPEC-ML-11 (LLM limits)

## Intent
The conceptual foundation for building agents: why LLMs need external memory (context-window limits,
knowledge cutoff), how vector databases + embeddings enable retrieval (RAG), and what MCP standardises
(tool/data access). Grounds every Agentic worked example.

## Learning objectives
- LO1 — Explain the context window as a finite token budget and its consequences (truncation, cost, "lost in the middle").
- LO2 — Explain embeddings + vector databases: semantic search via nearest-neighbour on vectors (ANN indexes: HNSW/IVF).
- LO3 — Explain RAG end to end: chunk → embed → store → retrieve → augment prompt → generate; and when RAG beats fine-tuning.
- LO4 — Explain MCP (Model Context Protocol): a standard way to expose tools/data to an LLM, and why a standard matters.

## Scope
In: context window, embeddings/vector DB/ANN, RAG pipeline + trade-offs, MCP concept + architecture. Diagrams + a tiny runnable similarity-search demo.
Out: full app builds (→ AGENT-2..5), MCP protocol wire-format depth (link).

## Outline
1. The problem — LLMs are stateless, bounded, and frozen at a cutoff.
2. Context windows — tokens in/out, cost, positional effects.
3. Embeddings + vector DBs — semantic search; ANN indexes; a tiny in-memory retrieval demo.
4. RAG — the pipeline; chunking strategies; RAG vs fine-tuning vs long-context.
5. MCP — standardising tools/data for agents; client/server roles.
6. Pitfalls — bad chunking, stale index, retrieval misses, over-stuffing context.

## Assets to produce
- Prose: "Agentic Engineering/Theory/theory.md"
- Code: "Agentic Engineering/Theory/code/tiny_rag_demo.py" (embed a few docs with a small local embedding model or hand-made vectors, do cosine top-k retrieval — CPU, no LLM key)
- Artefacts: a RAG-pipeline diagram; an MCP client/server diagram; a retrieval-result table.

## Claims to ground (Haiku, before writing)
- [ ] Verify definitions of RAG, ANN indexes (HNSW/IVF), and MCP against authoritative sources (MCP spec, pgvector/FAISS docs).
- [ ] Confirm a small local embedding approach for the demo (sentence-transformers small model, or numpy hand vectors) that runs on CPU without an API key; verify the API.

## Acceptance criteria
- [ ] AC1 — LOs delivered. AC2 — tiny_rag_demo.py runs on CPU (no key) and produces the retrieval table; snippet-check passes. AC3 — RAG/ANN/MCP definitions grounded with citations. AC4 — context-window limits → RAG/MCP motivation clear; tied to ML-11.

## Gates
Entry: approved; notes landed. Exit: DoD checklist.

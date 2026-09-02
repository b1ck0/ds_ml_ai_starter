# SPEC-AGENT-3: RAG over PDFs

**Status:** approved
**Subject:** Agentic Engineering
**Section:** Worked Examples
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8
**Prerequisites:** SPEC-AGENT-1 (RAG theory), SPEC-AGENT-0 (pgvector)
**Nature:** MIXED — ingestion + embedding + retrieval RUN locally (small local embedding model, no key);
the final answer-generation step is key-gated (shown, with a graceful no-key fallback that returns the
retrieved context).

## Intent
Build a working RAG pipeline over PDFs: extract text, chunk, embed, store in a vector DB, retrieve the
top-k for a question, and (optionally) generate a grounded answer. The retrieval half is fully
runnable without any LLM key, so the reader sees RAG's mechanics honestly.

## Learning objectives
- LO1 — Extract and chunk PDF text with sensible overlap; explain chunking trade-offs.
- LO2 — Embed chunks with a local embedding model and store vectors (pgvector or a local index).
- LO3 — Retrieve top-k by cosine similarity for a query and inspect what was retrieved.
- LO4 — Assemble a grounded prompt (context + question) and generate an answer (key-gated), with citations back to chunks.

## Scope
In: PDF parsing, chunking, local embeddings, vector store + retrieval (runnable), prompt assembly, key-gated generation with a no-key fallback.
Out: production ingestion pipelines, re-ranking depth (mention), multi-modal PDFs.

## Outline
1. What & why — RAG recap; the pipeline stages.
2. Parse + chunk a sample PDF; chunk size/overlap choices.
3. Embed (local model) + store (pgvector / local index).
4. Retrieve top-k for a question; show the retrieved chunks + scores (runnable).
5. Generate a grounded answer (key-gated); fallback prints the assembled context; cite chunks.
6. Pitfalls — bad chunking, embedding/model mismatch, retrieval misses, ungrounded generation (hallucination).

## Assets to produce
- Prose: "Agentic Engineering/Worked Examples/rag-over-pdfs.md"
- Code: "Agentic Engineering/Worked Examples/code/rag_pdf/" (ingest.py, retrieve.py, answer.py)
- Datasets: a small public-domain sample PDF (or a generated one) under datasets/.
- Artefacts: a retrieval result table (query → top-k chunks + scores), captured from a real run.

## Claims to ground (Haiku, before writing)
- [ ] Verify a PDF-parsing lib (pypdf / pdfplumber) current version + API, and a small local embedding model (sentence-transformers all-MiniLM-L6-v2 or similar) that runs on CPU + its licence/size + API.
- [ ] Verify how to store + query vectors in pgvector (SQL) OR a local FAISS/numpy fallback; confirm the cosine query.
- [ ] Verify the current provider SDK call for the key-gated generation step (mark clearly as requiring a key).

## Acceptance criteria
- [ ] AC1 — LOs delivered. AC2 — ingest + retrieve RUN locally on a sample PDF and produce the retrieval table (no key needed); answer.py is key-gated with a working no-key fallback; snippet-check passes; no fabricated retrieval output. AC3 — PDF lib + embedding model + vector-store + provider APIs grounded. AC4 — chunking/grounding pitfalls made concrete; ties to AGENT-1.

## Gates
Entry: approved; grounding landed. Exit: DoD checklist.

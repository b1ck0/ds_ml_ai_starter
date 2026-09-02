# SPEC-AGENT-4: Invoice Agent — PDF in, structured rows out via MCP

**Status:** done (written by Sonnet, grounded by Haiku, independently reviewed + merged 2026-09-03)
**Subject:** Agentic Engineering
**Section:** Worked Examples
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8
**Prerequisites:** SPEC-AGENT-2 (MCP), SPEC-AGENT-3 (PDF parsing)
**Nature:** MIXED — PDF parsing, the MCP write-tool, and the DB all RUN locally; the LLM extraction step
is key-gated with a deterministic rule-based fallback extractor so the whole flow is testable end-to-end
without a key.

## Intent
A capstone that composes the earlier pieces: give the agent a PDF invoice; it extracts the fields
(vendor, number, date, line items, total) and calls an MCP tool to persist them to the database. Shows
the full "unstructured input → agent → tool → structured store" loop a backend dev can appreciate.

## Learning objectives
- LO1 — Define an extraction schema (typed fields) for an invoice and validate against it.
- LO2 — Extract fields from a PDF: an LLM path (key-gated) AND a deterministic fallback so it's testable.
- LO3 — Persist via an MCP write-tool (parameterised INSERT) — reusing/extending the AGENT-2 server with a write capability + validation.
- LO4 — Handle failures: missing fields, malformed PDFs, idempotency (don't double-insert).

## Scope
In: invoice schema + validation (pydantic), PDF field extraction (LLM + fallback), an MCP insert tool, persistence, error handling, idempotency.
Out: OCR of scanned images (mention), a full accounts-payable workflow.

## Outline
1. What & why — the unstructured→structured loop; where agents add value over regex.
2. The schema — pydantic model; validation rules.
3. Extract — LLM extraction (key-gated) + a deterministic fallback on a known sample invoice.
4. Persist via MCP — a validated, parameterised insert tool; confirm the row landed.
5. Robustness — missing fields, bad PDF, idempotency key.
6. Pitfalls — trusting unvalidated LLM output, injection via extracted text, double-inserts.

## Assets to produce
- Prose: "Agentic Engineering/Worked Examples/invoice-agent.md"
- Code: "Agentic Engineering/Worked Examples/code/invoice_agent/" (schema.py, extract.py, mcp_write_tool.py, run.py)
- Datasets: a generated sample invoice PDF under datasets/.
- Artefacts: a captured end-to-end run transcript (PDF → fields → DB row) using the fallback extractor (real).

## Claims to ground (Haiku, before writing)
- [ ] Verify pydantic (v2) current version + API for the schema/validation. Reuse AGENT-2/3 notes for FastMCP + PDF parsing.
- [ ] Verify the provider SDK call for structured extraction (tool/JSON mode) — mark key-gated; confirm the fallback approach (regex/rules on a known template) is sound.

## Acceptance criteria
- [ ] AC1 — LOs delivered. AC2 — the full flow RUNS end-to-end with the deterministic fallback (parse → validate → MCP insert → verify row), no key needed; the LLM path is key-gated; snippet-check passes; the transcript is from a real run. AC3 — pydantic + MCP + extraction APIs grounded. AC4 — validation/injection/idempotency handled; composition of AGENT-2+3 clear.

## Gates
Entry: approved; grounding landed. Exit: DoD checklist.

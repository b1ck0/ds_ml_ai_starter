# SPEC-AGENT-5: Elders Tribunal — a multi-agent debate that reports consensus

**Status:** done (written by Sonnet, grounded by Haiku, independently reviewed + merged 2026-09-03)
**Subject:** Agentic Engineering
**Section:** Worked Examples
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8
**Prerequisites:** SPEC-AGENT-1, SPEC-ML-11
**Nature:** MIXED — the orchestration engine (turn-taking, rounds, aggregation) RUNS locally and is unit-
tested with SCRIPTED/mock "elder" responses; wiring real LLM providers is key-gated. This keeps the
multi-agent LOGIC fully testable without keys.

## Intent
A multi-agent application: several "elder" agents — ideally backed by different LLMs — debate a topic
over rounds, then a moderator agent synthesises a consensus and reports it. Teaches multi-agent
orchestration patterns (roles, rounds, aggregation) with a testable core.

## Learning objectives
- LO1 — Design a multi-agent protocol: roles (elders + moderator), rounds, turn-taking, termination.
- LO2 — Implement the orchestration engine so it's provider-agnostic (an Elder interface with a `respond()` method) and unit-testable with fakes.
- LO3 — Wire real, DIFFERENT LLM providers behind that interface (key-gated) and run a live debate.
- LO4 — Aggregate to a consensus (voting / moderator synthesis) and report it; reason about disagreement.

## Scope
In: the role/round protocol, a provider-agnostic Elder interface, a runnable mock-backed debate + tests, key-gated real-provider wiring, consensus aggregation.
Out: elaborate agent frameworks (keep it minimal/legible), long-running/async orchestration depth.

## Outline
1. What & why — why multiple models/roles beat one; the "panel of experts" framing.
2. The protocol — roles, rounds, turn order, stop condition.
3. The engine — an Elder interface + orchestrator; unit tests with scripted elders (runnable, no key).
4. Real providers — wire ≥2 different LLMs behind the interface (key-gated); a live-debate example (reference).
5. Consensus — voting vs moderator synthesis; report format; handling deadlock.
6. Pitfalls — cost blowup, echo chambers, prompt injection between agents, non-determinism.

## Assets to produce
- Prose: "Agentic Engineering/Worked Examples/elders-tribunal.md"
- Code: "Agentic Engineering/Worked Examples/code/elders_tribunal/" (elder.py interface, orchestrator.py, providers.py [key-gated], test_orchestrator.py, run.py)
- Artefacts: a debate-flow diagram; a captured mock-debate transcript + consensus (real, from the fake-backed run).

## Claims to ground (Haiku, before writing)
- [ ] Verify current provider SDKs for ≥2 different LLMs (e.g. anthropic, openai, google-genai) — package names, current versions, and the minimal chat-call API — all clearly marked key-gated. Prefer the latest Claude models for the Anthropic elder.
- [ ] Confirm whether Google ADK offers a native multi-agent primitive worth using vs a hand-rolled orchestrator; recommend the clearer teaching path.

## Acceptance criteria
- [ ] AC1 — LOs delivered. AC2 — the orchestrator + tests RUN with scripted elders and produce a real consensus transcript (no key); real-provider wiring is key-gated and clearly separated; snippet-check passes; no fabricated "live" transcripts presented as real. AC3 — provider SDKs + ADK option grounded. AC4 — the multi-agent protocol is legible and testable; disagreement/consensus handled thoughtfully.

## Gates
Entry: approved; grounding landed. Exit: DoD checklist.

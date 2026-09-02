# SPEC-SDLC-1: Theory — Prompts, Rules, Hooks, Gates, Tools, Sub-agents, Skills

**Status:** approved
**Subject:** AI-assisted-sdlc
**Section:** Theory
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8
**Prerequisites:** SPEC-SDLC-0

## Intent
Define the primitives of agentic software development so the reader can reason about them like build/CI
concepts they already know. Each primitive is defined, motivated, and — crucially — pointed at where
THIS repository uses it, so the theory has a live referent.

## Learning objectives
- LO1 — Define prompts and rules (persistent instructions, e.g. CLAUDE.md) and how they steer an agent — like coding standards + a linter config.
- LO2 — Define hooks and gates: deterministic automation around agent actions (pre/post) and pass/fail quality gates — like git hooks + CI gates.
- LO3 — Define tools and MCP: the capabilities an agent can call — like injected service dependencies with typed contracts.
- LO4 — Define sub-agents and skills: delegating scoped work to specialised agents, and packaged procedures — like microservices + runbooks.

## Scope
In: crisp definition + motivation + a "you already know this as X" analogy + a pointer to this repo's use, for each primitive.
Out: vendor-specific config exhaustiveness (link official docs), building each (→ SDLC-2).

## Outline
1. The mental model — an agent as a very capable junior engineer that needs SOPs, tools, and guardrails.
2. Prompts & rules — CLAUDE.md, persistent standards. (This repo: CLAUDE.md, docs/style-guide.md.)
3. Hooks & gates — deterministic automation + Definition of Done. (This repo: .claude/hooks/*, docs/definition-of-done.md.)
4. Tools & MCP — typed capabilities. (This repo: the Agentic subject's MCP servers.)
5. Sub-agents & skills — delegation + packaged procedures. (This repo: .claude/agents/*, .claude/skills/*.)
6. Putting it together — how they compose into a governed workflow (this repo is the worked example).

## Claims to ground (Haiku, before writing)
- [ ] Verify the current definitions/mechanics of Claude Code hooks, sub-agents, skills, MCP, and settings from official docs (2026) — names and behaviour, not invented.

## Assets to produce
- Prose: "AI-assisted-sdlc/Theory/theory.md"
- Artefacts: a diagram of how prompts/rules/hooks/gates/tools/sub-agents/skills compose into a governed SDLC loop (SVG/matplotlib).

## Acceptance criteria
- [ ] AC1 — each primitive defined with a Java/CI analogy + a pointer to this repo's concrete use. AC2 — no fabricated mechanics; each claim grounded in official docs; snippet-check passes (diagram script or none). AC3 — the primitives + their mechanics grounded with dated citations. AC4 — the reader can map every primitive to something they already do in Java/CI.

## Gates
Entry: approved; grounding landed. Exit: DoD checklist.

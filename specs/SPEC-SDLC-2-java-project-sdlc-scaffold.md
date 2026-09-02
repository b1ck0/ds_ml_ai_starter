# SPEC-SDLC-2: Scaffolding a governed SDLC for a new Java project

**Status:** done (written by Sonnet, grounded by Haiku, independently reviewed + merged 2026-09-03)
**Subject:** AI-assisted-sdlc
**Section:** Worked Examples
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8
**Prerequisites:** SPEC-SDLC-0, SPEC-SDLC-1
**Nature:** RUNNABLE where possible — the Java project builds/tests; the .claude/ scaffold files are real
and validated (JSON/markdown well-formed); agent runs are shown as reference transcripts.

## Intent
The capstone of the whole book: create a new Java project and set up the SDLC documents + agent roster
so agents follow a Standard Operating Procedure — a researcher subagent, a QA subagent, an implementer
subagent, and the architect. THIS repository's own `.claude/` scaffold is the reference implementation;
the chapter teaches the reader to reproduce it for a Java stack.

## Learning objectives
- LO1 — Author a CLAUDE.md architect charter + docs (architecture, definition-of-done) for a Java project with real gates (mvn/gradle build, test, checkstyle/spotless).
- LO2 — Define the agent roster: researcher (Haiku), implementer (Sonnet), QA/reviewer (fresh Sonnet), architect (Opus) — mirroring this repo's `.claude/agents/`.
- LO3 — Add hooks (verify on edit runs the Java gate; guard blocks dangerous shell) and a spec/DoD workflow (tests-first).
- LO4 — Run the loop on a small feature: spec → (research) → failing test → implement → gate → review → merge.

## Scope
In: a real .claude/ scaffold for a Java project (CLAUDE.md, docs, agents, hooks, settings), the Java gate commands, a walked-through feature loop (reference transcript), and a comparison to this repo's scaffold.
Out: language-agnostic exhaustiveness; CI wiring depth (mention + link to cloud section).

## Outline
1. What & why — agents need SOPs; the governed loop recap (SDLC-1).
2. The charter + docs — CLAUDE.md, architecture.md, definition-of-done.md with Java gates.
3. The agent roster — researcher/implementer/reviewer/architect files (adapted for Java).
4. Hooks + settings — verify.sh runs `mvn -q test` / checkstyle on edit; guard.sh; context.sh.
5. The loop in action — a small feature (e.g. a validator) taken spec→tests→implement→gate→review→merge (reference transcript).
6. Pitfalls — no failing test first, weakening tests, unbounded agent scope, skipping review.
7. Reference — how this differs from THIS repo's content-authoring scaffold, and what carries over.

## Claims to ground (Haiku, before writing)
- [ ] Verify current Java quality-gate tools + commands (JUnit 5, Checkstyle/Spotless/Spotbugs, `mvn test` / `gradle test`) and versions (2026).
- [ ] Verify the current Claude Code file conventions for CLAUDE.md, .claude/agents/*.md (frontmatter: name/description/model), .claude/skills, .claude/hooks, settings.json from official docs — the scaffold must match real, current schemas.

## Assets to produce
- Prose: "AI-assisted-sdlc/Worked Examples/java-sdlc-scaffold.md"
- Code: "AI-assisted-sdlc/Worked Examples/code/java-project/" (a small Maven/Gradle project + a real .claude/ scaffold: CLAUDE.md, docs/, .claude/agents/{researcher,implementer,reviewer}.md, .claude/hooks/{verify,guard,context}.sh, .claude/settings.json)
- Artefacts: a governed-loop diagram; a reference feature-loop transcript; validated JSON/frontmatter.

## Acceptance criteria
- [ ] AC1 — LOs delivered; the scaffold files are REAL and well-formed (settings.json parses; agent frontmatter valid). AC2 — the Java project builds/tests if a JDK is present (captured), else the gate commands are verified reference; the .claude/ files validate; snippet-check passes; agent-run transcripts clearly marked as reference. AC3 — Java gate tools + Claude Code file schemas grounded with dated citations (the schema MUST match current docs). AC4 — the reader can reproduce a governed SDLC on their own Java project; explicit tie-back to this repository as the live example.

## Gates
Entry: approved; grounding (esp. current Claude Code file schemas) landed. Exit: DoD checklist.

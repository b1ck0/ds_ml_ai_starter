# SPEC-SDLC-0: Local Environment Setup (AI-assisted SDLC)

**Status:** done (written by Sonnet, grounded by Haiku, independently reviewed + merged 2026-09-03)
**Subject:** AI-assisted-sdlc
**Section:** Local Environment Setup
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8
**Prerequisites:** none (the reader's home turf: Java)

## Intent
Set up the AI-assisted development toolchain on ground the reader already owns — a Java project — plus
Claude Code. Establish the workflow this whole repository already exemplifies.

## Learning objectives
- LO1 — Install a modern JDK + Maven/Gradle and scaffold a runnable Java project (the reader's comfort zone).
- LO2 — Install and authenticate Claude Code; understand what it is (an agentic CLI in the terminal/IDE).
- LO3 — Point Claude Code at the Java project and run a first task; understand the permission model.
- LO4 — Understand the artefacts that steer an agent (CLAUDE.md, .claude/ skills/agents/hooks) — previewing SDLC-2.

## Scope
In: JDK + build tool + a hello Java project; Claude Code install/auth/first-run; the permission model; the steering-file map.
Out: deep agent authoring (→ SDLC-2), CI integration (→ cloud section).

## Outline
1. What & why — AI-assisted SDLC on familiar Java ground.
2. JDK + Maven/Gradle + a starter project (verify it builds/tests).
3. Install + authenticate Claude Code; the terminal/IDE surfaces.
4. First task on the project; the permission prompts and why they exist.
5. The steering files — CLAUDE.md, .claude/ — a map (this repo is the reference example).

## Claims to ground (Haiku, before writing)
- [ ] Verify the current recommended JDK LTS version (2026) + a current Maven or Gradle version and the `mvn`/`gradle` starter command.
- [ ] Verify the current Claude Code install method + auth from official docs (do NOT invent commands); confirm supported surfaces (CLI, IDE, desktop).

## Assets to produce
- Prose: "AI-assisted-sdlc/Local Environment Setup/local-environment-setup.md"
- Code: "AI-assisted-sdlc/Local Environment Setup/code/hello-java/" (a minimal Maven/Gradle project + one test).
- Artefacts: a captured `mvn test`/`gradle test` output (```text); a steering-file map diagram.

## Acceptance criteria
- [ ] AC1 — LOs delivered. AC2 — the hello-java project BUILDS + TESTS (real output captured) if a JDK is available in-sandbox; if not, the build/test commands are given as verified reference and the writer notes the sandbox limitation honestly. snippet-check passes (java blocks fenced appropriately; no python to break). AC3 — JDK/build-tool/Claude-Code install steps grounded with dated citations. AC4 — leverages the reader's Java fluency; the steering-file map orients them.

## Gates
Entry: approved; grounding landed. Exit: DoD checklist.

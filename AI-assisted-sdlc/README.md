# AI-assisted SDLC

Part 4 of the crash course. See [`../docs/curriculum.md`](../docs/curriculum.md) for the full
backlog. Chapters are written from approved `specs/SPEC-SDLC-*.md`.

The best reference example for this subject is **this repository itself**: the `.claude/` scaffold —
architect charter (`CLAUDE.md`), model routing, chapter specs, gates, hooks, and the researcher /
writer / reviewer subagents — is exactly the kind of SOP this subject teaches. Point the reader here.

## Theory
Prompts · hooks · rules · gates · sub-agents · tools · skills.

## Local Environment Setup
Java · Claude Code.

## Worked Examples
- Create a new Java project and set up all the SDLC documents so agents follow the SOP: a researcher
  subagent, a QA subagent, an implementer subagent, and the architect — mirroring this repo's own
  `.claude/agents/` and `docs/`.

## Cloud Environment Setup
CI integration and running agents in a pipeline (cross-reference the team's CI/CD conventions).

## Production Considerations
Guardrails, review gates, and keeping humans in the loop for irreversible actions.

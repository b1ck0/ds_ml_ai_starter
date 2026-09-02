---
name: chapter-scoper
description: Turn one curriculum backlog item into a rigorous specs/SPEC-*.md chapter spec before any writing begins. Use whenever a new chapter is requested and no approved spec exists. Produces learning objectives, scope, an outline, the assets to produce, the claims that must be grounded first, and testable acceptance criteria.
---

# chapter-scoper

Write a `specs/SPEC-<SUBJECT>-<n>-<slug>.md` that the **chapter writer (Sonnet)** can build against
with zero ambiguity. The architect (Opus) approves it with the owner before writing begins.

## Rules
- One spec = one chapter (one teachable unit). If it needs more than a long chapter, split it and
  sequence the parts.
- **Learning objectives** are the spine: each is observable ("the reader can …") and each maps to at
  least one acceptance criterion.
- **Name the claims to ground.** Any external fact the chapter will assert — a current package
  version, a dataset URL + licence, a library's real API, a metric's exact formula, a "best
  dataset/framework" judgement — goes in a "Claims to ground" list that MUST be researched by the
  Haiku researcher (`.claude/skills/research-brief`) and land as `research/NOTE-*.md` BEFORE writing.
- **Name the runnable assets.** List the exact prose/code/dataset/artefact paths, so the writer knows
  what to produce and the reviewer knows what to check. Every code artefact must be runnable; every
  worked example must end in a visible artefact.
- Pin the audience: everything is for a senior Java dev new to Python/ML (see `docs/style-guide.md`).
- State what is **out of scope** and link the later chapter that covers it.
- Reuse the structure in `specs/SPEC-TEMPLATE.md`.

## Definition of a good chapter spec
It states WHAT the reader will learn and WHY it matters to them, lists concrete learning objectives
each tied to an acceptance criterion, enumerates the exact assets to produce, calls out every claim
that must be grounded first, and names the exit gate. The writer should never have to guess or make a
pedagogy decision.

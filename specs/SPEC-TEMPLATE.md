# SPEC-<SUBJECT>-<n>: <chapter title>

**Status:** draft | approved | in-progress | in-review | done
**Subject:** Data Science | Machine Learning | Agentic Engineering | AI-assisted-sdlc
**Section:** Theory | Local Environment Setup | Worked Examples | Cloud Environment Setup | Production Considerations
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8
**Prerequisites:** <chapters the reader should have read first, by SPEC id>

## Intent
<one paragraph: what this chapter teaches, and where a senior Java dev meets this problem>

## Learning objectives
After this chapter the reader can:
- LO1 …
- LO2 …

## Scope
In scope: …
Out of scope: … (things a reader might expect but we deliberately defer — link the later chapter)

## Outline (section-by-section)
1. What & why …
2. Concept (grounded) …
3. Worked example: <dataset> → <code> → <artefact> → interpretation …
4. Pitfalls …
5. Recap & next …

## Assets to produce
- Prose: `<Subject>/<section>/<slug>.md`
- Code: `<Subject>/<section>/code/<slug>*.py` (runnable, deps declared)
- Dataset: `<Subject>/<section>/datasets/…` (or a documented download step + licence)
- Artefacts: `<Subject>/<section>/artefacts/…` (plots/tables the code reproduces)

## Claims to ground (Haiku research brief — do BEFORE writing)
Delegate to `.claude/skills/research-brief` → dispatch `.claude/agents/researcher.md`. Each item
becomes a `research/NOTE-*.md` with a verified answer + source URL + date:
- [ ] Package versions to pin: <libs>
- [ ] Dataset URL + licence: <dataset>
- [ ] Metric/API definitions to verify: <e.g. exact formula of X; does library Y expose method Z>
- [ ] Any "best dataset / best framework" claim to confirm or correct

## Acceptance criteria (each maps to evidence)
- [ ] AC1 — <objective delivered> → evidence: <snippet id / NOTE id / artefact>
- [ ] AC2 — every snippet in the chapter runs → evidence: verify.sh compile pass + manual run log
- [ ] AC3 — every version/dataset/claim grounded → evidence: NOTE ids listed
- [ ] AC4 — audience-fit (Java-dev framing, jargon explained, artefact shown)

## Gates
Entry: this spec approved by the owner; the research NOTEs for all "claims to ground" have landed.
Exit: all ACs satisfied with evidence; snippets run; links resolve; independent fresh-Sonnet review
sign-off; architect merge approval. (See `docs/definition-of-done.md`.)

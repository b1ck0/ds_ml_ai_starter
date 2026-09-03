# SPEC-SDLC-3: How this repo was built — a governed multi-agent SDLC, dissected

**Status:** in-review (written by Sonnet; self-grounded against this repo's own files, `git log`, and
two light web checks — the MathJax `\text{}`-escaping error family and GitHub commit-permalink
resolution, both confirmed live; `check_snippets.py` and `check_markdown_render.py` both pass; pending
independent fresh-Sonnet review + architect merge)
**Subject:** AI-assisted-sdlc
**Section:** Worked Examples
**Routing:** writer=Sonnet 4.6 · research=Haiku (light) · review=Sonnet (fresh) · architect=Opus 4.8
**Prerequisites:** SPEC-SDLC-1 (Theory — prompts, rules, hooks, gates, tools, sub-agents, skills),
SPEC-SDLC-2 (Scaffolding a governed SDLC for a Java project).

## Intent
SDLC-2 shows the reader how to scaffold a governed AI-assisted pipeline for a *toy* Java project.
This chapter turns the camera on the book itself: **this repository is a real, non-trivial artefact
built by exactly such a pipeline**, and its whole history is inspectable. The reader has just read
57 chapters; now they see the factory that produced them. The teaching move is credibility —
"the thing you are reading was built the way we told you to build things" — and concreteness: every
abstract role from SDLC-1 (architect, writer, reviewer, researcher; rules, hooks, gates, specs) maps
to a real file and a real commit in this repo. A senior engineer evaluating whether to adopt this way
of working gets an honest, warts-and-all case study rather than a marketing diagram.

## Learning objectives
After this chapter the reader can:
- LO1 — Describe the multi-model division of labour used here (Opus 4.8 scopes/reviews/merges;
  Sonnet 4.6 writes one chapter at a time; Haiku grounds claims by web research; a *fresh* Sonnet
  reviews before merge) and the reason each task is routed to the model it is.
- LO2 — Trace one chapter from intent to merge through the real artefacts: `SPEC-*.md` →
  `research/NOTE-*.md` → chapter prose/code/artefacts → gate output → review verdict → commit.
- LO3 — Explain the governance layer that made this safe to run largely unattended: `CLAUDE.md`
  golden rules, the `.claude/hooks/` gates (`verify.sh` snippet-compile, `check_snippets.py`,
  `check_markdown_render.py` GitHub-render lint, `guard.sh` dangerous-shell block), and the
  Definition of Done — including *why each gate exists* (a real failure it caught).
- LO4 — Read the honest limitations: where the pipeline needed human escalation, a gate false
  positive (e.g. `guard.sh` flagging `echo … token…`), a stale-render bug that motivated the render
  lint, and the concurrency/rate-limit lesson from running many agents at once.

## Scope
In scope: a documented case study of *this* repo's actual structure, roles, artefacts, gates, and a
few real incidents, all pointing at real committed files and using real (lightly trimmed) transcripts/
outputs. A Mermaid diagram of the pipeline. This chapter is **self-referential and self-grounding** —
the evidence is the repository the reader already has checked out.
Out of scope: re-teaching SDLC-1 theory (link it); a step-by-step "do this yourself" tutorial (that
is SDLC-2); any claim about model internals or benchmarks. Do NOT invent metrics ("N% faster") —
only state what the repo's own history supports.

## Outline (section-by-section)
1. **Cold open** — the reader has finished the book; reveal that it was written by the pipeline it
   describes, and that they can `git log` every claim in this chapter.
2. **The org chart** — the four roles and the model routing, with the Mermaid pipeline diagram
   (intent → spec → grounding → draft → gate → review → merge), each box linked to the real file
   (`CLAUDE.md`, `.claude/agents/*.md`, `.claude/skills/*`).
3. **One chapter, end to end** — pick one real merged chapter and walk its actual trail: its
   `specs/SPEC-*.md`, the `research/NOTE-*.md` it cites, the chapter file, its artefacts, and the
   commit. Show the real spec→evidence mapping.
4. **The guardrails** — the gates as real files: what each checks, and for `check_markdown_render.py`
   the *real* bug class it was created to catch (unescaped `_` in a `\text{…}` run dumping raw TeX on
   GitHub — MathJax's "'_' allowed only in math mode"). Quote the DoD checklist.
5. **Where it needed a human / where it broke** — honest incidents: escalations, the `guard.sh`
   false positive, the render-lint origin, the rate-limit/concurrency lesson. This section is what
   makes the chapter trustworthy.
6. **Recap & next** — when this pattern is worth it, when it's overkill, and the transfer back to a
   product codebase (SDLC-2).

## Assets to produce
- Prose: `04-ai-assisted-sdlc/03-worked-examples/02-how-this-repo-was-built.md`
- Artefacts: a lightly-trimmed real spec→NOTE→commit trail excerpt and a real gate-output excerpt,
  under `04-ai-assisted-sdlc/03-worked-examples/artefacts/` (reuse existing transcript artefacts where
  they fit). No new runnable code is required; any shell shown (`git log …`, running a hook) must be
  real and copy-pasteable against this repo.

## Claims to ground (Haiku research brief — light; do BEFORE writing)
Most grounding is *internal* — verify against the repo's own files and `git log`, NOT the web. The
writer MUST confirm every file path, agent name, hook name, and rule quoted actually exists at HEAD
(paths drift). Web grounding only for:
- [ ] Any external claim about a model's intended role/strengths — keep to what is publicly documented,
      cite it, or cut it. Do not assert relative model capabilities from memory.
- [ ] The MathJax error-string ("'_' allowed only in math mode") — confirm it is the real message.

## Acceptance criteria (each maps to evidence)
- [ ] AC1 (LO1–LO4) — roles, one end-to-end trail, the gates, and the honest-limitations section all
      present, each pointing at a real file/commit → evidence: the links resolve to HEAD.
- [ ] AC2 — every path/file/hook/agent named exists at HEAD → evidence: a verification pass listing them.
- [ ] AC3 — no invented metric or ungrounded capability claim → evidence: reviewer confirms.
- [ ] AC4 — audience-fit; honest about limits, not a sales pitch.
- [ ] AC5 — renders on GitHub → `check_markdown_render.py` pass; Mermaid diagram eyeballed.

## Gates
Entry: this spec approved. Exit: all ACs satisfied; every internal link resolves at HEAD; fresh-Sonnet
review sign-off; architect merge. (See `docs/definition-of-done.md`.)

# ds_ml_ai_starter — Authoring Architecture & SDLC (living document)

Single source of truth for HOW this textbook gets built: the workflow, the roles, the content
quality gates, and the repository shape. Keep it current — update it in the same PR that changes the
thing it describes.

## 1. Operating model

Spec-driven, grounded, multi-model **content authoring**. The owner states intent; the
**architect (Opus 4.8)** turns it into an approved **chapter spec** with acceptance criteria; the
**Haiku researcher** grounds every external fact first; **Sonnet 4.6** writes exactly that chapter; a
**fresh Sonnet** reviews for correctness, grounding, and audience-fit; the architect merges. Nothing
merges until its exit gate (§5) passes.

This mirrors a test-first engineering workflow, but the "tests" are **verifiable content criteria**:
a snippet that runs, a link that resolves, a claim that traces to a source, a learning objective that
the prose actually delivers.

## 2. Roles & model routing

| Role | Model | Responsibility |
|---|---|---|
| Architect | Opus 4.8 (main session) | Scope chapters, sequence, gate decisions, review, merge, escalation |
| Writer | Sonnet 4.6 (`.claude/agents/chapter-writer.md`) | Write ONE approved chapter: prose + runnable code + artefacts |
| Researcher | Haiku (`.claude/agents/researcher.md`) | Search the web to ground claims / versions / dataset links → `research/NOTE-*.md` |
| Reviewer | Sonnet, fresh (`.claude/agents/chapter-reviewer.md`) | Independent QA against the spec + Definition of Done before merge |

## 3. Workflow (per chapter)

1. **Intent → Chapter spec.** Architect writes `specs/SPEC-<subject>-<n>-<slug>.md` using
   `.claude/skills/chapter-scoper`, drawing the backlog item from `docs/curriculum.md`. Owner/architect
   approves. No writing before approval.
2. **Ground the unknowns.** For every external fact the chapter will assert — a current package
   version, a dataset URL and licence, a library's real API, a metric's exact definition — the
   architect writes a research brief (`.claude/skills/research-brief`) and dispatches the **Haiku
   researcher**, which returns `research/NOTE-*.md`. No prose depends on an unverified fact.
3. **Write.** Dispatch **Sonnet** against the one approved spec. It writes the chapter markdown, the
   runnable `code/` snippets, generates the `artefacts/` (plots, tables), and cites the NOTEs.
4. **Gate.** Run the full content gate (§5). Snippets execute; links resolve; claims are grounded.
5. **Review.** Dispatch a **fresh Sonnet** reviewer for independent QA against the ACs and the
   Definition of Done.
6. **Merge.** Architect merges; PR body maps every AC → its evidence.

## 4. Repository shape

- `Data Science/`, `Machine Learning/`, `Agentic Engineering/`, `AI-assisted-sdlc/` — the four
  subjects. Each is organised under five sections: **Theory**, **Local Environment Setup**,
  **Worked Examples**, **Cloud Environment Setup**, **Production Considerations**. Chapter prose lives
  in `.md`; runnable code in `code/`; data in `datasets/`; generated outputs in `artefacts/`.
- `specs/` — one `SPEC-*.md` per chapter (+ `SPEC-TEMPLATE.md`).
- `research/` — `NOTE-*.md` grounding outputs (with the source URLs and dates verified).
- `docs/` — this file + `curriculum.md` (full backlog) + `definition-of-done.md` + `style-guide.md`.
- `.claude/` — `skills/` (chapter-scoper, research-brief), `agents/` (chapter-writer, researcher,
  chapter-reviewer), `hooks/` (verify, guard, context), `settings.json`.

## 5. Gates (Definition of Done)

Every chapter must clear this before it is "done":

- **Fidelity:** covers every learning objective in its approved spec; nothing silently out of scope.
- **Grounded:** every technical claim, package version, dataset link, API name, and metric definition
  traces to a `research/NOTE-*.md` or an inline authoritative citation. Versions and URLs verified live.
- **Runnable:** every code snippet executes as written against the declared environment (or is fenced
  as pseudocode). Artefacts (plots/tables) reproduce from the code.
- **Links resolve:** every external link (datasets, docs) is reachable.
- **Audience-fit:** written for an experienced Java dev new to Python/ML — analogies where they help,
  no unexplained jargon, artefacts shown.
- **Reviewed:** independent fresh-Sonnet review sign-off + architect merge approval.

Full checklist: `docs/definition-of-done.md`. Fast checks automated on edit by `.claude/hooks/verify.sh`.

## 6. Decision log

Non-obvious decisions, newest first — so future work doesn't re-litigate them.

- 2026-09-02 — Adapted the code-oriented SDLC scaffold to content authoring: "specs" → chapter specs,
  "tests-first" → verifiable content criteria (runnable snippets, resolving links, grounded claims).
  Model routing kept verbatim: Opus scopes, Sonnet writes, Haiku grounds, fresh Sonnet reviews.

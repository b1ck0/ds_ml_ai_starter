# java-project — Governed SDLC (living document)

Single source of truth for HOW this codebase gets built: the workflow, the roles, the quality
gates, and the repository shape. Keep it current — update it in the same PR that changes the thing
it describes. This file, and the rest of this project's `.claude/` scaffold, is a direct port of
`ds_ml_ai_starter`'s own content-authoring scaffold (see that repo's `docs/architecture.md`) onto a
Java/Maven stack — the mapping is spelled out in
`AI-assisted-sdlc/Worked Examples/java-sdlc-scaffold.md` §7 of the textbook this project ships as a
worked example for.

## 1. Operating model

Spec-driven, grounded, test-first engineering. The owner states a feature request; the **architect
(Opus)** turns it into an approved feature spec with acceptance criteria; the **researcher (Haiku)**
grounds any external unknown first; **Sonnet** implements — a failing test before any production
code — until every gate is green; a **fresh Sonnet** reviews for correctness, test-first discipline,
and scope; the architect merges. Nothing merges until its exit gate (§5) passes.

## 2. Roles & model routing

| Role | Model | Responsibility |
|---|---|---|
| Architect | Opus (main session) | Scope features, sequence, gate decisions, review, merge, escalation |
| Implementer | Sonnet (`.claude/agents/implementer.md`) | Implement ONE approved feature: failing test → production code → green gate |
| Researcher | Haiku (`.claude/agents/researcher.md`) | Ground external claims (dependency versions, API behaviour, CVEs) → a note the implementer cites |
| Reviewer | Sonnet, fresh (`.claude/agents/reviewer.md`) | Independent QA against the spec + Definition of Done before merge |

## 3. Workflow (per feature)

1. **Request → feature spec.** Architect writes `docs/features/FEATURE-<n>-<slug>.md`: intent,
   acceptance criteria, and any claims that need grounding. Owner/architect approves. No code before
   approval.
2. **Ground the unknowns (if any).** For every external fact the feature will rely on — a
   dependency's pinned version, a library's documented behaviour, a known-CVE check — the architect
   dispatches the **researcher**, which returns a grounding note. Skipped when the feature needs no
   external fact (e.g. a self-contained algorithm already fully specified in the spec).
3. **Failing test first.** Dispatch the **implementer**. It writes a JUnit 5 test that encodes the
   acceptance criteria, confirms it fails for the *right* reason (the behaviour doesn't exist yet,
   not a typo), then writes the minimum production code to make it pass.
4. **Gate.** `mvn clean test`, `mvn checkstyle:check`, `mvn spotless:check`,
   `mvn spotbugs:check` all pass. `.claude/hooks/verify.sh` runs the fast subset automatically on
   every edit.
5. **Review.** Dispatch a **fresh reviewer** for independent QA against the acceptance criteria, the
   Definition of Done, and — specifically — the commit order (test before code).
6. **Merge.** Architect merges; the PR body maps every acceptance criterion to its evidence.

## 4. Repository shape

- `src/main/java`, `src/test/java` — standard Maven layout; production code and JUnit 5 tests.
- `docs/features/` — one `FEATURE-<n>-<slug>.md` per feature.
- `docs/` — this file + `definition-of-done.md`.
- `.claude/` — `agents/` (researcher, implementer, reviewer), `hooks/` (verify, guard, context),
  `settings.json`.
- `pom.xml` — dependencies and the four gate plugins (Surefire, Checkstyle, Spotless, SpotBugs).
- `checkstyle.xml` — the style ruleset `checkstyle:check` enforces.

## 5. Gates (Definition of Done)

Every feature must clear this before it is "done":

- **Fidelity:** every acceptance criterion in the approved feature spec is met.
- **Test-first:** a JUnit 5 test existed, and failed for the right reason, before the production
  code that satisfies it.
- **Grounded:** any dependency version, library-behaviour claim, or CVE check traces to a
  researcher's note or a live, dated citation.
- **Green gate:** `mvn clean test` passes; `checkstyle:check`, `spotless:check`, `spotbugs:check`
  all pass clean.
- **Reviewed:** independent fresh-Sonnet review sign-off + architect merge approval.

Full checklist: `docs/definition-of-done.md`. Fast checks automated on edit by
`.claude/hooks/verify.sh`.

## 6. Decision log

Non-obvious decisions, newest first — so future work doesn't re-litigate them.

- 2026-09-03 — Scaffolded from `ds_ml_ai_starter`'s content-authoring `.claude/` framework, adapted
  to a Java/Maven stack: chapter specs → feature specs, snippet-compile gate → `mvn test` +
  Checkstyle/Spotless/SpotBugs, "chapter" roles → researcher/implementer/reviewer/architect (same
  model routing: Haiku grounds, Sonnet implements, fresh Sonnet reviews, Opus scopes and merges).

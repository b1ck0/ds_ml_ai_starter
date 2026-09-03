# Definition of Done — feature gate checklist

A feature is DONE only when **every** box below is checked. No exceptions "to be fixed later".

## Fidelity to the spec
- [ ] Every acceptance criterion in the approved `docs/features/FEATURE-*.md` is met.
- [ ] Anything cut from the spec is recorded in the spec's "Out of scope", not silently dropped.

## Test-first (the implementer's discipline)
- [ ] A JUnit 5 test encoding the acceptance criteria was written and run BEFORE the production code
      that satisfies it — and it failed for the right reason (missing behaviour, not a typo).
- [ ] No test was weakened (loosened assertion, deleted case, `@Disabled`) to make the gate pass.

## Grounded
- [ ] Every dependency version pinned in `pom.xml`, every library-behaviour claim the design relies
      on, and every CVE/security check traces to a researcher's note or a live, dated citation.
- [ ] No claim rests on model memory alone.

## Green gate
- [ ] `mvn clean test` passes.
- [ ] `mvn checkstyle:check` passes clean.
- [ ] `mvn spotless:check` passes clean (or `spotless:apply` was run before commit).
- [ ] `mvn spotbugs:check` passes clean.
- [ ] `.claude/hooks/verify.sh` (the fast per-edit subset) is green.

## Hygiene
- [ ] No secrets committed; any `.env.example` updated if config changed.
- [ ] Nothing outside the feature spec's stated files changed.

## Process
- [ ] One feature per PR; PR body maps each acceptance criterion → its evidence (test name / gate
      log / grounding note).
- [ ] Independent review by a **fresh** reviewer (not the implementer) — sign-off recorded.
- [ ] Architect (Opus) merge approval.

## Escalate instead of forcing
Stop and ask the owner if a feature's scope is ambiguous, a claim can't be grounded from available
sources, a dependency turns out to have a known CVE or is unmaintained, or a design choice conflicts
with existing code.

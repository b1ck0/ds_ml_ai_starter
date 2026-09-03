# Reference transcript — FEATURE-1 through the governed loop

> **This is a REFERENCE transcript, not a captured session log.** It narrates, stage by stage, what
> dispatching `java-project`'s architect/researcher/implementer/reviewer roster actually produces —
> using the real files this chapter committed under
> [`code/java-project/`](../code/java-project/) — so you can see the shape of the loop without
> paying for a live run. Two things in it ARE real, captured evidence, called out explicitly where
> they appear: the `javac` compile of `LuhnValidator.java`, and the smoke-check run of its logic
> against the same nine cases the JUnit suite encodes (this sandbox has a JDK but no Maven — see
> [`java-sdlc-scaffold.md`](../01-java-sdlc-scaffold.md) §5 for why, and for that real, captured
> output). Everywhere a `mvn` command's output is shown below, it is **illustrative** — the command
> and the file names are real; the exact console bytes are not a transcript of an actual run in this
> sandbox.

---

## Stage 1 — Request → spec

**Owner prompt to the architect:**
> "Support traced three chargebacks to typo'd card numbers this month. Add a check that rejects an
> obviously-malformed card number before we call the payment gateway."

**Architect (Opus), under `CLAUDE.md` golden rule 1** ("no code is written without an approved
feature spec"), writes
[`docs/features/FEATURE-1-luhn-validator.md`](../code/java-project/docs/features/FEATURE-1-luhn-validator.md):
intent, four acceptance criteria (AC1–AC4), and a "claims to ground" section that concludes **no
external research is needed** — the Luhn checksum is a fixed public algorithm, cited inline with the
date checked, not a package version or a "library X does Y" claim. The owner approves; status is set
to `approved`.

## Stage 2 — Ground the unknowns

Skipped for this feature, by design — the spec's own "Claims to ground" section states why (no
dependency added, no library behaviour asserted, the algorithm is fully specified in the spec
itself). This is what the parenthesised `(research)` in `docs/architecture.md` §3 step 2 means in
practice: the stage exists in the loop, but it is a no-op when a feature spec has nothing left to
verify. Compare: a feature that *did* add a new Maven dependency would stop here for a
`.claude/agents/researcher.md` (Haiku) dispatch against Maven Central + the NVD before the
implementer touches `pom.xml`.

## Stage 3 — Failing test first, then implement

**Architect dispatches the implementer** (`.claude/agents/implementer.md`, Sonnet) against
`FEATURE-1-luhn-validator.md`.

**Implementer, step 1 — write the test before the code exists to satisfy it:**

```
$ implementer writes src/test/java/com/example/sdlcdemo/LuhnValidatorTest.java
$ implementer writes an EMPTY src/main/java/com/example/sdlcdemo/LuhnValidator.java
    (package declaration + a stub `isValid` that just `throw new UnsupportedOperationException()`)
$ mvn -q test
```

```text
[REFERENCE — illustrative Surefire output, not captured in this sandbox]
[ERROR] Tests run: 9, Failures: 0, Errors: 9, Skipped: 0
[ERROR]   LuhnValidatorTest.acceptsKnownValidNumbers[1]  Time elapsed: 0.012 s  <<< ERROR!
java.lang.UnsupportedOperationException
	at com.example.sdlcdemo.LuhnValidator.isValid(LuhnValidator.java:9)
...
[INFO] BUILD FAILURE
```

Good — it fails for the *right* reason (the behaviour genuinely doesn't exist yet), not a typo in
the test. That's the check `CLAUDE.md` golden rule 2 and `docs/definition-of-done.md`'s "Test-first"
section both demand before a single line of real logic is written.

**Implementer, step 2 — write the minimum production code:**

```
$ implementer replaces the stub with the real Luhn checksum in LuhnValidator.java
```

**Real, captured evidence for this step** (this sandbox: JDK 24.0.1, no Maven — see
`java-sdlc-scaffold.md` §5 for the full explanation and the exact commands):

```text
$ javac -d out -Xlint:all src/main/java/com/example/sdlcdemo/LuhnValidator.java
[no output — clean compile]

$ java -cp out SmokeCheck   # ad-hoc driver running the same 9 cases LuhnValidatorTest.java encodes
PASS  acceptsKnownValidNumbers("4111111111111111") -> true
PASS  acceptsKnownValidNumbers("79927398713") -> true
PASS  acceptsKnownValidNumbers("4012 8888 8888 1881") -> true
PASS  rejectsKnownInvalidNumbers("4111111111111112") -> false
PASS  rejectsKnownInvalidNumbers("1234567890123456") -> false
PASS  rejectsKnownInvalidNumbers("79927398710") -> false
PASS  rejectsNonDigitCharacters -> false
PASS  rejectsEmptyString -> false
PASS  rejectsNull (IllegalArgumentException thrown) -> true

9 passed, 0 failed
```

## Stage 4 — Gate

`.claude/hooks/guard.sh` (PreToolUse) has already been running silently under every `Bash` call the
implementer made — it would have blocked, for example, `mvn -q test -DskipTests` outright (exit 2)
had the implementer ever tried to route around a failing test instead of fixing it. `verify.sh`
(PostToolUse) fires after every `Edit`/`Write` to a `.java` file:

```text
[REFERENCE — illustrative; a real run needs mvn on PATH, which this sandbox does not have]
[verify] java gate: src/main/java/com/example/sdlcdemo/LuhnValidator.java
  $ mvn -q test
  $ mvn -q checkstyle:check
  $ mvn -q spotless:check
  $ mvn -q spotbugs:check
[verify] all four gates green
```

The implementer reports the full `docs/definition-of-done.md` checklist back to the architect,
mapping AC1–AC4 to the test methods that cover them:

| AC | Test method |
|---|---|
| AC1 | `acceptsKnownValidNumbers` |
| AC2 | `rejectsKnownInvalidNumbers` |
| AC3 | `rejectsNonDigitCharacters`, `rejectsEmptyString` |
| AC4 | `rejectsNull` |

## Stage 5 — Review

**Architect dispatches a FRESH reviewer** (`.claude/agents/reviewer.md`, a new Sonnet context that
never saw the implementer's scratch work — same discipline as this book's own
`chapter-reviewer.md`). The reviewer re-runs the gate independently rather than trusting the
implementer's report, checks each AC against its named test method, confirms the test file's first
version really did fail (via the implementer's reported step-1 output above), and checks that
nothing outside `FEATURE-1`'s "Assets to produce" changed.

**Reviewer verdict:**
> **APPROVE.** AC1–AC4 each map to a passing test method; the test-first order is evidenced by the
> step-1 failure log; `checkstyle`/`spotless`/`spotbugs` are all clean; no file outside the spec's
> two listed assets changed. One note, not blocking: `LuhnValidator.isValid` throws
> `IllegalArgumentException` rather than returning `false` for `null` — confirmed intentional per
> AC4 and the spec's own rationale (a `null` reaching this method is a caller bug, not a malformed
> card number), not an oversight.

## Stage 6 — Merge

**Architect merges**, under `CLAUDE.md`'s merge-approval rule. PR body maps every acceptance
criterion to its evidence (the table above + the gate log). The loop reopens at stage 1 for the next
feature request — exactly the same six-stage shape `docs/architecture.md` §3 describes, and the same
shape [`AI-assisted-sdlc/Theory/theory.md`](../../01-theory/01-theory.md) §6 walked through for a chapter
instead of a feature.

# Scaffolding a Governed SDLC for a New Java Project

*AI-assisted-sdlc · Worked Examples · SPEC-SDLC-2 (the capstone)*

## 1. What & why — the same four primitives, a different build

[SPEC-SDLC-1 (Theory)](../01-theory/01-theory.md) named four scaffolding primitives an agentic coding
tool needs before it behaves like a disciplined engineer instead of a fast, opinion-free intern:
**prompts & rules** (persistent, advisory instructions — `CLAUDE.md`), **hooks & gates**
(deterministic automation that can actually block a bad action), **tools & MCP** (typed
capabilities, built-in or externally registered), and **sub-agents & skills** (a fresh-context
specialist dispatch versus a packaged procedure loaded into the current one). That chapter proved
every claim against `ds_ml_ai_starter`'s own `.claude/` folder — the exact scaffold that produced
the chapter you're reading right now.

This chapter reproduces that same scaffold for the stack you actually work in: a Java project built
with Maven, tested with JUnit 5, gated by Checkstyle/Spotless/SpotBugs. Nothing about the four
primitive categories changes — a hook is still a hook, a sub-agent is still a sub-agent, the
`.claude/settings.json` schema is byte-for-byte the same JSON Schema. What changes is *what the gate
checks*: this book's own scaffold byte-compiles a Python snippet
([`.claude/hooks/check_snippets.py`](../../.claude/hooks/check_snippets.py)); a Java project's
gate compiles, tests, and style-checks real production code —
[`research/NOTE-SDLC-3-java-gates.md`](../../research/NOTE-SDLC-3-java-gates.md) (verified
2026-09-02) pins the exact tool versions used below. You already run an equivalent gate today,
probably as a Maven `verify` phase wired into CI; the only new idea here is that an *agent*, not
just a human, is now a citizen of that gate — and needs the same guardrails a new hire would.

Everything this chapter builds lives under
[`code/java-project/`](code/java-project/) — a complete, standalone Maven project with its own
`.claude/` scaffold nested inside it, exactly as it would sit in a real repository. Section 2–4 tour
that scaffold file by file, comparing each one directly against the file in *this* repository's root
that plays the same role. Section 5 runs one real feature — a card-number checksum validator —
through the full spec → (research) → failing test → implement → gate → review → merge loop. Section
6 lists what breaks the loop if you skip a step. Section 7 is the direct comparison table: what
carries over unchanged from a content-authoring scaffold to a code scaffold, and what's genuinely
different.

**Toolchain baseline**, all pinned in
[`research/NOTE-SDLC-1-java-toolchain.md`](../../research/NOTE-SDLC-1-java-toolchain.md)
(verified 2026-09-02) and matching this subject's own
[`Local Environment Setup/code/hello-java/pom.xml`](../02-local-environment-setup/code/hello-java/pom.xml):
JDK 25 LTS, Maven 3.9.16, JUnit 5.14.3 (`org.junit.jupiter:junit-jupiter`), Maven Surefire 3.5.6
(the stable line — 3.6.0-M1 is a milestone, not used here). This chapter's sandbox has JDK 24.0.1
but no Maven or Gradle installed; §5 is explicit about exactly which evidence below is real,
captured output from *this* sandbox, and which is the verified-reference `mvn`/`gradle` command you
run on a machine that has them.

## 2. The charter + docs — one-to-one with this repository's own

[`code/java-project/CLAUDE.md`](code/java-project/CLAUDE.md) is structured identically to this
repository's own [`CLAUDE.md`](../../CLAUDE.md) — same seven-rule shape, same model-routing
table, same escalation section — with every noun swapped from "chapter" to "feature" and the
grounding/gate specifics swapped from Python-textbook rules to Java-engineering rules:

| This repository (content authoring) | `java-project/` (Java engineering) |
|---|---|
| No chapter without an approved `specs/SPEC-*.md` | No code without an approved `docs/features/FEATURE-*.md` |
| Every claim grounded (NOTE or citation) | Every dependency version / API claim / CVE check grounded |
| Every snippet must run | Every JUnit 5 test must exist and FAIL before the code that satisfies it exists — a rule this repo's own scaffold has no equivalent for, because chapter code has no "before/after" ordering requirement the way test-first engineering does |
| Exit gate: snippet compile + link check + review | Exit gate: `mvn clean test` + `checkstyle:check` + `spotless:check` + `spotbugs:check` + review |

[`code/java-project/docs/architecture.md`](code/java-project/docs/architecture.md) mirrors
[`docs/architecture.md`](../../docs/architecture.md)'s six sections (operating model, roles,
per-unit workflow, repository shape, gates, decision log) point for point — including a decision
log entry recording exactly this port, the same way this repo's own architecture doc keeps a running
log of non-obvious calls.
[`code/java-project/docs/definition-of-done.md`](code/java-project/docs/definition-of-done.md) is
the Java analogue of [`docs/definition-of-done.md`](../../docs/definition-of-done.md): the
categories (fidelity, grounded, runnable/green-gate, hygiene, process, escalate-instead-of-forcing)
are the same shape, with "runnable" split into an explicit "test-first" checklist item this repo's
own DoD doesn't need — a textbook chapter's code doesn't need to have failed first; a production
feature's test absolutely does, because that failure is the only proof the test was ever capable of
catching the bug it claims to guard against.

## 3. The agent roster — researcher / implementer / reviewer, architect stays human-driven

The model-routing table is identical to this repo's own
([`CLAUDE.md`](../../CLAUDE.md) "Model routing"): the architect (Opus) is the **main session** —
not a `.claude/agents/*.md` file, here or in this repository — because scoping, gate decisions, and
merge approval are exactly the judgment calls you don't want a dispatched, disposable-context
sub-agent making unsupervised. The three roles that *are* files:

- [`.claude/agents/researcher.md`](code/java-project/.claude/agents/researcher.md) — Haiku. Same
  job as this repo's own [`researcher.md`](../../.claude/agents/researcher.md) — verify an
  external fact and write it down — retargeted from "PyPI version + dataset licence" to "Maven
  Central version + known-CVE check via the NVD."
- [`.claude/agents/implementer.md`](code/java-project/.claude/agents/implementer.md) — Sonnet. The
  Java-project counterpart of
  [`chapter-writer.md`](../../.claude/agents/chapter-writer.md), with one structural addition
  neither this repo's writer nor its reviewer needs: an explicit, numbered "write the failing test
  first" step, because — unlike a chapter, which either runs or doesn't — a feature's test is only
  trustworthy if it was seen to fail for the right reason before the code existed to pass it.
- [`.claude/agents/reviewer.md`](code/java-project/.claude/agents/reviewer.md) — a **fresh** Sonnet,
  same non-negotiable property as
  [`chapter-reviewer.md`](../../.claude/agents/chapter-reviewer.md): it must never be the same
  context as the implementer, because a reviewer that remembers writing the code cannot
  independently doubt it. Its process explicitly re-runs the gate itself rather than trusting the
  implementer's report — see §5.

All three frontmatter blocks use only the **core** sub-agent fields — required `name` + `description`,
plus `model` from the documented set
[source: Subagents — Claude Code Docs](https://code.claude.com/docs/en/sub-agents) (checked
2026-09-03: the live page's frontmatter table lists `name` (required), `description` (required),
`tools`, and `model` — with the documented values `sonnet`, `opus`, `haiku`, `fable`, a full model ID,
or `inherit`). These four are all this scaffold needs.
[`research/NOTE-SDLC-2-claude-code.md`](../../research/NOTE-SDLC-2-claude-code.md) and
[`research/NOTE-SDLC-3-java-gates.md`](../../research/NOTE-SDLC-3-java-gates.md) additionally catalog
the **optional** extended fields (`permissionMode`, `maxTurns`, `skills`, `mcpServers`, `hooks`,
`memory`, `isolation`, `color`, and others) for when you need them — see NOTE-SDLC-2 for the full list.
This isn't asserted from memory:
[`code/validate_frontmatter.py`](code/validate_frontmatter.py) parses all three files' YAML
frontmatter with PyYAML and checks it against that documented schema programmatically. Real,
captured run:

```text
$ .venv/Scripts/python.exe "AI-assisted-sdlc/Worked Examples/code/validate_frontmatter.py" \
    "AI-assisted-sdlc/Worked Examples/code/java-project/.claude/agents/researcher.md" \
    "AI-assisted-sdlc/Worked Examples/code/java-project/.claude/agents/implementer.md" \
    "AI-assisted-sdlc/Worked Examples/code/java-project/.claude/agents/reviewer.md"
OK   ...researcher.md: name='researcher' model='haiku' fields=['description', 'model', 'name', 'tools']
OK   ...implementer.md: name='implementer' model='sonnet' fields=['description', 'model', 'name', 'tools']
OK   ...reviewer.md: name='reviewer' model='sonnet' fields=['description', 'model', 'name', 'tools']
```

Full output in [`artefacts/validation-log.md`](artefacts/validation-log.md) §2.

## 4. Hooks + settings — the enforced layer, ported

[`code/java-project/.claude/settings.json`](code/java-project/.claude/settings.json) is
**byte-for-byte the same structure** as this repository's own
[`.claude/settings.json`](../../.claude/settings.json) — same three events, same handler shape —
because the settings schema doesn't know or care what language your hooks check:

```json
{
  "$schema": "https://json.schemastore.org/claude-code-settings.json",
  "hooks": {
    "PostToolUse": [
      { "matcher": "Edit|Write", "hooks": [{ "type": "command", "command": ".claude/hooks/verify.sh" }] }
    ],
    "PreToolUse": [
      { "matcher": "Bash", "hooks": [{ "type": "command", "command": ".claude/hooks/guard.sh" }] }
    ],
    "SessionStart": [
      { "hooks": [{ "type": "command", "command": ".claude/hooks/context.sh" }] }
    ]
  }
}
```

This is grounded past the documentation prose: the published schema itself
[source: Claude Code settings schema — SchemaStore](https://json.schemastore.org/claude-code-settings.json)
(checked 2026-09-03) was fetched and parsed directly. Its top-level `hooks` property is `type:
object`, `additionalProperties: false`, with one array-valued key per lifecycle event —
`PreToolUse`, `PostToolUse`, `SessionStart`, and 25 more — each event's items validated against a
shared `hookMatcher` definition requiring a `hooks` array (each entry a `hookCommand` — `type:
command|http|mcp_tool|prompt|agent`, plus handler-specific fields) and an *optional* `matcher`
string. That is exactly the shape [`research/NOTE-SDLC-2-claude-code.md`](../../research/NOTE-SDLC-2-claude-code.md)
and [`research/NOTE-SDLC-3-java-gates.md`](../../research/NOTE-SDLC-3-java-gates.md) documented
from the prose docs, now confirmed against the machine-readable schema itself, and it's exactly what
both settings.json files above declare. `python -m json.tool` — the same well-formedness check this
book's own `check_snippets.py` gate would demand of any JSON asset — confirms the file parses; full
output in [`artefacts/validation-log.md`](artefacts/validation-log.md) §1.

The three hook scripts are line-for-line adaptations of this repo's own, with the language-specific
gate swapped in:

- **[`guard.sh`](code/java-project/.claude/hooks/guard.sh)** (`PreToolUse` on `Bash`) keeps every
  deny-rule from [this repo's `guard.sh`](../../.claude/hooks/guard.sh) (`rm -rf /`, forced
  `git push`, printing a secret, shutdown/reboot) and adds one Java-specific rule this project
  couldn't have without it: it blocks `mvn ... -DskipTests` outright. That flag exists precisely to
  let a broken build "succeed" by not running the tests that would catch the break — exactly the
  kind of self-defeating shortcut an agent under deadline pressure might reach for, and exactly what
  a `PreToolUse` veto is for.
- **[`verify.sh`](code/java-project/.claude/hooks/verify.sh)** (`PostToolUse` on `Edit|Write`) is
  the gate itself: on any `.java` edit it runs `mvn -q test`, then `checkstyle:check`,
  `spotless:check`, `spotbugs:check` — the four tools
  [`research/NOTE-SDLC-3-java-gates.md`](../../research/NOTE-SDLC-3-java-gates.md) names as the
  current standard Java quality gate (JUnit 5 via Surefire; Checkstyle 14.1.0 via the Maven plugin
  3.6.0; Spotless 2.46.1; SpotBugs 4.10.3.0 — versions pinned in
  [`pom.xml`](code/java-project/pom.xml)) — on a `pom.xml` edit it runs `mvn -q validate` instead,
  catching a broken POM before the next full gate run bothers compiling anything.
- **[`context.sh`](code/java-project/.claude/hooks/context.sh)** (`SessionStart`) prints the same
  kind of orientation banner as [this repo's own](../../.claude/hooks/context.sh) — read-first
  docs, the roster, and (ported directly from this repo's spec-listing loop) every
  `docs/features/FEATURE-*.md` and its status, so a new session immediately sees what's approved and
  waiting for an implementer.

All three were actually executed against synthetic hook payloads (exactly the JSON Claude Code
would pipe to a `command` hook on stdin), not just read and trusted:

```text
$ printf '{"command":"mvn -q -DskipTests package"}' | bash guard.sh
[guard] BLOCKED: mvn -DskipTests (the test-first gate exists precisely so this never happens silently)
exit=2

$ printf '{"file_path":"src/main/java/com/example/sdlcdemo/LuhnValidator.java"}' | bash verify.sh
[verify] mvn not on PATH — skipping gate for src/main/java/com/example/sdlcdemo/LuhnValidator.java
exit=0
```

Full run (benign command allowed, `rm -rf /` blocked, the skip-tests block above, `context.sh`'s
banner, and `verify.sh`'s graceful degrade when `mvn` is absent — this sandbox's actual condition)
is in [`artefacts/validation-log.md`](artefacts/validation-log.md) §§1–3.

**Gradle equivalent**, for a reader whose team standardised on Gradle instead of Maven — shown here
as a grounded reference translation, not a second maintained project (`java-project/` stays Maven
throughout, matching this subject's `hello-java` baseline from SPEC-SDLC-0):

```kotlin
// build.gradle.kts — equivalent gate wiring, versions from research/NOTE-SDLC-3-java-gates.md
plugins {
    id("com.diffplug.spotless") version "8.10.1"
    id("com.github.spotbugs") version "6.5.11"
}

dependencies {
    testImplementation("org.junit.jupiter:junit-jupiter:5.14.3")
}

tasks.test { useJUnitPlatform() }
spotless { java { googleJavaFormat() } }
spotbugs { ignoreFailures = false }
```

`verify.sh`'s Gradle equivalent would run `gradle test spotlessCheck spotbugsMain`. One honest gap,
flagged rather than hidden: `research/NOTE-SDLC-3-java-gates.md` records that **Gradle has no
official Checkstyle plugin** — Checkstyle here is Maven-only; a Gradle project needs either a custom
task invoking the Checkstyle JAR directly, or to drop Checkstyle and rely on Spotless's formatting
enforcement plus SpotBugs's bug detection.

## 5. The loop in action — FEATURE-1, spec to merge

Support traced three chargeback disputes this month to card numbers typo'd at checkout — a digit
transposed, one dropped — that were never validated before hitting the payment gateway. That's a
real, small, self-contained feature: reject an obviously malformed card number for free, before
paying a gateway round-trip to find out. It runs through every stage of the loop exactly once.

![Diagram of a six-stage rectangular loop for the java-project feature workflow: 1. Request -> spec, 2. Ground (if needed), 3. Failing test -> implement (top row, left to right), then down to 4. Gate, 5. Review, 6. Merge (bottom row, right to left), with an arrow looping back up from Merge to Request labelled "loop closes -> next feature." Each stage box lists the concrete primitives active there, colour-coded exactly as in the Theory chapter's diagram: blue for prompts & rules, red for hooks & gates, green for tools & MCP, purple for sub-agents & skills, each row naming the actual file in java-project/ that plays that role.](artefacts/java_feature_loop_diagram.png)

*(Generated by [`code/java_feature_loop_diagram.py`](code/java_feature_loop_diagram.py) — reproduce
with `.venv/Scripts/python.exe "AI-assisted-sdlc/Worked Examples/code/java_feature_loop_diagram.py"`.)*

The full stage-by-stage narration — architect writing the spec, the researcher stage being skipped
with its reasoning stated, the implementer's failing-test-first sequence, the gate, the fresh
reviewer's verdict, and the merge — is
[`artefacts/feature-loop-transcript.md`](artefacts/feature-loop-transcript.md). It is explicitly
labelled a **reference transcript**: the file names, the spec, the code, and the review verdict's
reasoning are all real and match what's committed under `code/java-project/`; the exact console
bytes of an `mvn test` run are illustrative, because this sandbox has no Maven installed. What *is*
real, captured evidence, run while writing this chapter (JDK 24.0.1, `javac`/`java`, no Maven or
Gradle on `PATH` — confirmed with `which mvn`, `which gradle`, and no `~/.m2` cache present):

```text
$ javac -d out -Xlint:all src/main/java/com/example/sdlcdemo/LuhnValidator.java
[no output — clean compile, zero warnings]

$ java -cp out SmokeCheck   # ad-hoc driver exercising the same 9 cases the JUnit suite encodes
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

The production code is [`LuhnValidator.java`](code/java-project/src/main/java/com/example/sdlcdemo/LuhnValidator.java);
the real JUnit 5 suite that ships with it (and that `mvn test` runs once Maven is on `PATH`) is
[`LuhnValidatorTest.java`](code/java-project/src/test/java/com/example/sdlcdemo/LuhnValidatorTest.java) —
9 test invocations across 5 methods (2 `@ParameterizedTest` with 3 cases each, 3 plain `@Test`),
matching the 9 smoke-check lines above one for one. Attempting to `javac`-compile the test file
directly (impossible without JUnit on the classpath) was also tried, specifically to confirm the
file has no *syntax* error hiding behind the missing-dependency errors — every reported error is
`package org.junit.jupiter... does not exist`, zero parse failures. Full log:
[`artefacts/validation-log.md`](artefacts/validation-log.md) §4.

The Luhn checksum itself — double every second digit counting from the right, subtract 9 from any
doubled digit over 9, sum everything, valid iff the total is a multiple of 10 — is grounded inline
in the feature spec and the Javadoc, not asserted from memory: [source: Luhn algorithm —
Wikipedia](https://en.wikipedia.org/wiki/Luhn_algorithm) (checked 2026-09-02). This is why
[`docs/features/FEATURE-1-luhn-validator.md`](code/java-project/docs/features/FEATURE-1-luhn-validator.md)'s
own "Claims to ground" section says **none required**: the algorithm is a fixed, fully-specified
piece of arithmetic with an authoritative inline citation, not a package version, a dataset licence,
or a "library X does Y" claim — the kind of fact this project's researcher role exists to verify.
That is stage 2 of the loop (§ "Ground the unknowns" in
[`docs/architecture.md`](code/java-project/docs/architecture.md)) working correctly by *not firing*
— the stage exists for the features that need it, and this one doesn't.

## 6. Pitfalls

- **Skipping the failing-test-first step.** If the implementer writes the production code and the
  test in the same pass, with the test never observed to fail, you've verified that the test and the
  code agree with *each other* — not that the test would have caught the bug it claims to guard
  against. `docs/definition-of-done.md`'s "Test-first" section and the reviewer's process both name
  this explicitly, because it's the single easiest step for a Sonnet under time pressure to quietly
  collapse into "write both together."
- **Weakening a test to make the gate pass.** A loosened assertion, a deleted edge case, or
  `@Disabled` on a stubborn test is a passing gate that proves nothing. `guard.sh`'s
  `-DskipTests` block stops the crude version of this; the fresh reviewer (§3, §5) is the backstop
  for the subtle version, because a hook can't tell a legitimately-relaxed assertion from a gamed
  one — only a human-directed second reader with the spec in hand can.
- **Unbounded agent scope.** The implementer's frontmatter and its "Boundaries" section both scope
  it to exactly the spec's "Assets to produce" — two files for `FEATURE-1`. An implementer that
  "while I'm in there" refactors an unrelated class has quietly expanded a one-feature PR into an
  unreviewable diff; the reviewer's step 5 (§ process above) exists specifically to catch this.
- **Skipping independent review.** The reviewer's whole value is that it did not write the code —
  it re-runs the gate itself rather than trusting the implementer's report (§3), which catches a
  gate that passed locally but not from a clean checkout, or a report that quietly omitted a step.
  Merging on the implementer's own say-so removes the one check in this loop that isn't automatable.
- **Pinning a milestone build instead of a stable release.** `research/NOTE-SDLC-1-java-toolchain.md`
  flags this by name: Maven Surefire 3.6.0-M1 is a milestone as of 2026-09-02, not the stable
  3.5.6 this project pins — an easy version-number mix-up (higher number looks newer) that a
  researcher-grounded pin, checked against the live release page rather than assumed, prevents.
- **Assuming Gradle mirrors Maven plugin-for-plugin.** §4's Gradle translation flags this directly:
  there's no official Gradle Checkstyle plugin — copying the Maven gate's four tools verbatim into a
  Gradle build silently drops one of them unless you notice and compensate.

## 7. Reference — what carries over from this repository's own scaffold, and what's Java-specific

| Layer | This repo (content authoring) | `java-project/` (Java engineering) | Carries over? |
|---|---|---|---|
| `.claude/settings.json` schema | 3 events, `command` handlers | identical 3 events, identical handler shape | **Unchanged** — the schema doesn't know what language the hook checks |
| Agent frontmatter schema | `name`, `description`, `model`, `tools` | identical fields, same documented value sets | **Unchanged** |
| `guard.sh` deny-list | `rm -rf /`, forced push, secrets, shutdown | same list **+** `mvn -DskipTests` | **Extended**, not replaced |
| `context.sh` orientation pattern | lists `specs/SPEC-*.md` + status | lists `docs/features/FEATURE-*.md` + status | **Unchanged pattern**, different glob |
| `verify.sh` gate | byte-compile a `.py`/fenced-`.md`-python snippet | `mvn test` + checkstyle + spotless + spotbugs | **Java-specific** — the one place the stack actually matters |
| Spec unit | one chapter, `specs/SPEC-*.md` | one feature, `docs/features/FEATURE-*.md` | Same shape, different granularity |
| "Runnable" gate criterion | snippet compiles | test existed and **failed** before the code, then passes | **Java-specific addition** — test-first has no equivalent in prose-writing |
| Roster | researcher / writer / reviewer / architect | researcher / implementer / reviewer / architect | **Unchanged roles**, renamed to fit the domain |
| Model routing | Haiku grounds, Sonnet writes, fresh Sonnet reviews, Opus scopes+merges | identical | **Unchanged** |

The takeaway worth over-learning: everything Claude Code itself provides — the settings schema, the
agent frontmatter schema, the permission model (Deny > Ask > Allow, enforced independently of any
prompt — [source: Configure permissions — Claude Code
Docs](https://code.claude.com/docs/en/permissions) (checked 2026-09-03)), the hook lifecycle events —
is stack-agnostic and ports verbatim. What you author *for* your stack is the gate's actual content
(`verify.sh`'s tool invocations) and the domain-specific rule your engineering discipline needs that
prose-writing doesn't (test-first). Reproducing this scaffold for your own Java project is
mechanical: copy `code/java-project/.claude/`, `CLAUDE.md`, and `docs/` verbatim into your repo root,
point `pom.xml`'s plugin versions at whatever your own researcher grounds as current when you do it
(the versions pinned here were current 2026-09-02 — check again), replace `FEATURE-1` with your
project's actual first feature, and the loop in §5 is yours to run for real, with a real `mvn` on
`PATH`.

Wiring the same four `mvn` gate commands into a CI branch-protection rule (GitHub Actions, or
whatever your team runs) is the natural next step for a real repository, and is explicitly out of
scope for this chapter — see [SPEC-SDLC-2](../../specs/SPEC-SDLC-2-java-project-sdlc-scaffold.md)
"Scope: Out." The gate commands themselves don't change moving into CI; only *where* they run does.

## 8. Recap & what's next

Four primitive categories, one governed loop, now proven twice: once for a textbook chapter
(SPEC-SDLC-1, this repository's own scaffold), once for a Java feature (this chapter,
`code/java-project/`). Prompts & rules steer intent without enforcing it; hooks & gates are the only
layer that can actually block a bad action, and here that means a real `mvn test` plus three style
and bug-detection tools instead of a snippet compile; tools & MCP are the typed capabilities
(`Bash` running `mvn`, in this chapter's case) an agent calls; sub-agents & skills are the
delegation boundary — a fresh researcher for facts, a fresh implementer for code, and critically, a
**fresh** reviewer that never shares context with the implementer it's checking.

This is the capstone worked example for the AI-assisted-sdlc subject — the loop described here and
in [SPEC-SDLC-1](../01-theory/01-theory.md) is the same one that governed the writing of every chapter in
this book, including the three you've just read. If you came to this book to learn Data Science or
Machine Learning rather than SDLC tooling, the **Data Science** subject's own Local Environment
Setup chapter is the natural next stop; if you're setting up a governed loop on your own Java
project right now, `code/java-project/` is a complete, ready-to-fork starting point.

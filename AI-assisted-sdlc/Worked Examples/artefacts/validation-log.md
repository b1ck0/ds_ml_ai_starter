# Validation log — java-project/ scaffold

Real, captured output from validating every file under
[`code/java-project/`](../code/java-project/) against the schemas in
[`research/NOTE-SDLC-2-claude-code.md`](../../../research/NOTE-SDLC-2-claude-code.md) and
[`research/NOTE-SDLC-3-java-gates.md`](../../../research/NOTE-SDLC-3-java-gates.md) (both official
Claude Code docs, verified 2026-09-02). Every command below was actually run in this repository's
sandbox (Windows 11, JDK 24.0.1, Python 3.13 in `.venv/`) while writing this chapter — nothing here
is a hypothetical or illustrative example. Environment gaps (no Maven/Gradle installed) are called
out explicitly rather than glossed over — see [`java-sdlc-scaffold.md`](../java-sdlc-scaffold.md)
§5 for the full explanation.

## 1. `settings.json` parses as valid JSON

```
$ .venv/Scripts/python.exe -m json.tool code/java-project/.claude/settings.json
{
    "$schema": "https://json.schemastore.org/claude-code-settings.json",
    "hooks": {
        "PostToolUse": [
            {
                "matcher": "Edit|Write",
                "hooks": [
                    {
                        "type": "command",
                        "command": ".claude/hooks/verify.sh"
                    }
                ]
            }
        ],
        "PreToolUse": [
            {
                "matcher": "Bash",
                "hooks": [
                    {
                        "type": "command",
                        "command": ".claude/hooks/guard.sh"
                    }
                ]
            }
        ],
        "SessionStart": [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": ".claude/hooks/context.sh"
                    }
                ]
            }
        ]
    }
}
```

`python -m json.tool` exits non-zero on malformed JSON and prints nothing above — a clean
pretty-printed echo back, as above, is the pass signal. Matches the hooks-block shape documented in
NOTE-SDLC-2 "Hooks: Events, Settings Schema, Handlers" and NOTE-SDLC-3 "Claude Code
`.claude/settings.json` Schema" — three events (`PostToolUse`, `PreToolUse`, `SessionStart`), each a
list of `{matcher, hooks: [{type: "command", command: ...}]}` groups.

## 2. Agent frontmatter is well-formed YAML and matches the documented schema

[`code/validate_frontmatter.py`](../code/validate_frontmatter.py) (parses the `---`-delimited
frontmatter block with PyYAML, checks `name` + `description` are present, every other key is in the
documented optional-field set from NOTE-SDLC-2 "Agents" / NOTE-SDLC-3 "Agent Frontmatter Schema",
`model` is one of the documented values, and `name` is lowercase-plus-hyphens) run against all three
roster files:

```
$ .venv/Scripts/python.exe validate_frontmatter.py \
    code/java-project/.claude/agents/researcher.md \
    code/java-project/.claude/agents/implementer.md \
    code/java-project/.claude/agents/reviewer.md
OK   code/java-project/.claude/agents/researcher.md: name='researcher' model='haiku' fields=['description', 'model', 'name', 'tools']
OK   code/java-project/.claude/agents/implementer.md: name='implementer' model='sonnet' fields=['description', 'model', 'name', 'tools']
OK   code/java-project/.claude/agents/reviewer.md: name='reviewer' model='sonnet' fields=['description', 'model', 'name', 'tools']
```

Model routing matches `docs/architecture.md` §2 exactly: researcher on Haiku, implementer and
reviewer on Sonnet — no `model:` field asserts a value outside the documented
`sonnet|opus|haiku|fable|inherit|<full-id>` set (NOTE-SDLC-2 "Agents" section).

## 3. Hooks actually execute and enforce what they claim to

`context.sh` (`SessionStart`) run directly:

```
$ bash code/java-project/.claude/hooks/context.sh
──────────────────────────────────────────────────────────
 java-project — a governed SDLC scaffold for a Java service
──────────────────────────────────────────────────────────
 Read first: docs/architecture.md · docs/definition-of-done.md · CLAUDE.md
 Roles: Opus (architect, main session) · .claude/agents/researcher.md (Haiku) ·
        .claude/agents/implementer.md (Sonnet) · .claude/agents/reviewer.md (fresh Sonnet)

 Feature specs:
   • FEATURE-1: Reject invalid card numbers before they reach the payment gateway  approved

 Gate reminders: failing JUnit 5 test committed before the code it covers ·
 mvn clean test passes · checkstyle/spotless/spotbugs all clean ·
 independent review before merge. (docs/definition-of-done.md)
──────────────────────────────────────────────────────────
exit=0
```

`guard.sh` (`PreToolUse`), fed synthetic hook payloads on stdin exactly as Claude Code would:

```
$ printf '{"command":"mvn -q test"}' | bash guard.sh
exit=0

$ printf '{"command":"rm -rf /"}' | bash guard.sh
[guard] BLOCKED: rm -rf /
exit=2

$ printf '{"command":"mvn -q -DskipTests package"}' | bash guard.sh
[guard] BLOCKED: mvn -DskipTests (the test-first gate exists precisely so this never happens silently)
exit=2
```

A benign build command passes through (exit 0); a destructive filesystem command and an attempt to
route around the test-first gate are both vetoed (exit 2, per the hook contract in NOTE-SDLC-2:
"non-zero exit signals a failing gate" / a `PreToolUse` hook "can block" the call it's watching).

`verify.sh` (`PostToolUse`), same synthetic-payload approach:

```
$ printf '{"file_path":"src/main/java/com/example/sdlcdemo/LuhnValidator.java"}' | bash verify.sh
[verify] mvn not on PATH — skipping gate for src/main/java/com/example/sdlcdemo/LuhnValidator.java
exit=0

$ printf '{"file_path":"README.md"}' | bash verify.sh
[verify] no checks for: README.md
exit=0
```

`verify.sh` degrades gracefully when `mvn` is absent (this sandbox) rather than failing the whole
hook — the same "skip gracefully when a tool is absent" contract this book's own
`.claude/hooks/verify.sh` documents for a missing `python`/`ruff`.

## 4. The production code compiles and the algorithm is correct

No Maven/Gradle is installed in this sandbox (checked: `which mvn`, `which gradle` both resolve to
nothing; no `~/.m2` cache either). `javac`/`java` (JDK 24.0.1) ARE present, so the production class
was compiled and its logic exercised directly, real and captured:

```
$ javac -d out -Xlint:all src/main/java/com/example/sdlcdemo/LuhnValidator.java
[no output — clean compile, zero warnings]

$ java -cp out SmokeCheck
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

`SmokeCheck.java` is an ad-hoc driver (not part of the shipped project — it exists only to exercise
the compiled `.class` file without a test runner) that asserts the exact 9 cases
`LuhnValidatorTest.java` encodes as `@Test`/`@ParameterizedTest` methods. Attempting to compile
`LuhnValidatorTest.java` itself without the JUnit 5 dependency on the classpath was also tried, to
confirm the test file has no syntax errors — every reported error is `package ... does not exist`
(an unresolved Maven dependency, expected without `mvn`), not a parse failure:

```
$ javac -d out-test -cp out src/test/java/com/example/sdlcdemo/LuhnValidatorTest.java
LuhnValidatorTest.java:3: error: package org.junit.jupiter.api does not exist
import static org.junit.jupiter.api.Assertions.assertFalse;
                                   ^
... (8 more errors, all "package ... does not exist" for org.junit.jupiter.* — zero syntax errors)
```

## Summary

| Check | Result |
|---|---|
| `settings.json` well-formed JSON | PASS |
| 3× agent frontmatter well-formed + schema-valid | PASS (3/3) |
| `context.sh` runs clean | PASS (exit 0) |
| `guard.sh` allows benign, blocks `rm -rf /` and `-DskipTests` | PASS (3/3 cases) |
| `verify.sh` degrades gracefully without `mvn` | PASS (2/2 cases) |
| `LuhnValidator.java` compiles clean | PASS (0 warnings) |
| `LuhnValidator` logic correct on 9 cases | PASS (9/9) |
| `LuhnValidatorTest.java` free of syntax errors | PASS (errors are unresolved-dependency only) |
| `mvn clean test` / `checkstyle:check` / `spotless:check` / `spotbugs:check` | **NOT run — no Maven in this sandbox.** Presented as verified reference per NOTE-SDLC-1/NOTE-SDLC-3 (exact commands, exact pinned versions); reproduce on a machine with Maven installed. |

Date verified: 2026-09-03.

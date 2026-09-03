# Artefact — real gate output for this chapter

*Referenced from [`02-how-this-repo-was-built.md`](../02-how-this-repo-was-built.md) §3 and §4.3.
Both commands below were run against this repository, against this chapter's own file, by the writer
who produced it — not fabricated for the prose. Re-run them yourself from the repo root; the output
should match exactly.*

## 1. Snippet-compile gate

```text
$ python .claude/hooks/check_snippets.py "04-ai-assisted-sdlc/03-worked-examples/02-how-this-repo-was-built.md"
[snippets] OK: all python blocks compiled in 04-ai-assisted-sdlc/03-worked-examples/02-how-this-repo-was-built.md
```

Exit code: `0`. This chapter has no fenced ` ```python ` blocks — every shell/output excerpt in the
prose is fenced ` ```text `, which `check_snippets.py`'s `SKIP_HINTS` set deliberately does not treat
as Python to compile. A file with zero Python blocks still reports `OK`, because there is nothing to
fail on — worth stating plainly rather than letting a clean pass look like "20 snippets verified" when
it verified none.

## 2. GitHub-render lint (LaTeX + Mermaid)

```text
$ python .claude/hooks/check_markdown_render.py "04-ai-assisted-sdlc/03-worked-examples/02-how-this-repo-was-built.md"
[render] OK: no LaTeX/Mermaid rendering issues in 1 file(s)
```

Exit code: `0`. This chapter's single Mermaid diagram (§1, the pipeline flowchart) was eyeballed as
rendered per `docs/definition-of-done.md`'s requirement that the checker is "necessary, not
sufficient" — every node label containing a `/` or `--` is plain text with no unquoted parentheses, so
none needed `"double quotes"` wrapping.

## 3. `guard.sh` — the live false-positive from §4.3, reproduced

This is the literal transcript from the session that wrote this chapter, captured while testing
`guard.sh`'s secret-detection rule from the inside with deliberately benign text:

```text
$ printf '%s' '{"command":"echo \"caching a fresh access token before the retry loop\""}' | bash .claude/hooks/guard.sh
```

Claude Code's own `PreToolUse` boundary intercepted the call before the shell ever ran it, and
returned this to the session:

```text
PreToolUse:Bash hook error: [.claude/hooks/guard.sh]: [guard] BLOCKED: printing a secret to stdout
```

`guard.sh`'s own source, at HEAD ([`.claude/hooks/guard.sh`](../../../.claude/hooks/guard.sh)):

```bash
# Policy: never print a secret to stdout.
printf '%s' "$cmd" | grep -Eiq 'echo[[:space:]].*(SECRET|API_KEY|TOKEN|PASSWORD|SERVICE_ROLE|PRIVATE_KEY)' && \
  deny "printing a secret to stdout"
```

The matched command contained the substring "token" as part of the ordinary English phrase "a fresh
access token before the retry loop" — no secret value, no environment variable, no credential of any
kind. The rule fired anyway, exactly as designed: it matches a keyword, not intent.

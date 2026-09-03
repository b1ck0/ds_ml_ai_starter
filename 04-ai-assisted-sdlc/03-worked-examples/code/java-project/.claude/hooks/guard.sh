#!/usr/bin/env bash
# PreToolUse(Bash): block dangerous or policy-violating shell commands. Exit non-zero to veto.
# Same deny-list shape as ds_ml_ai_starter's own .claude/hooks/guard.sh, plus a Java-specific
# rule: never skip the test phase to force a broken build to "succeed".
set -uo pipefail

payload="$(cat)"
cmd="$(printf '%s' "$payload" | grep -oE '"command"[[:space:]]*:[[:space:]]*"([^"\\]|\\.)*"' | head -1 | sed -E 's/^"command"[[:space:]]*:[[:space:]]*"//; s/"$//')"

deny() { echo "[guard] BLOCKED: $1" >&2; exit 2; }

# Destructive filesystem / git operations.
printf '%s' "$cmd" | grep -Eq 'rm[[:space:]]+-rf[[:space:]]+/([[:space:]]|$)' && deny "rm -rf /"
printf '%s' "$cmd" | grep -Eq 'git[[:space:]]+push[[:space:]].*--force([[:space:]]|$)' && deny "git push --force (use --force-with-lease and only when asked)"
printf '%s' "$cmd" | grep -Eq '(^|[[:space:];&|])(shutdown|reboot|mkfs|:\(\)\{)' && deny "system-level command"

# Policy: never print a secret to stdout.
printf '%s' "$cmd" | grep -Eiq 'echo[[:space:]].*(SECRET|API_KEY|TOKEN|PASSWORD|SERVICE_ROLE|PRIVATE_KEY)' && \
  deny "printing a secret to stdout"

# Project policy: don't let an agent talk itself past a failing gate by skipping tests.
printf '%s' "$cmd" | grep -Eq 'mvn[[:space:]].*-DskipTests' && \
  deny "mvn -DskipTests (the test-first gate exists precisely so this never happens silently)"

exit 0

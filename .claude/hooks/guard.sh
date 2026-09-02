#!/usr/bin/env bash
# PreToolUse(Bash): block dangerous or policy-violating shell commands. Exit non-zero to veto.
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

# Project policy: large datasets are downloaded by a documented step, not committed as blobs.
# (Advisory reminder only — not a hard block.)

exit 0

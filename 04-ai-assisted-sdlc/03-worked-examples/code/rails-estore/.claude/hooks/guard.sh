#!/usr/bin/env bash
# PreToolUse(Bash): block dangerous or policy-violating shell commands. Exit non-zero to veto.
# Same deny-list shape as ds_ml_ai_starter's own .claude/hooks/guard.sh and java-project's port,
# plus two Rails/e-commerce-specific rules a Java content-authoring project has no equivalent for:
# a real Stripe secret key must never appear in a command, and a git commit must never bypass hooks.
set -uo pipefail

payload="$(cat)"
cmd="$(printf '%s' "$payload" | grep -oE '"command"[[:space:]]*:[[:space:]]*"([^"\\]|\\.)*"' | head -1 | sed -E 's/^"command"[[:space:]]*:[[:space:]]*"//; s/"$//')"

deny() { echo "[guard] BLOCKED: $1" >&2; exit 2; }

# Destructive filesystem / git operations.
printf '%s' "$cmd" | grep -Eq 'rm[[:space:]]+-rf[[:space:]]+/([[:space:]]|$)' && deny "rm -rf /"
printf '%s' "$cmd" | grep -Eq 'git[[:space:]]+push[[:space:]].*--force([[:space:]]|$)' && deny "git push --force (use --force-with-lease and only when asked)"
printf '%s' "$cmd" | grep -Eq 'git[[:space:]]+commit[[:space:]].*--no-verify' && deny "git commit --no-verify (skips the RSpec/RuboCop/Brakeman pre-commit gate)"
printf '%s' "$cmd" | grep -Eq '(^|[[:space:];&|])(shutdown|reboot|mkfs|:\(\)\{)' && deny "system-level command"

# Policy: never print a secret to stdout.
printf '%s' "$cmd" | grep -Eiq 'echo[[:space:]].*(SECRET|API_KEY|TOKEN|PASSWORD|CREDENTIALS|MASTER_KEY)' && \
  deny "printing a secret to stdout"

# Rails/Stripe-specific: a LIVE key must never appear in a command (test keys start pk_test_/sk_test_
# and are fine — this pattern matches only the live prefixes). This is the guard that makes "no real
# secret ever appears in the repo" enforceable, not just a written rule.
printf '%s' "$cmd" | grep -Eq '(sk|pk|rk)_live_[A-Za-z0-9]' && \
  deny "a live Stripe key pattern (sk_live_/pk_live_/rk_live_) appeared in a command — use pk_test_/sk_test_ only, from .env, never inline"

# Destructive database operations against anything but the local test/dev db.
printf '%s' "$cmd" | grep -Eq 'RAILS_ENV=production[[:space:]]+.*rails[[:space:]]+db:(drop|reset)' && \
  deny "rails db:drop/db:reset against RAILS_ENV=production"

exit 0

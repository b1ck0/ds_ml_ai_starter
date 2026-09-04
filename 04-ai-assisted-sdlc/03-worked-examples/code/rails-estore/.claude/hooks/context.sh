#!/usr/bin/env bash
# SessionStart: orient the agent — charter, gates, feature specs, agent roster.
set -uo pipefail

echo "──────────────────────────────────────────────────────────"
echo " rails-estore — a governed SDLC scaffold for a Rails e-store"
echo "──────────────────────────────────────────────────────────"
echo " Read first: docs/architecture.md · docs/definition-of-done.md · CLAUDE.md"
echo " Roles: Opus (architect, main session) · .claude/agents/researcher.md (Haiku) ·"
echo "        .claude/agents/implementer.md (Sonnet) · .claude/agents/reviewer.md (fresh Sonnet)"
echo "        UI features also: .claude/agents/seo-optimizer.md · .claude/agents/frontend-qa.md (Sonnet)"
echo

if ls docs/features/FEATURE-*.md >/dev/null 2>&1; then
  echo " Feature specs:"
  for f in docs/features/FEATURE-*.md; do
    [ -e "$f" ] || continue
    title="$(grep -m1 '^# ' "$f" | sed 's/^# //')"
    status="$(grep -m1 -iE '^\*\*?status' "$f" | sed -E 's/.*:[[:space:]]*//; s/\*//g')"
    printf "   • %-58s %s\n" "$title" "${status:-(no status)}"
  done
else
  echo " No feature specs yet — write one in docs/features/ before touching app/."
fi

echo
echo " Gate reminders: failing RSpec example committed before the code it covers ·"
echo " bundle exec rspec passes · rubocop clean · brakeman -q --no-summary clean ·"
echo " no live Stripe key (sk_live_/pk_live_) anywhere · independent review before merge."
echo " (docs/definition-of-done.md)"
echo "──────────────────────────────────────────────────────────"

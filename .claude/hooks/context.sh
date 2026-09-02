#!/usr/bin/env bash
# SessionStart: orient the agent — framework entry points, active chapter specs, the gates.
set -uo pipefail

echo "──────────────────────────────────────────────────────────"
echo " ds_ml_ai_starter — a textbook for a senior Java dev new to Python/ML"
echo "──────────────────────────────────────────────────────────"
echo " Read first: docs/architecture.md · docs/curriculum.md · docs/style-guide.md · CLAUDE.md"
echo " Roles: Opus 4.8 scopes chapters · Sonnet 4.6 writes (.claude/agents/chapter-writer.md) ·"
echo "        Haiku grounds claims (.claude/agents/researcher.md) · fresh Sonnet reviews."
echo

if ls specs/SPEC-*.md >/dev/null 2>&1; then
  echo " Chapter specs:"
  for f in specs/SPEC-*.md; do
    [ -e "$f" ] || continue
    case "$f" in */SPEC-TEMPLATE.md) continue;; esac
    title="$(grep -m1 '^# ' "$f" | sed 's/^# //')"
    status="$(grep -m1 -iE '^\*\*?status' "$f" | sed -E 's/.*:[[:space:]]*//; s/\*//g')"
    printf "   • %-46s %s\n" "$title" "${status:-(no status)}"
  done
else
  echo " No chapter specs yet — scope one with .claude/skills/chapter-scoper before writing."
fi

echo
echo " Gate reminders: chapter matches its spec · every claim grounded (NOTE/citation) ·"
echo " every snippet runs · every link resolves · audience-fit · independent review before merge."
echo " (docs/definition-of-done.md)"
echo "──────────────────────────────────────────────────────────"

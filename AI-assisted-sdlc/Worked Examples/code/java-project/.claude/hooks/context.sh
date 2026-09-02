#!/usr/bin/env bash
# SessionStart: orient the agent — charter, gates, feature specs, agent roster.
set -uo pipefail

echo "──────────────────────────────────────────────────────────"
echo " java-project — a governed SDLC scaffold for a Java service"
echo "──────────────────────────────────────────────────────────"
echo " Read first: docs/architecture.md · docs/definition-of-done.md · CLAUDE.md"
echo " Roles: Opus (architect, main session) · .claude/agents/researcher.md (Haiku) ·"
echo "        .claude/agents/implementer.md (Sonnet) · .claude/agents/reviewer.md (fresh Sonnet)"
echo

if ls docs/features/FEATURE-*.md >/dev/null 2>&1; then
  echo " Feature specs:"
  for f in docs/features/FEATURE-*.md; do
    [ -e "$f" ] || continue
    title="$(grep -m1 '^# ' "$f" | sed 's/^# //')"
    status="$(grep -m1 -iE '^\*\*?status' "$f" | sed -E 's/.*:[[:space:]]*//; s/\*//g')"
    printf "   • %-52s %s\n" "$title" "${status:-(no status)}"
  done
else
  echo " No feature specs yet — write one in docs/features/ before touching src/."
fi

echo
echo " Gate reminders: failing JUnit 5 test committed before the code it covers ·"
echo " mvn clean test passes · checkstyle/spotless/spotbugs all clean ·"
echo " independent review before merge. (docs/definition-of-done.md)"
echo "──────────────────────────────────────────────────────────"

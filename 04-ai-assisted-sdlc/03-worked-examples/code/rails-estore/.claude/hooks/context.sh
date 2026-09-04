#!/usr/bin/env bash
# SessionStart: ориентира агента — харта, гейтове, feature spec-ове, състав от агенти.
set -uo pipefail

echo "──────────────────────────────────────────────────────────"
echo " rails-estore — управляван SDLC скелет за Rails онлайн магазин"
echo "──────────────────────────────────────────────────────────"
echo " Прочети първо: docs/architecture.md · docs/definition-of-done.md · CLAUDE.md"
echo " Роли: Opus (архитект, главна сесия) · .claude/agents/researcher.md (Haiku) ·"
echo "        .claude/agents/implementer.md (Sonnet) · .claude/agents/reviewer.md (свеж Sonnet)"
echo "        За UI функционалности също: .claude/agents/seo-optimizer.md · .claude/agents/frontend-qa.md (Sonnet)"
echo

if ls docs/features/FEATURE-*.md >/dev/null 2>&1; then
  echo " Feature spec-ове:"
  for f in docs/features/FEATURE-*.md; do
    [ -e "$f" ] || continue
    title="$(grep -m1 '^# ' "$f" | sed 's/^# //')"
    status="$(grep -m1 -iE '^\*\*?status' "$f" | sed -E 's/.*:[[:space:]]*//; s/\*//g')"
    printf "   • %-58s %s\n" "$title" "${status:-(без статус)}"
  done
else
  echo " Все още няма feature spec-ове — напиши един в docs/features/, преди да пипаш app/."
fi

echo
echo " Напомняния за гейта: провалящ се RSpec пример commit-нат преди кода, който покрива ·"
echo " bundle exec rspec минава · rubocop чист · brakeman -q --no-summary чист ·"
echo " никъде няма жив Stripe ключ (sk_live_/pk_live_) · независим review преди merge."
echo " (docs/definition-of-done.md)"
echo "──────────────────────────────────────────────────────────"

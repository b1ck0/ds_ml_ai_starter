#!/usr/bin/env bash
# PreToolUse(Bash): блокира опасни или нарушаващи политиката shell команди. Изход, различен от нула, вето-ва.
# Същата форма на deny-list като собствения .claude/hooks/guard.sh на ds_ml_ai_starter и порта на java-project,
# плюс две правила, специфични за Rails/е-търговия, за които Java content-authoring проект няма еквивалент:
# реален Stripe secret key никога не трябва да се появява в команда, и git commit никога не трябва да заобикаля hook-овете.
set -uo pipefail

payload="$(cat)"
cmd="$(printf '%s' "$payload" | grep -oE '"command"[[:space:]]*:[[:space:]]*"([^"\\]|\\.)*"' | head -1 | sed -E 's/^"command"[[:space:]]*:[[:space:]]*"//; s/"$//')"

deny() { echo "[guard] BLOCKED: $1" >&2; exit 2; }

# Деструктивни файлова система / git операции.
printf '%s' "$cmd" | grep -Eq 'rm[[:space:]]+-rf[[:space:]]+/([[:space:]]|$)' && deny "rm -rf /"
printf '%s' "$cmd" | grep -Eq 'git[[:space:]]+push[[:space:]].*--force([[:space:]]|$)' && deny "git push --force (използвай --force-with-lease и само когато е поискано)"
printf '%s' "$cmd" | grep -Eq 'git[[:space:]]+commit[[:space:]].*--no-verify' && deny "git commit --no-verify (пропуска pre-commit гейта на RSpec/RuboCop/Brakeman)"
printf '%s' "$cmd" | grep -Eq '(^|[[:space:];&|])(shutdown|reboot|mkfs|:\(\)\{)' && deny "команда на ниво система"

# Политика: никога не отпечатвай тайна в stdout.
printf '%s' "$cmd" | grep -Eiq 'echo[[:space:]].*(SECRET|API_KEY|TOKEN|PASSWORD|CREDENTIALS|MASTER_KEY)' && \
  deny "отпечатване на тайна в stdout"

# Специфично за Rails/Stripe: ЖИВ ключ никога не трябва да се появява в команда (тестовите ключове започват с pk_test_/sk_test_
# и са наред — този шаблон засича само живите префикси). Това е guard-ът, който прави "никаква реална
# тайна никога не се появява в хранилището" приложимо на практика, не само писано правило.
printf '%s' "$cmd" | grep -Eq '(sk|pk|rk)_live_[A-Za-z0-9]' && \
  deny "в команда се появи шаблон на жив Stripe ключ (sk_live_/pk_live_/rk_live_) — използвай само pk_test_/sk_test_, от .env, никога inline"

# Деструктивни операции с база данни срещу нещо различно от локалната test/dev база.
printf '%s' "$cmd" | grep -Eq 'RAILS_ENV=production[[:space:]]+.*rails[[:space:]]+db:(drop|reset)' && \
  deny "rails db:drop/db:reset срещу RAILS_ENV=production"

exit 0

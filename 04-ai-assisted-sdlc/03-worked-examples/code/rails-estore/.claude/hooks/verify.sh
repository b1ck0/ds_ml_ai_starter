#!/usr/bin/env bash
# PostToolUse(Edit|Write): бърз гейт за съдържанието на редактирания файл.
#   * app/**/*.rb, spec/**/*.rb -> bundle exec rspec, после rubocop, после brakeman
#   * app/views/**/*.erb        -> FRONTEND гейтът: системните спекове за axe + SEO, после html-proofer
#                                   (SPEC-SDLC-4-ADDENDUM-seo-frontend-qa-agents — агентите seo-optimizer
#                                   и frontend-qa карат ръчно същите инструменти по време на review;
#                                   този hook просто ги пуска бързо, при всяка редакция на view)
#   * Gemfile                   -> bundle check (улавя веднага нерешим/счупен Gemfile,
#                                   преди следващото пускане на пълния гейт изобщо да опита нещо)
# Чете hook payload-а на stdin, за да намери редактирания файл. Изход, различен от нула, сигнализира провалящ се гейт.
# POSIX bash; на Windows се пуска под Git Bash. Прескача елегантно, когато bundle липсва.
set -uo pipefail

payload="$(cat)"
file="$(printf '%s' "$payload" | grep -oE '"file_path"[[:space:]]*:[[:space:]]*"[^"]+"' | head -1 | sed -E 's/.*:"([^"]+)"/\1/')"

# Нормализирай Windows обратните наклонени черти, за да намират инструментите файла.
file="${file//\\//}"

BUNDLE="$(command -v bundle || true)"
fail=0
run() { echo "  \$ $*"; "$@" || fail=1; }

case "$file" in
  *.rb)
    if [ -n "$BUNDLE" ]; then
      echo "[verify] ruby гейт: $file"
      run "$BUNDLE" exec rspec
      run "$BUNDLE" exec rubocop
      run "$BUNDLE" exec brakeman -q --no-summary
    else
      echo "[verify] bundle не е на PATH — прескача се гейтът за $file"
    fi
    ;;
  *.erb)
    if [ -n "$BUNDLE" ]; then
      echo "[verify] frontend гейт: $file"
      run "$BUNDLE" exec rspec spec/system/accessibility_spec.rb spec/system/seo_spec.rb
      run "$BUNDLE" exec rake html_proofer:check
      echo "  (само референтно, не се пуска от този hook -- Node toolchain: npx lhci autorun)"
    else
      echo "[verify] bundle не е на PATH — прескача се frontend гейтът за $file"
    fi
    ;;
  *Gemfile)
    if [ -n "$BUNDLE" ]; then
      echo "[verify] Gemfile е променен — проверка"
      run "$BUNDLE" check
    else
      echo "[verify] bundle не е на PATH — прескача се проверката"
    fi
    ;;
  *)
    echo "[verify] няма проверки за: ${file:-<unknown>}" ;;
esac

exit $fail

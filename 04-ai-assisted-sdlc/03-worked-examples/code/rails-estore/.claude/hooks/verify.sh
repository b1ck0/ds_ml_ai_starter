#!/usr/bin/env bash
# PostToolUse(Edit|Write): fast content gate for the edited file.
#   * app/**/*.rb, spec/**/*.rb -> bundle exec rspec, then rubocop, then brakeman
#   * Gemfile                   -> bundle check (catches an unresolvable/broken Gemfile immediately,
#                                   before the next full gate run bothers running anything)
# Reads the hook payload on stdin to find the edited file. Non-zero exit signals a failing gate.
# POSIX bash; on Windows it runs under Git Bash. Skips gracefully when bundle is absent.
set -uo pipefail

payload="$(cat)"
file="$(printf '%s' "$payload" | grep -oE '"file_path"[[:space:]]*:[[:space:]]*"[^"]+"' | head -1 | sed -E 's/.*:"([^"]+)"/\1/')"

# Normalise Windows backslashes so tools find the file.
file="${file//\\//}"

BUNDLE="$(command -v bundle || true)"
fail=0
run() { echo "  \$ $*"; "$@" || fail=1; }

case "$file" in
  *.rb)
    if [ -n "$BUNDLE" ]; then
      echo "[verify] ruby gate: $file"
      run "$BUNDLE" exec rspec
      run "$BUNDLE" exec rubocop
      run "$BUNDLE" exec brakeman -q --no-summary
    else
      echo "[verify] bundle not on PATH — skipping gate for $file"
    fi
    ;;
  *Gemfile)
    if [ -n "$BUNDLE" ]; then
      echo "[verify] Gemfile changed — checking"
      run "$BUNDLE" check
    else
      echo "[verify] bundle not on PATH — skipping check"
    fi
    ;;
  *)
    echo "[verify] no checks for: ${file:-<unknown>}" ;;
esac

exit $fail

#!/usr/bin/env bash
# PostToolUse(Edit|Write): fast content gate for the edited file.
#   * .py  -> byte-compile (and ruff if available)
#   * .md  -> compile the fenced ```python snippets via check_snippets.py
# Reads the hook payload on stdin to find the edited file. Non-zero exit signals a failing gate.
# POSIX bash; on Windows it runs under Git Bash. Skips gracefully when a tool is absent.
set -uo pipefail

payload="$(cat)"
file="$(printf '%s' "$payload" | grep -oE '"file_path"[[:space:]]*:[[:space:]]*"[^"]+"' | head -1 | sed -E 's/.*:"([^"]+)"/\1/')"

# Normalise Windows backslashes so tools find the file.
file="${file//\\//}"

PY="$(command -v python || command -v python3 || true)"
fail=0
run() { echo "  \$ $*"; "$@" || fail=1; }

case "$file" in
  *.py)
    if [ -n "$PY" ]; then
      echo "[verify] python snippet: $file"
      run "$PY" -m py_compile "$file"
      command -v ruff >/dev/null 2>&1 && run ruff check "$file"
    else
      echo "[verify] python not on PATH — skipping compile of $file"
    fi
    ;;
  *.md)
    if [ -n "$PY" ] && [ -f ".claude/hooks/check_snippets.py" ]; then
      echo "[verify] markdown snippets: $file"
      run "$PY" .claude/hooks/check_snippets.py "$file"
    else
      echo "[verify] snippet checker unavailable — skipping $file"
    fi
    ;;
  *)
    echo "[verify] no checks for: ${file:-<unknown>}" ;;
esac

exit $fail

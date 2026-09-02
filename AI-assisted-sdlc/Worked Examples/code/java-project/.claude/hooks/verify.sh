#!/usr/bin/env bash
# PostToolUse(Edit|Write): fast content gate for the edited file.
#   * .java      -> mvn -q test (compiles + runs the JUnit 5 suite), then checkstyle/spotless/spotbugs
#   * pom.xml    -> mvn -q validate (catches a broken POM immediately, before the next full gate run)
# Reads the hook payload on stdin to find the edited file. Non-zero exit signals a failing gate.
# POSIX bash; on Windows it runs under Git Bash. Skips gracefully when mvn is absent.
set -uo pipefail

payload="$(cat)"
file="$(printf '%s' "$payload" | grep -oE '"file_path"[[:space:]]*:[[:space:]]*"[^"]+"' | head -1 | sed -E 's/.*:"([^"]+)"/\1/')"

# Normalise Windows backslashes so tools find the file.
file="${file//\\//}"

MVN="$(command -v mvn || true)"
fail=0
run() { echo "  \$ $*"; "$@" || fail=1; }

case "$file" in
  *.java)
    if [ -n "$MVN" ]; then
      echo "[verify] java gate: $file"
      run "$MVN" -q test
      run "$MVN" -q checkstyle:check
      run "$MVN" -q spotless:check
      run "$MVN" -q spotbugs:check
    else
      echo "[verify] mvn not on PATH — skipping gate for $file"
    fi
    ;;
  *pom.xml)
    if [ -n "$MVN" ]; then
      echo "[verify] pom.xml changed — validating"
      run "$MVN" -q validate
    else
      echo "[verify] mvn not on PATH — skipping validate"
    fi
    ;;
  *)
    echo "[verify] no checks for: ${file:-<unknown>}" ;;
esac

exit $fail

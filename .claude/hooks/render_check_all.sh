#!/usr/bin/env bash
# Run the GitHub-render lint (check_markdown_render.py) over EVERY tracked Markdown file.
# Wired into git as a pre-push hook (see .git/hooks/pre-push) so a push that would ship a
# broken formula/diagram is blocked. Also runnable by hand:  bash .claude/hooks/render_check_all.sh
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 0
PY="$(command -v python || command -v python3 || true)"
if [ -z "$PY" ]; then echo "[render-all] python not found — skipping"; exit 0; fi
mapfile -t md < <(git ls-files '*.md')
if [ ${#md[@]} -eq 0 ]; then exit 0; fi
"$PY" .claude/hooks/check_markdown_render.py "${md[@]}"

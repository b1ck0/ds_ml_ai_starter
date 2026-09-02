#!/usr/bin/env python3
"""Extract fenced ```python code blocks from a Markdown chapter and byte-compile each.

Part of the content gate: every runnable snippet in a chapter must at least parse/compile.
A block tagged ```python-pseudocode (or ```text) is skipped. Compilation is not execution —
the writer/reviewer still runs the examples — but this catches broken snippets cheaply and
cross-platform (Windows/macOS/Linux).

Usage:  python .claude/hooks/check_snippets.py <file.md> [<file2.md> ...]
Exit:   0 = all python blocks compiled;  1 = at least one failed to compile.
"""
from __future__ import annotations

import re
import sys

FENCE = re.compile(
    r"^[ \t]*```(?P<lang>[^\n`]*)\n(?P<code>.*?)^[ \t]*```",
    re.DOTALL | re.MULTILINE,
)
# Languages we try to compile. Explicitly-pseudocode fences are skipped.
PY_LANGS = {"python", "py", "python3"}
SKIP_HINTS = {"pseudocode", "no-run", "norun", "text", "console", "output"}


def check_markdown(path: str) -> int:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            src = fh.read()
    except OSError as exc:
        print(f"[snippets] cannot read {path}: {exc}")
        return 0  # nothing to check

    failures = 0
    for i, m in enumerate(FENCE.finditer(src), start=1):
        lang = m.group("lang").strip().lower()
        if lang not in PY_LANGS or any(h in lang for h in SKIP_HINTS):
            continue
        code = m.group("code")
        line_no = src[: m.start()].count("\n") + 1
        try:
            compile(code, f"{path}:block{i}@L{line_no}", "exec")
        except SyntaxError as exc:
            failures += 1
            print(f"[snippets] SYNTAX ERROR {path} block #{i} (near line {line_no}): {exc.msg} "
                  f"at line {exc.lineno}")
    if failures == 0:
        print(f"[snippets] OK: all python blocks compiled in {path}")
    return 1 if failures else 0


def main(argv: list[str]) -> int:
    md_files = [a for a in argv[1:] if a.lower().endswith(".md")]
    if not md_files:
        print("[snippets] no .md files given — nothing to check")
        return 0
    rc = 0
    for path in md_files:
        rc |= check_markdown(path)
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

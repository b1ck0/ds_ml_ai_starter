#!/usr/bin/env python3
"""Lint a Markdown chapter for GitHub RENDERING bugs that a Python compile check misses.

GitHub renders LaTeX with MathJax and Mermaid natively. Two whole classes of defect never
touch the Python snippet gate yet dump raw source (or an error box) on the rendered page:

  1. LaTeX in `$...$` / `$$...$$` / ```math that MathJax rejects — most commonly an UNESCAPED
     special char (`_ ^ # % & ~`) inside a `\\text{...}` / `\\mathrm{...}` / `\\operatorname{...}`
     run. MathJax then prints:  "'_' allowed only in math mode"  and the formula renders as raw
     TeX. (NOTE: KaTeX renders these without complaint, so a KaTeX check does NOT catch this —
     this rule targets MathJax's actual failure mode.) Also flags unbalanced `$$` / inline `$`.
  2. Mermaid blocks that fail to render: a bad start keyword, or a node label `[...]` / `{...}`
     containing an unquoted parenthesis (must be wrapped in "double quotes").

Usage:  python .claude/hooks/check_markdown_render.py <file.md> [file2.md ...]
Exit:   0 = clean; 1 = at least one rendering defect found.
Cross-platform, no third-party dependencies (stdlib only).
"""
from __future__ import annotations

import re
import sys

FENCE = re.compile(r"^([ \t]*)(`{3,}|~{3,})[ \t]*([^\s`~]*)", re.MULTILINE)
TEXT_RUN = re.compile(r"\\(?:text|mathrm|mathbf|mathit|mathsf|mathtt|operatorname|textbf|textit|textrm)\s*\{([^{}]*)\}")
# a special char that MathJax needs escaped inside a text-run, when NOT already backslash-escaped
UNESCAPED_SPECIAL = re.compile(r"(?<!\\)[_^#%&~]")
MERMAID_START = ("flowchart", "graph", "sequenceDiagram", "timeline", "journey", "stateDiagram",
                 "mindmap", "classDiagram", "erDiagram", "gantt", "pie", "quadrantChart",
                 "gitGraph", "block-beta", "requirementDiagram", "C4Context")
NODE_LABEL = re.compile(r"\[([^\[\]]*)\]|\{([^{}]*)\}")


def split_blocks(md: str):
    """Yield (kind, content, start_line) for prose, mermaid, and math-fence blocks.

    Fenced code blocks are separated out so `$` inside code and diagram syntax are not
    mistaken for math. ```mermaid and ```math fences are yielded with their kind; every
    other fenced block is dropped (it is code, not rendered markdown).
    """
    lines = md.split("\n")
    i = 0
    in_fence = False
    info = ""
    buf: list[str] = []
    start = 0
    prose: list[tuple[int, str]] = []
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^[ \t]*(`{3,}|~{3,})[ \t]*([^\s`~]*)", line)
        if m:
            if not in_fence:
                in_fence, info, buf, start = True, (m.group(2) or "").lower(), [], i + 1
            else:
                if info in ("mermaid", "math"):
                    yield (info, "\n".join(buf), start)
                in_fence, info = False, ""
            i += 1
            continue
        if in_fence:
            buf.append(line)
        else:
            # blank inline `code` so a `$5` in prose is not read as math
            prose.append((i + 1, re.sub(r"`[^`]*`", lambda x: " " * len(x.group()), line)))
        i += 1
    yield ("prose", "\n".join(t for _, t in prose), 1)


def check_math(tex: str, file: str, line: int, issues: list):
    for m in TEXT_RUN.finditer(tex):
        run = m.group(1)
        if UNESCAPED_SPECIAL.search(run):
            bad = UNESCAPED_SPECIAL.search(run).group()
            issues.append(f"{file}:{line}  LaTeX: unescaped '{bad}' inside a \\text-run "
                          f"(MathJax rejects it -- write '\\{bad}'): \\text{{{run}}}")


def check_markdown(path: str, issues: list):
    try:
        md = open(path, encoding="utf-8").read()
    except OSError as e:
        print(f"[render] cannot read {path}: {e}")
        return
    prose_text = ""
    for kind, content, line in split_blocks(md):
        if kind == "math":
            check_math(content, path, line, issues)
        elif kind == "prose":
            prose_text = content
        elif kind == "mermaid":
            body = [l for l in content.splitlines() if l.strip()]
            first = body[0].strip() if body else ""
            if not any(first.startswith(k) for k in MERMAID_START):
                issues.append(f"{path}:{line}  Mermaid: unrecognised start keyword {first!r}")
            for lm in NODE_LABEL.finditer(content):
                label = next(g for g in lm.groups() if g is not None).strip()
                # Skip shape-wrappers ([(cyl)], (["stadium"]), etc.) and already-quoted labels —
                # those parens/quotes are Mermaid syntax, not text that needs quoting.
                if label and label[0] not in '"([{' and (")" in label or "(" in label):
                    issues.append(f"{path}:{line}  Mermaid: unquoted '(' or ')' in node label "
                                  f"(wrap the label in \"...\"): {label[:60]}")
    # LaTeX text-runs inside DISPLAY math ($$...$$) and inline math ($...$).
    no_display = prose_text
    for m in re.finditer(r"\$\$([\s\S]+?)\$\$", prose_text):
        check_math(m.group(1), path, 0, issues)
        no_display = no_display.replace(m.group(0), " " * len(m.group(0)), 1)
    for m in re.finditer(r"(?<!\\)\$(?!\$)([^\n$]+?)\$(?!\$)", no_display):
        check_math(m.group(1), path, 0, issues)
    if prose_text.count("$$") % 2:
        issues.append(f"{path}  LaTeX: odd number of '$$' delimiters (an unclosed display-math block)")


def main(argv: list[str]) -> int:
    files = [a for a in argv[1:] if a.lower().endswith(".md")]
    if not files:
        print("[render] no .md files given")
        return 0
    issues: list[str] = []
    for f in files:
        check_markdown(f, issues)
    if issues:
        for x in issues:
            print("  " + x)
        print(f"[render] {len(issues)} rendering issue(s) across {len(files)} file(s)")
        return 1
    print(f"[render] OK: no LaTeX/Mermaid rendering issues in {len(files)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

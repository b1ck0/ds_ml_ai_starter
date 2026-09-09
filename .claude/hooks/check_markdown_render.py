#!/usr/bin/env python3
r"""Lint a Markdown chapter for GitHub RENDERING bugs a Python compile check never sees.

GitHub renders math with MathJax, but it runs its Markdown/HTML processor over `$...$` and
`$$...$$` FIRST — and that processor corrupts common LaTeX before MathJax ever sees it. A
```math fenced block is NOT markdown-processed, so it is the only reliable home for anything
tricky. This lint encodes the failure modes seen live on GitHub:

  1. ESCAPE EATEN.  Inside `$...$`/`$$...$$`, a markdown backslash-escape (`\_ \^ \# \% \& \|`)
     has its backslash stripped by Markdown, so `\text{one\_hot}` reaches MathJax as
     `\text{one_hot}` -> red error "'_' allowed only in math mode". Fix: put it in a ```math block.
  2. `\\` / MATRICES in `$...$`.  Markdown mangles the `\\` row separators (worst in a table
     cell), so `$\begin{bmatrix}1&2\\3&4\end{bmatrix}$` renders as raw text. Fix: a ```math block.
  3. `<` FOLLOWED BY A LETTER.  GitHub's HTML sanitiser reads `x_{<i}` 's `<i` as an `<i>` tag and
     shreds the math ("Extra open brace"). Fix: `\lt ` or a space, or a ```math block.
  4. INLINE MATH SPLIT ACROSS A LINE BREAK.  GitHub renders inline `$...$` only within ONE line;
     prose reflow that wraps a formula leaves the raw `$...` showing.
  5. `\text{}` with an unescaped `_ ^ # % & ~`; unbalanced `{ }`; unbalanced `$$`.
  6. Unsupported everywhere: the `\( \) \[ \]` delimiters; the `align`/`equation`/`gather`/
     `multline` environments (use `aligned` in a ```math block); user macros (`\newcommand`/`\def`).
  7. Mermaid: a bad start keyword, or a `[...]`/`{...}` node label with an unquoted parenthesis.

Usage:  python .claude/hooks/check_markdown_render.py <file.md> [file2.md ...]
Exit:   0 = clean; 1 = at least one rendering defect found.  stdlib only, cross-platform.
"""
from __future__ import annotations

import re
import sys

FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})[ \t]*([^\s`~]*)")
TEXT_RUN = re.compile(r"\\(?:text|mathrm|mathbf|mathit|mathsf|mathtt|operatorname|textbf|textit|textrm)\s*\{([^{}]*)\}")
UNESCAPED_SPECIAL = re.compile(r"(?<!\\)[_^#%&~]")
LATEX_CMD = re.compile(r"\\[A-Za-z]{2,}")
# backslash-escapes markdown itself consumes inside $-math (so `\_`->`_` reaches MathJax broken).
# `\%` `\&` are NOT markdown-escapable, so they survive fine -- do not flag them.
ESCAPED_MD_SPECIAL = re.compile(r"\\[_#{}~]")
HTML_TAGISH = re.compile(r"<[A-Za-z/]")
MATRIX_ENV = re.compile(r"\\begin\{[a-zA-Z]*matrix\}")
BAD_ENV = re.compile(r"\\begin\{(align|equation|gather|multline)\*?\}")
BAD_MACRO = re.compile(r"\\(?:newcommand|renewcommand|def|DeclareMathOperator)\b")
BAD_DELIM = re.compile(r"\\[()\[\]]")
MERMAID_START = ("flowchart", "graph", "sequenceDiagram", "timeline", "journey", "stateDiagram",
                 "mindmap", "classDiagram", "erDiagram", "gantt", "pie", "quadrantChart",
                 "gitGraph", "block-beta", "requirementDiagram", "C4Context")
NODE_LABEL = re.compile(r"\[([^\[\]]*)\]|\{([^{}]*)\}")


def check_span(tex: str, kind: str, file: str, line, issues: list):
    """Validate one math span. kind is 'mathfence' | 'display' | 'inline'.

    mathfence (```math) is markdown-safe; display/inline ($$.. / $..) are markdown-processed and
    so also fail on eaten escapes, mangled `\\`, and matrices.
    """
    for m in TEXT_RUN.finditer(tex):
        bad = UNESCAPED_SPECIAL.search(m.group(1))
        if bad:
            issues.append(f"{file}:{line}  math: unescaped '{bad.group()}' inside \\text{{{m.group(1)}}} "
                          f"(MathJax: \"'{bad.group()}' allowed only in math mode\") -- escape it AND put "
                          f"the equation in a ```math block")
    st = re.sub(r"\\[{}]", "", tex)
    if st.count("{") != st.count("}"):
        issues.append(f"{file}:{line}  math: unbalanced '{{'/'}}' (MathJax hard error): {tex.strip()[:70]}")
    hm = HTML_TAGISH.search(tex)
    if hm:
        issues.append(f"{file}:{line}  math: '{tex[hm.start():hm.start()+3]}' -- GitHub reads '<' before a "
                      f"letter as an HTML tag and breaks the math; write '\\lt ' (or add a space): {tex.strip()[:60]}")
    if kind in ("display", "inline"):
        em = ESCAPED_MD_SPECIAL.search(tex)
        if em:
            issues.append(f"{file}:{line}  math: '{em.group()}' inside {'$$' if kind=='display' else '$'}-math -- "
                          f"GitHub Markdown strips the backslash before MathJax; move the equation into a "
                          f"```math fenced block: {tex.strip()[:60]}")
        if kind == "inline" and ("\\\\" in tex or MATRIX_ENV.search(tex)):
            issues.append(f"{file}:{line}  math: '\\\\'/matrix inside inline $...$ renders as raw text on "
                          f"GitHub (Markdown mangles the '\\\\' row breaks, worst in a table cell) -- put it in "
                          f"a ```math fenced block: {tex.strip()[:60]}")


def inline_spans(s: str):
    """Yield (open, close) of valid GitHub inline `$...$` spans on one line."""
    i, n = 0, len(s)
    while i < n:
        if s[i] == "$" and (i == 0 or s[i - 1] not in "$\\") and i + 1 < n and not s[i + 1].isspace() and s[i + 1] != "$":
            j = i + 1
            while j < n:
                if s[j] == "$" and s[j - 1] not in " \t\\" and (j + 1 >= n or (not s[j + 1].isdigit() and s[j + 1] != "$")):
                    break
                j += 1
            if j < n and s[j] == "$":
                yield (i, j)
                i = j + 1
                continue
        i += 1


def check_markdown(path: str, issues: list):
    try:
        md = open(path, encoding="utf-8").read()
    except OSError as e:
        print(f"[render] cannot read {path}: {e}")
        return
    lines = md.split("\n")

    # Pass 1: walk fences; collect ```math / ```mermaid; gather prose lines (with inline `code` blanked).
    prose: list[tuple[int, str]] = []
    i, in_fence, info, buf, fstart = 0, False, "", [], 0
    while i < len(lines):
        m = FENCE.match(lines[i])
        if m:
            if not in_fence:
                in_fence, info, buf, fstart = True, (m.group(2) or "").lower(), [], i + 2
            else:
                if info == "math":
                    check_span("\n".join(buf), "mathfence", path, fstart, issues)
                elif info == "mermaid":
                    body = [l for l in buf if l.strip()]
                    first = body[0].strip() if body else ""
                    if not any(first.startswith(k) for k in MERMAID_START):
                        issues.append(f"{path}:{fstart}  Mermaid: unrecognised start keyword {first!r}")
                    for lm in NODE_LABEL.finditer("\n".join(buf)):
                        label = next(g for g in lm.groups() if g is not None).strip()
                        if label and label[0] not in '"([{' and ("(" in label or ")" in label):
                            issues.append(f"{path}:{fstart}  Mermaid: unquoted '(' or ')' in node label "
                                          f"(wrap in \"...\"): {label[:60]}")
                in_fence = False
            i += 1
            continue
        if in_fence:
            buf.append(lines[i])
        else:
            prose.append((i + 1, re.sub(r"`[^`]*`", lambda x: " " * len(x.group()), lines[i])))
        i += 1

    # Pass 2: display $$...$$ across the prose blob (line number from the newline prefix).
    text = "\n".join(t for _, t in prose)
    linenos = [ln for ln, _ in prose]

    def lineno_at(pos):
        return linenos[text.count("\n", 0, pos)] if linenos else 0

    masked = list(text)
    for m in re.finditer(r"\$\$([\s\S]+?)\$\$", text):
        check_span(m.group(1), "display", path, lineno_at(m.start()), issues)
        for k in range(m.start(), m.end()):
            if masked[k] != "\n":
                masked[k] = "\x02"
    text_wo_display = "".join(masked)

    # Pass 3: inline $...$ per line; unpaired $ adjacent to LaTeX => split-across-lines.
    disp_lines = text_wo_display.split("\n")
    for idx, (ln, _) in enumerate(prose):
        line = disp_lines[idx] if idx < len(disp_lines) else ""
        if BAD_DELIM.search(line):
            issues.append(f"{path}:{ln}  LaTeX delimiter \\( \\) \\[ \\] not rendered by GitHub "
                          f"(use $...$ or a ```math block): {line.strip()[:70]}")
        if BAD_ENV.search(line):
            issues.append(f"{path}:{ln}  '{BAD_ENV.search(line).group(1)}' environment unsupported by "
                          f"GitHub (use \\begin{{aligned}} in a ```math block): {line.strip()[:60]}")
        if BAD_MACRO.search(line):
            issues.append(f"{path}:{ln}  user macro {BAD_MACRO.search(line).group()} unsupported by GitHub MathJax")
        s = line.replace("\\$", "\x00")
        chars = list(s)
        for a, b in inline_spans(s):
            check_span(s[a + 1:b], "inline", path, ln, issues)
            for k in range(a, b + 1):
                chars[k] = "\x01"
        has_cmd = bool(LATEX_CMD.search(s))
        for k, c in enumerate(chars):
            if c != "$":
                continue
            after = chars[k + 1] if k + 1 < len(chars) else ""
            if after in ("\\", "{", "^", "_"):
                issues.append(f"{path}:{ln}  inline '$' opens math but never closes on this line -- GitHub "
                              f"renders '$...$' within ONE line; a formula was split across a line break: "
                              f"...{line.strip()[:75]}")
            elif has_cmd and after and not after.isdigit() and after not in (" ", "\t", "."):
                issues.append(f"{path}:{ln}  unbalanced inline '$' on a line with LaTeX (split '$...$' across a "
                              f"line break, or a stray '$'): ...{line.strip()[:75]}")
            elif has_cmd and after == "":
                issues.append(f"{path}:{ln}  dangling inline '$' ending a line with LaTeX (the other half of a "
                              f"split '$...$' is on an adjacent line): ...{line.strip()[-75:]}")

    if text.count("$$") % 2:   # count in prose only (code fences dropped, inline `code` blanked)
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

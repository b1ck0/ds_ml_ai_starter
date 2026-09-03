"""Validate .claude/agents/*.md frontmatter against the schema documented in
research/NOTE-SDLC-2-claude-code.md and research/NOTE-SDLC-3-java-gates.md (official Claude Code
sub-agent docs, https://code.claude.com/docs/en/sub-agents, verified 2026-09-02 and re-checked
2026-09-03 while writing this chapter): a `name` + `description` field are required, every other
frontmatter key must come from the documented optional-field set, `model` (if present) must be one
of the documented values, and `name` must be lowercase-plus-hyphens.

This is the Java-project equivalent of this book's own snippet gate
(.claude/hooks/check_snippets.py) applied to agent files instead of markdown code fences — cheap,
deterministic, and exactly what a `PostToolUse` hook on `.claude/agents/*.md` would run in a real
project (see java-sdlc-scaffold.md SS4 "what carries over").

Environment: PyYAML==6.0.3 (installed in this project's .venv; verify with
`.venv/Scripts/python.exe -c "import yaml; print(yaml.__version__)"`), Python 3.12+.

Usage:
    .venv/Scripts/python.exe "AI-assisted-sdlc/Worked Examples/code/validate_frontmatter.py" \
        "AI-assisted-sdlc/Worked Examples/code/java-project/.claude/agents/researcher.md" \
        "AI-assisted-sdlc/Worked Examples/code/java-project/.claude/agents/implementer.md" \
        "AI-assisted-sdlc/Worked Examples/code/java-project/.claude/agents/reviewer.md"

Exit: 0 if every given file's frontmatter is well-formed and schema-valid; 1 otherwise.
"""
from __future__ import annotations

import re
import sys

import yaml

REQUIRED = {"name", "description"}
KNOWN_OPTIONAL = {
    "model", "tools", "disallowedTools", "permissionMode", "maxTurns", "skills",
    "mcpServers", "hooks", "memory", "background", "effort", "isolation", "color",
    "initialPrompt", "experimental",
}
VALID_MODELS = {"sonnet", "opus", "haiku", "fable", "inherit"}

FENCE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def validate(path: str) -> bool:
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()

    match = FENCE.match(text)
    if not match:
        print(f"FAIL {path}: no YAML frontmatter block found")
        return False

    try:
        data = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        print(f"FAIL {path}: YAML parse error: {exc}")
        return False

    if not isinstance(data, dict):
        print(f"FAIL {path}: frontmatter did not parse to a mapping")
        return False

    missing = REQUIRED - data.keys()
    if missing:
        print(f"FAIL {path}: missing required field(s): {missing}")
        return False

    unknown = set(data.keys()) - REQUIRED - KNOWN_OPTIONAL
    if unknown:
        print(f"FAIL {path}: undocumented frontmatter field(s): {unknown}")
        return False

    if "model" in data and data["model"] not in VALID_MODELS and not str(
        data["model"]
    ).startswith("claude-"):
        print(f"FAIL {path}: model {data['model']!r} not in documented set {VALID_MODELS}")
        return False

    if not re.fullmatch(r"[a-z][a-z0-9-]*", data["name"]):
        print(f"FAIL {path}: name {data['name']!r} is not lowercase+hyphens")
        return False

    print(
        f"OK   {path}: name={data['name']!r} model={data.get('model')!r} "
        f"fields={sorted(data.keys())}"
    )
    return True


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: validate_frontmatter.py <agent.md> [<agent2.md> ...]")
        return 1
    ok = True
    for path in argv[1:]:
        ok &= validate(path)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

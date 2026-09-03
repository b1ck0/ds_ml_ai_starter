"""Render the steering-file map for SPEC-SDLC-0 (Local Environment Setup).

Draws CLAUDE.md as the hub of the four `.claude/` artefacts that steer Claude Code in THIS
repository -- agents, skills, hooks, and settings.json -- each satellite box listing the actual
files that play that role here, plus the one distinction from NOTE-SDLC-2 that most surprises a
newcomer: CLAUDE.md is *instructions* (Claude reads and tries to follow it, but nothing enforces
it), while settings.json's permission rules are *enforced* by Claude Code itself before every tool
call. SPEC-SDLC-2 goes deep on authoring these; this diagram is only the map, so LO4's reader can
orient before that chapter.

Grounded in research/NOTE-SDLC-2-claude-code.md (official Claude Code docs -- code.claude.com/docs,
verified 2026-09-02): agent/skill/hook frontmatter schemas, the Deny > Ask > Allow permission
precedence, and the quoted line "Instructions in your prompt or CLAUDE.md shape what Claude tries
to do, but they don't change what Claude Code allows." (code.claude.com/docs/en/permissions).
Concrete repo pointers (CLAUDE.md, .claude/agents/*.md, .claude/skills/*, .claude/hooks/*.sh,
.claude/settings.json) are this repository's own files, read directly.

Environment: matplotlib==3.11.1 (research/NOTE-2-package-versions.md, verified against PyPI
2026-09-02; confirmed live against this project's installed .venv interpreter), Python 3.12+.
Run with the project .venv:

    .venv/Scripts/python.exe "AI-assisted-sdlc/Local Environment Setup/code/steering_file_map.py"

Writes: ../artefacts/steering_file_map.png
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HUB_COLOR = "#1B2A4A"
HUB_FACE = "#4C72B0"
AGENTS_COLOR = "#8172B2"
SKILLS_COLOR = "#55A868"
HOOKS_COLOR = "#C44E52"
SETTINGS_COLOR = "#DD8452"
BOX_FACE = "#EAF1FB"

# The four `.claude/` satellites around CLAUDE.md, each with the real files this repository uses
# and a one-line Java-world analogy (docs/style-guide.md: "bridge from Java/JVM mental models").
SATELLITES = [
    {
        "title": ".claude/agents/",
        "color": AGENTS_COLOR,
        "pos": (2.6, 7.4),
        "files": [
            "chapter-writer.md (Sonnet)",
            "researcher.md (Haiku)",
            "chapter-reviewer.md (fresh Sonnet)",
        ],
        "analogy": "like DI-wired worker\nbeans -- charter + tools + model",
    },
    {
        "title": ".claude/skills/",
        "color": SKILLS_COLOR,
        "pos": (10.4, 7.4),
        "files": [
            "chapter-scoper/",
            "research-brief/",
        ],
        "analogy": "a packaged runbook --\ninvoked by name, not always-on",
    },
    {
        "title": ".claude/hooks/",
        "color": HOOKS_COLOR,
        "pos": (2.6, 1.6),
        "files": [
            "guard.sh (PreToolUse: Bash)",
            "verify.sh (PostToolUse: Edit|Write)",
            "context.sh (SessionStart)",
        ],
        "analogy": "git hooks, but for every\ntool call -- pre-commit / post-merge",
    },
    {
        "title": ".claude/settings.json",
        "color": SETTINGS_COLOR,
        "pos": (10.4, 1.6),
        "files": [
            "wires hooks to lifecycle events",
            "permission rules: Deny > Ask > Allow",
            "ENFORCED by Claude Code itself",
        ],
        "analogy": "an ACL / security policy file,\nnot a suggestion",
    },
]

BOX_W, BOX_H = 4.3, 2.0
HUB_W, HUB_H = 3.0, 1.5
HUB_POS = (6.5, 4.5)


def _draw_hub(ax) -> None:
    cx, cy = HUB_POS
    box = FancyBboxPatch(
        (cx - HUB_W / 2, cy - HUB_H / 2), HUB_W, HUB_H,
        boxstyle="round,pad=0.08,rounding_size=0.12",
        linewidth=2.2, edgecolor=HUB_COLOR, facecolor=HUB_FACE,
    )
    ax.add_patch(box)
    ax.text(cx, cy + 0.32, "CLAUDE.md", ha="center", va="center",
             fontsize=14, fontweight="bold", color="white")
    ax.text(cx, cy - 0.05, "persistent charter -- INSTRUCTIONS,", ha="center", va="center",
             fontsize=8.3, color="white")
    ax.text(cx, cy - 0.32, "read every session, not enforced", ha="center", va="center",
             fontsize=8.3, color="white")


def _draw_satellite(ax, sat: dict) -> None:
    cx, cy = sat["pos"]
    box = FancyBboxPatch(
        (cx - BOX_W / 2, cy - BOX_H / 2), BOX_W, BOX_H,
        boxstyle="round,pad=0.07,rounding_size=0.1",
        linewidth=1.8, edgecolor=sat["color"], facecolor=BOX_FACE,
    )
    ax.add_patch(box)
    ax.text(cx, cy + BOX_H / 2 - 0.28, sat["title"], ha="center", va="center",
             fontsize=12, fontweight="bold", color=sat["color"])
    y = cy + BOX_H / 2 - 0.62
    for f in sat["files"]:
        ax.text(cx, y, f, ha="center", va="center", fontsize=8.3, color="#222222",
                 family="monospace")
        y -= 0.28
    ax.text(cx, cy - BOX_H / 2 + 0.24, sat["analogy"], ha="center", va="center",
             fontsize=7.6, color="#555555", style="italic")


def _arrow(ax, start: tuple[float, float], end: tuple[float, float], color: str) -> None:
    arrow = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=15,
        linewidth=1.8, color=color, shrinkA=4, shrinkB=4,
    )
    ax.add_patch(arrow)


def draw_diagram(output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(13, 10.3))

    _draw_hub(ax)
    for sat in SATELLITES:
        _draw_satellite(ax, sat)

    hcx, hcy = HUB_POS
    for sat in SATELLITES:
        scx, scy = sat["pos"]
        # Direction from hub edge toward satellite edge (both boxes roughly rectangular; a
        # straight line between edge midpoints reads cleanly at this layout's angles).
        dx, dy = scx - hcx, scy - hcy
        dist = (dx**2 + dy**2) ** 0.5
        ux, uy = dx / dist, dy / dist
        start = (hcx + ux * HUB_W / 2, hcy + uy * HUB_H / 2)
        end = (scx - ux * BOX_W / 2, scy - uy * BOX_H / 2)
        _arrow(ax, start, end, sat["color"])

    ax.text(
        6.5, 8.85,
        "The steering files that shape an agent in this repository\n"
        "CLAUDE.md is instructions; only settings.json's rules are enforced",
        ha="center", va="center", fontsize=12.5, color=HUB_COLOR, fontweight="bold",
    )
    ax.text(
        6.5, -0.45,
        'source: research/NOTE-SDLC-2-claude-code.md -- code.claude.com/docs (verified 2026-09-02)',
        ha="center", va="center", fontsize=7.6, color="#666666", style="italic",
    )
    ax.text(
        6.5, -0.75,
        '"Instructions in your prompt or CLAUDE.md shape what Claude tries to do, '
        'but they don\'t change what Claude Code allows."',
        ha="center", va="center", fontsize=7.6, color="#666666", style="italic",
    )

    ax.set_xlim(0, 13)
    ax.set_ylim(-1.05, 9.3)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    artefacts_dir = os.path.normpath(os.path.join(here, "..", "artefacts"))
    os.makedirs(artefacts_dir, exist_ok=True)
    output_path = os.path.join(artefacts_dir, "steering_file_map.png")
    draw_diagram(output_path)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()

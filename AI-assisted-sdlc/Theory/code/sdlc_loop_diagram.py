"""Render the governed SDLC loop diagram for SPEC-SDLC-1 (Theory).

Draws the six-stage loop this repository actually runs for every chapter -- intent -> spec,
ground the unknowns, write, gate, review, merge -- as a closed rectangular flow (the loop reopens
at "merge" with the next chapter's intent). Each stage box lists the concrete primitives active
there (prompts & rules / hooks & gates / tools & MCP / sub-agents & skills), colour-coded by
category, each row pointing at the actual file in THIS repository that plays that role -- not a
generic illustration. Mechanics (hooks, sub-agents, skills, MCP) are grounded in
research/NOTE-SDLC-2-claude-code.md (official Claude Code docs, verified 2026-09-02); the
concrete repo pointers (CLAUDE.md, docs/style-guide.md, .claude/hooks/*, .claude/agents/*,
.claude/skills/*, docs/definition-of-done.md) are this repository's own files, read directly.

Environment: matplotlib==3.11.1 (research/NOTE-2-package-versions.md, verified against PyPI
2026-09-02; confirmed live against this project's installed .venv interpreter), Python 3.11+.
Run with the project .venv:

    .venv/Scripts/python.exe "AI-assisted-sdlc/Theory/code/sdlc_loop_diagram.py"

Writes: ../artefacts/sdlc_loop_diagram.png
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Colour-per-primitive-category, held constant across every stage box (same technique as
# Data Science/Cloud Environment Setup/code/platform_workflow_diagram.py's per-cloud colours).
PROMPTS_COLOR = "#4C72B0"   # prompts & rules
HOOKS_COLOR = "#C44E52"     # hooks & gates
TOOLS_COLOR = "#55A868"     # tools & MCP
AGENTS_COLOR = "#8172B2"    # sub-agents & skills
BOX_EDGE = "#1B2A4A"
BOX_FACE = "#EAF1FB"

# The six stages of the loop this repository runs for every chapter (docs/architecture.md
# section 3 "Workflow (per chapter)"), each annotated with the primitives active there and the
# exact file in this repo that plays that role.
STAGES = [
    {
        "title": "1. Intent -> spec",
        "rows": [
            ("Sub-agents & skills", "chapter-scoper", AGENTS_COLOR),
            ("Prompts & rules", "CLAUDE.md golden rules", PROMPTS_COLOR),
        ],
    },
    {
        "title": "2. Ground the unknowns",
        "rows": [
            ("Sub-agents & skills", "researcher.md (Haiku)", AGENTS_COLOR),
            ("Sub-agents & skills", "research-brief skill", AGENTS_COLOR),
            ("Tools & MCP", "WebFetch / WebSearch", TOOLS_COLOR),
        ],
    },
    {
        "title": "3. Write the chapter",
        "rows": [
            ("Sub-agents & skills", "chapter-writer.md (Sonnet)", AGENTS_COLOR),
            ("Prompts & rules", "docs/style-guide.md", PROMPTS_COLOR),
            ("Tools & MCP", "Read / Write / Bash", TOOLS_COLOR),
        ],
    },
    {
        "title": "4. Gate",
        "rows": [
            ("Hooks & gates", "guard.sh (PreToolUse)", HOOKS_COLOR),
            ("Hooks & gates", "verify.sh (PostToolUse)", HOOKS_COLOR),
            ("Hooks & gates", "docs/definition-of-done.md", HOOKS_COLOR),
        ],
    },
    {
        "title": "5. Review",
        "rows": [
            ("Sub-agents & skills", "chapter-reviewer.md (fresh Sonnet)", AGENTS_COLOR),
            ("Prompts & rules", "docs/definition-of-done.md checklist", PROMPTS_COLOR),
        ],
    },
    {
        "title": "6. Merge",
        "rows": [
            ("Prompts & rules", "CLAUDE.md merge approval", PROMPTS_COLOR),
            ("Sub-agents & skills", "architect (Opus, main session)", AGENTS_COLOR),
        ],
    },
]

BOX_W, BOX_H = 3.05, 0.85
TOP_Y, BOTTOM_Y = 6.4, 1.9
ROW_H = 0.34


def _draw_box(ax, cx: float, cy: float, stage: dict) -> None:
    box = FancyBboxPatch(
        (cx - BOX_W / 2, cy), BOX_W, BOX_H,
        boxstyle="round,pad=0.06,rounding_size=0.09",
        linewidth=1.6, edgecolor=BOX_EDGE, facecolor=BOX_FACE,
    )
    ax.add_patch(box)
    ax.text(cx, cy + BOX_H * 0.72, stage["title"], ha="center", va="center",
             fontsize=11, fontweight="bold", color=BOX_EDGE)

    # Rows list below the box; each row is "category (coloured) -- this repo's file".
    y0 = cy - 0.28
    for i, (category, pointer, color) in enumerate(stage["rows"]):
        y = y0 - i * ROW_H
        ax.text(cx, y, category, ha="center", va="center", fontsize=7.6,
                 color=color, fontweight="bold")
        ax.text(cx, y - 0.165, pointer, ha="center", va="center", fontsize=7.6,
                 color="#333333")


def _straight_arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    arrow = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=16,
        linewidth=1.6, color="#555555", shrinkA=2, shrinkB=2,
    )
    ax.add_patch(arrow)


def draw_diagram(output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 7.6))

    # Column centres for a 3-wide top row and 3-wide bottom row (a racetrack loop: top row
    # runs left -> right, bottom row runs right -> left, closing back to stage 1).
    col_x = [1.9, 6.75, 11.6]
    top_centers = [(col_x[0], TOP_Y), (col_x[1], TOP_Y), (col_x[2], TOP_Y)]
    bottom_centers = [(col_x[2], BOTTOM_Y), (col_x[1], BOTTOM_Y), (col_x[0], BOTTOM_Y)]

    for (cx, cy), stage in zip(top_centers, STAGES[0:3]):
        _draw_box(ax, cx, cy, stage)
    for (cx, cy), stage in zip(bottom_centers, STAGES[3:6]):
        _draw_box(ax, cx, cy, stage)

    # Flow arrows: 1 -> 2 -> 3 (top row), 3 -> 4 (down the right side), 4 -> 5 -> 6 (bottom
    # row), 6 -> 1 (up the left side, closing the loop for the next chapter).
    _straight_arrow(ax, (col_x[0] + BOX_W / 2, TOP_Y + BOX_H / 2),
                     (col_x[1] - BOX_W / 2, TOP_Y + BOX_H / 2))
    _straight_arrow(ax, (col_x[1] + BOX_W / 2, TOP_Y + BOX_H / 2),
                     (col_x[2] - BOX_W / 2, TOP_Y + BOX_H / 2))
    _straight_arrow(ax, (col_x[2], TOP_Y), (col_x[2], BOTTOM_Y + BOX_H))
    _straight_arrow(ax, (col_x[2] - BOX_W / 2, BOTTOM_Y + BOX_H / 2),
                     (col_x[1] + BOX_W / 2, BOTTOM_Y + BOX_H / 2))
    _straight_arrow(ax, (col_x[1] - BOX_W / 2, BOTTOM_Y + BOX_H / 2),
                     (col_x[0] + BOX_W / 2, BOTTOM_Y + BOX_H / 2))
    _straight_arrow(ax, (col_x[0], BOTTOM_Y + BOX_H), (col_x[0], TOP_Y))

    ax.text(col_x[0] - 0.05, (TOP_Y + BOTTOM_Y + BOX_H) / 2, "loop closes ->\nnext chapter",
             ha="right", va="center", fontsize=8.2, color="#555555", style="italic", rotation=90)

    # Legend for the four primitive categories (SPEC-SDLC-1 LO1-LO4).
    legend_items = [
        ("Prompts & rules", PROMPTS_COLOR),
        ("Hooks & gates", HOOKS_COLOR),
        ("Tools & MCP", TOOLS_COLOR),
        ("Sub-agents & skills", AGENTS_COLOR),
    ]
    lx = 1.0
    for label, color in legend_items:
        ax.add_patch(plt.Rectangle((lx, 0.35), 0.28, 0.28, facecolor=color, edgecolor="none"))
        ax.text(lx + 0.4, 0.49, label, ha="left", va="center", fontsize=9, color="#1B2A4A")
        lx += 0.4 + len(label) * 0.11 + 0.6

    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 7.9)
    ax.axis("off")
    ax.set_title(
        "The governed SDLC loop -- prompts/rules, hooks/gates, tools/MCP, and sub-agents/skills\n"
        "composing into one chapter's lifecycle (this repository is the worked example)",
        fontsize=12, color="#1B2A4A", pad=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    artefacts_dir = os.path.normpath(os.path.join(here, "..", "artefacts"))
    os.makedirs(artefacts_dir, exist_ok=True)
    output_path = os.path.join(artefacts_dir, "sdlc_loop_diagram.png")
    draw_diagram(output_path)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()

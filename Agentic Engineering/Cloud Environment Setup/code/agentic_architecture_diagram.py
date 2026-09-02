"""Render the reference production architecture diagram for SPEC-AGENT-6.

Draws the GCP reference deployment this chapter walks through -- a container API
(Cloud Run) that fronts a hosted LLM (Vertex Generative API), a managed vector store
(Cloud SQL with the pgvector extension) for RAG retrieval, and Google Secret Manager
for credentials -- plus the four cross-cutting guardrails a security-minded engineer
adds around that core (rate limiting + input validation at the edge, a human-in-the-loop
approval queue gating any side-effecting tool call, and trace export for observability).

Service names and the reference-deployment shape are grounded in
research/NOTE-AGENT-5-cloud.md ("Reference Deployment (GCP -- Container API + Managed
Vector Store + Secrets)" and "Guardrails for production" sections) -- nothing in this
diagram's labels is invented. This is the ONLY runnable code artefact for this chapter:
the gcloud CLI / Python SDK snippets in the chapter prose are reference-only (no GCP
project or credentials exist in this sandbox to execute them against), matching the
same "grounded conceptual" pattern used in
Machine Learning/Cloud Environment Setup/code/training_workflow_diagram.py and
Data Science/Cloud Environment Setup/code/platform_workflow_diagram.py, whose Google
brand-blue / stage palette this script reuses for visual consistency across the three
"cloud" chapters in this course.

Environment: matplotlib==3.11.1 (pinned in research/NOTE-2-package-versions.md,
verified against PyPI 2026-09-02; installed version confirmed live via
`import matplotlib; matplotlib.__version__` in this project's .venv). Python 3.12+.

Run with the project's shared virtual environment:

    .venv/Scripts/python.exe "Agentic Engineering/Cloud Environment Setup/code/agentic_architecture_diagram.py"

Writes: ../artefacts/agentic_architecture_diagram.png
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Same palette used in the sibling "cloud" chapters (Machine Learning and Data Science
# Cloud Environment Setup diagrams) so the three chapters read as one visual system.
VERTEX_COLOR = "#4285F4"       # Google Cloud brand blue -- used for GCP-managed boxes
STAGE_COLOR = "#4C72B0"        # neutral infra border (client, generic flow)
GUARDRAIL_COLOR = "#B00020"    # guardrail border -- human-in-the-loop / approval gate
GUARDRAIL_EDGE = "#C77700"     # guardrail annotation -- rate limit / input validation
CLIENT_FILL = "#F2F2F2"
COMPUTE_FILL = "#E8F0FE"       # light Google blue -- Cloud Run
LLM_FILL = "#E8F0FE"
DB_FILL = "#E6F4EA"            # light green -- Cloud SQL / pgvector
SECRET_FILL = "#F3E8FD"        # light purple -- Secret Manager
TRACE_FILL = "#EEEEEE"
TOOL_FILL = "#FDEFE3"          # light orange -- side-effecting tool call
APPROVAL_FILL = "#FBE4E4"      # light red -- human-in-the-loop gate
EXT_FILL = "#F2F2F2"


def _box(ax, cx, cy, w, h, title, subtitle, fill, edge, edge_width=1.6, dashed=False):
    """Draw one rounded box centred at (cx, cy) with a bold title and italic subtitle."""
    style = "dashed" if dashed else "solid"
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.06,rounding_size=0.08",
        linewidth=edge_width, edgecolor=edge, facecolor=fill, linestyle=style,
    )
    ax.add_patch(box)
    if subtitle:
        ax.text(cx, cy + h * 0.16, title, ha="center", va="center",
                 fontsize=9.6, fontweight="bold", color="#1B2A4A")
        ax.text(cx, cy - h * 0.22, subtitle, ha="center", va="center",
                 fontsize=7.6, color="#3A4A6B", style="italic")
    else:
        ax.text(cx, cy, title, ha="center", va="center",
                 fontsize=9.6, fontweight="bold", color="#1B2A4A")


def _arrow(ax, start, end, color="#555555", label="", linestyle="-", label_dy=0.0,
           connectionstyle="arc3,rad=0.0"):
    arrow = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=14,
        linewidth=1.4, linestyle=linestyle, color=color,
        connectionstyle=connectionstyle,
    )
    ax.add_patch(arrow)
    if label:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mx, my + label_dy, label, ha="center", va="center",
                 fontsize=7.2, color="#333333",
                 bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                            edgecolor="none", alpha=0.85))


def draw_diagram(output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 9.2))

    # --- Client -------------------------------------------------------------------
    client = (6.2, 8.3)
    _box(ax, *client, 2.6, 0.8, "Client / caller", "", CLIENT_FILL, STAGE_COLOR)

    # --- Agent API hub (Cloud Run) -- guardrails live at this edge ----------------
    hub = (6.2, 6.7)
    _box(ax, *hub, 4.2, 1.15, "Cloud Run: agent API (FastAPI)",
         "rate limit + input validation at the edge", COMPUTE_FILL, VERTEX_COLOR,
         edge_width=2.2)
    ax.text(hub[0] - 2.05, hub[1] - 0.85, "GUARDRAIL: rate limiting +\nprompt-injection input\nsanitisation before any\nprompt is built", fontsize=6.6,
            color=GUARDRAIL_EDGE, ha="left", va="top", style="italic")

    # --- Hosted LLM (right) --------------------------------------------------------
    llm = (10.6, 6.7)
    _box(ax, *llm, 2.9, 1.0, "Vertex Generative API", "hosted LLM (Gemini)",
         LLM_FILL, VERTEX_COLOR)

    # --- Managed vector store (left) -----------------------------------------------
    vecdb = (1.7, 6.7)
    _box(ax, *vecdb, 2.9, 1.0, "Cloud SQL (pgvector)", "RAG vector store",
         DB_FILL, "#188038")

    # --- Secret Manager (bottom-left, dashed startup read) --------------------------
    secmgr = (1.7, 4.6)
    _box(ax, *secmgr, 2.9, 1.0, "Secret Manager", "LLM key + DB credentials",
         SECRET_FILL, "#7B1FA2")

    # --- Cloud Trace (right, observability) -----------------------------------------
    trace = (10.6, 4.6)
    _box(ax, *trace, 2.9, 1.0, "Cloud Trace", "spans + latency, per request",
         TRACE_FILL, "#5F6368")

    # --- Side-effecting tool call path (bottom centre) -------------------------------
    tool = (6.2, 4.6)
    _box(ax, *tool, 3.2, 0.95, "Side-effecting tool call", "e.g. write DB row, call external API",
         TOOL_FILL, GUARDRAIL_EDGE, edge_width=2.0)

    approval = (6.2, 3.0)
    _box(ax, *approval, 3.4, 1.0, "Human-in-the-loop approval queue",
         "blocks until a person approves or denies", APPROVAL_FILL, GUARDRAIL_COLOR,
         edge_width=2.4)

    external = (6.2, 1.4)
    _box(ax, *external, 2.9, 0.85, "External system", "executes only after approval",
         EXT_FILL, STAGE_COLOR)

    # --- Arrows ----------------------------------------------------------------------
    _arrow(ax, (client[0], client[1] - 0.4), (hub[0], hub[1] + 0.58),
           label="HTTPS request", label_dy=0.0)

    _arrow(ax, (hub[0] + 2.1, hub[1] + 0.15), (llm[0] - 1.45, llm[1] + 0.15),
           label="generate() call", label_dy=0.22)
    _arrow(ax, (hub[0] - 2.1, hub[1] + 0.15), (vecdb[0] + 1.45, vecdb[1] + 0.15),
           label="ANN search (RAG)", label_dy=0.22)

    _arrow(ax, (secmgr[0], secmgr[1] + 0.5), (hub[0] - 1.9, hub[1] - 0.35),
           color="#7B1FA2", linestyle="--",
           label="IAM-scoped secret read\n(startup, not per request)",
           connectionstyle="arc3,rad=-0.15")

    _arrow(ax, (hub[0] + 1.9, hub[1] - 0.45), (trace[0] - 1.3, trace[1] + 0.35),
           color="#5F6368", linestyle=":", label="spans/traces exported",
           connectionstyle="arc3,rad=0.25")

    _arrow(ax, (hub[0], hub[1] - 0.58), (tool[0], tool[1] + 0.475),
           color=GUARDRAIL_EDGE, label="model requests a tool call", label_dy=0.22)
    _arrow(ax, (tool[0], tool[1] - 0.475), (approval[0], approval[1] + 0.5),
           color=GUARDRAIL_COLOR, label="held for review")
    _arrow(ax, (approval[0], approval[1] - 0.5), (external[0], external[1] + 0.425),
           color=GUARDRAIL_COLOR, label="approved -> executes")

    # --- Legend ------------------------------------------------------------------
    legend_x, legend_y = 0.35, 1.85
    ax.text(legend_x, legend_y + 0.55, "Line styles", fontsize=8.2, fontweight="bold",
             color="#1B2A4A")
    ax.plot([legend_x, legend_x + 0.55], [legend_y + 0.25, legend_y + 0.25],
             color="#555555", linewidth=1.4)
    ax.text(legend_x + 0.68, legend_y + 0.25, "synchronous call", fontsize=7.4,
             va="center", color="#333333")
    ax.plot([legend_x, legend_x + 0.55], [legend_y, legend_y],
             color="#7B1FA2", linewidth=1.4, linestyle="--")
    ax.text(legend_x + 0.68, legend_y, "secret fetch (startup)", fontsize=7.4,
             va="center", color="#333333")
    ax.plot([legend_x, legend_x + 0.55], [legend_y - 0.25, legend_y - 0.25],
             color="#5F6368", linewidth=1.4, linestyle=":")
    ax.text(legend_x + 0.68, legend_y - 0.25, "trace/observability export", fontsize=7.4,
             va="center", color="#333333")
    ax.text(legend_x, legend_y - 0.6, "Red / orange borders = guardrail", fontsize=7.4,
             color=GUARDRAIL_COLOR, fontweight="bold")

    ax.set_xlim(0, 12.5)
    ax.set_ylim(0.5, 9.1)
    ax.axis("off")
    ax.set_title(
        "Reference production architecture -- GCP: Cloud Run + Cloud SQL (pgvector) + Secret Manager\n"
        "(service names verified live -- research/NOTE-AGENT-5-cloud.md, checked 2026-09-02)",
        fontsize=11.5, color="#1B2A4A", pad=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    artefacts_dir = os.path.normpath(os.path.join(here, "..", "artefacts"))
    os.makedirs(artefacts_dir, exist_ok=True)
    output_path = os.path.join(artefacts_dir, "agentic_architecture_diagram.png")
    draw_diagram(output_path)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()

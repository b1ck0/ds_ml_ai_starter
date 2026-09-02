"""Render the universal managed-ML-platform workflow diagram for SPEC-DS-15.

Draws the four-stage pipeline every managed ML platform implements under different
service names -- managed notebook -> training pipeline -> model registry -> deployment
endpoint -- as a single horizontal flow, with the concrete Vertex AI / Azure ML /
SageMaker service name for each stage underneath it. Service names are grounded in
research/NOTE-18-managed-platforms.md (verified against each cloud's 2026 docs).

This is the ONLY runnable code artefact for this chapter (the SDK snippets in the
chapter prose are reference-only -- cloud services cannot execute in this sandbox).

Environment: matplotlib==3.11.1 (pinned in research/NOTE-2-package-versions.md,
verified against PyPI 2026-09-02), Python 3.12+. Run with the project .venv:

    .venv/Scripts/python.exe "Data Science/Cloud Environment Setup/code/platform_workflow_diagram.py"

Writes: ../artefacts/platform_workflow_diagram.png
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, FancyBboxPatch

# The four universal MLOps stages (spec outline section 2), and the concrete service
# name in each cloud (research/NOTE-18-managed-platforms.md evidence table).
STAGES = [
    {
        "title": "Managed notebook",
        "subtitle": "exploration, prototyping",
        "vertex": "Workbench",
        "azure": "Compute Instances",
        "sagemaker": "SageMaker Studio",
    },
    {
        "title": "Training pipeline",
        "subtitle": "reproducible, versioned steps",
        "vertex": "Vertex AI Pipelines",
        "azure": "Pipelines / ML Jobs",
        "sagemaker": "SageMaker Pipelines",
    },
    {
        "title": "Model registry",
        "subtitle": "versioned, governed artefacts",
        "vertex": "Model Registry",
        "azure": "Model Registry",
        "sagemaker": "Model Registry",
    },
    {
        "title": "Deployment endpoint",
        "subtitle": "serves predictions",
        "vertex": "Endpoints",
        "azure": "Online / Batch Endpoints",
        "sagemaker": "SageMaker Endpoints",
    },
]

STAGE_COLOR = "#4C72B0"
VERTEX_COLOR = "#4285F4"
AZURE_COLOR = "#0078D4"
SAGEMAKER_COLOR = "#FF9900"
BOX_W, BOX_H = 2.6, 1.05
GAP = 0.75


def draw_diagram(output_path: str) -> None:
    n = len(STAGES)
    fig_w = n * BOX_W + (n - 1) * GAP + 1.2
    fig, ax = plt.subplots(figsize=(fig_w, 5.4))

    x = 0.6
    centers = []
    for stage in STAGES:
        box = FancyBboxPatch(
            (x, 3.0), BOX_W, BOX_H,
            boxstyle="round,pad=0.06,rounding_size=0.08",
            linewidth=1.5, edgecolor=STAGE_COLOR, facecolor="#EAF1FB",
        )
        ax.add_patch(box)
        cx = x + BOX_W / 2
        centers.append(cx)
        ax.text(cx, 3.0 + BOX_H * 0.62, stage["title"], ha="center", va="center",
                fontsize=11, fontweight="bold", color="#1B2A4A")
        ax.text(cx, 3.0 + BOX_H * 0.25, stage["subtitle"], ha="center", va="center",
                fontsize=8.5, color="#3A4A6B", style="italic")

        # Per-cloud service name labels stacked below the stage box.
        rows = [
            ("Vertex AI", stage["vertex"], VERTEX_COLOR),
            ("Azure ML", stage["azure"], AZURE_COLOR),
            ("SageMaker", stage["sagemaker"], SAGEMAKER_COLOR),
        ]
        y0 = 2.55
        for i, (cloud, svc, color) in enumerate(rows):
            y = y0 - i * 0.62
            ax.text(cx, y, svc, ha="center", va="center", fontsize=8.5,
                    color="#1B2A4A", fontweight="normal")
            ax.text(cx, y - 0.22, cloud, ha="center", va="center", fontsize=6.8,
                    color=color, fontweight="bold")

        x += BOX_W + GAP

    # Arrows connecting the stage boxes (drawn between box edges, not centers).
    for i in range(n - 1):
        start_x = centers[i] + BOX_W / 2
        end_x = centers[i + 1] - BOX_W / 2
        arrow = FancyArrow(
            start_x + 0.05, 3.0 + BOX_H / 2, (end_x - start_x) - 0.1, 0,
            width=0.015, head_width=0.14, head_length=0.12,
            length_includes_head=True, color="#555555",
        )
        ax.add_patch(arrow)

    ax.set_xlim(0, fig_w)
    ax.set_ylim(0.6, 4.5)
    ax.axis("off")
    ax.set_title(
        "The universal managed-ML-platform workflow\n"
        "(same four stages, different service name per cloud -- source: NOTE-18, verified 2026)",
        fontsize=11, color="#1B2A4A", pad=10,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    artefacts_dir = os.path.normpath(os.path.join(here, "..", "artefacts"))
    os.makedirs(artefacts_dir, exist_ok=True)
    output_path = os.path.join(artefacts_dir, "platform_workflow_diagram.png")
    draw_diagram(output_path)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()

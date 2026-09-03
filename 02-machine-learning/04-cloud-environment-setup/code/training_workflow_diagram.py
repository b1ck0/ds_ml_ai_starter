"""Render the cloud accelerator training-workflow diagram for SPEC-ML-12.

Draws the shape every cloud GPU/TPU training job has in common: object/blob storage
holds the sharded dataset, a stream of batches feeds an accelerator training job
(GPU or TPU workers), and the job writes checkpoints back to object/blob storage on a
schedule -- with a feedback path showing that a restarted or preempted job resumes
from the last checkpoint instead of starting over. The concrete service name under
each stage is filled in per cloud (Google Cloud, AWS, Azure), verified live against
each vendor's own docs -- see the chapter's citations and research/NOTE-18-managed-
platforms.md for the shared SDK/service-name grounding this diagram reuses.

This is the ONLY runnable code artefact for this chapter (the CLI/SDK snippets in the
chapter prose are reference-only -- no GCP/AWS/Azure account exists in this sandbox to
execute them against).

Environment: matplotlib==3.11.1 (pinned in research/NOTE-2-package-versions.md,
verified against PyPI 2026-09-02), Python 3.12+. Run with the project's ML venv:

    .venv-ml/Scripts/python.exe "Machine Learning/Cloud Environment Setup/code/training_workflow_diagram.py"

Writes: ../artefacts/training_workflow_diagram.png
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, FancyArrowPatch, FancyBboxPatch

# Same cloud brand colours used in Data Science/Cloud Environment Setup/code/
# platform_workflow_diagram.py, for a consistent palette across the two "cloud" chapters.
VERTEX_COLOR = "#4285F4"
AZURE_COLOR = "#0078D4"
SAGEMAKER_COLOR = "#FF9900"
STAGE_COLOR = "#4C72B0"
STORAGE_FILL = "#EAF1FB"
COMPUTE_FILL = "#FDEFE3"

# The three stages of the workflow (spec outline sections 4-6), each with the concrete
# per-cloud service name -- grounded live (see chapter Section 3/4 citations):
#   Google Cloud: Cloud Storage (GCS) -> Vertex AI custom training job (GPU or TPU) -> GCS
#   AWS:          S3 -> SageMaker Training job (GPU or Trainium) -> S3
#   Azure:        Blob Storage -> Azure ML command job (GPU) -> Blob Storage
STAGES = [
    {
        "title": "Object / blob storage",
        "subtitle": "sharded training data",
        "fill": STORAGE_FILL,
        "vertex": "Cloud Storage (GCS)",
        "azure": "Blob Storage",
        "sagemaker": "S3",
    },
    {
        "title": "Accelerator training job",
        "subtitle": "GPU or TPU workers",
        "fill": COMPUTE_FILL,
        "vertex": "Vertex AI custom job",
        "azure": "Azure ML command job",
        "sagemaker": "SageMaker Training",
    },
    {
        "title": "Object / blob storage",
        "subtitle": "checkpoints + final model",
        "fill": STORAGE_FILL,
        "vertex": "Cloud Storage (GCS)",
        "azure": "Blob Storage",
        "sagemaker": "S3",
    },
]

BOX_W, BOX_H = 3.0, 1.05
GAP = 1.5


def draw_diagram(output_path: str) -> None:
    n = len(STAGES)
    fig_w = n * BOX_W + (n - 1) * GAP + 1.4
    fig, ax = plt.subplots(figsize=(fig_w, 6.4))

    x = 0.7
    centers = []
    for stage in STAGES:
        box = FancyBboxPatch(
            (x, 3.2), BOX_W, BOX_H,
            boxstyle="round,pad=0.06,rounding_size=0.08",
            linewidth=1.5, edgecolor=STAGE_COLOR, facecolor=stage["fill"],
        )
        ax.add_patch(box)
        cx = x + BOX_W / 2
        centers.append(cx)
        ax.text(cx, 3.2 + BOX_H * 0.62, stage["title"], ha="center", va="center",
                 fontsize=11, fontweight="bold", color="#1B2A4A")
        ax.text(cx, 3.2 + BOX_H * 0.25, stage["subtitle"], ha="center", va="center",
                 fontsize=8.5, color="#3A4A6B", style="italic")

        # Per-cloud service name labels stacked below the stage box.
        rows = [
            ("Google Cloud", stage["vertex"], VERTEX_COLOR),
            ("AWS", stage["sagemaker"], SAGEMAKER_COLOR),
            ("Azure", stage["azure"], AZURE_COLOR),
        ]
        y0 = 2.75
        for i, (cloud, svc, color) in enumerate(rows):
            y = y0 - i * 0.55
            ax.text(cx, y, svc, ha="center", va="center", fontsize=8.5,
                     color="#1B2A4A", fontweight="normal")
            ax.text(cx, y - 0.22, cloud, ha="center", va="center", fontsize=6.8,
                     color=color, fontweight="bold")

        x += BOX_W + GAP

    # Forward arrows: storage -> training job -> storage.
    forward_labels = [
        "stream sharded batches\n(tf.data / WebDataset / S3 streaming)",
        "write checkpoint\nevery N steps",
    ]
    for i in range(n - 1):
        start_x = centers[i] + BOX_W / 2
        end_x = centers[i + 1] - BOX_W / 2
        arrow = FancyArrow(
            start_x + 0.05, 3.2 + BOX_H / 2, (end_x - start_x) - 0.1, 0,
            width=0.015, head_width=0.14, head_length=0.12,
            length_includes_head=True, color="#555555",
        )
        ax.add_patch(arrow)
        ax.text((start_x + end_x) / 2, 3.2 + BOX_H + 0.35, forward_labels[i],
                 ha="center", va="bottom", fontsize=7.6, color="#333333")

    # Feedback path: checkpoint storage -> training job, routed as an elbow well below
    # the per-cloud label stack (so it never overlaps the service-name text), labelled
    # with the restart/preemption behaviour that makes checkpointing worth doing.
    box_bottom = 3.2 - 0.15
    elbow_y = 0.55
    feedback_color = "#B00020"
    ax.plot([centers[2], centers[2]], [box_bottom, elbow_y],
             linestyle="--", linewidth=1.4, color=feedback_color)
    ax.plot([centers[2], centers[1]], [elbow_y, elbow_y],
             linestyle="--", linewidth=1.4, color=feedback_color)
    feedback_up = FancyArrowPatch(
        (centers[1], elbow_y), (centers[1], box_bottom),
        arrowstyle="-|>", mutation_scale=14,
        linewidth=1.4, linestyle="--", color=feedback_color,
    )
    ax.add_patch(feedback_up)
    ax.text((centers[1] + centers[2]) / 2, elbow_y - 0.15,
             "resume from last checkpoint on restart / preemption",
             ha="center", va="top", fontsize=7.8, color=feedback_color)

    ax.set_xlim(0, fig_w)
    ax.set_ylim(0.0, 5.0)
    ax.axis("off")
    ax.set_title(
        "Cloud GPU/TPU training workflow: blob storage -> accelerators -> blob storage\n"
        "(same shape on every cloud, different service name -- verified live, see chapter citations)",
        fontsize=11, color="#1B2A4A", pad=10,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    artefacts_dir = os.path.normpath(os.path.join(here, "..", "artefacts"))
    os.makedirs(artefacts_dir, exist_ok=True)
    output_path = os.path.join(artefacts_dir, "training_workflow_diagram.png")
    draw_diagram(output_path)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()

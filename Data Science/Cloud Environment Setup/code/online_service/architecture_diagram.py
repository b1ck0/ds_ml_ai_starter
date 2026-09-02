"""Render the batch-vs-online inference architecture diagram for this chapter.

Not part of the online service itself — a documentation-generation script, run with the
project's shared .venv (matplotlib==3.11.1, pinned per research/NOTE-2-package-versions.md),
not the dedicated .venv-serving used to run the FastAPI service.

Run:  .venv/Scripts/python architecture_diagram.py
Produces: ../../artefacts/batch_vs_online_architecture.png
"""
from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

out_path = (
    pathlib.Path(__file__).parent.parent.parent
    / "artefacts"
    / "batch_vs_online_architecture.png"
)

fig, (ax_online, ax_batch) = plt.subplots(1, 2, figsize=(13, 6))


def box(ax, xy, w, h, text, facecolor):
    b = FancyBboxPatch(
        xy, w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.4, edgecolor="#333333", facecolor=facecolor,
    )
    ax.add_patch(b)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
             fontsize=10, wrap=True)


def arrow(ax, start, end, label=None):
    a = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=15,
        linewidth=1.4, color="#333333",
    )
    ax.add_patch(a)
    if label:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mx, my + 0.05, label, ha="center", va="bottom", fontsize=8.5, style="italic")


# --- Online: request/response, like a REST service ---
ax_online.set_title("Online inference — REST service\n(this chapter's runnable service)",
                     fontsize=11, fontweight="bold")
box(ax_online, (0.02, 0.55), 0.28, 0.22, "Client\n(curl / requests)", "#dce8f5")
box(ax_online, (0.36, 0.55), 0.32, 0.22, "FastAPI app\n/predict endpoint\n(uvicorn)", "#fde9c8")
box(ax_online, (0.74, 0.55), 0.24, 0.22, "model.joblib\n(loaded once\nat startup)", "#e3f1de")
arrow(ax_online, (0.30, 0.66), (0.36, 0.66), "HTTP POST\nJSON body")
arrow(ax_online, (0.68, 0.66), (0.74, 0.66), "pipeline\n.predict()")
box(ax_online, (0.36, 0.15), 0.32, 0.22,
    "pydantic validates\nrequest at the boundary\n(422 on bad input)", "#f8d7da")
arrow(ax_online, (0.52, 0.55), (0.52, 0.37), "before handler\nruns")
ax_online.text(0.5, 0.03, "Latency: milliseconds · one row per call · always fresh code path",
               ha="center", fontsize=8.5, color="#555555")
ax_online.set_xlim(0, 1)
ax_online.set_ylim(0, 1)
ax_online.axis("off")

# --- Batch: scheduled job over many rows ---
ax_batch.set_title("Batch inference — scheduled job\n(reference pattern, Section 3)",
                    fontsize=11, fontweight="bold")
box(ax_batch, (0.02, 0.55), 0.28, 0.22, "Scheduler\n(Airflow DAG,\ncron-like trigger)", "#dce8f5")
box(ax_batch, (0.36, 0.55), 0.32, 0.22, "Batch job\n(reads millions\nof rows)", "#fde9c8")
box(ax_batch, (0.74, 0.55), 0.24, 0.22, "Model\n(same artifact,\nbulk-scored)", "#e3f1de")
arrow(ax_batch, (0.30, 0.66), (0.36, 0.66), "triggers on\na schedule")
arrow(ax_batch, (0.68, 0.66), (0.74, 0.66), "batch_predict()\n/ transform()")
box(ax_batch, (0.36, 0.15), 0.32, 0.22,
    "Output table\n(predictions land in\na warehouse/store)", "#f8d7da")
arrow(ax_batch, (0.52, 0.55), (0.52, 0.37), "writes results,\nno caller waiting")
ax_batch.text(0.5, 0.03, "Latency: minutes-hours · millions of rows/run · freshness = job interval",
              ha="center", fontsize=8.5, color="#555555")
ax_batch.set_xlim(0, 1)
ax_batch.set_ylim(0, 1)
ax_batch.axis("off")

fig.suptitle("Batch vs online inference — REST service vs scheduled job", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"wrote {out_path}")

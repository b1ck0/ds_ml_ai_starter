"""Schematic diagram: the warehouse-as-model-server pattern vs. container + endpoint.

Companion code for:
  Data Science/Cloud Environment Setup/in-database-ml.md

What it does:
  Draws one figure, two side-by-side panels, both schematic (not learned from
  data or measurements -- same "draw the process, don't fit it" approach as
  plot_promotion_decision() in
  Data Science/Production Considerations/code/drift_detection.py):

    LEFT panel  -- in-database ML (BigQuery ML / Redshift ML): the table
                   already sitting in the warehouse becomes the model. Train,
                   evaluate, and predict are all SQL statements issued against
                   the warehouse; there is no container image, no deployed
                   endpoint, and no network hop to a separate serving process.
    RIGHT panel -- the container + endpoint pattern (SPEC-DS-16's online-
                   serving chapter): the same journey requires exporting a
                   trained artefact, packaging it in a Docker image, deploying
                   that image behind a REST endpoint, and calling it over the
                   network for every prediction.

  This is the picture behind LO1 and LO4 of SPEC-DS-18: showing, not just
  telling, why "a SQL query returns predictions" is a materially shorter path
  to production for a SQL-fluent backend engineer when the data already lives
  in the warehouse and the use case is batch scoring.

Grounded:
  - BigQuery ML CREATE MODEL / ML.EVALUATE / ML.PREDICT and Redshift ML
    CREATE MODEL / generated prediction function shapes: research/NOTE-21-in-
    database-ml.md (Google Cloud + AWS docs, checked 2026-08-15 to 2026-09-01).
  - The container + endpoint pattern (train -> pickle -> FastAPI -> Docker ->
    deployed endpoint) mirrors Data Science/Cloud Environment Setup/code/
    online_service/ (SPEC-DS-16), not re-verified here -- shown only as the
    contrasting shape, no version claims made about it in this script.

Environment (research/NOTE-2-package-versions.md, checked 2026-09-02):
    matplotlib==3.11.1, Python 3.11+ (this script was run and gated against
    exactly the installed version in this project's .venv).

Run:
    python warehouse_ml_diagram.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this script only saves a figure, never shows one
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ARTEFACTS_DIR = Path(__file__).resolve().parent.parent / "artefacts"

COLOR_DATA = "#4C72B0"      # blue  -- data at rest
COLOR_COMPUTE = "#55A868"   # green -- where training/serving compute runs
COLOR_INFRA = "#C44E52"     # red   -- infrastructure you must build/operate
COLOR_QUERY = "#8172B2"     # purple -- the thing the caller issues


def box(ax, xy, w, h, text, facecolor, fontsize=8.5, textcolor="white"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor=facecolor, edgecolor="white", linewidth=1.2,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, color=textcolor, wrap=True)
    return x, y, w, h


def arrow(ax, start, end, text=None, color="#333333"):
    patch = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                             color=color, linewidth=1.3)
    ax.add_patch(patch)
    if text:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mx, my + 0.12, text, ha="center", va="bottom", fontsize=7.5, color=color)


def plot_warehouse_as_model_server() -> Path:
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 7.5))

    # ---------------- LEFT: warehouse-as-model-server ----------------
    ax_l.set_title("In-database ML\n(BigQuery ML / Redshift ML)", fontsize=11, fontweight="bold")

    box(ax_l, (0.5, 8.0), 5.0, 0.9, "Table already in the warehouse", COLOR_DATA)
    box(ax_l, (0.5, 6.3), 5.0, 0.9,
        "CREATE MODEL ... OPTIONS(model_type=...)\n(training compute runs INSIDE the\nwarehouse / its managed backend)",
        COLOR_COMPUTE)
    box(ax_l, (0.5, 4.6), 5.0, 0.9, "Model object stored in the warehouse\n(a schema object, not a deployment)", COLOR_DATA)
    box(ax_l, (0.5, 2.9), 5.0, 0.9,
        "ML.EVALUATE(...) / ML.PREDICT(...)\n(BigQuery)  --or--  SELECT schema.fn(...)\n(Redshift, generated prediction function)",
        COLOR_QUERY)
    box(ax_l, (0.5, 1.0), 5.0, 0.9, "Caller: a plain SQL SELECT\n(no client library, no network hop\nto a separate service)", COLOR_DATA)

    arrow(ax_l, (3.0, 8.0), (3.0, 7.2))
    arrow(ax_l, (3.0, 6.3), (3.0, 5.5))
    arrow(ax_l, (3.0, 4.6), (3.0, 3.8))
    arrow(ax_l, (3.0, 2.9), (3.0, 1.9))

    ax_l.text(3.0, 0.35, "No Docker image. No deployed endpoint.\nEverything above is SQL.",
              ha="center", va="center", fontsize=8.5, style="italic", color="#333333")

    ax_l.set_xlim(-0.3, 6.3)
    ax_l.set_ylim(-0.2, 9.2)
    ax_l.axis("off")

    # ---------------- RIGHT: container + endpoint ----------------
    ax_r.set_title("Container + endpoint\n(e.g. the FastAPI service, SPEC-DS-16)", fontsize=11, fontweight="bold")

    box(ax_r, (0.5, 8.6), 5.0, 0.7, "Training data (warehouse, files, ...)", COLOR_DATA)
    box(ax_r, (0.5, 7.4), 5.0, 0.7, "train_model.py -> pickled/joblib artefact", COLOR_COMPUTE)
    box(ax_r, (0.5, 6.2), 5.0, 0.7, "Package artefact into a Docker image\n(FastAPI app + model + deps)", COLOR_INFRA)
    box(ax_r, (0.5, 5.0), 5.0, 0.7, "Push image; deploy behind a running endpoint\n(container host / managed endpoint)", COLOR_INFRA)
    box(ax_r, (0.5, 3.8), 5.0, 0.7, "Endpoint process listens for requests\n(must stay up, scaled, patched, monitored)", COLOR_INFRA)
    box(ax_r, (0.5, 1.9), 5.0, 0.9,
        "Caller: HTTP POST /predict over the network\n(client library / curl, serialize request,\ndeserialize response)",
        COLOR_QUERY)

    arrow(ax_r, (3.0, 8.6), (3.0, 8.1))
    arrow(ax_r, (3.0, 7.4), (3.0, 6.9))
    arrow(ax_r, (3.0, 6.2), (3.0, 5.7))
    arrow(ax_r, (3.0, 5.0), (3.0, 4.5))
    arrow(ax_r, (3.0, 3.8), (3.0, 2.8))

    ax_r.text(3.0, 0.9, "Five things to build and operate\nbefore the first prediction.",
              ha="center", va="center", fontsize=8.5, style="italic", color="#333333")

    ax_r.set_xlim(-0.3, 6.3)
    ax_r.set_ylim(-0.2, 9.6)
    ax_r.axis("off")

    fig.suptitle("The warehouse-as-model-server pattern vs. a container + endpoint",
                 fontsize=12.5, fontweight="bold", y=1.0)
    fig.tight_layout()

    out_path = ARTEFACTS_DIR / "warehouse_as_model_server.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = plot_warehouse_as_model_server()
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

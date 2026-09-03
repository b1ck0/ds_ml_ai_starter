"""Render the offline-vs-online architecture diagram referenced from
feature-store-feast.md.

Companion code for:
  Data Science/Worked Examples/feature-store-feast.md

Not part of the Feast pipeline itself -- this only draws the picture of how
the pieces fit together (one FeatureView definition, two stores, two
retrieval paths) using matplotlib (version pinned per NOTE-2-package-versions,
matching the rest of this project's .venv).

Run:
    .venv-feast/Scripts/python.exe make_architecture_diagram.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this script only saves a figure, never shows one
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_PATH = Path(__file__).parents[2] / "artefacts" / "feast_offline_online_architecture.png"

BOX_STYLE = dict(boxstyle="round,pad=0.4,rounding_size=0.08", linewidth=1.6)
COLORS = {
    "source": "#e8e8e8",
    "defs": "#cfe8ff",
    "offline": "#ffe8b3",
    "online": "#c8f2c8",
    "consumer": "#f0d9ff",
}


def box(ax, xy, w, h, text, color, fontsize=10.5, weight="normal"):
    x, y = xy
    patch = FancyBboxPatch((x, y), w, h, facecolor=color, edgecolor="#333333", **BOX_STYLE)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, weight=weight, wrap=True)
    return (x, y, w, h)


def arrow(ax, start, end, label="", color="#333333", style="-|>", connectionstyle="arc3,rad=0.0"):
    a = FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=16,
                         color=color, linewidth=1.6, connectionstyle=connectionstyle)
    ax.add_patch(a)
    if label:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mx, my + 0.15, label, ha="center", va="bottom", fontsize=9, color=color)


def main() -> None:
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # Row 1: raw data / sources
    src = box(ax, (0.4, 7.6), 3.0, 0.9, "Raw event data\n(trip logs, app clicks)", COLORS["source"])
    # Feature definitions -- the ONE spec that drives everything below.
    defs = box(ax, (4.0, 7.6), 3.4, 0.9,
               "feature_definitions.py\nEntity + FeatureView(s)\n(one definition, LO1)",
               COLORS["defs"], weight="bold")
    cli = box(ax, (7.9, 7.6), 2.7, 0.9, "feast apply\n(registers to registry.db)", COLORS["source"])

    arrow(ax, (src[0] + src[2], src[1] + src[3] / 2), (defs[0], defs[1] + defs[3] / 2), "schema for")
    arrow(ax, (defs[0] + defs[2], defs[1] + defs[3] / 2), (cli[0], cli[1] + cli[3] / 2))

    # Row 2: offline store (left) and online store (right), both fed by the same defs.
    offline = box(ax, (0.6, 5.1), 4.4, 1.6,
                   "OFFLINE STORE (slow)\nParquet file, full history\n"
                   "FileSource(path=..., timestamp_field=...)\n"
                   "-- every event ever recorded",
                   COLORS["offline"], weight="bold")
    online = box(ax, (6.0, 5.1), 4.4, 1.6,
                 "ONLINE STORE (fast)\nSQLite (Redis/DynamoDB in prod)\n"
                 "one row per entity: LATEST value only\n"
                 "-- overwritten by materialize()",
                 COLORS["online"], weight="bold")

    arrow(ax, (defs[0] + defs[2] * 0.3, defs[1]), (offline[0] + offline[2] * 0.5, offline[1] + offline[3]))
    arrow(ax, (cli[0] + cli[2] * 0.3, cli[1]), (online[0] + online[2] * 0.5, online[1] + online[3]),
          "registers schema for")
    arrow(ax, (offline[0] + offline[2], offline[1] + offline[3] * 0.6),
          (online[0], online[1] + online[3] * 0.6), "materialize()\n/ materialize_incremental()",
          color="#8a5b00")

    # Row 3: retrieval calls
    hist_call = box(ax, (0.6, 3.0), 4.4, 1.1,
                     "get_historical_features(entity_df, features)\npoint-in-time-correct join",
                     "#fff3d6")
    online_call = box(ax, (6.0, 3.0), 4.4, 1.1,
                       "get_online_features(features, entity_rows)\nlatest value, single-digit ms",
                       "#e2fbe2")

    arrow(ax, (offline[0] + offline[2] * 0.5, offline[1]), (hist_call[0] + hist_call[2] * 0.5, hist_call[1] + hist_call[3]))
    arrow(ax, (online[0] + online[2] * 0.5, online[1]), (online_call[0] + online_call[2] * 0.5, online_call[1] + online_call[3]))

    # Row 4: consumers
    training = box(ax, (0.6, 0.9), 4.4, 1.1,
                    "TRAINING\nbuild a labeled dataset\n(batch, offline, reproducible)",
                    COLORS["consumer"])
    serving = box(ax, (6.0, 0.9), 4.4, 1.1,
                   "SERVING\nlive prediction request\n(low latency, current state)",
                   COLORS["consumer"])

    arrow(ax, (hist_call[0] + hist_call[2] * 0.5, hist_call[1]), (training[0] + training[2] * 0.5, training[1] + training[3]))
    arrow(ax, (online_call[0] + online_call[2] * 0.5, online_call[1]), (serving[0] + serving[2] * 0.5, serving[1] + serving[3]))

    ax.text(5.5, 0.15,
            "Same FeatureView definitions drive both columns → no train/serve skew (LO1, LO4)",
            ha="center", va="center", fontsize=10.5, style="italic", color="#333333")

    ax.set_title("Feast: one feature definition, two stores, two retrieval paths",
                  fontsize=13, weight="bold", pad=14)

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

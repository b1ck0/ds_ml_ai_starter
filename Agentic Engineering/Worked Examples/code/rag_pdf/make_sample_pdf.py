"""Generate the sample PDF used by ingest.py / retrieve.py / answer.py.

Companion code for:
  Agentic Engineering/Worked Examples/rag-over-pdfs.md

What it does:
  Synthesizes a small, entirely fictional internal engineering handbook
  ("Acme Robotics — Field Service Engineering Handbook") as a real, multi-page,
  text-based PDF -- three pages, six numbered policy sections, each with a
  distinct, checkable fact (a rollback window, an on-call rotation length, an
  API rate limit, ...). This gives the RAG pipeline in this chapter a small
  corpus where you can ask a question and *know* which chunk should answer it,
  which makes retrieval quality easy to eyeball.

  This is a synthetic document generated for teaching purposes. "Acme Robotics"
  is a placeholder name (the classic generic-company stand-in), not a real
  company; every policy, number, and process described is invented for this
  exercise.

Environment note (deliberately different from the rest of this chapter):
  This ONE script needs matplotlib, which lives in the project's shared
  `.venv` (matplotlib==3.11.1, verified in NOTE-2-package-versions, checked
  2026-09-02) -- not in `.venv-agent`, which has no PDF-authoring library.
  It uses matplotlib's built-in `PdfPages` backend purely as a text-layout
  engine to lay out real, extractable PDF text (not a rasterized image) --
  no extra dependency beyond what the rest of the book already installs.
  Everything downstream (ingest.py, retrieve.py, answer.py) needs only
  `.venv-agent` and never touches matplotlib.

Run (one-time; the output PDF is committed, so readers don't have to):
    .venv/Scripts/python.exe make_sample_pdf.py
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this script only saves a PDF, never shows one
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUT_PATH = Path(__file__).parent / "sample" / "acme_handbook.pdf"

# Each page is a list of (heading, paragraph) blocks. Headings are numbered
# policy sections so retrieval results can cite "section 4" etc. back to a
# human-checkable source.
PAGE_1_TITLE = "ACME ROBOTICS -- FIELD SERVICE ENGINEERING HANDBOOK (v3.2, internal)"

PAGES: list[list[tuple[str, str]]] = [
    [
        (
            "1. Deployment Runbook",
            "All production deployments to the fleet-control service go through a canary "
            "rollout: the new build first serves 5 percent of regional traffic for a minimum "
            "of twenty minutes before promotion. A canary is considered failed if error rate "
            "exceeds 2 percent or p99 latency exceeds 800 milliseconds during that window. If "
            "the canary fails, the on-call engineer must initiate a rollback within fifteen "
            "minutes of the failure being flagged; rollbacks beyond that window require a "
            "written incident postmortem regardless of customer impact. Deployments are only "
            "permitted Monday through Thursday before 16:00 local time, to keep a full "
            "business day of on-call coverage available in case of a slow-burn regression. "
            "Emergency hotfixes outside that window require sign-off from the on-call lead "
            "and the service owner.",
        ),
        (
            "2. Incident Response Policy",
            "Incidents are classified SEV-1 through SEV-4 by customer impact. A SEV-1 "
            "incident -- full outage of the fleet-control API or any safety-critical robot "
            "command path -- requires the on-call engineer to acknowledge the page within "
            "five minutes and open a dedicated incident channel within ten minutes of "
            "acknowledgement. A SEV-2 incident, defined as degraded service for more than ten "
            "percent of fleets, carries a thirty-minute acknowledgement window. Every SEV-1 "
            "and SEV-2 incident requires a written postmortem within three business days of "
            "resolution, filed against the incident's tracking ticket and reviewed by the "
            "engineering lead before it is closed.",
        ),
    ],
    [
        (
            "3. Database Migration Policy",
            "Schema migrations against the fleet-telemetry database run through Flyway and "
            "must be backward compatible with the currently deployed application version for "
            "at least one full deploy cycle -- additive changes only (new nullable columns, "
            "new tables, new indexes); no column drops or type changes in the same migration "
            "that removes application code depending on the old shape. Every migration "
            "requires review and approval from two engineers, at least one of whom is not the "
            "migration's author, before it can run against the staging database. Migrations "
            "that touch a table larger than ten million rows must run during the Tuesday or "
            "Wednesday maintenance window and must include a tested rollback script in the "
            "same pull request.",
        ),
        (
            "4. On-Call Rotation & Escalation",
            "The primary on-call rotation is one week long, handing off every Monday at "
            "09:00 local time. Each engineer on the fleet-control team rotates through "
            "primary on-call roughly once every six weeks, assuming a team of six. If the "
            "primary on-call engineer does not acknowledge a page within ten minutes, the "
            "incident automatically escalates to the secondary on-call engineer; if the "
            "secondary does not acknowledge within a further ten minutes, it escalates to "
            "the engineering lead directly. Planned on-call swaps must be requested at least "
            "48 hours in advance through the on-call calendar tool.",
        ),
    ],
    [
        (
            "5. API Rate Limits",
            "The public fleet-status API enforces a rate limit of 100 requests per minute "
            "per API key, with a burst allowance of 20 additional requests over any "
            "five-second window before throttling kicks in. Requests beyond the limit "
            "receive an HTTP 429 response with a Retry-After header. Enterprise-tier API "
            "keys may request a raised limit of up to 500 requests per minute by filing a "
            "capacity request with the platform team at least five business days before the "
            "increased limit is needed. Rate limit counters reset on a rolling sixty-second "
            "window per key, not a fixed calendar minute, so bursts near a minute boundary "
            "are still capped correctly.",
        ),
        (
            "6. Logging & Data Retention",
            "Application logs from the fleet-control service are retained for 30 days in "
            "hot storage and then moved to cold storage for an additional 11 months before "
            "deletion, for a total retention period of 12 months. Logs containing customer "
            "location data must be redacted of precise GPS coordinates (rounded to the "
            "nearest 1 kilometre grid cell) before they leave hot storage. Access to "
            "unredacted location logs requires a documented business justification and "
            "sign-off from the data protection lead, logged in the access-request system.",
        ),
    ],
]

WRAP_WIDTH = 92
LINE_HEIGHT = 0.026
FONT = "DejaVu Serif"


def render_page(fig, page_blocks: list[tuple[str, str]], title: str | None) -> None:
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.axis("off")
    y = 0.95

    if title:
        ax.text(0.07, y, title, fontsize=12, fontweight="bold", family=FONT, va="top")
        y -= LINE_HEIGHT * 3

    for heading, paragraph in page_blocks:
        ax.text(0.07, y, heading, fontsize=11, fontweight="bold", family=FONT, va="top")
        y -= LINE_HEIGHT * 1.6
        for line in textwrap.wrap(paragraph, width=WRAP_WIDTH):
            ax.text(0.07, y, line, fontsize=10, family=FONT, va="top")
            y -= LINE_HEIGHT
        y -= LINE_HEIGHT * 1.2  # gap before next section


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT_PATH) as pdf:
        for i, page_blocks in enumerate(PAGES):
            fig = plt.figure(figsize=(8.5, 11))
            render_page(fig, page_blocks, title=PAGE_1_TITLE if i == 0 else None)
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size} bytes, {len(PAGES)} pages)")


if __name__ == "__main__":
    main()

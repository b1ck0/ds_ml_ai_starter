"""Render this chapter's concrete RAG pipeline as a PNG diagram, with Pillow (no
matplotlib needed -- matches the approach already used for the RAG pipeline / MCP
diagrams in Agentic Engineering/Theory/code/tiny_rag_demo.py, which lives in this same
.venv-agent environment: Pillow 12.3.0 -- verified via `import PIL; PIL.__version__`).

Companion code for:
  Agentic Engineering/Worked Examples/rag-over-pdfs.md

Where this differs from Theory/theory.md's generic RAG pipeline diagram
(artefacts/rag_pipeline_diagram.png, offline chunk->embed->store / online
query->embed->search->augment->generate): this diagram is this chapter's own pipeline,
naming the actual scripts (ingest.py, retrieve.py, answer.py) and marking exactly where
the no-key boundary sits -- everything left of the dashed line runs with no API key at
all; only the final generation step is key-gated.

Run:
    .venv-agent/Scripts/python.exe "Agentic Engineering/Worked Examples/code/rag_pdf/make_pipeline_diagram.py"
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ARTEFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "artefacts"
OUT_PATH = ARTEFACTS_DIR / "rag_pdf_pipeline_diagram.png"


def _font(size: int) -> ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def _box(draw: ImageDraw.ImageDraw, xy, text: str, font, fill="#eef3fb", outline="black") -> None:
    x0, y0, x1, y1 = xy
    draw.rectangle(xy, outline=outline, fill=fill, width=2)
    lines = text.split("\n")
    line_h = font.size + 4
    total_h = line_h * len(lines)
    ty = y0 + ((y1 - y0) - total_h) // 2
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        tx = x0 + ((x1 - x0) - tw) // 2
        draw.text((tx, ty), line, font=font, fill="black")
        ty += line_h


def _harrow(draw: ImageDraw.ImageDraw, x0: int, x1: int, y: int, label: str = "", font=None) -> None:
    draw.line([(x0, y), (x1, y)], fill="black", width=2)
    draw.polygon([(x1, y), (x1 - 10, y - 5), (x1 - 10, y + 5)], fill="black")
    if label and font:
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(((x0 + x1) // 2 - tw // 2, y - font.size - 20), label, font=font, fill="#333333")


def draw() -> None:
    W, H = 1500, 480
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    title_font = _font(18)
    font = _font(13)
    small = _font(11)

    d.text((20, 12), "RAG over PDFs -- this chapter's pipeline (ingest.py -> retrieve.py -> answer.py)", font=title_font, fill="black")

    # --- Row 1: the always-runnable, no-key pipeline -------------------------------
    y0, y1 = 90, 170
    bw = 190
    boxes = [
        (20, "acme_handbook.pdf\n(sample/)"),
        (230, "ingest.py:\nparse (pdfplumber)\n+ chunk (100w/20w)"),
        (440, "ingest.py:\nembed\n(all-MiniLM-L6-v2)"),
        (650, "index/\nembeddings.npy\n+ chunks.jsonl"),
        (860, "retrieve.py:\ncosine top-k\n(numpy, unit vectors)"),
        (1070, "answer.py:\nassemble grounded\nprompt (context+Q)"),
    ]
    for x, label in boxes:
        _box(d, (x, y0, x + bw, y1), label, font)
    for (x1, _), (x2, _) in zip(boxes, boxes[1:]):
        _harrow(d, x1 + bw, x2, (y0 + y1) // 2)

    d.text((20, y0 - 22), "NO API KEY NEEDED for any of this -- pure CPU, local model, numpy", font=small, fill="#2a7a2a")

    # --- Row 2: the key-gated decision, fed by answer.py's assembled prompt --------
    gate_y0, gate_y1 = 250, 320
    down_x = 1070 + bw // 2  # = 1165, center of the answer.py box above
    d.line([(down_x, y1), (down_x, gate_y0)], fill="black", width=2)
    d.polygon([(down_x, gate_y0), (down_x - 5, gate_y0 - 10), (down_x + 5, gate_y0 - 10)], fill="black")

    gate_box = (down_x - 130, gate_y0, down_x + 130, gate_y1)  # (1035, 250, 1295, 320)
    _box(d, gate_box, "ANTHROPIC_API_KEY +\nANTHROPIC_MODEL +\n'anthropic' installed?", font, fill="#fdf3e0")
    d.text((down_x - 95, gate_y0 - 20), "KEY-GATED BOUNDARY", font=small, fill="#b04a2f")

    # --- Row 3: the two outcomes, each fed by its own arrow off the gate box -------
    out_y0, out_y1 = 370, 450

    no_box = (730, out_y0, 1080, out_y1)  # width 350, left of gate center
    _box(
        d,
        no_box,
        "NO-KEY FALLBACK (answer.py):\nprint the assembled context\n(chunks + scores) to stdout,\nexit 0 -- no network call made",
        font,
        fill="#fbe9e7",
    )
    no_x = gate_box[0] + 30  # 1065, lands inside no_box's x-range
    d.line([(no_x, gate_y1), (no_x, out_y0)], fill="black", width=2)
    d.polygon([(no_x, out_y0), (no_x - 5, out_y0 - 10), (no_x + 5, out_y0 - 10)], fill="black")
    d.text((no_x - 30, gate_y1 + 8), "no", font=font, fill="#333333")

    yes_box = (1110, out_y0, 1460, out_y1)  # width 350, right of gate center
    _box(
        d,
        yes_box,
        "YES (answer.py):\nanthropic.Anthropic().messages\n  .create(model=..., messages=[prompt])\n-> generated answer, cited to chunk ids",
        font,
        fill="#e6f4ea",
    )
    yes_x = gate_box[2] - 30  # 1265, lands inside yes_box's x-range
    d.line([(yes_x, gate_y1), (yes_x, out_y0)], fill="black", width=2)
    d.polygon([(yes_x, out_y0), (yes_x - 5, out_y0 - 10), (yes_x + 5, out_y0 - 10)], fill="black")
    d.text((yes_x + 10, gate_y1 + 8), "yes", font=font, fill="#333333")

    d.text(
        (20, 460),
        "Every fact in the generated answer traces back to a [chunk id, page] pair from the retrieved context -- that's what \"grounded\" means here.",
        font=font,
        fill="#333333",
    )

    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    img.save(OUT_PATH)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    draw()

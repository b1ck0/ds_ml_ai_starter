"""Generate a small sample invoice PDF with no external PDF-writing library.

Why hand-built: this chapter's environment (.venv-agent) has pdfplumber 0.11.10 and
pypdf 6.16.2 for *reading* PDFs, but neither reportlab, fpdf2, nor matplotlib is
installed for *writing* one, and pypdf itself has no text-drawing canvas -- it can
only assemble/transform existing pages (verified live in this venv: see the chapter's
"Local Environment Setup" note). The PDF file format's text-drawing primitives are
simple enough to emit directly: a handful of PDF objects (catalog, pages, one page, a
base-14 font, one content stream) plus a cross-reference table. This keeps the sample
data reproducible without adding a dependency the rest of the chapter doesn't need.

Run:
    .venv-agent/Scripts/python.exe generate_sample_invoice.py
"""
from __future__ import annotations

from pathlib import Path

OUT_PATH = Path(__file__).parent / "datasets" / "sample_invoice.pdf"

# The invoice body, line by line, as it will be drawn top-to-bottom on one page.
# Courier (a PDF base-14 font -- no font file to embed) is monospaced, so the
# whitespace padding below lines up both on screen and in the extracted text: each
# line is one PDF text-showing operation, so the spaces are real characters in the
# content stream, not just visual gaps. That is what lets extract.py's regex fallback
# split columns on runs of whitespace.
LINES: list[str] = [
    "ACME WIDGETS INC.",
    "123 Foundry Lane, Springfield, IL 62701",
    "",
    "                              INVOICE",
    "",
    "Invoice Number: INV-2026-0042",
    "Invoice Date:   2026-08-15",
    "Bill To:        Contoso Ltd, 77 Market St, Boston MA 02108",
    "",
    "Description                    Qty   Unit Price       Amount",
    "------------------------------------------------------------",
    "Widget A - Standard              10       12.50        125.00",
    "Widget B - Heavy Duty             5       40.00        200.00",
    "Shipping and Handling             1       15.00         15.00",
    "------------------------------------------------------------",
    "                                          Subtotal:     340.00",
    "                                          Tax (8%):      27.20",
    "                                          Total Due:    367.20",
]

FONT_SIZE = 10
LEADING = 13  # points between baselines
TOP_Y = 760
LEFT_X = 56
PAGE_W = 612  # US Letter, points
PAGE_H = 792


def _escape(text: str) -> str:
    """Escape the three bytes that are special inside a PDF literal string: ( ) \\ ."""
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _content_stream() -> bytes:
    parts = ["BT", f"/F1 {FONT_SIZE} Tf", f"{LEADING} TL", f"{LEFT_X} {TOP_Y} Td"]
    for i, line in enumerate(LINES):
        if i > 0:
            parts.append("T*")  # advance to next line using the leading set above
        parts.append(f"({_escape(line)}) Tj")
    parts.append("ET")
    return ("\n".join(parts) + "\n").encode("latin-1")


def build_pdf() -> bytes:
    """Assemble a minimal, spec-valid single-page PDF as raw bytes."""
    stream_bytes = _content_stream()

    objects: list[bytes] = []
    # 1: Catalog
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    # 2: Pages
    objects.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    # 3: Page
    objects.append(
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] "
        f"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>".encode("latin-1")
    )
    # 4: Font (base-14, no embedding needed)
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")
    # 5: Content stream
    objects.append(
        f"<< /Length {len(stream_bytes)} >>\nstream\n".encode("latin-1")
        + stream_bytes
        + b"endstream"
    )

    out = bytearray()
    out += b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"  # header + binary marker comment

    offsets = [0]  # object 0 is the free-list head, filled in below
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode("latin-1")
        out += body
        out += b"\nendobj\n"

    xref_start = len(out)
    n_objs = len(objects) + 1
    out += f"xref\n0 {n_objs}\n".encode("latin-1")
    out += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        out += f"{off:010d} 00000 n \n".encode("latin-1")

    out += (
        f"trailer\n<< /Size {n_objs} /Root 1 0 R >>\n"
        f"startxref\n{xref_start}\n%%EOF"
    ).encode("latin-1")
    return bytes(out)


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pdf_bytes = build_pdf()
    OUT_PATH.write_bytes(pdf_bytes)
    print(f"Wrote {OUT_PATH} ({len(pdf_bytes)} bytes)")


if __name__ == "__main__":
    main()

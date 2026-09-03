"""Two ways to turn a PDF invoice into the raw dict schema.py's Invoice validates.

    extract_fields_rule_based(pdf_path)  -- deterministic, regex-on-text. No network
                                             call, no API key, no LLM. This is the
                                             path run.py exercises for AC2: it must
                                             work end-to-end with nothing but the
                                             sample PDF and this repo's environment.

    extract_fields_llm(pdf_path)         -- KEY-GATED. Sends the extracted page text
                                             to Claude with a tool schema derived
                                             from Invoice.model_json_schema() and
                                             asks it to call that tool with the
                                             structured fields. Mirrors the
                                             tools=[...] / tool_use bridge already
                                             built in code/mcp_db/llm_client.py, and
                                             the anthropic SDK call shape grounded in
                                             NOTE-AGENT-4-provider-sdks.md.

Both return the SAME raw dict shape (str/number fields, ISO date string, list of
line-item dicts) -- neither one validates its own output. Validation happens exactly
once, in schema.py's Invoice(**raw), regardless of which extractor produced the raw
dict. That is deliberate: an LLM's structured "tool call" is still just JSON the
model decided to emit, not a guarantee -- see the chapter's Pitfalls section.

pdfplumber 0.11.10 is installed in .venv-agent (confirmed via `pip show`); its
`.extract_text()` method is grounded in NOTE-AGENT-2-rag-primitives.md.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pdfplumber


class ExtractionError(Exception):
    """Raised when a PDF can't be opened, or a required field can't be found.

    Kept as one exception type with a clear message rather than returning None /
    a half-filled dict -- a missing field should fail loudly here, at the extraction
    boundary, not silently propagate into schema.py as a confusing ValidationError
    three layers down.
    """


# --- Rule-based (deterministic, no key) -------------------------------------------

_FIELD_PATTERNS: dict[str, re.Pattern[str]] = {
    "invoice_number": re.compile(r"^Invoice Number:\s*(?P<value>.+)$"),
    "invoice_date": re.compile(r"^Invoice Date:\s*(?P<value>.+)$"),
    "bill_to": re.compile(r"^Bill To:\s*(?P<value>.+)$"),
    "subtotal": re.compile(r"^Subtotal:\s*(?P<value>[\d,]+\.\d{2})$"),
    "tax": re.compile(r"^Tax\s*\([^)]*\):\s*(?P<value>[\d,]+\.\d{2})$"),
    "total": re.compile(r"^Total Due:\s*(?P<value>[\d,]+\.\d{2})$"),
}

# A line-item row, as pdfplumber's extract_text() renders it: a description
# (letters/digits/spaces/hyphens), then three whitespace-separated numeric columns
# -- quantity (integer), unit price, amount (both decimal, two places). pdfplumber
# collapses each row's original column gaps down to single spaces (verified by
# running extract_text() on generate_sample_invoice.py's output while writing this
# chapter), so a plain single-space split is enough; no fixed-width column math
# needed.
_LINE_ITEM_RE = re.compile(
    r"^(?P<description>.+?)\s+(?P<quantity>\d+)\s+(?P<unit_price>[\d,]+\.\d{2})\s+(?P<amount>[\d,]+\.\d{2})$"
)
_SEPARATOR_RE = re.compile(r"^-{5,}$")
_LINE_ITEM_HEADER = "Description Qty Unit Price Amount"


def _read_pdf_text(pdf_path: Path) -> str:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if not pdf.pages:
                raise ExtractionError(f"{pdf_path}: PDF has no pages")
            text = pdf.pages[0].extract_text()
    except ExtractionError:
        raise
    except Exception as exc:  # pdfplumber/pypdfium2 raise several exception types
        # for a corrupt/non-PDF file; normalise all of them to ExtractionError so
        # callers only ever need to catch one thing.
        raise ExtractionError(f"{pdf_path}: could not be read as a PDF ({exc!r})") from exc
    if not text:
        raise ExtractionError(f"{pdf_path}: no extractable text on page 1 (scanned image? OCR is out of scope)")
    return text


def extract_fields_rule_based(pdf_path: Path) -> dict[str, Any]:
    """Deterministic extraction: regex over pdfplumber's page text. No key, no LLM.

    Raises ExtractionError with a specific, actionable message if the PDF can't be
    opened, or if a required field's line is missing entirely -- this only works
    because the sample invoice's layout is fixed (see generate_sample_invoice.py);
    a rule-based extractor is inherently template-specific, which is exactly why
    Section 3's LLM path exists for invoices this chapter has never seen.
    """
    text = _read_pdf_text(pdf_path)
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    fields: dict[str, Any] = {}
    line_items: list[dict[str, Any]] = []
    in_items = False
    for line in lines:
        if line == _LINE_ITEM_HEADER:
            continue
        if _SEPARATOR_RE.match(line):
            in_items = not in_items  # first dash line opens the table, second closes it
            continue
        if in_items:
            m = _LINE_ITEM_RE.match(line)
            if not m:
                raise ExtractionError(f"could not parse line-item row: {line!r}")
            line_items.append(
                {
                    "description": m.group("description"),
                    "quantity": int(m.group("quantity")),
                    "unit_price": m.group("unit_price").replace(",", ""),
                    "amount": m.group("amount").replace(",", ""),
                }
            )
            continue
        for field, pattern in _FIELD_PATTERNS.items():
            m = pattern.match(line)
            if m:
                fields[field] = m.group("value").strip().replace(",", "") if field in (
                    "subtotal",
                    "tax",
                    "total",
                ) else m.group("value").strip()
                break

    # Vendor name and address are the two lines above everything else in this
    # template -- there's no label to regex against, so they're positional. Only
    # safe because generate_sample_invoice.py's layout is fixed; a different
    # vendor's template would need its own rule (or the LLM path).
    if len(lines) < 2:
        raise ExtractionError("PDF text has fewer than 2 lines; cannot locate vendor name/address")
    fields["vendor_name"] = lines[0]
    fields["vendor_address"] = lines[1]
    fields["line_items"] = line_items

    required = [
        "vendor_name",
        "vendor_address",
        "invoice_number",
        "invoice_date",
        "bill_to",
        "subtotal",
        "tax",
        "total",
    ]
    missing = [f for f in required if f not in fields]
    if missing or not line_items:
        all_missing = missing + ([] if line_items else ["line_items"])
        raise ExtractionError(
            f"{pdf_path}: rule-based extraction could not find required field(s): {all_missing}"
        )
    return fields


# --- LLM-based (KEY-GATED, makes a real, billed API call when a key is present) ---


def _record_invoice_tool_schema() -> dict[str, Any]:
    """Derive the Anthropic tool schema from the SAME pydantic model extraction
    output must satisfy, via BaseModel.model_json_schema() -- the current pydantic
    v2 API for JSON Schema generation (grounded 2026-09-03, pydantic docs). Reusing
    it means the LLM is asked for exactly the shape schema.py will validate, instead
    of a hand-duplicated schema the two could drift apart from.
    """
    from schema import Invoice  # local import: only needed on this key-gated path

    invoice_schema = Invoice.model_json_schema()
    return {
        "name": "record_invoice",
        "description": "Record the structured fields extracted from an invoice PDF.",
        "input_schema": invoice_schema,
    }


def extract_fields_llm(pdf_path: Path) -> dict[str, Any]:
    """LLM extraction via Claude's tool-use API -- KEY-GATED, same bridge shape as
    code/mcp_db/llm_client.py. Requires ANTHROPIC_API_KEY, ANTHROPIC_MODEL, and the
    `anthropic` package (not one of this chapter's pinned deps -- `pip install
    anthropic` yourself to exercise this path). With no key set, returns None
    immediately and makes no network call -- callers should fall back to
    extract_fields_rule_based(), which is exactly what run.py does.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "ANTHROPIC_API_KEY is not set -- skipping the LLM extraction path.\n"
            "Set it (and ANTHROPIC_MODEL) in .env to exercise this. No network call made."
        )
        return None

    model = os.environ.get("ANTHROPIC_MODEL")
    if not model:
        print("ANTHROPIC_API_KEY is set but ANTHROPIC_MODEL is not -- skipping.")
        return None

    try:
        import anthropic
    except ImportError:
        print("The 'anthropic' package is not installed -- run `pip install anthropic` to use this path.")
        return None

    text = _read_pdf_text(pdf_path)
    tool = _record_invoice_tool_schema()
    client = anthropic.Anthropic(api_key=api_key)
    # anthropic SDK call shape (client.messages.create with tools=[...] and a
    # forced tool_choice) grounded in NOTE-AGENT-4-provider-sdks.md, section
    # "Minimal Chat API Signatures" (checked 2026-09-02).
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        tools=[tool],
        tool_choice={"type": "tool", "name": "record_invoice"},
        messages=[
            {
                "role": "user",
                "content": (
                    "Extract the invoice fields from this text and call "
                    f"record_invoice with them:\n\n{text}"
                ),
            }
        ],
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == "record_invoice":
            return dict(block.input)
    raise ExtractionError("Claude did not call record_invoice -- no structured output returned")


if __name__ == "__main__":
    sample = Path(__file__).parent / "datasets" / "sample_invoice.pdf"
    result = extract_fields_rule_based(sample)
    print(json.dumps(result, indent=2))

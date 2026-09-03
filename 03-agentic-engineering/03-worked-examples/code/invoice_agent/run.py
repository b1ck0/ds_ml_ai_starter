"""End-to-end invoice agent: PDF -> extract -> validate -> MCP insert -> verify row.

This is the file the chapter's acceptance criterion (AC2) points at: every step
below runs with the deterministic, rule-based extractor -- no API key, no network
call -- and every printed value comes from a real run against a real SQLite
database through a real MCP tool call, the same "no LLM required" discipline as
code/mcp_db/test_client.py in the earlier MCP chapter.

Steps 1-4 are the happy path (LO1-LO3). Steps 5-7 are the robustness checks LO4
asks for: idempotency (don't double-insert), a missing required field, and a
malformed PDF -- each one caught and reported, not a silent failure.

Run:
    .venv-agent/Scripts/python.exe run.py
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastmcp import Client
from pydantic import ValidationError

from extract import ExtractionError, extract_fields_rule_based
from mcp_write_tool import DB_PATH
from schema import Invoice

HERE = Path(__file__).parent
SERVER_PATH = HERE / "mcp_write_tool.py"
SAMPLE_PDF = HERE / "datasets" / "sample_invoice.pdf"


async def main() -> None:
    # Fresh database on every run, so the transcript is reproducible and Step 5's
    # idempotency check always demonstrates a real duplicate, never a leftover one
    # from a previous run -- same discipline as mcp_db/seed.py dropping and
    # recreating its tables every time.
    if DB_PATH.exists():
        DB_PATH.unlink()

    if not SAMPLE_PDF.exists():
        from generate_sample_invoice import main as generate_sample

        generate_sample()

    print("=== Step 1: extract (deterministic, rule-based, no API key) ===")
    raw = extract_fields_rule_based(SAMPLE_PDF)
    print(json.dumps(raw, indent=2))

    print("\n=== Step 2: validate against schema.Invoice ===")
    invoice = Invoice(**raw)
    print(f"validated OK: invoice_number={invoice.invoice_number} total={invoice.total}")

    # model_dump(mode="json") turns Decimal -> str and date -> "YYYY-MM-DD" -- the
    # same JSON-safe primitives the MCP tool call below must send over JSON-RPC.
    payload = invoice.model_dump(mode="json")

    async with Client(SERVER_PATH) as client:
        print("\n=== Step 3: MCP insert_invoice() ===")
        result = await client.call_tool("insert_invoice", payload)
        print(json.dumps(result.data, indent=2))
        assert result.data["status"] == "inserted"

        print("\n=== Step 4: MCP get_invoice() -- verify the row landed ===")
        verify = await client.call_tool("get_invoice", {"invoice_number": invoice.invoice_number})
        print(json.dumps(verify.data, indent=2))
        assert verify.data["invoice"]["invoice_number"] == invoice.invoice_number
        assert len(verify.data["line_items"]) == len(invoice.line_items)

        print("\n=== Step 5: robustness -- idempotency (re-insert same invoice_number) ===")
        dup = await client.call_tool("insert_invoice", payload)
        print(json.dumps(dup.data, indent=2))
        assert dup.data["status"] == "duplicate"
        assert dup.data["invoice_id"] == result.data["invoice_id"]

    print("\n=== Step 6: robustness -- missing required field ===")
    broken_raw = dict(raw)
    del broken_raw["total"]
    try:
        Invoice(**broken_raw)
    except ValidationError as exc:
        print(f"ValidationError raised as expected ({len(exc.errors())} error(s)):")
        for err in exc.errors():
            print(f"  - {'.'.join(str(p) for p in err['loc'])}: {err['msg']}")

    print("\n=== Step 7: robustness -- malformed PDF ===")
    bad_pdf = HERE / "datasets" / "not_a_pdf.pdf"
    bad_pdf.write_bytes(b"this is not a pdf file, just plain bytes")
    try:
        extract_fields_rule_based(bad_pdf)
    except ExtractionError as exc:
        print(f"ExtractionError raised as expected: {exc}")
    finally:
        bad_pdf.unlink(missing_ok=True)

    print("\n=== Done: end-to-end flow (parse -> validate -> MCP insert -> verify) ===")
    print(f"exactly one row for {invoice.invoice_number} exists in {DB_PATH.name} "
          f"(Step 5 proved the retry did not create a second one).")


if __name__ == "__main__":
    asyncio.run(main())

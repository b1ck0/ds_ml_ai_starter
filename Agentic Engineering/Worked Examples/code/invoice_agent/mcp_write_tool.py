"""FastMCP server exposing a WRITE-capable, validated, parameterised insert tool.

This is code/mcp_db/server.py's read-only pattern
(../mcp_db/server.py -- see the MCP-Database-Query-Layer chapter) extended with the
one thing that chapter deliberately left out: a tool that mutates data. The two
disciplines that made the read-only server safe still apply, plus a third that only
matters once you can write:

    1. Allowlist identifiers, parameterise values  -- same as server.py; every SQL
       string below either contains only hard-coded table/column names or `?`
       placeholders, never caller-supplied text spliced into the query.
    2. Validate at the boundary                    -- the tool re-runs the caller's
       arguments through schema.py's Invoice model BEFORE touching the database, so
       this server is safe to call even from a client that skipped validation
       itself (never trust the caller -- the same reason a REST controller
       re-validates a request body even if the frontend already did).
    3. Idempotency                                  -- invoice_number is UNIQUE; the
       tool checks for an existing row before inserting and returns a "duplicate"
       status instead of a second row. Without this, re-processing the same PDF
       (a retry, a duplicate upload, an agent looping) double-counts revenue.

Two tools:
    insert_invoice(vendor_name, vendor_address, invoice_number, invoice_date,
                    bill_to, line_items, subtotal, tax, total) -> dict
    get_invoice(invoice_number) -> dict                    # read-only, verifies a row

Run directly for manual testing (stdio transport -- see run.py for how a client
launches this file as a subprocess):
    .venv-agent/Scripts/python.exe mcp_write_tool.py
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from pydantic import ValidationError

from schema import Invoice

DB_PATH = Path(__file__).parent / "invoices.db"

mcp = FastMCP("invoice_write_layer")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """Create the schema if it doesn't exist yet. Safe to call on every startup.

    Money columns (unit_price, amount, subtotal, tax, total) are TEXT, not REAL --
    sqlite's REAL is an IEEE-754 double, and storing "27.20" as a float re-introduces
    the exact rounding error pydantic's Decimal fields were chosen to avoid (the
    same reason you'd reach for java.math.BigDecimal, not double, for currency in
    Java). TEXT preserves the validated decimal string byte-for-byte.
    """
    conn = _connect()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS invoices (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                invoice_number  TEXT NOT NULL UNIQUE,
                vendor_name     TEXT NOT NULL,
                vendor_address  TEXT NOT NULL,
                invoice_date    TEXT NOT NULL,
                bill_to         TEXT NOT NULL,
                subtotal        TEXT NOT NULL,
                tax             TEXT NOT NULL,
                total           TEXT NOT NULL,
                created_at      TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS invoice_line_items (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                invoice_id  INTEGER NOT NULL REFERENCES invoices(id),
                description TEXT NOT NULL,
                quantity    INTEGER NOT NULL,
                unit_price  TEXT NOT NULL,
                amount      TEXT NOT NULL
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


@mcp.tool()
def insert_invoice(
    vendor_name: str,
    vendor_address: str,
    invoice_number: str,
    invoice_date: str,
    bill_to: str,
    line_items: list[dict[str, Any]],
    subtotal: str,
    tax: str,
    total: str,
) -> dict[str, Any]:
    """Validate and persist one invoice. Idempotent on invoice_number.

    Args:
        vendor_name: issuing vendor's name.
        vendor_address: issuing vendor's address.
        invoice_number: vendor-assigned invoice number. Unique key -- calling this
            tool twice with the same invoice_number does NOT insert a second row;
            it returns {"status": "duplicate", ...} instead.
        invoice_date: ISO date string, "YYYY-MM-DD".
        bill_to: the customer being billed.
        line_items: list of {description, quantity, unit_price, amount}.
        subtotal: sum of line item amounts, as a decimal string.
        tax: tax amount, as a decimal string.
        total: subtotal + tax, as a decimal string.

    Every argument is re-validated against schema.Invoice before any SQL runs --
    this tool never trusts that the caller already validated (it might be a hand
    -written test, a different extractor, or an LLM that skipped a step).
    """
    try:
        invoice = Invoice(
            vendor_name=vendor_name,
            vendor_address=vendor_address,
            invoice_number=invoice_number,
            invoice_date=invoice_date,
            bill_to=bill_to,
            line_items=line_items,
            subtotal=subtotal,
            tax=tax,
            total=total,
        )
    except ValidationError as exc:
        return {"status": "rejected", "errors": exc.errors()}

    conn = _connect()
    try:
        existing = conn.execute(
            "SELECT id FROM invoices WHERE invoice_number = ?",
            (invoice.invoice_number,),
        ).fetchone()
        if existing is not None:
            return {
                "status": "duplicate",
                "invoice_id": existing["id"],
                "invoice_number": invoice.invoice_number,
                "message": "invoice_number already exists -- not inserted again",
            }

        cur = conn.execute(
            """
            INSERT INTO invoices
                (invoice_number, vendor_name, vendor_address, invoice_date,
                 bill_to, subtotal, tax, total)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                invoice.invoice_number,
                invoice.vendor_name,
                invoice.vendor_address,
                invoice.invoice_date.isoformat(),
                invoice.bill_to,
                str(invoice.subtotal),
                str(invoice.tax),
                str(invoice.total),
            ),
        )
        invoice_id = cur.lastrowid
        conn.executemany(
            """
            INSERT INTO invoice_line_items
                (invoice_id, description, quantity, unit_price, amount)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (invoice_id, item.description, item.quantity, str(item.unit_price), str(item.amount))
                for item in invoice.line_items
            ],
        )
        conn.commit()
    finally:
        conn.close()

    return {
        "status": "inserted",
        "invoice_id": invoice_id,
        "invoice_number": invoice.invoice_number,
        "line_item_count": len(invoice.line_items),
    }


@mcp.tool()
def get_invoice(invoice_number: str) -> dict[str, Any]:
    """Read back one invoice by number, with its line items. Read-only.

    Used to verify a row actually landed after insert_invoice() -- the same
    "don't trust the write, read it back" discipline as asserting on a row
    fetched fresh from the database after a JDBC INSERT, rather than trusting
    the caller's own copy of what it thinks it wrote.
    """
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM invoices WHERE invoice_number = ?", (invoice_number,)
        ).fetchone()
        if row is None:
            return {"error": f"no invoice found with invoice_number={invoice_number!r}"}
        items = conn.execute(
            "SELECT description, quantity, unit_price, amount FROM invoice_line_items "
            "WHERE invoice_id = ? ORDER BY id",
            (row["id"],),
        ).fetchall()
    finally:
        conn.close()
    return {
        "invoice": dict(row),
        "line_items": [dict(item) for item in items],
    }


if __name__ == "__main__":
    init_db()
    # show_banner=False keeps stdout/stderr clean when a client (like run.py)
    # launches this file as a stdio subprocess.
    mcp.run(transport="stdio", show_banner=False)

"""FastMCP server exposing a small SQLite database as a read-only tool boundary.

Three tools, each with an explicit, typed input schema (FastMCP generates the JSON
schema from the Python type hints -- see the chapter prose):

    list_tables()                                -> which entities exist
    describe_table(entity)                        -> columns + row count for one entity
    query(entity, filters=None, limit=20)          -> filtered rows, parameterised SQL

No tool accepts raw SQL, and no tool can touch a table or column that isn't in the
SCHEMA allowlist below -- see the chapter's Pitfalls section for why that matters.

Run directly for manual testing (stdio transport, the FastMCP default -- see
test_client.py for how a client launches this file as a subprocess):

    .venv-agent/Scripts/python.exe server.py
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

DB_PATH = Path(__file__).parent / "mcp_demo.db"

# --- Least-privilege allowlist ---------------------------------------------------
# The tools NEVER accept a raw table or column name from the caller and interpolate
# it into SQL. Every entity and every filterable column this server will ever touch
# is enumerated here. Anything not in this dict is rejected before a single line of
# SQL is built -- this is what makes the f-string below (which builds table/column
# names, never values) safe. Table/column names cannot be passed as `?` parameters
# in sqlite3, so allowlisting is the mechanism that replaces parameterisation for
# them.
SCHEMA: dict[str, list[str]] = {
    "customers": ["id", "name", "email", "country"],
    "orders": ["id", "customer_id", "item", "amount", "status"],
}

MAX_LIMIT = 100

mcp = FastMCP("db_query_layer")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@mcp.tool()
def list_tables() -> dict[str, Any]:
    """List the entities (tables) this server exposes, with their columns.

    Read-only, no arguments. Call this first to discover what can be queried --
    the server exposes nothing beyond this allowlist, so there is no way to
    accidentally (or maliciously) reach a table that isn't listed here.
    """
    return {"entities": SCHEMA}


@mcp.tool()
def describe_table(entity: str) -> dict[str, Any]:
    """Describe one entity: its columns and current row count.

    Args:
        entity: table name. Must be one of the entities returned by list_tables().
    """
    if entity not in SCHEMA:
        return {"error": f"unknown entity '{entity}'. Call list_tables() for the allowed list."}
    conn = _connect()
    try:
        # `entity` is safe to interpolate here only because it was just checked
        # against SCHEMA above -- it can never be arbitrary caller text.
        cur = conn.execute(f"SELECT COUNT(*) FROM {entity}")
        (count,) = cur.fetchone()
    finally:
        conn.close()
    return {"entity": entity, "columns": SCHEMA[entity], "row_count": count}


@mcp.tool()
def query(entity: str, filters: dict[str, Any] | None = None, limit: int = 20) -> dict[str, Any]:
    """Query one entity with optional equality filters. Read-only.

    Args:
        entity: table name. Must be one of the entities returned by list_tables().
        filters: optional {column: value} equality filters. Every column must be
            one of that entity's allowlisted columns (see list_tables()).
        limit: maximum rows to return, capped at 100 so a single call can never
            dump an entire table.
    """
    if entity not in SCHEMA:
        return {"error": f"unknown entity '{entity}'. Call list_tables() for the allowed list."}

    allowed_columns = set(SCHEMA[entity])
    filters = filters or {}
    bad_columns = set(filters) - allowed_columns
    if bad_columns:
        return {"error": f"unknown column(s) {sorted(bad_columns)} for entity '{entity}'"}

    limit = max(1, min(int(limit), MAX_LIMIT))

    # Table and column names below come only from SCHEMA (an allowlist keyed by the
    # already-validated `entity`), never from unvalidated caller text -- that is
    # what makes this f-string safe. Every VALUE, in contrast, goes through a `?`
    # placeholder and is never concatenated into the SQL text. See the chapter's
    # Pitfalls section for the injection this prevents and why the two cases (names
    # vs. values) need different defences.
    sql = f"SELECT * FROM {entity}"
    params: list[Any] = []
    if filters:
        where = " AND ".join(f"{col} = ?" for col in filters)
        sql += f" WHERE {where}"
        params = list(filters.values())
    sql += " LIMIT ?"
    params.append(limit)

    conn = _connect()
    try:
        cur = conn.execute(sql, params)
        rows = [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()
    return {"entity": entity, "sql": sql, "row_count": len(rows), "rows": rows}


if __name__ == "__main__":
    # show_banner=False keeps stdout/stderr clean when a client (like test_client.py
    # or llm_client.py) launches this file as a stdio subprocess.
    mcp.run(transport="stdio", show_banner=False)

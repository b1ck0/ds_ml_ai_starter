"""Direct MCP client test -- no LLM involved.

Connects to server.py over stdio (FastMCP launches the server file as a
subprocess), calls each tool, and asserts on the real rows that come back. This is
how you unit-test an MCP tool boundary without spending a single LLM token.

Run (after seeding the database):
    .venv-agent/Scripts/python.exe seed.py
    .venv-agent/Scripts/python.exe test_client.py
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastmcp import Client

SERVER_PATH = Path(__file__).parent / "server.py"


async def main() -> None:
    # Passing a Path (not a bare string) selects the stdio transport explicitly --
    # fastmcp 4.0.1 still infers stdio from a plain string but prints a
    # FastMCPDeprecationWarning telling you to do exactly this instead.
    async with Client(SERVER_PATH) as client:
        print("=== list_tables() ===")
        result = await client.call_tool("list_tables", {})
        tables = result.data
        print(json.dumps(tables, indent=2))
        assert "customers" in tables["entities"]
        assert "orders" in tables["entities"]

        print("\n=== describe_table(entity='orders') ===")
        result = await client.call_tool("describe_table", {"entity": "orders"})
        described = result.data
        print(json.dumps(described, indent=2))
        assert described["row_count"] == 8

        print("\n=== query(entity='customers', filters={'country': 'BG'}) ===")
        result = await client.call_tool(
            "query", {"entity": "customers", "filters": {"country": "BG"}}
        )
        customers = result.data
        print(json.dumps(customers, indent=2))
        assert customers["row_count"] == 1
        assert customers["rows"][0]["name"] == "Ana Petrova"

        print("\n=== query(entity='orders', filters={'status': 'shipped'}, limit=3) ===")
        result = await client.call_tool(
            "query",
            {"entity": "orders", "filters": {"status": "shipped"}, "limit": 3},
        )
        orders = result.data
        print(json.dumps(orders, indent=2))
        assert orders["row_count"] == 3
        assert all(row["status"] == "shipped" for row in orders["rows"])

        print("\n=== query(entity='orders', filters={'dropped_table': 'x'})  [rejected] ===")
        result = await client.call_tool(
            "query", {"entity": "orders", "filters": {"dropped_table": "x"}}
        )
        rejected = result.data
        print(json.dumps(rejected, indent=2))
        assert "error" in rejected

        print("\nAll assertions passed -- every row above came from the real SQLite")
        print("database via the real MCP tool call. No LLM was involved.")


if __name__ == "__main__":
    asyncio.run(main())

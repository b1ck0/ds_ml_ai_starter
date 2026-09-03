"""LLM client for the MCP database query layer -- KEY-GATED, makes a real API call.

Unlike test_client.py (which calls tools with fixed, hand-written arguments),
this script hands the MCP server's tool schemas to an LLM (Anthropic Claude, via
its tool-use / "function calling" API) and lets the model itself decide which
tool to call, with which arguments, based on a natural-language question. That is
the whole point of MCP: the server publishes typed tools once, and any MCP-aware
client -- a fixed test script or an LLM -- can drive them the same way.

This file is DELIBERATELY separate from the runnable, no-key path of this chapter:

* It requires a real ANTHROPIC_API_KEY (see .env.example) and, when a key is
  present, makes a real, billed call to the Anthropic API.
* It requires the `anthropic` package, which is NOT one of this chapter's pinned,
  verified dependencies (only fastmcp 4.0.1 is -- see NOTE-AGENT-3). Install it
  yourself with `pip install anthropic` if you want to run this path.
* The exact Anthropic Messages API field names below (tool_use, tool_result,
  input_schema, messages.create(...)) match the documented shape of that API,
  confirmed 2026-09-03 against
  https://platform.claude.com/docs/en/api/messages (docs.anthropic.com/en/api/messages
  redirects there as of this check). Re-check it against your installed
  `anthropic` version before relying on this in anything real -- the API
  surface can change between when this was written and when you run it.
* The model id is read from an environment variable rather than hard-coded, on
  purpose: model ids change over time, and this chapter will not assert one from
  memory. Set ANTHROPIC_MODEL to a current model id from the Anthropic docs.

With no key set, this script makes no network call at all -- it prints what it
needs and exits. That "no key" output IS captured for real in the chapter (see
artefacts/llm_client_no_key_transcript.txt); the LLM's tool-calling transcript
itself is not, because reproducing it here would mean fabricating what a live
model would say, which this chapter does not do.

Run:
    .venv-agent/Scripts/python.exe llm_client.py
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from fastmcp import Client

SERVER_PATH = Path(__file__).parent / "server.py"


def _mcp_tool_to_anthropic_schema(tool: object) -> dict:
    """Translate one FastMCP tool definition into an Anthropic `tools=[...]` entry.

    This is the "bridge": MCP describes a tool as {name, description, input_schema}
    (JSON Schema) -- see server.py's tool docstrings and type hints, which FastMCP
    turned into exactly this shape (confirmed by inspecting a real server -- see
    the chapter prose). Anthropic's tool-use API wants the same three fields under
    the same names, so the translation is a straight pass-through here. A
    different LLM provider's tool-calling API would need the same three pieces of
    information, just re-shaped into its own field names.
    """
    return {
        "name": tool.name,
        "description": tool.description or "",
        "input_schema": tool.input_schema,
    }


async def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "ANTHROPIC_API_KEY is not set -- skipping the live LLM call.\n"
            "Set it in .env (never commit it) and set ANTHROPIC_MODEL to a current\n"
            "model id, then re-run to see Claude discover and call the query tool\n"
            "itself. This script makes no network call without a key."
        )
        return

    model = os.environ.get("ANTHROPIC_MODEL")
    if not model:
        print(
            "ANTHROPIC_API_KEY is set but ANTHROPIC_MODEL is not.\n"
            "Set ANTHROPIC_MODEL to a current Claude model id (see the Anthropic\n"
            "docs) -- this chapter does not hard-code one, since model ids change."
        )
        return

    try:
        import anthropic
    except ImportError:
        print(
            "The 'anthropic' package is not installed in this environment.\n"
            "Install it with: pip install anthropic\n"
            "(Not one of this chapter's pinned dependencies -- only needed for\n"
            "this optional, key-gated LLM path.)"
        )
        return

    client = anthropic.Anthropic(api_key=api_key)

    async with Client(SERVER_PATH) as mcp_client:
        mcp_tools = await mcp_client.list_tools()
        anthropic_tools = [_mcp_tool_to_anthropic_schema(t) for t in mcp_tools]

        messages: list[dict] = [
            {
                "role": "user",
                "content": (
                    "Which orders for customers in Bulgaria are shipped? "
                    "Use the tools you have to find out, then answer in one sentence."
                ),
            }
        ]

        # Bridge loop: ask Claude, execute any tool_use blocks against the real MCP
        # server, feed the results back as tool_result blocks, repeat until Claude
        # answers in plain text instead of asking for another tool call.
        while True:
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                tools=anthropic_tools,
                messages=messages,
            )
            messages.append({"role": "assistant", "content": response.content})

            tool_uses = [block for block in response.content if block.type == "tool_use"]
            if not tool_uses:
                for block in response.content:
                    if block.type == "text":
                        print(block.text)
                break

            tool_results = []
            for call in tool_uses:
                print(f"[Claude called] {call.name}({json.dumps(call.input)})")
                result = await mcp_client.call_tool(call.name, call.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": json.dumps(result.data),
                    }
                )
            messages.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    asyncio.run(main())

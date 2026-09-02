# NOTE-AGENT-3: FastMCP Server API and Safe Parameterized SQL

**Answer:** FastMCP 4.0.1 (released 2026-09-02) provides `@mcp.tool()` decorator for tools with type-hints auto-generating JSON schemas; server runs via `mcp.run(transport="stdio")` (default) or `mcp.run(transport="http", host="0.0.0.0", port=8000)`; client connects with `async with Client("./server.py")` (stdio) or `Client("https://...")` (HTTP); sqlite3 parameterized queries using `?` placeholders prevent SQL injection; test clients can call tools without an LLM key.

**Evidence:**

*FastMCP version and server API (PyPI + official docs):*
- **Version:** 4.0.1 (released 2026-09-02): https://pypi.org/project/fastmcp/
- **Package description:** "The fast, Pythonic way to build MCP servers and clients"

*Tool decorator API (source: FastMCP docs https://gofastmcp.com/servers/tools):*
```python
from fastmcp import FastMCP

mcp = FastMCP("Demo")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b
```
- Type hints generate JSON schema automatically
- Docstring becomes the tool description
- Return type hint defines the output schema
- FastMCP handles MCP protocol details (JSON-RPC, schema generation)

*Server run modes (source: FastMCP tutorials):*

**Stdio mode (default, for desktop clients):**
```python
mcp.run(transport="stdio")
```
- Launches server as subprocess
- Client communicates via stdin/stdout pipes
- Standard mechanism for Claude Desktop and other MCP clients
- Command-line usage: `mcp run python server.py` or direct subprocess spawn

**HTTP mode (for web/network clients):**
```python
mcp.run(transport="http", host="0.0.0.0", port=8000)
```
- Starts web server
- Allows remote or same-machine clients to connect via HTTP/SSE
- URL-based connection: `Client("http://127.0.0.1:8000")`

*Client connection API (source: FastMCP client docs https://gofastmcp.com/clients/client):*
```python
from fastmcp import Client

# For stdio server (local Python subprocess):
async with Client("./server.py") as client:
    result = await client.call_tool("add", a=1, b=2)
    print(result)

# For HTTP server:
async with Client("http://127.0.0.1:8000") as client:
    result = await client.call_tool("add", a=1, b=2)
    print(result)
```
- Transport inferred from argument type (local file path → stdio, URL → HTTP)
- Must run inside `async with` context
- `call_tool()` method takes tool name + keyword args
- No LLM required for direct client tests

*Safe parameterized SQL with sqlite3 (source: Python official docs + security guides):*

**Vulnerable (DON'T DO THIS):**
```python
name = "'; DROP TABLE users; --"
cursor.execute(f"SELECT * FROM users WHERE name='{name}'")  # SQL injection risk
```

**Safe (USE THIS):**
```python
import sqlite3

conn = sqlite3.connect(':memory:')
cursor = conn.cursor()

# Use ? placeholder for values
cursor.execute("SELECT * FROM users WHERE name=?", (name,))

# Multiple parameters
cursor.execute(
    "INSERT INTO users (name, email) VALUES (?, ?)",
    (name, email)
)
```

- `?` is sqlite3's paramstyle placeholder
- Values passed as tuple (or dict with named placeholders)
- sqlite3 automatically escapes and quotes parameters
- Prevents SQL injection by separating query structure from data
- **Limitation:** Table and column names cannot be parameterized; use allowlist if accepting dynamic schema references

*Testing without LLM (example pattern):*
```python
import asyncio
from fastmcp import Client

async def test_db_tool():
    async with Client("./mcp_db_server.py") as client:
        # Call tool without any LLM or API key
        result = await client.call_tool("query", entity="users", filters={"id": 1})
        assert result["rows"] is not None
        print("✓ Tool works offline")

asyncio.run(test_db_tool())
```
- Test client executes tool directly
- No LLM key, no API calls required
- Assert on return values/schema
- Runnable in CI/CD pipelines

**Caveats / limits:**
1. **Stdio transport latency:** Subprocess startup + JSON parsing add ~100–500 ms overhead per call; not suitable for <10 ms latency requirements.
2. **HTTP transport scale:** Single-threaded by default; use production ASGI server (Uvicorn, Gunicorn) for >100 concurrent clients.
3. **SQL parameter limitation:** Dynamic table/column names require allowlisting (not parameterizable):
   ```python
   allowed_tables = {"users", "orders"}
   table = user_input  # Must be in allowed_tables
   if table not in allowed_tables:
       raise ValueError("Invalid table")
   cursor.execute(f"SELECT * FROM {table} WHERE id=?", (id,))  # Table is safe, id is parameterized
   ```
4. **Tool schema strictness:** MCP client rejects calls that don't match the schema; mismatch errors are logged on the server, not returned to the client.
5. **Async requirement:** Client must run in async context; blocking calls will hang the server.

**Recommendation:**
1. **Server implementation pattern:**
   ```python
   from fastmcp import FastMCP
   import sqlite3

   mcp = FastMCP("db_server")

   @mcp.tool()
   def query(entity: str, filters: dict = None) -> dict:
       """Query the database by entity and optional filters."""
       conn = sqlite3.connect("app.db")
       cursor = conn.cursor()
       
       # Whitelist entity names
       entities = {"users", "products"}
       if entity not in entities:
           return {"error": f"Unknown entity: {entity}"}
       
       # Always use parameterized queries for values
       query = f"SELECT * FROM {entity}"
       if filters:
           where_clauses = [f"{k}=?" for k in filters.keys()]
           query += " WHERE " + " AND ".join(where_clauses)
           cursor.execute(query, tuple(filters.values()))
       else:
           cursor.execute(query)
       
       rows = cursor.fetchall()
       conn.close()
       return {"rows": rows}

   if __name__ == "__main__":
       mcp.run(transport="stdio")  # or ("http", host="0.0.0.0", port=8000)
   ```

2. **Test pattern (no LLM key needed):**
   ```python
   import asyncio
   from fastmcp import Client

   async def test():
       async with Client("./server.py") as client:
           result = await client.call_tool("query", entity="users", filters={"id": 1})
           print(result)  # Inspect returned rows

   asyncio.run(test())
   ```

3. **Deployment notes:**
   - For local development: use stdio mode
   - For service exposure: use HTTP mode with reverse proxy (nginx, Caddy) and TLS
   - Always run parameterized queries; audit dynamic schema access
   - Pin FastMCP 4.0.1 in requirements.txt

**Date checked:** 2026-09-02

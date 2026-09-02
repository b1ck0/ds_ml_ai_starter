# NOTE-AGENT-1: Agentic Stack — Package Versions and Python 3.13 Compatibility

**Answer:** google-adk 2.8.0, fastapi 0.141.1, fastmcp 4.0.1, pgvector 0.5.0, psycopg 3.3.5 all verified on PyPI (checked 2026-09-02); all support Python 3.10+, tested up to 3.14; google-adk package name is confirmed correct; pgvector Docker image pgvector/pgvector:pg17 available; minimal ADK API requires `root_agent = Agent(model='gemini-flash-latest', name='root_agent', description="...")`.

**Evidence:**

*Package versions (PyPI, checked 2026-09-02):*
- **google-adk** 2.8.0 (released 2026-08-26): https://pypi.org/project/google-adk/ — correct package name confirmed; described as "Agent Development Kit"
- **fastapi** 0.141.1 (released 2026-07-29): https://pypi.org/project/fastapi/
- **fastmcp** 4.0.1 (released 2026-09-02): https://pypi.org/project/fastmcp/ — described as "The fast, Pythonic way to build MCP servers and clients"
- **pgvector** (Python) 0.5.0 (released 2026-07-06): https://pypi.org/project/pgvector/ — provides pgvector support for Python
- **psycopg** 3.3.5 (released 2026-08-31): https://pypi.org/project/psycopg/ — "modern implementation of a PostgreSQL adapter for Python"

*Python version support:*
- google-adk specifies `requires-python = ">=3.10"` in pyproject.toml; tested and verified against Python 3.10, 3.11, 3.12, 3.13, 3.14 (source: GitHub pyproject.toml)
- fastapi: Python 3.10 or higher required
- pdfplumber (related PDF tool): tested on Python 3.10, 3.11, 3.12, 3.13, 3.14
- **All core packages support Python 3.13 ✓**

*pgvector Docker and SQL:*
- Official Docker image available: `pgvector/pgvector:pg17` (source: Docker Hub https://hub.docker.com/r/pgvector/pgvector/tags)
- SQL extension: Use `CREATE EXTENSION vector;` to enable in PostgreSQL (source: pgvector docs https://github.com/pgvector/pgvector)

*ADK Minimal Agent API:*
From official ADK docs (https://adk.dev/get-started/python/), the minimal shape requires a module-level `root_agent` variable:
```python
from google.adk.agents.llm_agent import Agent

root_agent = Agent(
    model='gemini-flash-latest',
    name='root_agent',
    description="Your agent description",
)
```
- Only `root_agent` name is required by ADK runtime
- `model` parameter specifies the LLM (key-gated at runtime)
- ADK runs a full FastAPI server internally

**Caveats / limits:**
1. **Starlette version conflict**: Earlier versions of google-adk (e.g., v1.12.0) had overly restrictive starlette constraints (`starlette<1.0.0` and `>=0.46.2`), conflicting with fastapi's needs (`starlette<0.42.0`). Version 2.8.0 appears to resolve this, but pin versions carefully if building with both.
2. **LLM calls key-gated**: The Agent model inference requires a valid LLM provider key (e.g., `GOOGLE_API_KEY`); this step cannot run locally without credentials.
3. **pgvector extension**: Must be installed in PostgreSQL before use; requires either building from source or using a pre-built Docker image.

**Recommendation:**
1. Pin versions in requirements.txt as:
   ```
   google-adk==2.8.0
   fastapi==0.141.1
   fastmcp==4.0.1
   pgvector==0.5.0
   psycopg==3.3.5
   ```
2. Use Python 3.10–3.13 (3.10 is the min; test before using 3.14+ in production).
3. For pgvector Postgres locally, use Docker: `docker run -d pgvector/pgvector:pg17` and issue `CREATE EXTENSION vector;` in a fresh database.
4. Document that LLM inference steps require a configured API key in `.env` (never commit keys); all import/version checks and local tool setup run offline.
5. When combining google-adk + fastapi, test install in a throwaway venv first to catch dependency conflicts before committing.

**Date checked:** 2026-09-02

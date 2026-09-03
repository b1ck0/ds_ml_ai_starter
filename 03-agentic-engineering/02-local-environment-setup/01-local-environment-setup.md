# Local Environment Setup (Agentic Engineering)

*Agentic Engineering · Local Environment Setup · SPEC-AGENT-0*

## The dependency conflict that resolved cleanly — this time

Run this exact command against this project's environment today, and it succeeds with zero conflicts:
install `google-adk==2.8.0` into a venv that already has `fastapi==0.141.1` sitting in it (Section 2.3
runs it for real, output and all). That's not something you get to take for granted, though — it's a
coin flip that happened to land right, and this project's own dependency history proves it. An earlier
release, `google-adk==1.12.0`, pinned `starlette<1.0.0,>=0.46.2` while `fastapi` needed
`starlette<0.42.0` at the time — two ranges with **no version number that satisfies both**
([NOTE-AGENT-1-stack.md](../../research/NOTE-AGENT-1-stack.md), caveat 1). That's not a hypothetical
risk; it's a resolver conflict that genuinely happened, on this exact library pairing, one version back.

Every Java engineer has felt this shape of failure before, just with a different toolchain's error
message: two libraries in a `pom.xml`, each pinning an incompatible range of the same transitive
dependency, and Maven simply cannot produce a dependency graph that satisfies both — the build fails
loudly, naming the conflict. `pip` hits the identical wall when two packages disagree on a shared
transitive pin; the only difference is which tool is doing the resolving.

```mermaid
flowchart LR
    OLD["google-adk==1.12.0<br/>needs starlette&lt;1.0.0,&gt;=0.46.2"] -->|"vs fastapi's<br/>starlette&lt;0.42.0"| CONFLICT["no version satisfies both --<br/>a real resolver conflict"]
    NEW["google-adk==2.8.0<br/>needs starlette&lt;2,&gt;=1.3.1"] -->|"vs installed<br/>starlette==1.6.0"| CLEAN["satisfied --<br/>pip install --dry-run proves it, Section 2.3"]
```

[`theory.md`](../01-theory/01-theory.md) (SPEC-AGENT-1) already explained *why* an agent needs RAG and
MCP at all: LLMs are stateless, bounded, and frozen at a training cutoff, so anything resembling
"memory" or "the ability to act" has to be bolted on from outside the model. This chapter is where you
stand up the four pieces that bolting-on actually requires, for real, on your own machine — and the
dependency story above is your first real taste of a theme running through the whole chapter: these
pieces are genuinely independent moving parts, not one framework, and treating them that way (separate
virtual environments, a defensive default, real dry-run verification instead of "it'll probably be
fine") is what Section 2 is built around.

```mermaid
flowchart LR
    FA["FastAPI<br/>the HTTP boundary"] --> ADK["ADK runtime<br/>orchestrates the request"]
    ADK -->|"tool call"| MCP["FastMCP<br/>standardised tool protocol"]
    ADK -->|"memory lookup"| PGV["pgvector on Postgres<br/>nearest-neighbour store"]
    ADK -->|"model inference<br/>(the ONLY key-gated arrow)"| LLM["LLM provider<br/>Gemini / Anthropic / OpenAI"]
```

Every install and every query behind those first four boxes runs locally, with no LLM API key. Only the
last arrow — the actual model-inference call — leaves your machine, and it's clearly marked wherever it
comes up (Section 4).

## Environment

```text
google-adk==2.8.0     (its own virtual environment, .venv-adk -- Section 2.3 explains why)
fastapi==0.141.1
fastmcp==4.0.1
pgvector==0.5.0
psycopg==3.3.5
pydantic==2.13.5
Python 3.13.7 (.venv-agent, plus a separate .venv-adk for google-adk)
Docker version 28.4.0 (build d8eb465) -- pgvector/pgvector:pg17
```

Every version above is installed and verified live in this project's environments — `pip show` /
`importlib.metadata`, checked 2026-09-03 — matching the pins in
[research/NOTE-AGENT-1-stack.md](../../research/NOTE-AGENT-1-stack.md) (PyPI, checked 2026-09-02). The
full commands are in Sections 2–3; captured output lives in
[`artefacts/verify_agentic_env_output.txt`](artefacts/verify_agentic_env_output.txt),
[`artefacts/adk_venv_install_output.txt`](artefacts/adk_venv_install_output.txt), and
[`artefacts/pgvector_docker_output.txt`](artefacts/pgvector_docker_output.txt).

## 1. What & why — the agent stack vs. the web backend you already know

You've built every *role* in this stack before under different names. The unfamiliar part is that an
agent framework puts a non-deterministic decision-maker — the LLM — in the middle of the request, where
you're used to your own code being the only thing making decisions:

| Piece in a Java web shop | Piece in the agent stack | What it does |
|---|---|---|
| A Spring Boot `@RestController` | **FastAPI** | The HTTP boundary: routes, request/response models, the thing an external client actually calls. |
| Your service/business-logic layer | **ADK runtime** (`root_agent = Agent(...)`) | Orchestrates a request — but *how* it's handled is decided at run time by the LLM's output, not by code you wrote branch-by-branch. |
| An internal RPC contract between microservices (an OpenAPI schema, a `.proto` file) | **MCP** (built with FastMCP) | A standard protocol for "here are the tools/data this service exposes" — write one server, any MCP-speaking host can use it, instead of a bespoke integration per caller. |
| Hibernate/JPA + Postgres, queried by primary key or an indexed column | **psycopg + pgvector** | Also Postgres, also queried from Python — but the query shape is "give me the K *nearest* rows," not "give me the row where `id = 42`." |
| A paid third-party HTTPS API you call with an API key (Stripe, Twilio) | **The LLM provider** (Gemini, Anthropic, OpenAI, …) | Stateless, billed per call, and the one piece in this whole stack that needs a secret credential to actually run. |

The load-bearing difference: in a normal backend, *your code* decides which branch executes. In the agent
stack, the ADK runtime hands control to the LLM for that decision — "does this request need a tool call,
a memory lookup, or can I just answer" — and only the LLM call itself is non-deterministic and key-gated.
Every other piece in the table (FastAPI, MCP, pgvector) is exactly as deterministic and locally testable as
the Java-world thing in the row next to it, which is precisely why this chapter can verify all of them
without ever touching an LLM provider.

```mermaid
flowchart LR
    subgraph JAVA["A normal backend -- your code decides"]
        REQ1["request"] --> CODE["if/else,<br/>your own branches"] --> RESP1["response"]
    end
    subgraph AGENT["The agent stack -- the LLM decides"]
        REQ2["request"] --> ADKR["ADK runtime hands the<br/>decision to the LLM"] --> DECIDE{"tool call, memory<br/>lookup, or just answer?"}
        DECIDE --> RESP2["response"]
    end
```

## 2. The Python stack — install, verify, and why ADK lives in its own venv

### 2.1 Two virtual environments, not one

This project keeps two separate virtualenvs for Agentic Engineering:

- **`.venv-agent`** — the shared stack used across this subject's chapters: `fastapi`, `fastmcp`,
  `pgvector`, `psycopg`, `pydantic`, plus the embedding/RAG tooling `theory.md` used
  (`sentence-transformers`, `pdfplumber`, …).
- **`.venv-adk`** — `google-adk` alone, in its own isolated environment.

```mermaid
flowchart LR
    subgraph AGENTV[".venv-agent -- the shared stack"]
        FA2["fastapi"]
        FM["fastmcp"]
        PGV2["pgvector (client)"]
        PSY["psycopg"]
        PYD["pydantic"]
        ST["sentence-transformers, pdfplumber, ..."]
    end
    subgraph ADKV[".venv-adk -- isolated, on purpose"]
        ADKPKG["google-adk alone"]
    end
```

If you've read [`Data Science/Local Environment Setup/local-environment-setup.md`](../../01-data-science/02-local-environment-setup/01-local-environment-setup.md)
(SPEC-DS-0), you already know *why* a venv exists at all — an isolated `site-packages`, the Python
equivalent of a Maven module's own dependency set. What's new here is *why this project uses two of them
for one subject* instead of one shared venv for everything, the way DS and ML each use a single venv.

### 2.2 Install the shared stack (`.venv-agent`)

```bash
python -m venv .venv-agent
.venv-agent\Scripts\activate      # Windows PowerShell/cmd
# source .venv-agent/bin/activate   # macOS/Linux
pip install fastapi==0.141.1 fastmcp==4.0.1 pgvector==0.5.0 psycopg==3.3.5 pydantic==2.13.5
```

Every pin matches [NOTE-AGENT-1-stack.md](../../research/NOTE-AGENT-1-stack.md) (checked against PyPI
2026-09-02): [fastapi](https://pypi.org/project/fastapi/), [fastmcp](https://pypi.org/project/fastmcp/),
[pgvector](https://pypi.org/project/pgvector/) (the Python client — not to be confused with the Postgres
*extension* of the same name, covered in Section 3), and [psycopg](https://pypi.org/project/psycopg/), a
modern PostgreSQL adapter for Python (roughly this ecosystem's JDBC driver — the thing that actually
speaks the Postgres wire protocol underneath `psycopg.connect(...)`).

### 2.3 Install `google-adk` — in its own venv, and why

Install ADK into a **separate** environment:

```bash
python -m venv .venv-adk
.venv-adk\Scripts\activate
pip install google-adk==2.8.0
```

The cold open at the top of this chapter already named the risk this isolation defends against:
`google-adk==1.12.0` and `fastapi` once pinned genuinely incompatible `starlette` ranges — a real
resolver conflict, not a hypothetical one
([NOTE-AGENT-1-stack.md](../../research/NOTE-AGENT-1-stack.md), caveat 1). Here's the honest version of
where the *current* pin actually stands, verified rather than assumed.

The current pin, `google-adk==2.8.0`, loosened that constraint to `starlette<2,>=1.3.1` — which the
`fastapi==0.141.1` / `starlette==1.6.0` already installed in `.venv-agent` satisfies. This project checked
that claim for real rather than asserting it, with `.venv-agent` left untouched (`pip install --dry-run`
computes the resolution without installing anything):

```text
$ .venv-agent\Scripts\python.exe -m pip install google-adk==2.8.0 --dry-run
Requirement already satisfied: starlette<2,>=1.3.1 in .\.venv-agent\Lib\site-packages
  (from google-adk==2.8.0) (1.6.0)
Requirement already satisfied: typing-inspection>=0.4.2 in .\.venv-agent\Lib\site-packages
  (from fastapi<1,>=0.133->google-adk==2.8.0) (0.4.4)
...
Would install aiohappyeyeballs-2.7.1 ... google-adk-2.8.0 google-auth-2.57.0 google-genai-2.22.0 ...
```

No conflict — `pip`'s resolver is satisfied. So as of *today's* pins, installing `google-adk` straight
into `.venv-agent` would work. Full log:
[`artefacts/adk_venv_install_output.txt`](artefacts/adk_venv_install_output.txt).

That's exactly why this project still keeps it separate. NOTE-AGENT-1's caveat 1 is proof this exact
library has already broken this exact adjacency once, between v1.12.0 and v2.8.0 — and the NOTE explicitly
recommends testing any google-adk + fastapi combination in a throwaway venv before committing to it
together. ADK also runs a full FastAPI server of its own internally (NOTE-AGENT-1) — it isn't a leaf
dependency, it's a framework that *embeds* the same library your own service uses, which is exactly the
shape of dependency most likely to fight over a shared transitive pin the next time either project bumps
a version. A shared venv makes a future `pip install --upgrade google-adk` free to yank `fastapi`/
`starlette` to whatever *it* now wants, silently, underneath every other chapter that also depends on that
`fastapi` pin — a lockfile-less version of a transitive Maven dependency getting bumped out from under
you by an unrelated `mvn install`. Two venvs means that upgrade is scoped to `.venv-adk` alone. This is a
judgment call, not a hard technical requirement today — say so plainly if you'd rather install both
together, now that you've seen the actual resolver output.

Verify the import — no key needed to construct the object, only to actually run it (Section 4):

```python
from google.adk.agents.llm_agent import Agent

root_agent = Agent(
    model="gemini-flash-latest",
    name="root_agent",
    description="A minimal ADK agent for the local-environment-setup gate check.",
)
print("root_agent created:", root_agent.name, root_agent.model, root_agent.description)
```

Run for real (`.venv-adk\Scripts\python.exe`), this printed:

```text
root_agent created: root_agent gemini-flash-latest A minimal ADK agent for the local-environment-setup gate check.
```

`root_agent` as a **module-level variable name** is the one convention ADK's runtime actually requires —
this shape is the minimal "hello agent" from the official docs
([source: ADK Python quickstart](https://adk.dev/get-started/python/), checked 2026-09-02;
[NOTE-AGENT-1-stack.md](../../research/NOTE-AGENT-1-stack.md)). `model`, `name`, and `description` are the
only fields that minimal example sets; `model` names *which* LLM this agent calls once you actually invoke
it — a decision, and a credential, this chapter defers to Section 4.

### 2.4 Verify the whole stack: `verify_agentic_env.py`

One script checks both environments, and handles the *other* one's packages being absent gracefully
instead of crashing — because that absence is the point, not a bug:

```mermaid
flowchart LR
    SCRIPT["verify_agentic_env.py<br/>(one script, two runs)"] -->|"run with<br/>.venv-agent's python.exe"| A["reports fastapi, fastmcp,<br/>pgvector, psycopg, pydantic --<br/>google-adk absent (by design)"]
    SCRIPT -->|"run with<br/>.venv-adk's python.exe"| B["reports google-adk --<br/>the rest absent (by design)"]
```

```python
"""Verify the Agentic Engineering local environment.

Two separate stacks, on purpose (the chapter explains why):

1. The shared `.venv-agent` stack -- fastapi, fastmcp, pgvector (Python client), psycopg,
   pydantic. This script's `check_agent_stack()` imports each and prints its installed
   version. Run this half with `.venv-agent`'s interpreter.
2. `google-adk`, which this project keeps in its OWN virtualenv, `.venv-adk` (see the
   chapter, Section 2.3, for the isolation rationale -- it is NOT about a live version
   conflict; empirically, `google-adk==2.8.0` resolves cleanly against
   `fastapi==0.141.1`/`starlette==1.6.0`, confirmed by `pip install --dry-run` and
   `pip check` against this project's real `.venv-agent`, both logged in the chapter).
   `check_adk()` handles ADK being ABSENT gracefully -- if you run this script with
   `.venv-agent`'s interpreter (where ADK is deliberately not installed), it reports that
   plainly instead of raising ImportError and killing the whole check.

Run against .venv-agent (fastapi/fastmcp/pgvector/psycopg -- ADK reported absent, by design):
    .venv-agent\\Scripts\\python.exe "Agentic Engineering/Local Environment Setup/code/verify_agentic_env.py"

Run against .venv-adk (only google-adk is checked; the other imports are reported absent there
by design too -- it is a dedicated, minimal environment):
    .venv-adk\\Scripts\\python.exe "Agentic Engineering/Local Environment Setup/code/verify_agentic_env.py"

Expects (pinned in this chapter; see local-environment-setup.md and
research/NOTE-AGENT-1-stack.md, checked 2026-09-02):
    fastapi==0.141.1  fastmcp==4.0.1  pgvector==0.5.0  psycopg==3.3.5
    google-adk==2.8.0 (separate venv)

No LLM API key is required for anything this script checks -- it only imports libraries and
reads their reported version. The LLM-call path (Section 4 of the chapter) is separately
marked key-gated and is not exercised here.
"""
from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version


def check_agent_stack() -> None:
    """Imports and reports the shared .venv-agent stack. Each import is independent --
    one missing package is reported and skipped, not a fatal crash, so this script gives
    a full picture even against a partially-installed environment."""
    packages = ["fastapi", "fastmcp", "pgvector", "psycopg", "pydantic"]
    print("-- shared .venv-agent stack --")
    for name in packages:
        try:
            __import__(name)
            print(f"{name:<12} {pkg_version(name)}")
        except ImportError:
            print(f"{name:<12} NOT INSTALLED in this interpreter")
        except PackageNotFoundError:
            print(f"{name:<12} imported, but no distribution metadata found")


def check_adk() -> None:
    """Imports google-adk if present; reports 'not installed here' if not, instead of
    raising -- this is the graceful-absence behaviour the chapter calls for, since this
    project deliberately does NOT install google-adk into .venv-agent."""
    print("-- google-adk (expected in a SEPARATE .venv-adk, not here) --")
    try:
        from google.adk.agents import Agent  # noqa: F401

        print(f"google-adk     {pkg_version('google-adk')}  (import OK: google.adk.agents.Agent)")
    except ImportError:
        print("google-adk     NOT INSTALLED in this interpreter (expected if this is .venv-agent)")


def main() -> None:
    print(f"Python:        {sys.version.split()[0]}")
    print(f"Executable:    {sys.executable}")
    print()
    check_agent_stack()
    print()
    check_adk()


if __name__ == "__main__":
    main()
```

The full script lives at [`code/verify_agentic_env.py`](code/verify_agentic_env.py). Running it against
both real environments (2026-09-03, no LLM key set in either) printed:

```text
$ .venv-agent\Scripts\python.exe "Agentic Engineering\Local Environment Setup\code\verify_agentic_env.py"
Python:        3.13.7
Executable:    C:\Users\...\ds_ml_ai_starter\.venv-agent\Scripts\python.exe

-- shared .venv-agent stack --
fastapi      0.141.1
fastmcp      4.0.1
pgvector     0.5.0
psycopg      3.3.5
pydantic     2.13.5

-- google-adk (expected in a SEPARATE .venv-adk, not here) --
google-adk     NOT INSTALLED in this interpreter (expected if this is .venv-agent)

$ .venv-adk\Scripts\python.exe "Agentic Engineering\Local Environment Setup\code\verify_agentic_env.py"
Python:        3.13.7
Executable:    C:\Users\...\ds_ml_ai_starter\.venv-adk\Scripts\python.exe

-- shared .venv-agent stack --
fastapi      0.141.1
fastmcp      NOT INSTALLED in this interpreter
pgvector     NOT INSTALLED in this interpreter
psycopg      NOT INSTALLED in this interpreter
pydantic     2.13.5

-- google-adk (expected in a SEPARATE .venv-adk, not here) --
google-adk     2.8.0  (import OK: google.adk.agents.Agent)
```

(`fastapi`/`pydantic` show up in the `.venv-adk` run too — `google-adk` pulls both in as its own
transitive dependencies, which is exactly the adjacency Section 2.3 discussed.) Full capture:
[`artefacts/verify_agentic_env_output.txt`](artefacts/verify_agentic_env_output.txt). Every version
printed matches Section 2.2/2.3's pins exactly — that match is what "the environment is ready" means for
this chapter, the same way it did in SPEC-DS-0.

## 3. Postgres + pgvector via Docker — a nearest-neighbour store, not a key-value one

### 3.1 Why pgvector, and what it actually adds

Ordinary Postgres answers "give me the row where `id = 42`" — an exact match, backed by a B-tree index,
exactly what Hibernate/JPA generates for you every day. An agent's memory needs a different query:
"give me the K rows whose embedding is *closest* to this one," where "closest" is a real distance
computation over hundreds of floating-point dimensions, not an equality check. **pgvector** is a Postgres
*extension* — same binary, same SQL, one `CREATE EXTENSION` away — that adds a `vector` column type and
distance operators to answer exactly that query
([source: pgvector](https://github.com/pgvector/pgvector), checked 2026-09-02;
[NOTE-AGENT-1-stack.md](../../research/NOTE-AGENT-1-stack.md)). `theory.md` Section 3 covered the concept
(HNSW/IVF, ANN vs. exact search); this section stands up the real thing.

```mermaid
flowchart LR
    Q1["WHERE id = 42"] --> BT["B-tree index --<br/>exact match<br/>(Hibernate/JPA, every day)"]
    Q2["ORDER BY embedding &lt;=&gt; query<br/>LIMIT k"] --> NN["pgvector --<br/>nearest-neighbour distance,<br/>not an equality check"]
```

### 3.2 Start it with Docker Compose

```yaml
# Local Postgres + pgvector for the Agentic Engineering chapters.
#
# Image and version verified against Docker Hub (research/NOTE-AGENT-1-stack.md, checked 2026-09-02):
# pgvector/pgvector:pg17 -- https://hub.docker.com/r/pgvector/pgvector/tags
#
# Start:  docker compose -f "Agentic Engineering/Local Environment Setup/code/docker-compose.yml" up -d
# Stop:   docker compose -f "Agentic Engineering/Local Environment Setup/code/docker-compose.yml" down
#
# The `vector` extension ships inside this image but is NOT enabled by default -- you still run
# `CREATE EXTENSION vector;` once per database (see local-environment-setup.md, Section 3).
services:
  pgvector:
    image: pgvector/pgvector:pg17
    container_name: agentic-pgvector
    restart: unless-stopped
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: agentdb
    ports:
      # host:container -- 55432 on the host to avoid colliding with a Postgres already
      # listening on the default 5432, exactly like binding a Java service to a non-default
      # port in docker-compose to dodge a clash with something else on your machine.
      - "55432:5432"
    volumes:
      - pgvector-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 10

volumes:
  pgvector-data:
```

The full file is [`code/docker-compose.yml`](code/docker-compose.yml). Port 55432 (not the Postgres
default 5432) isn't decoration — this machine, right now, already has another container bound to a
non-default Postgres port for exactly the same reason: two services both wanting `5432` on the host is a
real, common collision, and docker-compose's `host:container` port mapping is the fix, the same move as
giving a second Java service its own listener port in application properties instead of fighting over
`8080`.

```bash
docker compose -f "Agentic Engineering/Local Environment Setup/code/docker-compose.yml" up -d
docker exec agentic-pgvector pg_isready -U postgres
docker exec -it agentic-pgvector psql -U postgres -d agentdb -c "CREATE EXTENSION vector;"
```

Run for real on this machine (Docker 28.4.0), this produced:

```text
$ docker compose -f "Agentic Engineering/Local Environment Setup/code/docker-compose.yml" up -d
 Network code_default  Creating
 Network code_default  Created
 Volume "code_pgvector-data"  Creating
 Volume "code_pgvector-data"  Created
 Container agentic-pgvector  Creating
 Container agentic-pgvector  Created
 Container agentic-pgvector  Starting
 Container agentic-pgvector  Started

$ docker exec agentic-pgvector pg_isready -U postgres
/var/run/postgresql:5432 - accepting connections

$ docker exec -it agentic-pgvector psql -U postgres -d agentdb -c "CREATE EXTENSION vector;"
CREATE EXTENSION
```

`CREATE EXTENSION vector;` is the whole install step — the extension ships inside the official image, it
just isn't turned on for a fresh database by default
([source: pgvector](https://github.com/pgvector/pgvector), checked 2026-09-02).

```mermaid
flowchart LR
    UP["docker compose up -d<br/>(pgvector/pgvector:pg17)"] --> READY["pg_isready --<br/>accepting connections"]
    READY --> EXT["CREATE EXTENSION vector;<br/>(once per database)"]
    EXT --> SQL["raw SQL: CREATE TABLE ... vector(3);<br/>INSERT; SELECT ... ORDER BY &lt;=&gt;"]
    EXT --> PY["psycopg + pgvector client:<br/>the same query, from Python"]
```

### 3.3 A real vector column, a real insert, a real nearest-neighbour query

Straight SQL first, run inside the container with `psql`, so you see pgvector's own syntax with nothing
else in the way:

```sql
CREATE TABLE items (id bigserial PRIMARY KEY, content text, embedding vector(3));

INSERT INTO items (content, embedding) VALUES
  ('sourdough bread recipe', '[0.9, 0.1, 0.0]'),
  ('formula one pit stop rules', '[0.1, 0.9, 0.0]'),
  ('vector index HNSW explained', '[0.0, 0.1, 0.9]');

SELECT id, content, embedding <=> '[0.85, 0.15, 0.05]' AS cosine_distance
  FROM items
  ORDER BY embedding <=> '[0.85, 0.15, 0.05]'
  LIMIT 3;
```

`vector(3)` declares a 3-dimensional column (a real embedding model like `all-MiniLM-L6-v2`, used in
`theory.md`, produces 384 — 3 keeps this legible). `<=>` is pgvector's cosine-distance operator: **smaller
is closer**, unlike a cosine *similarity* score where bigger is closer — a detail worth getting right
before you write an `ORDER BY` the wrong direction. Run for real, against the container from Section 3.2:

```text
 id |           content           |    cosine_distance
----+-----------------------------+-----------------------
  1 | sourdough bread recipe      | 0.0037184699510960373
  2 | formula one pit stop rules  |     0.718997508381225
  3 | vector index HNSW explained |    0.9233629581289488
(3 rows)
```

The query vector `[0.85, 0.15, 0.05]` is closest, by construction, to the "sourdough" row's
`[0.9, 0.1, 0.0]` — and the result confirms it: distance 0.0037, an order of magnitude closer than either
other row. Full transcript, including `docker compose up`:
[`artefacts/pgvector_docker_output.txt`](artefacts/pgvector_docker_output.txt).

Now the same query from Python, through `psycopg` + the `pgvector` client — the code path every later
Agentic Engineering chapter's memory layer actually uses, instead of a `psql` shell:

```python
"""Sanity-check the local pgvector instance: connect, create a vector column, insert a
few rows, and run a real nearest-neighbour query -- the smallest possible proof that the
"agent memory" layer (Section 3 of the chapter) actually works end to end.

No LLM key needed. This talks to Postgres only.

Prerequisite: the pgvector container from docker-compose.yml is up and `CREATE EXTENSION
vector;` has been run once (chapter Section 3.2):

    docker compose -f "Agentic Engineering/Local Environment Setup/code/docker-compose.yml" up -d
    docker exec -it agentic-pgvector psql -U postgres -d agentdb -c "CREATE EXTENSION vector;"

Run with the shared .venv-agent interpreter:

    .venv-agent\\Scripts\\python.exe "Agentic Engineering/Local Environment Setup/code/pgvector_sanity_check.py"

Reads the connection string from the DATABASE_URL env var, matching .env.example, falling
back to this chapter's own docker-compose port (55432) if unset -- see Section 4 for why
this lives in env, not in the script.
"""
from __future__ import annotations

import os

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

DEFAULT_DATABASE_URL = "postgresql://postgres:postgres@localhost:55432/agentdb"


def main() -> None:
    database_url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    conn = psycopg.connect(database_url, autocommit=True)
    register_vector(conn)  # teaches psycopg to adapt numpy arrays <-> the `vector` SQL type
    cur = conn.cursor()

    # A 3-dimensional embedding table -- real embedding models (all-MiniLM-L6-v2, used
    # elsewhere in this subject) produce 384 dims; 3 keeps this sanity check legible.
    cur.execute("DROP TABLE IF EXISTS sanity_items")
    cur.execute(
        "CREATE TABLE sanity_items (id bigserial PRIMARY KEY, content text, embedding vector(3))"
    )

    rows = [
        ("sourdough bread recipe", np.array([0.9, 0.1, 0.0])),
        ("formula one pit stop rules", np.array([0.1, 0.9, 0.0])),
        ("vector index HNSW explained", np.array([0.0, 0.1, 0.9])),
    ]
    cur.executemany(
        "INSERT INTO sanity_items (content, embedding) VALUES (%s, %s)",
        rows,
    )
    print(f"inserted {len(rows)} rows")

    query_vec = np.array([0.85, 0.15, 0.05])  # closest, by construction, to the bread row
    cur.execute(
        "SELECT id, content, embedding <=> %s AS cosine_distance "
        "FROM sanity_items ORDER BY embedding <=> %s LIMIT 3",
        (query_vec, query_vec),
    )
    print("nearest neighbours to query vector [0.85, 0.15, 0.05]:")
    for row_id, content, distance in cur.fetchall():
        print(f"  id={row_id}  distance={distance:.6f}  content={content!r}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
```

Full script: [`code/pgvector_sanity_check.py`](code/pgvector_sanity_check.py). Run for real:

```text
$ .venv-agent\Scripts\python.exe "Agentic Engineering\Local Environment Setup\code\pgvector_sanity_check.py"
inserted 3 rows
nearest neighbours to query vector [0.85, 0.15, 0.05]:
  id=1  distance=0.003718  content='sourdough bread recipe'
  id=2  distance=0.718998  content='formula one pit stop rules'
  id=3  distance=0.923363  content='vector index HNSW explained'
```

Same distances as the raw SQL run, to six decimal places — `register_vector(conn)` is what lets
`psycopg` hand a plain `numpy.ndarray` straight to a `vector` column and get one back, the same role a
JDBC type mapper plays for a custom SQL type: without it, you'd be hand-formatting `'[0.9, 0.1, 0.0]'`
strings yourself, the way the raw-SQL version above had to.

## 4. Secrets in env — the provider-key model

### 4.1 What actually needs a key, and what doesn't

Section 2.3 already showed this concretely: constructing `Agent(model="gemini-flash-latest", ...)`
printed a working object with **no key set anywhere in this shell** (confirmed —
`GOOGLE_API_KEY`/`GEMINI_API_KEY` were both unset for that run). Building the agent, wiring MCP tools,
and querying pgvector are all key-free, local operations. The **one** call this chapter marks key-gated —
reference only, not executed as part of this chapter's gate — is actually *running* the agent, i.e.
having it call the LLM:

```mermaid
flowchart LR
    A["Agent(model=..., name=..., description=...)<br/>construction -- NO key needed"] --> B["InMemoryRunner(agent=...)<br/>construction -- NO key needed"]
    B --> C["runner.run(...)<br/>actually calls the LLM --<br/>THE one key-gated step"]
```

```python
# KEY-GATED -- reference only, not run as part of this chapter's gate.
# Requires a real GOOGLE_API_KEY (or another supported provider's key) in the environment.
# ADK reads GOOGLE_API_KEY from the environment by default for Gemini models
# (NOTE-AGENT-1-stack.md, caveat 2). This is the one call in the whole chapter that leaves
# your machine and costs money/quota.
#
# InMemoryRunner(agent=...) is a real, importable class -- confirmed directly against the
# installed google-adk==2.8.0 package (`inspect.signature`, checked 2026-09-03: its __init__
# takes an `agent: Optional[BaseAgent]` keyword), not asserted from memory.
from google.adk.runners import InMemoryRunner

runner = InMemoryRunner(agent=root_agent)
# runner.run(...) or the async equivalent actually calls the model -- requires the key above.
```

Without a valid key, that call fails at the provider boundary (an auth error from Google's API), not
silently — the same "key missing → 401, not a hang" behaviour you'd expect calling any paid HTTPS API
without credentials.

### 4.2 `.env` / `.env.example` — the pattern this whole project follows

This project's root [`.env.example`](../../.env.example) is the checked-in template: every variable name
a chapter needs, commented out, with a placeholder or a one-line explanation — never a real value.
Copy it to a real `.env` (git-ignored — `.gitignore` excludes `.env` and `.env.*` but explicitly
un-ignores `.env.example`, so the template stays committed and the real file never can be) and fill in
only what the chapter you're running actually needs:

```bash
# --- Agentic Engineering ---
# ANTHROPIC_API_KEY=
# OPENAI_API_KEY=
# GOOGLE_API_KEY=        # required for an ADK agent's actual model calls (Section 4.1) --
#                        # ADK reads GOOGLE_API_KEY by default for Gemini models
# DATABASE_URL=postgresql://postgres:postgres@localhost:55432/agentdb   # matches
#                        # docker-compose.yml (Section 3.2) -- port 55432 avoids colliding
#                        # with a Postgres already on the default 5432
```

The rule this project follows, no exceptions: a credential is either an env var read at run time
(`os.environ["GOOGLE_API_KEY"]`, or a library reading it implicitly the way ADK does) or it doesn't
exist in this codebase at all. Nothing is hard-coded, nothing is committed, and `.env.example` is kept
current whenever a chapter introduces a new variable — this chapter added `GOOGLE_API_KEY` and updated
`DATABASE_URL` to match Section 3's actual port and database name.

If you've deployed a Spring Boot service with `application.yml` pulling from environment variables or a
secrets manager rather than a checked-in properties file, this is the identical discipline — the
`.env`/`.env.example` split is just this ecosystem's name for "config lives in env, template lives in
git, secret never does."

## 5. The component map — how it all connects

Section 1's table and Section 4's key-gated/key-free split both collapse into one picture. Rendered
first as a GitHub-native Mermaid diagram; the equivalent plain-text version follows immediately below
it — that one is this project's actual committed artefact, for the reason explained right after it.

```mermaid
flowchart TB
    CLIENT["HTTP client<br/>(browser / curl / another service)"] -->|"HTTP request"| FASTAPI["FastAPI<br/>your service's HTTP boundary"]
    FASTAPI -->|"invokes"| ADKR["ADK Runtime<br/>root_agent = Agent(model=..., name=..., description=...)<br/>decides: tool call? memory lookup? call the LLM?"]
    ADKR -->|"tool call (MCP protocol)"| MCP["MCP Server(s)<br/>built with FastMCP --<br/>Tools / Resources / Prompts"]
    ADKR -->|"memory / retrieval"| PGV["Postgres + pgvector<br/>CREATE EXTENSION vector;<br/>embedding &lt;=&gt; query ORDER BY ... LIMIT k"]
    MCP -->|"wraps"| SYS["The actual system(s) --<br/>a database, a filesystem, an internal REST API"]
    ADKR -->|"model inference call --<br/>the ONLY key-gated arrow"| LLM["LLM Provider<br/>Gemini via GOOGLE_API_KEY<br/>(or Anthropic / OpenAI, per NOTE-AGENT-1)"]
```

```text
  HTTP client                                            LLM Provider (key-gated)
  (browser / curl /                                      Gemini, via GOOGLE_API_KEY
   another service)                                      (or Anthropic / OpenAI, per NOTE-AGENT-1)
        |                                                            ^
        | HTTP request                                              | model inference call
        v                                                            | (the ONLY step in this
  +----------------------------+                                    |  diagram that needs a key)
  |          FastAPI            |                                    |
  |  your service's HTTP        |                                    |
  |  boundary -- routes,        |                                    |
  |  request/response models    |                                    |
  +--------------+---------------+                                   |
                 |                                                   |
                 | invokes                                           |
                 v                                                   |
  +----------------------------------------------------------------------+
  |                          ADK Runtime                                  |
  |   root_agent = Agent(model=..., name=..., description=...)            |
  |   orchestrates: reads the request, decides whether to call a tool,    |
  |   whether to retrieve memory, when to call the LLM, and how to        |
  |   assemble the final response                                        |
  +---------------------+------------------------------+-----------------+
                         |                              |
          tool call      |                              |  memory / retrieval
          (MCP protocol) |                              |  (pgvector)
                         v                              v
  +----------------------------------+     +--------------------------------------+
  |         MCP Server(s)             |     |        Postgres + pgvector             |
  |         (built with FastMCP)      |     |  CREATE EXTENSION vector;              |
  |  exposes Tools / Resources /      |     |  embedding <=> query_vector            |
  |  Prompts over a standard,         |     |  ORDER BY ... LIMIT k                  |
  |  JSON-RPC-based protocol --       |     |  (nearest-neighbour search, not an     |
  |  one server per wrapped system    |     |   exact-key lookup -- Section 3's      |
  +---------------+-------------------+     |   pgvector_sanity_check.py runs this)  |
                  |                          +--------------------------------------+
                  | wraps
                  v
  +----------------------------------+
  |   The actual system(s)            |
  |   a database, a filesystem,       |
  |   an internal REST API --         |
  |   whatever the tool call reaches  |
  +----------------------------------+
```

Committed as [`artefacts/component_map.txt`](artefacts/component_map.txt) — rendered as text because
`.venv-agent` has no `matplotlib` installed (confirmed 2026-09-03: `ModuleNotFoundError: No module named
'matplotlib'`), and this course doesn't add a plotting dependency to the agent stack just to draw one
diagram.

Read it top to bottom, mapping straight back to Section 1's table: a request enters through **FastAPI**
(the piece you've built dozens of times), gets handed to the **ADK runtime**, which decides — per turn,
based on the LLM's own output — whether it needs a **tool** (out over **MCP** to a FastMCP server
wrapping some real system) or a **memory lookup** (**pgvector**, a nearest-neighbour search over
embeddings) before it ever calls the **LLM provider** itself. Every arrow except the one reaching the LLM
provider runs locally, for free, with nothing but this chapter's Docker container and two Python
environments — which is exactly what Sections 2–3 just proved by actually running all of it.

`theory.md` Section 5 already named the three MCP roles this diagram's "MCP Server(s)" box plays: Host
(the ADK runtime, initiating the connection), Client (the connector inside it, one per server), Server
(the FastMCP process on the other end). SPEC-AGENT-2 builds a real one.

## 6. Pitfalls

- **Assuming a shared venv "should" work forever because it works today.** Section 2.3's dry-run proved
  `google-adk==2.8.0` resolves cleanly against this project's `fastapi`/`starlette` pins *right now* —
  but NOTE-AGENT-1's caveat 1 is documented proof the exact same pairing broke as recently as ADK v1.12.0.
  A clean `pip install --dry-run` today is not a guarantee a future `pip install --upgrade` stays clean;
  it's evidence for *today's* pins only. Re-run the dry-run check after bumping either pin, the same
  discipline as re-running a build after bumping a Maven dependency.
- **Forgetting `CREATE EXTENSION vector;` on a fresh database.** The pgvector *extension* ships inside the
  Docker image, but a new database doesn't have it enabled — `CREATE TABLE ... vector(3)` fails with
  `type "vector" does not exist` until you run it once per database (Section 3.2). This is a one-time
  step per database, not per connection.
- **Confusing `<=>` (distance) with similarity.** pgvector's cosine operator returns a *distance* —
  smaller means closer, the opposite direction from a similarity score where bigger means closer. An
  `ORDER BY embedding <=> query DESC` when you meant `ASC` (the default, and what Section 3.3 used)
  silently returns the *furthest* matches first, with no error to flag the mistake.
- **A port collision with an existing local Postgres.** Binding the pgvector container to the Postgres
  default port 5432 fails outright — or worse, silently connects your client to the *wrong* Postgres — if
  anything else on the machine already owns 5432. Section 3.2's `55432:5432` mapping in
  `docker-compose.yml` sidesteps this by design; confirm nothing else already owns whatever host port you
  choose (`docker ps` lists every container's port mappings).
- **Treating "the object constructed" as "the key works."** Section 4.1 showed `Agent(...)` succeeding
  with zero credentials configured — construction alone proves nothing about whether the model call
  behind it would actually authenticate. Don't take a clean import or a clean object construction as
  proof the key-gated path is ready; the only way to know is the actual call, deliberately not run as
  part of this chapter's key-free gate.
- **A leaked key in `.env` committed by accident.** `.gitignore` blocks `.env` and `.env.*` while
  explicitly keeping `.env.example` — but that protection only holds if you never `git add -f` your way
  around it. Never paste a real key into `.env.example`, and never remove the `.gitignore` entries to "just
  get it working."

## 7. Recap & what's next

- The agent stack maps cleanly onto roles you already know from a Java web shop (Section 1): FastAPI is
  your HTTP boundary, the ADK runtime is your service layer except the LLM picks the branch, MCP is a
  standardised internal-RPC contract, pgvector is Postgres answering "nearest" instead of "equals," and
  the LLM provider is the one paid, key-gated, external call in the whole system.
- `google-adk` lives in its own `.venv-adk`, verified by a real `pip install --dry-run` against
  `.venv-agent` to show **no current conflict** with `fastapi`/`starlette` — the isolation is a defensive
  choice grounded in this exact library's documented history (NOTE-AGENT-1, caveat 1), not evidence of a
  conflict that exists today (Section 2.3).
- Postgres + pgvector runs locally via Docker Compose; `CREATE EXTENSION vector;` is the entire install
  step; a real insert and a real `<=>` nearest-neighbour query ran both in raw SQL and through
  `psycopg`/`pgvector` from Python, with matching distances (Section 3).
- Every credential lives in `.env`, never committed, following the template in `.env.example`; only the
  actual LLM inference call needs one, and everything else in this chapter — imports, versions, Docker,
  SQL, the vector query — ran key-free (Section 4).
- The component map (Section 5) is the picture to keep in your head for every chapter that follows: a
  request flows FastAPI → ADK runtime → (MCP tool call *or* pgvector memory lookup) → LLM, and only the
  last arrow costs money.

Back to the map from the cold open — every box is now either built or ready:

```mermaid
flowchart LR
    FA["FastAPI -- installed & verified, S2.2"] --> ADK["ADK runtime -- agent<br/>constructed & verified, S2.3"]
    ADK -->|"tool call"| MCP["FastMCP -- installed, S2.2;<br/>a real server is SPEC-AGENT-2"]
    ADK -->|"memory lookup"| PGV["pgvector on Postgres --<br/>real nearest-neighbour query, S3"]
    ADK -->|"model inference<br/>(key-gated)"| LLM["LLM provider --<br/>.env pattern ready, S4;<br/>not called in this chapter's gate"]
```

Next: [SPEC-AGENT-2](../03-worked-examples/01-mcp-database-query-layer.md) builds a real MCP server with
FastMCP — Tools and Resources wrapping an actual database — tested with a plain client, still with no LLM
in the loop, before any chapter wires an agent to it for real.

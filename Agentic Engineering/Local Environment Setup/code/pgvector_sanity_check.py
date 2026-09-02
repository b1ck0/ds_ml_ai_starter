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

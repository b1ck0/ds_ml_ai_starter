"""Tiny RAG demo -- companion code for
Agentic Engineering/Theory/theory.md (SPEC-AGENT-1).

This script is the runnable half of the RAG (Retrieval-Augmented Generation) section of
the theory chapter. It does the two things every RAG pipeline does before it ever talks
to an LLM:

  1. EMBED a small, fixed knowledge base of 10 short documents with a real, local,
     CPU-only sentence embedding model (sentence-transformers/all-MiniLM-L6-v2, 384-D,
     Apache 2.0 -- research/NOTE-AGENT-2-rag-primitives.md, "Embedding model" section,
     citing https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2, checked
     2026-09-02). No API key, no network call at query time, no LLM involved anywhere in
     this script -- this demo stops at "retrieve", which is exactly the embeddings +
     vector-search step Chapter theory.md Section 3 explains (RAG's full pipeline,
     including the augment/generate steps this demo does NOT do, is Section 4).

  2. RETRIEVE the top-k most similar documents for three example queries via a brute-force
     cosine top-k search implemented directly in numpy: every query vector is compared
     against every document vector and the highest-scoring k win. This is precisely what
     a FAISS "Flat" index or an unindexed pgvector `<=>` scan does under the hood
     (NOTE-AGENT-2, "Vector search approaches" -- Option A/B) -- exact, not approximate,
     which is fine at 10 documents and is *why* HNSW/IVF exist once a corpus grows past
     the point where scanning every row is cheap (see theory.md Section 3, "ANN indexes").

The script prints every step to stdout (which embeddings source loaded, every query's
ranked results) and renders three PNG artefacts with PIL (Pillow, already installed in
this project's .venv-agent -- no matplotlib is used or required):

  - artefacts/retrieval_result_table.png  -- the REAL retrieval table (9 rows: 3 queries
    x top-3), built from this run's own cosine scores, not invented numbers.
  - artefacts/rag_pipeline_diagram.png    -- the offline indexing path + online query
    path of a RAG pipeline (chunk -> embed -> store, and query -> embed -> search ->
    augment -> generate).
  - artefacts/mcp_client_server_diagram.png -- MCP's host/client/server roles and the
    Tools/Resources/Prompts primitives a server can expose (NOTE-AGENT-2, "MCP" section,
    citing the official spec https://modelcontextprotocol.io/specification/2026-07-28,
    checked 2026-09-02).

Every claim behind the diagrams and the retrieval mechanics is grounded in
research/NOTE-AGENT-2-rag-primitives.md or an inline citation in theory.md -- nothing in
this docstring or the code below is asserted from memory.

Environment (installed versions in this project's .venv-agent, verified live via
`pip show` on 2026-09-02 -- the pins in research/NOTE-AGENT-2 line up):
    sentence-transformers==6.0.1, numpy==2.5.2, pillow==12.3.0, torch==2.14.0,
    transformers==5.16.1, Python 3.13 (venv reports this; project targets 3.11+).
    CPU only. No API key. No network access needed at retrieval time (only the first
    ever run needs network access, to download and cache the ~90 MB model weights from
    Hugging Face -- NOTE-AGENT-2 caveat 1).

Run:
    .venv-agent/Scripts/python.exe "Agentic Engineering/Theory/code/tiny_rag_demo.py"
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ARTEFACTS_DIR = Path(__file__).resolve().parent.parent / "artefacts"
SEED = 42  # Embedding inference here is deterministic (no sampling, no dropout at eval
# time), so nothing in this script actually draws from a random number generator. SEED
# is kept as a named constant anyway, per this project's convention of always stating a
# seed explicitly, and would be the first thing to pass to np.random.default_rng(SEED)
# if this demo ever grew a stochastic step (e.g. randomly sampling a larger corpus).

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3

# A tiny fixed "knowledge base": 10 short documents a real RAG app might have chunked
# and indexed. Deliberately mixed-topic -- some are about vector search / RAG / MCP
# itself, some are unrelated Java/Python trivia, some are totally unrelated (bread,
# motorsport) -- so a correct retrieval has to actually discriminate on meaning, not
# just return "whatever's first".
DOCS: dict[str, str] = {
    "vector_index_hnsw": (
        "HNSW (Hierarchical Navigable Small World) is a graph-based approximate "
        "nearest-neighbor index: it organizes vectors into layered graphs so a search "
        "can jump through long-range links first, then refine locally, giving high "
        "recall at low query latency."
    ),
    "vector_index_ivf": (
        "IVF (Inverted File) indexes speed up nearest-neighbor search by clustering "
        "vectors with k-means into partitions, then at query time only scanning the "
        "handful of partitions closest to the query vector instead of the whole dataset."
    ),
    "pgvector_postgres": (
        "pgvector is a PostgreSQL extension that adds a vector column type and "
        "nearest-neighbor operators such as <=> for cosine distance, letting you run "
        "similarity search with ordinary SQL instead of standing up a separate vector "
        "database."
    ),
    "mcp_protocol": (
        "The Model Context Protocol (MCP) is a JSON-RPC based open standard that lets "
        "an LLM application discover and call tools, read resources, and use prompt "
        "templates exposed by a separate MCP server, without a custom integration for "
        "every data source."
    ),
    "rag_pipeline": (
        "Retrieval-Augmented Generation (RAG) retrieves relevant text chunks from an "
        "external store and inserts them into the model's prompt before generation, so "
        "the model can answer using up-to-date or private information it was never "
        "trained on."
    ),
    "context_window": (
        "A context window is the fixed maximum number of tokens a language model can "
        "hold in a single request, covering the prompt, any retrieved context, and the "
        "generated reply combined; exceeding it forces truncation or a request error."
    ),
    "java_garbage_collection": (
        "Java's garbage collector reclaims heap memory automatically by tracing which "
        "objects are still reachable from GC roots and freeing the rest, so a Java "
        "developer almost never calls free() or delete() directly."
    ),
    "python_gil": (
        "CPython's Global Interpreter Lock (GIL) allows only one thread to execute "
        "Python bytecode at a time per process, which is why CPU-bound Python workloads "
        "often reach for multiprocessing instead of threading."
    ),
    "sourdough_bread": (
        "A basic sourdough loaf needs only flour, water, salt, and a mature starter; "
        "the long fermentation develops flavor and gluten structure without any "
        "commercial yeast."
    ),
    "formula_one_pitstop": (
        "In Formula 1, cars must complete a mandatory pit stop using at least two "
        "different tyre compounds during a dry race, a rule intended to add strategic "
        "variation."
    ),
}

# query -> (natural-language question, expected top-1 doc id). The expectation is
# asserted at runtime -- if a future model swap ever broke this demo's retrieval
# quality, the script fails loudly instead of silently printing a wrong "example".
QUERIES: list[tuple[str, str]] = [
    (
        "How does a graph-based nearest-neighbor index use layered graphs and "
        "long-range links to avoid scanning every vector?",
        "vector_index_hnsw",
    ),
    (
        "How does an LLM-based agent call an external tool through a standardized "
        "protocol instead of a one-off integration?",
        "mcp_protocol",
    ),
    (
        "What ingredients go into a basic sourdough loaf?",
        "sourdough_bread",
    ),
]


# ---------------------------------------------------------------------------
# 1. Embed the corpus with a real, local, CPU-only sentence embedding model
# ---------------------------------------------------------------------------

def embed(texts: list[str]) -> np.ndarray:
    """Encodes `texts` with sentence-transformers/all-MiniLM-L6-v2, returning
    L2-normalized (unit-length) 384-D vectors. Because the vectors are unit-norm,
    cosine similarity reduces to a plain dot product (NOTE-ML-4 / NOTE-AGENT-2's shared
    grounding: cos_sim(a, b) = (a.b) / (||a|| ||b||), which is just a.b when
    ||a||=||b||=1) -- that identity is what `cosine_topk` below relies on.

    NOTE-AGENT-2 evidence: all-MiniLM-L6-v2 outputs a 384-dimensional dense vector
    space, Apache 2.0 licensed, CPU-compatible, no API key
    (https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2, checked 2026-09-02).
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    vectors = model.encode(
        texts, normalize_embeddings=True, show_progress_bar=False
    ).astype(np.float64)
    print(
        f"[embed] loaded sentence-transformers/{MODEL_NAME} on CPU -> "
        f"{vectors.shape[0]} vectors, {vectors.shape[1]}-D, L2-normalized"
    )
    return vectors


# ---------------------------------------------------------------------------
# 2. Brute-force cosine top-k -- the "Flat index" every ANN index approximates
# ---------------------------------------------------------------------------

def cosine_topk(
    query_vec: np.ndarray, doc_matrix: np.ndarray, doc_ids: list[str], k: int
) -> list[tuple[str, float]]:
    """Ranks every document in `doc_matrix` against `query_vec` by cosine similarity
    and returns the top `k` as (doc_id, score) pairs, highest first.

    Both `query_vec` and every row of `doc_matrix` are already unit-norm (see `embed`),
    so cosine similarity is exactly the dot product -- no division needed. This is a
    full O(n) scan over every document: an exact "Flat" index, not an approximation.
    NOTE-AGENT-2: "Flat index: brute-force exact search" is the baseline FAISS offers
    before IVF/HNSW; pgvector's `<=>` operator does the same row-by-row scan when there
    is no index on the column ("unindexed `<=>` queries on large tables are O(n)").
    At 10 documents this scan is effectively instant; theory.md Section 3 ("ANN indexes")
    explains why it stops being instant at scale, and what HNSW/IVF trade off to fix that.
    """
    scores = doc_matrix @ query_vec  # (n_docs,) cosine similarities
    order = np.argsort(-scores)[:k]
    return [(doc_ids[i], float(scores[i])) for i in order]


def run_retrieval_demo(
    doc_matrix: np.ndarray, doc_ids: list[str]
) -> list[tuple[str, int, str, float, str]]:
    """Runs every query in QUERIES through `cosine_topk`, prints the ranked results,
    asserts each query's top-1 hit matches the expected document (LO2/LO3's retrieval
    step made concrete and self-checking), and returns the flat list of table rows
    (query, rank, doc_id, score, snippet) used to render the retrieval-result table.
    """
    query_texts = [q for q, _ in QUERIES]
    query_vectors = embed(query_texts)

    rows: list[tuple[str, int, str, float, str]] = []
    print("\n--- retrieval demo: top-{} cosine matches per query ---".format(TOP_K))
    for (query_text, expected_top1), qvec in zip(QUERIES, query_vectors):
        ranked = cosine_topk(qvec, doc_matrix, doc_ids, TOP_K)
        print(f"\nquery: {query_text!r}")
        for rank, (doc_id, score) in enumerate(ranked, start=1):
            snippet = DOCS[doc_id][:70] + ("..." if len(DOCS[doc_id]) > 70 else "")
            print(f"  #{rank}  cos={score:.4f}  {doc_id:<24s} {snippet}")
            rows.append((query_text, rank, doc_id, score, snippet))
        top1_id, top1_score = ranked[0]
        assert top1_id == expected_top1, (
            f"expected top-1 doc {expected_top1!r} for query {query_text!r}, "
            f"got {top1_id!r} (cos={top1_score:.4f}) -- embedding source may have changed"
        )
        print(f"  -> top-1 matches expected doc {expected_top1!r}: OK")
    return rows


# ---------------------------------------------------------------------------
# 3. Render the retrieval-result table as a PNG with Pillow (no matplotlib needed)
# ---------------------------------------------------------------------------

def _load_font(size: int) -> ImageFont.ImageFont:
    """Pillow's bundled default font, requested at a real pixel size (supported since
    Pillow 10.1; this project's .venv-agent has 12.3.0). Avoids depending on any font
    file being present on the OS, so this renders identically on Windows/macOS/Linux."""
    return ImageFont.load_default(size=size)


def render_retrieval_table_png(
    rows: list[tuple[str, int, str, float, str]], path: Path
) -> None:
    """Draws the 9-row (3 queries x top-3) retrieval table produced by
    `run_retrieval_demo` as a PNG, using Pillow's ImageDraw directly. This is the "REAL
    retrieval-result table" artefact the chapter spec asks for -- every number in it
    came from this run's own cosine_topk calls, not a mock-up."""
    col_labels = ["query", "rank", "doc id", "cosine", "snippet"]
    col_widths = [330, 50, 190, 70, 430]
    header_h = 34
    row_h = 30
    pad = 12

    width = sum(col_widths) + 2 * pad
    height = header_h + row_h * len(rows) + 2 * pad + 40

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = _load_font(13)
    font_bold = _load_font(14)
    title_font = _load_font(16)

    draw.text(
        (pad, 8),
        "Tiny RAG demo: real top-3 cosine retrieval results "
        "(sentence-transformers/all-MiniLM-L6-v2, brute-force cosine top-k)",
        font=title_font,
        fill="black",
    )

    y = 40
    x = pad
    for label, w in zip(col_labels, col_widths):
        draw.rectangle([x, y, x + w, y + header_h], outline="black", fill="#dfe7f5")
        draw.text((x + 6, y + 8), label, font=font_bold, fill="black")
        x += w

    last_query = None
    y += header_h
    for query_text, rank, doc_id, score, snippet in rows:
        x = pad
        query_display = "" if query_text == last_query else _wrap(query_text, 46)
        last_query = query_text
        cells = [query_display, str(rank), doc_id, f"{score:.4f}", snippet]
        fill = "#fdf3e0" if rank == 1 else "white"
        for value, w in zip(cells, col_widths):
            draw.rectangle([x, y, x + w, y + row_h], outline="black", fill=fill)
            draw.text((x + 6, y + 7), value, font=font, fill="black")
            x += w
        y += row_h

    draw.text(
        (pad, y + 8),
        "Highlighted rows are each query's top-1 match; every score is a real cosine "
        "similarity from this run, computed as a plain dot product on unit-norm vectors.",
        font=font,
        fill="#444444",
    )

    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    img.save(path)
    print(f"wrote {path}")


def _wrap(text: str, width: int) -> str:
    """Very small word-wrap helper for the query column (Pillow's ImageDraw has no
    built-in wrapping); breaks at `width` characters, first line only shown here since
    the query column is wide enough for these short example questions."""
    if len(text) <= width:
        return text
    cut = text.rfind(" ", 0, width)
    cut = cut if cut != -1 else width
    return text[:cut] + "..."


# ---------------------------------------------------------------------------
# 4. RAG pipeline diagram (offline indexing path + online query path)
# ---------------------------------------------------------------------------

def _box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str = "#eef3fb",
    outline: str = "black",
) -> None:
    x0, y0, x1, y1 = xy
    draw.rectangle(xy, outline=outline, fill=fill, width=2)
    lines = text.split("\n")
    line_h = font.size + 4
    total_h = line_h * len(lines)
    ty = y0 + ((y1 - y0) - total_h) // 2
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        tx = x0 + ((x1 - x0) - tw) // 2
        draw.text((tx, ty), line, font=font, fill="black")
        ty += line_h


def _harrow(draw: ImageDraw.ImageDraw, x0: int, x1: int, y: int, label: str = "", font=None) -> None:
    draw.line([(x0, y), (x1, y)], fill="black", width=2)
    draw.polygon([(x1, y), (x1 - 10, y - 5), (x1 - 10, y + 5)], fill="black")
    if label and font:
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(((x0 + x1) // 2 - tw // 2, y - font.size - 8), label, font=font, fill="#333333")


def _varrow(draw: ImageDraw.ImageDraw, x: int, y0: int, y1: int, label: str = "", font=None) -> None:
    draw.line([(x, y0), (x, y1)], fill="black", width=2)
    draw.polygon([(x, y1), (x - 5, y1 - 10), (x + 5, y1 - 10)], fill="black")
    if label and font:
        draw.text((x + 8, (y0 + y1) // 2 - font.size // 2), label, font=font, fill="#333333")


def draw_rag_pipeline_diagram(path: Path) -> None:
    """Draws the end-to-end RAG pipeline: the offline indexing path (documents -> chunk
    -> embed -> vector store) running once/periodically, and the online query path
    (query -> embed -> ANN search -> augment prompt -> LLM -> answer) running per
    request. Matches theory.md Section 4's "chunk -> embed -> store -> retrieve ->
    augment -> generate" outline (SPEC-AGENT-1 outline item 4) and NOTE-AGENT-2's RAG
    definition (IBM/Pinecone, checked 2026-09-02)."""
    W, H = 1150, 520
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    font = _load_font(14)
    small = _load_font(12)
    title_font = _load_font(18)
    section_font = _load_font(13)

    draw.text((20, 12), "RAG pipeline: offline indexing (top) and online query (bottom)", font=title_font, fill="black")

    # --- offline indexing path (top row) ---
    top_y0, top_y1 = 60, 140
    boxes_top = [
        (30, "Source\ndocuments\n(PDFs, wiki, ...)"),
        (250, "Chunk\n(split into\npassages)"),
        (470, "Embed\n(sentence-\ntransformer)"),
        (690, "Vector store\n(embeddings +\nANN index)"),
    ]
    bw = 190
    draw.text((30, top_y0 - 24), "OFFLINE -- runs once per document, and again whenever the corpus changes", font=section_font, fill="#555555")
    for x, label in boxes_top:
        _box(draw, (x, top_y0, x + bw, top_y1), label, font)
    for (x1, _), (x2, _) in zip(boxes_top, boxes_top[1:]):
        _harrow(draw, x1 + bw, x2, (top_y0 + top_y1) // 2)

    # --- online query path (bottom row) ---
    bot_y0, bot_y1 = 260, 340
    boxes_bot = [
        (30, "User query"),
        (250, "Embed query\n(same model)"),
        (470, "ANN search\n(top-k vs.\nvector store)"),
        (690, "Augment prompt\n(query + top-k\nchunks)"),
        (910, "LLM\ngenerate"),
    ]
    draw.text((30, bot_y0 - 24), "ONLINE -- runs on every request", font=section_font, fill="#555555")
    for x, label in boxes_bot:
        _box(draw, (x, bot_y0, x + bw, bot_y1), label, font)
    for (x1, _), (x2, _) in zip(boxes_bot, boxes_bot[1:]):
        _harrow(draw, x1 + bw, x2, (bot_y0 + bot_y1) // 2)

    # vector store (top) feeds the ANN search (bottom) -- the link between the two paths
    store_x = 690 + bw // 2
    search_x = 470 + bw // 2
    draw.line([(store_x, top_y1), (store_x, 200), (search_x, 200), (search_x, bot_y0)], fill="#b04a2f", width=2)
    draw.polygon(
        [(search_x, bot_y0), (search_x - 5, bot_y0 - 10), (search_x + 5, bot_y0 - 10)],
        fill="#b04a2f",
    )
    draw.text((search_x + 10, 185), "top-k vectors looked up here", font=small, fill="#b04a2f")

    # final answer
    ans_x = 910 + bw // 2
    draw.text(
        (ans_x - 70, bot_y1 + 20),
        "Answer, grounded in\nretrieved chunks",
        font=font,
        fill="black",
    )
    draw.line([(ans_x, bot_y1), (ans_x, bot_y1 + 18)], fill="black", width=2)
    draw.polygon(
        [(ans_x, bot_y1 + 18), (ans_x - 5, bot_y1 + 8), (ans_x + 5, bot_y1 + 8)],
        fill="black",
    )

    draw.text(
        (30, 420),
        "\"chunk -> embed -> store\" happens offline; \"query -> embed -> search -> augment -> generate\"\n"
        "happens per request. See theory.md Section 4 for what each step does and for when RAG beats\n"
        "fine-tuning or a longer context window.",
        font=font,
        fill="#333333",
    )

    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    img.save(path)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# 5. MCP client/server diagram
# ---------------------------------------------------------------------------

def draw_mcp_diagram(path: Path) -> None:
    """Draws MCP's host/client/server architecture: a Host (the LLM application) holds
    one Client per connection, each Client talks JSON-RPC to one MCP Server, and each
    Server exposes some mix of Tools / Resources / Prompts. Matches NOTE-AGENT-2's MCP
    definition, citing the official spec
    https://modelcontextprotocol.io/specification/2026-07-28 (checked 2026-09-02):
    "Hosts: LLM applications that initiate connections. Clients: connectors within the
    host application. Servers: services that provide context and capabilities."""
    W, H = 1000, 510
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    font = _load_font(14)
    small = _load_font(12)
    title_font = _load_font(18)

    draw.text((20, 12), "MCP: host / client / server roles (Model Context Protocol spec, 2026-07-28)", font=title_font, fill="black")

    # Host application (large outer box)
    host_box = (30, 60, 430, 420)
    _box(draw, host_box, "", font, fill="#f5f5f5")
    draw.text((45, 68), "Host application (e.g. an agent / IDE / chat app)", font=font, fill="black")

    # LLM inside host
    _box(draw, (60, 100, 400, 150), "LLM", font, fill="#eef3fb")

    # Two clients inside host, each -> one server
    client_boxes = [(60, 190, 400, 240, "Client A"), (60, 260, 400, 310, "Client B")]
    for x0, y0, x1, y1, label in client_boxes:
        _box(draw, (x0, y0, x1, y1), f"{label}\n(1:1 connection)", font)

    draw.text((60, 330), "1 client per server connection --\nMCP calls this the host/client/server split.", font=small, fill="#555555")

    # Arrow LLM -> clients (host wires the LLM's tool calls to the right client)
    _varrow(draw, 230, 150, 190)

    # Servers, each in its own box to the right
    servers = [
        (520, 100, 900, 210, "MCP Server: database\nTools: run_query()\nResources: schema docs"),
        (520, 250, 900, 360, "MCP Server: filesystem\nTools: read_file(), search()\nResources: file contents"),
    ]
    for x0, y0, x1, y1, label in servers:
        _box(draw, (x0, y0, x1, y1), label, font, fill="#fdf3e0")

    # JSON-RPC arrows: Client A -> DB server, Client B -> filesystem server
    _harrow(draw, 400, 520, 215, "JSON-RPC", small)
    _harrow(draw, 400, 520, 285, "JSON-RPC", small)

    draw.text(
        (30, 435),
        "Each server independently exposes Tools (functions the model can call), Resources (data the client can\n"
        "read) and Prompts (reusable templates) -- the same three primitives regardless of what the server wraps.\n"
        "The host never needs a bespoke integration per data source: it speaks one protocol to every server.",
        font=font,
        fill="#333333",
    )

    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    img.save(path)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
def main() -> None:
    doc_ids = list(DOCS.keys())
    doc_matrix = embed([DOCS[d] for d in doc_ids])

    rows = run_retrieval_demo(doc_matrix, doc_ids)
    render_retrieval_table_png(rows, ARTEFACTS_DIR / "retrieval_result_table.png")

    draw_rag_pipeline_diagram(ARTEFACTS_DIR / "rag_pipeline_diagram.png")
    draw_mcp_diagram(ARTEFACTS_DIR / "mcp_client_server_diagram.png")


if __name__ == "__main__":
    main()

"""Retrieve the top-k most relevant chunks for a question -- the "online"/query half of
the RAG pipeline. Fully runnable with no LLM API key: this is pure numpy cosine similarity
over vectors produced by ingest.py.

Companion code for:
  Agentic Engineering/Worked Examples/rag-over-pdfs.md

What it does:
  1. Load the index ingest.py already built (index/embeddings.npy + index/chunks.jsonl).
  2. Embed the incoming question with the SAME model used to embed the chunks (LO2/LO3 --
     mixing embedding models here would silently break retrieval, see the chapter's
     pitfalls section).
  3. Score every chunk by cosine similarity (a plain dot product, since ingest.py already
     L2-normalised every vector) and return the top k.

Run (after ingest.py has produced index/ at least once):
    .venv-agent/Scripts/python.exe "Agentic Engineering/Worked Examples/code/rag_pdf/retrieve.py"
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from ingest import CHUNKS_PATH, EMBEDDING_MODEL_NAME, EMBEDDINGS_PATH, Chunk

# A handful of questions the sample handbook (make_sample_pdf.py) was written to have
# clean, checkable answers for -- one per policy section, phrased the way a person would
# actually ask, not by quoting the document's own words back at it.
SAMPLE_QUESTIONS: list[str] = [
    "How long does the on-call rotation last, and how often does an engineer take primary on-call?",
    "What is the rate limit for the public fleet-status API?",
    "How long are application logs retained before they are deleted?",
    "How many engineers must review a database migration before it can run against staging?",
]


def load_index() -> tuple[list[Chunk], np.ndarray]:
    """Load the on-disk index built by ingest.py. Raises a clear error if ingest.py has
    not been run yet, rather than silently rebuilding -- in a real pipeline, indexing and
    querying are separate stages run by separate processes (often separate machines), and
    conflating them here would hide that."""
    if not EMBEDDINGS_PATH.exists() or not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"No index found at {EMBEDDINGS_PATH.parent}/. Run ingest.py first:\n"
            f"  .venv-agent/Scripts/python.exe {Path(__file__).parent / 'ingest.py'}"
        )
    embeddings = np.load(EMBEDDINGS_PATH)
    chunks: list[Chunk] = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            chunks.append(Chunk(**record))
    return chunks, embeddings


def top_k(
    query: str,
    chunks: list[Chunk],
    embeddings: np.ndarray,
    model: SentenceTransformer,
    k: int = 3,
) -> list[tuple[Chunk, float]]:
    """Return the k chunks with highest cosine similarity to `query`, highest first.

    Both `embeddings` (from ingest.py) and the query embedding below are L2-normalised
    unit vectors, so `embeddings @ query_vector` computes cosine similarity for every
    chunk in one matrix-vector product -- no separate normalisation or a
    scipy.spatial.distance.cosine() call per row needed.
    """
    query_vec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    query_vec = query_vec.astype(np.float32)[0]
    scores = embeddings @ query_vec
    top_idx = np.argsort(-scores)[:k]
    return [(chunks[i], float(scores[i])) for i in top_idx]


def main() -> None:
    chunks, embeddings = load_index()
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"Loaded index: {len(chunks)} chunks, {embeddings.shape[1]}-dim embeddings\n")

    for question in SAMPLE_QUESTIONS:
        print(f"Q: {question}")
        for chunk, score in top_k(question, chunks, embeddings, model, k=3):
            preview = chunk.text[:100].replace("\n", " ")
            print(f"  [{score:.4f}] chunk {chunk.chunk_id} (page {chunk.page}): {preview}...")
        print()


if __name__ == "__main__":
    main()

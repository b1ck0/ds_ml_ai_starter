"""Ingest a PDF into a local, on-disk retrieval index: parse -> chunk -> embed -> save.

Companion code for:
  Agentic Engineering/Worked Examples/rag-over-pdfs.md

What it does (the "offline" half of a RAG pipeline -- runs once, ahead of any query):
  1. Parse: pull real text out of every page of the sample PDF with pdfplumber.
  2. Chunk: split each page's text into overlapping word windows (a chunk is the unit
     retrieval will compare against a question).
  3. Embed: turn every chunk into a 384-dimensional vector with a small local
     sentence-transformers model -- CPU only, no API key.
  4. Save: write the embeddings (a numpy array) and the chunk records (JSON Lines) to
     `index/`, so retrieve.py and answer.py can load them without re-parsing the PDF
     or re-embedding anything.

Environment: this file, retrieve.py, and answer.py all run in the shared `.venv-agent`
virtualenv (sentence-transformers==6.0.1, pdfplumber==0.11.10, numpy==2.5.2 -- installed
versions verified directly in this environment; sentence-transformers and pdfplumber also
match the pinned versions in NOTE-AGENT-2-rag-primitives, checked 2026-09-02).

Run (from anywhere -- Python adds this script's own directory to sys.path, so the
`import ingest` in retrieve.py/answer.py resolves regardless of your current directory):
    .venv-agent/Scripts/python.exe "Agentic Engineering/Worked Examples/code/rag_pdf/ingest.py"
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer

PDF_PATH = Path(__file__).parent / "sample" / "acme_handbook.pdf"
INDEX_DIR = Path(__file__).parent / "index"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"

# sentence-transformers/all-MiniLM-L6-v2: 384-dim output, Apache-2.0 licence, CPU-only,
# no API key -- verified in NOTE-AGENT-2-rag-primitives (checked 2026-09-02).
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Chunking: NOTE-AGENT-2's recommendation for a real-sized corpus is ~512 tokens with a
# 50-token overlap. This sample handbook is only three pages, so that would put almost
# the whole document in one chunk -- useless for demonstrating retrieval. We scale down
# to 100 words per chunk (this codebase has no tokenizer among its pinned deps, so a
# whitespace-split "word" stands in for a "token" here -- an approximation, not a token
# count; see the chapter prose for why that's an honest simplification, not a real
# tokenizer) with a 20-word overlap (20%, same ratio as the 512/50 recommendation),
# which yields several chunks per policy section in the sample PDF.
CHUNK_SIZE_WORDS = 100
CHUNK_OVERLAP_WORDS = 20


@dataclass
class Chunk:
    """One retrievable unit: a window of words from one page, plus enough metadata to
    cite it back to a human-checkable source (page number)."""

    chunk_id: int
    page: int
    text: str


def extract_pages(pdf_path: Path) -> list[str]:
    """Return one string of extracted text per PDF page, in page order."""
    with pdfplumber.open(pdf_path) as pdf:
        return [page.extract_text() or "" for page in pdf.pages]


def chunk_page(text: str, page_num: int, start_id: int, size: int, overlap: int) -> list[Chunk]:
    """Slide a `size`-word window across `text` with `overlap` words shared between
    consecutive windows, so a fact sitting near a chunk boundary still appears whole in
    at least one chunk."""
    words = text.split()
    if not words:
        return []
    step = size - overlap
    chunks: list[Chunk] = []
    cid = start_id
    i = 0
    while i < len(words):
        window = words[i : i + size]
        chunks.append(Chunk(chunk_id=cid, page=page_num, text=" ".join(window)))
        cid += 1
        if i + size >= len(words):
            break
        i += step
    return chunks


def chunk_document(
    pages: list[str], size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS
) -> list[Chunk]:
    """Chunk every page independently (chunks never span a page boundary here, which
    keeps the page-citation metadata exact) and assign globally increasing chunk ids."""
    all_chunks: list[Chunk] = []
    next_id = 0
    for page_num, page_text in enumerate(pages, start=1):
        page_chunks = chunk_page(page_text, page_num, next_id, size, overlap)
        all_chunks.extend(page_chunks)
        next_id += len(page_chunks)
    return all_chunks


def embed_chunks(chunks: list[Chunk], model: SentenceTransformer) -> np.ndarray:
    """Embed every chunk's text, L2-normalised, as float32.

    `normalize_embeddings=True` makes every vector unit length, so a plain dot product
    between two embeddings equals their cosine similarity -- retrieve.py relies on this
    to avoid a separate normalisation step at query time.
    """
    texts = [c.text for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return embeddings.astype(np.float32)


def build_index(pdf_path: Path = PDF_PATH) -> tuple[list[Chunk], np.ndarray]:
    """Run the full parse -> chunk -> embed pipeline in memory (no disk I/O)."""
    pages = extract_pages(pdf_path)
    chunks = chunk_document(pages)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = embed_chunks(chunks, model)
    return chunks, embeddings


def save_index(
    chunks: list[Chunk], embeddings: np.ndarray, index_dir: Path = INDEX_DIR
) -> None:
    """Persist the index to disk: one numpy array of vectors, one JSON-Lines file of
    chunk records (same row order, so row i of the array is chunk i)."""
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c)) + "\n")


def main() -> None:
    chunks, embeddings = build_index()
    save_index(chunks, embeddings)
    print(f"Parsed {PDF_PATH.name}: {len(chunks)} chunks from {len(extract_pages(PDF_PATH))} pages")
    print(f"Embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}")
    print(f"Saved index to {INDEX_DIR}/ (embeddings.npy, chunks.jsonl)")
    print("\nFirst 3 chunks:")
    for c in chunks[:3]:
        preview = c.text[:90].replace("\n", " ")
        print(f"  chunk {c.chunk_id} (page {c.page}, {len(c.text.split())} words): {preview}...")


if __name__ == "__main__":
    main()

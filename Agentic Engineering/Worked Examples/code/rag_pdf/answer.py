"""Assemble a grounded prompt from retrieved chunks and (optionally) generate a cited
answer -- the final, KEY-GATED stage of the RAG pipeline.

Companion code for:
  Agentic Engineering/Worked Examples/rag-over-pdfs.md

What it does:
  1. Retrieve the top-k chunks for a question (reuses retrieve.py -- no key needed).
  2. Assemble a grounded prompt: the retrieved chunks as context, the question, and an
     instruction to answer ONLY from that context and cite chunk ids.
  3. Generate: if ANTHROPIC_API_KEY and ANTHROPIC_MODEL are set, call the Anthropic
     Messages API for a real, cited answer. If not, print the assembled context and
     prompt -- exactly what would have been sent -- and exit cleanly. NO NETWORK CALL IS
     EVER MADE WITHOUT A KEY.

This file is DELIBERATELY separate from the no-key-required path (ingest.py, retrieve.py):

* Steps 1-2 (retrieve + assemble) run entirely on this chapter's pinned, verified
  dependencies (sentence-transformers, numpy) and make no network call.
* Step 3 (generate) requires a real ANTHROPIC_API_KEY (see .env.example) and the
  `anthropic` package, which is NOT one of this chapter's pinned dependencies -- install
  it yourself with `pip install anthropic` if you want to run this path.
* The `anthropic.Anthropic().messages.create(model=..., max_tokens=..., messages=[...])`
  call shape is verified against research/NOTE-AGENT-4-provider-sdks.md (anthropic SDK
  v1.3.0, checked against PyPI / the official Anthropic docs on 2026-09-02) -- not
  asserted from memory. The model id is read from ANTHROPIC_MODEL rather than
  hard-coded, on purpose: model ids change over time (NOTE-AGENT-4's verified current id
  at time of writing was `claude-opus-5`, checked 2026-09-02) and this chapter will not
  assert one from memory as a default that silently goes stale.

Run:
    .venv-agent/Scripts/python.exe "Agentic Engineering/Worked Examples/code/rag_pdf/answer.py"
"""
from __future__ import annotations

import os

from sentence_transformers import SentenceTransformer

from ingest import EMBEDDING_MODEL_NAME
from retrieve import Chunk, load_index, top_k

QUESTION = "How long does the on-call rotation last, and how often does an engineer take primary on-call?"
TOP_K = 3

PROMPT_TEMPLATE = """Answer the question using ONLY the context below. Cite the chunk id(s) \
and page number(s) you used. If the context does not contain the answer, say so -- do not guess.

Context:
{context}

Question: {question}
"""


def assemble_context(hits: list[tuple[Chunk, float]]) -> str:
    """Render retrieved chunks as labelled, citable blocks -- this is what makes the
    eventual answer "grounded": every fact the model is allowed to use is traceable back
    to a chunk id and page number here."""
    blocks = []
    for chunk, score in hits:
        blocks.append(
            f"[chunk {chunk.chunk_id}, page {chunk.page}, similarity {score:.4f}]\n{chunk.text}"
        )
    return "\n\n".join(blocks)


def build_prompt(question: str, hits: list[tuple[Chunk, float]]) -> str:
    return PROMPT_TEMPLATE.format(context=assemble_context(hits), question=question)


def main() -> None:
    chunks, embeddings = load_index()
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    hits = top_k(QUESTION, chunks, embeddings, model, k=TOP_K)
    prompt = build_prompt(QUESTION, hits)

    print(f"Question: {QUESTION}\n")
    print("Retrieved context (runs with no key -- this is the RAG part):\n")
    print(assemble_context(hits))
    print()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "--- NO-KEY FALLBACK ---\n"
            "ANTHROPIC_API_KEY is not set, so no network call is made. Above is the full\n"
            "grounded prompt's context -- exactly what would have been sent to the model,\n"
            "with every fact traceable to a chunk id and page number. This is RAG's honest\n"
            "mechanics: retrieval works and is inspectable without any LLM at all; only the\n"
            "final prose-generation step needs a key. Set ANTHROPIC_API_KEY (and optionally\n"
            "ANTHROPIC_MODEL) in .env -- see .env.example -- and re-run for a generated,\n"
            "cited answer."
        )
        return

    model_id = os.environ.get("ANTHROPIC_MODEL")
    if not model_id:
        print(
            "ANTHROPIC_API_KEY is set but ANTHROPIC_MODEL is not.\n"
            "Set ANTHROPIC_MODEL to a current Claude model id (NOTE-AGENT-4-provider-sdks\n"
            "verified 'claude-opus-5' as current on 2026-09-02 -- check the Anthropic docs\n"
            "for a newer id if you're reading this later) -- this chapter does not hard-code\n"
            "one, since model ids change."
        )
        return

    try:
        import anthropic
    except ImportError:
        print(
            "ANTHROPIC_API_KEY is set, but the 'anthropic' package is not installed in\n"
            "this environment. Install it with: pip install anthropic\n"
            "(Not one of this chapter's pinned dependencies -- only needed for this\n"
            "optional, key-gated generation step. See NOTE-AGENT-4-provider-sdks.)"
        )
        return

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model_id,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    print(f"--- GENERATED ANSWER (model={model_id}) ---\n")
    for block in response.content:
        if block.type == "text":
            print(block.text)


if __name__ == "__main__":
    main()

# RAG over PDFs — retrieval you can run, generation you can gate

*Agentic Engineering · Worked Examples · SPEC-AGENT-3*

[`theory.md`](../Theory/theory.md) (SPEC-AGENT-1) named the fix and drew the pipeline: **chunk → embed
→ store** offline, **query → embed → search → augment → generate** online. The previous chapter,
[`mcp-database-query-layer.md`](mcp-database-query-layer.md) (SPEC-AGENT-2), gave an agent its first
kind of external capability — structured rows, behind a narrow, typed tool boundary. This chapter
gives it a second kind: unstructured text, behind the same discipline. A PDF handbook is not a
database — there is no `SCHEMA` dict, no `SELECT ... WHERE`, no exact match to ask for. What it has
instead is *meaning*, and the tool for finding "the paragraph that's actually about this question"
is exactly what Section 3 of `theory.md` covered: embeddings, cosine similarity, and a vector index.

This chapter builds all of it, for real, over a real (if synthetic) three-page engineering handbook:
parse the PDF, chunk it, embed every chunk with a small local model, retrieve the top-k chunks for a
question by cosine similarity, and — only at the very last step, and only if you provide an API
key — generate a prose answer that cites which chunks it came from. Everything through retrieval runs
on a CPU with no key at all; **that is the actual point of this chapter**, not a simplification for
teaching purposes. RAG's retrieval half is honest, inspectable, ordinary code. Only the final "write
me a sentence" step needs an LLM.

## 1. What & why — a PDF is not a database

If SPEC-AGENT-2's `server.py` is a `@RestController` in front of a SQL database, this chapter's
`retrieve.py` is the equivalent in front of a document. Both exist for the same underlying reason: an
LLM has no way to reach into your data on its own (`theory.md` Section 1 — LLMs are stateless and
frozen at a training cutoff). The difference is the shape of the data and the shape of the query.
`query(entity="orders", filters={"status": "shipped"})` is exact — either a row's `status` column
equals `"shipped"` or it doesn't. "What's the on-call rotation policy?" against a 3-page handbook has
no exact-match analogue: the words "on-call rotation" might not even appear verbatim near the answer,
and there's no column to filter on. What you actually want is *the passage whose meaning is closest to
the question* — which is precisely what an embedding model plus cosine similarity computes
(`theory.md` Section 3).

A Java analogy worth keeping, and where it breaks: a full-text search index (Lucene, Postgres
`tsvector`) is closer to `query()` than you'd think — it's still fundamentally matching *tokens*, just
with stemming and ranking on top. An embedding index answers a different question entirely: not "which
documents contain these words" but "which documents mean something close to this." Two passages that
share almost no vocabulary can still embed close together if they're about the same thing — and two
passages that share a lot of vocabulary can embed far apart if they're not.

## 2. Parse + chunk the sample PDF

### The sample document

This chapter needs a real, multi-page, text-based PDF with facts specific enough that a retrieval
result can be checked by eye — "did it find the right paragraph" needs a ground truth to check against.
`make_sample_pdf.py` (companion code, reused as-is from this chapter's own scaffolding) synthesizes a
three-page, six-section fictional internal handbook, **"Acme Robotics — Field Service Engineering
Handbook."** "Acme Robotics" is the classic placeholder company name — nothing in it describes a real
organization. Each of its six sections (deployment runbook, incident response, database migrations,
on-call rotation, API rate limits, log retention) states one or two concrete, checkable numbers — a
rotation length, a rate limit, a retention period — so a question like "how long are logs retained"
has exactly one right paragraph to find. It needs `matplotlib` (already in this project's shared
`.venv`, used purely as a text-layout engine — not `.venv-agent`, which has no PDF-authoring library),
and its output is committed at
[`code/rag_pdf/sample/acme_handbook.pdf`](code/rag_pdf/sample/acme_handbook.pdf) so you don't have to
regenerate it to run the rest of this chapter.

### Parsing — pdfplumber

```python
import pdfplumber

with pdfplumber.open("sample/acme_handbook.pdf") as pdf:
    print(len(pdf.pages))
    print(pdf.pages[0].extract_text()[:200])
```

```text
3
ACME ROBOTICS -- FIELD SERVICE ENGINEERING HANDBOOK (v3.2, internal)
1. Deployment Runbook
All production deployments to the fleet-control service go through a canary
rollout: the new build first serves 5 percent of regional traffic for a minimum
of twenty minutes before promotion. A canary is considered failed if error rate
exceeds 2 percent or p99 latency
```

`pdfplumber.open(path)` returns a `PDF` object with one `Page` per PDF page; `page.extract_text()`
returns that page's text as one string, preserving reading order well enough for prose (it's built on
`pdfminer.six`) — output captured for real above, `.venv-agent`, checked 2026-09-03.
`pdfplumber==0.11.10` ([source: PyPI](https://pypi.org/project/pdfplumber/), checked 2026-09-03) is
this chapter's parsing library — NOTE-AGENT-2-rag-primitives.md's recommendation for layout-preserving
extraction, and the version pinned there matches what's actually installed in `.venv-agent`
(`pdfplumber 0.11.10`, confirmed via `importlib.metadata.version`). **`pypdf` is the alternative**
NOTE-AGENT-2 names, with more control over low-level page objects and less layout preservation;
`.venv-agent` has `pypdf 6.16.2` installed — newer than the `5.7.0` NOTE-AGENT-2 captured on
2026-09-02, a reminder of how fast this ecosystem moves and why this chapter checks the *installed*
version directly rather than trusting a snapshot. Either library would work for this chapter's fairly
simple, single-column layout; pdfplumber is what the code below actually uses.

### Chunking — a word-count proxy for tokens, with overlap

A **chunk** is the unit retrieval actually compares against a question — not a whole document (too
broad to match anything specific) and not a single sentence (too narrow to carry context). NOTE-AGENT-2's
recommendation for a real-sized corpus is roughly 512 tokens per chunk with a 50-token overlap between
consecutive chunks, tuned from there against measured retrieval quality. This project has no tokenizer
among its pinned dependencies, so `ingest.py` splits on whitespace and counts **words**, not tokens —
an honest approximation stated as one, not a claim that a word equals a token (it doesn't; a real
tokenizer like the ones covered in
[`llm-text-generation.md`](../../Machine%20Learning/Worked%20Examples/llms/llm-text-generation.md),
SPEC-ML-11, would split words further into sub-word pieces). This sample handbook is only three pages —
512 words would swallow almost an entire page into one chunk, leaving nothing to demonstrate retrieval
against — so this chapter scales down to **100 words per chunk, 20 words of overlap** (the same 20%
overlap ratio as NOTE-AGENT-2's 512/50 recommendation), which produces several chunks per policy
section:

```python
def chunk_page(text: str, page_num: int, start_id: int, size: int, overlap: int) -> list[Chunk]:
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
```

`step = size - overlap` is the whole trick: each new window starts `step` words after the previous
one started, so the last `overlap` words of one chunk are the first `overlap` words of the next. That
is what keeps a fact from vanishing off both sides of a chunk boundary — Section 6 shows what happens
when `overlap=0`. Full file: [code/rag_pdf/ingest.py](code/rag_pdf/ingest.py). Real run:

```text
.venv-agent/Scripts/python.exe "Agentic Engineering/Worked Examples/code/rag_pdf/ingest.py"
```

```text
Parsed acme_handbook.pdf: 9 chunks from 3 pages
Embeddings: shape=(9, 384), dtype=float32
Saved index to .../code/rag_pdf/index/ (embeddings.npy, chunks.jsonl)

First 3 chunks:
  chunk 0 (page 1, 100 words): ACME ROBOTICS -- FIELD SERVICE ENGINEERING HANDBOOK (v3.2, internal) 1. Deployment Runbook...
  chunk 1 (page 1, 100 words): flagged; rollbacks beyond that window require a written incident postmortem regardless of ...
  chunk 2 (page 1, 78 words): any safety-critical robot command path -- requires the on-call engineer to acknowledge the...
```

Nine chunks from a three-page document, three-ish per page — small enough to read every one by eye
(the full set is in the retrieval table artefact, Section 4), large enough that retrieval has to
actually discriminate between them.

## 3. Embed + store — a local model, a local index

### The embedding model

**`sentence-transformers/all-MiniLM-L6-v2`** — the same model `theory.md`'s demo used — embeds each
chunk as a 384-dimensional dense vector, runs on CPU, needs no API key, and is Apache 2.0 licensed
([source: Hugging Face model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
confirmed live 2026-09-03: "maps sentences & paragraphs to a 384 dimensional dense vector space",
"License: apache-2.0"; NOTE-AGENT-2-rag-primitives.md). Its **max input length is 256 tokens** — this
chapter's 100-word chunks sit comfortably under that; Section 6 covers what happens if a chunk doesn't.
Package: `sentence-transformers==6.0.1`
([source: PyPI](https://pypi.org/project/sentence-transformers/), checked against NOTE-AGENT-2 on
2026-09-02) — confirmed installed at that exact version in `.venv-agent` while writing this chapter.

```python
def embed_chunks(chunks: list[Chunk], model: SentenceTransformer) -> np.ndarray:
    texts = [c.text for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return embeddings.astype(np.float32)
```

`normalize_embeddings=True` is the detail worth pausing on: it makes every output vector unit length
(`‖v‖ = 1`). Cosine similarity is normally `(a·b) / (‖a‖ ‖b‖)` — but when both vectors already have
unit length, the denominator is `1`, so cosine similarity collapses to a plain dot product. Section 4's
`embeddings @ query_vector` relies on exactly this identity to score every chunk against a query in one
matrix-vector multiply, with no explicit division anywhere.

### The "store" — a local numpy index instead of pgvector

NOTE-AGENT-2 names two ways to run vector search: **pgvector** (a PostgreSQL extension — `CREATE
EXTENSION vector`, a `<=>` cosine-distance operator, ordinary SQL — the production-shaped option,
which SPEC-AGENT-0 stands up as this project's local vector store) and **local FAISS or plain numpy**
for development, testing, or a genuinely small corpus. Nine chunks is about as small a corpus as
exists, and this chapter's environment has no running Postgres/pgvector instance to reach — so
`ingest.py` writes the embeddings to a plain `.npy` array and the chunk records to a `.jsonl` file:

```python
def save_index(chunks: list[Chunk], embeddings: np.ndarray, index_dir: Path = INDEX_DIR) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c)) + "\n")
```

This is a **Flat index** in the vocabulary `theory.md` Section 3 introduced: brute-force, exact,
O(n) per query — no HNSW graph, no IVF clusters, because at 9 rows a full scan is instant and an ANN
index would add complexity for zero benefit. The same code, run against a real pgvector table, is
what an unindexed `<=>` query does under the hood; the difference is only where the vectors live, not
the math. `index/embeddings.npy` and `index/chunks.jsonl` are committed alongside this chapter's code
so `retrieve.py` and `answer.py` don't need `ingest.py` re-run first — but they're fully reproducible
from it (same PDF, same chunking, deterministic embedding inference — no sampling, nothing stochastic
anywhere in this pipeline).

## 4. Retrieve top-k — real cosine search, no key, real output

`retrieve.py` loads the saved index, embeds an incoming question with the **same** model that embedded
the chunks (`theory.md` Section 4 flagged this as "load-bearing" — Section 6 here shows why), and scores
every chunk with one matrix-vector product:

```python
def top_k(
    query: str, chunks: list[Chunk], embeddings: np.ndarray, model: SentenceTransformer, k: int = 3,
) -> list[tuple[Chunk, float]]:
    query_vec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    query_vec = query_vec.astype(np.float32)[0]
    scores = embeddings @ query_vec
    top_idx = np.argsort(-scores)[:k]
    return [(chunks[i], float(scores[i])) for i in top_idx]
```

Full file: [code/rag_pdf/retrieve.py](code/rag_pdf/retrieve.py). Real run, four natural-language
questions the sample handbook was written to have one clear answer for each, against the committed
index, `.venv-agent`, checked 2026-09-03:

```text
.venv-agent/Scripts/python.exe "Agentic Engineering/Worked Examples/code/rag_pdf/retrieve.py"
```

```text
Loaded index: 9 chunks, 384-dim embeddings

Q: How long does the on-call rotation last, and how often does an engineer take primary on-call?
  [0.7590] chunk 4 (page 2): before it can run against the staging database. Migrations that touch a table larger than ten millio...
  [0.7047] chunk 5 (page 2): If the primary on-call engineer does not acknowledge a page within ten minutes, the incident automat...
  [0.3741] chunk 1 (page 1): flagged; rollbacks beyond that window require a written incident postmortem regardless of customer i...

Q: What is the rate limit for the public fleet-status API?
  [0.7618] chunk 6 (page 3): 5. API Rate Limits The public fleet-status API enforces a rate limit of 100 requests per minute per ...
  [0.4831] chunk 7 (page 3): the increased limit is needed. Rate limit counters reset on a rolling sixty-second window per key, n...
  [0.3927] chunk 0 (page 1): ACME ROBOTICS -- FIELD SERVICE ENGINEERING HANDBOOK (v3.2, internal) 1. Deployment Runbook All produ...

Q: How long are application logs retained before they are deleted?
  [0.5274] chunk 7 (page 3): the increased limit is needed. Rate limit counters reset on a rolling sixty-second window per key, n...
  [0.3179] chunk 8 (page 3): precise GPS coordinates (rounded to the nearest 1 kilometre grid cell) before they leave hot storage...
  [0.1964] chunk 4 (page 2): before it can run against the staging database. Migrations that touch a table larger than ten millio...

Q: How many engineers must review a database migration before it can run against staging?
  [0.6816] chunk 3 (page 2): 3. Database Migration Policy Schema migrations against the fleet-telemetry database run through Flyw...
  [0.4179] chunk 4 (page 2): before it can run against the staging database. Migrations that touch a table larger than ten millio...
  [0.2857] chunk 0 (page 1): ACME ROBOTICS -- FIELD SERVICE ENGINEERING HANDBOOK (v3.2, internal) 1. Deployment Runbook All produ...
```

Full table (all four questions, top-3 each, with **full** chunk text — not truncated to 100
characters): [artefacts/retrieval_table.md](artefacts/retrieval_table.md).

**Read the top-1 result for the first question before trusting the preview.** The 100-character preview
printed above for chunk 4 is *"before it can run against the staging database. Migrations that touch a
table larger than ten millio..."* — which looks, at a glance, like the wrong section entirely (that's
migration-policy text, not on-call policy). It is not a miss. Chunk 4 is a 100-word overlapping window
that happens to *start* mid-sentence in the migration-policy section and, a few dozen words later,
*contains* the entire on-call answer: "The primary on-call rotation is one week long, handing off every
Monday at 09:00 local time. Each engineer on the fleet-control team rotates through primary on-call
roughly once every six weeks" (full text in the artefact table, or run `retrieve.py` yourself). The
same is true of the third question's top-1 hit (chunk 7): its preview shows leftover rate-limit text,
and the actual retention numbers ("30 days in hot storage... 11 months... 12 months total") sit later
in the same chunk. **The lesson, not just the anecdote:** a short preview string is not the chunk — it
is showing you where a chunk's window *starts*, not whether it contains the fact you asked about.
Section 5's assembled prompt always includes the full chunk text for exactly this reason; never judge a
retrieval hit by a truncated preview.

Every top-1 result above is, on inspection, the chunk that actually answers its question — a clean
result, and Section 6 shows a chunking configuration where that stops being true.

One lever this chapter deliberately doesn't pull: **re-ranking**. A production pipeline often follows
this exact cosine top-k step with a second, more expensive model — a cross-encoder that looks at the
query and each candidate chunk *together* (rather than as two independently-embedded vectors) and
re-scores them for precision — before handing the final, re-ranked top-k to the LLM. That's a real
technique worth knowing the name of; it's out of scope here because this chapter's nine-chunk corpus is
too small for a coarse-then-precise two-stage pipeline to demonstrate any benefit over plain cosine
top-k.

## 5. Generate a grounded answer — key-gated, with a working no-key fallback

Retrieval hands back chunks and numbers; it does not hand back a sentence. Turning "chunk 4, score
0.7590, contains a paragraph about on-call rotation" into "The on-call rotation is one week long" is
the LLM's job — the one step in this pipeline that genuinely needs a model call. `answer.py` splits
that cleanly into an always-runnable part and a key-gated part:

```python
def assemble_context(hits: list[tuple[Chunk, float]]) -> str:
    blocks = []
    for chunk, score in hits:
        blocks.append(
            f"[chunk {chunk.chunk_id}, page {chunk.page}, similarity {score:.4f}]\n{chunk.text}"
        )
    return "\n\n".join(blocks)


PROMPT_TEMPLATE = """Answer the question using ONLY the context below. Cite the chunk id(s) \
and page number(s) you used. If the context does not contain the answer, say so -- do not guess.

Context:
{context}

Question: {question}
"""
```

Every fact the model is *allowed* to use is labelled with a chunk id and page number before it ever
reaches the prompt — that labelling is what makes the eventual answer "grounded" instead of a plain
chat completion: the model is instructed to answer only from the supplied context and to say so if the
context doesn't cover the question, rather than filling the gap from its own training data (Section 6
covers what happens if you skip that instruction). Then the key gate, mirroring the exact
three-check pattern SPEC-AGENT-2's `llm_client.py` established (no key → stop; key but no model id →
stop; key and model but package missing → stop; only then call):

```python
# ...continuing inside answer.py's main() (full context in the file itself):
def _no_key_gate(api_key: str | None) -> bool:
    if not api_key:
        print(
            "--- NO-KEY FALLBACK ---\n"
            "ANTHROPIC_API_KEY is not set, so no network call is made. Above is the full\n"
            "grounded prompt's context -- exactly what would have been sent to the model, ..."
        )
        return True
    return False
```

Full file: [code/rag_pdf/answer.py](code/rag_pdf/answer.py). Real, captured run with no key set —
`.venv-agent`, checked 2026-09-03:

```text
.venv-agent/Scripts/python.exe "Agentic Engineering/Worked Examples/code/rag_pdf/answer.py"
```

```text
Question: How long does the on-call rotation last, and how often does an engineer take primary on-call?

Retrieved context (runs with no key -- this is the RAG part):

[chunk 4, page 2, similarity 0.7590]
before it can run against the staging database. Migrations that touch a table larger than ten million rows must run during the Tuesday or Wednesday maintenance window and must include a tested rollback script in the same pull request. 4. On-Call Rotation & Escalation The primary on-call rotation is one week long, handing off every Monday at 09:00 local time. Each engineer on the fleet-control team rotates through primary on-call roughly once every six weeks, assuming a team of six. If the primary on-call engineer does not acknowledge a page within ten minutes, the incident automatically escalates to the secondary

[chunk 5, page 2, similarity 0.7047]
If the primary on-call engineer does not acknowledge a page within ten minutes, the incident automatically escalates to the secondary on-call engineer; if the secondary does not acknowledge within a further ten minutes, it escalates to the engineering lead directly. Planned on-call swaps must be requested at least 48 hours in advance through the on-call calendar tool.

[chunk 1, page 1, similarity 0.3741]
flagged; rollbacks beyond that window require a written incident postmortem regardless of customer impact. Deployments are only permitted Monday through Thursday before 16:00 local time, to keep a full business day of on-call coverage available in case of a slow-burn regression. Emergency hotfixes outside that window require sign-off from the on-call lead and the service owner. 2. Incident Response Policy Incidents are classified SEV-1 through SEV-4 by customer impact. A SEV-1 incident -- full outage of the fleet-control API or any safety-critical robot command path -- requires the on-call engineer to acknowledge the page within five minutes and open a

--- NO-KEY FALLBACK ---
ANTHROPIC_API_KEY is not set, so no network call is made. Above is the full
grounded prompt's context -- exactly what would have been sent to the model,
with every fact traceable to a chunk id and page number. This is RAG's honest
mechanics: retrieval works and is inspectable without any LLM at all; only the
final prose-generation step needs a key. Set ANTHROPIC_API_KEY (and optionally
ANTHROPIC_MODEL) in .env -- see .env.example -- and re-run for a generated,
cited answer.
```

Full transcript: [artefacts/answer_no_key_transcript.txt](artefacts/answer_no_key_transcript.txt).
Notice this is a genuine run with real, non-fabricated numbers — a top-1 similarity of `0.7590` for
chunk 4, the exact same chunk Section 4 already showed you, the same "one week long... every six weeks"
answer sitting inside it. **No answer was invented** for this chapter: with no key, none is generated —
consistent with SPEC-AGENT-2's `llm_client.py` no-key transcript, and for the same reason this project's
rules give: typing out what a live model would say without a live model having said it is exactly the
kind of ungrounded claim this book forbids.

If you *do* set `ANTHROPIC_API_KEY` and `ANTHROPIC_MODEL`, and have `pip install anthropic`'d into
`.venv-agent` (the `anthropic` package is **not** one of this chapter's pinned, verified dependencies —
same posture as SPEC-AGENT-2's `llm_client.py`), the call itself is:

```python
client = anthropic.Anthropic(api_key=api_key)
response = client.messages.create(
    model=model_id,
    max_tokens=512,
    messages=[{"role": "user", "content": prompt}],
)
```

That exact shape — `anthropic.Anthropic(api_key=...)`, `.messages.create(model=..., max_tokens=...,
messages=[...])` — is verified against `anthropic` SDK **v1.3.0**
([source: PyPI](https://pypi.org/project/anthropic/), confirmed live 2026-09-03: "The most recent
version... is 1.3.0") and the official Anthropic docs
([source: platform.claude.com](https://platform.claude.com/docs/en/build-with-claude/working-with-messages),
both cited in `research/NOTE-AGENT-4-provider-sdks.md`, checked 2026-09-02) — not asserted from memory.
The model id is deliberately read from `ANTHROPIC_MODEL` rather than hard-coded: NOTE-AGENT-4's
verified-current id at time of writing was `claude-opus-5` (checked 2026-09-02), and model ids move on;
this chapter will not bake in a default that quietly goes stale.

![RAG-over-PDFs pipeline: acme_handbook.pdf through ingest.py's parse+chunk+embed, into the local index, through retrieve.py's cosine top-k, into answer.py's prompt assembly, hitting a key-gated boundary that either falls back to printing the assembled context (no key) or calls the Anthropic Messages API for a generated, cited answer (key set)](artefacts/rag_pdf_pipeline_diagram.png)

## 6. Pitfalls

**Bad chunking splits one answer across two retrieval units.** Section 4's 100-word/20-word-overlap
chunking put the complete "one week long... every six weeks" answer inside a single chunk. Cut the
overlap to zero and shrink chunks to 20 words — a real, run configuration, not a hypothetical — and the
same two facts land in *adjacent, non-overlapping* chunks instead of one:

```text
33 chunks (size=20, overlap=0)
[0.8027] chunk 19 (page 2): Each engineer on the fleet-control team rotates through primary on-call roughly once every six weeks, assuming a team of six.
[0.7792] chunk 18 (page 2): On-Call Rotation & Escalation The primary on-call rotation is one week long, handing off every Monday at 09:00 local time.
[0.5799] chunk 20 (page 2): If the primary on-call engineer does not acknowledge a page within ten minutes, the incident automatically escalates to the secondary
```

Both facts still make the top-3 here — but if a pipeline retrieved only the top-1 (chunk 19, "every six
weeks"), the answer to "how long does the rotation last" would be silently missing, even though the
document plainly states it two sentences earlier. Wider overlap is precisely the fix: it's what let
Section 4's 100/20 chunking keep both sentences inside one chunk (chunk 4) in the first place. **Always
inspect real retrieved chunks for real questions before trusting a chunking configuration** — a score
looking reasonable is not the same as the answer actually being present in what you retrieved.

**Bad chunking can also push the right chunk out of top-k entirely**, not just split it. The same
20-word/no-overlap configuration, asked "how long are application logs retained":

```text
#1 [0.6238] chunk 30: months before deletion, for a total retention period of 12 months. Logs containing customer location data must be redacted of
#2 [0.4859] chunk 28: minute, so bursts near a minute boundary are still capped correctly. 6. Logging & Data Retention Application logs from the
#3 [0.3947] chunk 32: logs requires a documented business justification and sign-off from the data protection lead, logged in the access-request system.
#4 [0.3532] chunk 29: fleet-control service are retained for 30 days in hot storage and then moved to cold storage for an additional 11
```

The chunk holding the actual "30 days... 11 months" numbers ranks **#4** — one place outside a
`k=3` retrieval. A pipeline built on this chunking would hand the model chunk 30 ("12 months" total,
with no breakdown) and chunk 28 (the section heading, no numbers at all) and never surface the
hot-storage/cold-storage split — not because the model failed, but because the chunk that had the
answer never made it into the context in the first place. This is the same failure `theory.md` Section
6 named generically ("retrieval misses — including near-misses that look like hits"); here it's the
same handbook, the same questions, run twice, so you can see the exact chunking choice that causes it.

**A chunk over the embedding model's 256-token input limit gets silently truncated**, not rejected
(NOTE-AGENT-2-rag-primitives.md). This chapter's 100-word chunks sit safely under that in practice, but
nothing in `SentenceTransformer.encode()` raises or warns if a much larger chunk size quietly drops its
back half before embedding — the resulting vector represents only the part of the chunk that fit.

**Embedding a query with a different model than the one that embedded your documents.** Named as
"load-bearing" in `theory.md` Section 4, and worth repeating as the pitfall it is: `EMBEDDING_MODEL_NAME`
in `ingest.py` and `retrieve.py` must be the literal same string, because two different embedding
models place identical text at different, incompatible coordinates — cosine similarity computed across
the two spaces is a number, but not a meaningful one, and nothing raises an error to tell you this
happened. Change the embedding model, and the whole `index/` directory needs re-embedding from scratch.

**Ungrounded generation — a prompt that doesn't force "I don't know."** Section 5's prompt template
explicitly instructs the model to answer *only* from the supplied context and to say so if the context
doesn't contain the answer. Drop that instruction and hand a general-purpose chat model the same
retrieved chunks plus a question outside their scope, and a fluent, confident, *wrong* answer is a real
risk — the model can lean on whatever it already "knows" from training instead of admitting the
retrieved context doesn't cover the question. Retrieval quality (the two pitfalls above) and prompt
discipline (this one) are two separate failure modes, and a RAG pipeline needs both addressed — good
retrieval feeding an ungrounded prompt still hallucinates; a grounded prompt fed bad retrieval just
confidently answers the wrong question.

**Stale index.** `index/embeddings.npy` and `index/chunks.jsonl` are a snapshot of the PDF at the moment
`ingest.py` last ran. If the source handbook changes — a policy number updated, a new section added —
retrieval keeps confidently returning the old numbers until someone re-runs `ingest.py`. `theory.md`
Section 6 named this as a cache-invalidation problem with a familiar shape; nothing in this chapter's
code re-indexes automatically, on purpose, so that the offline/online split (Section 3) stays visible
rather than hidden behind a "just works" abstraction.

## 7. Recap & what's next

- A PDF has no schema and no exact match to query — the retrieval question is "which passage means the
  closest thing to this," not "which row satisfies this predicate" (Section 1).
- **Parse** with `pdfplumber.open(path).pages[i].extract_text()` (`pdfplumber==0.11.10`); **chunk**
  with a sliding, overlapping window (`step = size - overlap`) so a fact near a boundary survives in at
  least one chunk — this chapter used 100 words/20 words overlap for a 3-page document, scaled down
  from NOTE-AGENT-2's 512/50-token recommendation for a real-sized corpus (Section 2).
- **Embed** every chunk with `sentence-transformers/all-MiniLM-L6-v2` (384-D, Apache 2.0, CPU, no key)
  with `normalize_embeddings=True`, so cosine similarity becomes a plain dot product; **store** the
  vectors as a local numpy array for this chapter's 9-chunk corpus, or in pgvector (SPEC-AGENT-0) for
  anything larger (Section 3).
- **Retrieve** with `embeddings @ query_vector` and `np.argsort` — an exact Flat scan, real output
  captured for all four sample questions, and a genuine reminder that a truncated preview is not proof
  a retrieval missed (Section 4).
- **Generate** is the one key-gated step: a grounded prompt template that forces "answer only from this
  context, or say you can't," `anthropic.Anthropic().messages.create(...)` (verified against
  NOTE-AGENT-4, `anthropic==1.3.0`) behind three explicit checks (key, model id, package installed),
  and a no-key fallback that prints the real assembled context instead of ever fabricating an answer
  (Section 5).
- Bad chunking can split an answer across chunks or push the right chunk out of top-k entirely — both
  demonstrated with real, run numbers, not hypotheticals; a 256-token embedding limit truncates
  silently; mismatched embedding models produce meaningless scores with no error; an undisciplined
  prompt lets the model hallucinate past what retrieval actually found; and a `index/` that never gets
  re-built is a stale cache with a familiar shape (Section 6).

**SPEC-AGENT-4 (Invoice Agent)**, next, composes this chapter's retrieval discipline with
SPEC-AGENT-2's MCP write-tool pattern: an agent that turns an unstructured PDF (an invoice, not a
policy handbook) into validated rows written through a narrow, typed MCP tool — structured extraction
instead of open-ended retrieval, but built on the same underlying instinct both this chapter and the
last one share: give the model a narrow, inspectable boundary in front of whatever it can't touch
directly, and never let it operate on data you haven't validated first.

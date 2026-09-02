# NOTE-AGENT-2: RAG Primitives — Embeddings, Vector Search, PDF Extraction, and Definitions

**Answer:** sentence-transformers all-MiniLM-L6-v2 outputs 384-dimensional vectors (Apache 2.0 license, runs on CPU with no API key); vector search via pgvector SQL with `<=>` cosine operator or local FAISS/IVF-Flat; PDF extraction via pdfplumber 0.11.10 or pypdf 5.7.0; RAG, ANN (HNSW/IVF), and MCP definitions sourced from official specs and authoritative docs.

**Evidence:**

*Embedding model (sentence-transformers all-MiniLM-L6-v2):*
- **Dimensions:** 384-dimensional dense vector space (source: Hugging Face model card https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **License:** Apache 2.0 (source: Hugging Face model metadata)
- **Max input length:** 256 tokens (longer input is truncated)
- **Local, offline, CPU-compatible:** No API key required; runs on CPU with transformers library
- **Package:** sentence-transformers 6.0.1 (released 2026-08-31): https://pypi.org/project/sentence-transformers/

*Vector search approaches:*

Option A — **pgvector SQL** (recommended for production):
- Uses PostgreSQL with pgvector extension installed via `CREATE EXTENSION vector;`
- Cosine distance operator: `<=>` (e.g., `SELECT * FROM embeddings ORDER BY embedding <=> query_vector LIMIT 10`)
- Cosine similarity computed as: `1 - cosine_distance`
- Also supports Euclidean (L2) and inner product distance metrics
- Source: pgvector docs https://github.com/pgvector/pgvector

Option B — **Local FAISS or numpy** (for development/testing):
- FAISS (Facebook AI Similarity Search): https://github.com/facebookresearch/faiss — C++ library with Python bindings, optimized for large-scale ANN; supports multiple index types (Flat, IVF, HNSW)
- Flat index: brute-force exact search
- IVF-Flat: inverted file + product quantization for scalable approximate search
- HNSW (in FAISS): hierarchical navigable small world graph-based index
- Simpler alternative: numpy cosine similarity with `scipy.spatial.distance.cosine()` for small datasets

*PDF extraction:*
- **pdfplumber** 0.11.10 (released 2026-06-15, tested on Python 3.10–3.14): https://pypi.org/project/pdfplumber/ — provides `extract_text()` method; built on pdfminer.six
- **pypdf** 5.7.0 (released 2025-06-29): alternative with similar APIs
- **Recommendation:** pdfplumber for layout-preserving text extraction; pypdf for more control over page objects

*Authoritative definitions:*

**RAG (Retrieval-Augmented Generation):**
"Retrieval-augmented generation (RAG) is a technique that enables large language models (LLMs) to retrieve and incorporate new information from external data sources. RAG inserts a data retrieval component into the response generation process: LLMs first refer to a specified set of documents, then respond to user queries, with these documents supplementing information from the LLM's pre-existing training data."
- Source: IBM https://www.ibm.com/think/topics/retrieval-augmented-generation; Pinecone https://www.pinecone.io/learn/retrieval-augmented-generation/
- Origin: 2020 publication "Retrieval-Augmented Generation for Knowledge-Intensive Tasks" (Meta AI Research)

**ANN (Approximate Nearest Neighbor) Search:**
"Approximate nearest neighbor search returns near-best results much faster than exact search by exploring only a subset of candidates. ANN algorithms aim to identify data points that are closest to a given query point approximately, dramatically reducing computational load while maintaining acceptable accuracy levels."
- Source: TiDB https://www.pingcap.com/article/approximate-nearest-neighbor-ann-search-explained-ivf-vs-hnsw-vs-pq/

**HNSW (Hierarchical Navigable Small World):**
"HNSW is a state-of-the-art graph-based method widely adopted for high recall with low query latency. HNSW organizes data points into a multi-layer graph structure where each layer forms a navigable small-world graph with nodes connected to their approximate nearest neighbors; higher layers provide coarse-grained shortcuts while lower layers enable fine-grained local search."
- Source: Medium https://medium.com/@adnanmasood/the-shortcut-through-space-hierarchical-navigable-small-worlds-hnsw-in-vector-search-4df5aa755100

**IVF (Inverted File):**
"IVF divides the data space into clusters using k-means; during a query, only the relevant clusters are searched, significantly speeding up the process. IVF is a two-stage algorithm combining a coarse quantizer with Product Quantization (PQ), first partitioning the database into Voronoi cells and selecting the nearest cells for a query vector, then applying PQ only to vectors in the selected cells."
- Source: TiDB https://www.pingcap.com/article/approximate-nearest-neighbor-ann-search-explained-ivf-vs-hnsw-vs-pq/

**MCP (Model Context Protocol):**
"MCP is an open protocol that enables seamless integration between LLM applications and external data sources and tools. MCP is a JSON-RPC-based standard that lets any AI application discover tools, reusable prompts, resources, and other context from remote MCP servers. The protocol defines four main primitives: Tools (functions the model can call with structured inputs), Resources (data or content the client can read), Prompts (reusable prompt templates), and Instructions (server-wide guidance)."
- Source: Official MCP Specification https://modelcontextprotocol.io/specification/2026-07-28

**Caveats / limits:**
1. **Embedding model size/latency:** all-MiniLM-L6-v2 is small (~90 MB) and fast on CPU, but may have lower quality than larger models (e.g., all-mpnet-base-v2, 384 dims, or OpenAI text-embedding-3-small). Trade-off: offline + fast vs. quality.
2. **pgvector performance:** unindexed `<=>` queries on large tables are O(n); for >1M vectors, create an IVF index or use a specialized vector DB (Pinecone, Weaviate).
3. **FAISS memory:** Flat index stores all vectors in RAM; for large corpora, use IVF or disk-based alternatives.
4. **PDF extraction:** complex layouts, scanned images (OCR needed), and multi-column text may require post-processing; pdfplumber handles layout better than pypdf for structured PDFs.
5. **Chunking strategy:** RAG requires careful chunk size/overlap tuning; oversized chunks dilute relevance, undersized chunks lack context.

**Recommendation:**
1. For the theory chapter (AGENT-1):
   - Define RAG as the retrieval + generation pipeline
   - Cite the official MCP spec for MCP definition
   - Use TiDB and Medium sources for HNSW/IVF diagrams and explanations
   - Demo: use sentence-transformers all-MiniLM-L6-v2 with numpy cosine similarity on a tiny corpus (no pgvector needed for theory)

2. For the RAG-over-PDFs worked example (AGENT-3):
   - Use pdfplumber for PDF text extraction (better layout preservation)
   - Embed with sentence-transformers all-MiniLM-L6-v2 (Apache 2.0, local, CPU)
   - Store in pgvector if PostgreSQL is available; fallback to local FAISS index or numpy
   - Retrieve top-k via `<=>` (pgvector) or FAISS/numpy cosine; show scores
   - Generate answer with key-gated LLM call; provide no-key fallback (return retrieved context)
   - Chunk size: start with 512 tokens, overlap 50; adjust based on retrieval quality

3. Pin versions:
   ```
   sentence-transformers==6.0.1
   pdfplumber==0.11.10
   faiss-cpu==1.8.0  # optional, if local vector store needed
   ```

**Date checked:** 2026-09-02

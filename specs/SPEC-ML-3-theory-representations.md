# SPEC-ML-3: Theory — Representations (Embeddings, Tokenizers, Similarity, Quantization, Fine-tuning)

**Status:** approved
**Subject:** Machine Learning
**Section:** Theory
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8
**Prerequisites:** SPEC-ML-1, SPEC-ML-2

## Intent
The glue concepts behind modern ML/LLMs: how text becomes numbers (tokenizers, word2vec, embeddings),
how we measure closeness (cosine similarity, Euclidean distance), and two production-critical ideas —
quantization (shrink models) and fine-tuning (adapt models). These underpin the Agentic subject too.

## Learning objectives
- LO1 — Explain tokenization (word/subword/BPE) and why subword tokenizers won.
- LO2 — Explain embeddings and word2vec: dense vectors where geometry encodes meaning.
- LO3 — Compute and contrast cosine similarity vs Euclidean distance; know when each is used (and why cosine dominates for embeddings).
- LO4 — Explain quantization (fp32→int8, what it trades) and fine-tuning (full vs LoRA/PEFT) at a working level.

## Scope
In: tokenizers, word2vec/embeddings, cosine vs Euclidean (runnable), quantization intuition, fine-tuning intuition.
Out: training embeddings from scratch at scale, full PEFT implementation (mention + link).

## Outline
1. Text → numbers — tokenizers (word/subword/BPE) with a tiny worked example.
2. Embeddings & word2vec — the "king − man + woman ≈ queen" geometry.
3. Similarity — cosine vs Euclidean, computed on real embedding vectors; normalization's role.
4. Quantization — precision vs size/speed; where int8 helps.
5. Fine-tuning — adapting a pretrained model; full vs LoRA; when to fine-tune vs prompt/RAG (forward-link to Agentic).
6. Pitfalls — tokenizer/model mismatch, comparing unnormalised vectors with cosine, over-quantizing, fine-tuning when RAG suffices.

## Assets to produce
- Prose: "Machine Learning/Theory/representations.md"
- Code: "Machine Learning/Theory/code/similarity_demo.py" (cosine vs Euclidean on small vectors + a tiny tokenizer illustration — CPU, numpy; embeddings can be small hand-made vectors to stay dependency-light)
- Artefacts: a 2D embedding scatter (project a few word vectors) + a cosine-vs-Euclidean comparison table.

## Claims to ground (Haiku, before writing)
- [ ] Verify the definitions/formulas of cosine similarity and Euclidean distance, BPE tokenization, and word2vec (skip-gram/CBOW) against authoritative sources.
- [ ] Verify the standard quantization levels (fp32/fp16/int8) and the LoRA idea against authoritative sources.

## Acceptance criteria
- [ ] AC1 — LOs delivered. AC2 — similarity_demo.py runs (numpy) and produces the scatter + table; snippet-check passes. AC3 — similarity/tokenizer/word2vec/quantization/LoRA claims grounded. AC4 — cosine-vs-Euclidean made concrete; fine-tune-vs-RAG decision previewed for the Agentic subject.

## Gates
Entry: approved; notes landed. Exit: DoD checklist.

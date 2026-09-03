# SPEC-ML-14: Text & NLP metrics — measuring classifiers, generators, and retrievers

**Status:** written by Sonnet, grounded by Haiku (NOTE-ML-14-package-versions, -metric-definitions,
-library-apis, -llm-judge-biases) — snippet-compile, py_compile, actual-run, and markdown-render gates
all pass — pending independent review + architect merge
**Subject:** Machine Learning
**Section:** Worked Examples
**Routing:** writer=Sonnet 4.6 · research=Haiku · review=Sonnet (fresh) · architect=Opus 4.8
**Prerequisites:** SPEC-DS-6 (classification metrics), SPEC-ML-8 (text classification),
SPEC-ML-9 (text generation), SPEC-ML-13 (fine-tuning). Mirrors SPEC-ML-7 (CV metrics).

## Intent
The book measures vision models thoroughly (ML-7: IoU, mAP, mAR) but never gives text its own metrics
chapter — text classification borrows DS-6's accuracy/F1, and generation quality is never quantified
at all. This chapter is the NLP counterpart to CV-metrics: it explains and *computes by hand, then
reproduces with a library*, the metrics for the three text tasks the book covers — classification,
generation, and semantic similarity / retrieval — and is honest about where automatic metrics
mislead. A Java dev who has shipped classifiers knows precision/recall; here they learn why BLEU and
ROUGE exist, what perplexity really measures, and why "the metric went up" and "the output got better"
are not the same claim for generation.

## Learning objectives
After this chapter the reader can:
- LO1 — Choose and compute the right *classification* metric for text: accuracy vs. macro/micro/
  weighted-F1, and why macro-F1 matters under class imbalance (tie back to DS-6/DS-8).
- LO2 — Explain and compute *generation* metrics: perplexity (what it is, why lower is better, its
  limits), BLEU and chrF (precision-oriented, n-gram overlap), ROUGE (recall-oriented, summarization),
  and an embedding-based metric (BERTScore) — by hand on a tiny example, then via a library.
- LO3 — Explain and compute *retrieval / semantic-similarity* metrics: cosine similarity, and ranked
  metrics Recall@k, MRR, and nDCG — the numbers a RAG system (AGENT-3) is actually judged on.
- LO4 — Explain why automatic generation metrics disagree with human judgement, and what
  "LLM-as-a-judge" is (and its own biases) — so the reader treats a single number with suspicion.

## Scope
In scope: metric *definitions* (grounded), a by-hand computation on a tiny hand-checkable example for
each, and a library reproduction. Small, CPU-friendly, no training (this is a metrics chapter; it
consumes model outputs, it doesn't produce models).
Out of scope: task-specific leaderboards (GLUE/SuperGLUE scoring harnesses), full RAG evaluation
frameworks (RAGAS etc.) beyond a named pointer, and human-eval study design. One-paragraph pointers.

## Outline (section-by-section)
1. **Cold open** — the problem: a summary that shares no exact words with the reference but means the
   same thing — how do you score it? Exact-match says 0; a human says "great." Pose the gap
   between *string overlap* and *meaning*, which the whole chapter is about.
2. **What & why** — a "you are here" map of the three task families and the metric family each needs;
   the Java tie-in (you already trust precision/recall; generation breaks the assumption that there's
   one right answer).
3. **Classification metrics for text** — macro vs micro vs weighted F1 on a small imbalanced
   label set, by hand then `sklearn`. Reuse/di reference DS-6.
4. **Generation metrics** — perplexity (compute from a model's token log-probs on a real short
   sentence), BLEU + chrF (`sacrebleu`) and ROUGE (`rouge-score`) on a tiny candidate/reference pair
   worked by hand first, then BERTScore (`bert-score`) to show an embedding-based score catches
   paraphrase where n-gram overlap fails. Show the three disagreeing on the same pair — that's the point.
5. **Retrieval / similarity metrics** — cosine similarity, then Recall@k / MRR / nDCG on a tiny ranked
   list by hand, then a library reproduction. Connect explicitly to AGENT-3 RAG.
6. **When the number lies** — automatic-vs-human gap; LLM-as-judge and its biases (position,
   verbosity, self-preference) — grounded, not asserted.
7. **Pitfalls & recap** — tokenization sensitivity of BLEU, perplexity not comparable across
   tokenizers, single-reference bias; recap table (task → metric → library).

## Assets to produce
- Prose: `02-machine-learning/03-worked-examples/02-natural-language/04-text-metrics.md`
- Code: `.../02-natural-language/code/text_metrics.py` (each metric by hand + library, seed set,
  deps pinned)
- Artefacts: a small comparison table/plot (the same candidate/reference scored by BLEU/ROUGE/
  BERTScore, disagreeing) under `.../02-natural-language/artefacts/`.

## Claims to ground (Haiku research brief — do BEFORE writing)
- [ ] Package versions to pin: `sacrebleu`, `rouge-score`, `bert-score`, `scikit-learn`,
      `evaluate`, `torch`/`transformers` (for BERTScore + perplexity) — current PyPI versions + dates.
- [ ] Metric definitions to verify (exact formulas, from an authoritative source, with the source):
      BLEU (brevity penalty + modified n-gram precision, sacrebleu's defaults), chrF, ROUGE-1/2/L,
      perplexity (= exp(mean negative log-likelihood per token)), BERTScore (P/R/F over token
      embeddings, with the rescale-with-baseline caveat), cosine similarity, Recall@k, MRR, nDCG
      (with the discount + ideal-DCG normalization).
- [ ] Confirm the current API of each library (e.g. `sacrebleu.corpus_bleu` vs sentence-level;
      `rouge_score.rouge_scorer`; `bert_score.score` signature and the model it downloads).
- [ ] "LLM-as-a-judge" — one authoritative reference for the known biases (cite it, don't assert).

## Acceptance criteria (each maps to evidence)
- [ ] AC1 (LO1–LO3) — each metric computed by hand AND reproduced by a library, numbers matching →
      evidence: runnable script + comparison artefact.
- [ ] AC2 — snippets run → `check_snippets.py` pass + run log.
- [ ] AC3 — every formula/version/claim grounded → NOTE ids.
- [ ] AC4 — audience-fit; the automatic-vs-human caveat is explicit.
- [ ] AC5 — renders on GitHub → `check_markdown_render.py` pass; formulas eyeballed (watch `\text{}`
      escaping in BLEU/nDCG formulas).

## Gates
Entry: this spec approved; research NOTEs landed. Exit: all ACs satisfied; snippets run; links
resolve; fresh-Sonnet review sign-off; architect merge. (See `docs/definition-of-done.md`.)
